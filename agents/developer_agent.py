"""Developer Agent – Backend and Frontend sub-agents that generate
real, functional code based on the Architect's output."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from config.llm_config import llm_developer
from config.paths import (
    PROJECT_ROOT,
    ensure_output_dir,
    project_dir,
    safe_join,
    UnsafePathError,
)
from tools.llm_helpers import parse_llm_json

from agents.architect_agent import ArchitectOutput
from agents.mcp_client import ThreadSafeMCPClient
from agents.ui_designer_agent import UIDesignerOutput


# ── Pydantic output models ──────────────────────────────────────────────────

class CodeFile(BaseModel):
    filename: str = Field(..., description="File name, e.g. app.module.ts")
    path: str = Field(..., description="Relative path, e.g. src/modules/expense")
    language: str = Field(..., description="Programming language, e.g. typescript, dart")
    content: str = Field(..., description="Full source code content of the file")
    description: str = Field(..., description="Brief description of the file's purpose")


class BackendOutput(BaseModel):
    files: list[CodeFile] = Field(..., description="Generated backend files")
    setup_commands: list[str] = Field(
        ..., description="Commands to set up and run the backend"
    )
    dependencies: list[str] = Field(
        ..., description="Required packages / dependencies"
    )


class FrontendOutput(BaseModel):
    files: list[CodeFile] = Field(..., description="Generated frontend files")
    setup_commands: list[str] = Field(
        ..., description="Commands to set up and run the frontend"
    )
    dependencies: list[str] = Field(
        ..., description="Required packages / dependencies"
    )


class DeveloperOutput(BaseModel):
    project_name: str = Field(..., description="Project name")
    backend: BackendOutput = Field(..., description="Backend output")
    frontend: FrontendOutput = Field(..., description="Frontend output")
    integration_notes: list[str] = Field(
        ..., description="Notes on how to connect frontend and backend"
    )


# ── System prompts ───────────────────────────────────────────────────────────

BACKEND_SYSTEM_PROMPT = """\
You are a Senior Backend Developer with over 10 years of experience, \
specializing in NestJS, TypeScript, and Clean Architecture.

Your task is to receive a software architecture document and generate \
real, functional, and complete code for the backend of the project.

PRIORITIES:
- Correct folder structure following Clean Architecture
- TypeORM entities with correct decorators and types
- Creation DTOs with validation using class-validator
- Services with complete CRUD business logic
- REST controllers with correct NestJS decorators
- Main application Module that imports all other modules

STRICT RULES:
1. Respond ONLY with a valid JSON. Do NOT include explanatory text, \
comments, or markdown blocks (```).
2. The JSON must follow EXACTLY this schema:

{{
  "files": [
    {{
      "filename": "app.module.ts",
      "path": "src",
      "language": "typescript",
      "content": "import {{ Module }} from '@nestjs/common';\\n...",
      "description": "Root application module"
    }}
  ],
  "setup_commands": [
    "npm install",
    "npm run start:dev"
  ],
  "dependencies": [
    "@nestjs/core",
    "@nestjs/typeorm",
    "class-validator"
  ]
}}

3. Generate real and functional TypeScript code, not placeholders.
4. Include at least: app.module.ts, one entity for each DB entity, \
one creation DTO per entity, a service with basic CRUD, and a controller \
with the main endpoints.
5. Correctly escape quotes and line breaks within the "content" field.
6. Do NOT use backticks, do NOT use markdown, ONLY pure JSON.
"""

FRONTEND_SYSTEM_PROMPT = """\
You are a Senior Frontend Developer with over 10 years of experience, \
specializing in Flutter, Dart, and Clean Architecture.

Your task is to receive a software architecture document and generate \
real, functional, and complete code for the mobile frontend of the project.

PRIORITIES:
- Feature structure following Clean Architecture
- Dart data models that map backend entities
- Repositories with HTTP calls to the backend using http or dio
- Main pages with functional Material Design widgets
- Navigation between screens

STRICT RULES:
1. Respond ONLY with a valid JSON. Do NOT include explanatory text, \
comments, or markdown blocks (```).
2. The JSON must follow EXACTLY this schema:

{{
  "files": [
    {{
      "filename": "main.dart",
      "path": "lib",
      "language": "dart",
      "content": "import 'package:flutter/material.dart';\\n...",
      "description": "Application entry point"
    }}
  ],
  "setup_commands": [
    "flutter pub get",
    "flutter run"
  ],
  "dependencies": [
    "flutter",
    "http",
    "provider"
  ]
}}

3. Generate real and functional Dart code, not placeholders.
4. Include at least: main.dart, one model per main entity, \
a repository with HTTP calls, and one page per high-priority user story.
5. Correctly escape quotes and line breaks within the "content" field.
6. Do NOT use backticks, do NOT use markdown, ONLY pure JSON.
"""

RETRY_PROMPT = """\
Your previous response was NOT valid JSON. The error was:
{error}

You MUST respond with pure and valid JSON. No extra text, no backticks, \
no explanations. Only the JSON object that complies with the indicated schema.
Make sure to:
- Escape internal quotes with \\\"
- Use \\n for line breaks inside strings
- Do not leave trailing commas in arrays or objects
- Close all braces and brackets correctly

Regenerate the complete response:
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_json_robust(text: str) -> Any:
    """Parse JSON (with json-repair fallback) and unwrap list-wrapped dicts."""
    data = parse_llm_json(text)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        data = data[0]
    return data


def _final_ai_text(messages_list: list) -> str:
    """Return the content of the last AI message, guarding against a trailing
    tool message being mistaken for the JSON answer. Returns "" if none found."""
    for msg in reversed(messages_list):
        if getattr(msg, "type", None) == "ai" and msg.content:
            return msg.content
    if messages_list:
        last_msg = messages_list[-1]
        if getattr(last_msg, "type", None) == "ai":
            return last_msg.content or ""
    return ""


def _clean_readme(text: str) -> str:
    """Clean README markdown from the LLM.

    Unlike ``clean_llm_response`` (which extracts a JSON value and would mangle
    markdown containing braces), this only removes ``<think>`` reasoning blocks and
    a single outer ```` ``` ```` fence if the whole body is wrapped in one.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


# ── Public API ───────────────────────────────────────────────────────────────

# Setup commands we are willing to auto-execute. Each entry is the exact argv
# prefix (after shlex.split) that a command must start with. Anything else is
# skipped. Commands are run via create_subprocess_exec (no shell) so there is no
# shell interpretation of metacharacters.
_SAFE_SETUP_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("npm", "install"),
    ("npm", "ci"),
    ("yarn", "install"),
    ("pnpm", "install"),
    ("flutter", "pub", "get"),
    ("pip", "install", "-r", "requirements.txt"),
)

# Shell metacharacters that must never appear in an auto-executed command's argv.
_SHELL_METACHARS = set(";&|<>`$(){}!*?\n\r")


def _write_generated_files(files: list[CodeFile], project_name: str):
    """Auto-save generated code files under the project's output directory.

    Each path is built from LLM-supplied components via :func:`safe_join`, so a
    malicious ``path``/``filename`` cannot escape the project sandbox. Files are
    NEVER written into the platform repo root.
    """
    project_root = project_dir(project_name)
    for f in files:
        try:
            file_path = safe_join(project_root, f.path, f.filename)
        except UnsafePathError as ue:
            print(f"⚠️ Skipping file with unsafe path {f.path}/{f.filename}: {ue}")
            continue
        file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving file: {file_path}")
        with open(file_path, "w", encoding="utf-8") as out:
            out.write(f.content)


async def save_files_to_filesystem(developer_output: DeveloperOutput, architect_output: Optional[ArchitectOutput] = None):
    """Save backend and frontend generated files to the filesystem using MCP or fallback to python, and auto-generate a professional project README.md."""
    from agents.mcp_client import get_mcp_tools, FILESYSTEM_MCP_CONFIG

    # 1. Connect to Filesystem MCP
    tools, mcp_client = get_mcp_tools(FILESYSTEM_MCP_CONFIG, "stdio", return_client=True)
    tool_map = {t.name: t for t in tools} if tools else {}

    ensure_output_dir()

    # Slugify the project name into a safe slug and resolve a validated, sandboxed
    # project output directory under OUTPUT_DIR.
    raw_slug = developer_output.project_name.lower().replace(" ", "-").replace("_", "-")
    project_clean = re.sub(r"[^a-z0-9-]", "", raw_slug).strip("-") or "project"
    project_output_dir = project_dir(project_clean)
    project_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Auto-Generate Project README.md ───────────────────────────────────────
    readme_content = ""
    try:
        project_name = developer_output.project_name
        
        # Tech stack compilation
        tech_summary = ""
        db_summary = ""
        api_summary = ""
        mermaid_code = ""
        decisions_summary = ""
        
        if architect_output:
            tech_summary = (
                f"- **Backend**: {architect_output.tech_stack.backend}\n"
                f"- **Frontend**: {architect_output.tech_stack.frontend}\n"
                f"- **Database**: {architect_output.tech_stack.database}\n"
                f"- **Infrastructure**: {architect_output.tech_stack.infrastructure}\n"
                f"- **Tools & Libraries**: {', '.join(architect_output.tech_stack.additional_tools)}\n"
                f"- **Architecture Pattern**: {architect_output.architecture_pattern}"
            )
            
            entities = []
            for ent in architect_output.database_entities:
                fields_str = ", ".join(ent.fields)
                rel_str = ", ".join(ent.relationships) if ent.relationships else "None"
                entities.append(f"| {ent.name} | {fields_str} | {rel_str} |")
            db_summary = "\n".join(entities) if entities else "| None | - | - |"
            
            endpoints = []
            for ep in architect_output.api_endpoints:
                endpoints.append(f"| `{ep.method}` | `{ep.path}` | {ep.description} |")
            api_summary = "\n".join(endpoints) if endpoints else "| None | - | - |"
            
            mermaid_code = architect_output.mermaid_diagram
            decisions_summary = "\n".join([f"- {d}" for d in architect_output.key_decisions])
        else:
            tech_summary = "- **Backend**: NestJS + TypeScript\n- **Frontend**: Flutter + Dart\n- **Database**: PostgreSQL"
            db_summary = "| Defined in database entities |"
            api_summary = "| Defined in API controllers |"
            
        integration_notes = "\n".join([f"- {note}" for note in developer_output.integration_notes])
        
        backend_deps = ", ".join(developer_output.backend.dependencies) if developer_output.backend else "None"
        backend_cmds = "\n".join([f"  $ {cmd}" for cmd in developer_output.backend.setup_commands]) if developer_output.backend else "None"
        
        frontend_deps = ", ".join(developer_output.frontend.dependencies) if developer_output.frontend else "None"
        frontend_cmds = "\n".join([f"  $ {cmd}" for cmd in developer_output.frontend.setup_commands]) if developer_output.frontend else "None"

        # Construct LLM prompt for the Technical Writer
        prompt = (
            f"You are an elite Technical Writer and Software Architect.\n"
            f"Your task is to write a highly professional, beautifully formatted, and comprehensive README.md "
            f"for the generated software project named '{project_name}'.\n\n"
            f"Here is the technical details and inputs compiled from the generation process:\n\n"
            f"--- PROJECT NAME ---\n"
            f"{project_name}\n\n"
            f"--- TECHNOLOGY STACK & ARCHITECTURE ---\n"
            f"{tech_summary}\n\n"
            f"--- DATABASE ENTITIES & SCHEMA ---\n"
            f"| Entity Name | Fields | Relationships |\n"
            f"|---|---|---|\n"
            f"{db_summary}\n\n"
            f"--- API REST ENDPOINTS ---\n"
            f"| Method | Path | Description |\n"
            f"|---|---|---|\n"
            f"{api_summary}\n\n"
            f"--- MERMAID ARCHITECTURE DIAGRAM ---\n"
            f"```mermaid\n"
            f"{mermaid_code}\n"
            f"```\n\n"
            f"--- KEY ARCHITECTURAL DECISIONS ---\n"
            f"{decisions_summary}\n\n"
            f"--- BACKEND SETUP & DEPENDENCIES ---\n"
            f"Dependencies: {backend_deps}\n"
            f"Setup commands:\n{backend_cmds}\n\n"
            f"--- FRONTEND SETUP & DEPENDENCIES ---\n"
            f"Dependencies: {frontend_deps}\n"
            f"Setup commands:\n{frontend_cmds}\n\n"
            f"--- INTEGRATION NOTES ---\n"
            f"{integration_notes}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Generate a stunning, complete README.md in English (and only in English).\n"
            f"2. Use premium styling with modern typography, markdown headers, emoji icons, beautifully formatted lists, and tables.\n"
            f"3. Do NOT use any placeholders. Write detailed, complete sections.\n"
            f"4. Structure the document with these exact sections:\n"
            f"   - **Header with Emojis & Description**\n"
            f"   - **Architecture & System Design** (including the Mermaid diagram and key architectural decisions)\n"
            f"   - **Database Schema & Entity Relationship** (with a clean summary table)\n"
            f"   - **API Endpoints Reference** (with a clean method table)\n"
            f"   - **Installation & Setup Guide** (providing step-by-step instructions for both backend and frontend, separating dependencies installation and run commands)\n"
            f"   - **Integration & Communication Guidelines** (explaining CORS, base URLs, authentication mechanisms, DTO mapping, and environment variables)\n"
            f"   - **Local Preview & DevOps Guidelines** (explaining how Docker Compose or local servers spin up the stack)\n"
            f"5. Return ONLY the raw markdown of the README. Do NOT wrap it in any additional markdown code block or backticks, just start with '# Project Name'."
        )
        
        messages = [
            SystemMessage(content="You are an elite Technical Writer and Software Architect. Write a gorgeous, extensive, production-ready project README in Markdown."),
            HumanMessage(content=prompt)
        ]
        
        print("📝 Generating project README.md using Technical Writer LLM...")
        res = llm_developer.invoke(messages)
        readme_content = _clean_readme(res.content)
    except Exception as exc:
        print(f"⚠️ Error generating README markdown via LLM: {exc}")
        readme_content = f"# {developer_output.project_name}\n\nAuto-generated project using devAIteam."

    all_files = []
    if developer_output.backend and developer_output.backend.files:
        all_files.extend(developer_output.backend.files)
    if developer_output.frontend and developer_output.frontend.files:
        all_files.extend(developer_output.frontend.files)
        
    mcp_available = "write_file" in tool_map

    try:
        for f in all_files:
            # Build the destination via safe_join so an LLM-supplied path/filename
            # can never escape this project's output directory (and never lands in
            # PROJECT_ROOT itself).
            try:
                dest_path = safe_join(project_output_dir, f.path, f.filename)
            except UnsafePathError as ue:
                print(f"⚠️ Skipping file with unsafe path {f.path}/{f.filename}: {ue}")
                continue

            rel_path = os.path.join(f.path, f.filename)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if mcp_available:
                try:
                    print(f"🚀 [Filesystem MCP] Saving file: {rel_path}...")
                    tool_map["write_file"].invoke({
                        "path": str(dest_path),
                        "content": f.content
                    })
                    print(f"📄 Saved: {rel_path}")
                    continue
                except Exception as e:
                    print(f"⚠️ [Filesystem MCP] Error writing {rel_path} via MCP: {e}. Using local fallback...")

            # Fallback to python
            try:
                with open(dest_path, "w", encoding="utf-8") as out_f:
                    out_f.write(f.content)
                print(f"📄 Saved (Local Fallback): {rel_path}")
            except Exception as e:
                print(f"❌ Error writing local fallback file: {e}")

        # ── Write Project README.md under the project's own output dir only ────
        if readme_content:
            readme_output_path = project_output_dir / "README.md"
            if mcp_available:
                try:
                    print(f"🚀 [Filesystem MCP] Saving project README to: {readme_output_path}...")
                    tool_map["write_file"].invoke({
                        "path": str(readme_output_path),
                        "content": readme_content
                    })
                    print(f"📄 Generated project README saved to: {readme_output_path}")
                except Exception as e:
                    print(f"⚠️ [Filesystem MCP] Error writing project README: {e}. Using local fallback...")
                    try:
                        readme_output_path.write_text(readme_content, encoding="utf-8")
                        print(f"📄 Generated project README saved to: {readme_output_path}")
                    except Exception as e2:
                        print(f"❌ Error writing project output README: {e2}")
            else:
                try:
                    readme_output_path.write_text(readme_content, encoding="utf-8")
                    print(f"📄 Generated project README saved to: {readme_output_path}")
                except Exception as e:
                    print(f"❌ Error writing project output README: {e}")

        # ── Safe Setup Command Execution ─────────────────────────────────────
        await _run_safe_setup_commands(developer_output, project_output_dir)
    finally:
        if mcp_client:
            mcp_client.close()


async def _run_safe_setup_commands(developer_output: DeveloperOutput, cwd) -> None:
    """Execute only allowlisted dependency-install commands, with NO shell.

    Each command is parsed with ``shlex.split`` and run via
    ``create_subprocess_exec`` (so shell metacharacters are never interpreted).
    Only commands whose argv prefix exactly matches an allowlist entry are run;
    everything else is skipped and logged.
    """
    all_cmds: list[str] = []
    if developer_output.backend and developer_output.backend.setup_commands:
        all_cmds.extend(developer_output.backend.setup_commands)
    if developer_output.frontend and developer_output.frontend.setup_commands:
        all_cmds.extend(developer_output.frontend.setup_commands)

    for cmd in all_cmds:
        try:
            argv = shlex.split(cmd)
        except ValueError as e:
            print(f"⛔ [Setup] Skipping unparseable command {cmd!r}: {e}")
            continue
        if not argv:
            continue

        # Reject any argv token containing shell metacharacters.
        if any(any(c in _SHELL_METACHARS for c in token) for token in argv):
            print(f"⛔ [Setup] Skipping command with shell metacharacters: {cmd!r}")
            continue

        # Only run if the argv prefix exactly matches an allowlist entry.
        allowed = next(
            (prefix for prefix in _SAFE_SETUP_COMMANDS if tuple(argv[: len(prefix)]) == prefix),
            None,
        )
        if allowed is None:
            print(f"⛔ [Setup] Skipping non-allowlisted command: {cmd!r}")
            continue

        print(f"⚙️ [Setup] Running (no shell): {' '.join(argv)} ...")
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                print(f"✅ [Setup] Successfully completed: {' '.join(argv)}")
            else:
                err_msg = stderr.decode(errors="replace").strip() or stdout.decode(errors="replace").strip()
                print(f"⚠️ [Setup] '{' '.join(argv)}' exited {proc.returncode}: {err_msg[:200]}")
        except Exception as exc:
            print(f"⚠️ [Setup] Error executing '{' '.join(argv)}': {exc}")


def run_backend_agent(architect_output: ArchitectOutput, designer_output: Optional[UIDesignerOutput] = None) -> BackendOutput:
    """Generate backend code from the architect's output and write files to disk."""
    
    # Connect Context7 and Filesystem
    context7_client = ThreadSafeMCPClient("npx", ["-y", "@upstash/context7-mcp@latest"])
    filesystem_client = ThreadSafeMCPClient(
        "npx", ["-y", "@modelcontextprotocol/server-filesystem", str(PROJECT_ROOT)]
    )

    try:
        tools = context7_client.get_tools() + filesystem_client.get_tools()

        llm = llm_developer

        agent = create_react_agent(llm, tools=tools)
        arch_json = architect_output.model_dump_json(indent=2)

        prompt = (
            f"{BACKEND_SYSTEM_PROMPT}\n\n"
            "Generate the complete backend code based on this architecture. "
            "If you need to consult documentation about NestJS, TypeORM, Postgres, etc., use Context7. "
            "Additionally, you can use Filesystem tools to verify files if needed. "
            "IMPORTANT: DO NOT search, list, or read files in directories outside the current project's backend, "
            "especially NEVER enter folders like 'gestor_gastos' or 'fintrack_app'. "
            "Only operate within the development context of this current project.\n\n"
            f"{arch_json}"
        )

        print("☕  Backend Developer Agent consulting tools and writing backend...")
        response = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": 100},
        )

        final_message = _final_ai_text(response["messages"])

        try:
            return BackendOutput.model_validate(_parse_json_robust(final_message))
        except Exception as exc:
            print(f"   ⚠️ Backend parsing failed: {exc}. Retrying with direct prompt...")
            try:
                messages = [
                    SystemMessage(content=BACKEND_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                    HumanMessage(content=RETRY_PROMPT.format(error=str(exc)))
                ]
                retry_res = llm.invoke(messages)
                return BackendOutput.model_validate(_parse_json_robust(retry_res.content))
            except Exception as retry_exc:
                print(f"   ❌ Backend retry parsing failed: {retry_exc}")
                raise
    finally:
        context7_client.close()
        filesystem_client.close()


def run_frontend_agent(architect_output: ArchitectOutput, designer_output: Optional[UIDesignerOutput] = None) -> FrontendOutput:
    """Generate frontend code from the architect's output and write files to disk."""
    
    # Connect Context7 and Filesystem
    context7_client = ThreadSafeMCPClient("npx", ["-y", "@upstash/context7-mcp@latest"])
    filesystem_client = ThreadSafeMCPClient(
        "npx", ["-y", "@modelcontextprotocol/server-filesystem", str(PROJECT_ROOT)]
    )

    try:
        tools = context7_client.get_tools() + filesystem_client.get_tools()

        llm = llm_developer

        agent = create_react_agent(llm, tools=tools)

        arch_json = architect_output.model_dump_json(indent=2)

        # Format DESIGN.md and screen references
        ui_context = ""
        system_prompt = FRONTEND_SYSTEM_PROMPT
        if designer_output:
            system_prompt += "\nImplement the screens faithfully following the design system from the attached DESIGN.md. Colors, typography, and spacing must match the defined tokens exactly."
            ui_context += "# DESIGN.md\n"
            ui_context += f"{designer_output.design_system_notes}\n\n"
            ui_context += "## SCREENS AND VIEWS REFERENCE:\n"
            for scr in designer_output.screens:
                ui_context += f"### Screen: {scr.name}\n"
                ui_context += f"Description: {scr.description}\n"
                if getattr(scr, 'stitch_screen_url', None):
                    ui_context += f"Stitch URL: {scr.stitch_screen_url}\n"
                if getattr(scr, 'html_content', None):
                    html_snippet = scr.html_content
                    if len(html_snippet) > 800:
                        html_snippet = html_snippet[:800] + "\n... [TRUNCATED FOR CONTEXT WINDOW] ..."
                    ui_context += f"HTML Reference (Truncated):\n```html\n{html_snippet}\n```\n"
                ui_context += "\n"
        else:
            ui_context = "No UI design provided."

        prompt = (
            f"Generate the complete Flutter frontend code based on this architecture and the provided screen design (Google Stitch). "
            "If you need to consult documentation about Flutter, Widgets, http, etc., use Context7. "
            "Additionally, you can use Filesystem tools to verify files if needed. "
            "IMPORTANT: DO NOT search, list, or read files in directories outside the current project's frontend, "
            "especially NEVER enter folders like 'gestor_gastos' or 'fintrack_app'. "
            "Only operate within the development context of this current project.\n\n"
            f"ARCHITECTURE:\n{arch_json}\n\n"
            f"INTERFACE DESIGN AND SCREENS:\n{ui_context}"
        )

        print("📱  Frontend Developer Agent consulting tools and writing frontend...")
        response = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": 100},
        )

        final_message = _final_ai_text(response["messages"])

        try:
            return FrontendOutput.model_validate(_parse_json_robust(final_message))
        except Exception as exc:
            print(f"   ⚠️ Frontend parsing failed: {exc}. Retrying with direct prompt...")
            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt),
                    HumanMessage(content=RETRY_PROMPT.format(error=str(exc)))
                ]
                retry_res = llm.invoke(messages)
                return FrontendOutput.model_validate(_parse_json_robust(retry_res.content))
            except Exception as retry_exc:
                print(f"   ❌ Frontend retry parsing failed: {retry_exc}")
                raise
    finally:
        context7_client.close()
        filesystem_client.close()
