"""LangGraph graph for the Developer Agent pipeline.

Flow: START → backend_node + frontend_node (parallel) → merge_node → formatter → END

Uses LangGraph's native fan-out: multiple edges from START run nodes
concurrently, and they fan-in at the merge_node.
"""

from __future__ import annotations

import traceback
from typing import Optional, Annotated

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field


def _keep_first(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Reducer: keep the first non-None error from concurrent fan-out branches."""
    return a or b

from agents.architect_agent import ArchitectOutput, _run_async_in_thread
from agents.developer_agent import (
    BackendOutput,
    FrontendOutput,
    DeveloperOutput,
    run_backend_agent,
    run_frontend_agent,
    save_files_to_filesystem,
)
from agents.ui_designer_agent import UIDesignerOutput


# ── Graph state ──────────────────────────────────────────────────────────────

class DeveloperState(BaseModel):
    architect_output: ArchitectOutput = Field(
        ..., description="Architect output from previous stage"
    )
    ui_designer_output: Optional[UIDesignerOutput] = Field(
        default=None, description="UI Designer output from previous stage"
    )
    backend_output: Optional[BackendOutput] = Field(
        default=None, description="Backend agent output"
    )
    frontend_output: Optional[FrontendOutput] = Field(
        default=None, description="Frontend agent output"
    )
    developer_output: Optional[DeveloperOutput] = Field(
        default=None, description="Combined developer output"
    )
    error: Annotated[Optional[str], _keep_first] = Field(
        default=None, description="Error message if any"
    )


# ── Nodes ────────────────────────────────────────────────────────────────────

def backend_node(state: DeveloperState) -> dict:
    """Execute the Backend Developer Agent."""
    try:
        print("   ⚙️  Backend Agent generating code...")
        result = run_backend_agent(state.architect_output, state.ui_designer_output)
        return {"backend_output": result}
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"Backend Agent error: {exc}"}


def frontend_node(state: DeveloperState) -> dict:
    """Execute the Frontend Developer Agent."""
    try:
        print("   📱 Frontend Agent generating code...")
        result = run_frontend_agent(state.architect_output, state.ui_designer_output)
        return {"frontend_output": result}
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"Frontend Agent error: {exc}"}


def merge_node(state: DeveloperState) -> dict:
    """Combine backend + frontend outputs into a single DeveloperOutput."""
    if state.error:
        return {}

    if state.backend_output is None or state.frontend_output is None:
        return {"error": "One or both sub-agents did not generate output."}

    project_name = state.architect_output.project_name

    integration_notes = [
        "Configure the backend base URL in the frontend (e.g., http://localhost:3000/api/v1)",
        "Implement JWT authentication: the backend issues tokens, the frontend stores them with flutter_secure_storage",
        "Enable CORS in NestJS to allow requests from the Flutter app",
        "Use environment variables (.env) to manage URLs and secrets in both environments",
        "The backend must serve JSON responses consistent with the frontend's Dart models",
        "Implement interceptors in Flutter (dio) to add auth tokens automatically",
    ]

    dev_output = DeveloperOutput(
        project_name=project_name,
        backend=state.backend_output,
        frontend=state.frontend_output,
        integration_notes=integration_notes,
    )

    return {"developer_output": dev_output}


def filesystem_node(state: DeveloperState) -> dict:
    """Save generated files to filesystem using MCP or Python fallback."""
    if state.error or state.developer_output is None:
        return {}
    print("💾 [Filesystem Node] Saving development files to the system...")
    try:
        _run_async_in_thread(save_files_to_filesystem(state.developer_output, state.architect_output))
    except Exception as exc:
        print(f"⚠️ [Filesystem Node] Error saving files: {exc}")
    return {}


def format_node(state: DeveloperState) -> dict:
    """Pretty-print the Developer output."""
    if state.error:
        print(f"\n❌ Error during Developer Agent execution:\n{state.error}")
        return {}

    output = state.developer_output
    if output is None:
        print("\n⚠️  No development output was generated.")
        return {}

    print("\n" + "=" * 60)
    print(f"💻 GENERATED CODE: {output.project_name}")
    print("=" * 60)

    # ── Backend ──────────────────────────────────────────────────────
    be = output.backend
    print(f"\n🔧 BACKEND ({len(be.files)} files generated):")
    print("-" * 60)
    for f in be.files:
        print(f"   📄 {f.path}/{f.filename} — {f.description}")
    print(f"\n   📦 Dependencies: {', '.join(be.dependencies)}")
    print(f"   🚀 Setup:")
    for cmd in be.setup_commands:
        print(f"      $ {cmd}")

    # ── Frontend ─────────────────────────────────────────────────────
    fe = output.frontend
    print(f"\n🎨 FRONTEND ({len(fe.files)} files generated):")
    print("-" * 60)
    for f in fe.files:
        print(f"   📄 {f.path}/{f.filename} — {f.description}")
    print(f"\n   📦 Dependencies: {', '.join(fe.dependencies)}")
    print(f"   🚀 Setup:")
    for cmd in fe.setup_commands:
        print(f"      $ {cmd}")

    # ── Integration notes ────────────────────────────────────────────
    print(f"\n🔗 INTEGRATION NOTES:")
    print("-" * 60)
    for i, note in enumerate(output.integration_notes, 1):
        print(f"   {i}. {note}")

    print(f"\n💾 Files saved to: ./output/")
    print("\n" + "=" * 60 + "\n")

    return {}


# ── Graph builder ────────────────────────────────────────────────────────────

def build_developer_graph():
    """Build and compile the Developer LangGraph pipeline with parallel fan-out."""
    graph = StateGraph(DeveloperState)

    # Add nodes
    graph.add_node("backend_node", backend_node)
    graph.add_node("frontend_node", frontend_node)
    graph.add_node("merge_node", merge_node)
    graph.add_node("filesystem_node", filesystem_node)
    graph.add_node("formatter", format_node)

    # Fan-out: START triggers both backend and frontend in parallel
    graph.add_edge(START, "backend_node")
    graph.add_edge(START, "frontend_node")

    # Fan-in: both converge at merge_node
    graph.add_edge("backend_node", "merge_node")
    graph.add_edge("frontend_node", "merge_node")

    # Then filesystem and format and finish
    graph.add_edge("merge_node", "filesystem_node")
    graph.add_edge("filesystem_node", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()


# Pre-built graph ready for import
developer_graph = build_developer_graph()
