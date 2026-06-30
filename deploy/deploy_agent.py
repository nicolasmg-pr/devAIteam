import os
import re
import json
import shutil
import asyncio
import subprocess
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from config.llm_config import llm_developer
from config.paths import validate_project_name, UnsafePathError
from langchain_core.messages import HumanMessage, SystemMessage

try:
    import tomllib as _toml  # Python 3.11+
    _TOML_LOADS = lambda s: _toml.loads(s)
except Exception:  # pragma: no cover
    try:
        import toml as _toml  # type: ignore
        _TOML_LOADS = lambda s: _toml.loads(s)
    except Exception:
        _TOML_LOADS = None


_MINIMAL_FLY_TOML_TEMPLATE = '''\
app = "{name}"
primary_region = "mad"

[build]

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0
'''

_MINIMAL_RAILWAY_JSON_TEMPLATE = '''\
{{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {{
    "builder": "DOCKERFILE"
  }},
  "deploy": {{
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }}
}}
'''

# ── Pydantic models ──────────────────────────────────────────────────────────

class DeployTarget(BaseModel):
    platform: str = Field(..., description='"fly" | "railway" | "render"')
    app_url: str = Field(..., description="URL where the app is deployed")
    deploy_status: str = Field(..., description='"success" | "failed" | "pending"')
    logs: List[str] = Field(default_factory=list)

class DeployOutput(BaseModel):
    project_name: str
    targets: List[DeployTarget]
    primary_url: str
    share_message: str
    instructions: str

# ── Helpers for cleaning response ────────────────────────────────────────────

def clean_txt_response(raw: str) -> str:
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL).strip()
    return cleaned

# ── Deploy Agent logic ───────────────────────────────────────────────────────

def check_deploy_tools() -> dict:
    """Check availability of flyctl, railway CLI, and RENDER_API_KEY."""
    fly_ok = bool(shutil.which("flyctl") or shutil.which("fly"))
    railway_ok = bool(shutil.which("railway"))
    
    render_key = os.getenv("RENDER_API_KEY", "")
    render_ok = bool(render_key and render_key != "your_render_api_key_here")
    
    return {
        "fly": fly_ok,
        "railway": railway_ok,
        "render": render_ok
    }

def generate_fly_config(project_name: str, docker_compose_path: str) -> str:
    """Generate fly.toml and save it under ./output/{project_name}/fly.toml"""
    print(f"   ⚙️  [Deploy] Generating fly.toml for Fly.io...")
    system_prompt = "You are an elite Cloud DevOps Engineer. Generate a professional and functional fly.toml file to deploy a NestJS backend or a Flutter Web frontend.\nRespond ONLY with the pure content of the fly.toml file, without markdown, without backticks, without explanations."
    human_prompt = f"Project: {project_name}\nGenerate fly.toml:"
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    response = llm_developer.invoke(messages)
    content = clean_txt_response(response.content)

    # Validate that the generated TOML parses; fall back to a known-good template.
    if _TOML_LOADS is not None:
        try:
            _TOML_LOADS(content)
        except Exception as e:
            print(f"   ⚠️  [Deploy] Generated fly.toml did not parse ({e}); using minimal template.")
            content = _MINIMAL_FLY_TOML_TEMPLATE.format(name=project_name)
    else:
        print("   ⚠️  [Deploy] No TOML parser available; cannot validate fly.toml content.")

    out_dir = f"./output/{project_name}"
    os.makedirs(out_dir, exist_ok=True)
    fly_path = os.path.join(out_dir, "fly.toml")
    try:
        with open(fly_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"   ✅ [Deploy] fly.toml saved in {fly_path}")
    except Exception as e:
        print(f"   ⚠️  [Deploy] Error saving fly.toml: {e}")
    return content

def generate_railway_config(project_name: str) -> str:
    """Generate railway.json and save it under ./output/{project_name}/railway.json"""
    print(f"   ⚙️  [Deploy] Generating railway.json for Railway...")
    system_prompt = "You are an elite Cloud DevOps Engineer. Generate a professional and functional railway.json file to deploy a modular web application.\nRespond ONLY with the pure JSON content, without markdown, without backticks, without explanations."
    human_prompt = f"Project: {project_name}\nGenerate railway.json:"
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    response = llm_developer.invoke(messages)
    content = clean_txt_response(response.content)

    # Validate that the generated JSON parses; fall back to a known-good template.
    try:
        json.loads(content)
    except Exception as e:
        print(f"   ⚠️  [Deploy] Generated railway.json did not parse ({e}); using minimal template.")
        content = _MINIMAL_RAILWAY_JSON_TEMPLATE

    out_dir = f"./output/{project_name}"
    os.makedirs(out_dir, exist_ok=True)
    railway_path = os.path.join(out_dir, "railway.json")
    try:
        with open(railway_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"   ✅ [Deploy] railway.json saved in {railway_path}")
    except Exception as e:
        print(f"   ⚠️  [Deploy] Error saving railway.json: {e}")
    return content

async def run_deploy(project_name: str, platform="auto") -> DeployOutput:
    """Deploy the project or generate instructions if no credentials/tools exist."""
    # Validate before the name reaches any subprocess argument (a leading '-'
    # would otherwise be parsed as a CLI flag).
    project_name = validate_project_name(project_name)

    tools = check_deploy_tools()

    out_dir = f"./output/{project_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    targets = []
    primary_url = ""
    share_message = ""
    instructions = ""
    
    # Check what platform to use
    selected_platform = None
    if platform == "auto":
        if tools["fly"]:
            selected_platform = "fly"
        elif tools["railway"]:
            selected_platform = "railway"
        elif tools["render"]:
            selected_platform = "render"
    else:
        selected_platform = platform.lower()
        
    print(f"   🚀 Starting deployment of '{project_name}' (Selected platform: {selected_platform or 'manual'})...")
    
    # ── FLY.IO DEPLOYMENT ────────────────────────────────────────────────────
    if selected_platform == "fly" and tools["fly"]:
        print("   📦 Building Docker image...")
        await asyncio.sleep(2)
        print("   🚀 Deploying on Fly.io...")
        
        try:
            # Generate config first
            generate_fly_config(project_name, os.path.join(out_dir, "docker-compose.yml"))
            
            # Subprocess commands:
            # 1. flyctl auth whoami
            res_whoami = subprocess.run(["fly", "auth", "whoami"], capture_output=True, text=True)
            if res_whoami.returncode != 0:
                raise Exception("Fly.io CLI not authenticated. Run 'fly auth login' first.")
                
            print("   🌐 Configuring domain...")
            # 2. flyctl launch --no-deploy
            launch_res = subprocess.run(["fly", "launch", "--no-deploy", "--name", project_name, "--region", "mad"], cwd=out_dir, capture_output=True, text=True)
            if launch_res.returncode != 0:
                raise Exception(f"'fly launch' failed (exit {launch_res.returncode}): {launch_res.stderr.strip()}")
            # 3. flyctl deploy --local-only
            deploy_res = subprocess.run(["fly", "deploy", "--local-only"], cwd=out_dir, capture_output=True, text=True)
            if deploy_res.returncode != 0:
                raise Exception(f"'fly deploy' failed (exit {deploy_res.returncode}): {deploy_res.stderr.strip()}")

            # Fetch status
            status_res = subprocess.run(["fly", "status"], cwd=out_dir, capture_output=True, text=True)
            primary_url = f"https://{project_name}.fly.dev"

            print(f"   ✅ Deployment completed: {primary_url}")
            targets.append(DeployTarget(platform="fly", app_url=primary_url, deploy_status="success", logs=[status_res.stdout]))

        except Exception as e:
            print(f"   ❌ [Deploy Fly] Error: {e}")
            targets.append(DeployTarget(platform="fly", app_url="", deploy_status="failed", logs=[str(e)]))
            selected_platform = None  # Fallback to manual
            
    # ── RAILWAY DEPLOYMENT ───────────────────────────────────────────────────
    elif selected_platform == "railway" and tools["railway"]:
        print("   📦 Building Docker image...")
        await asyncio.sleep(2)
        print("   🚀 Deploying on Railway...")
        
        try:
            generate_railway_config(project_name)
            # Check login
            res_login = subprocess.run(["railway", "whoami"], capture_output=True, text=True)
            if res_login.returncode != 0:
                raise Exception("Railway CLI not authenticated. Run 'railway login' first.")
                
            print("   🌐 Configuring domain...")
            init_res = subprocess.run(["railway", "init", "--name", project_name], cwd=out_dir, capture_output=True, text=True)
            if init_res.returncode != 0:
                raise Exception(f"'railway init' failed (exit {init_res.returncode}): {init_res.stderr.strip()}")
            up_res = subprocess.run(["railway", "up"], cwd=out_dir, capture_output=True, text=True)
            if up_res.returncode != 0:
                raise Exception(f"'railway up' failed (exit {up_res.returncode}): {up_res.stderr.strip()}")

            domain_res = subprocess.run(["railway", "domain"], cwd=out_dir, capture_output=True, text=True)
            primary_url = domain_res.stdout.strip() if domain_res.stdout else f"https://{project_name}.up.railway.app"

            print(f"   ✅ Deployment completed: {primary_url}")
            targets.append(DeployTarget(platform="railway", app_url=primary_url, deploy_status="success", logs=[domain_res.stdout]))
            
        except Exception as e:
            print(f"   ❌ [Deploy Railway] Error: {e}")
            targets.append(DeployTarget(platform="railway", app_url="", deploy_status="failed", logs=[str(e)]))
            selected_platform = None  # Fallback to manual
            
    # ── RENDER DEPLOYMENT ────────────────────────────────────────────────────
    elif selected_platform == "render" and tools["render"]:
        print("   🚀 Render deployment requested...")
        # The Render REST API integration is not implemented yet. Do NOT fabricate
        # a *.onrender.com URL or claim success; report it honestly and fall back
        # to manual instructions below.
        msg = (
            "Automated Render deployment is not implemented yet (no API call is made). "
            "Use the manual instructions below to deploy via dashboard.render.com."
        )
        print(f"   ⚠️  [Deploy Render] {msg}")
        targets.append(DeployTarget(platform="render", app_url="", deploy_status="pending", logs=[msg]))
        primary_url = ""
        selected_platform = None

    # ── FALLBACK TO MANUAL INSTRUCTIONS ──────────────────────────────────────
    if not primary_url or selected_platform is None:
        print("   ⚠️  Automated deployment not available due to missing tools or credentials.")
        print("   🧠 Generating detailed deployment instructions for Fly.io, Railway, and Render...")
        
        system_prompt = """\
You are an expert Senior Cloud DevOps Architect (Fly.io, Railway, Render).
Your task is to generate a technical document with the precise manual instructions and exact console commands \
to deploy this NestJS + Flutter + PostgreSQL project on the three specified platforms.
Interpolate the project name in a personalized way.
"""
        human_prompt = f"""
Project Name: {project_name}
Generate the step-by-step guide with complete console commands and brief explanations.
"""
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        try:
            response = llm_developer.invoke(messages)
            instructions = clean_txt_response(response.content)
        except Exception as e:
            print(f"   ⚠️  [Deploy] Could not generate detailed instructions via LLM: {e}")
            instructions = f"Basic deployment guide for {project_name}:\n- Fly.io: fly launch && fly deploy\n- Railway: railway init && railway up\n- Render: connect your repository on dashboard.render.com"

        primary_url = "not_deployed"
        
    # Generate share message via LLM if we deployed successfully
    if primary_url != "not_deployed":
        system_prompt = "Generate a short and enthusiastic message ready to share on social media about the project deployment."
        human_prompt = f"Project: {project_name}\nURL: {primary_url}\nGenerate the share message:"
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        try:
            response = llm_developer.invoke(messages)
            share_message = clean_txt_response(response.content)
        except Exception as e:
            print(f"   ⚠️  [Deploy] Could not generate share message via LLM: {e}")
            share_message = f"I just deployed {project_name} with devAIteam 🚀\nYou can see it at: {primary_url} — Generated 100% by AI locally"
    else:
        share_message = "Project not automatically deployed."

    return DeployOutput(
        project_name=project_name,
        targets=targets,
        primary_url=primary_url,
        share_message=share_message,
        instructions=instructions
    )
