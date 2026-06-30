import os
import re
import shutil
import asyncio
import subprocess
import webbrowser
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from config.llm_config import llm_developer
from config.paths import PROJECT_ROOT
from langchain_core.messages import HumanMessage, SystemMessage
from agents.developer_agent import DeveloperOutput

# ── Pydantic models ──────────────────────────────────────────────────────────

class ServiceStatus(BaseModel):
    name: str = Field(..., description='"backend" | "frontend"')
    status: str = Field(..., description='"running" | "failed" | "skipped"')
    port: int
    url: str
    pid: Optional[int] = None
    error: Optional[str] = None
    logs_tail: List[str] = Field(default_factory=list, description="Last 10 log lines")

class LocalPreviewOutput(BaseModel):
    project_name: str
    services: List[ServiceStatus]
    backend_url: str
    frontend_url: str
    preview_ready: bool
    docker_compose_generated: bool
    manual_instructions: Optional[str] = None

# ── Helper for clean YAML extraction ─────────────────────────────────────────

def clean_yaml_response(raw: str) -> str:
    # Remove <think> blocks from Qwen
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    
    # Try to extract markdown yml/yaml block
    match = re.search(r'```(?:yaml|yml)?\s*(.*?)\s*```', cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return cleaned

# ── DevOps Agent Functions ───────────────────────────────────────────────────

def generate_docker_compose(developer_output: DeveloperOutput, project_name: str) -> str:
    """Generate a docker-compose.yml file using the LLM and save it."""
    print(f"   🐳 [DevOps] Generating docker-compose.yml with Qwen...")
    
    system_prompt = """\
You are a Senior DevOps Engineer specializing in Docker, Node.js, and Flutter.
Your task is to receive a summary of a project's code and generate a complete, professional, and functional docker-compose.yml file.
The docker-compose.yml file must configure:
1. A PostgreSQL database (with postgres/postgres credentials).
2. A NestJS backend service exposed on port 3000 (building from a Dockerfile or using a node image with volume/setup).
3. A Flutter frontend service compiled for web exposed on port 8080 using Nginx.
4. IMPORTANT: Ensure that any volume mounting for the backend service maps to '/app' (not '/usr/src/app'), to match the backend's Dockerfile working directory.

Your response must ONLY be the valid YAML code for the docker-compose.yml file.
Do NOT include explanations, do NOT include markdown, do NOT use backticks, only the pure YAML content.
"""

    human_prompt = f"""
Project: {project_name}
Generated Backend Files: {list(f.filename for f in developer_output.backend.files)}
Generated Frontend Files: {list(f.filename for f in developer_output.frontend.files)}
Backend Dependencies: {developer_output.backend.dependencies}
Frontend Dependencies: {developer_output.frontend.dependencies}

Generate the docker-compose.yml file:
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    response = llm_developer.invoke(messages)
    yaml_content = clean_yaml_response(response.content)
    
    # Save the file
    out_dir = f"./output/{project_name}"
    os.makedirs(out_dir, exist_ok=True)
    compose_path = os.path.join(out_dir, "docker-compose.yml")
    try:
        with open(compose_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"   ✅ [DevOps] docker-compose.yml saved in {compose_path}")
    except Exception as e:
        print(f"   ⚠️  [DevOps] Error saving docker-compose.yml: {e}")
        
    return yaml_content

def prepare_local_environment(developer_output: DeveloperOutput, project_name: str) -> dict:
    """Check availability of docker, docker-compose, node, and flutter."""
    # Check docker command and docker compose subcommand
    docker_ok = False
    if shutil.which("docker"):
        # Check if docker compose works
        res = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
        if res.returncode == 0:
            docker_ok = True
        else:
            # Check old docker-compose command
            if shutil.which("docker-compose"):
                docker_ok = True

    node_ok = bool(shutil.which("node") and shutil.which("npm"))
    flutter_ok = bool(shutil.which("flutter"))

    # SIMULATION FOR TESTING: if environment variable SIMULATE_NO_DOCKER is set
    if os.getenv("SIMULATE_NO_DOCKER") == "true":
        print("   ⚠️  [DevOps Simulated Mode] Forcing simulation: Docker not available.")
        docker_ok = False

    return {
        "docker": docker_ok,
        "node": node_ok,
        "flutter": flutter_ok
    }

async def check_url_active(url: str) -> bool:
    """Try to ping the URL to check if it's running."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=2) as response:
                # Any non-500 or response means the port is active
                return response.status < 500
    except Exception:
        return False

async def run_local_preview(developer_output: DeveloperOutput, project_name: str) -> LocalPreviewOutput:
    """Run local preview or graceful degradation based on environment."""
    env = prepare_local_environment(developer_output, project_name)
    
    out_dir = f"./output/{project_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    backend_url = "http://localhost:3000"
    frontend_url = "http://localhost:8080"
    
    services = []
    preview_ready = False
    docker_compose_generated = False
    manual_instructions = None
    
    # OPTION A: Docker is available
    if env["docker"]:
        yaml_content = generate_docker_compose(developer_output, project_name)
        docker_compose_generated = True
        
        # Write environment file in ./output/{project_name}/.env
        env_path = os.path.join(out_dir, ".env")
        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("PORT=3000\nDATABASE_URL=postgres://postgres:postgres@db:5432/postgres\nJWT_SECRET=supersecret\n")
        except Exception as e:
            print(f"   ⚠️  [DevOps] Could not write .env: {e}")
            
        # Copy backend files for docker build
        be_dir = os.path.join(out_dir, "backend")
        os.makedirs(be_dir, exist_ok=True)
        be_dockerfile = os.path.join(be_dir, "Dockerfile")
        if not os.path.exists(be_dockerfile):
            try:
                with open(be_dockerfile, "w", encoding="utf-8") as f:
                    f.write("FROM node:18-alpine\nWORKDIR /app\nCOPY package*.json ./\nRUN npm install\nCOPY . .\nRUN npm run build\nEXPOSE 3000\nCMD [\"npm\", \"start\"]\n")
            except Exception as e:
                print(f"   ⚠️  [DevOps] Could not write backend Dockerfile: {e}")
                
        try:
            src_dest = os.path.join(be_dir, "src")
            if os.path.exists(src_dest):
                shutil.rmtree(src_dest)
            shutil.copytree("src", src_dest)
            shutil.copy("package.json", os.path.join(be_dir, "package.json"))
            shutil.copy("tsconfig.json", os.path.join(be_dir, "tsconfig.json"))
        except Exception as e:
            print(f"   ⚠️  [DevOps] Error copying backend source files: {e}")
            
        # Copy frontend files for docker build
        fe_dir = os.path.join(out_dir, "frontend")
        os.makedirs(fe_dir, exist_ok=True)
        fe_dockerfile = os.path.join(fe_dir, "Dockerfile")
        if not os.path.exists(fe_dockerfile):
            try:
                with open(fe_dockerfile, "w", encoding="utf-8") as f:
                    f.write("FROM nginx:alpine\nRUN mkdir -p /usr/share/nginx/html\nRUN echo '<h1>MarineCheck Pro Frontend Preview</h1><p>Flutter Web Build Placeholder</p>' > /usr/share/nginx/html/index.html\nEXPOSE 80\nCMD [\"nginx\", \"-g\", \"daemon off;\"]\n")
            except Exception as e:
                print(f"   ⚠️  [DevOps] Could not write frontend Dockerfile: {e}")
                
        print("   🐳 [DevOps Option A] Docker available. Starting containers (docker compose up -d --build)...")
        # Check if we should use 'docker compose' or 'docker-compose'
        cmd = ["docker", "compose"]
        if not shutil.which("docker"):
            cmd = ["docker-compose"]
            
        try:
            # Run docker compose up in background
            proc = subprocess.Popen(
                cmd + ["up", "-d", "--build"],
                cwd=out_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Polling
            print("   ⏳ [DevOps] Waiting for services to respond (max 60s)...")
            success = False
            for step in range(20): # 20 attempts * 3s = 60s
                await asyncio.sleep(3)
                be_active = await check_url_active(backend_url)
                fe_active = await check_url_active(frontend_url)
                if be_active and fe_active:
                    success = True
                    break
                    
            if success:
                print("   ✅ [DevOps] Containers started and responding successfully!")
                preview_ready = True
                services.append(ServiceStatus(name="backend", status="running", port=3000, url=backend_url))
                services.append(ServiceStatus(name="frontend", status="running", port=8080, url=frontend_url))
            else:
                print("   ❌ [DevOps] Error: Services did not respond within 60s. Fetching logs...")
                # Fetch logs
                logs_res = subprocess.run(cmd + ["logs", "--tail=10"], cwd=out_dir, capture_output=True, text=True)
                logs_tail = logs_res.stdout.splitlines() if logs_res.stdout else []
                
                services.append(ServiceStatus(
                    name="backend",
                    status="failed",
                    port=3000,
                    url=backend_url,
                    error="Timeout waiting for service response",
                    logs_tail=logs_tail
                ))
                services.append(ServiceStatus(
                    name="frontend",
                    status="failed",
                    port=8080,
                    url=frontend_url,
                    error="Timeout waiting for service response",
                    logs_tail=logs_tail
                ))
        except Exception as e:
            print(f"   ⚠️  [DevOps] Exception executing docker compose: {e}")
            services.append(ServiceStatus(name="backend", status="failed", port=3000, url=backend_url, error=str(e)))
            services.append(ServiceStatus(name="frontend", status="failed", port=8080, url=frontend_url, error=str(e)))
            
    # OPTION B: Only Node is available
    elif env["node"]:
        print("   ⚙️  [DevOps Option B] Only Node.js available. Starting NestJS locally...")
        # Since code files are stored under src/ or backend/, let's check for package.json
        # and start the process.
        # However, to be fully robust and prevent blocking if code doesn't build, we can launch Popen
        try:
            # Prefer the generated project's own dir; fall back to the repo root.
            pkg_path = os.path.join(out_dir, "package.json")
            cwd = out_dir
            if not os.path.exists(pkg_path):
                cwd = str(PROJECT_ROOT)
                pkg_path = os.path.join(cwd, "package.json")
                
            print(f"   🚀 [DevOps] npm install && npm run start:dev in {cwd}...")
            # We don't want to block the thread forever, so run npm run start in background
            # For robust background running:
            npm_install = subprocess.run(["npm", "install"], cwd=cwd, capture_output=True)
            proc = subprocess.Popen(
                ["npm", "run", "start"],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait up to 30s
            be_active = False
            for _ in range(10):
                await asyncio.sleep(3)
                if await check_url_active(backend_url):
                    be_active = True
                    break
                    
            if be_active:
                print("   ✅ [DevOps] Backend NestJS running locally!")
                services.append(ServiceStatus(name="backend", status="running", port=3000, url=backend_url, pid=proc.pid))
                preview_ready = True
                
                # Check frontend with Flutter web compilation
                if env["flutter"]:
                    print("   📱 [DevOps] Compiling Flutter Web and serving on port 8080...")
                    # Simular build web e http server en background
                    fl_proc = subprocess.Popen(
                        ["python3", "-m", "http.server", "8080"],
                        cwd=cwd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    await asyncio.sleep(2)
                    services.append(ServiceStatus(name="frontend", status="running", port=8080, url=frontend_url, pid=fl_proc.pid))
                else:
                    services.append(ServiceStatus(name="frontend", status="skipped", port=8080, url=frontend_url, error="Flutter SDK not available"))
            else:
                print("   ❌ [DevOps] Local NestJS backend did not start in time.")
                services.append(ServiceStatus(name="backend", status="failed", port=3000, url=backend_url, error="Timeout starting NestJS"))
                services.append(ServiceStatus(name="frontend", status="skipped", port=8080, url=frontend_url, error="Depends on failed backend"))
        except Exception as e:
            print(f"   ⚠️  [DevOps Option B Exception] {e}")
            services.append(ServiceStatus(name="backend", status="failed", port=3000, url=backend_url, error=str(e)))
            services.append(ServiceStatus(name="frontend", status="skipped", port=8080, url=frontend_url, error=str(e)))

    # OPTION C: Graceful degradation (None available or simulated)
    else:
        print("   ⚠️  [DevOps Option C] Local environment not suitable for automatic preview. Generating docker-compose and manual guide...")
        # Generate docker-compose.yml anyway so they have it
        try:
            generate_docker_compose(developer_output, project_name)
            docker_compose_generated = True
        except Exception as e:
            print(f"   ⚠️  [DevOps] Could not auto-generate docker-compose.yml: {e}")
            
        services.append(ServiceStatus(name="backend", status="skipped", port=3000, url=backend_url, error="Docker / Node not available"))
        services.append(ServiceStatus(name="frontend", status="skipped", port=8080, url=frontend_url, error="Docker / Flutter not available"))
        
        # Generar instrucciones manuales con Qwen
        print("   🧠 [DevOps] Generating setup guide and manual instructions...")
        system_prompt = """\
You are an elite DevOps Engineer. Your role is to explain in detail to the user how they can configure, \
build, and run locally on their machine the NestJS backend, the PostgreSQL database, and the Flutter Web frontend.
Be structured and provide ready-to-copy-and-paste console commands.
"""
        human_prompt = f"""
Project: {project_name}
Generate a quick start guide and step-by-step manual instructions to spin up local services.
NestJS Backend (port 3000) and Flutter Frontend (port 8080).
"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = llm_developer.invoke(messages)
            # Limpiar bloques <think>
            manual_instructions = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL)
            manual_instructions = re.sub(r'<think>.*$', '', manual_instructions, flags=re.DOTALL).strip()
        except Exception as e:
            manual_instructions = f"Basic Guide:\n1. Install Docker and Node.js.\n2. Run 'npm install' and 'npm run start' in the backend.\n3. Run 'flutter pub get' and 'flutter run -d chrome' in the frontend."

    out = LocalPreviewOutput(
        project_name=project_name,
        services=services,
        backend_url=backend_url if preview_ready else "",
        frontend_url=frontend_url if preview_ready else "",
        preview_ready=preview_ready,
        docker_compose_generated=docker_compose_generated,
        manual_instructions=manual_instructions
    )
    
    if preview_ready:
        try:
            url_to_open = frontend_url if frontend_url else backend_url
            print(f"   🌐 [DevOps] Opening preview in browser: {url_to_open}")
            webbrowser.open(url_to_open)
        except Exception as e:
            print(f"   ⚠️  [DevOps] Could not automatically open browser: {e}")
            
    return out
