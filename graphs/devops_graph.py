import sys
import traceback
from typing import Optional
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from agents.developer_agent import DeveloperOutput
from agents.devops_agent import LocalPreviewOutput, run_local_preview
from agents.architect_agent import _run_async_in_thread

class DevOpsState(BaseModel):
    developer_output: DeveloperOutput = Field(..., description="Developer output from stage 3")
    project_name: str = Field(..., description="Lowercase project name with dashes")
    local_preview_output: Optional[LocalPreviewOutput] = Field(default=None, description="Preview agent output")
    error: Optional[str] = Field(default=None, description="Error message if any")

def preview_node(state: DevOpsState) -> dict:
    """Execute run_local_preview, capturing exceptions."""
    try:
        # run_local_preview is async, so run it using _run_async_in_thread
        preview_output = _run_async_in_thread(run_local_preview(state.developer_output, state.project_name))
        return {"local_preview_output": preview_output}
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"DevOps Agent error: {exc}"}

def format_node(state: DevOpsState) -> dict:
    """Format and print local preview status."""
    if state.error:
        print(f"\n❌ Error in DevOps Agent:\n{state.error}")
        return {}

    o = state.local_preview_output
    if not o:
        return {}

    print("\n" + "━" * 40)
    print(f"🚀 LOCAL PREVIEW — {state.project_name}")
    print("━" * 40)

    for s in o.services:
        status_icon = "✅" if s.status == "running" else ("❌" if s.status == "failed" else "⏳")
        pid_info = f" (PID: {s.pid})" if s.pid else ""
        print(f"{status_icon} {s.name}: {s.url}{pid_info}")
        if s.error:
            print(f"   ⚠️  {s.error}")
        if s.logs_tail:
            print("   Logs (last 3 lines):")
            for line in s.logs_tail[-3:]:
                print(f"     | {line}")

    print()
    if o.preview_ready:
        print(f"🌐 App available at: {o.frontend_url}")
        print(f"🔌 API available at: {o.backend_url}")
        print(f"📂 Code in: ./output/{state.project_name}/")
        if o.docker_compose_generated:
            print(f"🐳 Docker Compose: ./output/{state.project_name}/docker-compose.yml")
        print()
        print("┌─────────────────────────────────────────┐")
        print("│  The app is running locally.            │")
        print("│  Test it in your browser.               │")
        print("│  When you are ready to deploy:          │")
        print("│                                         │")
        print(f"│  devAIteam deploy {state.project_name}        │")
        print("└─────────────────────────────────────────┘")
    else:
        print("⚠️  Automatic preview not available.")
        if o.manual_instructions:
            print("\n📋 MANUAL SETUP AND EXECUTION INSTRUCTIONS:")
            print(o.manual_instructions)
            
    print("━" * 40)
    return {}

def human_preview_node(state: DevOpsState) -> dict:
    """Standard Python input() blocking prompt (not a LangGraph interrupt)."""
    # Skip the blocking prompt in non-interactive environments (e.g. CI, pipes).
    if not sys.stdin.isatty():
        print("\n⏩ Non-interactive session: continuing automatically.")
        return {}
    # Wait for the user to press ENTER
    try:
        input("\n⏸  Press ENTER when you have tested the app to continue...")
    except (KeyboardInterrupt, EOFError):
        print("\nContinuing automatically...")
    return {}

def build_devops_graph():
    graph = StateGraph(DevOpsState)
    
    graph.add_node("preview_node", preview_node)
    graph.add_node("format_node", format_node)
    graph.add_node("human_preview_node", human_preview_node)
    
    graph.add_edge(START, "preview_node")
    graph.add_edge("preview_node", "format_node")
    graph.add_edge("format_node", "human_preview_node")
    graph.add_edge("human_preview_node", END)
    
    return graph.compile()

devops_graph = build_devops_graph()
