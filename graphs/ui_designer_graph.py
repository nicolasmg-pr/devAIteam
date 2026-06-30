import os
from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END

from agents.architect_agent import ArchitectOutput
from agents.ui_designer_agent import UIDesignerOutput, run_ui_designer_agent

# ── State Definition ───────────────────────────────────────────────────────────

class UIDesignerState(TypedDict):
    architect_output: ArchitectOutput
    ui_designer_output: UIDesignerOutput | None

# ── Nodes ──────────────────────────────────────────────────────────────────────

def ui_designer_node(state: UIDesignerState) -> dict:
    """Executes the UI Designer Agent using Google Stitch."""
    print("🎨 Starting UI Designer Agent...")
    stitch_api_key = os.environ.get("STITCH_API_KEY")
    if not stitch_api_key:
        print("⚠️  [UI Designer] STITCH_API_KEY was not found in the environment variables.")
    try:
        output = run_ui_designer_agent(state["architect_output"], stitch_api_key)
        return {"ui_designer_output": output}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"⚠️  [UI Designer] Agent failed: {exc}. Continuing without UI.")
        return {"ui_designer_output": None}

# ── Graph Construction ─────────────────────────────────────────────────────────

builder = StateGraph(UIDesignerState)
builder.add_node("ui_designer_node", ui_designer_node)

builder.add_edge(START, "ui_designer_node")
builder.add_edge("ui_designer_node", END)

ui_designer_graph = builder.compile()
