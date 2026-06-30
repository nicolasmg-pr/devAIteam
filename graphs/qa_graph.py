"""LangGraph graph for the QA Agent pipeline.

Flow: START → test_generator_node + code_reviewer_node (parallel) → merge_node → formatter → END
"""

from __future__ import annotations

import traceback
from typing import Optional, Annotated

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field


def _keep_first(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Reducer: keep the first non-None error from concurrent fan-out branches."""
    return a or b

from agents.pm_agent import PMOutput
from agents.developer_agent import DeveloperOutput
from agents.architect_agent import _run_async_in_thread
from agents.qa_agent import (
    QAOutput,
    run_test_generator,
    run_code_reviewer,
    merge_qa_outputs,
    run_e2e_tests,
    generate_test_files_with_filesystem,
)


# ── Graph state ──────────────────────────────────────────────────────────────

class QAState(BaseModel):
    pm_output: PMOutput = Field(..., description="PM output from stage 1")
    developer_output: DeveloperOutput = Field(
        ..., description="Developer output from stage 3"
    )
    test_output: Optional[dict] = Field(
        default=None, description="Test generator output"
    )
    review_output: Optional[dict] = Field(
        default=None, description="Code reviewer output"
    )
    qa_output: Optional[QAOutput] = Field(
        default=None, description="Combined QA output"
    )
    e2e_output: Optional[dict] = Field(
        default=None, description="E2E test results"
    )
    test_saved: bool = Field(
        default=False, description="Whether test files were successfully saved"
    )
    error: Annotated[Optional[str], _keep_first] = Field(
        default=None, description="Error message if any"
    )


# ── Nodes ────────────────────────────────────────────────────────────────────

def test_generator_node(state: QAState) -> dict:
    """Execute the Test Generator sub-agent."""
    try:
        print("   🧪 Test Generator generating test cases and code...")
        result = run_test_generator(state.pm_output, state.developer_output)
        return {"test_output": result}
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"Test Generator error: {exc}"}


def code_reviewer_node(state: QAState) -> dict:
    """Execute the Code Reviewer sub-agent."""
    try:
        print("   🔍 Code Reviewer analyzing code quality...")
        result = run_code_reviewer(state.developer_output)
        return {"review_output": result}
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"Code Reviewer error: {exc}"}


def merge_node(state: QAState) -> dict:
    """Combine test + review outputs into a single QAOutput."""
    if state.error:
        return {}

    if state.test_output is None or state.review_output is None:
        return {"error": "One or both QA sub-agents did not generate output."}

    project_name = state.developer_output.project_name
    qa_output = merge_qa_outputs(project_name, state.test_output, state.review_output)
    return {"qa_output": qa_output}


def e2e_node(state: QAState) -> dict:
    """Execute E2E verification tests using Playwright or HTTP fallback."""
    if state.error or state.qa_output is None:
        return {}
        
    print("🎭 [QA E2E Node] Starting navigation and verification tests...")
    try:
        e2e_res = _run_async_in_thread(run_e2e_tests(state.developer_output))
        return {"e2e_output": e2e_res}
    except Exception as exc:
        print(f"⚠️ [QA E2E Node] Error in E2E tests: {exc}")
        return {"e2e_output": {"error": str(exc)}}


def save_tests_node(state: QAState) -> dict:
    """Save generated test files using filesystem tools."""
    if state.error or state.qa_output is None:
        return {}
        
    print("💾 [QA Save Tests Node] Saving test files...")
    try:
        _run_async_in_thread(generate_test_files_with_filesystem(state.qa_output.test_files))
        return {"test_saved": True}
    except Exception as exc:
        print(f"⚠️ [QA Save Tests Node] Error saving tests: {exc}")
        return {"test_saved": False}


def format_node(state: QAState) -> dict:
    """Pretty-print the QA output with structured formatting."""
    if state.error:
        print(f"\n❌ Error during QA Agent execution:\n{state.error}")
        return {}

    output = state.qa_output
    if output is None:
        print("\n⚠️  No QA output was generated.")
        return {}

    print("\n" + "=" * 60)
    print(f"🧪 QA REPORT: {output.project_name}")
    print("=" * 60)

    # ── Quality score bar ────────────────────────────────────────────
    score = output.quality_score
    filled = round(score / 10)
    empty = 10 - filled
    bar = "█" * filled + "░" * empty

    if score >= 80:
        score_emoji = "🟢"
    elif score >= 60:
        score_emoji = "🟡"
    else:
        score_emoji = "🔴"

    print(f"\n{score_emoji} QUALITY SCORE: {score}/100  [{bar}]")
    print(f"📝 {output.quality_summary}")

    # ── Test cases ───────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"🧪 TEST CASES ({len(output.test_cases)} total):")
    print(f"{'─' * 60}")

    priority_icons = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢",
    }

    # Group by priority order
    priority_order = ["critical", "high", "medium", "low"]
    for prio in priority_order:
        cases = [tc for tc in output.test_cases if tc.priority == prio]
        if not cases:
            continue
        icon = priority_icons.get(prio, "⚪")
        print(f"\n   {icon} {prio.upper()} ({len(cases)}):")
        for tc in cases:
            print(f"      {tc.id} [{tc.test_type}] — {tc.title}  (→ {tc.user_story_id})")

    # ── Test files ───────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"📁 GENERATED TEST FILES ({len(output.test_files)} files):")
    print(f"{'─' * 60}")
    for tf in output.test_files:
        print(f"   📄 {tf.path}/{tf.filename} — {tf.description}")

    # ── Code issues ──────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"🐛 CODE ISSUES ({len(output.code_issues)} found):")
    print(f"{'─' * 60}")

    severity_icons = {
        "critical": "🔴 CRITICAL",
        "major": "🟠 MAJOR",
        "minor": "🟡 MINOR",
        "info": "🔵 INFO",
    }

    severity_order = ["critical", "major", "minor", "info"]
    for sev in severity_order:
        issues = [ci for ci in output.code_issues if ci.severity == sev]
        if not issues:
            continue
        icon_label = severity_icons.get(sev, "⚪ UNKNOWN")
        print(f"\n   {icon_label} ({len(issues)}):")
        for issue in issues:
            print(f"      [{issue.category}] {issue.file}")
            print(f"         {issue.description}")
            print(f"         💡 {issue.suggestion}")

    # ── E2E Results ──────────────────────────────────────────────────
    if state.e2e_output:
        print(f"\n{'─' * 60}")
        print("🎭 E2E TESTS / VERIFICATION RESULTS:")
        print(f"{'─' * 60}")
        for url, res in state.e2e_output.items():
            if not isinstance(res, dict):
                print(f"   ❌ {url} : {res}")
                continue
            status = res.get("status")
            if status == "server_offline":
                status_icon = "⚠️"
                status_text = "offline (skipped - server not running yet)"
            elif status in ["success", "simulated"]:
                status_icon = "✅"
                status_text = status
            else:
                status_icon = "❌"
                status_text = f"failed ({status})"
                
            print(f"   {status_icon} {url} : Status={status_text}")
            if status != "server_offline" and "http_code" in res:
                print(f"      HTTP Code: {res.get('http_code')}")
            if "screenshot_path" in res:
                print(f"      Visual Capture: {res.get('screenshot_path')}")
                
    if state.test_saved:
        print(f"\n💾 Test files saved to: ./output/tests/")

    print("\n" + "=" * 60 + "\n")

    return {}


# ── Graph builder ────────────────────────────────────────────────────────────

def build_qa_graph():
    """Build and compile the QA LangGraph pipeline with parallel fan-out."""
    graph = StateGraph(QAState)

    graph.add_node("test_generator_node", test_generator_node)
    graph.add_node("code_reviewer_node", code_reviewer_node)
    graph.add_node("merge_node", merge_node)
    graph.add_node("e2e_node", e2e_node)
    graph.add_node("save_tests_node", save_tests_node)
    graph.add_node("formatter", format_node)

    # Fan-out: START triggers both test generator and code reviewer in parallel
    graph.add_edge(START, "test_generator_node")
    graph.add_edge(START, "code_reviewer_node")

    # Fan-in: both converge at merge_node
    graph.add_edge("test_generator_node", "merge_node")
    graph.add_edge("code_reviewer_node", "merge_node")

    # E2E node and file saving
    graph.add_edge("merge_node", "e2e_node")
    graph.add_edge("e2e_node", "save_tests_node")
    graph.add_edge("save_tests_node", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()


# Pre-built graph ready for import
qa_graph = build_qa_graph()
