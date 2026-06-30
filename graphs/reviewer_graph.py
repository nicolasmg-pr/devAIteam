"""LangGraph graph for the Reviewer Agent with real Human-in-the-Loop."""

from __future__ import annotations

import traceback
import uuid
from typing import Optional, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from config.paths import project_dir, safe_join, UnsafePathError
from agents.developer_agent import DeveloperOutput
from agents.qa_agent import QAOutput
from agents.architect_agent import _run_async_in_thread
from agents.reviewer_agent import (
    ReviewRound,
    HumanDecision,
    ReviewerOutput,
    CodeChange,
    run_reviewer,
    create_github_pr,
)


# ── Graph state ──────────────────────────────────────────────────────────────

class ReviewerState(BaseModel):
    developer_output: DeveloperOutput
    qa_output: QAOutput
    
    # We use list to append state across rounds
    rounds: Annotated[list[ReviewRound], operator.add] = Field(default_factory=list)
    human_decisions: Annotated[list[HumanDecision], operator.add] = Field(default_factory=list)
    
    current_round: Optional[ReviewRound] = None
    human_decision: Optional[HumanDecision] = None
    final_output: Optional[ReviewerOutput] = None
    github_output: Optional[dict] = Field(default=None, description="GitHub PR creation output")
    max_rounds: int = Field(default=3)
    error: Optional[str] = None


# ── Nodes ────────────────────────────────────────────────────────────────────

def review_node(state: ReviewerState) -> dict:
    """Run the reviewer agent to produce a ReviewRound."""
    try:
        round_num = len(state.rounds) + 1
        print(f"   🔍 Reviewer Agent generating review (Round {round_num})...")
        
        last_decision = None
        if state.human_decisions:
            last_decision = state.human_decisions[-1].feedback
 
        new_round = run_reviewer(
            state.developer_output,
            state.qa_output,
            state.rounds,
            last_decision
        )
        
        return {
            "current_round": new_round,
            "rounds": [new_round], # Annotated with operator.add
            "human_decision": None # Clear previous decision
        }
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"Reviewer Agent error: {exc}"}


def format_review_node(state: ReviewerState) -> dict:
    """Format and print the current review round."""
    if state.error:
        print(f"\n❌ Error in Code Review:\n{state.error}")
        return {}

    r = state.current_round
    if not r:
        return {}

    emoji = "🟢" if r.overall_verdict in {"approved", "approved_with_suggestions"} else "🔴"
    
    print("\n" + "━" * 40)
    print(f"👁  CODE REVIEW — Round {r.round_number}")
    print("━" * 40)

    if getattr(r, "think_content", None):
        print("🧠 REVIEWER REASONING:")
        for line in r.think_content.strip().split("\n"):
            print(f"  > {line}")
        print()

    print(f"Verdict: {emoji} {r.overall_verdict} — {r.verdict_reason}")
    print(f"🔴 Blocking: {r.blocking_count} | 🟠 Important: {r.important_count} | 💡 Suggestions: {r.suggestion_count}\n")

    severities = ["blocking", "important", "suggestion"]
    icons = {"blocking": "🔴 BLOCKING", "important": "🟠 IMPORTANT", "suggestion": "💡 SUGGESTION"}

    for sev in severities:
        comments = [c for c in r.comments if c.severity == sev]
        for c in comments:
            print(f"[{icons[sev]}]")
            print(f"  📄 {c.file} ({c.line_reference})")
            print(f"  {c.comment}")
            if c.change:
                print(f"    BEFORE:  {c.change.original_snippet}")
                print(f"    AFTER:   {c.change.proposed_snippet}")
                print(f"    Reason:  {c.change.reason}")
            print()
    print("━" * 40)
    return {}


def human_input_node(state: ReviewerState) -> dict:
    """Pause the graph and ask for human input."""
    # This invokes the langgraph interrupt, suspending the graph execution.
    # The string passed to interrupt is returned to the client (main.py).
    human_input = interrupt("Waiting for human input...")

    # When resumed, human_input receives the payload from Command(resume=value)
    human_input = str(human_input).strip()

    # Parse the leading token strictly: approve only on exactly a/s/f.
    # 'r' and anything unrecognized count as not-approved.
    tokens = human_input.lower().split()
    leading = tokens[0] if tokens else ""
    approved = leading in {"a", "s", "f"}

    from datetime import datetime
    decision = HumanDecision(
        approved=approved,
        feedback=human_input,
        timestamp=datetime.now().isoformat()
    )
    
    return {
        "human_decision": decision,
        "human_decisions": [decision] # Annotated with operator.add
    }


def should_continue_node(state: ReviewerState) -> str:
    """Route based on human decision and max rounds."""
    if state.error:
        return "finalize_node"
        
    decision = state.human_decision
    if not decision:
        # Fallback if no decision found, though it shouldn't happen
        return "finalize_node"
        
    if decision.approved:
        return "finalize_node"
        
    if len(state.rounds) >= state.max_rounds:
        return "finalize_node"
        
    return "review_node"


import os
import subprocess
from agents.mcp_client import ThreadSafeMCPClient

def _apply_code_change(change: CodeChange, project_name: str):
    """Locate the original snippet in the generated project's file and replace it.

    Changes are applied to the GENERATED project's directory under output/, never
    to the platform's own repo.
    """
    # Slugify the project name the same way main.py does.
    slug = project_name.lower().replace(" ", "-").replace("_", "-")
    try:
        base = project_dir(slug)
    except UnsafePathError as e:
        print(f"⚠️  [Reviewer] Unsafe project name, skipping change: {e}")
        return

    try:
        full_path = safe_join(base, change.file)
    except UnsafePathError as e:
        print(f"⚠️  [Reviewer] Unsafe file path, skipping change: {e}")
        return

    if not os.path.exists(full_path):
        print(f"⚠️  [Reviewer] File not found to apply change: {full_path}")
        return

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        orig = change.original_snippet.strip()
        prop = change.proposed_snippet

        if orig in content:
            # Replace only the FIRST occurrence.
            content = content.replace(orig, prop, 1)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ [Reviewer] Change successfully applied locally in: {change.file}")
        else:
            # Simple fallback if exact match fails: search without leading/trailing whitespace
            lines = content.splitlines()
            orig_lines = orig.splitlines()
            # Try to match the subset of lines
            found = False
            for i in range(len(lines) - len(orig_lines) + 1):
                chunk = "\n".join(lines[i:i+len(orig_lines)])
                if chunk.strip() == orig:
                    lines[i:i+len(orig_lines)] = prop.splitlines()
                    content = "\n".join(lines)
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"✅ [Reviewer] Change applied locally (soft match) in: {change.file}")
                    found = True
                    break
            if not found:
                print(f"⚠️  [Reviewer] Could not find the exact original snippet in: {change.file}")
    except Exception as e:
        print(f"❌ [Reviewer] Error applying local change in {change.file}: {e}")


def _get_git_repo_name() -> Optional[tuple[str, str]]:
    """Get GitHub owner and repo from local git remote url."""
    try:
        url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], text=True).strip()
        if "github.com" in url:
            parts = url.split("github.com")[-1].strip(":/").replace(".git", "").split("/")
            if len(parts) >= 2:
                return parts[-2], parts[-1]
    except Exception:
        pass
    return None


def finalize_node(state: ReviewerState) -> dict:
    """Build the final output and apply code changes locally."""
    if state.error:
        return {}

    final_status = "rejected"
    if state.human_decisions:
        last_decision = state.human_decisions[-1]
        if last_decision.approved:
            # Check the AI's verdict from the last round
            last_round = state.rounds[-1]
            if last_round.overall_verdict == "approved_with_suggestions":
                final_status = "approved_with_suggestions"
            else:
                final_status = "approved"

    # Gather all changes proposed
    total_changes = 0
    approved_changes = []
    for r in state.rounds:
        for c in r.comments:
            if c.change:
                total_changes += 1
                if final_status in ["approved", "approved_with_suggestions"]:
                    approved_changes.append(c.change)

    # Apply approved changes locally
    if final_status in ["approved", "approved_with_suggestions"] and approved_changes:
        print(f"💾 Applying {len(approved_changes)} approved changes to the workspace...")
        for change in approved_changes:
            _apply_code_change(change, state.developer_output.project_name)

    out = ReviewerOutput(
        project_name=state.developer_output.project_name,
        rounds=state.rounds,
        human_decisions=state.human_decisions,
        final_status=final_status,
        total_changes_proposed=total_changes,
        approved_changes=approved_changes
    )
    return {"final_output": out}


def github_node(state: ReviewerState) -> dict:
    """Create GitHub branch, push changes, and open PR if approved."""
    if state.error or state.final_output is None:
        return {}
        
    if state.final_output.final_status not in ["approved", "approved_with_suggestions"]:
        return {}
        
    if not state.final_output.approved_changes:
        return {}
        
    print("🐙 [Github Node] Creating branch and opening Pull Request on GitHub...")
    try:
        github_res = _run_async_in_thread(create_github_pr(state.final_output, state.developer_output))
        return {"github_output": github_res}
    except Exception as exc:
        print(f"⚠️ [Github Node] GitHub integration failed: {exc}")
        return {"github_output": {"status": "failed", "error": str(exc)}}


def format_final_node(state: ReviewerState) -> dict:
    """Print the final summary of the Code Review phase."""
    if state.error:
        return {}
        
    out = state.final_output
    if not out:
        return {}

    print("\n╔══════════════════════════════════════════╗")
    print("║          CODE REVIEW COMPLETED           ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║ Status: {out.final_status.ljust(32)} ║")
    print(f"║ Rounds: {str(len(out.rounds)).ljust(2)} of {str(state.max_rounds).ljust(25)} ║")
    print(f"║ Changes proposed: {str(out.total_changes_proposed).ljust(22)} ║")
    print(f"║ Changes approved: {str(len(out.approved_changes)).ljust(22)} ║")
    print(f"║ Human decisions: {str(len(out.human_decisions)).ljust(22)} ║")
    
    if state.github_output:
        res = state.github_output
        if res.get("status") in ["success", "simulated"]:
            pr_val = res.get("pr_url", "")
            branch_val = res.get("branch", "")
            print(f"║ PR: {pr_val[:35].ljust(36)} ║")
            print(f"║ Branch: {branch_val[:31].ljust(32)} ║")
        else:
            print("║ PR: Failed to create on GitHub          ║")
    else:
        print("║ PR: GitHub push was not required         ║")
        
    print("╚══════════════════════════════════════════╝\n")
    return {}


# ── Graph builder ────────────────────────────────────────────────────────────

def build_reviewer_graph():
    graph = StateGraph(ReviewerState)

    graph.add_node("review_node", review_node)
    graph.add_node("format_review_node", format_review_node)
    graph.add_node("human_input_node", human_input_node)
    graph.add_node("finalize_node", finalize_node)
    graph.add_node("github_node", github_node)
    graph.add_node("format_final_node", format_final_node)

    graph.add_edge(START, "review_node")
    graph.add_edge("review_node", "format_review_node")
    graph.add_edge("format_review_node", "human_input_node")
    
    # Conditional edge from human input
    graph.add_conditional_edges(
        "human_input_node",
        should_continue_node,
        {
            "finalize_node": "finalize_node",
            "review_node": "review_node"
        }
    )
    
    graph.add_edge("finalize_node", "github_node")
    graph.add_edge("github_node", "format_final_node")
    graph.add_edge("format_final_node", END)

    # Use MemorySaver to allow pausing and resuming with interrupt()
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

# Pre-built graph ready for import
reviewer_graph = build_reviewer_graph()


def make_reviewer_config() -> dict:
    """Return a fresh reviewer config with a unique thread_id per run.

    Each pipeline run must use its own checkpoint thread so concurrent or
    sequential runs do not collide on a shared MemorySaver thread.
    """
    return {"configurable": {"thread_id": f"review-{uuid.uuid4()}"}}


# Backward-compat default config for imports; main.py calls the factory per run.
reviewer_config = make_reviewer_config()
