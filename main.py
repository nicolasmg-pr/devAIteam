"""AI Dev Team – Entry point.

Invokes the full pipeline: PM Agent → Architect Agent → Developer Agent → QA Agent → Reviewer Agent → DevOps Agent.
"""

import os
import sys

# Ensure the project root directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv()

from graphs.pm_graph import pm_graph
from graphs.architect_graph import architect_graph
from graphs.ui_designer_graph import ui_designer_graph
from graphs.developer_graph import developer_graph
from graphs.qa_graph import qa_graph
from graphs.reviewer_graph import reviewer_graph, make_reviewer_config
from graphs.devops_graph import devops_graph

from langgraph.types import Command

def main() -> None:
    args = sys.argv[1:]

    # Without arguments: show help and request interactive input
    if not args:
        print("""
  ╔══════════════════════════════════════════════════════╗
  ║                    devAIteam                        ║
  ║         Your local AI development team             ║
  ╠══════════════════════════════════════════════════════╣
  ║  GENERATE project:                                  ║
  ║    devAIteam "describe your app"                     ║
  ║                                                     ║
  ║  MANAGE projects:                                   ║
  ║    devAIteam list                  list projects    ║
  ║    devAIteam deploy <project>      deploy project   ║
  ║    devAIteam rm <project>          delete local     ║
  ║    devAIteam rm <project> --all    delete all       ║
  ║    devAIteam prune                 clean registry   ║
  ║  DIAGNOSTICS & SYSTEM:                              ║
  ║    devAIteam doctor                check MCP tools  ║
  ║    devAIteam clean                 clean system RAM ║
  ╚══════════════════════════════════════════════════════╝
      """)
        print("🤖 Describe your project (or CTRL+C to exit):")
        try:
            requirement = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return
        if not requirement:
            return
        args = [requirement]

    command = args[0].lower()

    # devAIteam clean
    if command == "clean":
        from deploy.clean_runner import run_clean_command
        run_clean_command()
        return

    # devAIteam doctor / check / mcp
    if command in ["doctor", "check", "mcp"]:
        from deploy.mcp_runner import run_mcp_check
        run_mcp_check()
        return

    # devAIteam list
    if command == "list":
        from deploy.list_runner import run_list_command
        run_list_command()
        return

    # devAIteam deploy <project>
    if command == "deploy":
        if len(args) < 2:
            print("❌ Usage: devAIteam deploy <project_name>")
            return
        from deploy.deploy_runner import run_deploy_command
        run_deploy_command(args[1])
        return

    # devAIteam rm <project> [--all]
    if command == "rm":
        if len(args) < 2:
            print("❌ Usage: devAIteam rm <project_name> [--all]")
            return
        delete_all = "--all" in args
        from deploy.rm_runner import run_rm_command
        run_rm_command(args[1], delete_all=delete_all)
        return

    # devAIteam prune
    if command == "prune":
        from deploy.rm_runner import run_prune_command
        run_prune_command()
        return

    # If it is not a CLI command, treat it as the user's original requirement
    requirement = " ".join(args)

    # ── Stage 1: PM Agent ────────────────────────────────────────────
    print("🤖 Running PM Agent...")
    pm_result = pm_graph.invoke({"requirement": requirement})

    if pm_result.get("error"):
        print("❌ PM Agent failed. Aborting pipeline.")
        return

    # ── Stage 2: Architect Agent ─────────────────────────────────────
    print("\n🏗️  Running Architect Agent...")
    arch_result = architect_graph.invoke({"pm_output": pm_result["pm_output"]})

    if arch_result.get("error"):
        print("❌ Architect Agent failed. Aborting pipeline.")
        return

    # ── Stage 2.5: UI Designer Agent ─────────────────────────────────
    print("\n🎨 Running UI Designer Agent...")
    ui_result = ui_designer_graph.invoke({"architect_output": arch_result["architect_output"]})

    if "ui_designer_output" not in ui_result or ui_result["ui_designer_output"] is None:
        print("⚠️ UI Designer Agent failed or did not return output. Continuing without UI.")
        ui_output = None
    else:
        ui_output = ui_result["ui_designer_output"]

    # ── Stage 3: Developer Agent (backend + frontend in parallel) ────
    print("\n💻 Running Developer Agent (backend + frontend in parallel)...")
    dev_result = developer_graph.invoke(
        {
            "architect_output": arch_result["architect_output"],
            "ui_designer_output": ui_output
        }
    )

    if dev_result.get("error"):
        print("❌ Developer Agent failed. Aborting pipeline.")
        return

    if dev_result.get("developer_output") is None:
        print("❌ Developer Agent produced no output. Aborting pipeline.")
        return

    # ── Stage 4: QA Agent (tests + code review in parallel) ──────────
    print("\n🧪 Running QA Agent (tests + code review in parallel)...")
    qa_result = qa_graph.invoke({
        "pm_output": pm_result["pm_output"],
        "developer_output": dev_result["developer_output"],
    })

    if qa_result.get("error"):
        print("❌ QA Agent failed. Aborting pipeline.")
        return

    # ── Stage 5: Reviewer Agent (Human-in-the-loop) ──────────────────
    print("\n👁  Running Code Reviewer (with human-in-the-loop)...")

    # Fresh checkpoint thread for this run; reused across all resume invokes below.
    reviewer_config = make_reviewer_config()

    # Initialize state
    initial_state = {
        "developer_output": dev_result["developer_output"],
        "qa_output": qa_result["qa_output"]
    }

    rev_result = reviewer_graph.invoke(initial_state, reviewer_config)

    # Loop for human-in-the-loop interruption, with an explicit iteration cap.
    MAX_HITL_ITERATIONS = 10
    hitl_iterations = 0
    while rev_result and not rev_result.get("final_output"):
        if hitl_iterations >= MAX_HITL_ITERATIONS:
            print("⚠️  Reached maximum human-in-the-loop iterations. Finalizing automatically.")
            rev_result = reviewer_graph.invoke(
                Command(resume="a Auto-approved (max iterations reached)"),
                reviewer_config,
            )
            break
        hitl_iterations += 1

        print("┌─────────────────────────────────────────┐")
        print("│  Do you approve this review?            │")
        print("│  [a] Approve                            │")
        print("│  [s] Approve with suggestions           │")
        print("│  [r] Reject and request new round       │")
        print("│  [f] Approve and finalize pipeline       │")
        print("└─────────────────────────────────────────┘")
        try:
            user_input = input("Enter your option and optional feedback: ")
        except (EOFError, KeyboardInterrupt):
            # Non-interactive / EOF: treat as a single terminal approve, then stop.
            print("\n⏩ Non-interactive input: auto-approving and finalizing.")
            rev_result = reviewer_graph.invoke(
                Command(resume="a Automatically approved (EOF)"),
                reviewer_config,
            )
            break

        rev_result = reviewer_graph.invoke(Command(resume=user_input), reviewer_config)
    
    # ── Stage 6: DevOps Agent (Local Preview) ────────────────────────
    print("\n🚀 Running DevOps Agent — spinning up local preview...")

    dev_output_obj = dev_result.get("developer_output")
    if dev_output_obj is None:
        print("❌ No developer output available for DevOps stage. Aborting pipeline.")
        return
    proj_name = dev_output_obj.project_name.lower().replace(" ", "-").replace("_", "-")

    initial_devops_state = {
        "developer_output": dev_result["developer_output"],
        "project_name": proj_name
    }
    
    try:
        devops_res = devops_graph.invoke(initial_devops_state)
        local_preview_output = devops_res.get("local_preview_output")
    except Exception as exc:
        print(f"⚠️  DevOps Agent failed: {exc}")
        local_preview_output = None

    # ── Pipeline data extraction ─────────────────────────────────────
    pm_out = pm_result["pm_output"]
    arch_out = arch_result["architect_output"]
    dev_out = dev_result["developer_output"]
    qa_out = qa_result.get("qa_output")
    rev_out = rev_result.get("final_output")

    n_stories = len(pm_out.user_stories)
    n_endpoints = len(arch_out.api_endpoints)
    n_entities = len(arch_out.database_entities)
    n_dev_files = len(dev_out.backend.files) + len(dev_out.frontend.files)

    if qa_out:
        n_tests = len(qa_out.test_cases)
        qa_score = qa_out.quality_score
    else:
        n_tests = 0
        qa_score = "N/A"
        
    if rev_out:
        rev_rounds = len(rev_out.rounds)
        rev_status = rev_out.final_status
    else:
        rev_rounds = 0
        rev_status = "N/A"

    # ── Save project metadata in registry ────────────────────────────
    try:
        from deploy.project_registry import ProjectMeta, save_project
        from deploy.rm_runner import get_output_size_mb
        from datetime import datetime
        
        size_mb = get_output_size_mb(proj_name)
        preview_ready = local_preview_output.preview_ready if local_preview_output else False
        pr_url = (rev_result.get("github_output") or {}).get("pr_url") if rev_result else None
        
        meta = ProjectMeta(
            project_name=proj_name,
            created_at=datetime.now().isoformat(),
            requirement=requirement,
            tech_stack=f"{arch_out.tech_stack.frontend} + {arch_out.tech_stack.backend} + {arch_out.tech_stack.database}",
            files_count=n_dev_files,
            quality_score=qa_score if isinstance(qa_score, int) else None,
            deploy_status="not_deployed",
            local_preview_available=preview_ready,
            github_pr_url=pr_url,
            output_size_mb=size_mb
        )
        save_project(meta)
        print(f"💾 Project metadata for '{proj_name}' successfully saved in central registry.")
    except Exception as exc:
        print(f"⚠️  Could not save project metadata in central registry: {exc}")

    # MCP Usage determination
    arch_mcp = "Real MCP (Context7)" if getattr(arch_out, "context7_enriched", False) else "Simulated (Fallback)"
    
    design_mcp = "Simulated (Fallback)"
    if ui_output and ui_output.screens:
        if any(s.stitch_screen_url for s in ui_output.screens):
            design_mcp = "Real MCP (Google Stitch)"
            
    dev_mcp = "Real MCP (Filesystem)"
    
    qa_playwright_mcp = "Simulated (aiohttp Fallback)"
    e2e_out = qa_result.get("e2e_output")
    if e2e_out:
        if any(
            isinstance(res, dict) and res.get("status") == "success"
            for res in e2e_out.values()
        ):
            qa_playwright_mcp = "Real MCP (Playwright)"

    qa_fs_mcp = "Real MCP (Filesystem)" if qa_result.get("test_saved") else "Simulated (Local Fallback)"

    rev_github_mcp = "Simulated (Local/Fallback)"
    gh_out = (rev_result.get("github_output") or {}) if rev_result else {}
    if gh_out.get("status") == "success":
        rev_github_mcp = f"Real MCP (GitHub PR: {gh_out.get('pr_url')})"

    # Overall success: reviewer finalized and developer output present.
    pipeline_ok = bool(rev_out) and dev_out is not None
    if pipeline_ok:
        banner_title = "║                        FULL PIPELINE STATUS ✅                           ║"
    else:
        banner_title = "║                     PARTIAL PIPELINE STATUS ⚠️                            ║"

    print("\n╔══════════════════════════════════════════════════════════════════════════╗")
    print(banner_title)
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"║ 📋 PM Agent:       {str(n_stories).ljust(2)} user stories.                                        ║")
    print(f"║ 🏗️  Architect:      {str(n_endpoints).ljust(2)} endpoints, {str(n_entities).ljust(2)} database entities.                   ║")
    print(f"║    ↳ Context7:     {arch_mcp.ljust(53)} ║")
    print(f"║ 🎨 Designer:       {str(len(ui_output.screens) if ui_output else 0).ljust(2)} screens generated.                                    ║")
    print(f"║    ↳ Stitch:       {design_mcp.ljust(53)} ║")
    print(f"║ 💻 Developer:      {str(n_dev_files).ljust(2)} code files generated.                                ║")
    print(f"║    ↳ Filesystem:   {dev_mcp.ljust(53)} ║")
    print(f"║ 🧪 QA Agent:       {str(n_tests).ljust(2)} tests, score: {str(qa_score).ljust(3)}/100.                                 ║")
    print(f"║    ↳ Playwright:   {qa_playwright_mcp.ljust(53)} ║")
    print(f"║    ↳ Filesystem:   {qa_fs_mcp.ljust(53)} ║")
    print(f"║ 👁  Reviewer:      {str(rev_rounds).ljust(2)} rounds, final status: {rev_status.ljust(25)} ║")
    print(f"║    ↳ GitHub:       {rev_github_mcp[:53].ljust(53)} ║")
    
    devops_val = "Not running"
    if local_preview_output and local_preview_output.preview_ready:
        devops_val = f"{local_preview_output.backend_url} · {local_preview_output.frontend_url}"
    elif local_preview_output and local_preview_output.docker_compose_generated:
        devops_val = "Simulated / Docker Compose Generated"
    print(f"║ 🚀 DevOps:        {devops_val[:53].ljust(53)} ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝\n")

    # ── Memory Cleanup Prompts ─────────────────────────────────────────
    interactive = sys.stdin.isatty()

    # A. Clean Docker VM RAM
    if local_preview_output and local_preview_output.preview_ready:
        print("💡 [Memory Cleanup]")
        print("   The local preview containers are currently running, consuming ~4GB in the Docker VM.")
        if not interactive:
            print("   ⏩ Non-interactive session: leaving Docker containers running.")
        else:
            try:
                cleanup_docker = input("❓ Do you want to stop the local preview Docker containers to free up RAM? [y/N]: ").strip().lower()
                if cleanup_docker in ["y", "yes"]:
                    import shutil
                    import subprocess
                    # Pick an available docker CLI up front; skip if neither exists.
                    if shutil.which("docker"):
                        cmd = ["docker", "compose", "down"]
                    elif shutil.which("docker-compose"):
                        cmd = ["docker-compose", "down"]
                    else:
                        cmd = None
                        print("⚠️ [Cleanup] Neither 'docker' nor 'docker-compose' found; skipping container teardown.")
                    if cmd is not None:
                        print(f"⚙️ [Cleanup] Stopping Docker containers for project '{proj_name}'...")
                        result = subprocess.run(cmd, cwd=f"./output/{proj_name}", capture_output=True)
                        if result.returncode == 0:
                            print("✅ [Cleanup] Docker preview containers successfully stopped. RAM reclaimed!")
                        else:
                            err = result.stderr.decode(errors="replace").strip() if result.stderr else ""
                            print(f"⚠️ [Cleanup] Failed to stop Docker containers (exit {result.returncode}). {err}")
            except (KeyboardInterrupt, EOFError):
                pass

    # B. Clean Python RAM from MLX local server
    if interactive:
        try:
            cleanup_mlx = input("❓ Do you want to stop the local MLX server to reclaim 20GB of Python memory? [y/N]: ").strip().lower()
            if cleanup_mlx in ["y", "yes"]:
                from deploy.clean_runner import kill_mlx_server
                kill_mlx_server()
        except (KeyboardInterrupt, EOFError):
            pass
    else:
        print("⏩ Non-interactive session: leaving local MLX server running.")

    # C. Exit cleanly to free memory and dangling threads.
    print("👋 Exiting cleanly...")
    sys.exit(0)

if __name__ == "__main__":
    main()
