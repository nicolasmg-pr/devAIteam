import os
import re
import shutil
import asyncio
import subprocess
from typing import Optional
from deploy.project_registry import (
    get_project,
    load_registry,
    save_project,
    remove_project,
    write_registry,
)
from config.paths import project_dir, validate_project_name, OUTPUT_DIR, UnsafePathError


def _safe_output_path(project_name: str) -> str:
    """Return the validated, sandbox-contained output path for a project.

    Validates the name and asserts the resolved path stays inside OUTPUT_DIR.
    Raises UnsafePathError on any violation.
    """
    path = str(project_dir(project_name))
    real_path = os.path.realpath(path)
    real_root = os.path.realpath(str(OUTPUT_DIR))
    if real_path != real_root and not real_path.startswith(real_root + os.sep):
        raise UnsafePathError(
            f"Resolved project path escapes output sandbox: {real_path}"
        )
    return path


def get_output_size_mb(project_name: str) -> float:
    """Calculate the recursive size of the project's output dir in MB."""
    total_size = 0
    try:
        output_path = _safe_output_path(project_name)
    except UnsafePathError:
        return 0.0
    if not os.path.exists(output_path):
        return 0.0
    for dirpath, dirnames, filenames in os.walk(output_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
            except Exception:
                pass
    return total_size / (1024 * 1024)

async def delete_github_resources(project_name: str, github_pr_url: Optional[str]) -> dict:
    """Connect to GitHub MCP, close active PRs, delete branches, and optionally delete the repo."""
    from agents.mcp_client import get_mcp_tools, GITHUB_MCP_CONFIG

    res = {
        "pr_closed": False,
        "branch_deleted": False,
        "repo_deleted": False,
        "error": None
    }

    if not github_pr_url:
        return res

    # Do not guess identity for destructive operations.
    owner = os.getenv("GITHUB_OWNER")
    repo = os.getenv("GITHUB_OUTPUT_REPO") or os.getenv("GITHUB_PROJECT_REPO")
    if not owner or not repo:
        msg = (
            "GITHUB_OWNER and GITHUB_OUTPUT_REPO/GITHUB_PROJECT_REPO must be set to "
            "delete GitHub resources; refusing to guess the target repository."
        )
        print(f"   ⚠️  [GitHub] {msg}")
        res["error"] = msg
        return res

    client = None
    try:
        tools, client = get_mcp_tools(GITHUB_MCP_CONFIG, "stdio", return_client=True)
        tool_map = {t.name: t for t in tools} if tools else {}

        # 1. Close PR if open and discover its actual branch name.
        pr_branch = None
        pr_match = re.search(r'pull/(\d+)', github_pr_url)
        pr_number = int(pr_match.group(1)) if pr_match else None

        if pr_number is not None:
            # Try to read the real head branch from the PR before closing it.
            if "get_pull_request" in tool_map:
                try:
                    pr_info = tool_map["get_pull_request"].invoke({
                        "owner": owner,
                        "repo": repo,
                        "pull_number": pr_number
                    })
                    pr_branch = _extract_branch_from_pr(pr_info)
                except Exception as e:
                    print(f"   ⚠️  [GitHub] Could not read PR details: {e}")

            if "update_pull_request" in tool_map:
                print(f"   🐙 [GitHub MCP] Closing Pull Request #{pr_number}...")
                try:
                    tool_map["update_pull_request"].invoke({
                        "owner": owner,
                        "repo": repo,
                        "pull_number": pr_number,
                        "state": "closed"
                    })
                    res["pr_closed"] = True
                except Exception as e:
                    print(f"   ⚠️  [GitHub] Could not close PR: {e}")

        # 2. Delete branch — prefer the branch derived from the PR, fall back to guess.
        branch_candidates = []
        if pr_branch:
            branch_candidates.append(pr_branch)
        branch_candidates.append(f"feature/ai-generated-{project_name}")
        # De-duplicate while preserving order.
        seen = set()
        branch_candidates = [b for b in branch_candidates if not (b in seen or seen.add(b))]

        if "delete_git_ref" in tool_map:
            for branch in branch_candidates:
                try:
                    print(f"   🐙 [GitHub MCP] Deleting branch 'heads/{branch}'...")
                    tool_map["delete_git_ref"].invoke({
                        "owner": owner,
                        "repo": repo,
                        "ref": f"heads/{branch}"
                    })
                    res["branch_deleted"] = True
                except Exception:
                    pass

        # 3. Full repository deletion — separate, explicit, typed-name confirmation,
        # and only after PR/branch cleanup above.
        if "delete_repository" in tool_map:
            print(
                f"\n   🚨 OPTIONAL: full deletion of the entire repository '{owner}/{repo}'."
                "\n      This is distinct from the PR/branch cleanup above and is irreversible."
            )
            try:
                typed = input(
                    f"      To delete the whole repo, retype exactly '{owner}/{repo}' (or press Enter to skip): "
                ).strip()
            except (KeyboardInterrupt, EOFError):
                typed = ""
            if typed == f"{owner}/{repo}":
                print(f"   🐙 [GitHub MCP] Deleting repository '{owner}/{repo}'...")
                try:
                    tool_map["delete_repository"].invoke({
                        "owner": owner,
                        "repo": repo
                    })
                    res["repo_deleted"] = True
                except Exception as e:
                    print(f"   ⚠️  [GitHub] Could not delete repository: {e}")
                    res["error"] = str(e)
            else:
                print("   ℹ️  Skipping full repository deletion.")

    except Exception as e:
        res["error"] = str(e)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    return res


def _extract_branch_from_pr(pr_info) -> Optional[str]:
    """Best-effort extraction of the PR head branch name from an MCP tool result."""
    import json

    data = pr_info
    if isinstance(pr_info, str):
        try:
            data = json.loads(pr_info)
        except Exception:
            # Fall back to a regex over the raw text payload.
            m = re.search(r'"ref"\s*:\s*"([^"]+)"', pr_info)
            return m.group(1) if m else None
    if isinstance(data, dict):
        head = data.get("head")
        if isinstance(head, dict):
            ref = head.get("ref")
            if isinstance(ref, str) and ref:
                return ref
    return None

def _docker_compose_teardown(docker_compose_path: str, with_volumes: bool = False):
    """Stop a project's Docker Compose stack, checking for the binary and result."""
    docker_bin = shutil.which("docker")
    compose_bin = shutil.which("docker-compose")
    if docker_bin:
        cmd = [docker_bin, "compose", "-f", docker_compose_path, "down"]
    elif compose_bin:
        cmd = [compose_bin, "-f", docker_compose_path, "down"]
    else:
        print("   ⚠️  Docker not found on PATH; skipping container teardown.")
        return
    if with_volumes:
        cmd.append("-v")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"   ⚠️  Docker teardown failed (exit {res.returncode}); containers may still be running.")
    except Exception as e:
        print(f"   ⚠️  Error running Docker teardown: {e}")


def _delete_dir_reporting(output_path: str, size_mb: float):
    """Delete a directory and report the ACTUAL result (no silent failure)."""
    errors = []

    def _onerror(func, path, exc_info):
        errors.append((path, exc_info[1]))

    try:
        shutil.rmtree(output_path, ignore_errors=False, onerror=_onerror)
    except Exception as e:
        errors.append((output_path, e))

    if not os.path.exists(output_path):
        print(f"   ✅ Local code deleted ({size_mb:.1f} MB freed)")
        return True
    else:
        if errors:
            print(f"   ⚠️  Error deleting local folder: {errors[-1][1]}")
        else:
            print("   ⚠️  Local folder still exists after deletion attempt.")
        return False


def run_rm_command(project_name: str, delete_all: bool = False):
    """Execute the project removal CLI command in soft or total mode."""
    try:
        project_name = validate_project_name(project_name)
        output_path = _safe_output_path(project_name)
    except UnsafePathError as e:
        print(f"\n❌ Error: Unsafe project name. {e}")
        return

    meta = get_project(project_name)

    if not meta and not os.path.exists(output_path):
        print(f"\n❌ Error: The project '{project_name}' does not exist locally or in the registry.")
        projects = load_registry()
        if projects:
            print("Registered projects:")
            for p in projects:
                print(f"  - {p.project_name}")
        else:
            print("No generated projects currently found.")
        return

    size_mb = get_output_size_mb(project_name)
    
    # ── SOFT MODE ────────────────────────────────────────────────────────────
    if not delete_all:
        github_pr_str = meta.github_pr_url if (meta and meta.github_pr_url) else "(no PR created)"
        deploy_str = meta.deploy_url if (meta and meta.deploy_url) else "not deployed"
        
        print(f"""
⚠️  You are about to delete local code of '{project_name}'
   📂 To delete: ./output/{project_name}/ ({size_mb:.1f} MB)
   🐙 To keep: code on GitHub {github_pr_str}
   🚀 To keep: deployment at {deploy_str}

Confirm? [y/N]: """, end="")
        
        try:
            confirm = input().strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n❌ Operation cancelled.")
            return
            
        if confirm not in ["y", "yes", "s", "si"]:
            print("❌ Operation cancelled.")
            return
            
        # Stop Docker compose if active
        docker_compose_path = os.path.join(output_path, "docker-compose.yml")
        if os.path.exists(docker_compose_path):
            print("   🐳 Stopping Docker services...")
            _docker_compose_teardown(docker_compose_path)

        # Delete local files
        _delete_dir_reporting(output_path, size_mb)

        # Update registry entry
        if meta:
            meta.local_preview_available = False
            save_project(meta)
            
        print(f"""
✅ '{project_name}' deleted locally.
   {"🐙 Code remains available at: " + meta.github_pr_url if meta and meta.github_pr_url else ""}
   {"🚀 Deployment remains active at: " + meta.deploy_url if meta and meta.deploy_url else ""}
   To regenerate: devAIteam "{meta.requirement[:60] + '...' if meta else 'your requirement'}"
""")

    # ── TOTAL MODE ───────────────────────────────────────────────────────────
    else:
        github_info = ""
        if meta and meta.github_pr_url:
            github_info = f"\n   🐙 Will close PR and delete branch on GitHub: {meta.github_pr_url}"
            
        deploy_warning = ""
        if meta and meta.deploy_url:
            deploy_warning = f"\n   ⚠️  ATTENTION: The deployment at {meta.deploy_url} will NOT be deleted automatically.\n      You will need to delete it manually from {meta.deploy_platform}."
            
        print(f"""
🚨 TOTAL DELETION of '{project_name}'
   📂 To delete: ./output/{project_name}/ ({size_mb:.1f} MB){github_info}{deploy_warning}

This action CANNOT be undone. Confirm? [y/N]: """, end="")
        
        try:
            confirm = input().strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n❌ Operation cancelled.")
            return
            
        if confirm not in ["y", "yes", "s", "si"]:
            print("❌ Operation cancelled.")
            return
            
        # 1. Stop Docker compose with volumes
        docker_compose_path = os.path.join(output_path, "docker-compose.yml")
        if os.path.exists(docker_compose_path):
            print("   🐳 Stopping Docker services (with volumes)...")
            _docker_compose_teardown(docker_compose_path, with_volumes=True)

        # 2. Delete local files
        _delete_dir_reporting(output_path, size_mb)

        # 3. Delete GitHub resources if present
        if meta and meta.github_pr_url:
            print("   🐙 Deleting GitHub resources...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                github_result = loop.run_until_complete(delete_github_resources(project_name, meta.github_pr_url))
                if github_result.get("pr_closed"):
                    print("   ✅ PR closed on GitHub")
                if github_result.get("branch_deleted"):
                    print("   ✅ Branch deleted on GitHub")
                if github_result.get("repo_deleted"):
                    print("   ✅ Repository deleted on GitHub")
                if github_result.get("error"):
                    print(f"   ⚠️  GitHub: {github_result['error']}")
            except Exception as e:
                print(f"   ⚠️  Error invoking GitHub MCP: {e}")
            finally:
                loop.close()
                 
        # 4. Remove from registry (atomic)
        try:
            if remove_project(project_name):
                print("   ✅ Project registry entry deleted from .registry.json")
            else:
                print("   ℹ️  No registry entry found to delete.")
        except Exception as e:
            print(f"   ⚠️  Error updating central registry: {e}")
            
        if meta and meta.deploy_url:
            print(f"""
✅ '{project_name}' completely deleted.
   ⚠️  Remember to manually delete the deployment at {meta.deploy_platform}:
       {meta.deploy_url}
""")
        else:
            print(f"\n✅ '{project_name}' completely deleted.\n")


def run_prune_command():
    """Detect and remove registry entries for projects whose local code directories no longer exist."""
    registry = load_registry()
    if not registry:
        print("\nRegistry is empty. Nothing to prune.\n")
        return

    to_prune = []
    for p in registry:
        try:
            output_path = str(project_dir(p.project_name))
        except UnsafePathError:
            # An entry whose name no longer validates can't have a safe local dir.
            to_prune.append(p)
            continue
        if not os.path.exists(output_path):
            to_prune.append(p)

    if not to_prune:
        print("\nAll registered projects exist locally. Nothing to prune.\n")
        return

    print("\n🔍 Detected projects in registry with missing local directories:")
    for idx, p in enumerate(to_prune, 1):
        print(f"  {idx}. {p.project_name} (Created: {p.created_at})")

    print(f"\nConfirm pruning these {len(to_prune)} entries from the registry? [y/N]: ", end="")
    try:
        confirm = input().strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n❌ Operation cancelled.")
        return

    if confirm not in ["y", "yes", "s", "si"]:
        print("❌ Operation cancelled.")
        return

    pruned_names = {p.project_name for p in to_prune}
    new_registry = [p for p in registry if p.project_name not in pruned_names]

    try:
        write_registry(new_registry)
        print(f"\n✅ Successfully pruned {len(to_prune)} missing entries from the registry!\n")
    except Exception as e:
        print(f"\n❌ Error updating project registry during prune: {e}\n")

