# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo actually is

This is **`devAIteam`** — a multi-agent, LangGraph-based pipeline that turns a one-line natural-language requirement into a full application (NestJS/Flutter by default). The pipeline runs 100% locally against an MLX/Ollama server, or against cloud OpenAI if `OPENAI_API_KEY` is set.

**Critical mental model:** the Python code under `agents/`, `graphs/`, `config/`, `tools/`, `deploy/`, plus `main.py` is the *system*. The web/app files committed at the repo root (`lib/`, `src/`, `prisma/`, `package.json`, `tsconfig.json`, `Dockerfile`, `docker-compose.yml`, `README_PROJECT.md`, `fintrack_app/`, `gestor_gastos/`) are **generated output of a past pipeline run**, not source you should treat as part of the platform. `package.json` is named `marineprecheck-backend` and `README_PROJECT.md` documents a "Pre-Sailing Checklist App" — both are artifacts. Newly generated projects normally land in `./output/{project-name}/`; the root pollution exists because the Filesystem MCP root and the reviewer's apply-patch base dir are both hardcoded to the repo root (see Gotchas).

## Running the system

The `devAIteam` CLI is the entry point (`./devAIteam` shell wrapper → `.venv/bin/python main.py`, also installed as a `project.script`).

```bash
uv sync                                 # install Python deps (Python 3.12, see .python-version)
source .venv/bin/activate

devAIteam "describe the app you want"   # run the full pipeline
devAIteam                               # no args → interactive dashboard prompt
devAIteam list                          # show generated-project registry table
devAIteam deploy <project>              # deploy to Fly.io/Railway/Render
devAIteam rm <project> [--all]          # delete local output (--all also closes PRs/branches via GitHub MCP)
devAIteam prune                         # clean stale entries in the registry
devAIteam doctor                        # check which MCP tools are reachable
devAIteam clean                         # kill the local MLX server + reclaim RAM
```

There is **no Python test suite or linter configured**. `test_mlx.py` is a standalone connectivity probe for the MLX server, not a pytest test. The `jest`/`tsc`/`prisma` scripts in the root `package.json` belong to generated output, not the platform.

## Prerequisites

- A local LLM server at `http://localhost:8000/v1` serving `Qwen3.6-35B-A3B-UD-MLX-4bit` (MLX or Ollama), **or** `OPENAI_API_KEY` exported (then it uses `gpt-4o`, and `o3-mini` for the reasoning-enabled Reviewer).
- `npx` available — all MCP servers (Context7, Filesystem, Playwright, GitHub) are launched on demand via `npx -y ...`.
- `.env` (gitignored) for optional integrations: `GITHUB_PERSONAL_ACCESS_TOKEN`, `GITHUB_OWNER`/`GITHUB_REPO`, `STITCH_API_KEY`, `FLY_API_TOKEN`/`RAILWAY_TOKEN`/`RENDER_API_KEY`.

## Architecture

The pipeline is a sequence of independent LangGraph graphs, wired together imperatively in `main.py`:

```
PM → Architect → UI Designer → Developer → QA → Reviewer (HITL) → DevOps → registry save
```

Each stage follows the **same two-file convention**:

- `agents/<role>_agent.py` — the agent logic: Pydantic output models, a strict "respond ONLY with JSON" system prompt, an LLM call, then `clean_llm_response()` + `json.loads` + `Model.model_validate`. The agent's typed Pydantic output is the contract passed to the next stage.
- `graphs/<role>_graph.py` — a `StateGraph` whose state is a Pydantic model; nodes call the agent, format/print results, and (where relevant) persist files. Each module ends with a pre-built `<role>_graph = build_..._graph()` ready to import.

Key cross-cutting pieces:

- **`config/llm_config.py`** — the *only* place LLM engine selection lives. `get_llm(temperature, thinking, max_tokens)` picks cloud-vs-local. Per-agent instances (`llm_pm`, `llm_developer`, etc.) are exposed via module `__getattr__` so **a fresh `ChatOpenAI` is built on every access** — this is deliberate, to avoid Pydantic concurrency errors when LangGraph runs nodes in parallel threads. Temperatures encode intent: 1.0 for PM/Architect (creative), 0.7 for Designer/Developer, 0.3 for QA/Reviewer (precise). Only the Reviewer uses `thinking=True`.
- **`tools/llm_helpers.py`** — `clean_llm_response()` strips Qwen `<think>...</think>` blocks (including unclosed ones) and extracts the outermost JSON object/array. Every agent depends on this for robust parsing of local-model output.
- **`agents/mcp_client.py`** — `ThreadSafeMCPClient` wraps async stdio MCP servers behind a sync API by running an asyncio loop in a background thread. `get_mcp_tools(config)` is the entry point and **degrades gracefully**: any connection failure returns `[]` so the pipeline continues with a non-MCP fallback. MCP server configs (`CONTEXT7_MCP_CONFIG`, `FILESYSTEM_MCP_CONFIG`, `PLAYWRIGHT_MCP_CONFIG`, `GITHUB_MCP_CONFIG`) live at the bottom of this file. The "Real MCP vs Simulated/Fallback" status table printed at the end of `main.py` reflects which servers actually connected.

### Parallelism and human-in-the-loop

- **Developer and QA graphs use LangGraph fan-out**: multiple `add_edge(START, ...)` run backend/frontend (or tests/code-review) nodes concurrently, then fan-in at a `merge_node`. Sub-agent errors are caught per-node and surfaced via an `error` field on the state rather than raising.
- **The Reviewer graph is the only interactive one.** It compiles with a `MemorySaver` checkpointer and uses `interrupt()` to pause for human approval. `main.py` drives the loop: invoke → if no `final_output`, prompt the user (`a`/`s`/`r`/`f`), then `reviewer_graph.invoke(Command(resume=user_input), reviewer_config)`. `reviewer_config` pins `thread_id="review-session-1"`. On approval, `finalize_node` applies the AI's proposed code patches to disk and `github_node` opens a PR via the GitHub MCP.

### Deploy / lifecycle layer

`deploy/` holds the non-pipeline CLI commands. `deploy/project_registry.py` reads/writes `./output/.registry.json` (the `ProjectMeta` index — file counts, QA score, PR URL, size, deploy status). `*_runner.py` files implement each `devAIteam` subcommand; `deploy/deploy_agent.py` handles cloud deploys.

## Gotchas

- **Hardcoded absolute paths.** `FILESYSTEM_MCP_CONFIG` in `agents/mcp_client.py` and `base_dir` in `graphs/reviewer_graph.py` (`_apply_code_change`) are hardcoded to `/Users/nikomendez/Documents/SWdevAIgency_project`. Anyone cloning elsewhere must update these.
- `main.py` ends with `os._exit(0)` to force-kill the process (and offers to stop Docker preview containers + the MLX server) — it does not return normally.
- Pipeline stages communicate via **typed Pydantic objects**, not dicts. When changing an agent's output schema, update both the agent's model and every downstream graph state / consumer that reads those fields (e.g. `main.py`'s summary extraction).
- Generated app code is written to the repo root via the reviewer's local patch-apply, which is why `git status` shows churn in `lib/` and `src/`. Don't confuse those diffs with platform changes.
