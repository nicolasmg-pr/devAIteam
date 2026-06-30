import asyncio
import atexit
import os
import threading
import weakref
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from config.paths import PROJECT_ROOT

# Bound how long we wait for an MCP server to connect or a tool to respond, so a
# hung server (or an `npx` blocked on a download) can't stall a pipeline node.
MCP_CONNECT_TIMEOUT = float(os.getenv("MCP_CONNECT_TIMEOUT", "60"))
MCP_CALL_TIMEOUT = float(os.getenv("MCP_CALL_TIMEOUT", "120"))
MCP_MAX_OUTPUT_CHARS = int(os.getenv("MCP_MAX_OUTPUT_CHARS", "100000"))

# Track live clients so we can close them at interpreter exit even when a caller
# does not own the lifecycle explicitly (legacy `get_mcp_tools` callers).
_LIVE_CLIENTS: "weakref.WeakSet[ThreadSafeMCPClient]" = weakref.WeakSet()


@atexit.register
def _close_live_clients() -> None:  # pragma: no cover - process teardown
    for client in list(_LIVE_CLIENTS):
        try:
            client.close()
        except Exception:
            pass


class ThreadSafeMCPClient:
    """A thread-safe synchronous wrapper around asynchronous MCP stdio clients.

    Manages the lifecycle of stdio MCP servers in a background event loop. Use as a
    context manager, or call :meth:`close` when done, to release the spawned
    subprocess, background thread, event loop, and file handles.
    """

    def __init__(self, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        self.command = command
        self.args = args
        self.env = env
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        self.raw_tools: list = []
        self.session = None
        self.client_context = None
        self.devnull = None
        self._closed = False

        # Connect synchronously and fetch tools
        try:
            self.raw_tools = self._run_async(self._connect_and_list(), timeout=MCP_CONNECT_TIMEOUT)
            print(f"🔌 [MCP] Connected to {command} {' '.join(args)}. {len(self.raw_tools)} tools loaded.")
            _LIVE_CLIENTS.add(self)
        except Exception as e:
            print(f"⚠️ [MCP] Error connecting to {command} {' '.join(args)}: {e}")
            # Tear down the loop/thread so we don't leak on a failed connect.
            self.close()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _run_async(self, coro, timeout: Optional[float] = None):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            future.cancel()
            raise TimeoutError(f"MCP operation timed out after {timeout}s")

    async def _connect_and_list(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )
        # Suppress redundant stderr warnings (e.g. Client does not support MCP Roots)
        self.devnull = open(os.devnull, "w", encoding="utf-8")
        self.client_context = stdio_client(server_params, errlog=self.devnull)
        self.read, self.write = await self.client_context.__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()

        tools_list = await self.session.list_tools()
        return tools_list.tools

    async def _aclose(self):
        if self.session is not None:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception:
                pass
            self.session = None
        if self.client_context is not None:
            try:
                await self.client_context.__aexit__(None, None, None)
            except Exception:
                pass
            self.client_context = None

    def close(self):
        """Release the MCP session, subprocess, event loop, thread, and file handle."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.loop.is_running():
                try:
                    self._run_async(self._aclose(), timeout=MCP_CONNECT_TIMEOUT)
                except Exception:
                    pass
                self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        try:
            if not self.loop.is_closed():
                self.loop.close()
        except Exception:
            pass
        if self.devnull is not None:
            try:
                self.devnull.close()
            except Exception:
                pass
            self.devnull = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def build_filtered_directory_tree(self, root_dir: str) -> str:
        ignore_dirs = {".git", ".venv", "node_modules", "__pycache__", ".DS_Store", ".mypy_cache", ".pytest_cache", "dist", "build"}
        tree_lines = [f"Directory tree of: {root_dir}"]

        def walk(directory: str, prefix: str = ""):
            try:
                items = sorted(os.listdir(directory))
            except Exception:
                return

            filtered_items = []
            for item in items:
                if item in ignore_dirs:
                    continue
                filtered_items.append(item)

            for i, item in enumerate(filtered_items):
                is_last = (i == len(filtered_items) - 1)
                item_path = os.path.join(directory, item)
                is_dir = os.path.isdir(item_path)

                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{item}{'/' if is_dir else ''}")

                if is_dir:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    walk(item_path, next_prefix)

        walk(root_dir)
        return "\n".join(tree_lines)

    def call_tool(self, name: str, arguments: dict) -> str:
        if name == "directory_tree":
            path = arguments.get("path", ".")
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            try:
                return self.build_filtered_directory_tree(path)
            except Exception as e:
                return f"Error building filtered directory tree: {e}"

        if self.session is None:
            return f"Error executing tool {name}: MCP session is not connected."

        try:
            return self._run_async(self._call_tool(name, arguments), timeout=MCP_CALL_TIMEOUT)
        except Exception as e:
            return f"Error executing tool {name}: {e}"

    async def _call_tool(self, name: str, arguments: dict) -> str:
        res = await self.session.call_tool(name, arguments)
        output_text = ""
        for block in res.content:
            if getattr(block, "type", None) == "text":
                output_text += getattr(block, "text", "") + "\n"

        output_stripped = output_text.strip()
        if len(output_stripped) > MCP_MAX_OUTPUT_CHARS:
            # Truncate on a line boundary so we don't cut mid-token / mid-JSON.
            head = output_stripped[:MCP_MAX_OUTPUT_CHARS]
            nl = head.rfind("\n")
            if nl > 0:
                head = head[:nl]
            print(f"⚠️  [MCP] Truncating response from {name} ({len(output_stripped)} characters)")
            output_stripped = head + "\n\n... [TRUNCATED - Output too long for LLM context window] ..."
        return output_stripped

    def get_tools(self) -> List[StructuredTool]:
        lc_tools = []
        for tool_def in self.raw_tools:
            name = tool_def.name
            description = tool_def.description or ""
            input_schema = tool_def.inputSchema or {}

            # Build Pydantic args model dynamically
            fields: dict[str, Any] = {}
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            for prop_name, prop_details in properties.items():
                prop_type_str = prop_details.get("type", "string")
                prop_desc = prop_details.get("description", "")

                python_type: Any = str
                if prop_type_str == "integer":
                    python_type = int
                elif prop_type_str == "boolean":
                    python_type = bool
                elif prop_type_str == "number":
                    python_type = float
                elif prop_type_str == "array":
                    python_type = list
                elif prop_type_str == "object":
                    python_type = dict

                is_req = prop_name in required
                default = ... if is_req else None
                fields[prop_name] = (python_type, Field(default=default, description=prop_desc))

            args_schema = create_model(f"{name}_args", **fields)

            def _create_func(tool_name: str):
                def _tool_func(**kwargs) -> str:
                    filtered_args = {k: v for k, v in kwargs.items() if v is not None}
                    return self.call_tool(tool_name, filtered_args)
                return _tool_func

            lc_tool = StructuredTool.from_function(
                func=_create_func(name),
                name=name,
                description=description,
                args_schema=args_schema,
            )
            lc_tools.append(lc_tool)
        return lc_tools


CONTEXT7_MCP_CONFIG = {
    "command": "npx",
    "args": ["-y", "@upstash/context7-mcp@latest"],
}

FILESYSTEM_MCP_CONFIG = {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", str(PROJECT_ROOT)],
}

PLAYWRIGHT_MCP_CONFIG = {
    "command": "npx",
    "args": ["-y", "@playwright/mcp@latest"],
}

GITHUB_MCP_CONFIG = {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
}


def get_mcp_tools(mcp_config: dict, transport: str = "stdio", return_client: bool = False):
    """Connect to an MCP server and return its tools, with graceful degradation.

    Returns a list of tools by default. Pass ``return_client=True`` to also receive
    the underlying :class:`ThreadSafeMCPClient` so the caller can ``close()`` it when
    finished (the tools keep the session alive while in use). On failure returns
    ``[]`` (or ``([], None)``).
    """
    try:
        env = None
        if "server-github" in "".join(mcp_config.get("args", [])):
            token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") or os.environ.get("GITHUB_TOKEN")
            if not token:
                print("⚠️ [MCP] GITHUB_PERSONAL_ACCESS_TOKEN or GITHUB_TOKEN not found in env.")
                return ([], None) if return_client else []
            # Preserve PATH/HOME etc. so the spawned `npx` can locate node.
            env = {**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": token}

        client = ThreadSafeMCPClient(
            command=mcp_config["command"],
            args=mcp_config["args"],
            env=env,
        )
        tools = client.get_tools()
        if not tools:
            print(f"⚠️ [MCP] No tools retrieved from {mcp_config.get('command')}")
            client.close()
            return ([], None) if return_client else []
        return (tools, client) if return_client else tools
    except Exception as e:
        print(f"⚠️ [MCP] Connection with MCP server {mcp_config.get('command')} failed: {e}")
        return ([], None) if return_client else []
