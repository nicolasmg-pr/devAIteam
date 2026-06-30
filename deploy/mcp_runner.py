import os
import shutil
import time
from typing import List
from agents.mcp_client import ThreadSafeMCPClient, CONTEXT7_MCP_CONFIG, FILESYSTEM_MCP_CONFIG, PLAYWRIGHT_MCP_CONFIG, GITHUB_MCP_CONFIG

def format_col(val: str, width: int, align="left") -> str:
    val = str(val)
    if len(val) > width:
        val = val[:width-3] + "..."
    if align == "right":
        return val.rjust(width)
    elif align == "center":
        return val.center(width)
    else:
        return val.ljust(width)

def check_mcp_connection(name: str, config: dict, env_required: str = None) -> tuple[bool, str, int]:
    """Test connecting to an MCP server and listing its tools."""
    start_time = time.time()

    # Resolve the actually-present token (accept GITHUB_TOKEN as a fallback) and
    # inject it under the canonical key the server expects, never as None.
    env = None
    if env_required:
        token = os.environ.get(env_required) or os.environ.get("GITHUB_TOKEN")
        if not token:
            return False, f"Missing {env_required} env variable", 0
        # Preserve PATH/HOME etc. so the spawned process can locate its runtime.
        env = {**os.environ, env_required: token}

    client = None
    try:
        # Check command availability
        if not shutil.which(config["command"]):
            return False, f"Command '{config['command']}' not found on system PATH", 0

        # Instantiate client which syncs and lists tools
        client = ThreadSafeMCPClient(
            command=config["command"],
            args=config["args"],
            env=env
        )

        tools = client.get_tools()
        elapsed = int((time.time() - start_time) * 1000)

        if not tools:
            return False, "Connected but returned 0 tools", elapsed

        return True, f"Success ({len(tools)} tools loaded)", elapsed
    except Exception as e:
        elapsed = int((time.time() - start_time) * 1000)
        # Keep the FULL message so the keyword-matching troubleshooting logic can
        # still see substrings like 'GITHUB_PERSONAL_ACCESS_TOKEN' or 'PATH'.
        return False, str(e), elapsed
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

def run_mcp_check():
    """Diagnostic tool to inspect all configured Model Context Protocol (MCP) connections."""
    print("\n" + "=" * 80)
    print("🩺  devAIteam — MODEL CONTEXT PROTOCOL (MCP) DIAGNOSTICS")
    print("=" * 80)
    print("Probing connection channels to MCP servers on stdio...\n")
    
    servers = [
        ("Context7 (Upstash Docs)", CONTEXT7_MCP_CONFIG, None),
        ("Filesystem (Workspace)", FILESYSTEM_MCP_CONFIG, None),
        ("Playwright (E2E Tests)", PLAYWRIGHT_MCP_CONFIG, None),
        ("GitHub (Pull Requests)", GITHUB_MCP_CONFIG, "GITHUB_PERSONAL_ACCESS_TOKEN"),
    ]
    
    results = []
    for name, config, env_req in servers:
        print(f"📡 Probing '{name}'...")
        status, details, elapsed = check_mcp_connection(name, config, env_req)
        results.append((name, status, details, elapsed))
        
    print("\n╔" + "═" * 78 + "╗")
    print("║" + "  MCP CONNECTION CHANNELS DIAGNOSTICS SUMMARY".ljust(78) + "║")
    print("╠" + "═" * 28 + "╦" + "═" * 12 + "╦" + "═" * 8 + "╦" + "═" * 26 + "╣")
    print("║ " + format_col("Server", 26) + " ║ " + format_col("Status", 10) + " ║ " + format_col("Latency", 6) + " ║ " + format_col("Details", 24) + " ║")
    print("╠" + "═" * 28 + "╬" + "═" * 12 + "╬" + "═" * 8 + "╬" + "═" * 26 + "╣")
    
    for name, status, details, elapsed in results:
        status_str = "✅ ACTIVE" if status else "❌ OFFLINE"
        lat_str = f"{elapsed}ms" if elapsed > 0 else "N/A"
        print("║ " + format_col(name, 26) + " ║ " + format_col(status_str, 10) + " ║ " + format_col(lat_str, 6, "right") + " ║ " + format_col(details, 24) + " ║")
        
    print("╚" + "═" * 28 + "╩" + "═" * 12 + "╩" + "═" * 8 + "╩" + "═" * 26 + "╝\n")
    
    # Render diagnostics help
    has_offline = any(not r[1] for r in results)
    if has_offline:
        print("💡 TROUBLESHOOTING OFFLINE SERVICES:")
        for name, status, details, _ in results:
            if not status:
                print(f"❌ {name}:")
                if "npx" in details or "PATH" in details:
                    print("   -> Verify that Node.js / NPX is installed and added to your system environment variables.")
                elif "GITHUB_PERSONAL_ACCESS_TOKEN" in details:
                    print("   -> Add a valid GITHUB_PERSONAL_ACCESS_TOKEN or GITHUB_TOKEN inside your root '.env' file.")
                elif "Playwright" in name:
                    print("   -> Make sure you have playwright packages installed: npm install -g playwright")
                else:
                    print(f"   -> {details}")
        print()
    else:
        print("🎉 All MCP connections are running perfectly! Your local team is fully equipped.\n")
