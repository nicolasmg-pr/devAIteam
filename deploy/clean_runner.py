import os
import shutil
import subprocess
import signal

def _args_target_mlx_server(args: str) -> bool:
    """True only when the process args reference the MLX/oMLX server as a whole token."""
    tokens = args.replace("/", " ").split()
    targets = {"mlx_lm.server", "oMLX.app"}
    return any(tok in targets for tok in tokens)


def kill_mlx_server():
    """Finds and terminates running mlx_lm.server / oMLX.app local server processes to free RAM."""
    try:
        res = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True)
        lines = res.stdout.splitlines()
        killed = False
        current_pid = os.getpid()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            pid_str, args = parts[0], parts[1]
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            if pid == current_pid:
                continue
            if _args_target_mlx_server(args):
                print(f"💀 [Cleanup] Terminating local MLX/oMLX server process (PID {pid})...")
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed = True
                except ProcessLookupError:
                    pass
        if killed:
            print("✅ [Cleanup] Local MLX/oMLX server successfully stopped. ~20GB RAM reclaimed!")
        else:
            print("ℹ️ [Cleanup] No active local MLX or oMLX server process was found.")
    except Exception as e:
        print(f"⚠️ [Cleanup] Error stopping MLX/oMLX server: {e}")

def run_clean_command():
    """Stops all running Docker Compose previews and terminates the local MLX server."""
    print("\n🧹 RAMPAGING MEMORY CLEANUP ACTIVATED...")
    print("=" * 60)
    
    # 1. Clean Docker Compose Previews
    output_dir = "./output"
    docker_cleaned = 0

    # Resolve the available container CLI once to avoid an uncaught FileNotFoundError.
    docker_bin = shutil.which("docker")
    compose_bin = shutil.which("docker-compose")
    if docker_bin:
        base_cmd = [docker_bin, "compose", "down"]
    elif compose_bin:
        base_cmd = [compose_bin, "down"]
    else:
        base_cmd = None

    if base_cmd is None:
        print("ℹ️ Neither 'docker' nor 'docker-compose' found on PATH; skipping container cleanup.")
    elif os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                compose_path = os.path.join(item_path, "docker-compose.yml")
                if os.path.exists(compose_path):
                    print(f"⚙️ Stopping Docker containers for project '{item}'...")
                    try:
                        res = subprocess.run(base_cmd, cwd=item_path, capture_output=True)
                    except Exception as e:
                        print(f"   ⚠️ Could not stop project '{item}' containers: {e}")
                        continue
                    if res.returncode == 0:
                        docker_cleaned += 1
                        print(f"   ✅ Project '{item}' containers stopped.")
                    else:
                        print(f"   ⚠️ Could not stop project '{item}' containers.")

    if docker_cleaned > 0:
        print(f"✅ Stopped preview containers for {docker_cleaned} projects. ~4GB RAM reclaimed in Docker VM!")
    else:
        print("ℹ️ No active Docker preview containers found to stop.")
        
    print("-" * 60)
    
    # 2. Clean MLX Server
    print("🔍 Searching for active local MLX server processes...")
    kill_mlx_server()
    
    print("=" * 60)
    print("🎉 Memory cleanup complete. Your machine is fresh and fast!\n")
