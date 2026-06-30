import os
import asyncio
from deploy.deploy_agent import run_deploy
from deploy.project_registry import load_registry, update_deploy_status
from config.paths import project_dir, validate_project_name, OUTPUT_DIR, UnsafePathError

def run_deploy_command(project_name: str):
    """Deploy the project, or print instructions, and update the registry."""
    # Validate the name and assert the resolved path stays inside the output sandbox.
    try:
        project_name = validate_project_name(project_name)
        output_path = str(project_dir(project_name))
        real_path = os.path.realpath(output_path)
        real_root = os.path.realpath(str(OUTPUT_DIR))
        if real_path != real_root and not real_path.startswith(real_root + os.sep):
            raise UnsafePathError(f"Resolved project path escapes output sandbox: {real_path}")
    except UnsafePathError as e:
        print(f"\n❌ Error: Unsafe project name. {e}")
        return

    if not os.path.exists(output_path):
        print(f"\n❌ Error: The project directory '{project_name}' does not exist in './output/'.")
        projects = load_registry()
        if projects:
            print("Available projects in the registry:")
            for p in projects:
                print(f"  - {p.project_name}")
        else:
            # Check physical directories
            if os.path.exists("./output"):
                dirs = [d for d in os.listdir("./output") if os.path.isdir(os.path.join("./output", d)) and not d.startswith(".")]
                if dirs:
                    print("Available project directories:")
                    for d in dirs:
                        print(f"  - {d}")
                else:
                    print("No generated projects found yet.")
            else:
                print("No generated projects found yet.")
        return

    print(f"\n🚀 Starting deployment process for '{project_name}'...")
    
    # Run the deployment asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        deploy_out = loop.run_until_complete(run_deploy(project_name))
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        try:
            update_deploy_status(project_name, "failed", "", "manual")
        except Exception as reg_err:
            print(f"⚠️  Could not persist failure status to registry: {reg_err}")
        return
    finally:
        loop.close()

    primary_url = deploy_out.primary_url
    
    if primary_url and primary_url != "not_deployed":
        print("\n" + "═" * 40)
        print("🎉 DEPLOYMENT SUCCESSFULLY COMPLETED")
        print("═" * 40)
        print(f"🌐 URL: {primary_url}")
        print(f"📢 Share Message:\n   \"{deploy_out.share_message}\"")
        
        # Determine platform deployed
        platform = "manual"
        for t in deploy_out.targets:
            if t.deploy_status == "success":
                platform = t.platform
                break
                
        # Update registry status
        update_deploy_status(project_name, "deployed", primary_url, platform)
        print("═" * 40 + "\n")
    else:
        print("\n" + "═" * 40)
        print("⚠️  MANUAL DEPLOYMENT GUIDE GENERATED")
        print("═" * 40)
        print(deploy_out.instructions)
        print("═" * 40 + "\n")
