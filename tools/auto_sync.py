import time
import subprocess
import os
import sys
from datetime import datetime

# Configuration
CHECK_INTERVAL = 10  # Check every 10 seconds
GIT_DIR = "/root/autodl-tmp/dky/foggy_test"

def run_command(cmd, cwd=None):
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Stderr: {e.stderr}")
        return None

def check_for_changes():
    # Check for uncommitted changes
    status = run_command("git status --porcelain", cwd=GIT_DIR)
    if status:
        return True
        
    # Check for unpushed commits
    # This requires a remote to be set up. If no remote, this might fail or return nothing.
    try:
        # Check if we are ahead of remote
        # We use 'git rev-list --count @{u}..HEAD' to see how many commits we are ahead of upstream
        # If no upstream is set, this command will fail.
        # Alternatively, try to push if we haven't pushed in a while? 
        # Easier: just try to push if we are ahead.
        ahead = run_command("git rev-list HEAD --not --remotes", cwd=GIT_DIR)
        return bool(ahead)
    except:
        return False

def sync_to_github():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Syncing...")
    
    # 1. Commit any local changes
    status = run_command("git status --porcelain", cwd=GIT_DIR)
    if status:
        print(f"[{timestamp}] Committing changes...")
        run_command("git add .", cwd=GIT_DIR)
        commit_msg = f"Auto sync: {timestamp}"
        run_command(f'git commit -m "{commit_msg}"', cwd=GIT_DIR)
    
    # 2. Push
    print(f"[{timestamp}] Pushing to GitHub...")
    push_res = run_command("git push origin master", cwd=GIT_DIR)
    if push_res is not None:
         print(f"[{timestamp}] Push successful.")
    else:
         print(f"[{timestamp}] Push failed.")

def main():
    print(f"Starting auto-sync monitor for {GIT_DIR}...")
    print(f"Check interval: {CHECK_INTERVAL} seconds")
    
    # Initial check
    if check_for_changes():
        sync_to_github()
        
    while True:
        try:
            time.sleep(CHECK_INTERVAL)
            if check_for_changes():
                sync_to_github()
        except KeyboardInterrupt:
            print("Auto-sync stopped.")
            break
        except Exception as e:
            print(f"Error in auto-sync loop: {e}")

if __name__ == "__main__":
    main()
