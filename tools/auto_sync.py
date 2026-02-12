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
    # Check if there are changes to track
    status = run_command("git status --porcelain", cwd=GIT_DIR)
    return bool(status)

def sync_to_github():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Detecting changes...")
    
    # Add all changes
    run_command("git add .", cwd=GIT_DIR)
    
    # Commit
    commit_msg = f"Auto sync: {timestamp}"
    commit_res = run_command(f'git commit -m "{commit_msg}"', cwd=GIT_DIR)
    
    if commit_res:
        print(f"[{timestamp}] Committed: {commit_msg}")
        
        # Push
        # Note: Credential helper should handle auth if configured
        push_res = run_command("git push origin master", cwd=GIT_DIR) # Assuming master/main
        if push_res is not None:
             print(f"[{timestamp}] Pushed to GitHub successfully.")
        else:
             print(f"[{timestamp}] Push failed. Check credentials/remote.")
    else:
        print(f"[{timestamp}] Commit failed (or nothing to commit).")

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
