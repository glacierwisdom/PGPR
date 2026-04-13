
import os
import sys
import zipfile
import paramiko
import shutil
from pathlib import Path
import time

# Configuration
HOST = "36.103.199.234"
USER = "ubuntu"
PASSWORD = "zfVNgmyzW.hwEx1Z"
REMOTE_DIR = "/home/ubuntu/GAPNPPI"
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROJECT_NAME = "GAPNPPI"
ZIP_NAME = "GAPNPPI_deploy.zip"

def create_zip():
    print(f"Creating zip archive of {LOCAL_DIR}...")
    zip_path = os.path.join(LOCAL_DIR, ZIP_NAME)
    
    # Files/Dirs to exclude
    exclude_dirs = {'.git', '__pycache__', 'logs', 'results', 'wandb', '.idea', '.vscode', 'checkpoints'}
    exclude_files = {ZIP_NAME, '.DS_Store'}
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(LOCAL_DIR):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files:
                    continue
                if file.endswith('.pyc'):
                    continue
                    
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, LOCAL_DIR)
                zipf.write(file_path, arcname)
                
    print(f"Zip created at {zip_path}, size: {os.path.getsize(zip_path) / 1024 / 1024:.2f} MB")
    return zip_path

def connect_ssh():
    print(f"Connecting to {HOST}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD)
    print("Connected successfully.")
    return client

def run_command(client, command, print_output=True):
    print(f"Running remote command: {command}")
    stdin, stdout, stderr = client.exec_command(command)
    
    exit_status = stdout.channel.recv_exit_status()
    
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    
    if print_output:
        if out: print(f"STDOUT:\n{out}")
        if err: print(f"STDERR:\n{err}")
        
    if exit_status != 0:
        print(f"Command failed with exit status {exit_status}")
        return False, out + "\n" + err
    return True, out

def deploy():
    # 1. Create Zip
    zip_path = create_zip()
    
    client = None
    try:
        # 2. Connect
        client = connect_ssh()
        sftp = client.open_sftp()
        
        # 3. Setup remote directory
        print(f"Setting up remote directory {REMOTE_DIR}...")
        run_command(client, f"mkdir -p {REMOTE_DIR}")
        
        # 4. Upload Zip
        remote_zip = f"{REMOTE_DIR}/{ZIP_NAME}"
        print(f"Uploading {zip_path} to {remote_zip}...")
        sftp.put(zip_path, remote_zip)
        print("Upload complete.")
        
        # 5. Unzip
        print("Unzipping...")
        run_command(client, f"cd {REMOTE_DIR} && unzip -o {ZIP_NAME}")
        
        # 6. Install dependencies
        print("Installing dependencies (this may take a while)...")
        # Install pip if missing (unlikely on ubuntu)
        run_command(client, "sudo apt-get update && sudo apt-get install -y python3-pip unzip")
        
        # Install torch with CUDA support
        print("Installing PyTorch with CUDA support...")
        run_command(client, "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        # Install other requirements
        print("Installing other requirements...")
        run_command(client, f"cd {REMOTE_DIR} && pip3 install -r requirements.txt")
        
        # 7. Verify GPU
        print("Verifying GPU...")
        success, out = run_command(client, "nvidia-smi")
        if not success:
            print("WARNING: nvidia-smi failed. GPU might not be available.")
        
        success, out = run_command(client, "python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"Device count: {torch.cuda.device_count()}\")'")
        
        # 8. Clean up zip
        run_command(client, f"rm {remote_zip}")
        
        print("\nDeployment successful!")
        print(f"Project deployed to {REMOTE_DIR}")
        
        # 9. Start Benchmark (Optional)
        # We don't start it automatically here to avoid hanging, but we can return instructions
        print("\nTo start training, run:")
        print(f"ssh {USER}@{HOST}")
        print(f"cd {REMOTE_DIR}")
        print("nohup python3 run_benchmark.py --strategies bs > benchmark.log 2>&1 &")
        
    except Exception as e:
        print(f"Deployment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            client.close()
        # Clean up local zip
        if os.path.exists(zip_path):
            os.remove(zip_path)

if __name__ == "__main__":
    deploy()
