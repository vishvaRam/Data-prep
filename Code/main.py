import torch
import subprocess
import sys

def get_versions():
    versions = {}

    # Python version
    versions['Python'] = sys.version.split()[0]

    # PyTorch version
    if torch.cuda.is_available():
        versions['PyTorch'] = torch.__version__
        versions['CUDA'] = torch.version.cuda
        try:
            gpu_name = torch.cuda.get_device_name(0)
            versions['GPU Name'] = gpu_name
        except Exception as e:
            versions['GPU Name'] = f"N/A (Error: {e})"
    else:
        versions['PyTorch'] = torch.__version__
        versions['CUDA'] = "N/A"
        versions['GPU Name'] = "N/A (CUDA not available)"

    # NVIDIA driver version (if available)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            versions['NVIDIA Driver'] = result.stdout.strip()
        else:
            versions['NVIDIA Driver'] = "N/A (nvidia-smi not found or failed)"
    except FileNotFoundError:
        versions['NVIDIA Driver'] = "N/A (nvidia-smi not found)"

    return versions

if __name__ == "__main__":
    versions = get_versions()
    for key, value in versions.items():
        print(f"{key}: {value}")