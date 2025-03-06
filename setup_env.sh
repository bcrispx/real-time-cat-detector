#!/bin/bash

# Ensure ~/.local/bin exists
mkdir -p "$HOME/.local/bin"

# Add local bin to PATH if it's not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if running on Jetson Nano
if [ -f "/etc/nv_tegra_release" ]; then
    echo "Detected Jetson Nano platform"
    
    # Print system information
    echo "System information:"
    cat /etc/nv_tegra_release
    nvidia-smi
    python3 --version
    
    # Check CUDA installation
    echo "CUDA installation:"
    if [ -f "/usr/local/cuda/version.txt" ]; then
        cat /usr/local/cuda/version.txt
    fi
    echo "CUDA Path:"
    which nvcc
    echo "CUDA Libraries:"
    ldconfig -p | grep -i cuda
    echo "CUDA Environment:"
    env | grep -i cuda
    
    # Try to detect JetPack version more reliably
    echo "Detecting JetPack version..."
    dpkg-query --show nvidia-l4t-core | grep -Po '(?<=-).*' || echo "Could not detect version"
    
    # Remove any existing torch installations
    echo "Removing existing PyTorch installations..."
    sudo apt-get remove -y python3-torch python3-torch-cuda
    sudo pip3 uninstall -y torch torchvision
    
    # Install Jetson-specific dependencies
    echo "Installing Jetson dependencies..."
    sudo apt-get update
    sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev python3-numpy python3-dev
    
    # Set CUDA environment variables
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    # Install Jetson-specific PyTorch
    echo "Installing Jetson-specific PyTorch..."
    
    # First try apt installation which should get the correct CUDA-enabled version
    echo "Installing PyTorch using apt..."
    sudo apt-get install -y python3-torch python3-torch-cuda
    
    # Verify the installation
    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "Apt installation didn't enable CUDA, trying pip installation..."
        # Try pip installation with specific version known to work on Jetson
        sudo pip3 install --no-cache-dir torch==1.8.0 torchvision==0.9.0 --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461
    fi
    
    # If that still fails, try the official NVIDIA forums solution
    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "Trying official NVIDIA solution..."
        # Install dependencies
        sudo apt-get install -y libopenblas-base libopenmpi-dev
        
        # Download and install the wheel directly from NVIDIA
        wget https://nvidia.box.com/shared/static/fjtbwnd5kunq5c3xz77blq9noafeq8vk.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
        sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
        
        # Install compatible torchvision
        sudo pip3 install --no-cache-dir torchvision==0.9.0
    fi
    
    # Verify PyTorch installation and CUDA status
    echo "Verifying PyTorch installation..."
    python3 -c "
import torch
import sys
print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Device count: {torch.cuda.device_count()}')
else:
    print('Torch build info:', torch.__config__.show())
    print('CUDA_HOME:', torch.utils.cpp_extension.CUDA_HOME)
"
    
    # Install other dependencies excluding torch and torchvision
    echo "Installing other dependencies..."
    grep -v "torch\|torchvision" requirements.txt | xargs -r sudo pip3 install
else
    # Install all dependencies normally on non-Jetson platforms
    pip install --user -r requirements.txt
fi

# Source the updated bashrc
source ~/.bashrc

echo "Environment setup complete!"
echo "If you see any PATH warnings, please run: source ~/.bashrc"
echo "or restart your terminal to apply the changes."
