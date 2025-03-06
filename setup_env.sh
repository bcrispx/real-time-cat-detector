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
    
    # Create a temporary directory for downloads
    TEMP_DIR=$(mktemp -d)
    cd $TEMP_DIR
    
    # Download and install the latest Jetson PyTorch wheel
    echo "Downloading PyTorch for Jetson..."
    wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.11-cp38-cp38-linux_aarch64.whl
    
    if [ $? -eq 0 ]; then
        echo "Installing PyTorch..."
        sudo pip3 install torch-2.1.0a0+41361538.nv23.11-cp38-cp38-linux_aarch64.whl
        
        echo "Installing torchvision..."
        wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torchvision-0.16.0a0+0b5855f.nv23.11-cp38-cp38-linux_aarch64.whl
        sudo pip3 install torchvision-0.16.0a0+0b5855f.nv23.11-cp38-cp38-linux_aarch64.whl
    else
        echo "Failed to download PyTorch wheel, trying alternative method..."
        # Try installing from NVIDIA repository
        sudo pip3 install --no-cache-dir torch torchvision --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512
    fi
    
    # Clean up
    cd -
    rm -rf $TEMP_DIR
    
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
