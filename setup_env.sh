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
    
    # Check CUDA installation
    echo "CUDA installation:"
    if [ -f "/usr/local/cuda/version.txt" ]; then
        cat /usr/local/cuda/version.txt
    fi
    ldconfig -p | grep cuda
    
    # Remove any existing torch installations
    echo "Removing existing PyTorch installations..."
    pip uninstall -y torch torchvision
    
    # Install Jetson-specific dependencies
    echo "Installing Jetson dependencies..."
    sudo apt-get update
    sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev
    
    # Install Jetson-specific PyTorch
    echo "Installing Jetson-specific PyTorch..."
    
    # Try the JetPack 5.1 version first
    if ! python3 -m pip install --no-cache-dir torch torchvision --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51/; then
        echo "Failed to install from JetPack 5.1 repository, trying alternative source..."
        # If that fails, try the PyTorch pip wheel for aarch64
        python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
    fi
    
    # Verify PyTorch installation
    echo "Verifying PyTorch installation..."
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Device count: {torch.cuda.device_count()}')
"
    
    # Install other dependencies excluding torch and torchvision
    echo "Installing other dependencies..."
    grep -v "torch\|torchvision" requirements.txt | xargs -r pip install --user
else
    # Install all dependencies normally on non-Jetson platforms
    pip install --user -r requirements.txt
fi

# Source the updated bashrc
source ~/.bashrc

echo "Environment setup complete!"
echo "If you see any PATH warnings, please run: source ~/.bashrc"
echo "or restart your terminal to apply the changes."
