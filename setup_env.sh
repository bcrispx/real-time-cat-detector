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
    
    # Remove any existing torch installations
    pip uninstall -y torch torchvision
    
    # Install Jetson-specific PyTorch
    # The specific version might need to be adjusted based on your JetPack version
    python3 -m pip install --no-cache-dir torch torchvision --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51/
    
    # Verify CUDA installation
    echo "Verifying CUDA installation..."
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    if [ -f "/usr/local/cuda/version.txt" ]; then
        cat /usr/local/cuda/version.txt
    fi
    
    # Install other dependencies excluding torch and torchvision
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
