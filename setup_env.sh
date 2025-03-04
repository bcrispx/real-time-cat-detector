#!/bin/bash

# Add local bin to PATH if it's not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install dependencies
pip install -r requirements.txt

echo "Environment setup complete. Please run 'source ~/.bashrc' or restart your terminal."
