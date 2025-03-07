@echo off
echo Setting up environment for CUDA with RTX 3090...

:: Create and activate a virtual environment
python -m venv venv
call venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Verify CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

:: Install other requirements
pip install -r requirements.txt

echo Environment setup complete!
echo To activate this environment in the future, run: venv\Scripts\activate.bat
