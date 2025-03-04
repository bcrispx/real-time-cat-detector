# Real-time Cat Detector v1.0.1

First release of the Real-time Cat Detector! This application uses your webcam to detect cats and other objects in real-time, providing gentle audio feedback when cats are detected.

## What's New in v1.0.1
- 🔊 Cross-platform audio support (Windows, Linux, Jetson Nano)
- 🛠️ Improved Linux installation experience
- 🔧 Fixed PATH-related warnings on Linux systems
- 📝 Enhanced documentation for Linux users

## Features
- 🐱 Real-time cat detection with audio alerts
- 🎯 General object detection powered by YOLOv8
- 👁️ Facial feature detection (eyes, nose, mouth)
- 🏠 Basic architectural element detection (doors, windows)
- 🔍 Object filtering capabilities
- 🚀 GPU acceleration when available

## Quick Start

### Windows
1. Download and extract `CatDetector-v1.0.1.zip`
2. Open a terminal in the extracted directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the detector:
   ```bash
   CatDetector.bat
   ```
   Or directly with Python:
   ```bash
   python realtime_classifier.py
   ```

### Linux (including Jetson Nano)
1. Download and extract `CatDetector-v1.0.1.zip`
2. Open a terminal in the extracted directory
3. Make the setup script executable:
   ```bash
   chmod +x setup_env.sh
   ```
4. Run the setup script:
   ```bash
   ./setup_env.sh
   ```
5. Run the detector:
   ```bash
   python realtime_classifier.py
   ```

## Controls
- Press 't' to filter for specific objects (separate multiple objects with commas)
- Press 'enter' to confirm filter
- Press 'esc' to cancel filter
- Press 'q' to quit

## System Requirements
- Python 3.8 or higher
- Windows, Linux, or Jetson Nano
- Webcam
- Internet connection (for first run only)
- NVIDIA GPU (optional, for faster performance)

## Notes
- On first run, the program will download the YOLOv8 model (~6MB)
- The program will create a `.cache` directory in your user folder to store the downloaded model
- Enhanced logging is enabled to help diagnose any potential issues
- On Linux systems, packages are installed in `~/.local/bin`. The setup script will automatically add this to your PATH
