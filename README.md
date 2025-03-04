# Real-time Cat Detector

A real-time object detection system with special focus on cat detection, built using YOLOv8 and MediaPipe. The system provides audio feedback when cats are detected and includes additional features for detecting facial features and architectural elements.

## Features
- 🐱 Real-time cat detection with gentle audio alerts
- 🎯 General object detection using YOLOv8
- 👁️ Facial feature detection (eyes, nose, mouth)
- 🏠 Architectural element detection (doors, windows)
- 🔍 Object filtering capabilities
- 🚀 GPU acceleration when available

## Latest Release: v1.0.1
What's new:
- 🔊 Cross-platform audio support (Windows, Linux, Jetson Nano)
- 🛠️ Improved Linux installation experience
- 🔧 Fixed PATH-related warnings on Linux systems
- 📝 Enhanced documentation for Linux users

See the [release notes](release_notes.md) for full details.

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- NVIDIA GPU (optional, for faster performance)

### Quick Start (Windows)
1. Download `CatDetector-v1.0.1.zip` from the [Releases](https://github.com/bcrispx/real-time-cat-detector/releases) page
2. Extract the ZIP file
3. Open a terminal in the extracted directory
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the detector:
   ```bash
   CatDetector.bat
   ```
   Or directly with Python:
   ```bash
   python realtime_classifier.py
   ```

### Quick Start (Linux/Jetson Nano)
1. Download and extract the release:
   ```bash
   wget https://github.com/bcrispx/real-time-cat-detector/releases/download/v1.0.1/CatDetector-v1.0.1.zip
   unzip CatDetector-v1.0.1.zip
   cd CatDetector-v1.0.1
   ```

2. Set up the environment and install dependencies:
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

3. Run the detector:
   ```bash
   python realtime_classifier.py
   ```

### Controls
- Press 't' to enter filter mode (separate multiple objects with commas)
- Press 'enter' to confirm filter
- Press 'esc' to cancel filter
- Press 'q' to quit

By default, the system detects all objects but provides special audio feedback for cats.

## Notes
- On first run, the program will download the YOLOv8 model (~6MB)
- The program will create a `.cache` directory in your user folder to store the downloaded model
- On Linux systems, packages are installed in `~/.local/bin`

## Troubleshooting

### Linux/Jetson Nano
- If you see warnings about scripts not being in PATH, run the `setup_env.sh` script which will fix this issue
- If you get webcam permission errors, ensure your user is in the `video` group:
  ```bash
  sudo usermod -a -G video $USER
  ```
  Then log out and back in for the changes to take effect

## Technologies Used
- YOLOv8 for general object detection
- MediaPipe for facial feature detection
- OpenCV for image processing and architectural detection
- PyTorch for deep learning inference
- Pygame for cross-platform audio feedback
