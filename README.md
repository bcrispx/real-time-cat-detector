# Real-time Cat Detector

A real-time object detection system with special focus on cat detection, built using YOLOv8 and MediaPipe. The system provides audio feedback when cats are detected and includes additional features for detecting facial features and architectural elements.

## Features
- Real-time cat detection with gentle audio alerts
- General object detection using YOLOv8
- Facial feature detection (eyes, nose, mouth)
- Architectural element detection (doors, windows)
- Object filtering capabilities
- GPU acceleration when available

## Requirements
- Python 3.8+
- Webcam
- NVIDIA GPU (optional, for faster inference)

## Installation
1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python realtime_classifier.py
```

### Controls
- Press 't' to enter filter mode (separate multiple objects with commas)
- Press 'enter' to confirm filter
- Press 'esc' to cancel filter
- Press 'q' to quit

By default, the system detects all objects but provides special audio feedback for cats.

## Technologies Used
- YOLOv8 for general object detection
- MediaPipe for facial feature detection
- OpenCV for image processing and architectural detection
- PyTorch for deep learning inference
- Windows sound system for audio feedback
