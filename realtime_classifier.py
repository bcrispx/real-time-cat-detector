import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch
import logging
import mediapipe as mp
import os
from datetime import datetime
import sys
import pygame

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Check if running on Jetson Nano
is_jetson = os.path.exists('/etc/nv_tegra_release')
if is_jetson:
    logging.info("Detected Jetson Nano platform")
    # Try to force CUDA initialization
    if not torch.cuda.is_available():
        logging.warning("CUDA not available initially, attempting to force initialization...")
        try:
            # Try to manually set CUDA_VISIBLE_DEVICES
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            # Force torch to reinitialize CUDA
            torch.cuda.init()
            # Clear the cache
            torch.cuda.empty_cache()
            logging.info("CUDA initialization attempted")
        except Exception as e:
            logging.error(f"Error during CUDA initialization: {str(e)}")
    
    # Log CUDA information
    logging.info(f"CUDA initialization: {torch.cuda.is_initialized()}")
    logging.info(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
        logging.info(f"CUDA version: {torch.version.cuda}")

# Log system information
logging.info(f"OpenCV version: {cv2.__version__}")
logging.info(f"PyTorch version: {torch.__version__}")
logging.info(f"CUDA available: {torch.cuda.is_available()}")
logging.info(f"Current directory: {os.getcwd()}")

class ObjectDetector:
    def __init__(self):
        try:
            # Check if CUDA is available
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f"Using device: {self.device}")
            
            if self.device == 'cuda':
                # Ensure CUDA is initialized
                if not torch.cuda.is_initialized():
                    torch.cuda.init()
                # Try to optimize CUDA performance
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.fastest = True
                logging.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)}")
                logging.info(f"CUDA memory cached: {torch.cuda.memory_reserved(0)}")
            
            # Initialize YOLOv8 for general object detection
            logging.info("Loading YOLOv8 model...")
            self.yolo_model = YOLO('yolov8n.pt')
            if self.device == 'cuda':
                self.yolo_model.to(self.device)
            
            # Initialize MediaPipe for face and facial features
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=3,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            
            # Initialize Haar cascades for architectural elements
            self.door_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.window_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # Get YOLO class names
            self.class_names = self.yolo_model.names
            logging.info(f"Loaded {len(self.class_names)} YOLO classes")
            
            # Additional class names for facial features and architecture
            self.additional_classes = {
                'nose': 'nose',
                'left_eye': 'eye',
                'right_eye': 'eye',
                'mouth': 'mouth',
                'door': 'door',
                'window': 'window'
            }
            
            # Initialize pygame for audio
            pygame.mixer.init()
            # Generate a gentle beep sound
            sample_rate = 44100
            duration = 0.15  # 150ms
            frequency = 600  # 600Hz
            t = np.linspace(0, duration, int(sample_rate * duration))
            beep_data = np.sin(2 * np.pi * frequency * t)
            beep_data = np.int16(beep_data * 32767)  # Convert to 16-bit integer
            self.beep_sound = pygame.mixer.Sound(beep_data)
            
            # Sound effect settings
            self.last_cat_sound = datetime.min
            self.sound_cooldown = 1.0  # seconds between sounds
            
            # Initialize search terms
            self.search_terms = set()  # Empty set means detect all objects
            self.conf_threshold = 0.3
            
            # Color map for different objects
            self.color_map = {}
            self.base_colors = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Blue
                (0, 255, 255),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 128),  # Light green
                (128, 0, 255),  # Purple
            ]
            
        except Exception as e:
            logging.error(f"Error initializing detector: {str(e)}")
            raise
    
    def get_object_color(self, class_name):
        if class_name not in self.color_map:
            # Assign a new color from the base colors
            color_idx = len(self.color_map) % len(self.base_colors)
            self.color_map[class_name] = self.base_colors[color_idx]
        return self.color_map[class_name]
    
    def play_cat_sound(self):
        try:
            now = datetime.now()
            if (now - self.last_cat_sound).total_seconds() >= self.sound_cooldown:
                self.beep_sound.play()
                self.last_cat_sound = now
        except Exception as e:
            logging.error(f"Error playing cat sound: {str(e)}")

    def detect_facial_features(self, frame):
        detections = []
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get image dimensions
                h, w, _ = frame.shape
                
                # Detect eyes
                left_eye = face_landmarks.landmark[33]  # Left eye center
                right_eye = face_landmarks.landmark[263]  # Right eye center
                
                # Detect nose
                nose = face_landmarks.landmark[4]  # Nose tip
                
                # Detect mouth
                mouth = face_landmarks.landmark[13]  # Upper lip
                
                # Convert normalized coordinates to pixel coordinates
                features = {
                    'left_eye': (int(left_eye.x * w), int(left_eye.y * h)),
                    'right_eye': (int(right_eye.x * w), int(right_eye.y * h)),
                    'nose': (int(nose.x * w), int(nose.y * h)),
                    'mouth': (int(mouth.x * w), int(mouth.y * h))
                }
                
                # Create detection boxes for each feature
                for name, (x, y) in features.items():
                    size = 30 if name == 'mouth' else 20  # Larger box for mouth
                    detections.append({
                        'box': (x-size//2, y-size//2, size, size),
                        'class': name,
                        'conf': 0.9  # High confidence for detected features
                    })
        
        return detections
    
    def detect_architecture(self, frame):
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect doors (using face cascade as placeholder - you might want to train a specific cascade)
        doors = self.door_cascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 200))
        for (x, y, w, h) in doors:
            if h/w > 1.5:  # Door-like aspect ratio
                detections.append({
                    'box': (x, y, w, h),
                    'class': 'door',
                    'conf': 0.7
                })
        
        # Detect windows (using eye cascade as placeholder - you might want to train a specific cascade)
        windows = self.window_cascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))
        for (x, y, w, h) in windows:
            if 0.5 < h/w < 2.0:  # Window-like aspect ratio
                detections.append({
                    'box': (x, y, w, h),
                    'class': 'window',
                    'conf': 0.7
                })
        
        return detections
    
    def process_frame(self, frame):
        try:
            all_detections = []
            cat_detected = False
            
            # Run YOLO detection
            yolo_results = self.yolo_model.predict(frame, 
                                                 conf=self.conf_threshold,
                                                 device=self.device,
                                                 verbose=False)[0]
            
            # Process YOLO results
            boxes = yolo_results.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id].lower()
                
                # Check for cat detection
                if class_name == 'cat':
                    cat_detected = True
                
                all_detections.append({
                    'box': (x1, y1, x2-x1, y2-y1),
                    'class': class_name,
                    'conf': confidence
                })
            
            # Play sound if cat detected
            if cat_detected:
                self.play_cat_sound()
            
            # Run facial feature detection
            facial_detections = self.detect_facial_features(frame)
            all_detections.extend(facial_detections)
            
            # Run architectural element detection
            arch_detections = self.detect_architecture(frame)
            all_detections.extend(arch_detections)
            
            # Draw all detections that match any search term
            for det in all_detections:
                class_name = det['class']
                if not self.search_terms or any(term.lower() in class_name for term in self.search_terms):
                    x, y, w, h = det['box']
                    conf = det['conf']
                    
                    # Get color for this object type
                    color = self.get_object_color(class_name)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Add label with confidence
                    label = f"{class_name} {conf:.2f}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
                    cv2.putText(frame, label, (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            return frame
        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}")
            logging.error(f"Error details: {str(e.__class__.__name__)}")
            import traceback
            traceback.print_exc()
            return frame

class TextBox:
    def __init__(self):
        self.text = ""
        self.active = False
        
    def draw(self, frame):
        # Draw text box background
        cv2.rectangle(frame, (10, frame.shape[0] - 60), (frame.shape[1] - 10, frame.shape[0] - 20), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, frame.shape[0] - 60), (frame.shape[1] - 10, frame.shape[0] - 20), (0, 255, 0), 2)
        
        # Draw prompt text
        prompt = "Type objects to detect (comma-separated): " if self.active else "Press 'T' to filter objects"
        cv2.putText(frame, prompt + self.text, (20, frame.shape[0] - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        
        if self.active:
            # Draw cursor
            text_size = cv2.getTextSize(prompt + self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            cursor_x = 20 + text_size[0] + 5
            cv2.line(frame, (cursor_x, frame.shape[0] - 45), (cursor_x, frame.shape[0] - 30), (0, 255, 0), 1)

def main():
    try:
        # Initialize detector and text box
        detector = ObjectDetector()
        text_box = TextBox()
        
        # Initialize webcam
        logging.info("Initializing webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Failed to open webcam. Please ensure your webcam is connected and not in use by another application.")
            raise Exception("Failed to open webcam. Please check your camera connection.")
        
        # Set lower resolution for better performance
        if not cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640):
            logging.warning("Failed to set webcam width")
        if not cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480):
            logging.warning("Failed to set webcam height")
        
        logging.info("Webcam initialized successfully")
        logging.info("Press 'q' to quit")
        logging.info("Press 't' to filter objects (separate multiple objects with commas)")
        logging.info("Press 'enter' to confirm")
        logging.info("Press 'esc' to cancel")
        logging.info("\nDetecting all objects by default")
        
        # For frame rate control
        fps = 30
        frame_time = 1/fps
        prev_time = time.time()
        
        while True:
            # Control frame rate
            current_time = time.time()
            elapsed = current_time - prev_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            prev_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to grab frame")
                break
            
            # Process frame with current search term
            frame = detector.process_frame(frame)
            
            # Draw text box
            text_box.draw(frame)
            
            # Show frame
            cv2.imshow('Object Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if text_box.active:
                if key == 13:  # Enter key
                    # Split by comma and strip whitespace
                    terms = [t.strip() for t in text_box.text.split(',') if t.strip()]
                    detector.search_terms = set(terms)
                    logging.info(f"Now filtering for: {', '.join(terms) if terms else 'all objects'}")
                    text_box.text = ""
                    text_box.active = False
                elif key == 27:  # Escape key
                    text_box.text = ""
                    text_box.active = False
                elif key == 8:  # Backspace
                    text_box.text = text_box.text[:-1]
                elif 32 <= key <= 126:  # Printable characters
                    text_box.text += chr(key)
            else:
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    text_box.active = True
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
