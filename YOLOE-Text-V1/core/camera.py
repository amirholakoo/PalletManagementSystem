"""
Camera module for YOLOE Smart Eye project.
Handles Raspberry Pi camera with configurable parameters.
"""

import time
import cv2
from threading import Thread, Lock
from .yoloe_model import YOLOEModel
from .config import config
import queue
import atexit

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

class Camera:
    """Camera class with configurable parameters and optimizations"""
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Camera, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if self._initialized:
            return
            
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError("PiCamera2 library is not installed. Please install with: pip install picamera2")

        try:
            try:
                tuning_file = config.TUNING_FILE
                tuning = Picamera2.load_tuning_file(tuning_file)
                self.picam2 = Picamera2(tuning=tuning)
                print(f"Camera initialized with tuning file: {tuning_file}")
            except Exception as tuning_error:
                print(f"Tuning file error: {tuning_error}, using default settings")
                self.picam2 = Picamera2()
            
            camera_config = self.picam2.create_video_configuration(
                main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"},
                raw={'size': (config.RAW_WIDTH, config.RAW_HEIGHT)},
                controls={
                    "FrameDurationLimits": (config.FRAME_DURATION, config.FRAME_DURATION),
                }
            )
            self.picam2.configure(camera_config)
            self.picam2.start()
            print("Camera started. Allowing to warm up...")
            time.sleep(config.CAMERA_WARMUP_TIME)
            print("Camera ready.")

            self.display_width = config.CAMERA_WIDTH
            self.display_height = config.CAMERA_HEIGHT
            
            self.model = YOLOEModel()
            
            self.frame_queue = queue.Queue(maxsize=config.QUEUE_SIZE)
            self.output_frame = None
            self.raw_frame = None  
            self.latest_results = None 
            self.frame_lock = Lock()
            self.running = True
            
            self.capture_thread = Thread(target=self._capture_frames, daemon=True)
            self.process_thread = Thread(target=self._process_frames, daemon=True)
            
            self.capture_thread.start()
            self.process_thread.start()
            
            self.frame_skip_count = config.FRAME_SKIP_COUNT
            self.frame_counter = 0
            
            atexit.register(self.stop)
            self._initialized = True
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            if hasattr(self, 'picam2'):
                try:
                    self.picam2.stop()
                    self.picam2.close()
                except:
                    pass
            raise

    def _capture_frames(self):
        """Continuously capture frames from camera."""
        while self.running:
            try:
                if not hasattr(self, 'picam2') or self.picam2 is None:
                    break
                    
                frame = self.picam2.capture_array()
                
                # Skip frames based on configuration
                self.frame_counter += 1
                if self.frame_counter % (self.frame_skip_count + 1) != 0:
                    continue
                
                # Add frame to queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame, block=False)
            except Exception as e:
                print(f"Capture error: {e}")
                break

    def _process_frames(self):
        """Process frames with YOLOE model using configured parameters."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)

                frame_resized = cv2.resize(frame, (config.INFERENCE_WIDTH, config.INFERENCE_HEIGHT))
                
                results, annotated_resized = self.model.predict(frame_resized)
                
                annotated_frame = cv2.resize(annotated_resized, (self.display_width, self.display_height))
                
                ret, jpeg = cv2.imencode('.jpg', annotated_frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                if ret:
                    with self.frame_lock:
                        self.output_frame = jpeg.tobytes()
                        self.raw_frame = cv2.resize(frame, (self.display_width, self.display_height))
                        self.latest_results = results
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue

    def get_frame(self):
        """Returns the latest processed frame."""
        with self.frame_lock:
            return self.output_frame
    
    def get_raw_frame(self):
        """Returns the latest raw frame for tracking."""
        with self.frame_lock:
            return self.raw_frame

    def get_latest_results(self):
        """Returns the latest prediction results from the model."""
        with self.frame_lock:
            return self.latest_results

    def stop(self):
        """Stops the camera stream."""
        if not self._initialized:
            return
            
        print("Stopping camera.")
        self.running = False
        
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=config.CAPTURE_THREAD_TIMEOUT)
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=config.PROCESS_THREAD_TIMEOUT)
        
        if hasattr(self, 'picam2') and self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception as e:
                print(f"Error stopping camera: {e}")

        Camera._instance = None
        Camera._initialized = False 
