#!/usr/bin/env python3
"""
YOLOE Smart Eye - Standalone Detection System
Real-time object detection using Raspberry Pi camera and YOLOE model.

Usage: python yoloe_detection.py
- Press 'q' to quit
- Press 's' to save current frame
- All settings can be modified in core/config.py
"""

import cv2
import time
import signal
import sys
from core import Camera, config


TEXT_PROMPT_CLASSES = ["bag","sack", "gunny", 
                       "sack with text on it", "bag of strach naterial",
                       "white strach sack with some texts on it", "sack of raw factory materials",
                       "plastic bag containing strach powder",]  

def signal_handler(sig, frame):
    """Handle cleanup on Ctrl+C"""
    print("\nShutting down...")
    global camera
    if camera is not None:
        camera.stop()
    sys.exit(0)

def main():
    """Main detection loop"""
    global camera
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🎯 YOLOE Smart Eye - Standalone Detection")
    print("=" * 50)
    config.print_config()
    
    if TEXT_PROMPT_CLASSES:
        print(f"🔍 Text Prompt Classes: {TEXT_PROMPT_CLASSES}")
    
    print("=" * 50)
    print("Starting camera and model...")
    
    try:
        camera = Camera()
        
        if TEXT_PROMPT_CLASSES:
            print("Setting up text prompts...")
            camera.model.set_text_prompt_classes(TEXT_PROMPT_CLASSES)
        
        print("✅ Camera and model ready!")
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        print("- Press 'c' to show config")
        print("\nStarting detection...")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            frame_bytes = camera.get_frame()
            
            if frame_bytes is not None:
                import numpy as np
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    frame_count += 1
                    elapsed = time.time() - start_time
                    if elapsed > 1.0:
                        overall_fps = frame_count / elapsed
                        print(f"📊 Overall FPS: {overall_fps:.1f}")
                        frame_count = 0
                        start_time = time.time()
                    
                    cv2.imshow('YOLOE Smart Eye', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        filename = f"detection_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"💾 Frame saved as {filename}")
                    elif key == ord('c'):
                        config.print_config()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean shutdown
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()
        print("🏁 Detection stopped.")

if __name__ == "__main__":
    camera = None
    main() 
