#!/usr/bin/env python3
"""
YOLOE Smart Eye - Tracking-Enabled Detection System
Real-time object detection and tracking using Raspberry Pi camera and YOLOE model.

Usage: python yoloe_tracking.py
- Press 'q' to quit
- Press 's' to save current frame
- Press 't' to toggle trajectory display
- Press 'r' to reset tracker
- All settings can be modified in core/config.py
"""

import cv2
import time
import signal
import sys
import numpy as np
from core import Camera, config
from core.tracker import SimpleObjectTracker

# =============================================================================
# TRACKING CONFIGURATION
# =============================================================================
TRACKING_CONFIG = {
    'max_misses': 30,        
    'iou_threshold': 0.2,    
    'show_trajectories': True,  
    'trajectory_length': 5,     
}

TEXT_PROMPT_CLASSES = ["bag","sack", "gunny", "sack with text on it", 
                       "bag of strach naterial", 
                       "white strach sack with some texts on it",
                       "sack of raw factory materials",
                       "plastic bag containing strach powder",]  

def signal_handler(sig, frame):
    """Handle cleanup on Ctrl+C"""
    print("\nShutting down...")
    global camera
    if camera is not None:
        camera.stop()
    sys.exit(0)

def draw_tracking_visualization(frame, tracks, show_trajectories=True):
    """
    Draw simple tracking visualization on frame.
    
    Args:
        frame: OpenCV frame to draw on
        tracks: List of SimpleTrack objects
        show_trajectories: Whether to show object trajectories
    """
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        
        if track.misses == 0:
            color = (0, 255, 0)      
        else:
            fade_factor = max(0.3, 1.0 - (track.misses / 5.0))
            color = (0, int(255 * fade_factor), int(255 * (1 - fade_factor)))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{track.id} {track.confidence:.2f}"
        if track.misses > 0:
            label += f" (pred)"
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if show_trajectories and len(track.center_history) > 1:
            points = list(track.center_history)
            for i in range(1, len(points)):
                # Fade trajectory points
                alpha = i / len(points)
                trajectory_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, points[i-1], points[i], trajectory_color, 2)
        
        center = track.center
        cv2.circle(frame, center, 4, color, -1)

        vx, vy = track.velocity
        if abs(vx) > 2 or abs(vy) > 2:  # Only draw if moving
            end_point = (center[0] + vx * 3, center[1] + vy * 3)
            cv2.arrowedLine(frame, center, end_point, color, 2)

def extract_detections_from_results(results, confidence_threshold=0.5):
    """
    Extract detections from YOLOE results and scale them for display.
    
    Args:
        results: YOLOE prediction results from the inference-sized image.
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        List of (bbox, class_id, confidence) tuples, scaled to camera resolution.
    """
    detections = []
    
    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        from_w = config.INFERENCE_WIDTH
        from_h = config.INFERENCE_HEIGHT
        to_w = config.CAMERA_WIDTH
        to_h = config.CAMERA_HEIGHT
        scale_x = to_w / from_w
        scale_y = to_h / from_h
        
        for box, class_id, conf in zip(boxes, class_ids, confidences):
            if conf >= confidence_threshold:
                x1, y1, x2, y2 = box
                
 
                scaled_bbox = (
                    int(x1 * scale_x), 
                    int(y1 * scale_y), 
                    int(x2 * scale_x), 
                    int(y2 * scale_y)
                )
                detections.append((scaled_bbox, int(class_id), float(conf)))
    
    return detections

def main():
    """Main tracking detection loop"""
    global camera

    signal.signal(signal.SIGINT, signal_handler)
    
    print("🎯 YOLOE Smart Eye - Tracking-Enabled Detection")
    print("=" * 60)
    config.print_config()
    
    print(f"🔍 Tracking Config:")
    for key, value in TRACKING_CONFIG.items():
        print(f"   {key}: {value}")
    
    if TEXT_PROMPT_CLASSES:
        print(f"🔍 Text Prompt Classes: {TEXT_PROMPT_CLASSES}")
    
    print("=" * 60)
    print("Starting camera, model, and tracker...")
    
    try:
        camera = Camera()
        tracker = SimpleObjectTracker(
            max_misses=TRACKING_CONFIG['max_misses'],
            iou_threshold=TRACKING_CONFIG['iou_threshold']
        )
        
        if TEXT_PROMPT_CLASSES:
            print("Setting up text prompts...")
            camera.model.set_text_prompt_classes(TEXT_PROMPT_CLASSES)
        
        print("✅ Camera, model, and tracker ready!")
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        print("- Press 't' to toggle trajectory display")
        print("- Press 'r' to reset tracker")
        print("- Press 'c' to show config")
        print("\nStarting tracking detection...")
        
        frame_count = 0
        start_time = time.time()
        show_trajectories = TRACKING_CONFIG['show_trajectories']
        
        while True:
            raw_frame = camera.get_raw_frame()
            results = camera.get_latest_results()
            
            if raw_frame is not None and results is not None:
                
                detections = extract_detections_from_results(results, config.CONFIDENCE_THRESHOLD)

                tracks = tracker.update(detections)
                
                vis_frame = raw_frame.copy()

                draw_tracking_visualization(vis_frame, tracks, show_trajectories)
                
                stats_text = f"Tracks: {len(tracks)} | Detections: {len(detections)}"
                cv2.putText(vis_frame, stats_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    fps = frame_count / elapsed
                    print(f"📊 FPS: {fps:.1f} | Active Tracks: {len(tracks)} | Detections: {len(detections)}")
                    frame_count = 0
                    start_time = time.time()

                fps_text = f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --"
                cv2.putText(vis_frame, fps_text, (vis_frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('YOLOE Smart Eye - Tracking', vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"tracking_{int(time.time())}.jpg"
                    cv2.imwrite(filename, vis_frame)
                    print(f"💾 Frame saved as {filename}")
                elif key == ord('t'):
                    show_trajectories = not show_trajectories
                    print(f"🎯 Trajectories: {'ON' if show_trajectories else 'OFF'}")
                elif key == ord('r'):
                    tracker.reset()
                    print("🔄 Tracker reset")
                elif key == ord('c'):
                    config.print_config()

            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()
        print("🏁 Tracking detection stopped.")

if __name__ == "__main__":
    camera = None
    main()
