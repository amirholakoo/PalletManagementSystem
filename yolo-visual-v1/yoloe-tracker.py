#!/usr/bin/env python3
"""
YOLOE Tracker - Object Tracking System
Real-time object tracking using YOLOE model with tracking capabilities.
"""

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import time

# Set up the camera with Picam
picam2 = Picamera2()
picam2.preview_configuration.main.size = (800, 800)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLOE model
model = YOLO("yoloe-11l-seg.onnx")

# Initialize tracking variables
tracked_objects = {}
frame_count = 0
start_time = time.time()

print("🎯 YOLOE Tracker Started")
print("Press 'q' to quit, 's' to save frame")

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()
    
    # Run YOLOE model with tracking
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    
    # Process tracking results
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        # Update tracked objects
        for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
            if track_id not in tracked_objects:
                tracked_objects[track_id] = {
                    'class': int(cls),
                    'first_seen': frame_count,
                    'total_detections': 0
                }
            
            tracked_objects[track_id]['total_detections'] += 1
            
            # Draw tracking info on frame
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{int(track_id)}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Output the visual detection data
    annotated_frame = results[0].plot(boxes=True, masks=False)
    
    # Calculate and display FPS
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()
    
    # Display tracking stats
    active_tracks = len(tracked_objects)
    stats_text = f'FPS: {fps:.1f} | Tracks: {active_tracks}'
    cv2.putText(annotated_frame, stats_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the resulting frame
    cv2.imshow("YOLOE Tracker", annotated_frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        filename = f"tracking_{int(time.time())}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"💾 Frame saved as {filename}")

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
print("🏁 Tracking stopped.") 
