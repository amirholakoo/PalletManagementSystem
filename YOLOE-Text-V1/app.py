"""
YOLOE Smart Eye - Main Application
Web-based object detection system using Raspberry Pi camera and YOLOE model.
"""

from flask import Flask, render_template, Response, jsonify, request
from core import Camera, config
from core.tracker import SimpleObjectTracker
import signal
import sys
import os
import time
import cv2
import numpy as np

app = Flask(__name__)

# Global camera and tracker instances
camera = None
tracker = None
tracking_enabled = False

def signal_handler(sig, frame):
    """Handle cleanup on SIGINT (Ctrl+C)"""
    global camera
    print('Cleaning up...')
    if camera is not None:
        camera.stop()
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def extract_detections_from_results(results, confidence_threshold=0.5):
    """
    Extract detections from YOLOE results and scale them to the display resolution.
    
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

def draw_simple_tracking_visualization(frame, tracks, detections_count=0):
    """
    Draw simple tracking visualization on frame.
    
    Args:
        frame: OpenCV frame to draw on
        tracks: List of SimpleTrack objects
        detections_count: Number of detections in current frame
    """
    for track in tracks:
        x1, y1, x2, y2 = track.bbox

        age_factor = min(1.0, track.age / 10.0)  
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

        if len(track.center_history) > 1:
            points = list(track.center_history)
            for i in range(1, len(points)):
                # Fade trajectory points
                alpha = i / len(points)
                trajectory_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, points[i-1], points[i], trajectory_color, 2)
        
        center = track.center
        cv2.circle(frame, center, 4, color, -1)

    stats_text = f"Tracks: {len(tracks)} | Detections: {detections_count}"
    cv2.putText(frame, stats_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def gen():
    """
    Video streaming generator function with tracking support.
    """
    global camera, tracker, tracking_enabled
    try:
        if camera is None:
            print("Initializing camera...")
            camera = Camera()
            print("Camera initialized successfully")

        while True:
            if tracking_enabled and tracker is not None:
                raw_frame = camera.get_raw_frame()
                results = camera.get_latest_results()

                if raw_frame is None or results is None:
                    time.sleep(0.02)
                    continue
                
                detections = extract_detections_from_results(results, config.CONFIDENCE_THRESHOLD)
                
                # Update tracker
                tracks = tracker.update(detections)
                
                # Debug logging
                if len(detections) > 0 or len(tracks) > 0:
                    print(f"🎯 Tracking: {len(detections)} detections, {len(tracks)} active tracks")
                
                vis_frame = raw_frame.copy()

                draw_simple_tracking_visualization(vis_frame, tracks, len(detections))
                
                status_text = f"TRACKING ENABLED - Active Tracks: {len(tracks)}"
                cv2.putText(vis_frame, status_text, (10, vis_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                ret, jpeg = cv2.imencode('.jpg', vis_frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                if ret:
                    frame_bytes = jpeg.tobytes()
                else:
                    time.sleep(0.01)
                    continue
            else:
                frame_bytes = camera.get_frame()
                if frame_bytes is None:
                    time.sleep(0.1)
                    continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    except Exception as e:
        print(f"Camera error: {e}")
        # Generate a simple error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, 'Camera Error', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        ret, jpeg = cv2.imencode('.jpg', error_frame)
        if ret:
            error_bytes = jpeg.tobytes()
            while True:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + error_bytes + b'\r\n')
                time.sleep(1)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/config')
def show_config():
    """Show current configuration (for debugging)."""
    global camera
    
    # Check if ONNX model exists
    model_path = config.MODEL_PATH
    onnx_path = model_path.replace('.pt', '.onnx')
    onnx_exists = os.path.exists(onnx_path)
    pt_exists = os.path.exists(model_path)
    
    # Get current model type
    current_model_type = "None"
    if camera and hasattr(camera, 'model'):
        current_model_type = camera.model.model_type
    
    config_text = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOE Configuration</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; background: #f8f9fa; }}
            .status {{ padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
            .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }}
            .danger {{ background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
            .btn {{ display: inline-block; padding: 10px 20px; margin: 5px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
            .btn:hover {{ background: #0056b3; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 YOLOE Configuration & Status</h1>
            
            <div class="section">
                <h2>🤖 Model Status</h2>
                <div class="status {'success' if current_model_type == 'ONNX' else 'warning' if current_model_type == 'PyTorch' else 'danger'}">
                    <strong>Current Model Type:</strong> {current_model_type}
                    {' ⚡ (Optimized)' if current_model_type == 'ONNX' else ' 🐌 (Consider converting to ONNX)' if current_model_type == 'PyTorch' else ' ❌ (Not loaded)'}
                </div>
                
                <table>
                    <tr><th>File</th><th>Status</th><th>Action</th></tr>
                    <tr>
                        <td>{model_path}</td>
                        <td>{'✅ Available' if pt_exists else '❌ Missing'}</td>
                        <td>{'Base PyTorch model' if pt_exists else 'Upload required'}</td>
                    </tr>
                    <tr>
                        <td>{onnx_path}</td>
                        <td>{'✅ Available' if onnx_exists else '❌ Missing'}</td>
                        <td>{'Fast ONNX model' if onnx_exists else 'Convert using convert_to_onnx.py'}</td>
                    </tr>
                </table>
                
                {('<div class="status warning"><strong>💡 Recommendation:</strong> Convert to ONNX for better performance using <code>python convert_to_onnx.py</code></div>' if pt_exists and not onnx_exists else '')}
            </div>
            
            <div class="section">
                <h2>📷 Camera Settings</h2>
                <table>
                    <tr><td>Resolution</td><td>{config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}</td></tr>
                    <tr><td>Framerate</td><td>{config.CAMERA_FRAMERATE} FPS</td></tr>
                    <tr><td>Exposure Time</td><td>{config.EXPOSURE_TIME}</td></tr>
                    <tr><td>Analogue Gain</td><td>{config.ANALOGUE_GAIN}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>🧠 AI Model Settings</h2>
                <table>
                    <tr><td>Model Path</td><td>{config.MODEL_PATH}</td></tr>
                    <tr><td>Inference Size</td><td>{config.INFERENCE_WIDTH}x{config.INFERENCE_HEIGHT}</td></tr>
                    <tr><td>Confidence Threshold</td><td>{config.CONFIDENCE_THRESHOLD}</td></tr>
                    <tr><td>IoU Threshold</td><td>{config.IOU_THRESHOLD}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>⚡ Performance Settings</h2>
                <table>
                    <tr><td>Frame Skip Count</td><td>{config.FRAME_SKIP_COUNT}</td></tr>
                    <tr><td>JPEG Quality</td><td>{config.JPEG_QUALITY}%</td></tr>
                    <tr><td>Queue Size</td><td>{config.QUEUE_SIZE}</td></tr>
                    <tr><td>Model Warmup</td><td>{'Enabled' if config.MODEL_WARMUP else 'Disabled'}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>🌐 Web Server Settings</h2>
                <table>
                    <tr><td>Host</td><td>{config.HOST}</td></tr>
                    <tr><td>Port</td><td>{config.PORT}</td></tr>
                    <tr><td>Debug Mode</td><td>{'Enabled' if config.DEBUG else 'Disabled'}</td></tr>
                </table>
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="/" class="btn">🎥 Back to Video Stream</a>
                <a href="/api/status" class="btn">📊 API Status</a>
            </div>
            
            <div class="status warning">
                <strong>📝 Note:</strong> To modify these settings, edit the values in <code>core/config.py</code> and restart the application.
            </div>
        </div>
    </body>
    </html>
    """
    return config_text

@app.route('/api/status')
def api_status():
    """API endpoint for getting system status."""
    global camera
    
    model_path = config.MODEL_PATH
    onnx_path = model_path.replace('.pt', '.onnx')
    
    status = {
        "camera_active": camera is not None,
        "model_type": camera.model.model_type if camera and hasattr(camera, 'model') else "None",
        "model_files": {
            "pytorch": {
                "path": model_path,
                "exists": os.path.exists(model_path)
            },
            "onnx": {
                "path": onnx_path,
                "exists": os.path.exists(onnx_path)
            }
        },
        "config": {
            "camera_resolution": f"{config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}",
            "inference_resolution": f"{config.INFERENCE_WIDTH}x{config.INFERENCE_HEIGHT}",
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "iou_threshold": config.IOU_THRESHOLD,
            "framerate": config.CAMERA_FRAMERATE
        }
    }
    
    return jsonify(status)

@app.route('/convert_to_onnx', methods=['POST'])
def convert_to_onnx():
    """API endpoint to trigger ONNX conversion."""
    try:
        import subprocess
        
        # Run the conversion script
        result = subprocess.run(['python', 'convert_to_onnx.py'], 
                              capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            return jsonify({
                "success": True,
                "message": "ONNX conversion completed successfully!",
                "output": result.stdout
            })
        else:
            return jsonify({
                "success": False,
                "message": "ONNX conversion failed",
                "error": result.stderr
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error during conversion: {str(e)}"
        }), 500

@app.route('/api/tracking/toggle', methods=['POST'])
def toggle_tracking():
    """API endpoint to toggle tracking on/off."""
    global tracking_enabled, tracker
    
    try:
        tracking_enabled = not tracking_enabled
        
        if tracking_enabled:
            if tracker is None:
                # Simple tracker with very persistent settings for robot vision
                tracker = SimpleObjectTracker(max_misses=30, iou_threshold=0.2)
                print("🎯 Persistent Simple Tracker initialized for robot vision")
            print("🎯 Tracking ENABLED")
        else:
            if tracker is not None:
                tracker.reset()
            print("🎯 Tracking DISABLED")
        
        return jsonify({
            "success": True,
            "tracking_enabled": tracking_enabled,
            "message": f"Tracking {'enabled' if tracking_enabled else 'disabled'}"
        })
    except Exception as e:
        print(f"❌ Error toggling tracking: {e}")
        return jsonify({
            "success": False,
            "message": f"Error toggling tracking: {str(e)}"
        }), 500

@app.route('/api/tracking/reset', methods=['POST'])
def reset_tracking():
    """API endpoint to reset tracker."""
    global tracker
    
    try:
        if tracker is not None:
            tracker.reset()
            return jsonify({
                "success": True,
                "message": "Tracker reset successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Tracker not initialized"
            }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error resetting tracker: {str(e)}"
        }), 500

@app.route('/api/tracking/status')
def tracking_status():
    """API endpoint to get tracking status."""
    global tracking_enabled, tracker
    
    active_tracks = len(tracker.get_tracks()) if tracker else 0
    
    return jsonify({
        "tracking_enabled": tracking_enabled,
        "active_tracks": active_tracks,
        "tracker_initialized": tracker is not None
    })

@app.route('/api/debug')
def debug_info():
    """API endpoint for debugging information."""
    global camera, tracker, tracking_enabled
    
    debug_data = {
        "camera_initialized": camera is not None,
        "tracking_enabled": tracking_enabled,
        "tracker_initialized": tracker is not None,
        "active_tracks": len(tracker.get_tracks()) if tracker else 0,
        "camera_has_raw_frame": False,
        "model_type": None
    }
    
    if camera:
        debug_data["camera_has_raw_frame"] = camera.get_raw_frame() is not None
        if hasattr(camera, 'model') and hasattr(camera.model, 'model_type'):
            debug_data["model_type"] = camera.model.model_type
    
    return jsonify(debug_data)

if __name__ == '__main__':
    try:
        print("🎯 Starting YOLOE Smart Eye (ONNX Edition)...")
        print("=" * 60)
        
        # Check model status
        model_path = config.MODEL_PATH
        onnx_path = model_path.replace('.pt', '.onnx')
        
        print(f"📦 PyTorch Model: {model_path} {'✅' if os.path.exists(model_path) else '❌'}")
        print(f"⚡ ONNX Model: {onnx_path} {'✅' if os.path.exists(onnx_path) else '❌'}")
        
        if os.path.exists(onnx_path):
            print("🚀 ONNX model detected - optimized performance enabled!")
        elif os.path.exists(model_path):
            print("💡 Convert to ONNX for better performance: python convert_to_onnx.py")
        else:
            print("⚠️  No model files found - check MODEL_PATH in config.py")
        
        print("=" * 60)
        print(f"🌐 Web interface: http://{config.HOST}:{config.PORT}")
        print(f"⚙️  Configuration: core/config.py")
        print(f"📋 Status page: http://{config.HOST}:{config.PORT}/config")
        print("=" * 60)
        
        # Run with configured parameters
        app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
    finally:
        # Cleanup on exit
        if camera is not None:
            camera.stop() 