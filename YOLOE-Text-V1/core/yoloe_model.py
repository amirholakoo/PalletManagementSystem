"""
YOLOE Model wrapper for object detection.
Uses configuration from config.py for all parameters.
"""

import cv2
from ultralytics import YOLOE
import os
import numpy as np
from .config import config

class YOLOEModel:
    """YOLOE model wrapper with ONNX support for better performance"""
    
    def __init__(self):
        """
        Initialize YOLOE model using configuration parameters.
        Prefers ONNX model if available for better performance.
        """
        model_path = config.MODEL_PATH
        onnx_path = model_path.replace('.pt', '.onnx')

        if os.path.exists(onnx_path):
            print(f"✅ Loading ONNX model from {onnx_path} (faster inference)...")
            self.model = YOLOE(onnx_path)
            self.model_type = "ONNX"
        else:
            print(f"📦 Loading PyTorch model from {model_path}...")
            print("💡 Tip: Convert to ONNX using 'python convert_to_onnx.py' for better performance!")
            self.model = YOLOE(model_path)
            self.model_type = "PyTorch"

        if config.MODEL_WARMUP:
            print("Warming up model...")
            dummy_frame = np.zeros((config.INFERENCE_HEIGHT, config.INFERENCE_WIDTH, 3), dtype=np.uint8)
            _ = self.model.predict(dummy_frame, verbose=False)
            print(f"Model warmed up and ready! ({self.model_type})")
    
    def set_text_prompt_classes(self, class_names):
        """
        Set custom classes using text prompts for YOLOE.
        Note: This only works with PyTorch models, not ONNX.
        For ONNX models, text prompts should be baked in during conversion.
        
        Args:
            class_names (list): List of class names for text prompts (e.g., ['black horse', 'car'])
        """
        if self.model_type == "ONNX":
            print("⚠️  Text prompts are already baked into ONNX model - skipping set_text_prompt_classes")
            return
            
        print(f"Setting text prompt classes: {class_names}")
        self.model.set_classes(class_names, self.model.get_text_pe(class_names))
        print("Text prompt classes set successfully!")

    def predict(self, frame):
        """
        Perform object detection using configured parameters.
        
        Args:
            frame (np.ndarray): Input image frame
            
        Returns:
            np.ndarray: Annotated frame with detections
        """
        results = self.model.predict(
            frame, 
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            verbose=False
        )
        
        annotated_frame = results[0].plot(
            boxes=config.SHOW_BOXES, 
            masks=config.SHOW_MASKS, 
            labels=config.SHOW_LABELS
        )
        
        if config.SHOW_FPS:
            inference_time = results[0].speed['inference']
            fps = 1000 / inference_time if inference_time > 0 else 0
            text = f'FPS: {fps:.1f} ({self.model_type})'
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, config.FPS_FONT_SCALE, 2)[0]
            text_x = annotated_frame.shape[1] - text_size[0] - 10
            text_y = text_size[1] + 10
            
            cv2.putText(annotated_frame, text, (text_x, text_y), 
                       font, config.FPS_FONT_SCALE, config.FPS_COLOR, 2, cv2.LINE_AA)
        
        return results, annotated_frame

if __name__ == '__main__':
    # Test the model
    model = YOLOEModel()
    print("Model test completed successfully!") 
