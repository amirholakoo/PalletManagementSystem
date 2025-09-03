#!/usr/bin/env python3
"""
Convert YOLOE model to ONNX with pre-baked text prompts for better performance
"""

from ultralytics import YOLOE

def convert_model_with_prompts(model_path, text_prompts, output_resolution=320):
    """
    Convert YOLOE model to ONNX with pre-baked text prompts
    
    Args:
        model_path (str): Path to the .pt model file
        text_prompts (list): List of text prompts to bake into the model
        output_resolution (int): Output resolution (must be multiple of 32)
    """
    print(f"Loading YOLOE model from {model_path}...")
    model = YOLOE(model_path)
    
    print(f"Setting text prompts: {text_prompts}")
    model.set_classes(text_prompts, model.get_text_pe(text_prompts))
    
    print(f"Exporting to ONNX with resolution {output_resolution}x{output_resolution}...")
    model.export(format="onnx", imgsz=output_resolution)
    
    onnx_path = model_path.replace('.pt', '.onnx')
    print(f"✅ ONNX model saved to: {onnx_path}")
    print(f"🚀 This model now has '{text_prompts}' prompts baked in for faster inference!")
    
    return onnx_path

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "yoloe-11m-seg.pt"
    TEXT_PROMPTS = ["bag","sack", "gunny", "sack with text on it",
                    "bag of strach naterial", 
                    "white strach sack with some texts on it", 
                    "sack of raw factory materials",
                    "plastic bag containing strach powder",]  
    RESOLUTION = 640  

    print("🔄 Converting YOLOE model to ONNX with text prompts...")
    print("=" * 60)
    
    try:
        onnx_path = convert_model_with_prompts(MODEL_PATH, TEXT_PROMPTS, RESOLUTION)
        print("=" * 60)
        print("✅ Conversion completed successfully!")
        print(f"📁 ONNX model: {onnx_path}")
        print("🎯 You can now use this ONNX model for much faster inference!")
    
    except Exception as e:
        print(f"❌ Error during conversion: {e}") 
