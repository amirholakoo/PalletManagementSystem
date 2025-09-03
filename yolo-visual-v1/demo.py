#!/usr/bin/env python3
"""
YOLOE Visual Prompt Demo Script
This script demonstrates the complete workflow in a simplified manner
"""

import os
import sys
import time

def print_header():
    """Print the demo header"""
    print("🎯" * 50)
    print("🎯 YOLOE Visual Prompt Object Detection System - Demo 🎯")
    print("🎯" * 50)
    print()

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'ultralytics',
        'opencv-python',
        'numpy',
        'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are available")
    return True

def check_files():
    """Check if required files exist"""
    print("\n📁 Checking project files...")
    
    required_files = [
        'imagecapture.py',
        'image-drow-box.py',
        'convertor.py',
        'yolorunmodel.py',
        'yoloe-tracker.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All project files are available")
    return True

def check_models():
    """Check if model files exist"""
    print("\n🤖 Checking model files...")
    
    model_files = [
        'yoloe-11l-seg.pt',
        'yoloe-11s-seg.pt'
    ]
    
    found_models = []
    
    for model in model_files:
        if os.path.exists(model):
            print(f"✅ {model}")
            found_models.append(model)
        else:
            print(f"⚠️  {model} (not found)")
    
    if not found_models:
        print("❌ No model files found. You'll need to download them.")
        return False
    
    print(f"✅ Found {len(found_models)} model file(s)")
    return True

def show_workflow():
    """Show the complete workflow"""
    print("\n🚀 Complete Workflow:")
    print("=" * 50)
    
    workflow_steps = [
        ("1. Image Capture", "python imagecapture.py", "Capture reference image with camera"),
        ("2. Annotation", "python image-drow-box.py", "Draw bounding boxes around objects"),
        ("3. Training", "python convertor.py", "Train YOLOE model with visual prompts"),
        ("4. Detection", "python yolorunmodel.py", "Run real-time object detection"),
        ("5. Tracking", "python yoloe-tracker.py", "Advanced object tracking (optional)")
    ]
    
    for step, command, description in workflow_steps:
        print(f"\n{step}")
        print(f"   Command: {command}")
        print(f"   Purpose: {description}")
    
    print("\n" + "=" * 50)

def show_usage_tips():
    """Show usage tips"""
    print("\n💡 Usage Tips:")
    print("-" * 30)
    
    tips = [
        "• Ensure good lighting for better detection accuracy",
        "• Use clear, high-contrast objects for training",
        "• Start with simple objects before moving to complex scenes",
        "• The system works best with Raspberry Pi camera module",
        "• Adjust camera resolution in scripts if needed",
        "• Use YOLOE-11S for faster inference on slower hardware"
    ]
    
    for tip in tips:
        print(tip)

def main():
    """Main demo function"""
    print_header()
    
    # Check system requirements
    if not check_dependencies():
        print("\n❌ System check failed. Please resolve the issues above.")
        return
    
    if not check_files():
        print("\n❌ Project files check failed. Please ensure all files are present.")
        return
    
    check_models()
    
    # Show workflow
    show_workflow()
    
    # Show tips
    show_usage_tips()
    
    print("\n🎉 Demo completed successfully!")
    print("\nReady to start using the YOLOE Visual Prompt system!")
    print("Follow the workflow steps above to get started.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Please check the error and try again")
