# YOLOE Visual Prompt Object Detection System

A complete object detection and tracking system using YOLOE (You Only Look Once Efficient) with visual prompts, designed for Raspberry Pi with camera integration.

## 🎯 Project Overview

This system demonstrates a complete workflow for training and deploying a custom YOLOE model using visual prompts. The process involves capturing reference images, annotating bounding boxes, training the model, and running real-time inference.

## 🚀 Quick Start (Existing Images)

If you already have trained models and want to run detection:

```bash
# Run real-time object detection
python yolorunmodel.py

# For object tracking (optional)
python yoloe-tracker.py
```

## 📋 Complete Workflow for New Images

**For users who want to capture new images and train new models, see:**
**[📖 Complete Workflow Guide](WORKFLOW_GUIDE.md)**

This guide covers the entire process from image capture to model training.

## 🎬 Current System Components

### 1. Image Capture (`imagecapture.py`)
- Captures reference photos using Raspberry Pi camera
- Position objects in frame and press SPACE to capture
- Press ESC to exit
- Saves captured image as `image.jpg`

### 2. Bounding Box Annotation (`image-drow-box.py`)
- Interactive tool for drawing bounding boxes around objects
- Click and drag to create boxes
- Each box gets a unique Class ID (0, 1, 2...)
- Press 'R' to reset all boxes, 'Q' to quit
- Annotates the captured image with bounding box coordinates

### 3. Model Training (`convertor.py`)
- Loads the annotated image and bounding boxes
- Trains YOLOE model using visual prompts
- Exports trained model to ONNX format
- Creates `yoloe-11L-seg.onnx` for deployment

### 4. Real-time Inference (`yolorunmodel.py`)
- Loads the trained ONNX model
- Runs real-time object detection on camera feed
- Displays detection results with bounding boxes
- Shows FPS performance metrics
- Press 'q' to quit

### 5. Object Tracking (`yoloe-tracker.py`)
- Advanced tracking system with persistent object IDs
- Tracks objects across frames
- Displays tracking statistics and FPS
- Press 'q' to quit, 's' to save frame

## 📋 Prerequisites

- Raspberry Pi with camera module
- Python 3.8+
- Sufficient storage for model files
- Good lighting conditions for image capture

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/amirholakoo/PalletManagementSystem

cd PalletManagementSystem/yoloe-visual-v1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your camera is properly connected and configured

## 🎯 Usage Scenarios

### Scenario 1: Use Existing Trained Models
If you already have trained models and want to run detection:
```bash
python yolorunmodel.py      # Run detection
python yoloe-tracker.py     # Run tracking (optional)
```

### Scenario 2: Train New Models with New Images
Follow the complete workflow guide:
1. Capture new images
2. Annotate bounding boxes
3. Train the model
4. Run detection

See **[📖 Complete Workflow Guide](WORKFLOW_GUIDE.md)** for detailed instructions.

## ⚙️ Configuration

### Camera Settings
- Resolution: 800x800 pixels
- Format: RGB888
- Adjustable in each script if needed

### Model Settings
- Default model: YOLOE-11L (large)
- Alternative: YOLOE-11S (small, faster)
- Input size: 640x640 (configurable)

## 🔧 Troubleshooting

### Common Issues:
1. **Camera not detected**: Ensure camera is properly connected and enabled
2. **Model training fails**: Check available memory and storage space
3. **Low FPS**: Consider using the smaller YOLOE-11S model
4. **Poor detection**: Ensure good lighting and clear reference images

### Performance Tips:
- Use YOLOE-11S for faster inference on slower hardware
- Optimize camera resolution based on your needs
- Ensure adequate lighting for better detection accuracy

## 📊 Performance Metrics

- **FPS**: Real-time performance indicator
- **Detection Accuracy**: Depends on training data quality
- **Memory Usage**: Varies with model size and input resolution

