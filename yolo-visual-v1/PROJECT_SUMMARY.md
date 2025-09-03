# YOLOE Visual Prompt Project Summary

## 🎯 Project Purpose
This project implements a complete object detection system using YOLOE (You Only Look Once Efficient) with visual prompts, designed specifically for Raspberry Pi with camera integration.

## 🔄 Complete Workflow

### 1. **Image Capture** (`imagecapture.py`)
- **Purpose**: Capture reference photos using Raspberry Pi camera
- **Input**: Camera feed
- **Output**: `image.jpg` (reference image)
- **Controls**: SPACE to capture, ESC to exit

### 2. **Bounding Box Annotation** (`image-drow-box.py`)
- **Purpose**: Interactive tool for drawing bounding boxes around objects
- **Input**: Captured image
- **Output**: Annotated image with bounding box coordinates
- **Controls**: Click and drag to draw boxes, R to reset, Q to quit

### 3. **Model Training** (`convertor.py`)
- **Purpose**: Train YOLOE model using visual prompts and annotations
- **Input**: Annotated image + bounding boxes
- **Output**: `yoloe-11l-seg.onnx` (trained model)
- **Process**: Visual prompt training with ultralytics

### 4. **Real-time Detection** (`yolorunmodel.py`)
- **Purpose**: Run real-time object detection on camera feed
- **Input**: Camera feed + trained model
- **Output**: Live detection results with bounding boxes
- **Controls**: Q to quit

### 5. **Object Tracking** (`yoloe-tracker.py`)
- **Purpose**: Advanced tracking with persistent object IDs
- **Input**: Camera feed + trained model
- **Output**: Live tracking with object persistence
- **Controls**: Q to quit, S to save frame

## 🛠️ Key Technologies

- **YOLOE Model**: Efficient object detection with visual prompts
- **Ultralytics**: YOLO framework implementation
- **OpenCV**: Computer vision and camera handling
- **Picamera2**: Raspberry Pi camera interface
- **PyTorch**: Deep learning backend
- **ONNX**: Model export format for deployment

## 📱 Hardware Requirements

- **Raspberry Pi** (3B+ or 4B recommended)
- **Camera Module** (Pi Camera v2 or v3)
- **Adequate Storage** (for model files and images)
- **Good Lighting** (for optimal detection accuracy)

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete workflow
python imagecapture.py      # Capture image
python image-drow-box.py    # Annotate boxes
python convertor.py         # Train model
python yolorunmodel.py      # Run detection
```

## 📊 Performance Characteristics

- **Resolution**: 800x800 pixels (configurable)
- **FPS**: Varies by hardware (typically 10-30 FPS)
- **Model Size**: YOLOE-11L (large) or YOLOE-11S (small)
- **Accuracy**: Depends on training data quality and lighting

## 🔧 Customization Options

- **Camera Resolution**: Adjustable in each script
- **Model Selection**: Choose between large (accurate) or small (fast) models
- **Detection Confidence**: Configurable threshold values
- **Input Size**: Model input dimensions can be modified

## 📁 File Organization

```
yoloe-visual2/
├── Core Scripts/           # Main workflow components
├── Models/                 # Pre-trained and trained models
├── Images/                 # Sample and captured images
├── Documentation/          # README, setup guides
└── Installation/           # Setup scripts for different platforms
```

## 🎯 Use Cases

- **Industrial Object Detection**: Manufacturing quality control
- **Security Monitoring**: Object presence detection
- **Research Projects**: Computer vision experimentation
- **Educational Purposes**: Learning object detection concepts
- **IoT Applications**: Smart camera systems

## 🔍 Key Features

- **Visual Prompt Learning**: Train without traditional datasets
- **Real-time Processing**: Live camera feed analysis
- **Object Tracking**: Persistent identification across frames
- **Easy Annotation**: Interactive bounding box tool
- **Model Export**: ONNX format for deployment
- **Cross-platform**: Works on Raspberry Pi and other systems

## 📈 Future Enhancements

- **Multi-class Detection**: Support for multiple object types
- **Cloud Integration**: Remote model training and deployment
- **Mobile App**: Remote monitoring and control
- **Advanced Analytics**: Detection statistics and reporting
- **Edge AI Optimization**: Better performance on limited hardware
