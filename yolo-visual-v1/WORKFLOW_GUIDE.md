# 📖 Complete Workflow Guide for New Images

This guide covers the complete process of capturing new images, annotating them, training the YOLOE model, and running detection and tracking.

## 🎯 Overview

The complete workflow consists of 5 main steps:
1. **Image Capture** - Take reference photos
2. **Bounding Box Annotation** - Draw boxes around objects
3. **Model Training** - Train YOLOE with visual prompts
4. **Real-time Detection** - Run detection on camera feed
5. **Object Tracking** - Advanced tracking (optional)

## 🚀 Step-by-Step Workflow

### Step 1: Image Capture (`imagecapture.py`)

**Purpose**: Capture reference photos using Raspberry Pi camera

**How to use**:
```bash
python imagecapture.py
```

**Instructions**:
- Position your objects in the camera view
- Ensure good lighting for clear images
- Press **SPACE** to capture the image
- Press **ESC** to exit
- Image is saved as `image.jpg`

**Tips**:
- Use high-contrast objects for better detection
- Ensure objects are clearly visible and not overlapping
- Capture from the same angle you'll use for detection

---

### Step 2: Bounding Box Annotation (`image-drow-box.py`)

**Purpose**: Draw bounding boxes around objects in the captured image

**How to use**:
```bash
python image-drow-box.py
```

**Instructions**:
- **Click and drag** to draw boxes around objects
- Each box gets a unique Class ID (0, 1, 2...)
- **Press 'R'** to reset all boxes if needed
- **Press 'Q'** to save and exit

**Annotation Process**:
1. Load the captured image
2. Draw boxes around each object you want to detect
3. Each box automatically gets a sequential class ID
4. The tool saves the bounding box coordinates

**Important Notes**:
- Draw boxes tightly around objects
- Include the entire object in each box
- Avoid overlapping boxes
- Each object should have its own box

---

### Step 3: Model Training (`convertor.py`)

**Purpose**: Train YOLOE model using visual prompts and annotations

**How to use**:
```bash
python convertor.py
```

**What happens**:
1. Loads the annotated image (`image.jpg`)
2. Reads the bounding box coordinates
3. Trains YOLOE model using visual prompts
4. Exports trained model to ONNX format
5. Creates `yoloe-11L-seg.onnx` for deployment

**Training Data Structure**:
The script uses this format for training:
```python
training_data = [
    {
        "image": "image.jpg",
        "box": [x1, y1, x2, y2],  # Bounding box coordinates
        "box": [x1, y1, x2, y2],  # Additional boxes...
    }
]
```

**Training Process**:
- Uses YOLOE-11L pre-trained model
- Applies visual prompt learning
- Trains with your specific objects
- Exports to ONNX format for deployment

**Expected Output**:
- `yoloe-11L-seg.onnx` - Trained model file
- Training completion message with object mapping

---

### Step 4: Real-time Detection (`yolorunmodel.py`)

**Purpose**: Run real-time object detection using the trained model

**How to use**:
```bash
python yolorunmodel.py
```

**What happens**:
1. Loads the trained ONNX model
2. Activates camera feed
3. Runs real-time detection on each frame
4. Displays results with bounding boxes
5. Shows FPS performance metrics

**Controls**:
- **Press 'q'** to quit the detection

**Display Information**:
- Live camera feed with detection boxes
- FPS counter in top-right corner
- Bounding boxes around detected objects
- Confidence scores for detections

---

### Step 5: Object Tracking (Optional) (`yoloe-tracker.py`)

**Purpose**: Advanced tracking with persistent object IDs

**How to use**:
```bash
python yoloe-tracker.py
```

**Features**:
- Persistent object tracking across frames
- Unique ID assignment for each object
- Tracking statistics and FPS display
- Frame saving capability

**Controls**:
- **Press 'q'** to quit tracking
- **Press 's'** to save current frame

**Tracking Information**:
- Object IDs that persist across frames
- Total detection count per object
- Frame-by-frame tracking statistics

---

## 🔧 Configuration and Customization

### Camera Settings
All scripts use these default camera settings:
- **Resolution**: 800x800 pixels
- **Format**: RGB888
- **Alignment**: Enabled for optimal performance

**To modify camera settings**:
Edit the camera configuration in each script:
```python
picam2.preview_configuration.main.size = (800, 800)
picam2.preview_configuration.main.format = "RGB888"
```

### Model Settings
- **Default Model**: YOLOE-11L (large, accurate)
- **Alternative**: YOLOE-11S (small, faster)
- **Input Size**: 640x640 pixels (configurable)

**To use different model**:
Modify the model loading line in scripts:
```python
# For large model (accurate)
model = YOLO("yoloe-11L-seg.onnx")

# For small model (faster)
model = YOLO("yoloe-11S-seg.onnx")
```

---

## 📊 Performance Optimization

### For Better Detection Accuracy:
- Ensure good lighting conditions
- Use high-contrast objects
- Avoid shadows and reflections
- Position objects clearly in frame

### For Better Performance:
- Use YOLOE-11S for faster inference
- Reduce camera resolution if needed
- Close unnecessary applications
- Ensure adequate cooling on Raspberry Pi

### For Better Training:
- Use clear, high-quality reference images
- Draw precise bounding boxes
- Include all objects you want to detect
- Avoid overlapping annotations

---

## 🚨 Troubleshooting

### Common Issues and Solutions:

**1. Camera Not Detected**
```
Error: Camera not accessible
Solution: Check camera connection and enable camera in Raspberry Pi settings
```

**2. Model Training Fails**
```
Error: Out of memory or training fails
Solution: Check available RAM and storage space, close other applications
```

**3. Low FPS**
```
Problem: Detection running slowly
Solution: Use YOLOE-11S model, reduce camera resolution, optimize lighting
```

**4. Poor Detection Accuracy**
```
Problem: Objects not detected properly
Solution: Improve lighting, use clearer objects, retrain with better images
```

**5. Bounding Box Tool Not Working**
```
Problem: Can't draw boxes or tool crashes
Solution: Ensure image.jpg exists, check image format, restart the tool
```

---

## 📋 Workflow Checklist

Before starting, ensure you have:
- [ ] Raspberry Pi with camera module
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Good lighting conditions
- [ ] Objects to detect ready

Complete workflow checklist:
- [ ] **Step 1**: Capture reference image (`imagecapture.py`)
- [ ] **Step 2**: Annotate bounding boxes (`image-drow-box.py`)
- [ ] **Step 3**: Train the model (`convertor.py`)
- [ ] **Step 4**: Run detection (`yolorunmodel.py`)
- [ ] **Step 5**: Run tracking (optional) (`yoloe-tracker.py`)

---

## 💡 Pro Tips

1. **Start Simple**: Begin with 1-2 objects before adding more
2. **Lighting Matters**: Good lighting significantly improves detection accuracy
3. **Box Precision**: Draw tight, accurate bounding boxes for better training
4. **Model Selection**: Use YOLOE-11S for speed, YOLOE-11L for accuracy
5. **Regular Retraining**: Retrain models if detection accuracy decreases
6. **Backup Models**: Keep copies of your trained models
7. **Test Incrementally**: Test each step before moving to the next

---

## 🔗 Related Documentation

- **[Main README](README.md)** - Project overview and quick start
- **[Requirements](requirements.txt)** - Python dependencies
- **[Project Summary](PROJECT_SUMMARY.md)** - Technical specifications

---

**Happy detecting! 🎯**
