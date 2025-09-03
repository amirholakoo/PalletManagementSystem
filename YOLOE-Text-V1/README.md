# YOLOE-v2-ONNX: Real-Time Object Detection & Tracking System

This high-performance computer vision application provides real-time object detection and tracking using **YOLOE models** with **ONNX optimization**. The system operates in **detection mode by default**, with the option to **enable advanced object tracking** when needed. Designed for deployment on **Raspberry Pi** with camera modules but works seamlessly on standard computers with video files or network streams.

The application features a **Flask web interface**, **standalone detection scripts**, and **advanced object tracking** capabilities, making it ideal for industrial automation, warehouse monitoring, and smart vision applications.

## 🚀 Key Features

- **Multi-Model Support**:
  - YOLOE-11 models (small, medium, large) with segmentation capabilities
  - Automatic ONNX conversion for 3-5x performance improvement
  - Text prompt-based detection with customizable classes

- **Flexible Deployment Options**:
  - **Web Interface**: Flask-based dashboard with live video streaming
  - **Standalone Detection**: Direct OpenCV display for development (default mode)
  - **Object Tracking**: Advanced multi-object tracking with trajectories (can be enabled)

- **Platform Support**:
  - **Raspberry Pi**: Native `picamera2` integration with hardware optimization
  - **Standard Systems**: OpenCV support for USB cameras, RTSP streams, and video files
  - **GPU Acceleration**: Automatic CUDA detection for enhanced performance

- **Smart Configuration**:
  - Easy parameter tuning through `core/config.py`
  - Dynamic model switching without code changes
  - Configurable inference resolution and quality settings

## 📋 System Requirements

### Hardware Requirements
- **Raspberry Pi**: Pi 4B (4GB+ RAM recommended) or Pi 5
- **Standard PC**: 8GB+ RAM, optional NVIDIA GPU for acceleration
- **Camera**: Raspberry Pi Camera Module v2/v3 or USB camera
- **Storage**: 4GB+ free space for models and dependencies

### Software Requirements
- **Python**: 3.8 or higher
- **Operating System**: Raspberry Pi OS, Ubuntu 20.04+, or Windows 10+
- **Dependencies**: Listed in `requirements.txt`

## 🛠️ Installation & Setup

**Do the installation commands in Home directory**

1. **Clone the AISM Repository:**
    ```bash
    git clone https://github.com/amirholakoo/PalletManagementSystem.git
    ```

2.  **Navigate to the Project Directory:**
    ```bash
    cd PalletManagementSystem/YOLOE-Text-V1
    ```

### On a Standard Computer (for Development)

1.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### On a Raspberry Pi (for Deployment)

Setting up on a Raspberry Pi requires installing system-level dependencies for the camera module.

1.  **Install System Dependencies**:
    These packages are required for `picamera2` to function correctly.
    ```bash
    sudo apt-get update
    sudo apt-get install -y build-essential libcamera-dev python3-libcamera
    ```

2.  **Create the Virtual Environment (with System Access)**:
    This critical step creates a virtual environment that can access the system-level `python3-libcamera` library.
    ```bash
    python3 -m venv --system-site-packages venv
    ```

3.  **Activate the Environment**:
    ```bash
    source venv/bin/activate
    ```

4.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🎯 Model Configuration & Management

### Available Models

The project supports multiple YOLOE-11 models with different performance characteristics:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yoloe-11s-seg.pt` | ~27MB | Fast | Good | Raspberry Pi, Real-time |
| `yoloe-11m-seg.pt` | ~57MB | Medium | Better | Balanced performance |
| `yoloe-11l-seg.pt` | ~68MB | Slow | Best | High accuracy needed |

### Changing Models and Resolution

⚠️ **CRITICAL**: When changing models or resolution, you must update **BOTH** `core/config.py` and `convert_to_onnx.py` with matching values.

#### 1. Update Model Configuration

Edit `core/config.py` to change the model:

```python
# MODEL CONFIGURATION
MODEL_PATH = "yoloe-11s-seg.pt"  # Change to desired model

# Inference resolution (must be multiple of 32)
INFERENCE_WIDTH = 640   # Options: 256, 320, 640, 1024
INFERENCE_HEIGHT = 640  # Must match width for square images
```

#### 2. Camera Resolution Configuration

```python
# CAMERA CONFIGURATION
CAMERA_WIDTH = 640      # Camera capture width
CAMERA_HEIGHT = 480     # Camera capture height
CAMERA_FRAMERATE = 10   # FPS (lower = better quality)
```

#### 3. Performance Tuning

```python
# PERFORMANCE CONFIGURATION
CONFIDENCE_THRESHOLD = 0.25    # Lower = more detections
IOU_THRESHOLD = 0.6           # Lower = more overlapping boxes
FRAME_SKIP_COUNT = 0          # 0 = every frame, 1 = every 2nd frame
JPEG_QUALITY = 85             # Web streaming quality (1-100)
```

### ONNX Conversion for Better Performance

⚠️ **IMPORTANT**: The model type and resolution in `convert_to_onnx.py` **MUST MATCH** the settings in `core/config.py`

1. **Remove Old ONNX File First**:
   ```bash
   # Remove existing ONNX model to avoid conflicts
   rm yoloe-11m-seg.onnx  # Or whatever model you're replacing
   ```

2. **Match Configuration Settings**:
   
   **In `core/config.py`**:
   ```python
   MODEL_PATH = "yoloe-11m-seg.pt"    # Your source model
   INFERENCE_WIDTH = 640              # Resolution setting
   INFERENCE_HEIGHT = 640             # Must match conversion
   ```
   
   **In `convert_to_onnx.py`** (must match above):
   ```python
   MODEL_PATH = "yoloe-11m-seg.pt"    # SAME as config.py
   RESOLUTION = 640                   # SAME as INFERENCE_WIDTH/HEIGHT
   
   # Customize your detection classes
   TEXT_PROMPTS = [
       "bag", "sack", "gunny", 
       "sack with text on it", 
       "bag of starch material",
       "white starch sack with some texts on it"
   ]
   ```

3. **Run Conversion**:
   ```bash
   python convert_to_onnx.py
   ```

4. **Automatic Model Selection**:
   The system automatically uses ONNX models when available:
   - `yoloe-11m-seg.pt` → `yoloe-11m-seg.onnx`
   - Faster inference with pre-baked prompts

#### Complete Model Change Workflow

When switching to a different model or resolution, follow this exact sequence:

```bash
# Step 1: Remove old ONNX file
rm yoloe-11m-seg.onnx  # Remove current ONNX model

# Step 2: Edit core/config.py
# MODEL_PATH = "yoloe-11s-seg.pt"  # New model
# INFERENCE_WIDTH = 320            # New resolution
# INFERENCE_HEIGHT = 320           # Must match width

# Step 3: Edit convert_to_onnx.py (MUST MATCH config.py)
# MODEL_PATH = "yoloe-11s-seg.pt"  # SAME as config
# RESOLUTION = 320                 # SAME as INFERENCE_WIDTH

# Step 4: Convert new model
python convert_to_onnx.py

# Step 5: Restart application
python app.py
```


## 🚀 Running the Application

The system provides **three modes of operation**: web interface (with toggleable tracking), standalone detection (default), and dedicated tracking mode.

### 1. Web Interface (Recommended)

```bash
python app.py
```

**Features:**
- Live video streaming at `http://localhost:5000`
- **Real-time detection with bounding boxes (default mode)**
- **Object tracking with trajectory visualization (can be toggled on/off)**
- Configuration dashboard at `/config`
- API endpoints for programmatic control

**Web Controls:**
- Toggle tracking on/off (detection runs by default)
- Reset tracker state
- View system configuration
- Monitor performance metrics

### 2. Standalone Detection (Default Mode)

```bash
python yoloe_detection.py
```

**Features:**
- Pure object detection with bounding boxes
- No tracking - fastest performance mode
- Direct OpenCV display

**Controls:**
- `q`: Quit application
- `s`: Save current frame
- `c`: Show configuration

### 3. Dedicated Tracking Mode

```bash
python yoloe_tracking.py
```

**Controls:**
- `q`: Quit application
- `s`: Save current frame
- `t`: Toggle trajectory display
- `r`: Reset tracker
- `c`: Show configuration

## 🔧 Configuration Reference

### Core Settings (`core/config.py`)



## 🎨 Customizing Detection Classes

### For PyTorch Models (`.pt` files)

Edit the `TEXT_PROMPT_CLASSES` in these files:

**In `yoloe_detection.py`**:
```python
TEXT_PROMPT_CLASSES = [
    "person", "car", "truck", 
    "forklift", "pallet", "box"
]
```

**In `yoloe_tracking.py`**:
```python
TEXT_PROMPT_CLASSES = [
    "person", "car", "truck", 
    "forklift", "pallet", "box"
]
```

### For ONNX Models (`.onnx` files)

Text prompts must be baked in during conversion:

```python
# In convert_to_onnx.py
TEXT_PROMPTS = [
    "your", "custom", "classes", 
    "with detailed descriptions"
]
```

## 🌐 API Endpoints

The web interface provides REST API endpoints:

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/` | GET | Main video streaming interface |
| `/config` | GET | Configuration dashboard |
| `/api/status` | GET | System status and model info |
| `/api/tracking/toggle` | POST | Enable/disable tracking |
| `/api/tracking/reset` | POST | Reset tracker state |
| `/api/tracking/status` | GET | Current tracking status |
| `/convert_to_onnx` | POST | Trigger ONNX conversion |

## 📊 Performance Optimization

### Resolution vs Performance

| Resolution | Speed | Quality | Recommended For |
|------------|-------|---------|----------------|
| 256x256 | Fastest | Basic | Raspberry Pi  |
| 320x320 | Fast | Good | Raspberry Pi  |
| 640x640 | Medium | High | Standard computers |
| 1024x1024 | Slow | Highest | GPU systems only |

### Optimization Tips

1. **Use ONNX Models**: 3-5x faster than PyTorch
2. **Reduce Inference Resolution**: Lower resolution = higher FPS
3. **Enable Frame Skipping**: Process every 2nd or 3rd frame
4. **GPU Acceleration**: Install CUDA for NVIDIA GPUs
5. **Lower Camera FPS**: Reduces processing load

**3. Performance Issues**
- Use ONNX models for better speed
- Reduce `INFERENCE_WIDTH/HEIGHT` in config
- Enable `FRAME_SKIP_COUNT > 0`
- Lower `CAMERA_FRAMERATE`

**4. Memory Errors**
- Use smaller models (`yoloe-11s-seg.pt`)
- Reduce `QUEUE_SIZE` in config
- Close other applications


