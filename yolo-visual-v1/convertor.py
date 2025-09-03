from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import numpy as np

# ========== CONFIGURATION ==========
# Add as many objects as you want:
training_data = [
     
    {
        "image": "image.jpg",
        "box": [278, 44, 554, 287],
        "box": [47, 38, 270, 289],
        "box": [347, 265, 663, 480],
        "box": [24, 270, 337, 471],
        "box": [347, 456, 686, 682],
        "box": [32, 470, 349, 685],

    }
    
]
# ===================================

model = YOLOE("yoloe-11L-seg.pt")

# Collect all bboxes and class IDs
all_bboxes = []
all_class_ids = []

for i, data in enumerate(training_data):
    all_bboxes.append(data["box"])
    all_class_ids.append(i)  # Class ID will be 0, 1, 2, etc.

visual_prompts = {
    'bboxes': np.array(all_bboxes),
    'cls': np.array(all_class_ids)
}

# Train with all objects at once
model.predict(
    training_data[0]["image"],
    refer_image=training_data[0]["image"], 
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    conf=0.1
)

model.export(format="onnx", imgsz=640)

print("Training complete!")
print("Object mapping:")
for i, data in enumerate(training_data):
    print(f"  ID {i}: {data['image']}")
