import cv2
import numpy as np

# ========== CONFIGURATION ==========
# Set your image path here:
image_path = "image.jpg"
# ===================================

class BoundingBoxTool:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"❌ Could not load image: {image_path}")
        
        self.original_image = self.image.copy()
        self.boxes = []
        self.current_box = []
        self.drawing = False
        self.class_id = 0
        self.mouse_x = 0
        self.mouse_y = 0
        
        cv2.namedWindow("Bounding Box Tool", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Bounding Box Tool", self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [x, y]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_image = self.image.copy()
                cv2.rectangle(temp_image, (self.current_box[0], self.current_box[1]), 
                            (x, y), (0, 255, 0), 2)
                self.draw_crosshairs(temp_image, x, y)
                cv2.imshow("Bounding Box Tool", temp_image)
            else:
                temp_image = self.image.copy()
                self.draw_crosshairs(temp_image, x, y)
                cv2.imshow("Bounding Box Tool", temp_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_box.extend([x, y])
            
            # Create bounding box [x1, y1, x2, y2]
            box = [
                min(self.current_box[0], self.current_box[2]),
                min(self.current_box[1], self.current_box[3]),
                max(self.current_box[0], self.current_box[2]),
                max(self.current_box[1], self.current_box[3])
            ]
            
            self.boxes.append({
                'bbox': box,
                'class_id': self.class_id
            })
            
            print(f"📦 Box {len(self.boxes)}: {box} (Class ID: {self.class_id})")
            self.class_id += 1
            self.update_display()
    
    def draw_crosshairs(self, image, x, y):
        height, width = image.shape[:2]
        cv2.line(image, (x, 0), (x, height), (255, 255, 0), 1)
        cv2.line(image, (0, y), (width, y), (255, 255, 0), 1)
        cv2.putText(image, f"({x}, {y})", (x + 10, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
    def update_display(self):
        self.image = self.original_image.copy()
        
        # Draw all boxes
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, box_info in enumerate(self.boxes):
            bbox = box_info['bbox']
            class_id = box_info['class_id']
            color = colors[class_id % len(colors)]
            
            cv2.rectangle(self.image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(self.image, f"ID:{class_id}", (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add instructions
        cv2.putText(self.image, "R: Reset | Q: Quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.image, f"Next Class ID: {self.class_id}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if hasattr(self, 'mouse_x') and hasattr(self, 'mouse_y'):
            self.draw_crosshairs(self.image, self.mouse_x, self.mouse_y)
        
        cv2.imshow("Bounding Box Tool", self.image)
        
    def run(self):
        print(f"🎯 Bounding Box Tool - Loaded: {self.image_path}")
        print("📋 Instructions:")
        print("   • Click and drag to draw boxes around objects")
        print("   • Each box gets a unique Class ID (0, 1, 2...)")
        print("   • Press 'R' to reset all boxes")
        print("   • Press 'Q' to quit")
        print()
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.boxes = []
                self.class_id = 0
                self.update_display()
                print("🔄 Reset all boxes")
        
        cv2.destroyAllWindows()

# Check if image exists
import os
if not os.path.exists(image_path):
    print(f"❌ Image not found: {image_path}")
    print("📁 Available images in current directory:")
    for file in os.listdir('.'):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"   {file}")
    exit()

# Run the tool
tool = BoundingBoxTool(image_path)
tool.run()
