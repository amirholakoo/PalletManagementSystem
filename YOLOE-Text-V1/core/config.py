"""
YOLOE Smart Eye Configuration
Easy-to-modify Python configuration class for all project parameters.
"""

class Config:
    """Configuration class with all adjustable parameters"""
    
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    
    CAMERA_FRAMERATE = 10

    TUNING_FILE = "imx219_noir.json"
    
    RAW_WIDTH = 1640
    RAW_HEIGHT = 1232
    

    MODEL_PATH = "yoloe-11m-seg.pt"
    
    INFERENCE_WIDTH = 640
    INFERENCE_HEIGHT = 640
    
    # Detection thresholds
    CONFIDENCE_THRESHOLD = 0.25    
    IOU_THRESHOLD = 0.6          
    
    FRAME_SKIP_COUNT = 0        
    QUEUE_SIZE = 2               
    
    JPEG_QUALITY = 85            
    
    # Threading settings
    CAPTURE_THREAD_TIMEOUT = 2  
    PROCESS_THREAD_TIMEOUT = 2   
    
    CAMERA_WARMUP_TIME = 2      
    MODEL_WARMUP = True         
    
    # ============================================================================
    # DISPLAY CONFIGURATION
    # ============================================================================
    
    # FPS counter settings
    SHOW_FPS = True            
    FPS_FONT_SCALE = 0.8        
    FPS_COLOR_R = 0            
    FPS_COLOR_G = 255           
    FPS_COLOR_B = 0             
    
    # Detection box settings
    SHOW_LABELS = True          
    SHOW_BOXES = True         
    SHOW_MASKS = False          
    
    # ============================================================================
    # WEB SERVER CONFIGURATION
    # ============================================================================
    
    HOST = "0.0.0.0"            
    PORT = 5000                
    DEBUG = False               
    
    # ============================================================================
    # CALCULATED PROPERTIES
    # ============================================================================
    
    @property
    def FRAME_DURATION(self):
        """Calculate frame duration for camera"""
        return int(1000000 / self.CAMERA_FRAMERATE)
    
    @property
    def FPS_COLOR(self):
        """Get FPS color in BGR format for OpenCV"""
        return (self.FPS_COLOR_B, self.FPS_COLOR_G, self.FPS_COLOR_R)
    
    def print_config(self):
        """Print current configuration for debugging"""
        print("=== YOLOE Configuration ===")
        print(f"Camera: {self.CAMERA_WIDTH}x{self.CAMERA_HEIGHT} @ {self.CAMERA_FRAMERATE}FPS")
        print(f"Model: {self.MODEL_PATH}")
        print(f"Inference: {self.INFERENCE_WIDTH}x{self.INFERENCE_HEIGHT}")
        print(f"Performance: Skip={self.FRAME_SKIP_COUNT}, Quality={self.JPEG_QUALITY}%")
        print("===========================")

# Global configuration instance
config = Config()

if __name__ == "__main__":
    # Test configuration
    config.print_config() 
