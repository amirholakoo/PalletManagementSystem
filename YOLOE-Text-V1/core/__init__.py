"""
YOLOE Smart Eye - Core Module
Modular object detection system with configurable parameters.
"""

from .config import config
from .yoloe_model import YOLOEModel
from .camera import Camera

__version__ = "1.0.0"
__author__ = "YOLOE Smart Eye Project"

# Print configuration on import for debugging
if __name__ != "__main__":
    config.print_config() 