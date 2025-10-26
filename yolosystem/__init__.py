"""
YOLOsystem - 去雾和目标检测系统
Dehazing and Object Detection System
"""

__version__ = "1.0.0"
__author__ = "YOLOsystem Team"

from .dehazing import DehazingModule
from .detection import YOLODetector
from .pipeline import DehazingDetectionPipeline

__all__ = [
    "DehazingModule",
    "YOLODetector", 
    "DehazingDetectionPipeline"
]
