"""
YOLOsystem - 去雾和目标检测系统
Dehazing and Object Detection System
"""

__version__ = "1.0.0"
__author__ = "YOLOsystem Team"

from .dehazing import DehazingModule
from .detection import YOLODetector, MultiModelDetector
from .pipeline import DehazingDetectionPipeline, MultiModelDetectionPipeline, FusionDetectionPipeline
from .fusion import FusionDetector, ImageQualityAssessment

__all__ = [
    "DehazingModule",
    "YOLODetector",
    "MultiModelDetector",
    "DehazingDetectionPipeline",
    "MultiModelDetectionPipeline",
    "FusionDetectionPipeline",
    "FusionDetector",
    "ImageQualityAssessment"
]
