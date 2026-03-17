"""
Oriented Object Detection using YOLO-OBB.
"""

from ultralytics import YOLO
import cv2
import numpy as np

class YOLOOBBDetection:
    def __init__(self, model_path='yolov8n-obb.pt'):
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Detect objects in the image.
        Args:
            image: numpy array or path to image
        Returns:
            results: detection results
        """
        results = self.model(image)
        return results

    def get_bboxes(self, results):
        """
        Extract bounding boxes from results.
        """
        # Assuming results[0] for single image
        result = results[0]
        boxes = result.obb  # oriented bounding boxes
        return boxes