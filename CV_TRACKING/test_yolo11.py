"""
Quick test to verify YOLO11 integration
"""
import sys
sys.path.insert(0, r'f:\desktop\CV_TRACKING')

from trackers.redetector import ReDetector
import cv2

# Test YOLO11 loading
print("="*50)
print("Testing YOLO11 Integration")
print("="*50)

print("\n1. Testing ReDetector initialization...")
detector = ReDetector(device='cuda')

print("\n2. Testing detection on a dummy frame...")
import numpy as np
# Create a simple test frame with colored rectangle
frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(frame, (200, 150), (400, 350), (0, 255, 0), -1)

detections = detector.detect(frame, conf_threshold=0.1)
print(f"   Detections found: {len(detections)}")

if len(detections) > 0:
    for i, det in enumerate(detections):
        x, y, w, h, conf, cls = det
        print(f"   Detection {i+1}: bbox=({x},{y},{w},{h}), conf={conf:.2f}, class={cls}")

print("\n✅ YOLO11 integration test completed!")
print("="*50)
