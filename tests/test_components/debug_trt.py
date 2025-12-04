#!/usr/bin/env python3
"""Debug TRT tracker outputs"""
import sys
import os
sys.path.append('/mnt/f/desktop/CV_Project/CV_TRACKING_ADVANCED')
os.chdir('/mnt/f/desktop/CV_Project/CV_TRACKING_ADVANCED')

from trackers.ostrack_trt_tracker import OSTrackFerrariTRT
import cv2
import numpy as np

print("Initializing TRT Tracker...")
tracker = OSTrackFerrariTRT()

# Load test video
video_path = "/mnt/f/desktop/CV_Project/CV_TRACKING_ADVANCED/test_videos/1_1.mp4"
cap = cv2.VideoCapture(video_path)

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    sys.exit(1)

# Initialize
bbox = (282, 260, 52, 23)
tracker.init(frame, bbox)
print(f"Initialized with bbox: {bbox}")

# Process one frame
ret, frame = cap.read()
if ret:
    bbox, conf, status = tracker.update(frame)
    print(f"\nFrame 1 Results:")
    print(f"  BBox: {[int(v) for v in bbox]}")
    print(f"  Confidence: {conf:.4f}")
    print(f"  Status: {status}")
    
    # Access internal state to debug
    print(f"\n  Template size: {tracker.template_size}")
    print(f"  Search size: {tracker.search_size}")
    print(f"  Feature size: {tracker.feat_sz}")
    print(f"  Output window shape: {tracker.output_window.shape}")
    print(f"  Output window min/max: {tracker.output_window.min():.4f} / {tracker.output_window.max():.4f}")

cap.release()
