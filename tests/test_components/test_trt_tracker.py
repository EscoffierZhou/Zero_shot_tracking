#!/usr/bin/env python3
import sys
sys.path.append('CV_TRACKING_ADVANCED')

from trackers.ostrack_trt_tracker import OSTrackFerrariTRT
import cv2

print("Testing TensorRT Tracker...")

# Initialize tracker
tracker = OSTrackFerrariTRT()
print("✓ Tracker initialized successfully!")

# Load test video
video_path = "F:/desktop/CV_Project/CV_TRACKING_ADVANCED/test_videos/1_1.mp4"
cap = cv2.VideoCapture(video_path)

# Read first frame
ret, frame = cap.read()
if not ret:
    print("❌ Failed to read video")
    sys.exit(1)

print(f"✓ Video loaded: {frame.shape}")

# Initialize with bbox (278,259) to (338,280)
bbox = (278, 259, 338-278, 280-259)  # (x, y, w, h)
tracker.init(frame, bbox)
print(f"✓ Tracker initialized with bbox: {bbox}")

# Track a few frames
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    
    result_bbox, confidence, status = tracker.update(frame)
    print(f"Frame {i+1}: bbox={[int(v) for v in result_bbox]}, status={status}")

cap.release()
print("\n✅ Test completed successfully!")
