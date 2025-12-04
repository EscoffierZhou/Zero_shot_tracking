#!/usr/bin/env python3
"""Test TRT tracker initialization only"""
import sys
sys.path.append('/mnt/f/desktop/CV_Project/CV_TRACKING_ADVANCED')

from trackers.ostrack_trt_tracker import OSTrackFerrariTRT

print("Testing TensorRT Tracker initialization...")

# Initialize tracker
tracker = OSTrackFerrariTRT()
print("✓ Tracker initialized successfully!")

print("\n✅ All systems go! The TensorRT tracker is ready to use.")
print("\nNext: Launch the Gradio app to test with real video:")
print("  Double-click: CV_TRACKING_ADVANCED/run_trt.bat")
