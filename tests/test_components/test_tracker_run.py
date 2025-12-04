import sys
import os
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'CV_TRACKING_ADVANCED'))

from CV_TRACKING_ADVANCED.trackers.ostrack_tracker import OSTrackFerrari

def test_tracker():
    video_path = 'test_video.mp4'
    if not os.path.exists(video_path):
        print("Video not found!")
        return

    # Initial bbox for the bouncing box video (center)
    # create_bouncing_box_video uses width=640, height=480, box_size=50
    # start x,y = 320, 240
    init_bbox = (320, 240, 50, 50)
    
    print("Initializing tracker...")
    try:
        tracker = OSTrackFerrari()
    except Exception as e:
        print(f"❌ Failed to instantiate tracker: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    print("Calling tracker.init()...")
    try:
        tracker.init(frame, init_bbox)
        print("✅ Init successful")
    except Exception as e:
        print(f"❌ Init failed: {e}")
        return

    print("Starting tracking loop...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            bbox, score, status = tracker.update(frame)
            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx}: {status} Score: {score:.2f} BBox: {bbox}")
        except Exception as e:
            print(f"❌ Update failed at frame {frame_idx}: {e}")
            break
            
        frame_idx += 1
        
    print("Test complete.")

if __name__ == "__main__":
    test_tracker()
