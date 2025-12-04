"""
FPS Benchmark Test for Hybrid Tracker
"""
import cv2
import time
import numpy as np
from trackers.hybrid_tracker import HybridTracker


def test_fps(video_path, duration=10, device='cuda'):
    """
    Measure tracking FPS on a video
    Args:
        video_path: Path to test video
        duration: Duration in seconds to test
        device: 'cuda' or 'cpu'
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {width}x{height} @ {fps_original:.1f} FPS")
    
    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Cannot read first frame")
        return
    
    # Use center region as initial bbox
    h, w = first_frame.shape[:2]
    bbox = (w//4, h//4, w//2, h//2)
    
    # Initialize tracker
    print(f"🚀 Initializing tracker on {device}...")
    tracker = HybridTracker(device=device)
    tracker.init(first_frame, bbox)
    
    # Track for specified duration
    max_frames = int(fps_original * duration)
    frame_times = []
    confidences = []
    statuses = {'TRACKING': 0, 'TRACKING_KALMAN': 0, 'PREDICTING': 0, 'REDETECTED': 0, 'LOST': 0}
    
    print(f"⏱️ Testing for {duration}s ({max_frames} frames)...")
    
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        bbox, confidence, status = tracker.update(frame)
        elapsed = time.time() - start_time
        
        frame_times.append(elapsed)
        confidences.append(confidence)
        statuses[status] = statuses.get(status, 0) + 1
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            avg_fps = 1.0 / np.mean(frame_times[-30:])
            print(f"  Frame {frame_count}/{max_frames} | FPS: {avg_fps:.1f} | Status: {status} | Conf: {confidence:.2f}")
    
    cap.release()
    
    # Calculate statistics
    frame_times = np.array(frame_times)
    confidences = np.array(confidences)
    
    avg_fps = 1.0 / np.mean(frame_times)
    min_fps = 1.0 / np.max(frame_times)
    max_fps = 1.0 / np.min(frame_times)
    p90_fps = 1.0 / np.percentile(frame_times, 90)
    
    avg_conf = np.mean(confidences)
    
    print("\n" + "="*50)
    print("📊 BENCHMARK RESULTS")
    print("="*50)
    print(f"Device: {device.upper()}")
    print(f"Video Resolution: {width}x{height}")
    print(f"Frames Processed: {frame_count}")
    print(f"\n🚀 FPS Metrics:")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Minimum FPS: {min_fps:.1f}")
    print(f"  Maximum FPS: {max_fps:.1f}")
    print(f"  90th Percentile: {p90_fps:.1f}")
    print(f"\n📈 Tracking Statistics:")
    print(f"  Average Confidence: {avg_conf:.3f}")
    for status, count in sorted(statuses.items()):
        pct = (count / frame_count) * 100
        print(f"  {status}: {count} frames ({pct:.1f}%)")
    print("="*50)
    
    # Performance verdict
    if avg_fps >= 50:
        print("✅ EXCELLENT: Exceeds 50 FPS target")
    elif avg_fps >= 30:
        print("✅ GOOD: Meets minimum 30 FPS requirement")
    else:
        print("⚠️ WARNING: Below 30 FPS target")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_fps.py <video_path> [duration] [device]")
        print("Example: python test_fps.py test.mp4 10 cuda")
        sys.exit(1)
    
    video_path = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    device = sys.argv[3] if len(sys.argv) > 3 else 'cuda'
    
    test_fps(video_path, duration, device)
