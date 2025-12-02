"""
Download sample test videos for benchmarking
"""
import cv2
import numpy as np
from pathlib import Path


def create_synthetic_test_video(output_path, width=1920, height=1080, fps=30, duration=10):
    """
    Create a synthetic test video with a moving object
    Args:
        output_path: Path to save video
        width, height: Video resolution
        fps: Frames per second
        duration: Duration in seconds
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = fps * duration
    
    # Object parameters
    obj_size = 100
    obj_color = (0, 255, 0)  # Green
    
    for i in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving object (circular motion)
        angle = (i / total_frames) * 2 * np.pi * 3  # 3 full circles
        radius = min(width, height) // 3
        center_x = width // 2 + int(radius * np.cos(angle))
        center_y = height // 2 + int(radius * np.sin(angle))
        
        # Draw object (rectangle)
        x1 = center_x - obj_size // 2
        y1 = center_y - obj_size // 2
        x2 = center_x + obj_size // 2
        y2 = center_y + obj_size // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), obj_color, -1)
        
        # Add some noise
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Created synthetic video: {output_path}")


def create_occlusion_test_video(output_path, width=1920, height=1080, fps=30, duration=15):
    """
    Create a test video with 3s occlusion in the middle
    Args:
        output_path: Path to save video
        width, height: Video resolution
        fps: Frames per second
        duration: Duration in seconds
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = fps * duration
    occlusion_start = fps * 6  # Start occlusion at 6s
    occlusion_end = fps * 9    # End occlusion at 9s (3s duration)
    
    obj_size = 100
    obj_color = (255, 0, 0)  # Blue
    occluder_color = (128, 128, 128)  # Gray
    
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving object (horizontal motion)
        progress = i / total_frames
        center_x = int(width * 0.2 + width * 0.6 * progress)
        center_y = height // 2
        
        # Draw object
        x1 = center_x - obj_size // 2
        y1 = center_y - obj_size // 2
        x2 = center_x + obj_size // 2
        y2 = center_y + obj_size // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), obj_color, -1)
        
        # Draw occluder during occlusion period
        if occlusion_start <= i < occlusion_end:
            occ_x1 = width // 3
            occ_y1 = 0
            occ_x2 = 2 * width // 3
            occ_y2 = height
            cv2.rectangle(frame, (occ_x1, occ_y1), (occ_x2, occ_y2), occluder_color, -1)
        
        # Add noise
        noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Created occlusion test video: {output_path}")


def download_test_videos():
    """Create test videos for benchmarking"""
    test_dir = Path("test_videos")
    test_dir.mkdir(exist_ok=True)
    
    print("🎬 Creating test videos...")
    
    # Create synthetic test video (1080p, 10s)
    create_synthetic_test_video(
        str(test_dir / "synthetic_1080p.mp4"),
        width=1920, height=1080, fps=30, duration=10
    )
    
    # Create occlusion test video (1080p, 15s with 3s occlusion)
    create_occlusion_test_video(
        str(test_dir / "occlusion_test.mp4"),
        width=1920, height=1080, fps=30, duration=15
    )
    
    # Create 720p version for faster testing
    create_synthetic_test_video(
        str(test_dir / "synthetic_720p.mp4"),
        width=1280, height=720, fps=30, duration=10
    )
    
    print("\n✅ All test videos created!")
    print(f"📁 Location: {test_dir.absolute()}")


if __name__ == "__main__":
    download_test_videos()
