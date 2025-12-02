"""
Gradio UI for Zero-shot Object Tracking
"""
import gradio as gr
import cv2
import numpy as np
import time
import os
from pathlib import Path
from trackers.hybrid_tracker import HybridTracker


# Global tracker instance
tracker = None
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Global state for ROI selection
roi_state = {
    'frame': None,
    'points': [],
    'bbox': None
}


def process_video(video_path, x1, y1, x2, y2):
    """
    Main tracking pipeline
    Args:
        video_path: Path to input video
        x1, y1, x2, y2: Bounding box coordinates
    Returns:
        output_video_path: Path to tracked video
        stats: Tracking statistics
    """
    global tracker
    
    # 验证输入
    if video_path is None:
        return None, "❌ 错误：请先上传视频！"
    
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None, "❌ 错误：请先选择目标区域！"
    
    try:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    except:
        return None, "❌ 错误：边界框坐标格式无效！"
    
    # 确保有效的边界框
    if x2 <= x1 or y2 <= y1:
        return None, "❌ 错误：无效的边界框！x2 必须大于 x1，y2 必须大于 y1"
    
    init_bbox = (x1, y1, x2-x1, y2-y1)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ 错误：无法打开视频文件！"
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video writer
    timestamp = int(time.time())
    output_path = OUTPUT_DIR / f"tracked_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Initialize tracker with first frame
    ret, first_frame = cap.read()
    if not ret:
        return None, "❌ Error: Cannot read first frame!"
    
    tracker = HybridTracker(device='cuda')
    tracker.init(first_frame, init_bbox)
    
    # Draw initial bbox on first frame
    vis_frame = first_frame.copy()
    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis_frame, "Initial Target", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out.write(vis_frame)
    
    # Tracking statistics
    stats = {
        'total_frames': total_frames,
        'tracked': 0,
        'lost': 0,
        'redetected': 0,
        'avg_fps': 0,
        'avg_confidence': 0
    }
    
    frame_idx = 1
    confidences = []
    start_time = time.time()
    
    print(f"🎯 Starting tracking with bbox: {init_bbox}")
    
    # Process remaining frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track object
        bbox, confidence, status = tracker.update(frame)
        confidences.append(confidence)
        
        # Update statistics
        if status == "REDETECTED":
            stats['redetected'] += 1
        elif status == "LOST":
            stats['lost'] += 1
        else:
            stats['tracked'] += 1
        
        # Visualize (only show bbox if confidence > 0.15)
        vis_frame = frame.copy()
        if bbox is not None and confidence > 0.15:
            x, y, w, h = [int(v) for v in bbox]
            
            # Color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green
            elif confidence > 0.4:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Draw bbox
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, 2)
            
            # Status text
            text = f"{status} | Conf: {confidence:.2f}"
            cv2.putText(vis_frame, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Frame counter (always show)
        cv2.putText(vis_frame, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(vis_frame)
        frame_idx += 1
        
        # Progress indicator
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Processing: {progress:.1f}% | Status: {status} | Conf: {confidence:.2f}")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Calculate final statistics
    elapsed_time = time.time() - start_time
    stats['avg_fps'] = total_frames / elapsed_time if elapsed_time > 0 else 0
    stats['avg_confidence'] = np.mean(confidences) if confidences else 0
    
    stats_text = f"""
    ✅ Tracking Complete!
    
    📊 Statistics:
    - Total Frames: {stats['total_frames']}
    - Successfully Tracked: {stats['tracked']}
    - Re-detections: {stats['redetected']}
    - Lost Frames: {stats['lost']}
    - Average FPS: {stats['avg_fps']:.1f}
    - Average Confidence: {stats['avg_confidence']:.3f}
    
    🎥 Output: {output_path.name}
    """
    
    return str(output_path), stats_text


def load_first_frame(video_path):
    """
    Extract first frame for ROI selection
    Args:
        video_path: Path to video
    Returns:
        rgb_frame: First frame in RGB format
        width, height: Frame dimensions
    """
    global roi_state
    
    if video_path is None:
        roi_state['frame'] = None
        roi_state['points'] = []
        roi_state['bbox'] = None
        return None, None, None, None, None, None, None
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Store frame globally
        roi_state['frame'] = frame
        roi_state['points'] = []
        roi_state['bbox'] = None
        
        # Convert BGR to RGB for Gradio
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # Initialize bbox to center quarter of the frame
        default_x1 = width // 4
        default_y1 = height // 4
        default_x2 = 3 * width // 4
        default_y2 = 3 * height // 4
        
        # Draw initial bbox on frame
        preview = rgb_frame.copy()
        cv2.rectangle(preview, (default_x1, default_y1), (default_x2, default_y2), (0, 255, 0), 2)
        cv2.putText(preview, "Adjust bbox below or click two points", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return preview, default_x1, default_y1, default_x2, default_y2, width, height
    
    return None, None, None, None, None, None, None


def update_bbox_preview(video_path, x1, y1, x2, y2):
    """
    Update bbox preview when coordinates change
    """
    global roi_state
    
    if roi_state['frame'] is None:
        return None
    
    # Convert to RGB
    preview = cv2.cvtColor(roi_state['frame'], cv2.COLOR_BGR2RGB).copy()
    
    # Draw bbox if valid
    try:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(preview, f"ROI: ({x1},{y1}) to ({x2},{y2})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            w, h = x2 - x1, y2 - y1
            cv2.putText(preview, f"Size: {w}x{h}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    except:
        pass
    
    return preview


def handle_image_click(video_path, x1, y1, x2, y2, evt: gr.SelectData):
    """
    Handle click on image to set bbox corners
    """
    global roi_state
    
    if roi_state['frame'] is None:
        return x1, y1, x2, y2
    
    click_x, click_y = evt.index[0], evt.index[1]
    roi_state['points'].append((click_x, click_y))
    
    print(f"🖱️ Click {len(roi_state['points'])}: ({click_x}, {click_y})")
    
    # After two clicks, set bbox
    if len(roi_state['points']) >= 2:
        p1, p2 = roi_state['points'][-2], roi_state['points'][-1]
        new_x1 = min(p1[0], p2[0])
        new_y1 = min(p1[1], p2[1])
        new_x2 = max(p1[0], p2[0])
        new_y2 = max(p1[1], p2[1])
        
        print(f"✓ BBox set: ({new_x1},{new_y1}) to ({new_x2},{new_y2})")
        
        return new_x1, new_y1, new_x2, new_y2
    
    return x1, y1, x2, y2


# Gradio Interface
with gr.Blocks(title="🎯 Zero-shot Object Tracker", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 Zero-shot Object Tracking System
    
    **Features:**
    - ✨ Track any object (zero-shot)
    - 🚀 Real-time performance (30-50 FPS)
    - 🔄 Occlusion recovery (3s+ handling)
    - 🎨 GPU accelerated (CUDA)
    
    **Instructions:**
    1. Upload a video
    2. **Method 1**: Click **two points** on the frame to define bounding box (top-left, bottom-right)
    3. **Method 2**: Adjust the coordinate sliders below the image
    4. Click "Start Tracking"
    5. Download the tracked video
    """)
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="📤 Upload Video")
            
            frame_display = gr.Image(
                label="🎯 Click TWO points to select ROI (or use sliders below)",
                interactive=False,
                type="numpy"
            )
            
            with gr.Row():
                x1_slider = gr.Number(label="X1 (left)", value=0, precision=0)
                y1_slider = gr.Number(label="Y1 (top)", value=0, precision=0)
            
            with gr.Row():
                x2_slider = gr.Number(label="X2 (right)", value=100, precision=0)
                y2_slider = gr.Number(label="Y2 (bottom)", value=100, precision=0)
            
            # Hidden dimensions
            frame_width = gr.State(value=None)
            frame_height = gr.State(value=None)
            
            track_btn = gr.Button("🚀 Start Tracking", variant="primary", size="lg")
        
        with gr.Column():
            output_video = gr.Video(label="📥 Tracked Video")
            stats_output = gr.Textbox(label="📊 Tracking Statistics", lines=12)
    
    # Event handlers
    video_input.change(
        fn=load_first_frame,
        inputs=[video_input],
        outputs=[frame_display, x1_slider, y1_slider, x2_slider, y2_slider, frame_width, frame_height]
    )
    
    # Update preview when sliders change
    for slider in [x1_slider, y1_slider, x2_slider, y2_slider]:
        slider.change(
            fn=update_bbox_preview,
            inputs=[video_input, x1_slider, y1_slider, x2_slider, y2_slider],
            outputs=[frame_display]
        )
    
    # Handle image clicks
    frame_display.select(
        fn=handle_image_click,
        inputs=[video_input, x1_slider, y1_slider, x2_slider, y2_slider],
        outputs=[x1_slider, y1_slider, x2_slider, y2_slider]
    )
    
    # Start tracking
    track_btn.click(
        fn=process_video,
        inputs=[video_input, x1_slider, y1_slider, x2_slider, y2_slider],
        outputs=[output_video, stats_output]
    )
    
    gr.Markdown("""
    ---
    ### 🔧 Technical Stack:
    - **Main Tracker**: OpenCV CSRT (Channel and Spatial Reliability Tracker)
    - **Motion Prediction**: Kalman Filter (8-state constant velocity model)
    - **Re-detection**: YOLO11n with CUDA acceleration (local model)
    - **Re-ID**: Multi-feature matching (color histogram + template + spatial proximity)
    
    ### 📈 Performance Optimization:
    - Layer 1: CSRT for frame-to-frame tracking (fastest)
    - Layer 2: Kalman prediction during low confidence
    - Layer 3: Trigger-based YOLO11 re-detection (only when needed)
    - Adaptive confidence thresholds for intelligent switching
    """)


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7999)
