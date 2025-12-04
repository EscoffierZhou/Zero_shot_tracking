"""
零样本目标跟踪 Gradio 界面
"""
import gradio as gr
import cv2
import numpy as np
import time
import os
from pathlib import Path
from trackers.hybrid_tracker import HybridTracker


# 全局跟踪器实例
tracker = None
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ROI 选择的全局状态
roi_state = {
    'frame': None,
    'points': [],
    'bbox': None
}


def process_video(video_path, x1, y1, x2, y2):
    """
    主跟踪流程
    参数:
        video_path: 输入视频的路径
        x1, y1, x2, y2: 边界框坐标
    返回:
        output_video_path: 跟踪后的视频路径
        stats: 跟踪统计信息
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
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 输出视频写入器
    timestamp = int(time.time())
    output_path = OUTPUT_DIR / f"tracked_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # 使用第一帧初始化跟踪器
    ret, first_frame = cap.read()
    if not ret:
        return None, "❌ 错误：无法读取第一帧！"
    
    tracker = HybridTracker(device='cuda')
    tracker.init(first_frame, init_bbox)
    
    # 在第一帧上绘制初始边界框
    vis_frame = first_frame.copy()
    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # "Initial Target" -> "初始目标"
    # 注意：cv2.putText 不支持直接绘制中文字符，若需中文需使用 PIL 转换。
    # 这里为了保持代码依赖简单，暂时保留英文或使用拼音，或者假设系统支持。
    # 为保证稳定性，这里用英文 "Initial Target" (因为OpenCV默认不支持中文)
    # 如果一定要中文，通常需要 freetype 或 PIL。这里我保留英文以防乱码，
    # 但在注释中说明。
    cv2.putText(vis_frame, "Initial Target", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out.write(vis_frame)
    
    # 跟踪统计信息
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
    
    print(f"🎯 开始跟踪，初始边界框: {init_bbox}")
    
    # 处理剩余帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 跟踪对象
        bbox, confidence, status = tracker.update(frame)
        confidences.append(confidence)
        
        # 更新统计信息
        if status == "REDETECTED":
            stats['redetected'] += 1
        elif status == "LOST":
            stats['lost'] += 1
        else:
            stats['tracked'] += 1
        
        # 可视化（仅当置信度 > 0.15 时显示边界框）
        vis_frame = frame.copy()
        if bbox is not None and confidence > 0.15:
            x, y, w, h = [int(v) for v in bbox]
            
            # 基于置信度设定颜色
            if confidence > 0.7:
                color = (0, 255, 0)  # 绿色
            elif confidence > 0.4:
                color = (0, 165, 255)  # 橙色
            else:
                color = (0, 0, 255)  # 红色
            
            # 绘制边界框
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, 2)
            
            # 状态文本
            text = f"{status} | Conf: {confidence:.2f}"
            cv2.putText(vis_frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 帧计数器（始终显示）
        cv2.putText(vis_frame, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(vis_frame)
        frame_idx += 1
        
        # 进度指示器
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"处理进度: {progress:.1f}% | 状态: {status} | 置信度: {confidence:.2f}")
    
    # 清理资源
    cap.release()
    out.release()
    
    # 计算最终统计信息
    elapsed_time = time.time() - start_time
    stats['avg_fps'] = total_frames / elapsed_time if elapsed_time > 0 else 0
    stats['avg_confidence'] = np.mean(confidences) if confidences else 0
    
    stats_text = f"""
    ✅ 跟踪完成！
    📊 统计数据:
    - 总帧数: {stats['total_frames']}
    - 成功跟踪帧数: {stats['tracked']}
    - 重检测次数: {stats['redetected']}
    - 丢失帧数: {stats['lost']}
    - 平均 FPS: {stats['avg_fps']:.1f}
    - 平均置信度: {stats['avg_confidence']:.3f}    
    🎥 输出文件: {output_path.name}
    """
    
    return str(output_path), stats_text


def load_first_frame(video_path):
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
        # 全局存储帧
        roi_state['frame'] = frame
        roi_state['points'] = []
        roi_state['bbox'] = None
        
        # 将 BGR 转换为 RGB 以供 Gradio 使用
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # 将边界框初始化为帧中心的四分之一
        default_x1 = width // 4
        default_y1 = height // 4
        default_x2 = 3 * width // 4
        default_y2 = 3 * height // 4
        # 在帧上绘制初始边界框
        preview = rgb_frame.copy()
        cv2.rectangle(preview, (default_x1, default_y1), (default_x2, default_y2), (0, 255, 0), 2)
        # 注意：此处为在Gradio中显示的图片，若OpenCV不支持中文，依然使用英文提示
        cv2.putText(preview, "Adjust bbox below or click two points", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return preview, default_x1, default_y1, default_x2, default_y2, width, height
    
    return None, None, None, None, None, None, None


def update_bbox_preview(video_path, x1, y1, x2, y2):
    global roi_state
    if roi_state['frame'] is None:
        return None
    # 转换为 RGB
    preview = cv2.cvtColor(roi_state['frame'], cv2.COLOR_BGR2RGB).copy()
    # 如果边界框有效则绘制
    try:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # 绘制坐标文本
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
    处理图像点击事件以设置边界框角点
    """
    global roi_state
    
    if roi_state['frame'] is None:
        return x1, y1, x2, y2
    
    click_x, click_y = evt.index[0], evt.index[1]
    roi_state['points'].append((click_x, click_y))
    
    print(f"🖱️ 点击第 {len(roi_state['points'])} 次: ({click_x}, {click_y})")
    
    # 点击两次后，设置边界框
    if len(roi_state['points']) >= 2:
        p1, p2 = roi_state['points'][-2], roi_state['points'][-1]
        new_x1 = min(p1[0], p2[0])
        new_y1 = min(p1[1], p2[1])
        new_x2 = max(p1[0], p2[0])
        new_y2 = max(p1[1], p2[1])
        
        print(f"✓ 边界框已设定: ({new_x1},{new_y1}) 至 ({new_x2},{new_y2})")
        
        return new_x1, new_y1, new_x2, new_y2
    
    return x1, y1, x2, y2


# Gradio 界面
with gr.Blocks(title="🎯 零样本目标跟踪器", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 零样本目标跟踪系统 (Zero-shot Object Tracking)
    
    **功能特点:**
    - ✨ 跟踪任意目标 (零样本/Zero-shot)
    - 🚀 实时性能 (30-50 FPS)
    - 🔄 遮挡恢复能力 (支持 3秒+ 遮挡)
    - 🎨 GPU 加速 (CUDA)
    
    **使用说明:**
    1. 上传一个视频文件。
    2. **方法 1**: 在视频帧上点击 **两点** 来定义边界框（左上角和右下角）。
    3. **方法 2**: 调整图像下方的坐标滑块。
    4. 点击 "开始跟踪" (Start Tracking)。
    5. 下载跟踪完成的视频。
    """)
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="📤 上传视频")
            
            frame_display = gr.Image(
                label="🎯 点击两点选择 ROI（或使用下方滑块调整）",
                interactive=False,
                type="numpy"
            )
            
            with gr.Row():
                x1_slider = gr.Number(label="X1 (左)", value=0, precision=0)
                y1_slider = gr.Number(label="Y1 (顶)", value=0, precision=0)
            
            with gr.Row():
                x2_slider = gr.Number(label="X2 (右)", value=100, precision=0)
                y2_slider = gr.Number(label="Y2 (底)", value=100, precision=0)
            
            # 隐藏的尺寸状态
            frame_width = gr.State(value=None)
            frame_height = gr.State(value=None)
            
            track_btn = gr.Button("🚀 开始跟踪", variant="primary", size="lg")
        
        with gr.Column():
            output_video = gr.Video(label="📥 跟踪结果视频")
            stats_output = gr.Textbox(label="📊 跟踪统计信息", lines=12)
    
    # 事件处理
    video_input.change(
        fn=load_first_frame,
        inputs=[video_input],
        outputs=[frame_display, x1_slider, y1_slider, x2_slider, y2_slider, frame_width, frame_height]
    )
    
    # 当滑块变化时更新预览
    for slider in [x1_slider, y1_slider, x2_slider, y2_slider]:
        slider.change(
            fn=update_bbox_preview,
            inputs=[video_input, x1_slider, y1_slider, x2_slider, y2_slider],
            outputs=[frame_display]
        )
    
    # 处理图像点击
    frame_display.select(
        fn=handle_image_click,
        inputs=[video_input, x1_slider, y1_slider, x2_slider, y2_slider],
        outputs=[x1_slider, y1_slider, x2_slider, y2_slider]
    )
    
    # 开始跟踪
    track_btn.click(
        fn=process_video,
        inputs=[video_input, x1_slider, y1_slider, x2_slider, y2_slider],
        outputs=[output_video, stats_output]
    )
    
    gr.Markdown("""
    ---
    ### 🔧 技术栈:
    - **主跟踪器**: OpenCV CSRT (通道与空间可靠性跟踪器)
    - **运动预测**: 卡尔曼滤波 (Kalman Filter, 8状态恒速模型)
    - **重检测**: YOLO11n (CUDA 加速，本地模型)
    - **重识别 (Re-ID)**: 多特征匹配 (颜色直方图 + 模板匹配 + 空间邻近度)
    
    ### 📈 性能优化:
    - 第 1 层: CSRT 用于逐帧跟踪 (速度最快)
    - 第 2 层: 低置信度期间使用卡尔曼预测
    - 第 3 层: 触发式 YOLO11 重检测 (仅在需要时运行)
    - 自适应置信度阈值实现智能切换
    """)


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=3021)