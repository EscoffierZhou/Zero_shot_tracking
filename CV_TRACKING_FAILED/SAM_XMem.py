import gradio as gr
import cv2
import numpy as np
import torch
import sys
import os
from segment_anything import sam_model_registry, SamPredictor

# ---------------------------------------------------------
# 1. 系统配置与模型加载 (System Config & Model Loading)
# ---------------------------------------------------------

# 自动选择计算设备
# [B, C, H, W] 上下文中的 Device 选择
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# 模型类型选择 (vit_b 是速度和精度的平衡点)
MODEL_TYPE = "vit_b"
# 请确保该文件存在，或者修改路径
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"


def load_sam_model():

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint {CHECKPOINT_PATH} not found.")
        print("Please download it from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        # 为了演示不报错，这里返回 None，实际运行需要下载
        return None

    try:
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        predictor = SamPredictor(sam)
        return predictor
    except Exception as e:
        print(f"Error loading SAM: {e}")
        return None



sam_predictor = load_sam_model()


# ---------------------------------------------------------
# 2. 算法核心逻辑 (Core Algorithms)
# ---------------------------------------------------------

def get_box_from_mask(mask):

    pos = np.where(mask > 0)
    if len(pos[0]) == 0:
        return None
    return np.array([np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])])


def run_tracking_sam(video_path, user_bbox_data, progress=gr.Progress()):
    """
    核心跟踪循环：SAM + Optical Flow

    理论分析 (Complexity Analysis):
    - SAM Encoder: O(H*W) - Transformer 基于 Patch，计算量巨大。
    - Optical Flow (LK): O(N * K^2) - N 是特征点数，K 是窗口大小。
    - 整体复杂度：每帧都跑 SAM Encoder 是瓶颈。
    """
    if sam_predictor is None:
        return None, "Error: SAM Model not loaded. Check checkpoint path."

    if video_path is None or user_bbox_data is None:
        return None, "Error: No video or box detected."

    # 1. 解析用户输入 (Parse User Input)
    # Gradio 的 ImageEditor 返回的数据结构
    mask_input = user_bbox_data["layers"][0]  # [H, W, 4] or [H, W]

    # 如果是 RGBA，转为单通道 Mask
    if len(mask_input.shape) == 3:
        mask_input = mask_input[:, :, 0]

    init_box = get_box_from_mask(mask_input)
    if init_box is None:
        return video_path, "No object selected."

    # 2. 视频流初始化
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 降采样以加快处理速度 (SAM 对高分辨率图像推理较慢)
    # [H, W, C]
    resize_factor = 0.5
    proc_w, proc_h = int(width * resize_factor), int(height * resize_factor)

    # 输出视频配置
    output_path = "tracked_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 3. 状态初始化
    # current_box: [x1, y1, x2, y2] (在原始分辨率下)
    current_box = init_box
    prev_gray = None
    prev_pts = None

    # 颜色定义
    mask_color = np.array([30, 144, 255], dtype=np.uint8)  # DodgerBlue

    for i in progress.tqdm(range(total_frames), desc="Tracking Objects"):
        ret, frame = cap.read()
        if not ret:
            break

        # 原始帧用于显示，缩小帧用于光流计算
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_small = cv2.resize(frame_gray, (proc_w, proc_h))

        # ---------------------------------------------------------
        # 阶段 A: 位置预测 (Position Prediction via Optical Flow)
        # 类似于 Kalman Filter 的 Predict 步骤，但利用像素级特征
        # ---------------------------------------------------------
        if prev_gray is not None and prev_pts is not None:
            # Lucas-Kanade Optical Flow
            # p1: [N, 1, 2] (New points)
            # st: Status (1 if found)
            # err: Error
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, frame_gray_small, prev_pts, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            # 选择好的跟踪点
            good_new = p1[st == 1]
            good_old = prev_pts[st == 1]

            if len(good_new) > 5:  # 如果有足够的点被跟踪
                # 计算平均位移向量 (dx, dy)
                movement = np.mean(good_new - good_old, axis=0)
                # 还原回原始分辨率
                movement /= resize_factor

                # 更新 Box 位置 (简单的平移)
                # [x1, y1, x2, y2]
                current_box[0] += movement[0]
                current_box[1] += movement[1]
                current_box[2] += movement[0]
                current_box[3] += movement[1]
            else:
                # 跟踪丢失处理：这里简单保持原位或扩大搜索范围
                pass

        # ---------------------------------------------------------
        # 阶段 B: SAM 细化 (Refinement via SAM)
        # 类似于 Kalman Filter 的 Update 步骤，利用观测值(Prompt)修正
        # ---------------------------------------------------------

        # 必须调用 set_image，这是最耗时的部分
        # SAM 需要 [H, W, 3] RGB input
        sam_predictor.set_image(frame_rgb)

        # 使用预测的 Box 作为 Prompt
        input_box = current_box[None, :]  # 增加 Batch 维度 -> [1, 4]

        masks, scores, logits = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,  # 只返回这一个物体
        )

        # masks: [1, H, W] boolean
        best_mask = masks[0]

        # ---------------------------------------------------------
        # 阶段 C: 状态更新 (State Update)
        # 为下一帧准备光流特征点
        # ---------------------------------------------------------

        # 根据新的 Mask 更新 Box，防止漂移
        refined_box = get_box_from_mask(best_mask)

        if refined_box is not None:
            # 可以在这里做动量平滑 (Momentum Smoothing)
            # current_box = alpha * current_box + (1-alpha) * refined_box
            current_box = refined_box

            # 提取 Mask 内部的特征点用于下一帧光流
            # 1. 创建 Mask 的缩小版
            mask_small = cv2.resize(best_mask.astype(np.uint8), (proc_w, proc_h))

            # 2. 仅在 Mask 区域内寻找角点 (Shi-Tomasi Corner Detector)
            prev_pts = cv2.goodFeaturesToTrack(
                frame_gray_small, mask=mask_small,
                maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
            )

        prev_gray = frame_gray_small

        # ---------------------------------------------------------
        # 阶段 D: 可视化 (Visualization)
        # ---------------------------------------------------------

        # 叠加 Mask
        # Mask [H, W] -> [H, W, 1]
        if refined_box is not None:
            viz_mask = best_mask[:, :, np.newaxis]
            # 简单的半透明叠加
            overlay = frame.copy()
            overlay[best_mask] = (frame[best_mask] * 0.5 + mask_color * 0.5).astype(np.uint8)
            frame = overlay

            # 画框
            p1, p2 = (int(current_box[0]), int(current_box[1])), (int(current_box[2]), int(current_box[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(frame, "SAM Tracker", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    return output_path, "Tracking Completed."


def get_first_frame(video_path):
    if video_path is None: return None
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


# ---------------------------------------------------------
# 3. Gradio 界面构建
# ---------------------------------------------------------

with gr.Blocks(title="Zero-Shot SAM Tracker") as app:
    gr.Markdown("# 🧬 Zero-Shot Object Tracking (SAM + Optical Flow)")
    gr.Markdown("""
    **核心机制**: 使用 Segment Anything Model (SAM) 获取高质量分割，利用光流法 (Optical Flow) 模拟时序记忆传递 Prompt。
    **注意**: 需要下载 `sam_vit_b_01ec64.pth` 权重文件。
    """)

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="1. 上传视频 (Upload Video)")
            extract_btn = gr.Button("2. 获取第一帧 (Get Frame)")
            # ImageEditor 用于画框
            image_input = gr.ImageEditor(
                label="3. 涂抹目标 (Paint over Target)",
                type="numpy",
                brush=gr.Brush(colors=["#FFFFFF"], default_size=20),
                interactive=True
            )
            track_btn = gr.Button("4. 开始跟踪 (Start Tracking)", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="跟踪结果 (Result)")
            status_text = gr.Textbox(label="状态 (Status)", interactive=False)

    extract_btn.click(fn=get_first_frame, inputs=video_input, outputs=image_input)
    track_btn.click(
        fn=run_tracking_sam,
        inputs=[video_input, image_input],
        outputs=[video_output, status_text]
    )

# ---------------------------------------------------------
# 4. CLI 运行声明 (CLI Command)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Starting SAM Tracker App...")
    print("Command: python tracker_sam_app.py")
    app.launch(share=False)