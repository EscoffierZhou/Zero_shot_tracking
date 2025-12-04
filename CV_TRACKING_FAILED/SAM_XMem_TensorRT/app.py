import os
# === 0. 环境变量优化 ===
# 开启显存碎片整理，这对 8GB 显存至关重要
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import torch
import numpy as np
import cv2
import gc
import time
import gradio as gr
from PIL import Image

# === 1. 路径与环境配置 ===
sys.path.append(os.path.join(os.getcwd(), 'XMem'))

from model.network import XMem
# from inference.inference_core import InferenceCore # Use custom core for occlusion handling
from inference_core_custom import InferenceCore
from segment_anything import sam_model_registry, SamPredictor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Running on: {device}")

# === 2. 模型加载 ===

# --- 加载 XMem (FP32 + Autocast) ---
print("Loading XMem model...")
# 保持 FP32 权重，依靠 Autocast 和 no_grad 优化显存
network = XMem(config={'enable_long_term': True, 'enable_short_term': True}).to(device).eval()
xmem_checkpoint = 'XMem.pth'

def load_xmem_weights(model, path):
    if not os.path.exists(path):
        print(f"❌ Error: Weights not found at {path}")
        return
    try:
        checkpoint = torch.load(path, map_location='cpu')
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint: state_dict = checkpoint['model']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ XMem weights loaded successfully from {path}")
    except Exception as e:
        print(f"❌ Failed to load XMem weights: {e}")

load_xmem_weights(network, xmem_checkpoint)

# --- 加载 SAM ---
print("Loading SAM model...")
sam_checkpoint = "sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
print("✅ SAM Loaded!")

# === 3. 全局状态 ===
class TrackerState:
    def __init__(self):
        self.cap = None
        self.first_frame = None
        self.mask = None
        self.video_path = None

state = TrackerState()

# === 4. 核心逻辑 ===

def on_video_upload(video_path):
    if video_path is None: return None

    state.video_path = video_path
    state.cap = cv2.VideoCapture(video_path)
    ret, frame = state.cap.read()
    if not ret: return None

    # BGR -> RGB
    state.first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print("⚡ Extracting SAM features for the first frame...")
    predictor.set_image(state.first_frame)
    state.mask = None
    return state.first_frame

def on_click(evt: gr.SelectData):
    if state.first_frame is None: return None
    print(f"🖱️ Clicked at: {evt.index}")

    try:
        input_point = np.array([[evt.index[0], evt.index[1]]])
        input_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        best_mask = masks[0]
        state.mask = best_mask.astype(np.uint8) * 255

        # 可视化
        overlay = state.first_frame.copy()
        red_map = np.zeros_like(overlay)
        red_map[:, :, 0] = 255

        alpha = 0.5
        mask_bool = state.mask > 0
        if mask_bool.any():
            overlay[mask_bool] = cv2.addWeighted(
                overlay[mask_bool], 1 - alpha,
                red_map[mask_bool], alpha, 0
            ).squeeze()

        cv2.circle(overlay, (evt.index[0], evt.index[1]), 6, (0, 255, 0), -1)
        return overlay

    except Exception as e:
        print(f"❌ SAM Prediction Error: {e}")
        return state.first_frame

@torch.no_grad()
def run_tracking(progress=gr.Progress()):
    if state.mask is None or state.cap is None:
        return None, "❌ Error: No mask or video loaded."

    print("🚀 Starting XMem Tracking Pipeline...")

    # 性能监控初始化
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    # 显存清理
    torch.cuda.empty_cache()
    gc.collect()

    # 初始化推理核心
    processor = InferenceCore(network, config={
        'enable_long_term': True,
        'enable_short_term': True,
        'enable_long_term_count_usage': True,
        'hidden_dim': 64,
        'key_dim': 64,
        'value_dim': 512,
        'num_prototypes': 128,
        'min_mid_term_frames': 10,      # 改为 10 (增加短期记忆容量)
        'max_mid_term_frames': 20,     # 改为 20 (增加短期记忆容量)
        'max_long_term_elements': 10000, # 改回 10000 (让它记得更久)
        'mem_every': 5,
        'deep_update_every': -1,
        'save_every': -1,
        'show_every': -1,
        'size': -1,
        'top_k': 30,
    })

    # 传入第一帧和 Mask
    mask_torch = torch.from_numpy(state.mask > 128).long().to(device)
    processor.set_all_labels([1])

    frame_torch = (torch.from_numpy(state.first_frame).permute(2, 0, 1).float().to(device) / 255.0)

    with torch.autocast("cuda"):
        processor.step(frame_torch, mask_torch[None, ...])

    # 视频写入设置
    width = int(state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = state.cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(state.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 尝试使用浏览器兼容性更好的编码
    # H.264 (avc1) > VP9 (vp09) > MP4V (兼容性最差但无需额外库)
    # 确保 output 目录存在
    os.makedirs("output", exist_ok=True)
    output_path = "output/tracking_result.mp4"
    
    codecs_to_try = ['avc1', 'vp09', 'mp4v']
    writer = None

    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"✅ Using video codec: {codec}")
                break
        except:
            continue

    if writer is None or not writer.isOpened():
        print("⚠️ Failed to initialize preferred codecs, falling back to default mp4v")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 处理循环
    for idx in progress.tqdm(range(total_frames), desc="Processing"):
        ret, frame = state.cap.read()
        if not ret: break

        vis = frame.copy()

        if idx == 0:
            pred_mask = (state.mask > 0)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = (torch.from_numpy(frame_rgb).permute(2, 0, 1).float().to(device) / 255.0)

            with torch.autocast("cuda"):
                prob = processor.step(frame_tensor)

            pred_mask = torch.argmax(prob, dim=0)
            pred_mask = (pred_mask.cpu().numpy() == 1)

        if pred_mask.any():
            green_map = np.zeros_like(vis)
            green_map[:, :, 1] = 255
            alpha = 0.5
            vis[pred_mask] = cv2.addWeighted(
                vis[pred_mask], 1-alpha,
                green_map[pred_mask], alpha, 0
            ).squeeze()

        writer.write(vis)

        if idx % 20 == 0:
            torch.cuda.empty_cache()

    writer.release()
    del processor
    torch.cuda.empty_cache()
    gc.collect()

    # === 性能统计 ===
    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = total_frames / total_time if total_time > 0 else 0
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB

    metrics_str = (
        f"✅ Tracking Completed!\n"
        f"⏱️ Total Time: {total_time:.2f} s\n"
        f"🎞️ Total Frames: {total_frames}\n"
        f"🚀 Average FPS: {avg_fps:.2f} fps\n"
        f"💾 Peak GPU Memory: {max_mem:.2f} MB"
    )
    print(metrics_str)

    return output_path, metrics_str

# === 5. UI ===
with gr.Blocks(title="SAM_Xmem Tracker (TensoRT优化)") as demo:
    gr.Markdown("# SAM_Xmem Tracker (TensoRT优化)")

    with gr.Row():
        # 左侧：交互区
        with gr.Column():
            gr.Markdown("### 1. 上传与选择")
            video_in = gr.Video(label="上传视频")
            click_img = gr.Image(label="点击选择对象", interactive=True, type="numpy")

        # 右侧：结果区
        with gr.Column():
            gr.Markdown("### 2. 结果预览")
            mask_preview = gr.Image(label="Mask 预览", interactive=False)
            track_btn = gr.Button("🚀 开始追踪 (保留原画质)", variant="primary")

            # 输出区域：视频 + 指标
            video_out = gr.Video(label="追踪结果")
            metrics_out = gr.Textbox(label="性能指标 (Performance Metrics)", lines=5)

    # 事件绑定
    video_in.upload(on_video_upload, inputs=video_in, outputs=click_img)
    click_img.select(on_click, inputs=None, outputs=mask_preview)

    # 修改：同时输出 视频 和 指标文本
    track_btn.click(run_tracking, inputs=None, outputs=[video_out, metrics_out])

if __name__ == "__main__":
    # 允许局域网访问
    demo.launch(server_name="0.0.0.0", server_port=7860)
