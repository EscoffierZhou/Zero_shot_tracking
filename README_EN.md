# 🚀 Zero-Shot Object Tracking: Technical Report & Evolution

This project explores the evolution of high-performance, zero-shot object tracking systems, moving from heavy segmentation models to optimized traditional CV methods, and finally to state-of-the-art Transformer-based tracking accelerated by TensorRT.

## 📊 Performance Benchmark

We evaluated 5 distinct iterations of the tracking system. The trade-off between **Stability** and **Speed** drove our research direction.

| Version | Directory | Algorithm | Speed (FPS) | Stability | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v5 (Best)** | `CV_TRACKING_FP32` | **OSTrack (TRT FP32)** | **120+** | ⭐⭐⭐⭐ | **Best Balance.** High precision, extreme speed, robust CUDA memory management. |
| **v4** | `CV_TRACKING_ADVANCED` | OSTrack (TRT FP16) | 100+ | ⭐⭐⭐ | Fast but slightly less stable due to FP16 precision loss in bounding box regression. |
| **v3** | `CV_TRACKING_INTER` | OSTrack (PyTorch) | 5~30 | ⭐⭐⭐⭐⭐ | **Most Stable.** Reference implementation. Slow on CPU/Unoptimized GPU. |
| **v2** | `CV_TRACKING` | CSRT + Kalman | 20+ | ⭐⭐⭐⭐ | "Surgical" traditional approach. Good for simple motion, fails on deformation/occlusion. |
| **v1** | `CV_TRACKING_FAILED` | SAM + XMem + TRT | < 10 | ⭐ | Failed experiment. Heavy computation, no occlusion recovery, unusable for real-time. |

### 🏆 Rankings
*   **Stability**: `CV_TRACKING_INTER` > `CV_TRACKING` > `CV_TRACKING_FP32` > `CV_TRACKING_ADVANCED`
*   **Speed**: `CV_TRACKING_FP32` > `CV_TRACKING_ADVANCED` > `CV_TRACKING` > `CV_TRACKING_INTER`

---

## 🛤️ Research Evolution Path

### Phase 1: The "Heavy" Approach (Failure)
*   **Goal**: Use Segment Anything Model (SAM) and XMem for pixel-perfect mask tracking.
*   **Outcome**: **FAILED**.
*   **Reason**: The pipeline was too heavy for real-time usage (<10 FPS). Crucially, it lacked a mechanism to recover from complete occlusion—once the target was lost, it was gone forever.

### Phase 2: The "Surgical" Approach (Traditional CV)
*   **Goal**: Debug tracking logic using established, explainable algorithms.
*   **Method**: Implemented **CSRT** (Channel and Spatial Reliability Tracker) augmented with a **Kalman Filter**.
*   **Outcome**: **Success for simple cases**.
*   **Key Learning**: We successfully decoupled detection from motion prediction. However, traditional features (HOG/Color Histograms) failed during significant deformation or similar-object distractors.

### Phase 3: End-to-End Deep Learning (OSTrack)
*   **Goal**: Leverage Transformers for robust feature extraction and global search capabilities.
*   **Method**: Deployed **OSTrack** (One-Stream Transformer Tracking).
*   **Outcome**: **High Stability**. The model could handle occlusion and deformation perfectly, but the native PyTorch implementation was too slow for high-refresh-rate applications.

### Phase 4 & 5: TensorRT Acceleration (The Breakthrough)
*   **Goal**: Optimize OSTrack for real-time performance (100+ FPS) without losing accuracy.
*   **Method**: Compiled the ViT backbone to TensorRT engines (FP16 and FP32).
*   **Result**: Achieved **120+ FPS** in the FP32 version (`CV_TRACKING_FP32`), making it viable for high-speed intercept and drone tracking scenarios.

---

## 🧠 Core Algorithm: OSTrack (One-Stream Transformer)

The heart of our system is **OSTrack**, which fundamentally changes how tracking is modeled compared to Siamese networks.

### 1. One-Stream Framework
Unlike Siamese trackers that process the *Template* (Target) and *Search Region* separately and then cross-correlate them, OSTrack feeds **both** into a single Vision Transformer (ViT).
*   **Input**: Concatenated tokens of Template $Z$ and Search Region $X$.
*   **Mechanism**: Self-attention layers allow immediate, bidirectional information flow between the target and the search area from the very first layer.
*   **Benefit**: This "early fusion" allows the model to learn powerful discriminative features that distinguish the target from the background much more effectively than late-fusion correlation.

### 2. Candidate Elimination (CE) Module
To speed up inference, OSTrack doesn't process all background tokens in the final layers.
*   **Logic**: Tokens in the search region that have low similarity to the template are "eliminated" (masked out) in deeper layers.
*   **Effect**: Reduces computational load significantly, as the Transformer only focuses on the potential target area.

### 3. Pure Transformer Architecture
*   **Backbone**: ViT-B (Vision Transformer Base) pretrained on MAE (Masked Autoencoders).
*   **Head**: A simple fully convolutional head predicts the score map (confidence), offset, and size directly from the transformer output.

---

## ⚡ Optimization Deep Dive: TensorRT FP32

The `CV_TRACKING_FP32` version represents the pinnacle of our optimization efforts. Here is how we achieved 120+ FPS:

### 1. Compilation Strategy
We built the TensorRT engine using the **Python API** (`tensorrt` library) rather than `trtexec` for finer control.
*   **Precision**: We explicitly chose **FP32** (32-bit floating point) over FP16.
    *   *Why?* While FP16 is theoretically faster, we found it introduced jitter in the bounding box regression (coordinate drift). FP32 maintained the "pixel-perfect" stability of the PyTorch version while still providing massive acceleration on modern GPUs.
*   **Explicit Batch**: Configured the network for explicit batch size (Batch=1) to optimize for latency over throughput.

### 2. CUDA Memory Management
A common bottleneck in Python-based AI is the overhead of moving data between CPU and GPU.
*   **Pinned Memory (Page-Locked)**: We allocate input/output buffers using `cuda.pagelocked_empty`. This allows the DMA (Direct Memory Access) controller to transfer data to the GPU without CPU involvement.
*   **Persistent Context**: We manually manage the CUDA context (`pycuda.driver.Context`) to prevent the expensive initialization cost on every frame.

### 3. Zero-Copy Preprocessing
In the standard pipeline, image normalization involves creating multiple temporary copies of the image array.
*   **Optimization**: We rewrote the `preprocess` function to write normalized float data **directly** into the pinned memory buffer.
*   **Effect**: Eliminated 2-3 redundant memory copies per frame, significantly reducing CPU usage and "feeding" the GPU faster.

### 4. Temporal Smoothing
To further stabilize the high-speed output, we implemented an **Exponential Moving Average (EMA)** filter on the bounding box coordinates:
$$ \text{State}_{t} = \alpha \cdot \text{Prediction}_{t} + (1 - \alpha) \cdot \text{State}_{t-1} $$
*   **Position**: $\alpha = 0.7$ (Responsive)
*   **Size**: $\alpha = 0.5$ (Smooth)
This eliminates high-frequency jitter often seen in raw neural network outputs.

---

## 🚀 How to Run

To experience the best version (**CV_TRACKING_FP32**):

1.  Navigate to the directory:
    ```bash
    cd F:\desktop\CV_Project\CV_TRACKING_FP32
    ```
2.  Run the launcher script:
    ```cmd
    run_trt_fp32.bat
    ```
3.  Upload a video, select a target, and enjoy **120+ FPS** tracking!

---
*Author: Escoffier Zhou & Antigravity AI*
*Date: December 2025*
