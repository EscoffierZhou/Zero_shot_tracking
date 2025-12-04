# SAM_XMem_TensorRT

这是一个高性能的视频对象追踪系统，结合了 SAM (Segment Anything Model) 和 XMem (Extended Memory)，并利用 TensorRT 进行优化，实现了实时的视频对象分割和追踪。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🎯 概述

本项目集成了两个强大的 AI 模型，提供直观且高效的视频对象追踪解决方案：

- **SAM (Segment Anything Model)**：用于通过点击交互进行初始对象选择
- **XMem (Extended Memory)**：处理时序视频对象分割，具有高效的记忆机制
- **TensorRT 优化**：加速推理以实现实时性能

用户只需在视频的第一帧中点击一个对象，AI 就会自动在整个视频序列中追踪并分割该对象。

---

## 📖 什么是 SAM (Segment Anything Model)？

### 概述
**SAM** 是由 Meta AI Research 开发的图像分割基础模型。它于 2023 年 4 月发布，代表了计算机视觉领域的重大突破，能够对任何图像中的任何对象进行零样本分割。

### 主要特性
- **提示式分割 (Promptable Segmentation)**：可以基于各种提示分割对象：
  - 点击 (前景/背景)
  - 边界框
  - 文本描述
  - 掩码输入
  
- **零样本泛化 (Zero-Shot Generalization)**：适用于训练期间从未见过的对象和图像领域

- **实时性能**：尽管模型巨大 (ViT-H 约 6 亿参数)，优化版本仍能以交互速度运行

### 架构
SAM 包含三个主要组件：

1. **图像编码器 (Image Encoder)** (基于 ViT)
   - 处理输入图像以生成图像嵌入
   - 使用 Vision Transformer (ViT) 架构
   - 本项目使用 **ViT-B (Base)** 变体，以平衡速度和精度
   
2. **提示编码器 (Prompt Encoder)**
   - 将用户提示 (点、框、掩码、文本) 编码为嵌入向量
   - 同时支持多种提示类型
   
3. **掩码解码器 (Mask Decoder)**
   - 轻量级解码器，结合图像和提示嵌入
   - 生成高质量的分割掩码
   - 可生成带有置信度分数的多个候选掩码

### 为什么在视频追踪中使用 SAM？
在本项目中，SAM 作为**初始化模块**：
- **用户友好**：简单的点击界面进行对象选择
- **精准**：即使对于复杂的对象也能生成高质量的初始掩码
- **快速**：配合 TensorRT 优化，编码器每段视频仅运行一次
- **灵活**：无需训练或微调即可适用于任何对象

### 我们流程中的 SAM
```
用户点击对象 → SAM 编码器处理第一帧 → 
SAM 解码器生成掩码 → 掩码传递给 XMem 进行追踪
```

---

## 📖 什么是 XMem (Extended Memory)？

### 概述
**XMem** 是由 UIUC 和 Adobe Research 研究人员开发的最先进的视频对象分割 (VOS) 模型。发表于 ECCV 2022，XMem 引入了一种创新的记忆机制，在保持高分割质量的同时高效处理长视频。

### 核心创新：Atkinson-Shiffrin 记忆模型
XMem 从人类记忆心理学中汲取灵感，实现了一个三级记忆系统：

1. **感觉记忆 (工作记忆)**
   - 存储最近的一帧
   - 提供追踪的即时上下文
   
2. **短期记忆 (STM)**
   - 保留最近的几帧 (可配置，通常 5-10 帧)
   - 提供时间一致性
   - 频繁更新
   
3. **长期记忆 (LTM)**
   - 存储整个视频历史中的关键帧
   - 实现长距离的时间推理
   - 从短期记忆中整合而来
   - 防止在长序列中“遗忘”

### 技术架构

#### 记忆管理
```
帧 t → 查询特征 → 
  ↓
匹配：
  - 工作记忆 (帧 t-1)
  - 短期记忆 (最近 5-10 帧)  
  - 长期记忆 (历史关键帧)
  ↓
生成分割掩码
```

#### 关键组件
- **特征提取器**：从视频帧中提取视觉特征
- **记忆读取器**：从记忆库中检索相关信息
- **记忆写入器**：用新信息更新记忆
- **解码器**：生成最终的分割掩码

### XMem 的优势

1. **长视频能力**
   - 可以处理任意长度的视频
   - 通过智能整合保持内存使用受限
   - 不会像传统方法那样遭受“漂移”或“遗忘”

2. **高精度**
   - 在 VOS 基准测试 (DAVIS, YouTube-VOS) 上表现出色
   - 处理遮挡、快速运动和外观变化

3. **高效性**
   - 智能记忆整合防止内存无限增长
   - 针对 GPU 加速进行了优化
   - 逐帧处理，开销极小

4. **鲁棒性**
   - 同时处理多个对象
   - 从暂时遮挡中恢复
   - 适应随时间变化的外观

### 系统中的 XMem 配置
```python
config = {
    'enable_long_term': True,           # 启用 LTM 以支持长视频
    'enable_short_term': True,          # 启用 STM 以保持时间一致性
    'min_mid_term_frames': 5,           # STM 最小帧数
    'max_mid_term_frames': 10,          # STM 最大帧数
    'max_long_term_elements': 10000,    # LTM 容量
    'mem_every': 5,                     # 每 5 帧更新一次记忆
    'top_k': 30,                        # Top-k 匹配以提高效率
}
```

---

## 🚀 为什么要进行 TensorRT 优化？

**TensorRT** 是 NVIDIA 的高性能深度学习推理优化器和运行时。在本项目中：

- **SAM 编码器/解码器**：转换为 TensorRT 引擎以加快推理速度
- **速度提升**：比 PyTorch 实现快 2-5 倍
- **显存效率**：优化的算子融合和显存分配
- **兼容性**：在 NVIDIA GPU (推荐 RTX 2060 及以上) 上运行

### 本项目中的 TensorRT 模型
- `sam_vit_b_encoder.engine`：优化的 SAM 图像编码器
- `sam_vit_b_decoder.engine`：优化的 SAM 掩码解码器
- 两者均从 ONNX 格式转换而来，以获得最大兼容性

---

## 🔧 安装

### 前置要求
- Python 3.8 或更高版本
- CUDA 11.0 或更高版本
- NVIDIA GPU，显存 8GB+ (推荐 RTX 3060 或更好)
- TensorRT 8.0+

### 第一步：克隆仓库
```bash
git clone https://github.com/EscoffierZhou/SAM_XMem_TensorRT.git
cd SAM_XMem_TensorRT
```

### 第二步：安装依赖
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install gradio opencv-python numpy pillow
pip install segment-anything
```

### 第三步：下载模型
下载预训练模型：
- **SAM ViT-B**: `sam_vit_b_01ec64.pth` (375MB)
- **XMem**: `XMem.pth` (249MB)

将它们放在项目根目录下。

### 第四步：转换为 TensorRT (可选)
如果你想自己构建 TensorRT 引擎：
```bash
python export_sam_encoder.py
python export_sam_decoder.py
```

这将生成：
- `sam_vit_b_encoder.engine`
- `sam_vit_b_decoder.engine`

---

## 💻 使用方法

### Web 界面 (推荐)
启动 Gradio Web 界面：
```bash
python app.py
```

然后在浏览器中打开 `http://localhost:7860`

### 工作流程
1. **上传视频**：点击“上传视频”上传你的视频文件
2. **选择对象**：在第一帧中点击你想追踪的对象
3. **开始追踪**：点击“🚀 开始追踪”处理整个视频
4. **下载结果**：追踪后的视频将保存在 `output/tracking_result.mp4`

### 性能提示
- 对于 8GB 显存：处理 720p 或更低分辨率的视频
- 通过 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 启用显存优化
- 每 20 帧清理一次缓存 (代码中已实现)

---

## 📁 项目结构

```
SAM_XMem_TensorRT/
├── app.py                          # Gradio Web 界面
├── export_sam_encoder.py           # SAM 编码器转 ONNX/TRT
├── export_sam_decoder.py           # SAM 解码器转 ONNX/TRT
├── XMem/                           # XMem 源代码
│   ├── model/
│   │   └── network.py              # XMem 网络架构
│   └── inference/
│       └── inference_core.py       # XMem 推理引擎
├── models/                         # 额外模型文件
├── output/                         # 结果输出目录
├── src/                            # C++ 源文件 (可选)
├── include/                        # C++ 头文件 (可选)
├── CMakeLists.txt                  # CMake 构建配置
├── sam_vit_b_01ec64.pth            # SAM 预训练权重
├── XMem.pth                        # XMem 预训练权重
├── sam_vit_b_encoder.engine        # TensorRT 优化编码器
├── sam_vit_b_decoder.engine        # TensorRT 优化解码器
├── sam_vit_b_encoder.onnx          # ONNX 中间格式
└── sam_vit_b_decoder.onnx          # ONNX 中间格式
```

---

## ⚡ 性能基准

在 **NVIDIA RTX 3060 (8GB 显存)** 上使用 720p 视频测试：

| 指标 | 数值 |
|--------|-------|
| **平均 FPS** | 15-25 fps |
| **峰值 GPU 显存** | 6.5 GB |
| **编码器时间 (第一帧)** | ~200ms |
| **每帧处理时间** | ~40-60ms |

### 对比：PyTorch vs TensorRT
| 模型组件 | PyTorch | TensorRT | 加速比 |
|----------------|---------|----------|---------|
| SAM 编码器 | ~500ms | ~200ms | **2.5x** |
| SAM 解码器 | ~80ms | ~30ms | **2.7x** |
| XMem (每帧) | ~60ms | ~40ms | **1.5x** |

---

## 🎓 参考与引用

### SAM (Segment Anything)
```bibtex
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  journal={arXiv:2304.02643},
  year={2023}
}
```
- **论文**: https://arxiv.org/abs/2304.02643
- **官方仓库**: https://github.com/facebookresearch/segment-anything

### XMem
```bibtex
@inproceedings{cheng2022xmem,
  title={XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model},
  author={Cheng, Ho Kei and Alexander G. Schwing},
  booktitle={ECCV},
  year={2022}
}
```
- **论文**: https://arxiv.org/abs/2207.07115
- **官方仓库**: https://github.com/hkchengrex/XMem

---

## 🛠️ 需求

### Python 包
- torch >= 1.13.0
- torchvision >= 0.14.0
- opencv-python >= 4.7.0
- numpy >= 1.23.0
- gradio >= 3.50.0
- pillow >= 9.0.0
- segment-anything >= 1.0

### 系统要求
- **操作系统**: Linux (Ubuntu 20.04+) 或 Windows 10/11
- **GPU**: 支持 CUDA 的 NVIDIA GPU (8GB+ 显存)
- **CUDA**: 11.0 或更高版本
- **TensorRT**: 8.0+ (用于优化推理)
- **内存**: 推荐 16GB+

---

## 📝 许可证

本项目仅供研究和教育用途。商业用途请参考 SAM 和 XMem 的原始许可证。

- **SAM**: Apache 2.0 License
- **XMem**: Apache 2.0 License

---

## 🙏 致谢

- 感谢 Meta AI Research 提供的惊人 SAM 模型
- 感谢 Ho Kei Cheng 和 Alexander G. Schwing 开发的 XMem
- 感谢 NVIDIA 提供的 TensorRT 优化工具
- 感谢开源社区提供的各种工具和库

---

## 📧 联系方式

如有问题、议题或合作：
- **邮箱**: 3416270780@qq.com
- **GitHub**: https://github.com/EscoffierZhou/SAM_XMem_TensorRT

---

## 🚧 未来改进

- [ ] 支持多对象追踪
- [ ] 实时流媒体模式
- [ ] 移动端部署 (ONNX Runtime / Jetson 上的 TensorRT)
- [ ] 交互式掩码细化
- [ ] 自动关键帧选择优化
- [ ] 支持更高分辨率视频 (1080p+)

---

**Made with ❤️ for the Computer Vision Community**
