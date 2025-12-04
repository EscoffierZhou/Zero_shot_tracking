#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

class SamDecoder {
public:
    SamDecoder(const std::string& engine_path);
    ~SamDecoder();

    // 核心函数：输入特征向量 + 点坐标，输出 Mask
    // image_embeddings: 来自 Encoder 的输出 (1x256x64x64)
    // points: 用户点击的点 (x, y)
    // labels: 1=前景, 0=背景
    // output_mask: 输出的掩码 (256x256)
    bool run(const std::vector<float>& image_embeddings,
             const std::vector<cv::Point2f>& points,
             const std::vector<float>& labels,
             cv::Mat& output_mask);

private:
    void loadEngine(const std::string& path);

    // TensorRT 组件
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // 显存指针数组 (TRT绑定)
    // 0: image_embeddings, 1: point_coords, 2: point_labels, 3: low_res_masks, 4: iou_predictions
    void* buffers_[5];

    // 维度常量 (我们之前固定的)
    const int MAX_POINTS = 20;
    const int EMBED_DIM = 256;
    const int EMBED_SIZE = 64;
    const int MASK_SIZE = 256;

    // 缓冲区大小记录
    size_t size_embed_;
    size_t size_coords_;
    size_t size_labels_;
    size_t size_masks_;
    size_t size_iou_;
};