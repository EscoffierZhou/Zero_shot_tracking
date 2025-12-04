#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
};

class SamEngine {
public:
    SamEngine(const std::string& engine_path);
    ~SamEngine();

    // 核心函数：输入 OpenCV 图片，输出特征向量
    bool run(const cv::Mat& img, std::vector<float>& output_embeddings);

private:
    void loadEngine(const std::string& path);
    void preprocess(const cv::Mat& img, float* gpu_input_buffer);

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // 显存指针
    void* buffers_[2]; // 0: Input, 1: Output

    // 维度信息
    const int INPUT_W = 1024;
    const int INPUT_H = 1024;
    const int INPUT_C = 3;
    const int EMBED_DIM = 256;
    const int EMBED_H = 64;
    const int EMBED_W = 64;

    // 缓冲区大小
    size_t input_size_;
    size_t output_size_;
};