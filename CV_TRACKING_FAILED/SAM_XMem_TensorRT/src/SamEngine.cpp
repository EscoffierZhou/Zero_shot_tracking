#include "SamEngine.h"
#include <cuda_runtime_api.h>

SamEngine::SamEngine(const std::string& engine_path) {
    loadEngine(engine_path);

    // 计算 Buffer 大小
    input_size_ = 1 * INPUT_C * INPUT_H * INPUT_W * sizeof(float); // Batch=1, FP32 (我们输入是FP32，TRT内部会转FP16)
    output_size_ = 1 * EMBED_DIM * EMBED_H * EMBED_W * sizeof(float);

    // 分配显存
    cudaMalloc(&buffers_[0], input_size_);
    cudaMalloc(&buffers_[1], output_size_);
}

SamEngine::~SamEngine() {
    cudaFree(buffers_[0]);
    cudaFree(buffers_[1]);
}

void SamEngine::loadEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Could not read engine file: " << path << std::endl;
        return;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), size));
    context_.reset(engine_->createExecutionContext());
}

// OpenCV 预处理：Resize -> Normalize -> CHW Layout
void SamEngine::preprocess(const cv::Mat& img, float* gpu_input) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));

    // 转换成 Float32 并且 归一化 (0-255 -> 0-1)
    // SAM 官方还需要减去均值除以方差，这里简化为 0-1，通常够用，严格复现需要做 standardize
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // HWC (OpenCV默认) -> CHW (TensorRT需要)
    // 这是一个非常耗时的步骤如果用 CPU 做。
    // 这里为了演示简单先用 CPU，后续可以用 cv::cuda::split 优化
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    // 将三个通道的数据拷贝到连续的内存中
    std::vector<float> chw_data;
    for (int i = 0; i < 3; ++i) {
        std::vector<float> data;
        channels[i] = channels[i].reshape(1, 1); // 展平
        data.assign((float*)channels[i].data, (float*)channels[i].data + channels[i].total());
        chw_data.insert(chw_data.end(), data.begin(), data.end());
    }

    // 拷贝到显存
    cudaMemcpy(gpu_input, chw_data.data(), input_size_, cudaMemcpyHostToDevice);
}

bool SamEngine::run(const cv::Mat& img, std::vector<float>& output_embeddings) {
    if (!context_) return false;

    // 1. 预处理并上传 GPU
    preprocess(img, (float*)buffers_[0]);

    // 2. 推理
    // TensorRT 10 可能需要用 executeV2
    context_->executeV2(buffers_);

    // 3. 结果传回 CPU
    output_embeddings.resize(output_size_ / sizeof(float));
    cudaMemcpy(output_embeddings.data(), buffers_[1], output_size_, cudaMemcpyDeviceToHost);

    return true;
}