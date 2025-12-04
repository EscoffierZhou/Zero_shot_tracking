#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <nvofapi.h>  // NVIDIA Optical Flow
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

namespace sam_tracker {

struct TrackingConfig {
    float resize_factor = 0.5f;
    float confidence_threshold = 0.7f;
    int memory_bank_size = 10;
    bool use_fp16 = true;
    int device_id = 0;
};

class MemoryBank {
public:
    MemoryBank(int capacity, const torch::Device& device);
    void add_embedding(const torch::Tensor& embedding, const cv::Rect& box);
    std::pair<torch::Tensor, cv::Rect> get_combined_embedding() const;
private:
    struct MemoryItem {
        torch::Tensor embedding;
        cv::Rect box;
        float confidence;
    };
    std::vector<MemoryItem> bank_;
    int capacity_;
};

class OpticalFlowEstimator {
public:
    OpticalFlowEstimator(int width, int height);
    ~OpticalFlowEstimator();
    cv::Point2f estimate(const cv::cuda::GpuMat& prev, const cv::cuda::GpuMat& curr, 
                         const cv::Rect& prev_box);
private:
    nvof::NvOFHandle of_handle_;
    cv::cuda::GpuMat flow_buffer_;
};

class SAMTracker {
public:
    SAMTracker(const TrackingConfig& config, const std::string& engine_path);
    ~SAMTracker();
    
    void initialize(const cv::Mat& first_frame, const cv::Rect& init_box);
    cv::Rect track_frame(const cv::Mat& frame);
    cv::Mat get_visualization() const;
    
private:
    TrackingConfig config_;
    torch::Device device_;
    
    // TensorRT Engines
    torch::jit::script::Module sam_encoder_;  // 可替换为 TensorRT
    torch::jit::script::Module sam_decoder_;
    
    std::unique_ptr<MemoryBank> memory_bank_;
    std::unique_ptr<OpticalFlowEstimator> flow_estimator_;
    
    // 预分配的张量 (内存池)
    torch::Tensor image_buffer_;
    torch::Tensor embedding_buffer_;
    cv::cuda::GpuMat gpu_frame_prev_;
    cv::Rect current_box_;
    
    // 核心算子 (C++ 实现)
    torch::Tensor mask_to_box(const torch::Tensor& mask);
    void preprocess_frame(const cv::Mat& frame, torch::Tensor& output);
};

} // namespace sam_tracker