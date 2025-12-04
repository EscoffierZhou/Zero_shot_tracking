#include "sam_tracker.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace sam_tracker {

    SAMTracker::SAMTracker(const TrackingConfig& config, const std::string& engine_path)
            : config_(config), device_(torch::kCUDA, config.device_id) {

        // 加载 TensorRT 引擎 (或用 TorchScript)
        try {
            sam_encoder_ = torch::jit::load(engine_path + "/sam_encoder.ts");
            sam_decoder_ = torch::jit::load(engine_path + "/sam_decoder.ts");
            sam_encoder_.to(device_);
            sam_decoder_.to(device_);
            if (config_.use_fp16) {
                sam_encoder_.to(torch::kFloat16);
                sam_decoder_.to(torch::kFloat16);
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load models: " + std::string(e.what()));
        }

        memory_bank_ = std::make_unique<MemoryBank>(config_.memory_bank_size, device_);
    }

    SAMTracker::~SAMTracker() = default;

    void SAMTracker::initialize(const cv::Mat& first_frame, const cv::Rect& init_box) {
        // 预分配 GPU 内存 (内存池)
        int proc_w = first_frame.cols * config_.resize_factor;
        int proc_h = first_frame.rows * config_.resize_factor;

        gpu_frame_prev_.upload(first_frame);
        image_buffer_ = torch::zeros({1, 3, proc_h, proc_w},
                                     torch::TensorOptions().dtype(config_.use_fp16 ? torch::kFloat16 : torch::kFloat32)
                                             .device(device_));

        preprocess_frame(first_frame, image_buffer_);
        embedding_buffer_ = sam_encoder_.forward({image_buffer_}).toTensor();

        memory_bank_->add_embedding(embedding_buffer_, init_box);
        current_box_ = init_box;
    }

    cv::Rect SAMTracker::track_frame(const cv::Mat& frame) {
        // 1. GPU 光流预测位置
        cv::cuda::GpuMat gpu_frame_curr;
        gpu_frame_curr.upload(frame);

        cv::Point2f movement = flow_estimator_->estimate(gpu_frame_prev_, gpu_frame_curr, current_box_);
        current_box_.x += movement.x;
        current_box_.y += movement.y;
        gpu_frame_prev_ = gpu_frame_curr.clone();

        // 2. **每10帧才运行一次 SAM Encoder** (关键优化)
        static int frame_counter = 0;
        if (frame_counter++ % 10 == 0) {
            preprocess_frame(frame, image_buffer_);
            embedding_buffer_ = sam_encoder_.forward({image_buffer_}).toTensor();
            memory_bank_->add_embedding(embedding_buffer_, current_box_);
        }

        // 3. 使用 Memory Bank 的 Embedding 运行轻量 Decoder
        auto [combined_embedding, avg_box] = memory_bank_->get_combined_embedding();

        // 4. 准备 Prompt (Box 坐标需要归一化)
        torch::Tensor boxes = torch::from_blob(&current_box_, {1, 4}, torch::kInt32)
                .to(device_).to(torch::kFloat32);
        boxes[0][0] /= frame.cols; boxes[0][1] /= frame.rows;
        boxes[0][2] /= frame.cols; boxes[0][3] /= frame.rows;

        // 5. **算子融合**: 单次前向传播
        auto decoder_output = sam_decoder_.forward({
                                                           combined_embedding, boxes
                                                   }).toTensor();

        // 后处理: Mask -> Box
        torch::Tensor mask = torch::sigmoid(decoder_output).squeeze().gt(0.5);
        current_box_ = mask_to_box(mask);

        return current_box_;
    }

    torch::Tensor SAMTracker::mask_to_box(const torch::Tensor& mask) {
        // 使用 CUDA Kernel 加速 Mask 到 Box 的转换
        auto mask_cpu = mask.to(torch::kCPU, torch::kUInt8);
        cv::Mat mask_mat(mask.size(0), mask.size(1), CV_8U, mask_cpu.data_ptr());
        cv::Rect box = cv::boundingRect(mask_mat);
        return torch::tensor({box.x, box.y, box.x + box.width, box.y + box.height}, torch::kInt32);
    }

    void SAMTracker::preprocess_frame(const cv::Mat& frame, torch::Tensor& output) {
        // Zero-copy GPU 预处理 (避免 CPU-GPU 传输)
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(frame);
        cv::cuda::resize(gpu_frame, gpu_frame,
                         cv::Size(output.size(3), output.size(2)));
        cv::cuda::cvtColor(gpu_frame, gpu_frame, cv::COLOR_BGR2RGB);

        // 直接将 GPU 内存映射到 Tensor (Advanced)
        // 注意: 这里简化处理，实际可用 custom allocator
        std::vector<torch::Tensor> channels;
        cv::cuda::split(gpu_frame, channels);

        for (int i = 0; i < 3; ++i) {
            output[0][i] = torch::from_blob(channels[i].data,
                                            {output.size(2), output.size(3)},
                                            torch::kUInt8).to(output.dtype());
        }
    }

} // namespace sam_tracker