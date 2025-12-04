#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <memory>

namespace sam_tracker {

/**
 * @brief GPU 加速光流估计器
 *
 * 支持两种后端:
 * 1. **NVIDIA Optical Flow SDK** (推荐，硬件加速)
 * 2. **OpenCV CUDA Dense Optical Flow** (备用，纯GPU计算)
 *
 * 相比 CPU LK 算法优势:
 * - 处理 640x480 帧: ~0.5ms (vs CPU 15ms)
 * - 支持稠密光流，精度更高
 * - 异步计算，可与 SAM 推理重叠
 */
    class OpticalFlowEstimator {
    public:
        /**
         * @param width 输入帧宽度
         * @param height 输入帧高度
         * @param use_nvidia_of 是否使用 NVIDIA Optical Flow SDK (需要 Ampere+ GPU)
         */
        OpticalFlowEstimator(int width, int height, bool use_nvidia_of = true);
        ~OpticalFlowEstimator();

        /**
         * @brief 估计物体在帧间的位移向量
         * @param prev 前一帧灰度图 (GPU Mat)
         * @param curr 当前帧灰度图 (GPU Mat)
         * @param prev_box 前一帧的边界框
         * @return cv::Point2f 位移向量 (dx, dy)，已缩放到原始分辨率
         *
         * 实现步骤:
         * 1. 在 Box 区域内提取光流 ROI
         * 2. 计算 ROI 内光流向量的中值 (Robust 估计)
         * 3. 乘上缩放因子还原到原图分辨率
         */
        cv::Point2f estimate(const cv::cuda::GpuMat& prev,
                             const cv::cuda::GpuMat& curr,
                             const cv::Rect& prev_box);

    private:
        bool use_nvidia_of_;                     // 后端选择标志
        cv::Size frame_size_;                    // 帧尺寸

        // OpenCV CUDA 实现
        cv::Ptr<cv::cuda::DensePyrLKOpticalFlow> lk_cuda_;

        // NVIDIA OF SDK 实现 (需要包含 <nvOpticalFlow.h>)
        void* nvidia_of_handle_ = nullptr;       // opaque handle
        cv::cuda::GpuMat flow_buffer_;           // 光流输出缓冲区 [H/4, W/4, 2]
        cv::cuda::GpuMat prev_gray_, curr_gray_; // 灰度图缓存

        // 预计算的缩放因子 (因为光流是下采样输出的)
        float flow_scale_x_, flow_scale_y_;
    };

} // namespace sam_tracker