#include "optical_flow.h"
#include <cuda_runtime.h>

namespace sam_tracker {

    OpticalFlowEstimator::OpticalFlowEstimator(int width, int height) {
        // 初始化 NVIDIA Optical Flow SDK
        nvof::InitParams init_params;
        init_params.width = width;
        init_params.height = height;
        init_params.gpu_id = 0;
        init_params.out_grid_size = nvof::NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;

        nvof::NvOFAPI::getInterface()->nvOFCreate(init_params, &of_handle_);

        // 预分配光流缓冲区
        flow_buffer_.create(height / 4, width / 4, CV_32FC2);
    }

    OpticalFlowEstimator::~OpticalFlowEstimator() {
        nvof::NvOFAPI::getInterface()->nvOFDestroy(of_handle_);
    }

    cv::Point2f OpticalFlowEstimator::estimate(const cv::cuda::GpuMat& prev,
                                               const cv::cuda::GpuMat& curr,
                                               const cv::Rect& prev_box) {
        // 计算稠密光流 (GPU 加速)
        nvof::ExecuteParams exec_params;
        exec_params.input_frame = prev.data;
        exec_params.reference_frame = curr.data;
        exec_params.output_buffer = flow_buffer_.data;

        nvof::NvOFAPI::getInterface()->nvOFExecute(of_handle_, &exec_params);

        // 在 Box 区域内平均光流向量
        cv::cuda::GpuMat region_flow(flow_buffer_, cv::Rect(prev_box.x / 4, prev_box.y / 4,
                                                            prev_box.width / 4, prev_box.height / 4));
        cv::Mat host_flow;
        region_flow.download(host_flow);

        cv::Scalar mean_flow = cv::mean(host_flow);
        return cv::Point2f(mean_flow[0], mean_flow[1]);  // dx, dy
    }

} // namespace sam_tracker