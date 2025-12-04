#include "SamDecoder.h"
#include <cuda_runtime_api.h>

// 简单的 Logger 定义 (复用之前的)
extern nvinfer1::ILogger* gLogger; // 如果在 main 里定义了全局 logger，这里可以用 extern，或者重新定义一个简单的

class DecoderLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) std::cout << "[Decoder] " << msg << std::endl;
    }
} d_logger;

SamDecoder::SamDecoder(const std::string& engine_path) {
    loadEngine(engine_path);

    // 计算各部分大小 (Batch=1)
    size_embed_  = 1 * 256 * 64 * 64 * sizeof(float);
    size_coords_ = 1 * MAX_POINTS * 2 * sizeof(float);
    size_labels_ = 1 * MAX_POINTS * sizeof(float);
    size_masks_  = 1 * 4 * 256 * 256 * sizeof(float); // 输出4个mask候选
    size_iou_    = 1 * 4 * sizeof(float);

    // 分配显存
    cudaMalloc(&buffers_[0], size_embed_);
    cudaMalloc(&buffers_[1], size_coords_);
    cudaMalloc(&buffers_[2], size_labels_);
    cudaMalloc(&buffers_[3], size_masks_);
    cudaMalloc(&buffers_[4], size_iou_);
}

SamDecoder::~SamDecoder() {
    for(int i=0; i<5; ++i) cudaFree(buffers_[i]);
}

void SamDecoder::loadEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error loading decoder engine!" << std::endl;
        return;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    runtime_.reset(nvinfer1::createInferRuntime(d_logger));
    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), size));
    context_.reset(engine_->createExecutionContext());
}

bool SamDecoder::run(const std::vector<float>& image_embeddings,
                     const std::vector<cv::Point2f>& points,
                     const std::vector<float>& labels,
                     cv::Mat& final_mask)
{
    // 1. 准备 Input: Image Embeddings
    // (在完全体中，这步应该直接在显存间传递，现在为了演示先从CPU拷过去)
    cudaMemcpy(buffers_[0], image_embeddings.data(), size_embed_, cudaMemcpyHostToDevice);

    // 2. 准备 Input: Points & Labels
    // 我们需要把用户点击的几个点，填充到固定的 20 个点
    std::vector<float> input_coords(MAX_POINTS * 2, 0.0f); // 初始化为 0
    std::vector<float> input_labels(MAX_POINTS, -1.0f);    // 初始化为 -1 (Padding)

    int num_user_points = points.size();
    if (num_user_points > MAX_POINTS) num_user_points = MAX_POINTS;

    for (int i = 0; i < num_user_points; ++i) {
        input_coords[i*2]     = points[i].x;
        input_coords[i*2 + 1] = points[i].y;
        input_labels[i]       = labels[i];
    }
    // 如果没有点，这里至少要保留 padding 逻辑 (我们已经在 ONNX 导出时处理了 -1 的逻辑)

    cudaMemcpy(buffers_[1], input_coords.data(), size_coords_, cudaMemcpyHostToDevice);
    cudaMemcpy(buffers_[2], input_labels.data(), size_labels_, cudaMemcpyHostToDevice);

    // 3. 推理
    if(!context_->executeV2(buffers_)) return false;

    // 4. 取回结果
    // SAM 输出 4 个 Mask，通常取 IOU 最高的那个，或者取第一个(索引0)作为最佳结果
    // 这里简单起见，我们直接取回所有数据，然后在 CPU 端选
    std::vector<float> gpu_masks(4 * 256 * 256);
    std::vector<float> gpu_iou(4);

    cudaMemcpy(gpu_masks.data(), buffers_[3], size_masks_, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_iou.data(),   buffers_[4], size_iou_,   cudaMemcpyDeviceToHost);

    // 5. 后处理：找到最佳 Mask (IOU 最高的)
    int best_idx = 0;
    float max_iou = -1000.0f;
    for(int i=0; i<4; ++i) {
        if(gpu_iou[i] > max_iou) {
            max_iou = gpu_iou[i];
            best_idx = i;
        }
    }

    // 6. 提取最佳 Mask 并二值化 (Sigmoid > 0.0 -> Logits > 0)
    // TRT 输出的是 Logits，大于0即为前景
    cv::Mat mask_logits(256, 256, CV_32FC1, gpu_masks.data() + best_idx * 256 * 256);

    // 简单二值化得到 Mask (0 或 255)
    // 注意：这里的 Mask 是 256x256 的，如果原图是 1024，后续还需要 resize 回去
    cv::threshold(mask_logits, final_mask, 0.0, 255, cv::THRESH_BINARY);
    final_mask.convertTo(final_mask, CV_8UC1);

    return true;
}