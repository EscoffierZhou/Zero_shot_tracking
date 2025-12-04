#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>

namespace sam_tracker {

/**
 * @brief 时序记忆银行：存储历史帧的SAM Embeddings和Bounding Boxes
 *
 * 工作机制：
 * - 采用滑动窗口策略，保留最近N帧的记忆
 * - 支持置信度加权和动量平滑 (Momentum = 0.9)
 * - 防止模型漂移和遮挡导致的跟踪失败
 */
    class MemoryBank {
    public:
        /**
         * @param capacity 最大记忆帧数 (建议: 10-15)
         * @param device 计算设备 (CUDA)
         */
        MemoryBank(int capacity, const torch::Device& device);
        ~MemoryBank() = default;

        /**
         * @brief 添加新的帧 embedding 和 box 到记忆库
         * @param embedding SAM Image Encoder 输出 [1, 256, 64, 64]
         * @param box 当前帧的边界框
         * @param confidence 置信度 (0-1)，由SAM Decoder的IoU分数决定
         */
        void add_embedding(const torch::Tensor& embedding, const cv::Rect& box, float confidence = 1.0f);

        /**
         * @brief 获取加权组合的 embedding 和平均 box
         * @return pair<combined_embedding, averaged_box>
         *
         * 算法:
         * - 对 embeddings 做加权平均 (权重 = confidence * exp(-t/T))
         * - 对 boxes 做指数平滑 EMA: box_t = α*box_t + (1-α)*box_{t-1}
         */
        std::pair<torch::Tensor, cv::Rect> get_combined_embedding() const;

        /** @brief 清空记忆库 (用于场景切换或手动重置) */
        void clear();

        /** @brief 获取当前记忆帧数 */
        size_t size() const { return bank_.size(); }

    private:
        struct MemoryItem {
            torch::Tensor embedding;  // SAM Image Embedding
            cv::Rect box;             // 对应边界框
            float confidence;         // 置信度分数
            float timestamp;          // 时间戳 (用于衰减权重)
        };

        std::deque<MemoryItem> bank_;  // 双端队列，O(1) 头尾操作
        int capacity_;                 // 最大容量
        torch::Device device_;         // 设备类型
        float momentum_ = 0.85f;       // Box 平滑动量系数

        // 时间衰减参数: w = exp(-age / tau)
        float temporal_decay_ = 0.95f; // 每帧衰减因子
    };

} // namespace sam_tracker