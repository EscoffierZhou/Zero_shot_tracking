#include "memory_bank.h"
#include <algorithm>

namespace sam_tracker {

    MemoryBank::MemoryBank(int capacity, const torch::Device& device)
            : capacity_(capacity) {
        bank_.reserve(capacity);
    }

    void MemoryBank::add_embedding(const torch::Tensor& embedding, const cv::Rect& box) {
        // LIFO 策略，保持最新记忆
        if (bank_.size() >= capacity_) {
            bank_.erase(bank_.begin());
        }

        // 动量平滑 (Momentum Smoothing)
        float confidence = 1.0f / (bank_.size() + 1);
        bank_.push_back({embedding.clone(), box, confidence});
    }

    std::pair<torch::Tensor, cv::Rect> MemoryBank::get_combined_embedding() const {
        if (bank_.empty()) {
            return {torch::Tensor(), cv::Rect()};
        }

        // 加权平均 Embedding (注意力机制)
        torch::Tensor combined = torch::zeros_like(bank_[0].embedding);
        cv::Rect avg_box(0, 0, 0, 0);
        float weight_sum = 0.0f;

        for (const auto& item : bank_) {
            combined += item.embedding * item.confidence;
            avg_box.x += item.box.x * item.confidence;
            avg_box.y += item.box.y * item.confidence;
            avg_box.width += item.box.width * item.confidence;
            avg_box.height += item.box.height * item.confidence;
            weight_sum += item.confidence;
        }

        return {combined / weight_sum,
                cv::Rect(avg_box.x / weight_sum, avg_box.y / weight_sum,
                         avg_box.width / weight_sum, avg_box.height / weight_sum)};
    }

} // namespace sam_tracker