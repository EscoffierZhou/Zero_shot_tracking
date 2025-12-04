#include <iostream>
#include "SamEngine.h"
#include "SamDecoder.h"

int main() {
    // 1. 初始化
    std::cout << "Loading Encoder..." << std::endl;
    SamEngine encoder("sam_vit_b_encoder.engine");

    std::cout << "Loading Decoder..." << std::endl;
    SamDecoder decoder("sam_vit_b_decoder.engine");

    // 2. 准备假数据
    cv::Mat img = cv::Mat::zeros(1024, 1024, CV_8UC3);
    cv::circle(img, cv::Point(512, 512), 100, cv::Scalar(255, 255, 255), -1); // 画个白球

    // 3. 运行 Encoder (耗时)
    std::cout << "Running Encoder..." << std::endl;
    std::vector<float> features;
    auto start = std::chrono::high_resolution_clock::now();
    encoder.run(img, features);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Encoder Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    // 4. 模拟点击 (点击中心点)
    std::vector<cv::Point2f> points;
    points.push_back(cv::Point2f(512, 512)); // 点击白球中心

    std::vector<float> labels;
    labels.push_back(1.0f); // 1 = 前景

    // 5. 运行 Decoder (极速)
    std::cout << "Running Decoder..." << std::endl;
    cv::Mat mask;
    start = std::chrono::high_resolution_clock::now();
    decoder.run(features, points, labels, mask);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Decoder Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    // 6. 保存结果验证
    cv::imwrite("result_mask.png", mask);
    std::cout << "Result saved to result_mask.png" << std::endl;

    return 0;
}