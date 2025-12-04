#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <chrono>

int main() {
    // 创建一张 4K 大图
    cv::Mat h_img = cv::Mat::zeros(2160, 3840, CV_8UC3);
    randu(h_img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    
    // CPU 测试
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cv::Mat h_result;
    cv::GaussianBlur(h_img, h_result, cv::Size(5, 5), 0);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count() << " ms" << std::endl;

    // GPU 测试 (包含上传下载时间)
    cv::cuda::GpuMat d_img, d_result;
    d_img.upload(h_img); // 第一次上传可能含初始化开销
    
    auto ptr = cv::cuda::createGaussianFilter(d_img.type(), d_img.type(), cv::Size(5, 5), 0);
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    ptr->apply(d_img, d_result);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    
    std::cout << "GPU Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count() << " ms" << std::endl;
    
    return 0;
}