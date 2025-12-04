#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>

int main() {
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    try {
        int device_count = cv::cuda::getCudaEnabledDeviceCount();
        std::cout << "CUDA Device Count: " << device_count << std::endl;

        if (device_count > 0) {
            cv::cuda::printShortCudaDeviceInfo(0);
            std::cout << "\n[SUCCESS] OpenCV is using CUDA!" << std::endl;
        } else {
            std::cout << "\n[WARNING] No CUDA devices found. Is CUDA installed correctly?" << std::endl;
        }
    } catch (const cv::Exception& ex) {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}