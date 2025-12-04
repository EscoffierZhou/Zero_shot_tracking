#include <iostream>
#include <NvInfer.h>

// 继承 ILogger 以捕获 TensorRT 的输出
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
            // 只有警告和错误才打印，避免信息过多
            if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} logger;

int main() {
    std::cout << "Checking TensorRT..." << std::endl;

    // 【修改点1】直接调用全局函数，不要加 nvinfer1::
    int ver = getInferLibVersion();
    std::cout << "TensorRT Version: " << ver << std::endl;

    // 创建 Builder
    auto builder = nvinfer1::createInferBuilder(logger);
    if (builder) {
        std::cout << "[SUCCESS] TensorRT Builder created successfully!" << std::endl;
        // 记得销毁对象 (TensorRT 10 推荐使用智能指针，这里为了演示先用 delete)
        delete builder;
    } else {
        std::cout << "[FAILED] Could not create TensorRT Builder." << std::endl;
    }

    return 0;
}