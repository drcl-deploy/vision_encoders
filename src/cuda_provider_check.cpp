#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main() {
  try {
    const std::vector<std::string> providers = Ort::GetAvailableProviders();
    if (std::find(providers.begin(), providers.end(),
                  "CUDAExecutionProvider") == providers.end()) {
      std::cerr << "CUDAExecutionProvider is not present in this ONNX Runtime "
                   "build\n";
      return 1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vision_encoders_cuda_check");
    Ort::SessionOptions options;
    OrtCUDAProviderOptions cuda_options{};
    options.AppendExecutionProvider_CUDA(cuda_options);
    std::cout << "ONNX Runtime CUDA provider loaded successfully\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ONNX Runtime CUDA provider check failed: " << error.what()
              << '\n';
    return 1;
  }
}
