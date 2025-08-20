#pragma once

#include <cuda_fp16.h>
#include <opencv2/core/cuda.hpp>

// Kernel declaration
__global__ void fp32_to_fp16_kernel(const float* input, __half* output, int size);

// Host wrapper
void convertFp32ToFp16(const cv::cuda::GpuMat& input, __half* output, int numElems);
