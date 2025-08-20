#include "detection_cpp/tools/fp16_utils.hpp"

__global__ void fp32_to_fp16_kernel(const float* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

void convertFp32ToFp16(const cv::cuda::GpuMat& input, __half* output, int numElems) {
    const float* inputPtr = reinterpret_cast<const float*>(input.ptr<float>());
    int blockSize = 256;
    int gridSize = (numElems + blockSize - 1) / blockSize;
    fp32_to_fp16_kernel<<<gridSize, blockSize>>>(inputPtr, output, numElems);
    cudaDeviceSynchronize();
}
