#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "CudaUtil.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__constant__ unsigned char cudaLUT[256];

__global__ void transformKernel8UC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < srcCols && y < srcRows)
    {
        const unsigned char* srcPtr = getRowPtr<unsigned char>(srcData, srcStep, y) + x * 4;
        unsigned char* dstPtr = getRowPtr<unsigned char>(dstData, dstStep, y) + x * 4;
        dstPtr[0] = cudaLUT[srcPtr[0]];
        dstPtr[1] = cudaLUT[srcPtr[1]];
        dstPtr[2] = cudaLUT[srcPtr[2]];
        dstPtr[3] = 0;
    }
}

void cudaTransform(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const std::vector<unsigned char>& lut)
{
    CV_Assert(src.data && src.type() == CV_8UC4 && lut.size() == 256);

    int rows = src.rows, cols = src.cols;
    dst.create(rows, cols, CV_8UC4);

    cudaSafeCall(cudaMemcpyToSymbol(cudaLUT, lut.data(), 256));

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    transformKernel8UC4<<<grid, block>>>(src.data, rows, cols, src.step, dst.data, rows, cols, dst.step);
    cudaSafeCall(cudaGetLastError());
}

__constant__ unsigned char cudaLUTB[256], cudaLUTG[256], cudaLUTR[256];

__global__ void transformBGRKernel8UC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < srcCols && y < srcRows)
    {
        const unsigned char* srcPtr = getRowPtr<unsigned char>(srcData, srcStep, y) + x * 4;
        unsigned char* dstPtr = getRowPtr<unsigned char>(dstData, dstStep, y) + x * 4;
        dstPtr[0] = cudaLUTB[srcPtr[0]];
        dstPtr[1] = cudaLUTG[srcPtr[1]];
        dstPtr[2] = cudaLUTR[srcPtr[2]];
        dstPtr[3] = 0;
    }
}

void cudaTransform(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const std::vector<std::vector<unsigned char> >& luts)
{
    CV_Assert(src.data && src.type() == CV_8UC4 && 
        luts.size() == 3 && luts[0].size() == 256 && luts[1].size() == 256 && luts[2].size() == 256);

    int rows = src.rows, cols = src.cols;
    dst.create(rows, cols, CV_8UC4);

    cudaSafeCall(cudaMemcpyToSymbol(cudaLUTB, luts[0].data(), 256));
    cudaSafeCall(cudaMemcpyToSymbol(cudaLUTG, luts[1].data(), 256));
    cudaSafeCall(cudaMemcpyToSymbol(cudaLUTR, luts[2].data(), 256));

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    transformBGRKernel8UC4<<<grid, block>>>(src.data, rows, cols, src.step, dst.data, rows, cols, dst.step);
    cudaSafeCall(cudaGetLastError());
}