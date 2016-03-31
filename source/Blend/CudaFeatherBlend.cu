#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/device/common.hpp>
#include "CudaUtil.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void accumulate8UC4To32SC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    const unsigned char* weightData, int weightRows, int weightCols, int weightStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < srcCols && y < srcRows)
    {
        getRowPtr<int4>(dstData, dstStep, y)[x] = getRowPtr<int4>(dstData, dstStep, y)[x] + 
            getElem<int>(weightData, weightStep, y, x) * getElem<uchar4>(srcData, srcStep, y, x);
    }
}

__global__ void normalize32SC4Feather(unsigned char* imageData, int imageRows, int imageCols, int imageStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < imageCols && y < imageRows)
    {
        getRowPtr<int4>(imageData, imageStep, y)[x] = getElem<int4>(imageData, imageStep, y, x) >> 16;
    }
}

void accumulate8UC4To32SC4(const cv::gpu::GpuMat& src, const cv::gpu::GpuMat& weight, cv::gpu::GpuMat& dst)
{
    CV_Assert(src.data && src.type() == CV_8UC4 &&
        weight.data && weight.type() == CV_32SC1 &&
        dst.data && dst.type() == CV_32SC4 &&
        src.size() == weight.size() && src.size() == dst.size());

    const dim3 block(32, 8);
    const dim3 grid(cv::gpu::divUp(src.cols, block.x), cv::gpu::divUp(src.rows, block.y));
    accumulate8UC4To32SC4<<<grid, block>>>(src.data, src.rows, src.cols, src.step, 
        weight.data, weight.rows, weight.cols, weight.step,
        dst.data, dst.rows, dst.cols, dst.step);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void normalize32SC4Feather(cv::gpu::GpuMat& image)
{
    CV_Assert(image.data && image.type() == CV_32SC4);
    const dim3 block(32, 8);
    const dim3 grid(cv::gpu::divUp(image.cols, block.x), cv::gpu::divUp(image.rows, block.y));
    normalize32SC4Feather<<<grid, block>>>(image.data, image.rows, image.cols, image.step);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}