#include "CudaUtil.cuh"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define COUNT_NON_ZERO_BLOCK_SIZE 256
#define UTIL_BLOCK_WIDTH 32
#define UTIL_BLOCK_HEIGHT 8

/*
__global__ void countNonZero8UC1(const unsigned char* data, int step, int rows, int cols, int* sum)
{
    __shared__ int ssum[UTIL_BLOCK_HEIGHT][UTIL_BLOCK_WIDTH];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tidx = threadIdx.x, tidy = threadIdx.y;

    ssum[tidy][tidx] = ((x < cols && y < rows) && getElem<unsigned char>(data, step, y, x) != 0) ? 1 : 0;
    __syncthreads();

    for (int gap = UTIL_BLOCK_WIDTH / 2; gap > 0; gap /= 2)
    {
        if (tidx < gap)
            ssum[tidy][tidx] += ssum[tidy][tidx + gap];
        __syncthreads();
    }    

    for (int gap = UTIL_BLOCK_HEIGHT / 2; gap > 0; gap /= 2)
    {
        if (tidy < gap && tidx == 0)
            ssum[tidy][0] += ssum[tidy + gap][0];
        __syncthreads();
    }    

    if (tidx == 0 && tidy == 0)
        atomicAdd(sum, ssum[0][0]);
}

int countNonZero8UC1(const cv::cuda::GpuMat& mat)
{
    CV_Assert(mat.data && mat.type() == CV_8UC1);

    cv::cuda::GpuMat dst(1, 1, CV_32SC1);
    dst.setTo(0);

    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(mat.cols, block.x), cv::cuda::device::divUp(mat.rows, block.y));
    countNonZero8UC1<<<grid, block>>>(mat.data, mat.step, mat.rows, mat.cols, (int*)dst.data);

    int count;
    dst.download(cv::Mat(1, 1, CV_32SC1, &count));
    return count;
}
*/

__global__ void countNonZero8UC1(const unsigned char* data, int step, int rows, int cols, int* sum)
{
    __shared__ int ssum[COUNT_NON_ZERO_BLOCK_SIZE];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    const int tidx = threadIdx.x;

    ssum[tidx] = ((x < cols && y < rows) && getElem<unsigned char>(data, step, y, x) != 0) ? 1 : 0;
    __syncthreads();

    for (int gap = COUNT_NON_ZERO_BLOCK_SIZE / 2; gap > 0; gap /= 2)
    {
        if (tidx < gap)
            ssum[tidx] += ssum[tidx + gap];
        __syncthreads();
    }

    if (tidx == 0)
        atomicAdd(sum, ssum[0]);
}

int countNonZero8UC1(const cv::cuda::GpuMat& mat)
{
    CV_Assert(mat.data && mat.type() == CV_8UC1);

    cv::cuda::GpuMat dst(1, 1, CV_32SC1);
    dst.setTo(0);

    const dim3 block(COUNT_NON_ZERO_BLOCK_SIZE);
    const dim3 grid(cv::cuda::device::divUp(mat.cols, block.x), mat.rows);
    countNonZero8UC1<<<grid, block>>>(mat.data, mat.step, mat.rows, mat.cols, (int*)dst.data);

    int count;
    dst.download(cv::Mat(1, 1, CV_32SC1, &count));
    return count;
}

__global__ void or8UC1(const unsigned char* aData, int aStep, const unsigned char* bData, int bStep,
    unsigned char* cData, int cStep, int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
    {
        getRowPtr<unsigned char>(cData, cStep, y)[x] = getElem<unsigned char>(aData, aStep, y, x) | getElem<unsigned char>(bData, bStep, y, x);
    }
}

void or8UC1(const cv::cuda::GpuMat& a, const cv::cuda::GpuMat& b, cv::cuda::GpuMat& c)
{
    CV_Assert(a.data && a.type() == CV_8UC1 &&
        b.data && b.type() == CV_8UC1 && a.size() == b.size());

    c.create(a.size(), CV_8UC1);

    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(a.cols, block.x), cv::cuda::device::divUp(a.rows, block.y));
    or8UC1<<<grid, block>>>(a.data, a.step, b.data, b.step, c.data, c.step, a.rows, a.cols);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}