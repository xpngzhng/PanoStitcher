#include "CudaUtil.cuh"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define COUNT_NON_ZERO_THREAD_SIZE 4
#define COUNT_NON_ZERO_BLOCK_SIZE 256
#define UTIL_BLOCK_WIDTH 32
#define UTIL_BLOCK_HEIGHT 8
#define UTIL_THREAD_SIZE 4

__device__ void warpReduce(volatile int* sdata, int tid) 
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

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

/*
__global__ void countNonZero8UC1(const unsigned char* data, int step, int rows, int cols, int* sum)
{
    __shared__ int ssum[UTIL_BLOCK_HEIGHT][UTIL_BLOCK_WIDTH];

    const int x = blockIdx.x * blockDim.x * UTIL_THREAD_SIZE + threadIdx.x;
    const int y = blockIdx.y * blockDim.y * UTIL_THREAD_SIZE + threadIdx.y;
    const int tidx = threadIdx.x, tidy = threadIdx.y;

    int threadSum = 0;
    for (int yy = y, i = 0; yy < rows && i < UTIL_THREAD_SIZE; yy += UTIL_BLOCK_HEIGHT, i++)
    {
        for (int xx = x, j = 0; xx < cols && j < UTIL_THREAD_SIZE; xx += UTIL_BLOCK_WIDTH, j++)
        {
            threadSum += (getElem<unsigned char>(data, step, yy, xx) != 0) ? 1 : 0;
        }
    }
    ssum[tidy][tidx] = threadSum;
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
*/

__global__ void countNonZero8UC1(const unsigned char* data, int step, int rows, int cols, int* sum)
{
    __shared__ int ssum[UTIL_BLOCK_HEIGHT * UTIL_BLOCK_WIDTH];

    const int x = blockIdx.x * blockDim.x * UTIL_THREAD_SIZE + threadIdx.x;
    const int y = blockIdx.y * blockDim.y * UTIL_THREAD_SIZE + threadIdx.y;
    const int tidx = threadIdx.x, tidy = threadIdx.y;
    const int tid = tidy * blockDim.x + tidx;

    int threadSum = 0;
    for (int yy = y, i = 0; yy < rows && i < UTIL_THREAD_SIZE; yy += UTIL_BLOCK_HEIGHT, i++)
    {
        for (int xx = x, j = 0; xx < cols && j < UTIL_THREAD_SIZE; xx += UTIL_BLOCK_WIDTH, j++)
        {
            threadSum += (getElem<unsigned char>(data, step, yy, xx) != 0) ? 1 : 0;
        }
    }
    ssum[tid] = threadSum;
    __syncthreads();

    for (int gap = UTIL_BLOCK_WIDTH * UTIL_BLOCK_HEIGHT / 2; gap > 32; gap /= 2)
    {
        if (tid < gap)
            ssum[tid] += ssum[tid + gap];
        __syncthreads();
    }

    if (tid < 32)
        warpReduce(ssum, tid);

    if (tid == 0)
        atomicAdd(sum, ssum[0]);
}

int countNonZero8UC1(const cv::cuda::GpuMat& mat)
{
    CV_Assert(mat.data && mat.type() == CV_8UC1);

    cv::cuda::GpuMat dst(1, 1, CV_32SC1);
    dst.setTo(0);

    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(mat.cols, block.x * UTIL_THREAD_SIZE), 
                    cv::cuda::device::divUp(mat.rows, block.y * UTIL_THREAD_SIZE));
    countNonZero8UC1<<<grid, block>>>(mat.data, mat.step, mat.rows, mat.cols, (int*)dst.data);

    int count;
    dst.download(cv::Mat(1, 1, CV_32SC1, &count));
    return count;
}


/*
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
*/

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

__global__ void not8UC1(const unsigned char* srcData, int srcStep, unsigned char* dstData, int dstStep,
    int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
    {
        getRowPtr<unsigned char>(dstData, dstStep, y)[x] = ~getElem<unsigned char>(srcData, srcStep, y, x);
    }
}

void not8UC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{
    CV_Assert(src.data && src.type() == CV_8UC1);

    dst.create(src.size(), CV_8UC1);

    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), cv::cuda::device::divUp(src.rows, block.y));
    not8UC1<<<grid, block>>>(src.data, src.step, dst.data, dst.step, src.rows, src.cols);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}