#include "CudaUtil.cuh"
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/device/common.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PYR_DOWN_BLOCK_SIZE 256

template<typename RowWiseReflectType>
__global__ void padLeftRight16SC1(unsigned char* data, int rows, int cols, int step, const RowWiseReflectType cb)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < rows)
    {
        getRowPtr<short>(data, step, y)[-2] = getElem<short>(data, step, y, cb.idx_col_low(-2));
        getRowPtr<short>(data, step, y)[-1] = getElem<short>(data, step, y, cb.idx_col_low(-1));
        getRowPtr<short>(data, step, y)[cols] = getElem<short>(data, step, y, cb.idx_col_high(cols));
        getRowPtr<short>(data, step, y)[cols + 1] = getElem<short>(data, step, y, cb.idx_col_high(cols + 1));
    }
}

template<typename ColWiseReflectType>
__global__ void padTopBottom16SC1(unsigned char* data, int rows, int padCols, int step, const ColWiseReflectType rb)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < padCols)
    {
        getRowPtr<short>(data, step, -2)[x] = getElem<short>(data, step, rb.idx_row_low(-2), x);
        getRowPtr<short>(data, step, -1)[x] = getElem<short>(data, step, rb.idx_row_low(-1), x);
        getRowPtr<short>(data, step, rows)[x] = getElem<short>(data, step, rb.idx_row_high(rows), x);
        getRowPtr<short>(data, step, rows + 1)[x] = getElem<short>(data, step, rb.idx_row_high(rows + 1), x);
    }
}

template<typename RowWiseReflectType, typename DataType>
__global__ void padLeftRight(unsigned char* data, int rows, int cols, int step, const RowWiseReflectType cb)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < rows)
    {
        getRowPtr<DataType>(data, step, y)[-2] = getElem<DataType>(data, step, y, cb.idx_col_low(-2));
        getRowPtr<DataType>(data, step, y)[-1] = getElem<DataType>(data, step, y, cb.idx_col_low(-1));
        getRowPtr<DataType>(data, step, y)[cols] = getElem<DataType>(data, step, y, cb.idx_col_high(cols));
        getRowPtr<DataType>(data, step, y)[cols + 1] = getElem<DataType>(data, step, y, cb.idx_col_high(cols + 1));
    }
}

template<typename ColWiseReflectType, typename DataType>
__global__ void padTopBottom(unsigned char* data, int rows, int padCols, int step, const ColWiseReflectType rb)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < padCols)
    {
        getRowPtr<DataType>(data, step, -2)[x] = getElem<DataType>(data, step, rb.idx_row_low(-2), x);
        getRowPtr<DataType>(data, step, -1)[x] = getElem<DataType>(data, step, rb.idx_row_low(-1), x);
        getRowPtr<DataType>(data, step, rows)[x] = getElem<DataType>(data, step, rb.idx_row_high(rows), x);
        getRowPtr<DataType>(data, step, rows + 1)[x] = getElem<DataType>(data, step, rb.idx_row_high(rows + 1), x);
    }
}

template<typename DataType>
__global__ void expand(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep)
{
    const int srcx = blockIdx.x * blockDim.x + threadIdx.x;
    const int srcy = blockIdx.y * blockDim.y + threadIdx.y;
    const int dstx = srcx * 2;
    const int dsty = srcy * 2;
    if (dstx < dstCols && dsty < dstRows)
        getRowPtr<DataType>(dstData, dstStep, dsty)[dstx] = getElem<DataType>(srcData, srcStep, srcy, srcx);
}

__global__ void pyrDown16SC1To32SC1(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep)
{
    __shared__ int smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    const int srcy = 2 * y;

    {
        int sum;
        sum =           getElem<short>(srcData, srcStep, srcy - 2, x);
        sum = sum + 4 * getElem<short>(srcData, srcStep, srcy - 1, x);
        sum = sum + 6 * getElem<short>(srcData, srcStep, srcy,     x);
        sum = sum + 4 * getElem<short>(srcData, srcStep, srcy + 1, x);
        sum = sum +     getElem<short>(srcData, srcStep, srcy + 2, x);
        smem[2 + threadIdx.x] = sum;
    }

    if (threadIdx.x < 2)
    {
        const int leftx = x - 2;
        int sum;
        sum =           getElem<short>(srcData, srcStep, srcy - 2, leftx);
        sum = sum + 4 * getElem<short>(srcData, srcStep, srcy - 1, leftx);
        sum = sum + 6 * getElem<short>(srcData, srcStep, srcy,     leftx);
        sum = sum + 4 * getElem<short>(srcData, srcStep, srcy + 1, leftx);
        sum = sum +     getElem<short>(srcData, srcStep, srcy + 2, leftx);
        smem[threadIdx.x] = sum;
    }

    if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
    {
        const int rightx = x + 2;
        int sum;
        sum =           getElem<short>(srcData, srcStep, srcy - 2, rightx);
        sum = sum + 4 * getElem<short>(srcData, srcStep, srcy - 1, rightx);
        sum = sum + 6 * getElem<short>(srcData, srcStep, srcy,     rightx);
        sum = sum + 4 * getElem<short>(srcData, srcStep, srcy + 1, rightx);
        sum = sum +     getElem<short>(srcData, srcStep, srcy + 2, rightx);
        smem[4 + threadIdx.x] = sum;
    }

    __syncthreads();

    if (threadIdx.x < PYR_DOWN_BLOCK_SIZE / 2)
    {
        const int tid2 = threadIdx.x * 2;
        int sum;
        sum =           smem[2 + tid2 - 2];
        sum = sum + 4 * smem[2 + tid2 - 1];
        sum = sum + 6 * smem[2 + tid2    ];
        sum = sum + 4 * smem[2 + tid2 + 1];
        sum = sum +     smem[2 + tid2 + 2];

        const int dstx = (blockIdx.x * blockDim.x + tid2) / 2;

        if (dstx < dstCols)
            getRowPtr<int>(dstData, dstStep, y)[dstx] = sum;
    }
}

__global__ void pyrDown16SC1To16SC1(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep)
{
    __shared__ int smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    const int srcy = 2 * y;

    {
        int sum;
        sum =           getElem<short>(srcData, srcStep, srcy - 2, x);
        sum = sum + 4 * getElem<short>(srcData, srcStep, srcy - 1, x);
        sum = sum + 6 * getElem<short>(srcData, srcStep, srcy,     x);
        sum = sum + 4 * getElem<short>(srcData, srcStep, srcy + 1, x);
        sum = sum +     getElem<short>(srcData, srcStep, srcy + 2, x);
        smem[2 + threadIdx.x] = sum;
    }

    if (threadIdx.x < 2)
    {
        const int leftx = x - 2;
        int sum;
        sum =           getElem<short>(srcData, srcStep, srcy - 2, leftx);
        sum = sum + 4 * getElem<short>(srcData, srcStep, srcy - 1, leftx);
        sum = sum + 6 * getElem<short>(srcData, srcStep, srcy,     leftx);
        sum = sum + 4 * getElem<short>(srcData, srcStep, srcy + 1, leftx);
        sum = sum +     getElem<short>(srcData, srcStep, srcy + 2, leftx);
        smem[threadIdx.x] = sum;
    }

    if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
    {
        const int rightx = x + 2;
        int sum;
        sum =           getElem<short>(srcData, srcStep, srcy - 2, rightx);
        sum = sum + 4 * getElem<short>(srcData, srcStep, srcy - 1, rightx);
        sum = sum + 6 * getElem<short>(srcData, srcStep, srcy,     rightx);
        sum = sum + 4 * getElem<short>(srcData, srcStep, srcy + 1, rightx);
        sum = sum +     getElem<short>(srcData, srcStep, srcy + 2, rightx);
        smem[4 + threadIdx.x] = sum;
    }

    __syncthreads();

    if (threadIdx.x < PYR_DOWN_BLOCK_SIZE / 2)
    {
        const int tid2 = threadIdx.x * 2;
        int sum;
        sum =           smem[2 + tid2 - 2];
        sum = sum + 4 * smem[2 + tid2 - 1];
        sum = sum + 6 * smem[2 + tid2    ];
        sum = sum + 4 * smem[2 + tid2 + 1];
        sum = sum +     smem[2 + tid2 + 2];

        const int dstx = (blockIdx.x * blockDim.x + tid2) / 2;

        if (dstx < dstCols)
            getRowPtr<short>(dstData, dstStep, y)[dstx] = (sum + 128) >> 8;
    }
}

__global__ void pyrDown16SC4To32SC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep)
{
    __shared__ int4 smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    const int srcy = 2 * y;

    {
        int4 sum;
        sum =       1 * getElem<short4>(srcData, srcStep, srcy - 2, x);
        sum = sum + 4 * getElem<short4>(srcData, srcStep, srcy - 1, x);
        sum = sum + 6 * getElem<short4>(srcData, srcStep, srcy,     x);
        sum = sum + 4 * getElem<short4>(srcData, srcStep, srcy + 1, x);
        sum = sum + 1 * getElem<short4>(srcData, srcStep, srcy + 2, x);
        smem[2 + threadIdx.x] = sum;
    }

    if (threadIdx.x < 2)
    {
        const int leftx = x - 2;
        int4 sum;
        sum =       1 * getElem<short4>(srcData, srcStep, srcy - 2, leftx);
        sum = sum + 4 * getElem<short4>(srcData, srcStep, srcy - 1, leftx);
        sum = sum + 6 * getElem<short4>(srcData, srcStep, srcy,     leftx);
        sum = sum + 4 * getElem<short4>(srcData, srcStep, srcy + 1, leftx);
        sum = sum + 1 * getElem<short4>(srcData, srcStep, srcy + 2, leftx);
        smem[threadIdx.x] = sum;
    }

    if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
    {
        const int rightx = x + 2;
        int4 sum;
        sum =       1 * getElem<short4>(srcData, srcStep, srcy - 2, rightx);
        sum = sum + 4 * getElem<short4>(srcData, srcStep, srcy - 1, rightx);
        sum = sum + 6 * getElem<short4>(srcData, srcStep, srcy,     rightx);
        sum = sum + 4 * getElem<short4>(srcData, srcStep, srcy + 1, rightx);
        sum = sum + 1 * getElem<short4>(srcData, srcStep, srcy + 2, rightx);
        smem[4 + threadIdx.x] = sum;
    }

    __syncthreads();

    if (threadIdx.x < PYR_DOWN_BLOCK_SIZE / 2)
    {
        const int tid2 = threadIdx.x * 2;
        int4 sum;
        sum =       1 * smem[2 + tid2 - 2];
        sum = sum + 4 * smem[2 + tid2 - 1];
        sum = sum + 6 * smem[2 + tid2    ];
        sum = sum + 4 * smem[2 + tid2 + 1];
        sum = sum + 1 * smem[2 + tid2 + 2];

        const int dstx = (blockIdx.x * blockDim.x + tid2) / 2;

        if (dstx < dstCols)
            getRowPtr<int4>(dstData, dstStep, y)[dstx] = sum;
    }
}

__global__ void pyrUp32SC4To32SC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep)
{
    __shared__ int4 smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    {
        int4 sum;
        sum =           getElem<int4>(srcData, srcStep, y - 2, x);
        sum = sum + 4 * getElem<int4>(srcData, srcStep, y - 1, x);
        sum = sum + 6 * getElem<int4>(srcData, srcStep, y,     x);
        sum = sum + 4 * getElem<int4>(srcData, srcStep, y + 1, x);
        sum = sum +     getElem<int4>(srcData, srcStep, y + 2, x);
        smem[2 + threadIdx.x] = sum;
    }

    if (threadIdx.x < 2)
    {
        const int leftx = x - 2;
        int4 sum;
        sum =           getElem<int4>(srcData, srcStep, y - 2, leftx);
        sum = sum + 4 * getElem<int4>(srcData, srcStep, y - 1, leftx);
        sum = sum + 6 * getElem<int4>(srcData, srcStep, y,     leftx);
        sum = sum + 4 * getElem<int4>(srcData, srcStep, y + 1, leftx);
        sum = sum +     getElem<int4>(srcData, srcStep, y + 2, leftx);
        smem[threadIdx.x] = sum;
    }

    if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
    {
        const int rightx = x + 2;
        int4 sum;
        sum =           getElem<int4>(srcData, srcStep, y - 2, rightx);
        sum = sum + 4 * getElem<int4>(srcData, srcStep, y - 1, rightx);
        sum = sum + 6 * getElem<int4>(srcData, srcStep, y,     rightx);
        sum = sum + 4 * getElem<int4>(srcData, srcStep, y + 1, rightx);
        sum = sum +     getElem<int4>(srcData, srcStep, y + 2, rightx);
        smem[4 + threadIdx.x] = sum;
    }

    __syncthreads();

    {
        const int tid = threadIdx.x;
        int4 sum;
        sum =           smem[2 + tid - 2];
        sum = sum + 4 * smem[2 + tid - 1];
        sum = sum + 6 * smem[2 + tid    ];
        sum = sum + 4 * smem[2 + tid + 1];
        sum = sum +     smem[2 + tid + 2];
        
        if (x < dstCols)
            getRowPtr<int4>(dstData, dstStep, y)[x] = roundCastShift6ToInt4(sum);
    }
}

__global__ void pyrUp16SC4To16SC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep)
{
    __shared__ int4 smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    {
        int4 sum;
        sum =       1 * getElem<short4>(srcData, srcStep, y - 2, x);
        sum = sum + 4 * getElem<short4>(srcData, srcStep, y - 1, x);
        sum = sum + 6 * getElem<short4>(srcData, srcStep, y,     x);
        sum = sum + 4 * getElem<short4>(srcData, srcStep, y + 1, x);
        sum = sum + 1 * getElem<short4>(srcData, srcStep, y + 2, x);
        smem[2 + threadIdx.x] = sum;
    }

    if (threadIdx.x < 2)
    {
        const int leftx = x - 2;
        int4 sum;
        sum =       1 * getElem<short4>(srcData, srcStep, y - 2, leftx);
        sum = sum + 4 * getElem<short4>(srcData, srcStep, y - 1, leftx);
        sum = sum + 6 * getElem<short4>(srcData, srcStep, y,     leftx);
        sum = sum + 4 * getElem<short4>(srcData, srcStep, y + 1, leftx);
        sum = sum + 1 * getElem<short4>(srcData, srcStep, y + 2, leftx);
        smem[threadIdx.x] = sum;
    }

    if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
    {
        const int rightx = x + 2;
        int4 sum;
        sum =       1 * getElem<short4>(srcData, srcStep, y - 2, rightx);
        sum = sum + 4 * getElem<short4>(srcData, srcStep, y - 1, rightx);
        sum = sum + 6 * getElem<short4>(srcData, srcStep, y,     rightx);
        sum = sum + 4 * getElem<short4>(srcData, srcStep, y + 1, rightx);
        sum = sum + 1 * getElem<short4>(srcData, srcStep, y + 2, rightx);
        smem[4 + threadIdx.x] = sum;
    }

    __syncthreads();

    {
        const int tid = threadIdx.x;
        int4 sum;
        sum =           smem[2 + tid - 2];
        sum = sum + 4 * smem[2 + tid - 1];
        sum = sum + 6 * smem[2 + tid    ];
        sum = sum + 4 * smem[2 + tid + 1];
        sum = sum +     smem[2 + tid + 2];
        
        if (x < dstCols)
            getRowPtr<short4>(dstData, dstStep, y)[x] = roundCastShift6ToShort4(sum);
    }
}

void pyramidDownPad16SC1To32SC1(cv::gpu::GpuMat& padSrc, cv::gpu::GpuMat& padDst, cv::Size padDstSize, bool horiWrap)
{
    CV_Assert(padSrc.data && padSrc.type() == CV_16SC1); 

    cv::gpu::GpuMat src(padSrc, cv::Rect(2, 2, padSrc.cols - 4, padSrc.rows - 4));
    if (padDstSize == cv::Size())
    {
        padDstSize.width = ((src.cols + 1) >> 1) + 4;
        padDstSize.height = ((src.rows + 1) >> 1) + 4;
    }
    padDst.create(padDstSize.height, padDstSize.width, CV_32SC1);
    cv::gpu::GpuMat dst(padDst, cv::Rect(2, 2, padDst.cols - 4, padDst.rows - 4));

    if (horiWrap)
    {
        BrdRowWrap cb(src.cols);
        padLeftRight<BrdRowWrap, short><<<cv::gpu::divUp(src.rows, 256), 256>>>(src.data, src.rows, src.cols, src.step, cb);
        cudaSafeCall(cudaGetLastError());
    }
    else
    {
        BrdRowReflect101 cb(src.cols);
        padLeftRight<BrdRowReflect101, short><<<cv::gpu::divUp(src.rows, 256), 256>>>(src.data, src.rows, src.cols, src.step, cb);
        cudaSafeCall(cudaGetLastError());
    }

    BrdColReflect101 rb(src.rows);    
    padTopBottom<BrdColReflect101, short><<<cv::gpu::divUp(padSrc.cols, 256), 256>>>(padSrc.data + 2 * padSrc.step, src.rows, padSrc.cols, padSrc.step, rb);
    cudaSafeCall(cudaGetLastError());

    const dim3 block(PYR_DOWN_BLOCK_SIZE);
    const dim3 grid(cv::gpu::divUp(src.cols, block.x), dst.rows);
    pyrDown16SC1To32SC1<<<grid, block>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidDownPad16SC1To16SC1(cv::gpu::GpuMat& padSrc, cv::gpu::GpuMat& padDst, cv::Size padDstSize, bool horiWrap)
{
    CV_Assert(padSrc.data && padSrc.type() == CV_16SC1); 

    cv::gpu::GpuMat src(padSrc, cv::Rect(2, 2, padSrc.cols - 4, padSrc.rows - 4));
    if (padDstSize == cv::Size())
    {
        padDstSize.width = ((src.cols + 1) >> 1) + 4;
        padDstSize.height = ((src.rows + 1) >> 1) + 4;
    }
    padDst.create(padDstSize.height, padDstSize.width, CV_16SC1);
    cv::gpu::GpuMat dst(padDst, cv::Rect(2, 2, padDst.cols - 4, padDst.rows - 4));

    if (horiWrap)
    {
        BrdRowWrap cb(src.cols);
        padLeftRight<BrdRowWrap, short><<<cv::gpu::divUp(src.rows, 256), 256>>>(src.data, src.rows, src.cols, src.step, cb);
        cudaSafeCall(cudaGetLastError());
    }
    else
    {
        BrdRowReflect101 cb(src.cols);
        padLeftRight<BrdRowReflect101, short><<<cv::gpu::divUp(src.rows, 256), 256>>>(src.data, src.rows, src.cols, src.step, cb);
        cudaSafeCall(cudaGetLastError());
    }

    BrdColReflect101 rb(src.rows);    
    padTopBottom<BrdColReflect101, short><<<cv::gpu::divUp(padSrc.cols, 256), 256>>>(padSrc.data + 2 * padSrc.step, src.rows, padSrc.cols, padSrc.step, rb);
    cudaSafeCall(cudaGetLastError());

    const dim3 block(PYR_DOWN_BLOCK_SIZE);
    const dim3 grid(cv::gpu::divUp(src.cols, block.x), dst.rows);
    pyrDown16SC1To16SC1<<<grid, block>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidDownPad16SC4To32SC4(cv::gpu::GpuMat& padSrc, cv::gpu::GpuMat& padDst, cv::Size padDstSize, bool horiWrap)
{
    CV_Assert(padSrc.data && padSrc.type() == CV_16SC4); 

    cv::gpu::GpuMat src(padSrc, cv::Rect(2, 2, padSrc.cols - 4, padSrc.rows - 4));
    if (padDstSize == cv::Size())
    {
        padDstSize.width = ((src.cols + 1) >> 1) + 4;
        padDstSize.height = ((src.rows + 1) >> 1) + 4;
    }
    padDst.create(padDstSize.height, padDstSize.width, CV_32SC4);
    cv::gpu::GpuMat dst(padDst, cv::Rect(2, 2, padDst.cols - 4, padDst.rows - 4));

    if (horiWrap)
    {
        BrdRowWrap cb(src.cols);
        padLeftRight<BrdRowWrap, short4><<<cv::gpu::divUp(src.rows, 256), 256>>>(src.data, src.rows, src.cols, src.step, cb);
        cudaSafeCall(cudaGetLastError());
    }
    else
    {
        BrdRowReflect101 cb(src.cols);
        padLeftRight<BrdRowReflect101, short4><<<cv::gpu::divUp(src.rows, 256), 256>>>(src.data, src.rows, src.cols, src.step, cb);
        cudaSafeCall(cudaGetLastError());
    }

    BrdColReflect101 rb(src.rows);    
    padTopBottom<BrdColReflect101, short4><<<cv::gpu::divUp(padSrc.cols, 256), 256>>>(padSrc.data + 2 * padSrc.step, src.rows, padSrc.cols, padSrc.step, rb);
    cudaSafeCall(cudaGetLastError());

    const dim3 block(PYR_DOWN_BLOCK_SIZE);
    const dim3 grid(cv::gpu::divUp(src.cols, block.x), dst.rows);
    pyrDown16SC4To32SC4<<<grid, block>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

// INFO: Due to different boundary treatment, pyramidUp functions in this file
// produce different boundary result than that in other file.
void pyramidUpPad32SC4To32SC4(const cv::gpu::GpuMat& padSrc, cv::gpu::GpuMat& padDst, cv::Size padDstSize, bool horiWrap)
{
    CV_Assert(padSrc.data && padSrc.type() == CV_32SC4);

    cv::gpu::GpuMat src(padSrc, cv::Rect(2, 2, padSrc.cols - 4, padSrc.rows - 4));
    if (padDstSize == cv::Size())
    {
        padDstSize.width = (src.cols << 1) + 4;
        padDstSize.height = (src.rows << 1) + 4;
    }
    cv::gpu::GpuMat padTmp(padDstSize, CV_32SC4);
    padTmp.setTo(0);
    cv::gpu::GpuMat tmp(padTmp, cv::Rect(2, 2, padTmp.cols - 4, padTmp.rows - 4));
    padDst.create(padDstSize, CV_32SC4);
    cv::gpu::GpuMat dst(padDst, cv::Rect(2, 2, padDst.cols - 4, padDst.rows - 4));
    
    dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(src.cols, block.x), cv::gpu::divUp(src.rows, block.y));
    expand<int4><<<grid, block>>>(src.data, src.rows, src.cols, src.step, tmp.data, tmp.rows, tmp.cols, tmp.step);
    cudaSafeCall(cudaGetLastError());

    if (horiWrap)
    {
        BrdRowWrap cb(tmp.cols);
        padLeftRight<BrdRowWrap, int4><<<cv::gpu::divUp(tmp.rows, 256), 256>>>(tmp.data, tmp.rows, tmp.cols, tmp.step, cb);
        cudaSafeCall(cudaGetLastError());
    }
    else
    {
        BrdRowReflect101 cb(tmp.cols);
        padLeftRight<BrdRowReflect101, int4><<<cv::gpu::divUp(tmp.rows, 256), 256>>>(tmp.data, tmp.rows, tmp.cols, tmp.step, cb);
        cudaSafeCall(cudaGetLastError());
    }

    BrdColReflect101 rb(tmp.rows);    
    padTopBottom<BrdColReflect101, int4><<<cv::gpu::divUp(padTmp.cols, 256), 256>>>(padTmp.data + 2 * padTmp.step, tmp.rows, padTmp.cols, padTmp.step, rb);
    cudaSafeCall(cudaGetLastError());

    block = dim3(PYR_DOWN_BLOCK_SIZE);
    grid = dim3(cv::gpu::divUp(dst.cols, block.x), dst.rows);
    pyrUp32SC4To32SC4<<<grid, block>>>(tmp.data, tmp.rows, tmp.cols, tmp.step, dst.data, dst.rows, dst.cols, dst.step);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidUpPad16SC4To16SC4(const cv::gpu::GpuMat& padSrc, cv::gpu::GpuMat& padDst, cv::Size padDstSize, bool horiWrap)
{
    CV_Assert(padSrc.data && padSrc.type() == CV_16SC4);

    cv::gpu::GpuMat src(padSrc, cv::Rect(2, 2, padSrc.cols - 4, padSrc.rows - 4));
    if (padDstSize == cv::Size())
    {
        padDstSize.width = (src.cols << 1) + 4;
        padDstSize.height = (src.rows << 1) + 4;
    }
    cv::gpu::GpuMat padTmp(padDstSize, CV_16SC4);
    padTmp.setTo(0);
    cv::gpu::GpuMat tmp(padTmp, cv::Rect(2, 2, padTmp.cols - 4, padTmp.rows - 4));
    padDst.create(padDstSize, CV_16SC4);
    cv::gpu::GpuMat dst(padDst, cv::Rect(2, 2, padDst.cols - 4, padDst.rows - 4));
    
    dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(src.cols, block.x), cv::gpu::divUp(src.rows, block.y));
    expand<short4><<<grid, block>>>(src.data, src.rows, src.cols, src.step, tmp.data, tmp.rows, tmp.cols, tmp.step);
    cudaSafeCall(cudaGetLastError());

    if (horiWrap)
    {
        BrdRowWrap cb(tmp.cols);
        padLeftRight<BrdRowWrap, short4><<<cv::gpu::divUp(tmp.rows, 256), 256>>>(tmp.data, tmp.rows, tmp.cols, tmp.step, cb);
        cudaSafeCall(cudaGetLastError());
    }
    else
    {
        BrdRowReflect101 cb(tmp.cols);
        padLeftRight<BrdRowReflect101, short4><<<cv::gpu::divUp(tmp.rows, 256), 256>>>(tmp.data, tmp.rows, tmp.cols, tmp.step, cb);
        cudaSafeCall(cudaGetLastError());
    }

    BrdColReflect101 rb(tmp.rows);    
    padTopBottom<BrdColReflect101, short4><<<cv::gpu::divUp(padTmp.cols, 256), 256>>>(padTmp.data + 2 * padTmp.step, tmp.rows, padTmp.cols, padTmp.step, rb);
    cudaSafeCall(cudaGetLastError());

    //padTmp.copyTo(padDst);
    //return;

    block = dim3(PYR_DOWN_BLOCK_SIZE);
    grid = dim3(cv::gpu::divUp(dst.cols, block.x), dst.rows);
    pyrUp16SC4To16SC4<<<grid, block>>>(tmp.data, tmp.rows, tmp.cols, tmp.step, dst.data, dst.rows, dst.cols, dst.step);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}