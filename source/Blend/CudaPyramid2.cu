#include "CudaUtil.cuh"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define PYR_DOWN_BLOCK_SIZE 256
#define PYR_UP_BLOCK_WIDTH 16
#define PYR_UP_BLOCK_HEIGHT 16
#define UTIL_BLOCK_WIDTH 32
#define UTIL_BLOCK_HEIGHT 8

template<typename ColWiseReflectType, typename RowWiseReflectType>
__global__ void pyrDown16SC1To32SC1(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep, 
    const ColWiseReflectType rb, const RowWiseReflectType cb)
{
    __shared__ int smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    const int srcy = 2 * y;

    if (srcy >= 2 && srcy < srcRows - 2 && x >= 2 && x < srcCols - 2)
    {
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
    }
    else
    {
        {
            int sum;
            sum =           getElem<short>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col_high(x));
            sum = sum + 4 * getElem<short>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col_high(x));
            sum = sum + 6 * getElem<short>(srcData, srcStep, srcy,                      cb.idx_col_high(x));
            sum = sum + 4 * getElem<short>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col_high(x));
            sum = sum +     getElem<short>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col_high(x));
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int leftx = x - 2;
            int sum;
            sum =           getElem<short>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col(leftx));
            sum = sum + 4 * getElem<short>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col(leftx));
            sum = sum + 6 * getElem<short>(srcData, srcStep, srcy,                      cb.idx_col(leftx));
            sum = sum + 4 * getElem<short>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col(leftx));
            sum = sum +     getElem<short>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col(leftx));
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int rightx = x + 2;
            int sum;
            sum =           getElem<short>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col_high(rightx));
            sum = sum + 4 * getElem<short>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col_high(rightx));
            sum = sum + 6 * getElem<short>(srcData, srcStep, srcy,                      cb.idx_col_high(rightx));
            sum = sum + 4 * getElem<short>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col_high(rightx));
            sum = sum +     getElem<short>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col_high(rightx));
            smem[4 + threadIdx.x] = sum;
        }
    }

    __syncthreads();

    //if (x == 0)
    //{
    //    printf("%d, %d, %d, %d, %d\n", smem[0], smem[1], smem[2], smem[3], smem[4]);
    //}
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

template<typename ColWiseReflectType, typename RowWiseReflectType>
__global__ void pyrDown16SC1To16SC1(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep, 
    const ColWiseReflectType rb, const RowWiseReflectType cb)
{
    __shared__ int smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    const int srcy = 2 * y;

    if (srcy >= 2 && srcy < srcRows - 2 && x >= 2 && x < srcCols - 2)
    {
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
    }
    else
    {
        {
            int sum;
            sum =           getElem<short>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col_high(x));
            sum = sum + 4 * getElem<short>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col_high(x));
            sum = sum + 6 * getElem<short>(srcData, srcStep, srcy,                      cb.idx_col_high(x));
            sum = sum + 4 * getElem<short>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col_high(x));
            sum = sum +     getElem<short>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col_high(x));
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int leftx = x - 2;
            int sum;
            sum =           getElem<short>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col(leftx));
            sum = sum + 4 * getElem<short>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col(leftx));
            sum = sum + 6 * getElem<short>(srcData, srcStep, srcy,                      cb.idx_col(leftx));
            sum = sum + 4 * getElem<short>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col(leftx));
            sum = sum +     getElem<short>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col(leftx));
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int rightx = x + 2;
            int sum;
            sum =           getElem<short>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col_high(rightx));
            sum = sum + 4 * getElem<short>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col_high(rightx));
            sum = sum + 6 * getElem<short>(srcData, srcStep, srcy,                      cb.idx_col_high(rightx));
            sum = sum + 4 * getElem<short>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col_high(rightx));
            sum = sum +     getElem<short>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col_high(rightx));
            smem[4 + threadIdx.x] = sum;
        }
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

template<typename ColWiseReflectType, typename RowWiseReflectType>
__global__ void pyrDown16SC4To32SC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep, 
    const ColWiseReflectType rb, const RowWiseReflectType cb)
{
    __shared__ int4 smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    const int srcy = 2 * y;

    if (srcy >= 2 && srcy < srcRows - 2 && x >= 2 && x < srcCols - 2)
    {
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
    }
    else
    {
        {
            int4 sum;
            sum =       1 * getElem<short4>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col_high(x));
            sum = sum + 4 * getElem<short4>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col_high(x));
            sum = sum + 6 * getElem<short4>(srcData, srcStep, srcy,                      cb.idx_col_high(x));
            sum = sum + 4 * getElem<short4>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col_high(x));
            sum = sum + 1 * getElem<short4>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col_high(x));
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int leftx = x - 2;
            int4 sum;
            sum =       1 * getElem<short4>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col(leftx));
            sum = sum + 4 * getElem<short4>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col(leftx));
            sum = sum + 6 * getElem<short4>(srcData, srcStep, srcy,                      cb.idx_col(leftx));
            sum = sum + 4 * getElem<short4>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col(leftx));
            sum = sum + 1 * getElem<short4>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col(leftx));
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int rightx = x + 2;
            int4 sum;
            sum =       1 * getElem<short4>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col_high(rightx));
            sum = sum + 4 * getElem<short4>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col_high(rightx));
            sum = sum + 6 * getElem<short4>(srcData, srcStep, srcy,                      cb.idx_col_high(rightx));
            sum = sum + 4 * getElem<short4>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col_high(rightx));
            sum = sum + 1 * getElem<short4>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col_high(rightx));
            smem[4 + threadIdx.x] = sum;
        }
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

template<typename ColWiseReflectType, typename RowWiseReflectType>
__global__ void pyrDown16SC4To16SC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep, 
    const ColWiseReflectType rb, const RowWiseReflectType cb)
{
    __shared__ int4 smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    const int srcy = 2 * y;

    if (srcy >= 2 && srcy < srcRows - 2 && x >= 2 && x < srcCols - 2)
    {
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
    }
    else
    {
        {
            int4 sum;
            sum =       1 * getElem<short4>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col_high(x));
            sum = sum + 4 * getElem<short4>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col_high(x));
            sum = sum + 6 * getElem<short4>(srcData, srcStep, srcy,                      cb.idx_col_high(x));
            sum = sum + 4 * getElem<short4>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col_high(x));
            sum = sum + 1 * getElem<short4>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col_high(x));
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int leftx = x - 2;
            int4 sum;
            sum =       1 * getElem<short4>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col(leftx));
            sum = sum + 4 * getElem<short4>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col(leftx));
            sum = sum + 6 * getElem<short4>(srcData, srcStep, srcy,                      cb.idx_col(leftx));
            sum = sum + 4 * getElem<short4>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col(leftx));
            sum = sum + 1 * getElem<short4>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col(leftx));
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int rightx = x + 2;
            int4 sum;
            sum =       1 * getElem<short4>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col_high(rightx));
            sum = sum + 4 * getElem<short4>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col_high(rightx));
            sum = sum + 6 * getElem<short4>(srcData, srcStep, srcy,                      cb.idx_col_high(rightx));
            sum = sum + 4 * getElem<short4>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col_high(rightx));
            sum = sum + 1 * getElem<short4>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col_high(rightx));
            smem[4 + threadIdx.x] = sum;
        }
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
            getRowPtr<short4>(dstData, dstStep, y)[dstx] = roundCastShift8ToShort4(sum);
    }
}

template<typename ColWiseReflectType, typename RowWiseReflectType>
__global__ void pyrUp32SC4To32SC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep, 
    const ColWiseReflectType rb, const RowWiseReflectType cb)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int4 s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 2][PYR_UP_BLOCK_WIDTH / 2 + 2];
    __shared__ int4 s_dstPatch[PYR_UP_BLOCK_HEIGHT + 4][PYR_UP_BLOCK_WIDTH];

    if ((threadIdx.x < PYR_UP_BLOCK_WIDTH / 2 + 2) && (threadIdx.y < PYR_UP_BLOCK_HEIGHT / 2 + 2))
    {
        int srcx = static_cast<int>((blockIdx.x * blockDim.x) / 2 + threadIdx.x) - 1;
        int srcy = static_cast<int>((blockIdx.y * blockDim.y) / 2 + threadIdx.y) - 1;

        srcx = srcx < 0 ? cb.idx_col_low(srcx) : (srcx >= srcCols) ? cb.idx_col_high(srcx) : srcx;
        srcy = srcy < 0 ? rb.idx_row_low(srcy) : (srcy >= srcRows) ? rb.idx_row_high(srcy) : srcy;

        s_srcPatch[threadIdx.y][threadIdx.x] = getElem<int4>(srcData, srcStep, srcy, srcx);
    }

    __syncthreads();

    int4 sum = make_int4(0, 0, 0, 0);

    const int evenFlag = static_cast<int>((threadIdx.x & 1) == 0);
    const int oddFlag  = static_cast<int>((threadIdx.x & 1) != 0);
    const int eveny = ((threadIdx.y & 1) == 0);
    const int tidx = threadIdx.x;

    if (eveny)
    {
        sum =       (evenFlag    ) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx - 2) >> 1)];
        sum = sum + ( oddFlag * 4) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx - 1) >> 1)];
        sum = sum + (evenFlag * 6) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx    ) >> 1)];
        sum = sum + ( oddFlag * 4) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx + 1) >> 1)];
        sum = sum + (evenFlag    ) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx + 2) >> 1)];
    }

    s_dstPatch[2 + threadIdx.y][threadIdx.x] = sum;

    if (threadIdx.y < 2)
    {
        if (eveny)
        {
            sum =       (evenFlag    ) * s_srcPatch[0][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[0][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 6) * s_srcPatch[0][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[0][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag    ) * s_srcPatch[0][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[threadIdx.y][threadIdx.x] = sum;
    }

    if (threadIdx.y > PYR_UP_BLOCK_HEIGHT - 3)
    {
        if (eveny)
        {
            sum =       (evenFlag    ) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 6) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag    ) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[4 + threadIdx.y][threadIdx.x] = sum;
    }

    __syncthreads();

    const int tidy = threadIdx.y;

    sum =           s_dstPatch[2 + tidy - 2][threadIdx.x];
    sum = sum + 4 * s_dstPatch[2 + tidy - 1][threadIdx.x];
    sum = sum + 6 * s_dstPatch[2 + tidy    ][threadIdx.x];
    sum = sum + 4 * s_dstPatch[2 + tidy + 1][threadIdx.x];
    sum = sum +     s_dstPatch[2 + tidy + 2][threadIdx.x];

    if (x < dstCols && y < dstRows)
        getRowPtr<int4>(dstData, dstStep, y)[x] = roundCastShift6ToInt4(sum);
}

template<typename ColWiseReflectType, typename RowWiseReflectType>
__global__ void pyrUp16SC4To16SC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep, 
    const ColWiseReflectType rb, const RowWiseReflectType cb)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ short4 s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 2][PYR_UP_BLOCK_WIDTH / 2 + 2];
    __shared__ int4 s_dstPatch[PYR_UP_BLOCK_HEIGHT + 4][PYR_UP_BLOCK_WIDTH];

    if ((threadIdx.x < PYR_UP_BLOCK_WIDTH / 2 + 2) && (threadIdx.y < PYR_UP_BLOCK_HEIGHT / 2 + 2))
    {
        int srcx = static_cast<int>((blockIdx.x * blockDim.x) / 2 + threadIdx.x) - 1;
        int srcy = static_cast<int>((blockIdx.y * blockDim.y) / 2 + threadIdx.y) - 1;

        srcx = srcx < 0 ? cb.idx_col_low(srcx) : (srcx >= srcCols) ? cb.idx_col_high(srcx) : srcx;
        srcy = srcy < 0 ? rb.idx_row_low(srcy) : (srcy >= srcRows) ? rb.idx_row_high(srcy) : srcy;

        s_srcPatch[threadIdx.y][threadIdx.x] = getElem<short4>(srcData, srcStep, srcy, srcx);
    }

    __syncthreads();

    int4 sum = make_int4(0, 0, 0, 0);

    const int evenFlag = static_cast<int>((threadIdx.x & 1) == 0);
    const int oddFlag  = static_cast<int>((threadIdx.x & 1) != 0);
    const int eveny = ((threadIdx.y & 1) == 0);
    const int tidx = threadIdx.x;

    if (eveny)
    {
        sum =       (evenFlag    ) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx - 2) >> 1)];
        sum = sum + ( oddFlag * 4) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx - 1) >> 1)];
        sum = sum + (evenFlag * 6) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx    ) >> 1)];
        sum = sum + ( oddFlag * 4) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx + 1) >> 1)];
        sum = sum + (evenFlag    ) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx + 2) >> 1)];
    }

    s_dstPatch[2 + threadIdx.y][threadIdx.x] = sum;

    if (threadIdx.y < 2)
    {
        if (eveny)
        {
            sum =       (evenFlag    ) * s_srcPatch[0][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[0][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 6) * s_srcPatch[0][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[0][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag    ) * s_srcPatch[0][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[threadIdx.y][threadIdx.x] = sum;
    }

    if (threadIdx.y > PYR_UP_BLOCK_HEIGHT - 3)
    {
        if (eveny)
        {
            sum =       (evenFlag    ) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 6) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag    ) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[4 + threadIdx.y][threadIdx.x] = sum;
    }

    __syncthreads();

    const int tidy = threadIdx.y;

    sum =           s_dstPatch[2 + tidy - 2][threadIdx.x];
    sum = sum + 4 * s_dstPatch[2 + tidy - 1][threadIdx.x];
    sum = sum + 6 * s_dstPatch[2 + tidy    ][threadIdx.x];
    sum = sum + 4 * s_dstPatch[2 + tidy + 1][threadIdx.x];
    sum = sum +     s_dstPatch[2 + tidy + 2][threadIdx.x];

    if (x < dstCols && y < dstRows)
        getRowPtr<short4>(dstData, dstStep, y)[x] = roundCastShift6ToShort4(sum);
}

__global__ void divide32SC4To16SC4(const unsigned char* srcImageData, int srcImageRows, int srcImageCols, int srcImageStep,
    const unsigned char* srcAlphaData, int srcAlphaRows, int srcAlphaCols, int srcAlphaStep, 
    unsigned char* dstImageData, int dstImageRows, int dstImageCols, int dstImageStep,
    unsigned char* dstAlphaData, int dstAlphaRows, int dstAlphaCols, int dstAlphaStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < srcImageCols && y < srcImageRows)
    {
        int4 srcImageVal = getElem<int4>(srcImageData, srcImageStep, y, x);
        int srcAlphaVal = getElem<int>(srcAlphaData, srcAlphaStep, y, x);
        if (srcAlphaVal)
        {
            getRowPtr<short4>(dstImageData, dstImageStep, y)[x] = ((srcImageVal << 8) - srcImageVal) / srcAlphaVal;
            getRowPtr<short>(dstAlphaData, dstAlphaStep, y)[x] = 256;
        }
        else
        {
            getRowPtr<short4>(dstImageData, dstImageStep, y)[x] = make_short4(0, 0, 0, 0);
            getRowPtr<short>(dstAlphaData, dstAlphaStep, y)[x] = 0;
        }
    }
}

__global__ void accumulate16SC4To32SC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    const unsigned char* weightData, int weightRows, int weightCols, int weightStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < srcCols && y < srcRows)
    {
        getRowPtr<int4>(dstData, dstStep, y)[x] = getRowPtr<int4>(dstData, dstStep, y)[x] + 
            getElem<short>(weightData, weightStep, y, x) * getElem<short4>(srcData, srcStep, y, x);
    }
}

__global__ void normalize32SC4(unsigned char* imageData, int imageRows, int imageCols, int imageStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < imageCols && y < imageRows)
    {
        getRowPtr<int4>(imageData, imageStep, y)[x] = (getElem<int4>(imageData, imageStep, y, x) + make_int4(128, 128, 128, 0)) >> 8;
    }
}

__global__ void accumulate16SC4To32SC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    const unsigned char* weightData, int weightRows, int weightCols, int weightStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep,
    unsigned char* dstWeightData, int dstWeightRows, int dstWeightCols, int dstWeightStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < srcCols && y < srcRows)
    {
        getRowPtr<int4>(dstData, dstStep, y)[x] = getRowPtr<int4>(dstData, dstStep, y)[x] +
            getElem<short>(weightData, weightStep, y, x) * getElem<short4>(srcData, srcStep, y, x);
        getRowPtr<int>(dstWeightData, dstWeightStep, y)[x] = getRowPtr<int>(dstWeightData, dstWeightStep, y)[x] +
            getElem<short>(weightData, weightStep, y, x);
    }
}

__global__ void normalize32SC4(unsigned char* imageData, int imageRows, int imageCols, int imageStep,
    const unsigned char* weightData, int weightRows, int weightCols, int weightStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < imageCols && y < imageRows)
    {
        int w = getElem<int>(weightData, weightStep, y, x);
        if (!w) w++;
        int4 result = getElem<int4>(imageData, imageStep, y, x);
        result.x /= w;
        result.y /= w;
        result.z /= w;
        result.w = 0;
        getRowPtr<int4>(imageData, imageStep, y)[x] = result;
    }
}

__global__ void scaledSet16SC1Mask16SC1(unsigned char* imageData, int imageRows, int imageCols, int imageStep,
    short val, const unsigned char* maskData, int maskStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < imageCols && y < imageRows)
    {
        getRowPtr<short>(imageData, imageStep, y)[x] = getElem<short>(maskData, maskStep, y, x) ? val : 0;
    }
}

__global__ void scaledSet16SC1Mask32SC1(unsigned char* imageData, int imageRows, int imageCols, int imageStep,
    short val, const unsigned char* maskData, int maskStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < imageCols && y < imageRows)
    {
        getRowPtr<short>(imageData, imageStep, y)[x] = getElem<int>(maskData, maskStep, y, x) ? val : 0;
    }
}

__global__ void divide32SC4To16SC4(const unsigned char* srcImageData, int srcImageRows, int srcImageCols, int srcImageStep,
    const unsigned char* srcAlphaData, int srcAlphaRows, int srcAlphaCols, int srcAlphaStep, 
    unsigned char* dstImageData, int dstImageRows, int dstImageCols, int dstImageStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < srcImageCols && y < srcImageRows)
    {
        int4 srcImageVal = getElem<int4>(srcImageData, srcImageStep, y, x);
        int srcAlphaVal = getElem<int>(srcAlphaData, srcAlphaStep, y, x);
        if (srcAlphaVal)
        {
            getRowPtr<short4>(dstImageData, dstImageStep, y)[x] = ((srcImageVal << 8) - srcImageVal) / srcAlphaVal;
        }
        else
        {
            getRowPtr<short4>(dstImageData, dstImageStep, y)[x] = make_short4(0, 0, 0, 0);
        }
    }
}

__global__ void subtract16SC4(const unsigned char* aData, int aStep, const unsigned char* bData, int bStep,
    unsigned char* cData, int cStep, int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
    {
        getRowPtr<short4>(cData, cStep, y)[x] = getElem<short4>(aData, aStep, y, x) - getElem<short4>(bData, bStep, y, x);
    }
}

__global__ void add32SC4(const unsigned char* aData, int aStep, const unsigned char* bData, int bStep,
    unsigned char* cData, int cStep, int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
    {
        getRowPtr<int4>(cData, cStep, y)[x] = getElem<int4>(aData, aStep, y, x) + getElem<int4>(bData, bStep, y, x);
    }
}

__global__ void accumulate16SC1To32SC1(const unsigned char* srcData, int srcStep, unsigned char* dstData, int dstStep,
    int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
    {
        getRowPtr<int>(dstData, dstStep, y)[x] = getElem<int>(dstData, dstStep, y, x) + getElem<short>(srcData, srcStep, y, x);
    }
}

void pyramidDown16SC1To16SC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::cuda::Stream& stream)
{
    CV_Assert(src.data && src.type() == CV_16SC1); 

    if (dstSize == cv::Size())
    {
        dstSize.width = ((src.cols + 1) >> 1);
        dstSize.height = ((src.rows + 1) >> 1);
    }
    dst.create(dstSize, CV_16SC1);

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(PYR_DOWN_BLOCK_SIZE);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), dst.rows);
    if (horiWrap)
    {
        BrdColReflect101 rb(src.rows);
        BrdRowWrap cb(src.cols);
        pyrDown16SC1To16SC1<<<grid, block, 0, st>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    else
    {
        BrdColReflect101 rb(src.rows);
        BrdRowReflect101 cb(src.cols);
        pyrDown16SC1To16SC1<<<grid, block, 0, st>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidDown16SC1To32SC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::cuda::Stream& stream)
{
    CV_Assert(src.data && src.type() == CV_16SC1); 

    if (dstSize == cv::Size())
    {
        dstSize.width = ((src.cols + 1) >> 1);
        dstSize.height = ((src.rows + 1) >> 1);
    }
    dst.create(dstSize, CV_32SC1);

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(PYR_DOWN_BLOCK_SIZE);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), dst.rows);
    if (horiWrap)
    {
        BrdColReflect101 rb(src.rows);
        BrdRowWrap cb(src.cols);
        pyrDown16SC1To32SC1 <<<grid, block, 0, st>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    else
    {
        BrdColReflect101 rb(src.rows);
        BrdRowReflect101 cb(src.cols);
        pyrDown16SC1To32SC1<<<grid, block, 0, st>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidDown16SC4To32SC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::cuda::Stream& stream)
{
    CV_Assert(src.data && src.type() == CV_16SC4); 

    if (dstSize == cv::Size())
    {
        dstSize.width = ((src.cols + 1) >> 1);
        dstSize.height = ((src.rows + 1) >> 1);
    }
    dst.create(dstSize, CV_32SC4);

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(PYR_DOWN_BLOCK_SIZE);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), dst.rows);
    if (horiWrap)
    {
        BrdColReflect101 rb(src.rows);
        BrdRowWrap cb(src.cols);
        pyrDown16SC4To32SC4<<<grid, block, 0, st>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    else
    {
        BrdColReflect101 rb(src.rows);
        BrdRowReflect101 cb(src.cols);
        pyrDown16SC4To32SC4<<<grid, block, 0, st>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }    
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidDown16SC4To16SC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::cuda::Stream& stream)
{
    CV_Assert(src.data && src.type() == CV_16SC4); 

    if (dstSize == cv::Size())
    {
        dstSize.width = ((src.cols + 1) >> 1);
        dstSize.height = ((src.rows + 1) >> 1);
    }
    dst.create(dstSize, CV_16SC4);

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(PYR_DOWN_BLOCK_SIZE);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), dst.rows);
    if (horiWrap)
    {
        BrdColReflect101 rb(src.rows);
        BrdRowWrap cb(src.cols);
        pyrDown16SC4To16SC4<<<grid, block, 0, st>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    else
    {
        BrdColReflect101 rb(src.rows);
        BrdRowReflect101 cb(src.cols);
        pyrDown16SC4To16SC4<<<grid, block, 0, st>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }    
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void divide32SC4To16SC4(const cv::cuda::GpuMat& srcImage, const cv::cuda::GpuMat& srcAlpha,
    cv::cuda::GpuMat& dstImage, cv::cuda::GpuMat& dstAlpha, cv::cuda::Stream& stream)
{
    CV_Assert(srcImage.data && srcImage.type() == CV_32SC4 &&
        srcAlpha.data && srcAlpha.type() == CV_32SC1 &&
        srcImage.size() == srcAlpha.size());

    dstImage.create(srcImage.size(), CV_16SC4);
    dstAlpha.create(srcAlpha.size(), CV_16SC1);

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(srcImage.cols, block.x), cv::cuda::device::divUp(srcImage.rows, block.y));
    divide32SC4To16SC4<<<grid, block, 0, st>>>(srcImage.data, srcImage.rows, srcImage.cols, srcImage.step,
        srcAlpha.data, srcAlpha.rows, srcAlpha.cols, srcAlpha.step,
        dstImage.data, dstImage.rows, dstImage.cols, dstImage.step,
        dstAlpha.data, dstAlpha.rows, dstAlpha.cols, dstAlpha.step);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidDown16SC4To16SC4(const cv::cuda::GpuMat& srcImage, const cv::cuda::GpuMat& srcAlpha, bool horiWrap, 
    cv::cuda::GpuMat& dstImage, cv::cuda::GpuMat& dstAlpha, cv::cuda::Stream& stream)
{
    CV_Assert(srcImage.data && srcImage.type() == CV_16SC4 &&
        srcAlpha.data && srcAlpha.type() == CV_16SC1 && srcImage.size() == srcAlpha.size());
    cv::cuda::GpuMat dstImage32S, dstAlpha32S;
    pyramidDown16SC4To32SC4(srcImage, dstImage32S, cv::Size(), horiWrap, stream);
    pyramidDown16SC1To32SC1(srcAlpha, dstAlpha32S, cv::Size(), horiWrap, stream);
    divide32SC4To16SC4(dstImage32S, dstAlpha32S, dstImage, dstAlpha, stream);
}

void pyramidUp32SC4To32SC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::cuda::Stream& stream)
{
    CV_Assert(src.data && src.type() == CV_32SC4);
    
    if (dstSize == cv::Size())
    {
        dstSize.width = (src.cols << 1);
        dstSize.height = (src.rows << 1);
    }
    dst.create(dstSize, CV_32SC4);

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(PYR_UP_BLOCK_WIDTH, PYR_UP_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(dst.cols, block.x), cv::cuda::device::divUp(dst.rows, block.y));
    if (horiWrap)
    {
        BrdColReflect101 rb(src.rows);
        BrdRowWrap cb(src.cols);
        pyrUp32SC4To32SC4<<<grid, block, 0, st>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    else
    {
        BrdColReflect101 rb(src.rows);
        BrdRowReflect101 cb(src.cols);
        pyrUp32SC4To32SC4<<<grid, block, 0, st>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidUp16SC4To16SC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::cuda::Stream& stream)
{
    CV_Assert(src.data && src.type() == CV_16SC4);
    
    if (dstSize == cv::Size())
    {
        dstSize.width = (src.cols << 1);
        dstSize.height = (src.rows << 1);
    }
    dst.create(dstSize, CV_16SC4);

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(PYR_UP_BLOCK_WIDTH, PYR_UP_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(dst.cols, block.x), cv::cuda::device::divUp(dst.rows, block.y));
    if (horiWrap)
    {
        BrdColReflect101 rb(src.rows);
        BrdRowWrap cb(src.cols);
        pyrUp16SC4To16SC4<<<grid, block, 0, st>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    else
    {
        BrdColReflect101 rb(src.rows);
        BrdRowReflect101 cb(src.cols);
        pyrUp16SC4To16SC4<<<grid, block, 0, st>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void accumulate16SC4To32SC4(const cv::cuda::GpuMat& src, const cv::cuda::GpuMat& weight, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream)
{
    CV_Assert(src.data && src.type() == CV_16SC4 &&
        weight.data && weight.type() == CV_16SC1 &&
        dst.data && dst.type() == CV_32SC4 &&
        src.size() == weight.size() && src.size() == dst.size());

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), cv::cuda::device::divUp(src.rows, block.y));
    accumulate16SC4To32SC4<<<grid, block, 0, st>>>(src.data, src.rows, src.cols, src.step, 
        weight.data, weight.rows, weight.cols, weight.step,
        dst.data, dst.rows, dst.cols, dst.step);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void normalize32SC4(cv::cuda::GpuMat& image, cv::cuda::Stream& stream)
{
    CV_Assert(image.data && image.type() == CV_32SC4);
    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(image.cols, block.x), cv::cuda::device::divUp(image.rows, block.y));
    normalize32SC4<<<grid, block, 0, st>>>(image.data, image.rows, image.cols, image.step);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void accumulate16SC4To32SC4(const cv::cuda::GpuMat& src, const cv::cuda::GpuMat& weight, 
    cv::cuda::GpuMat& dst, cv::cuda::GpuMat& dstWeight, cv::cuda::Stream& stream)
{
    CV_Assert(src.data && src.type() == CV_16SC4 &&
        weight.data && weight.type() == CV_16SC1 &&
        dst.data && dst.type() == CV_32SC4 &&
        dstWeight.data && dstWeight.type() == CV_32SC1 &&
        src.size() == weight.size() && src.size() == dst.size() && src.size() == dstWeight.size());

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), cv::cuda::device::divUp(src.rows, block.y));
    accumulate16SC4To32SC4<<<grid, block, 0, st>>>(src.data, src.rows, src.cols, src.step,
        weight.data, weight.rows, weight.cols, weight.step,
        dst.data, dst.rows, dst.cols, dst.step,
        dstWeight.data, dstWeight.rows, dstWeight.cols, dstWeight.step);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void normalize32SC4(cv::cuda::GpuMat& image, const cv::cuda::GpuMat& weight, cv::cuda::Stream& stream)
{
    CV_Assert(image.data && image.type() == CV_32SC4 &&
        weight.data && weight.type() == CV_32SC1 && image.size() == weight.size());
    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(image.cols, block.x), cv::cuda::device::divUp(image.rows, block.y));
    normalize32SC4<<<grid, block, 0, st>>>(image.data, image.rows, image.cols, image.step,
        weight.data, weight.rows, weight.cols, weight.step);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void scaledSet16SC1Mask16SC1(cv::cuda::GpuMat& image, short val, const cv::cuda::GpuMat& mask, cv::cuda::Stream& stream)
{
    CV_Assert(image.data && image.type() == CV_16SC1 &&
        mask.data && mask.type() == CV_16SC1 && image.size() == mask.size());
    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(image.cols, block.x), cv::cuda::device::divUp(image.rows, block.y));
    scaledSet16SC1Mask16SC1<<<grid, block, 0, st>>>(image.data, image.rows, image.cols, image.step, val, mask.data, mask.step);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void scaledSet16SC1Mask32SC1(cv::cuda::GpuMat& image, short val, const cv::cuda::GpuMat& mask, cv::cuda::Stream& stream)
{
    CV_Assert(image.data && image.type() == CV_16SC1 &&
        mask.data && mask.type() == CV_32SC1 && image.size() == mask.size());
    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(image.cols, block.x), cv::cuda::device::divUp(image.rows, block.y));
    scaledSet16SC1Mask32SC1<<<grid, block, 0, st>>>(image.data, image.rows, image.cols, image.step, val, mask.data, mask.step);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void divide32SC4To16SC4(const cv::cuda::GpuMat& srcImage, const cv::cuda::GpuMat& srcAlpha, cv::cuda::GpuMat& dstImage, cv::cuda::Stream& stream)
{
    CV_Assert(srcImage.data && srcImage.type() == CV_32SC4 &&
        srcAlpha.data && srcAlpha.type() == CV_32SC1 &&
        srcImage.size() == srcAlpha.size());

    dstImage.create(srcImage.size(), CV_16SC4);

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(32, 8);
    const dim3 grid(cv::cuda::device::divUp(srcImage.cols, block.x), cv::cuda::device::divUp(srcImage.rows, block.y));
    divide32SC4To16SC4<<<grid, block, 0, st>>>(srcImage.data, srcImage.rows, srcImage.cols, srcImage.step,
        srcAlpha.data, srcAlpha.rows, srcAlpha.cols, srcAlpha.step,
        dstImage.data, dstImage.rows, dstImage.cols, dstImage.step);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void subtract16SC4(const cv::cuda::GpuMat& a, const cv::cuda::GpuMat& b, cv::cuda::GpuMat& c, cv::cuda::Stream& stream)
{
    CV_Assert(a.data && a.type() == CV_16SC4 &&
        b.data && b.type() == CV_16SC4 && a.size() == b.size());

    c.create(a.size(), CV_16SC4);

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(a.cols, block.x), cv::cuda::device::divUp(a.rows, block.y));
    subtract16SC4<<<grid, block, 0, st>>>(a.data, a.step, b.data, b.step, c.data, c.step, a.rows, a.cols);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void add32SC4(const cv::cuda::GpuMat& a, const cv::cuda::GpuMat& b, cv::cuda::GpuMat& c, cv::cuda::Stream& stream)
{
    CV_Assert(a.data && a.type() == CV_32SC4 &&
        b.data && b.type() == CV_32SC4 && a.size() == b.size());

    c.create(a.size(), CV_32SC4);

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(a.cols, block.x), cv::cuda::device::divUp(a.rows, block.y));
    add32SC4<<<grid, block, 0, st>>>(a.data, a.step, b.data, b.step, c.data, c.step, a.rows, a.cols);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void accumulate16SC1To32SC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream)
{
    CV_Assert(src.data && src.type() == CV_16SC1 &&
        dst.data && dst.type() == CV_32SC1 && src.size() == dst.size());

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), cv::cuda::device::divUp(src.rows, block.y));
    accumulate16SC1To32SC1<<<grid, block, 0, st>>>(src.data, src.step, dst.data, dst.step, src.rows, src.cols);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

__global__ void func(const unsigned char* src, int srcStep,
    unsigned char* dst, int dstStep, int rows, int cols)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < cols && y < rows)
    {
        float val = getElem<float>(src, srcStep, y, x);
        val = val * val * val;
        val = val * val;
        val = val - val * val * val;
        getRowPtr<float>(dst, dstStep, y)[x] = val;
    }
}

void func(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{
    CV_Assert(src.data && src.type() == CV_32FC1);
    dst.create(src.size(), src.type());
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), cv::cuda::device::divUp(src.rows, block.y));
    func<<<grid, block>>>(src.data, src.step, dst.data, dst.step, src.rows, src.cols);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}