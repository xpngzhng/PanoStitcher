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

#define NUM_1_DEN_16 0.0625F
#define NUM_4_DEN_16 0.25F
#define NUM_6_DEN_16 0.375F

template<typename ColWiseReflectType, typename RowWiseReflectType>
__global__ void pyrDown32FC1(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep,
    const ColWiseReflectType rb, const RowWiseReflectType cb)
{
    __shared__ float smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    const int srcy = 2 * y;

    if (srcy >= 2 && srcy < srcRows - 2 && x >= 2 && x < srcCols - 2)
    {
        {
            float sum;
            sum =       NUM_1_DEN_16 * getElem<float>(srcData, srcStep, srcy - 2, x);
            sum = sum + NUM_4_DEN_16 * getElem<float>(srcData, srcStep, srcy - 1, x);
            sum = sum + NUM_6_DEN_16 * getElem<float>(srcData, srcStep, srcy,     x);
            sum = sum + NUM_4_DEN_16 * getElem<float>(srcData, srcStep, srcy + 1, x);
            sum = sum + NUM_1_DEN_16 * getElem<float>(srcData, srcStep, srcy + 2, x);
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int leftx = x - 2;
            float sum;
            sum =       NUM_1_DEN_16 * getElem<float>(srcData, srcStep, srcy - 2, leftx);
            sum = sum + NUM_4_DEN_16 * getElem<float>(srcData, srcStep, srcy - 1, leftx);
            sum = sum + NUM_6_DEN_16 * getElem<float>(srcData, srcStep, srcy,     leftx);
            sum = sum + NUM_4_DEN_16 * getElem<float>(srcData, srcStep, srcy + 1, leftx);
            sum = sum + NUM_1_DEN_16 * getElem<float>(srcData, srcStep, srcy + 2, leftx);
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int rightx = x + 2;
            float sum;
            sum =       NUM_1_DEN_16 * getElem<float>(srcData, srcStep, srcy - 2, rightx);
            sum = sum + NUM_4_DEN_16 * getElem<float>(srcData, srcStep, srcy - 1, rightx);
            sum = sum + NUM_6_DEN_16 * getElem<float>(srcData, srcStep, srcy,     rightx);
            sum = sum + NUM_4_DEN_16 * getElem<float>(srcData, srcStep, srcy + 1, rightx);
            sum = sum + NUM_1_DEN_16 * getElem<float>(srcData, srcStep, srcy + 2, rightx);
            smem[4 + threadIdx.x] = sum;
        }
    }
    else
    {
        {
            float sum;
            sum =       NUM_1_DEN_16 * getElem<float>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col_high(x));
            sum = sum + NUM_4_DEN_16 * getElem<float>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col_high(x));
            sum = sum + NUM_6_DEN_16 * getElem<float>(srcData, srcStep, srcy,                      cb.idx_col_high(x));
            sum = sum + NUM_4_DEN_16 * getElem<float>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col_high(x));
            sum = sum + NUM_1_DEN_16 * getElem<float>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col_high(x));
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int leftx = x - 2;
            float sum;
            sum =       NUM_1_DEN_16 * getElem<float>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col(leftx));
            sum = sum + NUM_4_DEN_16 * getElem<float>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col(leftx));
            sum = sum + NUM_6_DEN_16 * getElem<float>(srcData, srcStep, srcy,                      cb.idx_col(leftx));
            sum = sum + NUM_4_DEN_16 * getElem<float>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col(leftx));
            sum = sum + NUM_1_DEN_16 * getElem<float>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col(leftx));
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int rightx = x + 2;
            float sum;
            sum =       NUM_1_DEN_16 * getElem<float>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col_high(rightx));
            sum = sum + NUM_4_DEN_16 * getElem<float>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col_high(rightx));
            sum = sum + NUM_6_DEN_16 * getElem<float>(srcData, srcStep, srcy,                      cb.idx_col_high(rightx));
            sum = sum + NUM_4_DEN_16 * getElem<float>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col_high(rightx));
            sum = sum + NUM_1_DEN_16 * getElem<float>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col_high(rightx));
            smem[4 + threadIdx.x] = sum;
        }
    }

    __syncthreads();

    if (threadIdx.x < PYR_DOWN_BLOCK_SIZE / 2)
    {
        const int tid2 = threadIdx.x * 2;
        float sum;
        sum =       NUM_1_DEN_16 * smem[2 + tid2 - 2];
        sum = sum + NUM_4_DEN_16 * smem[2 + tid2 - 1];
        sum = sum + NUM_6_DEN_16 * smem[2 + tid2    ];
        sum = sum + NUM_4_DEN_16 * smem[2 + tid2 + 1];
        sum = sum + NUM_1_DEN_16 * smem[2 + tid2 + 2];

        const int dstx = (blockIdx.x * blockDim.x + tid2) / 2;

        if (dstx < dstCols)
            getRowPtr<float>(dstData, dstStep, y)[dstx] = sum;            
    }
}

template<typename ColWiseReflectType, typename RowWiseReflectType>
__global__ void pyrDown32FC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep,
    const ColWiseReflectType rb, const RowWiseReflectType cb)
{
    __shared__ float4 smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    const int srcy = 2 * y;

    if (srcy >= 2 && srcy < srcRows - 2 && x >= 2 && x < srcCols - 2)
    {
        {
            float4 sum;
            sum =       NUM_1_DEN_16 * getElem<float4>(srcData, srcStep, srcy - 2, x);
            sum = sum + NUM_4_DEN_16 * getElem<float4>(srcData, srcStep, srcy - 1, x);
            sum = sum + NUM_6_DEN_16 * getElem<float4>(srcData, srcStep, srcy,     x);
            sum = sum + NUM_4_DEN_16 * getElem<float4>(srcData, srcStep, srcy + 1, x);
            sum = sum + NUM_1_DEN_16 * getElem<float4>(srcData, srcStep, srcy + 2, x);
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int leftx = x - 2;
            float4 sum;
            sum =       NUM_1_DEN_16 * getElem<float4>(srcData, srcStep, srcy - 2, leftx);
            sum = sum + NUM_4_DEN_16 * getElem<float4>(srcData, srcStep, srcy - 1, leftx);
            sum = sum + NUM_6_DEN_16 * getElem<float4>(srcData, srcStep, srcy,     leftx);
            sum = sum + NUM_4_DEN_16 * getElem<float4>(srcData, srcStep, srcy + 1, leftx);
            sum = sum + NUM_1_DEN_16 * getElem<float4>(srcData, srcStep, srcy + 2, leftx);
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int rightx = x + 2;
            float4 sum;
            sum =       NUM_1_DEN_16 * getElem<float4>(srcData, srcStep, srcy - 2, rightx);
            sum = sum + NUM_4_DEN_16 * getElem<float4>(srcData, srcStep, srcy - 1, rightx);
            sum = sum + NUM_6_DEN_16 * getElem<float4>(srcData, srcStep, srcy,     rightx);
            sum = sum + NUM_4_DEN_16 * getElem<float4>(srcData, srcStep, srcy + 1, rightx);
            sum = sum + NUM_1_DEN_16 * getElem<float4>(srcData, srcStep, srcy + 2, rightx);
            smem[4 + threadIdx.x] = sum;
        }
    }
    else
    {
        {
            float4 sum;
            sum =       NUM_1_DEN_16 * getElem<float4>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col_high(x));
            sum = sum + NUM_4_DEN_16 * getElem<float4>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col_high(x));
            sum = sum + NUM_6_DEN_16 * getElem<float4>(srcData, srcStep, srcy,                      cb.idx_col_high(x));
            sum = sum + NUM_4_DEN_16 * getElem<float4>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col_high(x));
            sum = sum + NUM_1_DEN_16 * getElem<float4>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col_high(x));
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int leftx = x - 2;
            float4 sum;
            sum =       NUM_1_DEN_16 * getElem<float4>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col(leftx));
            sum = sum + NUM_4_DEN_16 * getElem<float4>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col(leftx));
            sum = sum + NUM_6_DEN_16 * getElem<float4>(srcData, srcStep, srcy,                      cb.idx_col(leftx));
            sum = sum + NUM_4_DEN_16 * getElem<float4>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col(leftx));
            sum = sum + NUM_1_DEN_16 * getElem<float4>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col(leftx));
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int rightx = x + 2;
            float4 sum;
            sum =       NUM_1_DEN_16 * getElem<float4>(srcData, srcStep, rb.idx_row_low(srcy - 2),  cb.idx_col_high(rightx));
            sum = sum + NUM_4_DEN_16 * getElem<float4>(srcData, srcStep, rb.idx_row_low(srcy - 1),  cb.idx_col_high(rightx));
            sum = sum + NUM_6_DEN_16 * getElem<float4>(srcData, srcStep, srcy,                      cb.idx_col_high(rightx));
            sum = sum + NUM_4_DEN_16 * getElem<float4>(srcData, srcStep, rb.idx_row_high(srcy + 1), cb.idx_col_high(rightx));
            sum = sum + NUM_1_DEN_16 * getElem<float4>(srcData, srcStep, rb.idx_row_high(srcy + 2), cb.idx_col_high(rightx));
            smem[4 + threadIdx.x] = sum;
        }
    }

    __syncthreads();

    if (threadIdx.x < PYR_DOWN_BLOCK_SIZE / 2)
    {
        const int tid2 = threadIdx.x * 2;
        float4 sum;
        sum =       NUM_1_DEN_16 * smem[2 + tid2 - 2];
        sum = sum + NUM_4_DEN_16 * smem[2 + tid2 - 1];
        sum = sum + NUM_6_DEN_16 * smem[2 + tid2    ];
        sum = sum + NUM_4_DEN_16 * smem[2 + tid2 + 1];
        sum = sum + NUM_1_DEN_16 * smem[2 + tid2 + 2];

        const int dstx = (blockIdx.x * blockDim.x + tid2) / 2;

        if (dstx < dstCols)
            getRowPtr<float4>(dstData, dstStep, y)[dstx] = sum;
    }
}

template<typename ColWiseReflectType, typename RowWiseReflectType>
__global__ void pyrUp32FC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep,
    const ColWiseReflectType rb, const RowWiseReflectType cb)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float4 s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 2][PYR_UP_BLOCK_WIDTH / 2 + 2];
    __shared__ float4 s_dstPatch[PYR_UP_BLOCK_HEIGHT + 4][PYR_UP_BLOCK_WIDTH];

    if ((threadIdx.x < PYR_UP_BLOCK_WIDTH / 2 + 2) && (threadIdx.y < PYR_UP_BLOCK_HEIGHT / 2 + 2))
    {
        int srcx = static_cast<int>((blockIdx.x * blockDim.x) / 2 + threadIdx.x) - 1;
        int srcy = static_cast<int>((blockIdx.y * blockDim.y) / 2 + threadIdx.y) - 1;

        srcx = srcx < 0 ? cb.idx_col_low(srcx) : (srcx >= srcCols) ? cb.idx_col_high(srcx) : srcx;
        srcy = srcy < 0 ? rb.idx_row_low(srcy) : (srcy >= srcRows) ? rb.idx_row_high(srcy) : srcy;

        s_srcPatch[threadIdx.y][threadIdx.x] = getElem<float4>(srcData, srcStep, srcy, srcx);
    }

    __syncthreads();

    float4 sum = make_float4(0, 0, 0, 0);

    const int evenFlag = static_cast<int>((threadIdx.x & 1) == 0);
    const int oddFlag = static_cast<int>((threadIdx.x & 1) != 0);
    const int eveny = ((threadIdx.y & 1) == 0);
    const int tidx = threadIdx.x;

    if (eveny)
    {
        sum =       (evenFlag * NUM_1_DEN_16) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx - 2) >> 1)];
        sum = sum + (oddFlag  * NUM_4_DEN_16) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx - 1) >> 1)];
        sum = sum + (evenFlag * NUM_6_DEN_16) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx)     >> 1)];
        sum = sum + (oddFlag  * NUM_4_DEN_16) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx + 1) >> 1)];
        sum = sum + (evenFlag * NUM_1_DEN_16) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx + 2) >> 1)];
    }

    s_dstPatch[2 + threadIdx.y][threadIdx.x] = sum;

    if (threadIdx.y < 2)
    {
        if (eveny)
        {
            sum =       (evenFlag * NUM_1_DEN_16) * s_srcPatch[0][1 + ((tidx - 2) >> 1)];
            sum = sum + (oddFlag  * NUM_4_DEN_16) * s_srcPatch[0][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * NUM_6_DEN_16) * s_srcPatch[0][1 + ((tidx)     >> 1)];
            sum = sum + (oddFlag  * NUM_4_DEN_16) * s_srcPatch[0][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag * NUM_1_DEN_16) * s_srcPatch[0][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[threadIdx.y][threadIdx.x] = sum;
    }

    if (threadIdx.y > PYR_UP_BLOCK_HEIGHT - 3)
    {
        if (eveny)
        {
            sum =       (evenFlag * NUM_1_DEN_16) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx - 2) >> 1)];
            sum = sum + (oddFlag  * NUM_4_DEN_16) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * NUM_6_DEN_16) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx)     >> 1)];
            sum = sum + (oddFlag  * NUM_4_DEN_16) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag * NUM_1_DEN_16) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[4 + threadIdx.y][threadIdx.x] = sum;
    }

    __syncthreads();

    const int tidy = threadIdx.y;

    sum =       NUM_1_DEN_16 * s_dstPatch[2 + tidy - 2][threadIdx.x];
    sum = sum + NUM_4_DEN_16 * s_dstPatch[2 + tidy - 1][threadIdx.x];
    sum = sum + NUM_6_DEN_16 * s_dstPatch[2 + tidy    ][threadIdx.x];
    sum = sum + NUM_4_DEN_16 * s_dstPatch[2 + tidy + 1][threadIdx.x];
    sum = sum + NUM_1_DEN_16 * s_dstPatch[2 + tidy + 2][threadIdx.x];

    if (x < dstCols && y < dstRows)
        getRowPtr<float4>(dstData, dstStep, y)[x] = 4.0F * sum;
}

__global__ void scaledSet32FC1Mask32FC1(unsigned char* imageData, int imageRows, int imageCols, int imageStep,
    float val, const unsigned char* maskData, int maskStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < imageCols && y < imageRows)
    {
        getRowPtr<float>(imageData, imageStep, y)[x] = getElem<float>(maskData, maskStep, y, x) > 0 ? val : 0;
    }
}

__global__ void divide32FC4(const unsigned char* srcImageData, int srcImageRows, int srcImageCols, int srcImageStep,
    const unsigned char* srcAlphaData, int srcAlphaStep, unsigned char* dstImageData, int dstImageStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < srcImageCols && y < srcImageRows)
    {
        float4 srcImageVal = getElem<float4>(srcImageData, srcImageStep, y, x);
        float srcAlphaVal = getElem<float>(srcAlphaData, srcAlphaStep, y, x);
        if (srcAlphaVal > 0)
        {
            getRowPtr<float4>(dstImageData, dstImageStep, y)[x] = srcImageVal / srcAlphaVal;
        }
        else
        {
            getRowPtr<float4>(dstImageData, dstImageStep, y)[x] = make_float4(0, 0, 0, 0);
        }
    }
}

__global__ void subtract32FC4(const unsigned char* aData, int aStep, const unsigned char* bData, int bStep,
    unsigned char* cData, int cStep, int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
    {
        getRowPtr<float4>(cData, cStep, y)[x] = getElem<float4>(aData, aStep, y, x) - getElem<float4>(bData, bStep, y, x);
    }
}

__global__ void add32FC4(const unsigned char* aData, int aStep, const unsigned char* bData, int bStep,
    unsigned char* cData, int cStep, int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
    {
        getRowPtr<float4>(cData, cStep, y)[x] = getElem<float4>(aData, aStep, y, x) + getElem<float4>(bData, bStep, y, x);
    }
}

__global__ void accumulate32FC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    const unsigned char* weightData, int weightStep, unsigned char* dstData, int dstStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < srcCols && y < srcRows)
    {
        getRowPtr<float4>(dstData, dstStep, y)[x] = getRowPtr<float4>(dstData, dstStep, y)[x] +
            getElem<float>(weightData, weightStep, y, x) * getElem<float4>(srcData, srcStep, y, x);
    }
}

__global__ void accumulate32FC1(const unsigned char* srcData, int srcStep, unsigned char* dstData, int dstStep,
    int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
    {
        getRowPtr<float>(dstData, dstStep, y)[x] = getElem<float>(dstData, dstStep, y, x) + getElem<float>(srcData, srcStep, y, x);
    }
}

__global__ void inverse32FC1(unsigned char* data, int step, int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
    {
        float val = getRowPtr<float>(data, step, y)[x];
        getRowPtr<float>(data, step, y)[x] = val > 0 ? 1.0F / val : 0;
    }
}

__global__ void scale32FC4(unsigned char* imageData, int imageStep,
    const unsigned char* weightData, int weightStep, int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
    {
        getRowPtr<float4>(imageData, imageStep, y)[x] = getRowPtr<float>(weightData, weightStep, y)[x] *
            getRowPtr<float4>(imageData, imageStep, y)[x];
    }
}

__global__ void scale32FC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    const unsigned char* weightData, int weightStep, unsigned char* dstData, int dstStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < srcCols && y < srcRows)
    {
        getRowPtr<float4>(dstData, dstStep, y)[x] = 
            getElem<float>(weightData, weightStep, y, x) * getElem<float4>(srcData, srcStep, y, x);
    }
}

void pyramidDown32FC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap)
{
    CV_Assert(src.data && src.type() == CV_32FC1);

    if (dstSize == cv::Size())
    {
        dstSize.width = ((src.cols + 1) >> 1);
        dstSize.height = ((src.rows + 1) >> 1);
    }
    dst.create(dstSize, CV_32FC1);

    const dim3 block(PYR_DOWN_BLOCK_SIZE);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), dst.rows);
    if (horiWrap)
    {
        BrdColReflect101 rb(src.rows);
        BrdRowWrap cb(src.cols);
        pyrDown32FC1 << <grid, block >> >(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    else
    {
        BrdColReflect101 rb(src.rows);
        BrdRowReflect101 cb(src.cols);
        pyrDown32FC1 << <grid, block >> >(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidDown32FC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap)
{
    CV_Assert(src.data && src.type() == CV_32FC4);

    if (dstSize == cv::Size())
    {
        dstSize.width = ((src.cols + 1) >> 1);
        dstSize.height = ((src.rows + 1) >> 1);
    }
    dst.create(dstSize, CV_32FC4);

    const dim3 block(PYR_DOWN_BLOCK_SIZE);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), dst.rows);
    if (horiWrap)
    {
        BrdColReflect101 rb(src.rows);
        BrdRowWrap cb(src.cols);
        pyrDown32FC4 << <grid, block >> >(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    else
    {
        BrdColReflect101 rb(src.rows);
        BrdRowReflect101 cb(src.cols);
        pyrDown32FC4 << <grid, block >> >(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidUp32FC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap)
{
    CV_Assert(src.data && src.type() == CV_32FC4);

    if (dstSize == cv::Size())
    {
        dstSize.width = (src.cols << 1);
        dstSize.height = (src.rows << 1);
    }
    dst.create(dstSize, CV_32FC4);

    const dim3 block(PYR_UP_BLOCK_WIDTH, PYR_UP_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(dst.cols, block.x), cv::cuda::device::divUp(dst.rows, block.y));
    if (horiWrap)
    {
        BrdColReflect101 rb(src.rows);
        BrdRowWrap cb(src.cols);
        pyrUp32FC4 << <grid, block >> >(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    else
    {
        BrdColReflect101 rb(src.rows);
        BrdRowReflect101 cb(src.cols);
        pyrUp32FC4 << <grid, block >> >(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    }
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void scaledSet32FC1Mask32FC1(cv::cuda::GpuMat& image, float val, const cv::cuda::GpuMat& mask)
{
    CV_Assert(image.data && image.type() == CV_32FC1 &&
        mask.data && mask.type() == CV_32FC1 && image.size() == mask.size());
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(image.cols, block.x), cv::cuda::device::divUp(image.rows, block.y));
    scaledSet32FC1Mask32FC1 << <grid, block >> >(image.data, image.rows, image.cols, image.step, val, mask.data, mask.step);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void divide32FC4(const cv::cuda::GpuMat& srcImage, const cv::cuda::GpuMat& srcAlpha, cv::cuda::GpuMat& dstImage)
{
    CV_Assert(srcImage.data && srcImage.type() == CV_32FC4 &&
        srcAlpha.data && srcAlpha.type() == CV_32FC1 &&
        srcImage.size() == srcAlpha.size());

    dstImage.create(srcImage.size(), CV_32FC4);

    const dim3 block(32, 8);
    const dim3 grid(cv::cuda::device::divUp(srcImage.cols, block.x), cv::cuda::device::divUp(srcImage.rows, block.y));
    divide32FC4 << <grid, block >> >(srcImage.data, srcImage.rows, srcImage.cols, srcImage.step,
        srcAlpha.data, srcAlpha.step, dstImage.data, dstImage.step);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void subtract32FC4(const cv::cuda::GpuMat& a, const cv::cuda::GpuMat& b, cv::cuda::GpuMat& c)
{
    CV_Assert(a.data && a.type() == CV_32FC4 &&
        b.data && b.type() == CV_32FC4 && a.size() == b.size());

    c.create(a.size(), CV_32FC4);

    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(a.cols, block.x), cv::cuda::device::divUp(a.rows, block.y));
    subtract32FC4 << <grid, block >> >(a.data, a.step, b.data, b.step, c.data, c.step, a.rows, a.cols);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void add32FC4(const cv::cuda::GpuMat& a, const cv::cuda::GpuMat& b, cv::cuda::GpuMat& c)
{
    CV_Assert(a.data && a.type() == CV_32FC4 &&
        b.data && b.type() == CV_32FC4 && a.size() == b.size());

    c.create(a.size(), CV_32FC4);

    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(a.cols, block.x), cv::cuda::device::divUp(a.rows, block.y));
    add32FC4 << <grid, block >> >(a.data, a.step, b.data, b.step, c.data, c.step, a.rows, a.cols);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void accumulate32FC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{
    CV_Assert(src.data && src.type() == CV_32FC1 &&
        dst.data && dst.type() == CV_32FC1 && src.size() == dst.size());

    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), cv::cuda::device::divUp(src.rows, block.y));
    accumulate32FC1 << <grid, block >> >(src.data, src.step, dst.data, dst.step, src.rows, src.cols);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void accumulate32FC4(const cv::cuda::GpuMat& src, const cv::cuda::GpuMat& weight, cv::cuda::GpuMat& dst)
{
    CV_Assert(src.data && src.type() == CV_32FC4 &&
        weight.data && weight.type() == CV_32FC1 &&
        dst.data && dst.type() == CV_32FC4 &&
        src.size() == weight.size() && src.size() == dst.size());

    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), cv::cuda::device::divUp(src.rows, block.y));
    accumulate32FC4 << <grid, block >> >(src.data, src.rows, src.cols, src.step,
        weight.data, weight.step, dst.data, dst.step);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void inverse32FC1(cv::cuda::GpuMat& mat)
{
    CV_Assert(mat.data && mat.type() == CV_32FC1);

    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(mat.cols, block.x), cv::cuda::device::divUp(mat.rows, block.y));
    inverse32FC1 << <grid, block >> >(mat.data, mat.step, mat.rows, mat.cols);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void scale32FC4(cv::cuda::GpuMat& image, const cv::cuda::GpuMat& alpha)
{
    CV_Assert(image.data && image.type() == CV_32FC4 &&
        alpha.data && alpha.type() == CV_32FC1 && image.size() == alpha.size());

    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(image.cols, block.x), cv::cuda::device::divUp(image.rows, block.y));
    scale32FC4<<<grid, block>>>(image.data, image.step, alpha.data, alpha.step, image.rows, image.cols);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

void scale32FC4(const cv::cuda::GpuMat& src, const cv::cuda::GpuMat& weight, cv::cuda::GpuMat& dst)
{
    CV_Assert(src.data && src.type() == CV_32FC4 &&
        weight.data && weight.type() == CV_32FC1 &&
        src.size() == weight.size());

    dst.create(src.size(), CV_32FC4);

    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), cv::cuda::device::divUp(src.rows, block.y));
    scale32FC4 << <grid, block >> >(src.data, src.rows, src.cols, src.step,
        weight.data, weight.step, dst.data, dst.step);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}