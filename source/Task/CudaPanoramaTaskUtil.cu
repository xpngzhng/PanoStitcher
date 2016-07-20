#include "CudaUtil.cuh"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define UTIL_BLOCK_WIDTH 16
#define UTIL_BLOCK_HEIGHT 16

__global__ void alphaBlend8UC4(unsigned char* data, int step,
    const unsigned char* blendData, int blendStep, int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
    {
        int ofs = x * 4;
        unsigned char* ptr = getRowPtr<unsigned char>(data, step, y) + ofs;
        const unsigned char* ptrBlend = getRowPtr<unsigned char>(blendData, blendStep, y) + ofs;
        if (ptrBlend[3])
        {
            int val = ptrBlend[3];
            int comp = 255 - ptrBlend[3];
            ptr[0] = (comp * ptr[0] + val * ptrBlend[0] + 254) / 255;
            ptr[1] = (comp * ptr[1] + val * ptrBlend[1] + 254) / 255;
            ptr[2] = (comp * ptr[2] + val * ptrBlend[2] + 254) / 255;
        }
    }
}

void alphaBlend8UC4(cv::cuda::GpuMat& target, const cv::cuda::GpuMat& blender)
{
    CV_Assert(target.data && target.type() == CV_8UC4 &&
        blender.data && blender.type() == CV_8UC4 && target.size() == blender.size());
    
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(target.cols, block.x), cv::cuda::device::divUp(target.rows, block.y));
    alphaBlend8UC4<<<grid, block>>>(target.data, target.step, blender.data, blender.step, target.rows, target.cols);
    cudaSafeCall(cudaDeviceSynchronize());
}

// Coefficients for RGB to YUV420p conversion
const int ITUR_BT_601_CRY = 269484;
const int ITUR_BT_601_CGY = 528482;
const int ITUR_BT_601_CBY = 102760;
const int ITUR_BT_601_CRU = -155188;
const int ITUR_BT_601_CGU = -305135;
const int ITUR_BT_601_CBU = 460324;
const int ITUR_BT_601_CGV = -385875;
const int ITUR_BT_601_CBV = -74448;

const int ITUR_BT_601_CY = 1220542;
const int ITUR_BT_601_CUB = 2116026;
const int ITUR_BT_601_CUG = -409993;
const int ITUR_BT_601_CVG = -852492;
const int ITUR_BT_601_CVR = 1673527;
const int ITUR_BT_601_SHIFT = 20;

const int shifted16 = (16 << ITUR_BT_601_SHIFT);
const int halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
const int shifted128 = (128 << ITUR_BT_601_SHIFT);

__device__ __forceinline__ int clamp0255(int val)
{
    return val < 0 ? 0 : (val > 255 ? 255 : val);
}

__device__ void cvtBGRToYUV2x2Block(const unsigned char* bgrTopData, const unsigned char* bgrBotData,
    unsigned char* yTopData, unsigned char* yBotData, unsigned char* uData, unsigned char* vData)
{
    int b00 = bgrTopData[0];      int g00 = bgrTopData[1];      int r00 = bgrTopData[2];
    int b01 = bgrTopData[4];      int g01 = bgrTopData[5];      int r01 = bgrTopData[6];
    int b10 = bgrBotData[0];      int g10 = bgrBotData[1];      int r10 = bgrBotData[2];
    int b11 = bgrBotData[4];      int g11 = bgrBotData[5];      int r11 = bgrBotData[6];
    
    int y00 = ITUR_BT_601_CRY * r00 + ITUR_BT_601_CGY * g00 + ITUR_BT_601_CBY * b00 + halfShift + shifted16;
    int y01 = ITUR_BT_601_CRY * r01 + ITUR_BT_601_CGY * g01 + ITUR_BT_601_CBY * b01 + halfShift + shifted16;
    int y10 = ITUR_BT_601_CRY * r10 + ITUR_BT_601_CGY * g10 + ITUR_BT_601_CBY * b10 + halfShift + shifted16;
    int y11 = ITUR_BT_601_CRY * r11 + ITUR_BT_601_CGY * g11 + ITUR_BT_601_CBY * b11 + halfShift + shifted16;

    yTopData[0] = clamp0255(y00 >> ITUR_BT_601_SHIFT);
    yTopData[1] = clamp0255(y01 >> ITUR_BT_601_SHIFT);
    yBotData[0] = clamp0255(y10 >> ITUR_BT_601_SHIFT);
    yBotData[1] = clamp0255(y11 >> ITUR_BT_601_SHIFT);
    
    int u00 = ITUR_BT_601_CRU * r00 + ITUR_BT_601_CGU * g00 + ITUR_BT_601_CBU * b00 + halfShift + shifted128;
    int v00 = ITUR_BT_601_CBU * r00 + ITUR_BT_601_CGV * g00 + ITUR_BT_601_CBV * b00 + halfShift + shifted128;
    int u10 = ITUR_BT_601_CRU * r10 + ITUR_BT_601_CGU * g10 + ITUR_BT_601_CBU * b10 + halfShift + shifted128;
    int v10 = ITUR_BT_601_CBU * r10 + ITUR_BT_601_CGV * g10 + ITUR_BT_601_CBV * b10 + halfShift + shifted128;

    *uData = clamp0255((u00 + u10) >> (ITUR_BT_601_SHIFT + 1));
    *vData = clamp0255((v00 + v10) >> (ITUR_BT_601_SHIFT + 1));
}

__global__ void cvtBGR32ToYUV420P(const unsigned char* bgrData, int bgrStep,
    unsigned char* yData, int yStep, unsigned char* uData, int uStep, unsigned char* vData, int vStep,
    int yRows, int yCols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int xx = x * 2, yy = y * 2;
    if (xx < yCols && yy < yRows)
    {
        cvtBGRToYUV2x2Block(bgrData + yy * bgrStep + xx * 4, bgrData + (yy + 1) * bgrStep + xx * 4,
            yData + yy * yStep + xx, yData + (yy + 1) * yStep + xx,
            uData + y * uStep + x, vData + y * vStep + x);
    }
}

__global__ void cvtBGR32ToNV12(const unsigned char* bgrData, int bgrStep,
    unsigned char* yData, int yStep, unsigned char* uvData, int uvStep, int yRows, int yCols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int xx = x * 2, yy = y * 2;
    if (xx < yCols && yy < yRows)
    {
        cvtBGRToYUV2x2Block(bgrData + yy * bgrStep + xx * 4, bgrData + (yy + 1) * bgrStep + xx * 4,
            yData + yy * yStep + xx, yData + (yy + 1) * yStep + xx,
            uvData + y * uvStep + xx, uvData + y * uvStep + xx + 1);
    }
}

void cvtBGR32ToYUV420P(const cv::cuda::GpuMat& bgr32, cv::cuda::GpuMat& y, cv::cuda::GpuMat& u, cv::cuda::GpuMat& v,
    cv::cuda::Stream& stream)
{
    CV_Assert(bgr32.data && bgr32.type() == CV_8UC4 && ((bgr32.rows & 1) == 0) && ((bgr32.cols & 1) == 0));
    int rows = bgr32.rows, cols = bgr32.cols;
    y.create(rows, cols, CV_8UC1);
    u.create(rows / 2, cols / 2, CV_8UC1);
    v.create(rows / 2, cols / 2, CV_8UC1);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(u.cols, block.x), cv::cuda::device::divUp(u.rows, block.y));
    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    if (st)
    {
        cvtBGR32ToYUV420P<<<grid, block, 0, st>>>(bgr32.data, bgr32.step, y.data, y.step, u.data, u.step, v.data, v.step, rows, cols);
        return;
    }
    cvtBGR32ToYUV420P<<<grid, block>>>(bgr32.data, bgr32.step, y.data, y.step, u.data, u.step, v.data, v.step, rows, cols);
    cudaSafeCall(cudaDeviceSynchronize());
}

void cvtBGR32ToNV12(const cv::cuda::GpuMat& bgr32, cv::cuda::GpuMat& y, cv::cuda::GpuMat& uv,
    cv::cuda::Stream& stream)
{
    CV_Assert(bgr32.data && bgr32.type() == CV_8UC4 && ((bgr32.rows & 1) == 0) && ((bgr32.cols & 1) == 0));
    int rows = bgr32.rows, cols = bgr32.cols;
    y.create(rows, cols, CV_8UC1);
    uv.create(rows / 2, cols , CV_8UC1);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(uv.cols / 2, block.x), cv::cuda::device::divUp(uv.rows, block.y));
    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    if (st)
    {
        cvtBGR32ToNV12<<<grid, block, 0, st>>>(bgr32.data, bgr32.step, y.data, y.step, uv.data, uv.step, rows, cols);
        return;
    }
    cvtBGR32ToNV12<<<grid, block>>>(bgr32.data, bgr32.step, y.data, y.step, uv.data, uv.step, rows, cols);
    cudaSafeCall(cudaDeviceSynchronize());
}

__device__ void cvtYUV2x2BlockToBGR(const unsigned char* yTopData, const unsigned char* yBotData,
    unsigned char uVal, unsigned char vVal, unsigned char* bgrTopData, unsigned char* bgrBotData)
{
    int u = int(uVal) - 128;
    int v = int(vVal) - 128;

    int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
    int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
    int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

    int y00 = max(0, int(yTopData[0]) - 16) * ITUR_BT_601_CY;
    bgrTopData[2] = clamp0255((y00 + ruv) >> ITUR_BT_601_SHIFT);
    bgrTopData[1] = clamp0255((y00 + guv) >> ITUR_BT_601_SHIFT);
    bgrTopData[0] = clamp0255((y00 + buv) >> ITUR_BT_601_SHIFT);

    int y01 = max(0, int(yTopData[1]) - 16) * ITUR_BT_601_CY;
    bgrTopData[6] = clamp0255((y01 + ruv) >> ITUR_BT_601_SHIFT);
    bgrTopData[5] = clamp0255((y01 + guv) >> ITUR_BT_601_SHIFT);
    bgrTopData[4] = clamp0255((y01 + buv) >> ITUR_BT_601_SHIFT);

    int y10 = max(0, int(yBotData[0]) - 16) * ITUR_BT_601_CY;
    bgrBotData[2] = clamp0255((y10 + ruv) >> ITUR_BT_601_SHIFT);
    bgrBotData[1] = clamp0255((y10 + guv) >> ITUR_BT_601_SHIFT);
    bgrBotData[0] = clamp0255((y10 + buv) >> ITUR_BT_601_SHIFT);

    int y11 = max(0, int(yBotData[1]) - 16) * ITUR_BT_601_CY;
    bgrBotData[6] = clamp0255((y11 + ruv) >> ITUR_BT_601_SHIFT);
    bgrBotData[5] = clamp0255((y11 + guv) >> ITUR_BT_601_SHIFT);
    bgrBotData[4] = clamp0255((y11 + buv) >> ITUR_BT_601_SHIFT);
}

__global__ void cvtYUV420PToBGR32(const unsigned char* yData, int yStep, const unsigned char* uData, int uStep,
    const unsigned char* vData, int vStep, unsigned char* bgrData, int bgrStep, int yRows, int yCols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int xx = x * 2, yy = y * 2;
    if (xx < yCols && yy < yRows)
    {
        cvtYUV2x2BlockToBGR(yData + yy * yStep + xx, yData + (yy + 1) * yStep + xx,
            (uData + y * uStep)[x], (vData + y * vStep)[x],
            bgrData + yy * bgrStep + xx * 4, bgrData + (yy + 1) * bgrStep + xx * 4);
    }
}

__global__ void cvtNV12ToBGR32(const unsigned char* yData, int yStep, const unsigned char* uvData, int uvStep,
    unsigned char* bgrData, int bgrStep, int yRows, int yCols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int xx = x * 2, yy = y * 2;
    if (xx < yCols && yy < yRows)
    {
        cvtYUV2x2BlockToBGR(yData + yy * yStep + xx, yData + (yy + 1) * yStep + xx,
            (uvData + y * uvStep)[xx], (uvData + y * uvStep)[xx + 1],
            bgrData + yy * bgrStep + xx * 4, bgrData + (yy + 1) * bgrStep + xx * 4);
    }
}

void cvtYUV420PToBGR32(const cv::cuda::GpuMat& y, const cv::cuda::GpuMat& u, const cv::cuda::GpuMat& v, cv::cuda::GpuMat& bgr32,
    cv::cuda::Stream& stream)
{
    CV_Assert(y.data && y.type() == CV_8UC1 && u.data && u.type() == CV_8UC1 &&
        v.data && v.type() == CV_8UC1 && y.rows == u.rows * 2 && y.cols == u.cols * 2 &&
        u.size() == v.size());
    bgr32.create(y.size(), CV_8UC4);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(u.cols, block.x), cv::cuda::device::divUp(u.rows, block.y));
    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    if (st)
    {
        cvtYUV420PToBGR32<<<grid, block, 0, st>>>(y.data, y.step, u.data, u.step, v.data, v.step, bgr32.data, bgr32.step, y.rows, y.cols);
        return;
    }
    cvtYUV420PToBGR32<<<grid, block>>>(y.data, y.step, u.data, u.step, v.data, v.step, bgr32.data, bgr32.step, y.rows, y.cols);
    cudaSafeCall(cudaDeviceSynchronize());
}

void cvtNV12ToBGR32(const cv::cuda::GpuMat& y, const cv::cuda::GpuMat& uv, cv::cuda::GpuMat& bgr32,
    cv::cuda::Stream& stream)
{
    CV_Assert(y.data && y.type() == CV_8UC1 && uv.data && uv.type() == CV_8UC1 &&
        y.rows == uv.rows * 2 && y.cols == uv.cols);
    bgr32.create(y.size(), CV_8UC4);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(uv.cols / 2, block.x), cv::cuda::device::divUp(uv.rows, block.y));
    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    if (st)
    {
        cvtNV12ToBGR32<<<grid, block, 0, st>>>(y.data, y.step, uv.data, uv.step, bgr32.data, bgr32.step, y.rows, y.cols);
        return;
    }
    cvtNV12ToBGR32<<<grid, block>>>(y.data, y.step, uv.data, uv.step, bgr32.data, bgr32.step, y.rows, y.cols);
    cudaSafeCall(cudaDeviceSynchronize());
}
