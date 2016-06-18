#include "CudaUtil.cuh"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda/common.hpp"
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

    *uData = clamp0255(u00 >> ITUR_BT_601_SHIFT);
    *vData = clamp0255(v00 >> ITUR_BT_601_SHIFT);
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

void cvtBGR32ToYUV420P(const cv::cuda::GpuMat& bgr32, cv::cuda::GpuMat& y, cv::cuda::GpuMat& u, cv::cuda::GpuMat& v)
{
    CV_Assert(bgr32.data && bgr32.type() == CV_8UC4 && ((bgr32.rows & 1) == 0) && ((bgr32.cols & 1) == 0));
    int rows = bgr32.rows, cols = bgr32.cols;
    y.create(rows, cols, CV_8UC1);
    u.create(rows / 2, cols / 2, CV_8UC1);
    v.create(rows / 2, cols / 2, CV_8UC1);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(u.cols, block.x), cv::cuda::device::divUp(u.rows, block.y));
    cvtBGR32ToYUV420P<<<grid, block>>>(bgr32.data, bgr32.step, y.data, y.step, u.data, u.step, v.data, v.step, rows, cols);
}

void cvtBGR32ToNV12(const cv::cuda::GpuMat& bgr32, cv::cuda::GpuMat& y, cv::cuda::GpuMat& uv)
{
    CV_Assert(bgr32.data && bgr32.type() == CV_8UC4 && ((bgr32.rows & 1) == 0) && ((bgr32.cols & 1) == 0));
    int rows = bgr32.rows, cols = bgr32.cols;
    y.create(rows, cols, CV_8UC1);
    uv.create(rows / 2, cols , CV_8UC1);
    const dim3 block(UTIL_BLOCK_WIDTH, UTIL_BLOCK_HEIGHT);
    const dim3 grid(cv::cuda::device::divUp(uv.cols / 2, block.x), cv::cuda::device::divUp(uv.rows, block.y));
    cvtBGR32ToNV12<<<grid, block>>>(bgr32.data, bgr32.step, y.data, y.step, uv.data, uv.step, rows, cols);
}