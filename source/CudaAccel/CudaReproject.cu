#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

texture<uchar4, 2> srcTexture;
texture<float, 2> xmapTexture, ymapTexture, weightTexture;

template<typename DstElemType>
__global__ void reprojectNearestNeighborKernel(unsigned char* dstData, 
    int dstWidth, int dstHeight, int dstStep, int srcWidth, int srcHeight)
{
    int dstx = threadIdx.x + blockIdx.x * blockDim.x;
    int dsty = threadIdx.y + blockIdx.y * blockDim.y;

    int srcx = tex2D(xmapTexture, dstx, dsty);
    int srcy = tex2D(ymapTexture, dstx, dsty);

    //unsigned char* ptrDst = dstData + dsty * dstStep + dstx * 4;
    DstElemType* ptrDst = (DstElemType*)(dstData + dsty * dstStep) + dstx * 4;
    if (srcx < 0 || srcx >= srcWidth || srcy < 0 || srcy >= srcHeight)
        ptrDst[3] = ptrDst[2] = ptrDst[1] = ptrDst[0] = 0;
    else
    {
        uchar4 val = tex2D(srcTexture, srcx, srcy);
        ptrDst[0] = val.x;
        ptrDst[1] = val.y;
        ptrDst[2] = val.z;
        ptrDst[3] = val.w;
    }
}

const int BILINEAR_INTER_SHIFT = 10;
const int BILINEAR_INTER_BACK_SHIFT = BILINEAR_INTER_SHIFT * 2;
const int BILINEAR_UNIT = 1 << BILINEAR_INTER_SHIFT;

template<typename DstElemType>
__global__ void reprojectLinearKernel(unsigned char* dstData, 
    int dstWidth, int dstHeight, int dstStep, int srcWidth, int srcHeight)
{
    int dstx = threadIdx.x + blockIdx.x * blockDim.x;
    int dsty = threadIdx.y + blockIdx.y * blockDim.y;

    float srcx = tex2D(xmapTexture, dstx, dsty);
    float srcy = tex2D(ymapTexture, dstx, dsty);

    //unsigned char* ptrDst = dstData + dsty * dstStep + dstx * 4;
    DstElemType* ptrDst = (DstElemType*)(dstData + dsty * dstStep) + dstx * 4;
    if (srcx < 0 || srcx >= srcWidth || srcy < 0 || srcy >= srcHeight)
        ptrDst[3] = ptrDst[2] = ptrDst[1] = ptrDst[0] = 0;
    else
    {
        int x0 = srcx, y0 = srcy;
        int x1 = x0 + (x0 < srcWidth - 1), y1 = y0 + (y0 < srcHeight - 1);
        int deltax0 = (srcx - x0) * BILINEAR_UNIT, deltax1 = BILINEAR_UNIT - deltax0;
        int deltay0 = (srcy - y0) * BILINEAR_UNIT, deltay1 = BILINEAR_UNIT - deltay0;
        int b = 0, g = 0, r = 0, w = 0;
        uchar4 val;
        
        val = tex2D(srcTexture, x0, y0);
        w = deltax1 * deltay1;
        b += val.x * w;
        g += val.y * w;
        r += val.z * w;
        
        val = tex2D(srcTexture, x1, y0);
        w = deltax0 * deltay1;
        b += val.x * w;
        g += val.y * w;
        r += val.z * w;

        val = tex2D(srcTexture, x0, y1);
        w = deltax1 * deltay0;
        b += val.x * w;
        g += val.y * w;
        r += val.z * w;

        val = tex2D(srcTexture, x1, y1);
        w = deltax0 * deltay0;
        b += val.x * w;
        g += val.y * w;
        r += val.z * w;

        ptrDst[0] = b >> BILINEAR_INTER_BACK_SHIFT;
        ptrDst[1] = g >> BILINEAR_INTER_BACK_SHIFT;
        ptrDst[2] = r >> BILINEAR_INTER_BACK_SHIFT;
        ptrDst[3] = 0;
    }
}

__device__ __forceinline__ unsigned char bicubic(const unsigned char rgb[4], const float w[4]) 
{
	int res = (int)(rgb[0] * w[0] + rgb[1] * w[1] + rgb[2] * w[2] + rgb[3] * w[3] + 0.5);
	res = res > 255 ? 255 : res;
	res = res < 0 ? 0 : res;
	return (unsigned char)res;
}

__device__ __forceinline__ void calcWeight(float deta, float weight[4])
{
	weight[3] = (deta * deta * (-1.0F + deta));								
	weight[2] = (deta * (1.0F + deta * (1.0F - deta)));						
	weight[1] = (1.0F + deta * deta * (-2.0F + deta)) ;		
	weight[0] = (-deta * (1.0F + deta * (-2.0F + deta))) ;  
}

template<typename DstElemType>
__device__ __forceinline__ void resampling(int width, int height, 
	float x, float y, DstElemType result[4]) 
{
	int x2 = (int)x;
	int y2 = (int)y;
	int nx[4];
	int ny[4];

	for (int i = 0; i < 4;++i)
	{
		nx[i] = (x2 - 1 + i);
		ny[i] = (y2 - 1 + i);
		if (nx[i] < 0) nx[i] = 0;
		if (nx[i] > width - 1) nx[i] = width - 1;
		if (ny[i] < 0) ny[i] = 0;
		if (ny[i] > height - 1) ny[i] = height - 1;
	}

	float u = (x - nx[1]);
	float v = (y - ny[1]);
	//u,v vertical while /100 horizontal
    float tweight1[4], tweight2[4];
	calcWeight(u, tweight1);//weight
	calcWeight(v, tweight2);//weight

    uchar4 val[4][4];
    for (int j = 0; j < 4; j++)
    {
        for (int i = 0; i < 4; i++)
            val[j][i] = tex2D(srcTexture, nx[i], ny[j]);
    }
    unsigned char* ptrVal = &val[0][0].x;
    unsigned char temp0[4], temp1[4];
	for (int k = 0; k < 3; ++k)
	{
		for (int j = 0; j < 4; j++)
		{
			// 按行去每个通道
			for (int i = 0; i < 4; i++)
			{
				temp0[i] = ptrVal[j * 16 + i * 4 + k];
			}
			//4*4区域的三个通道
			temp1[j] = bicubic(temp0, tweight1);
		}
		result[k] = bicubic(temp1, tweight2);
	}
    result[3] = 0;
}

template<typename DstElemType>
__global__ void reprojectCubicKernel(unsigned char* dstData, 
    int dstWidth, int dstHeight, int dstStep, int srcWidth, int srcHeight)
{
    int dstx = threadIdx.x + blockIdx.x * blockDim.x;
    int dsty = threadIdx.y + blockIdx.y * blockDim.y;

    float srcx = tex2D(xmapTexture, dstx, dsty);
    float srcy = tex2D(ymapTexture, dstx, dsty);

    //unsigned char* ptrDst = dstData + dsty * dstStep + dstx * 4;
    DstElemType* ptrDst = (DstElemType*)(dstData + dsty * dstStep) + dstx * 4;
    if (srcx < 0 || srcx >= srcWidth || srcy < 0 || srcy >= srcHeight)
        ptrDst[3] = ptrDst[2] = ptrDst[1] = ptrDst[0] = 0;
    else
        resampling(srcWidth, srcHeight, srcx, srcy, ptrDst);
}

__global__ void reprojectWeightedAccumulate(unsigned char* dstData,
    int dstWidth, int dstHeight, int dstStep, int srcWidth, int srcHeight)
{
    int dstx = threadIdx.x + blockIdx.x * blockDim.x;
    int dsty = threadIdx.y + blockIdx.y * blockDim.y;

    float srcx = tex2D(xmapTexture, dstx, dsty);
    float srcy = tex2D(ymapTexture, dstx, dsty);
    
    if (srcx < 0 || srcx >= srcWidth || srcy < 0 || srcy >= srcHeight)
        ;
    else
    {
        float temp[4];
        float w = tex2D(weightTexture, dstx, dsty);
        resampling(srcWidth, srcHeight, srcx, srcy, temp);
        float* ptrDst = (float*)(dstData + dsty * dstStep) + dstx * 4;
        ptrDst[0] += temp[0] * w;
        ptrDst[1] += temp[1] * w;
        ptrDst[2] += temp[2] * w;
        ptrDst[3] = 0;
    }
}

void cudaReproject(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, 
    const cv::cuda::GpuMat& xmap, const cv::cuda::GpuMat& ymap, cv::cuda::Stream& stream)
{
    CV_Assert(src.data && src.type() == CV_8UC4 &&
        xmap.data && xmap.type() == CV_32FC1 && ymap.data && ymap.type() == CV_32FC1 &&
        xmap.size() == ymap.size());

    cv::Size dstSize = xmap.size();
    dst.create(dstSize, CV_8UC4);

    cudaChannelFormatDesc chanDescUchar4 = cudaCreateChannelDesc<uchar4>();
    cudaChannelFormatDesc chanDescFloat = cudaCreateChannelDesc<float>();

    cudaSafeCall(cudaBindTexture2D(NULL, srcTexture, src.data, chanDescUchar4, src.cols, src.rows, src.step));
    cudaSafeCall(cudaBindTexture2D(NULL, xmapTexture, xmap.data, chanDescFloat, xmap.cols, xmap.rows, xmap.step));
    cudaSafeCall(cudaBindTexture2D(NULL, ymapTexture, ymap.data, chanDescFloat, ymap.cols, ymap.rows, ymap.step));

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    dim3 block(16, 16);
    dim3 grid((dstSize.width + block.x - 1) / block.x, (dstSize.height + block.y - 1) / block.y);
    reprojectCubicKernel<unsigned char><<<grid, block, 0, st>>>(dst.data, dstSize.width, dstSize.height, dst.step, src.cols, src.rows);
    cudaSafeCall(cudaGetLastError());

    cudaSafeCall(cudaUnbindTexture(srcTexture));
    cudaSafeCall(cudaUnbindTexture(xmapTexture));
    cudaSafeCall(cudaUnbindTexture(ymapTexture));

    //cudaSafeCall(cudaDeviceSynchronize());
}

void cudaReprojectTo16S(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
    const cv::cuda::GpuMat& xmap, const cv::cuda::GpuMat& ymap, cv::cuda::Stream& stream)
{
    CV_Assert(src.data && src.type() == CV_8UC4 &&
        xmap.data && xmap.type() == CV_32FC1 && ymap.data && ymap.type() == CV_32FC1 &&
        xmap.size() == ymap.size());

    cv::Size dstSize = xmap.size();
    dst.create(dstSize, CV_16SC4);

    cudaChannelFormatDesc chanDescUchar4 = cudaCreateChannelDesc<uchar4>();
    cudaChannelFormatDesc chanDescFloat = cudaCreateChannelDesc<float>();

    cudaSafeCall(cudaBindTexture2D(NULL, srcTexture, src.data, chanDescUchar4, src.cols, src.rows, src.step));
    cudaSafeCall(cudaBindTexture2D(NULL, xmapTexture, xmap.data, chanDescFloat, xmap.cols, xmap.rows, xmap.step));
    cudaSafeCall(cudaBindTexture2D(NULL, ymapTexture, ymap.data, chanDescFloat, ymap.cols, ymap.rows, ymap.step));

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    dim3 block(16, 16);
    dim3 grid((dstSize.width + block.x - 1) / block.x, (dstSize.height + block.y - 1) / block.y);
    reprojectCubicKernel<short><<<grid, block, 0, st>>>(dst.data, dstSize.width, dstSize.height, dst.step, src.cols, src.rows);
    cudaSafeCall(cudaGetLastError());

    cudaSafeCall(cudaUnbindTexture(srcTexture));
    cudaSafeCall(cudaUnbindTexture(xmapTexture));
    cudaSafeCall(cudaUnbindTexture(ymapTexture));

    //cudaSafeCall(cudaDeviceSynchronize());
}

void cudaReprojectWeightedAccumulateTo32F(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
    const cv::cuda::GpuMat& xmap, const cv::cuda::GpuMat& ymap, const cv::cuda::GpuMat& weight,
    cv::cuda::Stream& stream)
{
    CV_Assert(src.data && src.type() == CV_8UC4 &&
        xmap.data && xmap.type() == CV_32FC1 && ymap.data && ymap.type() == CV_32FC1 &&
        weight.data && weight.type() == CV_32FC1 && 
        xmap.size() == ymap.size() && xmap.size() == weight.size());

    cv::Size dstSize = xmap.size();
    dst.create(dstSize, CV_32FC4);

    cudaChannelFormatDesc chanDescUchar4 = cudaCreateChannelDesc<uchar4>();
    cudaChannelFormatDesc chanDescFloat = cudaCreateChannelDesc<float>();

    cudaSafeCall(cudaBindTexture2D(NULL, srcTexture, src.data, chanDescUchar4, src.cols, src.rows, src.step));
    cudaSafeCall(cudaBindTexture2D(NULL, xmapTexture, xmap.data, chanDescFloat, xmap.cols, xmap.rows, xmap.step));
    cudaSafeCall(cudaBindTexture2D(NULL, ymapTexture, ymap.data, chanDescFloat, ymap.cols, ymap.rows, ymap.step));
    cudaSafeCall(cudaBindTexture2D(NULL, weightTexture, weight.data, chanDescFloat, weight.cols, weight.rows, weight.step));

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    dim3 block(16, 16);
    dim3 grid((dstSize.width + block.x - 1) / block.x, (dstSize.height + block.y - 1) / block.y);
    reprojectWeightedAccumulate<<<grid, block, 0, st>>>(dst.data, dstSize.width, dstSize.height, dst.step, src.cols, src.rows);
    cudaSafeCall(cudaGetLastError());

    cudaSafeCall(cudaUnbindTexture(srcTexture));
    cudaSafeCall(cudaUnbindTexture(xmapTexture));
    cudaSafeCall(cudaUnbindTexture(ymapTexture));
    cudaSafeCall(cudaUnbindTexture(weightTexture));
}
