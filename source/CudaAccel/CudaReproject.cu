#include "Warp/ZReproject.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

typedef double CalcType;

struct CudaRemapParam
{
    enum ImageType
    {
        ImageTypeRectlinear = 0,
        ImageTypeFullFrameFishEye = 1,
        ImageTypeDrumFishEye = 2,
        ImageTypeCircularFishEye = 3
    };
    CalcType srcTX, srcTY;
    CalcType destTX, destTY;
    CalcType scale[2];
    CalcType shear[2];
    CalcType rot[2];
    void *perspect[2];
    CalcType rad[6];
    CalcType mt[3][3];
    CalcType distance;
    CalcType horizontal;
    CalcType vertical;
    CalcType PI;
    CalcType cropX;
    CalcType cropY;
    CalcType cropWidth;
    CalcType cropHeight;
    CalcType centx;
    CalcType centy;
    CalcType sqrDist;
    int imageType;
};

void copyParam(const Remap& src, CudaRemapParam& dst, CalcType x, CalcType y,
    CalcType width, CalcType height, CalcType centx, CalcType centy, CalcType sqrDist, int type)
{
    dst.srcTX = src.srcTX;
    dst.srcTY = src.srcTY;
    dst.destTX = src.destTX;
    dst.destTY = src.destTY;
    dst.scale[0] = src.mp.scale[0];
    dst.scale[1] = src.mp.scale[1];
    dst.shear[0] = src.mp.shear[0];
    dst.shear[1] = src.mp.shear[1];
    dst.rot[0] = src.mp.rot[0];
    dst.rot[1] = src.mp.rot[1];
    dst.perspect[0] = src.mp.perspect[0];
    dst.perspect[1] = src.mp.perspect[1];
    for (int i = 0; i < 6; i++)
        dst.rad[i] = src.mp.rad[i];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            dst.mt[i][j] = src.mp.mt[i][j];
    }
    dst.distance = src.mp.distance;
    dst.horizontal = src.mp.horizontal;
    dst.vertical = src.mp.vertical;
    dst.PI = 3.1415926535898;
    dst.cropX = x;
    dst.cropY = y;
    dst.cropWidth = width;
    dst.cropHeight = height;
    dst.centx = centx;
    dst.centy = centy;
    dst.sqrDist = sqrDist;
    dst.imageType = type;
}

__constant__ CudaRemapParam param;

__device__ void dstToSrc(float* srcx, float* srcy, int dstx, int dsty, int mapWidth, int mapHeight)
{
    if (dstx >= mapWidth || dsty >= mapHeight)
        return;

    CalcType x_src = dstx, y_src = dsty;

    x_src -= param.srcTX - 0.5;
    y_src -= param.srcTY - 0.5;

    CalcType tx_dest, ty_dest;

    //rotate_erect  中心归一化
    tx_dest = x_src + param.rot[1];

    //if (tx_dest < -param.rot[0])
    //{
    //    int i = -tx_dest / 2 / param.rot[0];
    //    tx_dest += (i + 1) * param.rot[0] * 2;
    //}

    //if (tx_dest > param.rot[0])
    //{
    //    int i = tx_dest / 2 / param.rot[0];
    //    tx_dest -= i * param.rot[0] * 2;
    //}

    while (tx_dest < -param.rot[0])
        tx_dest += 2 * param.rot[0];

    while (tx_dest >   param.rot[0])
        tx_dest -= 2 * param.rot[0];

    ty_dest = y_src;

    x_src = tx_dest;
    y_src = ty_dest;

    //sphere_tp_erect 球面坐标转化为现实坐标
    CalcType phi, theta, r;
    CalcType v[3];
    phi = x_src / param.distance; //
    theta = -y_src / param.distance + param.PI / 2; //
    if (theta < 0)
    {
        theta = -theta;
        phi += param.PI;
    }
    if (theta > param.PI)
    {
        theta = param.PI - (theta - param.PI);
        phi += param.PI;
    }

    v[0] = sin(theta) * sin(phi);
    v[1] = cos(theta);
    v[2] = sin(theta) * cos(phi);

    //摄像机外参
    CalcType v0 = v[0];
    CalcType v1 = v[1];
    CalcType v2 = v[2];

    for (int i = 0; i<3; i++)
    {
        v[i] = param.mt[0][i] * v0 + param.mt[1][i] * v1 + param.mt[2][i] * v2;
    }

    r = sqrt(v[0] * v[0] + v[1] * v[1]);
    if (r == 0.0)
        theta = 0.0;
    else
        theta = param.distance * atan2(r, v[2]) / r;
    tx_dest = theta * v[0];
    ty_dest = theta * v[1];
    x_src = tx_dest;
    y_src = ty_dest;

    if (param.imageType == CudaRemapParam::ImageTypeRectlinear)                                    // rectilinear image
    {
        //SetDesc(m_stack[i],   rect_sphere_tp,         &(m_mp.distance) ); i++; // Convert rectilinear to spherical
        CalcType rho, theta, r;
        r = sqrt(x_src * x_src + y_src * y_src);
        theta = r / param.distance;

        if (theta >= param.PI / 2.0)
            rho = 1.6e16;
        else if (theta == 0.0)
            rho = 1.0;
        else
            rho = tan(theta) / theta;
        tx_dest = rho * x_src;
        ty_dest = rho * y_src;
        x_src = tx_dest;
        y_src = ty_dest;
    }

    //摄像机内参
    //SetDesc(  stack[i],   resize,                 param.scale       ); i++; // Scale image
    tx_dest = x_src * param.scale[0];
    ty_dest = y_src * param.scale[1];

    x_src = tx_dest;
    y_src = ty_dest;

    CalcType rt, scale;

    rt = (sqrt(x_src*x_src + y_src*y_src)) / param.rad[4];
    if (rt < param.rad[5])
    {
        scale = ((param.rad[3] * rt + param.rad[2]) * rt +
            param.rad[1]) * rt + param.rad[0];
    }
    else
        scale = 1000.0;

    tx_dest = x_src * scale;
    ty_dest = y_src * scale;

    x_src = tx_dest;
    y_src = ty_dest;

    //摄像机水平竖直矫正
    if (param.vertical != 0.0)
    {
        //SetDesc(stack[i],   vert,                   &(param.vertical));   i++;
        tx_dest = x_src;
        ty_dest = y_src + param.vertical;
        x_src = tx_dest;
        y_src = ty_dest;
    }

    if (param.horizontal != 0.0)
    {
        //SetDesc(stack[i],   horiz,                  &(param.horizontal)); i++;
        tx_dest = x_src + param.horizontal;
        ty_dest = y_src;
        x_src = tx_dest;
        y_src = ty_dest;
    }

    if (param.shear[0] != 0 || param.shear[1] != 0)
    {
        //SetDesc( stack[i],  shear,                  param.shear       ); i++;
        tx_dest = x_src + param.shear[0] * y_src;
        ty_dest = y_src + param.shear[1] * x_src;
    }

    tx_dest += param.destTX - 0.5;
    ty_dest += param.destTY - 0.5;

    if (param.imageType == CudaRemapParam::ImageTypeDrumFishEye ||
        param.imageType == CudaRemapParam::ImageTypeCircularFishEye)
    {
        CalcType diffx = tx_dest - param.centx;
        CalcType diffy = ty_dest - param.centy;
        if (tx_dest >= param.cropX && tx_dest < param.cropX + param.cropWidth &&
            ty_dest >= param.cropY && ty_dest < param.cropY + param.cropHeight &&
            diffx * diffx + diffy * diffy < param.sqrDist)
        {
            *srcx = tx_dest;
            *srcy = ty_dest;
        }
        else
        {
            *srcx = -1.0F;
            *srcy = -1.0F;
        }
    }
    else
    {
        if (tx_dest >= param.cropX && tx_dest < param.cropX + param.cropWidth &&
            ty_dest >= param.cropY && ty_dest < param.cropY + param.cropHeight)
        {
            *srcx = tx_dest;
            *srcy = ty_dest;
        }
        else
        {
            *srcx = -1.0F;
            *srcy = -1.0F;
        }
    }
}

__global__ void generateMapKernel(unsigned char* xMapData, int xMapStep,
    unsigned char* yMapData, int yMapStep, int mapWidth, int mapHeight)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= mapWidth || y >= mapHeight)
        return;

    dstToSrc((float*)(xMapData + y * xMapStep) + x, (float*)(yMapData + y * yMapStep) + x, x, y, mapWidth, mapHeight);
}

__global__ void generateMapAndMaskKernel(unsigned char* xMapData, int xMapStep,
    unsigned char* yMapData, int yMapStep, unsigned char* maskData, int maskStep, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height)
        return;

    float xpos, ypos;
    dstToSrc(&xpos, &ypos, x, y, width, height);
    *((float*)(xMapData + y * xMapStep) + x) = xpos;
    *((float*)(yMapData + y * yMapStep) + x) = ypos;
    *(maskData + y * maskStep + x) = (xpos == -1.0F || ypos == -1.0F) ? 0 : 255;
}

static void prepareConstantRemapParam(const PhotoParam& photoParam_,
    const cv::Size& srcSize, const cv::Size& dstSize)
{
    int dstWidth = dstSize.width, dstHeight = dstSize.height;
    int srcWidth = srcSize.width, srcHeight = srcSize.height;

    bool fullImage = (photoParam_.imageType == PhotoParam::ImageTypeRectlinear) ||
        (photoParam_.imageType == PhotoParam::ImageTypeFullFrameFishEye);
    PhotoParam photoParam = photoParam_;
    if (fullImage)
    {
        photoParam.cropX = 0;
        photoParam.cropY = 0;
        photoParam.cropWidth = srcWidth;
        photoParam.cropHeight = srcHeight;
    }
    CalcType centx = 0, centy = 0, sqrDist = 0;
    if (photoParam.circleR == 0)
    {
        centx = photoParam.cropX + photoParam.cropWidth / 2;
        centy = photoParam.cropY + photoParam.cropHeight / 2;
        sqrDist = photoParam.cropWidth > photoParam.cropHeight ?
            photoParam.cropWidth * photoParam.cropWidth * 0.25 :
            photoParam.cropHeight * photoParam.cropHeight * 0.25;
    }
    else
    {
        centx = photoParam.circleX;
        centy = photoParam.circleY;
        sqrDist = photoParam.circleR * photoParam.circleR;
    }

    Remap remap;
    remap.init(photoParam, dstWidth, dstHeight, srcWidth, srcHeight);
    CudaRemapParam cudaParam;
    copyParam(remap, cudaParam,
        photoParam.cropX, photoParam.cropY, photoParam.cropWidth, photoParam.cropHeight,
        centx, centy, sqrDist, photoParam.imageType);
    cudaSafeCall(cudaMemcpyToSymbol(param, &cudaParam, sizeof(CudaRemapParam)));
}

void cudaGenerateReprojectMap(const PhotoParam& photoParam_,
    const cv::Size& srcSize, const cv::Size& dstSize, cv::cuda::GpuMat& xmap, cv::cuda::GpuMat& ymap)
{
    CV_Assert(srcSize.width > 0 && srcSize.height > 0 &&
        dstSize.width > 0 && dstSize.height > 0 && dstSize.width == 2 * dstSize.height);

    prepareConstantRemapParam(photoParam_, srcSize, dstSize);
    xmap.create(dstSize, CV_32FC1);
    ymap.create(dstSize, CV_32FC1);

    dim3 block(16, 16);
    dim3 grid((dstSize.width + block.x - 1) / block.x, (dstSize.height + block.y - 1) / block.y);
    generateMapKernel<<<grid, block>>>(xmap.data, xmap.step, ymap.data, ymap.step, dstSize.width, dstSize.height);
    cudaSafeCall(cudaGetLastError());
}

void cudaGenerateReprojectMaps(const std::vector<PhotoParam>& params,
    const cv::Size& srcSize, const cv::Size& dstSize, std::vector<cv::cuda::GpuMat>& xmaps, std::vector<cv::cuda::GpuMat>& ymaps)
{
    int num = params.size();
    xmaps.resize(num);
    ymaps.resize(num);
    for (int i = 0; i < num; i++)
        cudaGenerateReprojectMap(params[i], srcSize, dstSize, xmaps[i], ymaps[i]);
}

void cudaGenerateReprojectMapAndMask(const PhotoParam& photoParam_, const cv::Size& srcSize, const cv::Size& dstSize, 
    cv::cuda::GpuMat& xmap, cv::cuda::GpuMat& ymap, cv::cuda::GpuMat& mask)
{
    CV_Assert(srcSize.width > 0 && srcSize.height > 0 &&
        dstSize.width > 0 && dstSize.height > 0 && dstSize.width == 2 * dstSize.height);

    prepareConstantRemapParam(photoParam_, srcSize, dstSize);
    xmap.create(dstSize, CV_32FC1);
    ymap.create(dstSize, CV_32FC1);
    mask.create(dstSize, CV_8UC1);

    dim3 block(16, 16);
    dim3 grid((dstSize.width + block.x - 1) / block.x, (dstSize.height + block.y - 1) / block.y);
    generateMapAndMaskKernel<<<grid, block>>>(xmap.data, xmap.step, ymap.data, ymap.step, mask.data, mask.step, dstSize.width, dstSize.height);
    cudaSafeCall(cudaGetLastError());
}

void cudaGenerateReprojectMapsAndMasks(const std::vector<PhotoParam>& params, const cv::Size& srcSize, const cv::Size& dstSize, 
    std::vector<cv::cuda::GpuMat>& xmaps, std::vector<cv::cuda::GpuMat>& ymaps, std::vector<cv::cuda::GpuMat>& masks)
{
    int num = params.size();
    xmaps.resize(num);
    ymaps.resize(num);
    masks.resize(num);
    for (int i = 0; i < num; i++)
        cudaGenerateReprojectMapAndMask(params[i], srcSize, dstSize, xmaps[i], ymaps[i], masks[i]);
}

texture<uchar4, 2> srcTexture;
texture<float, 2> xmapTexture, ymapTexture, weightTexture;

template<typename DstElemType>
__global__ void reprojectNNWithMapKernel(unsigned char* dstData, 
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
__global__ void reprojectLinearWithMapKernel(unsigned char* dstData, 
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
__global__ void reprojectCubicWithMapKernel(unsigned char* dstData, 
    int dstWidth, int dstHeight, int dstStep, int srcWidth, int srcHeight)
{
    int dstx = threadIdx.x + blockIdx.x * blockDim.x;
    int dsty = threadIdx.y + blockIdx.y * blockDim.y;
    if (dstx >= dstWidth || dsty >= dstHeight)
        return;

    float srcx = tex2D(xmapTexture, dstx, dsty);
    float srcy = tex2D(ymapTexture, dstx, dsty);

    //unsigned char* ptrDst = dstData + dsty * dstStep + dstx * 4;
    DstElemType* ptrDst = (DstElemType*)(dstData + dsty * dstStep) + dstx * 4;
    if (srcx < 0 || srcx >= srcWidth || srcy < 0 || srcy >= srcHeight)
        ptrDst[3] = ptrDst[2] = ptrDst[1] = ptrDst[0] = 0;
    else
        resampling(srcWidth, srcHeight, srcx, srcy, ptrDst);
}

template<typename DstElemType>
__global__ void reprojectCubicNoMapKernel(unsigned char* dstData,
    int dstWidth, int dstHeight, int dstStep, int srcWidth, int srcHeight)
{
    int dstx = threadIdx.x + blockIdx.x * blockDim.x;
    int dsty = threadIdx.y + blockIdx.y * blockDim.y;
    if (dstx >= dstWidth || dsty >= dstHeight)
        return;

    float srcx, srcy;
    dstToSrc(&srcx, &srcy, dstx, dsty, dstWidth, dstHeight);

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
    if (dstx >= dstWidth || dsty >= dstHeight)
        return;

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

void cudaReproject(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Size& dstSize,
    const PhotoParam& param, cv::cuda::Stream& stream)
{
    CV_Assert(src.data && src.type() == CV_8UC4);

    prepareConstantRemapParam(param, src.size(), dstSize);
    dst.create(dstSize, CV_8UC4);

    cudaChannelFormatDesc chanDescUchar4 = cudaCreateChannelDesc<uchar4>();
    cudaSafeCall(cudaBindTexture2D(NULL, srcTexture, src.data, chanDescUchar4, src.cols, src.rows, src.step));

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    dim3 block(16, 16);
    dim3 grid((dstSize.width + block.x - 1) / block.x, (dstSize.height + block.y - 1) / block.y);
    reprojectCubicNoMapKernel<unsigned char><<<grid, block, 0, st>>>(dst.data, dstSize.width, dstSize.height, dst.step, src.cols, src.rows);
    cudaSafeCall(cudaGetLastError());

    cudaSafeCall(cudaUnbindTexture(srcTexture));
}

void cudaReprojectTo16S(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Size& dstSize,
    const PhotoParam& param, cv::cuda::Stream& stream)
{
    CV_Assert(src.data && src.type() == CV_8UC4);

    prepareConstantRemapParam(param, src.size(), dstSize);
    dst.create(dstSize, CV_16SC4);

    cudaChannelFormatDesc chanDescUchar4 = cudaCreateChannelDesc<uchar4>();
    cudaSafeCall(cudaBindTexture2D(NULL, srcTexture, src.data, chanDescUchar4, src.cols, src.rows, src.step));

    cudaStream_t st = cv::cuda::StreamAccessor::getStream(stream);
    dim3 block(16, 16);
    dim3 grid((dstSize.width + block.x - 1) / block.x, (dstSize.height + block.y - 1) / block.y);
    reprojectCubicNoMapKernel<short><<<grid, block, 0, st>>>(dst.data, dstSize.width, dstSize.height, dst.step, src.cols, src.rows);
    cudaSafeCall(cudaGetLastError());

    cudaSafeCall(cudaUnbindTexture(srcTexture));
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
    reprojectCubicWithMapKernel<unsigned char><<<grid, block, 0, st>>>(dst.data, dstSize.width, dstSize.height, dst.step, src.cols, src.rows);
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
    reprojectCubicWithMapKernel<short><<<grid, block, 0, st>>>(dst.data, dstSize.width, dstSize.height, dst.step, src.cols, src.rows);
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
