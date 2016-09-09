#pragma once

#include "Warp/ZReproject.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"

class CudaTilingMultibandBlend
{
public:
    CudaTilingMultibandBlend() : numImages(0), rows(0), cols(0), numLevels(0), success(false) {}
    bool prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength);
    void tile(const cv::cuda::GpuMat& image, const cv::cuda::GpuMat& mask, int index);
    void composite(cv::cuda::GpuMat& blendImage);
    void blend(const std::vector<cv::cuda::GpuMat>& images, const std::vector<cv::cuda::GpuMat>& masks, cv::cuda::GpuMat& blendImage);

private:
    std::vector<cv::cuda::GpuMat> uniqueMasks;
    std::vector<cv::cuda::GpuMat> resultPyr;
    std::vector<cv::cuda::GpuMat> imagePyr, image32SPyr;
    std::vector<cv::cuda::GpuMat> alphaPyr, alpha32SPyr;
    std::vector<cv::cuda::GpuMat> imageUpPyr, resultUpPyr;
    std::vector<cv::cuda::GpuMat> weightPyr, resultWeightPyr;
    cv::cuda::GpuMat image16S, aux16S, maskNot;
    int numImages;
    int rows, cols;
    int numLevels;
    bool success;
};

class CudaTilingMultibandBlendFast
{
public:
    CudaTilingMultibandBlendFast() : numImages(0), rows(0), cols(0), numLevels(0), success(false) {}
    bool prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength);
    void blend(const std::vector<cv::cuda::GpuMat>& images, cv::cuda::GpuMat& blendImage);
    void blend(const std::vector<cv::cuda::GpuMat>& images, const std::vector<cv::cuda::GpuMat>& masks,
        cv::cuda::GpuMat& blendImage);
    void getUniqueMasks(std::vector<cv::cuda::GpuMat>& masks) const;

private:
    std::vector<cv::cuda::GpuMat> uniqueMasks;
    std::vector<cv::cuda::GpuMat> resultPyr, resultUpPyr, resultWeightPyr;
    std::vector<cv::cuda::GpuMat> imagePyr, imageUpPyr;
    std::vector<std::vector<cv::cuda::GpuMat> > alphaPyrs, weightPyrs;
    cv::cuda::GpuMat maskNot;
    int numImages;
    int rows, cols;
    int numLevels;
    bool fullMask;
    bool success;

    std::vector<cv::cuda::GpuMat> customResultWeightPyr;
    std::vector<std::vector<cv::cuda::GpuMat> > customWeightPyrs;
    cv::cuda::GpuMat customAux, customMaskNot;
};

class CudaTilingMultibandBlendFast32F
{
public:
    CudaTilingMultibandBlendFast32F() : numImages(0), rows(0), cols(0), numLevels(0), success(false) {}
    bool prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength);
    void blend(const std::vector<cv::cuda::GpuMat>& images, cv::cuda::GpuMat& blendImage);

private:
    std::vector<cv::cuda::GpuMat> resultPyr, resultUpPyr, resultScalePyr;
    std::vector<cv::cuda::GpuMat> imagePyr, imageDownPyr, imageUpPyr;
    std::vector<std::vector<cv::cuda::GpuMat> > alphaPyrs, weightPyrs;
    cv::cuda::GpuMat maskNot;
    int numImages;
    int rows, cols;
    int numLevels;
    bool fullMask;
    bool success;
};

class CudaTilingLinearBlend
{
public:
    CudaTilingLinearBlend() : numImages(0), rows(0), cols(0), success(false) {}
    bool prepare(const std::vector<cv::Mat>& masks, int radius);
    void blend(const std::vector<cv::cuda::GpuMat>& images, cv::cuda::GpuMat& blendImage);
private:
    std::vector<cv::cuda::GpuMat> weights;
    cv::cuda::GpuMat accumImage;
    int numImages;
    int rows, cols;
    bool success;
};

void cudaTransform(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const std::vector<unsigned char>& lut);

void cudaGenerateReprojectMap(const PhotoParam& param,
    const cv::Size& srcSize, const cv::Size& dstSize, cv::cuda::GpuMat& xmap, cv::cuda::GpuMat& ymap);

void cudaGenerateReprojectMaps(const std::vector<PhotoParam>& params,
    const cv::Size& srcSize, const cv::Size& dstSize, std::vector<cv::cuda::GpuMat>& xmaps, std::vector<cv::cuda::GpuMat>& ymaps);

void cudaGenerateReprojectMapAndMask(const PhotoParam& param, const cv::Size& srcSize, const cv::Size& dstSize, 
    cv::cuda::GpuMat& xmap, cv::cuda::GpuMat& ymap, cv::cuda::GpuMat& mask);

void cudaGenerateReprojectMaps(const std::vector<PhotoParam>& params, const cv::Size& srcSize, const cv::Size& dstSize, 
    std::vector<cv::cuda::GpuMat>& xmaps, std::vector<cv::cuda::GpuMat>& ymaps, std::vector<cv::cuda::GpuMat>& masks);

void cudaReproject(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Size& dstSize,
    const PhotoParam& param, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void cudaReprojectTo16S(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Size& dstSize,
    const PhotoParam& param, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void cudaReproject(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
    const cv::cuda::GpuMat& xmap, const cv::cuda::GpuMat& ymap, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void cudaReprojectTo16S(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
    const cv::cuda::GpuMat& xmap, const cv::cuda::GpuMat& ymap, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void cudaReprojectWeightedAccumulateTo32F(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
    const cv::cuda::GpuMat& xmap, const cv::cuda::GpuMat& ymap, const cv::cuda::GpuMat& weight,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void alphaBlend8UC4(cv::cuda::GpuMat& target, const cv::cuda::GpuMat& blender);

void cvtBGR32ToYUV420P(const cv::cuda::GpuMat& bgr32, cv::cuda::GpuMat& y, cv::cuda::GpuMat& u, cv::cuda::GpuMat& v,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void cvtBGR32ToNV12(const cv::cuda::GpuMat& bgr32, cv::cuda::GpuMat& y, cv::cuda::GpuMat& uv,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void cvtYUV420PToBGR32(const cv::cuda::GpuMat& y, const cv::cuda::GpuMat& u, const cv::cuda::GpuMat& v, cv::cuda::GpuMat& bgr32,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void cvtNV12ToBGR32(const cv::cuda::GpuMat& y, const cv::cuda::GpuMat& uv, cv::cuda::GpuMat& bgr32,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void resize8UC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize);
