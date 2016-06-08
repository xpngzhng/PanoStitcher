#pragma once

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <atomic>

void compensate(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results);
void compensate3(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results);

class GainCompensate
{
public:
    GainCompensate() :numImages(0), maxMeanIndex(0), rows(0), cols(0), success(false) {}
    bool prepare(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks);
    bool compensate(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& results) const;
private:
    std::vector<double> gains;
    std::vector<std::vector<unsigned char> > LUTs;
    int numImages;
    int maxMeanIndex;
    int rows, cols;
    int success;
};

void compensateGray(const std::vector<cv::Mat>& images,
    const std::vector<cv::Mat>& masks, int refIndex, std::vector<cv::Mat>& results);
void compensateLightAndSaturation(const std::vector<cv::Mat>& images,
    const std::vector<cv::Mat>& masks, int refIndex, std::vector<cv::Mat>& results);

struct BlendConfig
{
    enum SeamMode
    {
        SEAM_SKIP,
        SEAM_DISTANCE_TRANSFORM,
        SEAM_GRAPH_CUT
    };
    enum BlendMode
    {
        BLEND_PASTE,
        BLEND_LINEAR,
        BLEND_MULTIBAND
    };
    BlendConfig(int seamMode_ = SEAM_GRAPH_CUT, 
        int blendMode_ = BLEND_MULTIBAND, int radiusForLinear_ = 125, 
        int maxLevelsForMultiBand_ = 16, int minLengthForMultiBand_ = 2,
        int padForGraphCut_ = 8, int scaleForGraphCut_ = 8, 
        int refineForGraphCut_ = 1, double ratioForGraphCut_ = 0.75)
        : seamMode(seamMode_), blendMode(blendMode_), radiusForLinear(radiusForLinear_), 
        maxLevelsForMultiBand(maxLevelsForMultiBand_), minLengthForMultiBand(minLengthForMultiBand_),
        padForGraphCut(padForGraphCut_), scaleForGraphCut(scaleForGraphCut_), 
        refineForGraphCut(refineForGraphCut_), ratioForGraphCut(ratioForGraphCut_)
    {};
    void setSeamSkip()
    {
        seamMode = SEAM_SKIP;
    }
    void setSeamDistanceTransform()
    {
        seamMode = SEAM_DISTANCE_TRANSFORM;
    }
    void setSeamGraphCut(int pad = 8, int scale = 8, int refine = 1, double ratio = 0.75)
    {
        padForGraphCut = pad;
        scaleForGraphCut = scale;
        refineForGraphCut = refine;
        ratioForGraphCut = ratio;
    }
    void setBlendPaste()
    {
        blendMode = BLEND_PASTE;
    }
    void setBlendLinear(int radius = 125)
    {
        blendMode = BLEND_LINEAR;
        radiusForLinear = radius;
    }
    void setBlendMultiBand(int maxLevels = 16, int minLength = 2)
    {
        blendMode = BLEND_MULTIBAND;
        maxLevelsForMultiBand = maxLevels;
        minLengthForMultiBand = minLength;
    }
    int seamMode;
    int blendMode;
    int radiusForLinear;
    int maxLevelsForMultiBand;
    int minLengthForMultiBand;
    int padForGraphCut;
    int scaleForGraphCut;
    int refineForGraphCut;
    double ratioForGraphCut;    
};

void serialBlend(const BlendConfig& config, const cv::Mat& image, const cv::Mat& mask, 
    cv::Mat& blendImage, cv::Mat& blendMask);

void parallelBlend(const BlendConfig& config, const std::vector<cv::Mat>& images,
    const std::vector<cv::Mat>& masks, cv::Mat& blendImage);

class TilingMultibandBlend
{
public:
    TilingMultibandBlend() : numImages(0), rows(0), cols(0), numLevels(0), success(false) {}
    bool prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength);
    void tile(const cv::Mat& image, const cv::Mat& mask, int index);
    void composite(cv::Mat& blendImage);
    void blend(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, cv::Mat& blendImage);
    void blendAndCompensate(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, cv::Mat& blendImage);

private:
    std::vector<cv::Mat> uniqueMasks;
    std::vector<cv::Mat> resultPyr;
    std::vector<cv::Mat> resultWeightPyr;
    int numImages;
    int rows, cols;
    int numLevels;
    bool success;
};

class TilingMultibandBlendFast
{
public:
    TilingMultibandBlendFast() : numImages(0), rows(0), cols(0), numLevels(0), success(false) {}
    bool prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength);
    void blend(const std::vector<cv::Mat>& images, cv::Mat& blendImage);
    void blend(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, cv::Mat& blendImage);
    void getUniqueMasks(std::vector<cv::Mat>& masks) const;

private:
    std::vector<cv::Mat> uniqueMasks;
    std::vector<cv::Mat> resultPyr, resultUpPyr, resultWeightPyr;
    std::vector<cv::Mat> imagePyr, image32SPyr, imageUpPyr;
    std::vector<std::vector<cv::Mat> > alphaPyrs, weightPyrs;    
    cv::Mat maskNot;
    int numImages;
    int rows, cols;
    int numLevels;
    bool fullMask;
    bool success;

    std::vector<cv::Mat> customResultWeightPyr;
    std::vector<std::vector<cv::Mat> > customWeightPyrs;
    cv::Mat customAux, customMaskNot;
};

// DEPRECATED
// Just only a little faster than TilingMultibandBlendFast at the expense of more memory consumption
class TilingMultibandBlendFastParallel
{
public:
    TilingMultibandBlendFastParallel() : numImages(0), rows(0), cols(0), numLevels(0), success(false), threadEnd(true) {}
    ~TilingMultibandBlendFastParallel();
    bool prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength);
    void blend(const std::vector<cv::Mat>& images, cv::Mat& blendImage);

private:
    std::vector<cv::Mat> resultPyr, resultUpPyr, resultWeightPyr;
    std::vector<std::vector<cv::Mat> > imagePyrs, image32SPyrs, imageUpPyrs;
    std::vector<std::vector<cv::Mat> > alphaPyrs, weightPyrs;
    std::vector<std::vector<unsigned char> > rowBuffers, tabBuffers;
    std::vector<unsigned char> restoreRowBuffer, restoreTabBuffer;
    cv::Mat maskNot;
    int numImages;
    int rows, cols;
    int numLevels;
    bool fullMask;
    bool success;

    void init();
    void endThreads();
    std::vector<cv::Mat> imageHeaders;
    std::vector<std::unique_ptr<std::thread> > threads;
    std::mutex mtxBuildPyr, mtxAccum;
    std::condition_variable cvBuildPyr, cvAccum;
    std::atomic<int> buildCount;
    bool threadEnd;
    void buildPyramid(int index);
};

class TilingLinearBlend
{
public:
    TilingLinearBlend() : numImages(0), rows(0), cols(0), success(false) {}
    bool prepare(const std::vector<cv::Mat>& masks, int radius);
    void blend(const std::vector<cv::Mat>& images, cv::Mat& blendImage) const;
private:
    std::vector<cv::Mat> weights;
    int numImages;
    int rows, cols;
    bool success;
};

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

private:
    std::vector<cv::cuda::GpuMat> resultPyr, resultUpPyr, resultWeightPyr;
    std::vector<cv::cuda::GpuMat> imagePyr, image32SPyr, imageUpPyr;
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

void prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength,
    std::vector<std::vector<cv::cuda::GpuMat> >& alphaPyrs, std::vector<std::vector<cv::cuda::GpuMat> >& weightPyrs,
    std::vector<cv::cuda::GpuMat>& resultPyr, std::vector<std::vector<cv::cuda::GpuMat> >& image32SPyrs,
    std::vector<std::vector<cv::cuda::GpuMat> >& imageUpPyrs, std::vector<cv::cuda::GpuMat>& resultUpPyr);
void calcImagePyramid(const cv::cuda::GpuMat& image, const std::vector<cv::cuda::GpuMat>& alphaPyr,
    std::vector<cv::cuda::GpuMat>& imagePyr, cv::cuda::Stream& stream,
    std::vector<cv::cuda::GpuMat>& image32SPyr, std::vector<cv::cuda::GpuMat>& imageUpPyr);
void calcResult(const std::vector<std::vector<cv::cuda::GpuMat> >& imagePyrs,
    const std::vector<std::vector<cv::cuda::GpuMat> >& weightPyrs, cv::cuda::GpuMat& result,
    std::vector<cv::cuda::GpuMat>& resultPyr, std::vector<cv::cuda::GpuMat>& resultUpPyr);

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

class MultibandBlendGainAdjust
{
public:
    MultibandBlendGainAdjust() : numImages(0), rows(0), cols(0), prepareSuccess(false), calcGainSuccess(false) {}
    bool prepare(const std::vector<cv::Mat>& masks, int radius);
    bool calcGain(const std::vector<cv::Mat>& images, std::vector<std::vector<unsigned char> >& luts);
private:
    int numImages;
    int rows, cols;
    bool prepareSuccess;
    bool calcGainSuccess;
    TilingMultibandBlendFast blender;
    cv::Mat blendImage;
    std::vector<cv::Mat> extendedMasks;
    std::vector<std::vector<unsigned char> > luts;
};

void transform(const cv::Mat& src, cv::Mat& dst, const std::vector<unsigned char>& lut, const cv::Mat& mask = cv::Mat());

void cudaTransform(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const std::vector<unsigned char>& lut);