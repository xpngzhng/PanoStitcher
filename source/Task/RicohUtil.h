#pragma once

#include "ZBlend.h"
#include "ZReproject.h"
#include "CudaInterface.h"
#include "PinnedMemoryPool.h"
#include "ConcurrentQueue.h"
#include "CustomMask.h"
#include "opencv2/core.hpp"
#include <memory>
#include <string>

class RicohPanoramaRender
{
public:
    RicohPanoramaRender() : success(0) {};
    bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize);
    void render(const cv::Mat& src, cv::Mat& dst);
private:
    cv::Size srcSize, dstSize;
    cv::Mat dstSrcMap1, dstSrcMap2;
    cv::Mat from1, from2, intersect;
    cv::Mat weight1, weight2;
    int success;
};

class DetuPanoramaRender
{
public:
    DetuPanoramaRender() : success(0) {};
    bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize);
    void render(const cv::Mat& src, cv::Mat& dst);
private:
    cv::Size srcSize, dstSize;
    cv::Mat dstSrcMap;
    int success;
};

class PanoramaRender
{
public:
    enum BlendType 
    {
        BlendTypeLinear, 
        BlendTypeMultiband
    };
    virtual ~PanoramaRender() {};
    virtual bool prepare(const std::string& path, int blendType, const cv::Size& srcSize, const cv::Size& dstSize) = 0;
    virtual bool render(const std::vector<cv::Mat>& src, cv::Mat& dst) = 0;
};

class DualGoProPanoramaRender : public PanoramaRender
{
public:
    DualGoProPanoramaRender() : success(0) {};
    ~DualGoProPanoramaRender() {};
    bool prepare(const std::string& path, int blendType, const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst);
    bool render(const std::vector<cv::Mat>& src, cv::Mat& dst);
private:
    cv::Size srcSize, dstSize;
    cv::Mat dstSrcMap1, dstSrcMap2;
    cv::Mat mask1, mask2;
    cv::Mat from1, from2, intersect;
    cv::Mat weight1, weight2;
    int success;
};

class CPUMultiCameraPanoramaRender : public PanoramaRender
{
public:
    CPUMultiCameraPanoramaRender() : success(0) {};
    ~CPUMultiCameraPanoramaRender() {};
    bool prepare(const std::string& path, int blendType, const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const std::vector<cv::Mat>& src, cv::Mat& dst);
private:
    cv::Size srcSize, dstSize;
    std::vector<cv::Mat> dstSrcMaps;
    std::vector<cv::Mat> masks;
    std::vector<cv::Mat> reprojImages;
    TilingMultibandBlendFast blender;
    int numImages;
    int success;
};

class CudaMultiCameraPanoramaRender : public PanoramaRender
{
public:
    CudaMultiCameraPanoramaRender(): success(0) {};
    ~CudaMultiCameraPanoramaRender() {};
    bool prepare(const std::string& path, int blendType, const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const std::vector<cv::Mat>& src, cv::Mat& dst);
private:
    cv::Size srcSize, dstSize;
    std::vector<cv::cuda::GpuMat> dstSrcXMapsGPU, dstSrcYMapsGPU;
    std::vector<cv::cuda::HostMem> srcMems;
    std::vector<cv::Mat> srcImages;
    std::vector<cv::cuda::GpuMat> srcImagesGPU;
    std::vector<cv::cuda::GpuMat> reprojImagesGPU;
    cv::cuda::GpuMat blendImageGPU;
    cv::Mat blendImage;
    std::vector<cv::cuda::Stream> streams;
    CudaTilingMultibandBlendFast blender;
    int numImages;
    int success;
};

// render accepts pinned memory cv::Mat
class CudaMultiCameraPanoramaRender2
{
public:
    CudaMultiCameraPanoramaRender2() : success(0) {};
    ~CudaMultiCameraPanoramaRender2() {};
    bool prepare(const std::string& path, int blendType, const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const std::vector<cv::Mat>& src, cv::cuda::GpuMat& dst);
    int getNumImages() const;
private:
    cv::Size srcSize, dstSize;
    std::vector<cv::cuda::GpuMat> dstSrcXMapsGPU, dstSrcYMapsGPU;
    std::vector<cv::cuda::GpuMat> srcImagesGPU;
    std::vector<cv::cuda::GpuMat> reprojImagesGPU;
    std::vector<cv::cuda::Stream> streams;
    int blendType;
    CudaTilingMultibandBlendFast mbBlender;
    CudaTilingLinearBlend lBlender;
    int numImages;
    int success;
};

// add gain adjust function compared with CudaMultiCameraPanoramaRender2
class CudaMultiCameraPanoramaRender3 : public PanoramaRender
{
public:
    CudaMultiCameraPanoramaRender3() : success(0) {};
    ~CudaMultiCameraPanoramaRender3() {};
    bool prepare(const std::string& path, int blendType, const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const std::vector<cv::Mat>& src, cv::Mat& dst);
private:
    cv::Size srcSize, dstSize;
    std::vector<std::vector<unsigned char> > luts;
    std::vector<cv::cuda::GpuMat> dstSrcXMapsGPU, dstSrcYMapsGPU;
    std::vector<cv::cuda::GpuMat> srcImagesGPU;
    std::vector<cv::cuda::GpuMat> reprojImagesGPU;
    cv::cuda::GpuMat blendImageGPU;
    cv::Mat blendImage;
    std::vector<cv::cuda::Stream> streams;
    MultibandBlendGainAdjust adjuster;
    CudaTilingLinearBlend blender;
    int numImages;
    int success;
};

// multithread version of CudaMultiCameraPanoramaRender2
class CudaPanoramaRender
{
public:
    CudaPanoramaRender() : success(0) {};
    ~CudaPanoramaRender() { clear(); };
    bool prepare(const std::string& path, const std::string& customMaskPath,
        int highQualityBlend, int completeQueue, const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const std::vector<cv::Mat>& src, const std::vector<long long int> timeStamps);
    bool getResult(cv::Mat& dst, long long int& timeStamp);
    void stop();
    void resume();
    void waitForCompletion();
    void clear();
    int getNumImages() const;
private:
    cv::Size srcSize, dstSize;
    std::vector<cv::cuda::GpuMat> dstUniqueMasksGPU, currMasksGPU;
    int useCustomMasks;
    std::vector<CudaCustomIntervaledMasks> customMasks;
    std::vector<cv::cuda::GpuMat> dstSrcXMapsGPU, dstSrcYMapsGPU;
    std::vector<cv::cuda::GpuMat> srcImagesGPU;
    std::vector<cv::cuda::GpuMat> reprojImagesGPU;
    PinnedMemoryPool pool;
    typedef ForceWaitRealTimeQueue<std::pair<cv::cuda::HostMem, long long int> > RealTimeQueue;
    typedef BoundedCompleteQueue<std::pair<cv::cuda::HostMem, long long int> > CompleteQueue;
    RealTimeQueue rtQueue;
    CompleteQueue cpQueue;
    std::vector<cv::cuda::Stream> streams;
    int highQualityBlend;
    int completeQueue;
    CudaTilingMultibandBlendFast mbBlender;
    std::vector<cv::cuda::GpuMat> weightsGPU;
    cv::cuda::GpuMat accumGPU;
    int numImages;
    int success;
};

// single thread version of CudaPanoramaRender
class CudaPanoramaRender2
{
public:
    CudaPanoramaRender2() : success(0), highQualityBlend(0), numImages(0) {};
    virtual ~CudaPanoramaRender2() { };
    virtual bool prepare(const std::string& path, int highQualityBlend, const cv::Size& srcSize, const cv::Size& dstSize);
    virtual bool render(const std::vector<cv::Mat>& src, cv::cuda::GpuMat& dst, 
        const std::vector<std::vector<unsigned char> >& luts = std::vector<std::vector<unsigned char> >());
    virtual void clear();
    virtual int getNumImages() const;
protected:
    cv::Size srcSize, dstSize;
    std::vector<PhotoParam> params;
    std::vector<cv::cuda::GpuMat> dstUniqueMasksGPU, currMasksGPU;
    std::vector<cv::cuda::GpuMat> dstSrcXMapsGPU, dstSrcYMapsGPU;
    std::vector<cv::cuda::GpuMat> srcImagesGPU;
    std::vector<cv::cuda::GpuMat> reprojImagesGPU;
    std::vector<cv::cuda::Stream> streams;
    int highQualityBlend;
    CudaTilingMultibandBlendFast mbBlender;
    std::vector<cv::cuda::GpuMat> weightsGPU;
    cv::cuda::GpuMat accumGPU;
    int numImages;
    int success;
};

class CudaRicohPanoramaRender : public CudaPanoramaRender2
{
public:
    CudaRicohPanoramaRender() {};
    virtual ~CudaRicohPanoramaRender() { };
    virtual bool prepare(const std::string& path, int highQualityBlend, const cv::Size& srcSize, const cv::Size& dstSize);
    virtual bool render(const std::vector<cv::Mat>& src, cv::cuda::GpuMat& dst,
        const std::vector<std::vector<unsigned char> >& luts = std::vector<std::vector<unsigned char> >());
    virtual void clear();
    virtual int getNumImages() const;
};

// cpu version of CudaPanoramaRender
class CPUPanoramaRender
{
public:
    CPUPanoramaRender() : success(0), highQualityBlend(0), numImages(0) {};
    virtual ~CPUPanoramaRender() { };
    virtual bool prepare(const std::string& path, int highQualityBlend, const cv::Size& srcSize, const cv::Size& dstSize);
    virtual bool render(const std::vector<cv::Mat>& src, cv::Mat& dst);
    virtual void clear();
    virtual int getNumImages() const;
protected:
    cv::Size srcSize, dstSize;
    std::vector<cv::Mat> maps;
    std::vector<cv::Mat> reprojImages;
    int highQualityBlend;
    std::unique_ptr<MultibandBlendBase> mbBlender;
    std::vector<cv::Mat> weights;
    cv::Mat accum;
    int numImages;
    int success;
};

class CPURicohPanoramaRender : public CPUPanoramaRender
{
public:
    CPURicohPanoramaRender() {};
    virtual ~CPURicohPanoramaRender() { };
    virtual bool prepare(const std::string& path, int highQualityBlend, const cv::Size& srcSize, const cv::Size& dstSize);
    virtual bool render(const std::vector<cv::Mat>& src, cv::Mat& dst);
    virtual void clear();
    virtual int getNumImages() const;
};

#include "CompileControl.h"
#if COMPILE_INTEL_OPENCL
#include "oclobject.hpp"
class IOclPanoramaRender
{
public:
    IOclPanoramaRender() : success(0) {};
    ~IOclPanoramaRender() { clear(); };
    bool prepare(const std::string& path, int highQualityBlend, int completeQueue,
        const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const std::vector<cv::Mat>& src, const std::vector<long long int>& timeStamps);
    bool getResult(cv::Mat& dst, long long int& timeStamp);
    void stop();
    void resume();
    void waitForCompletion();
    void clear();
    int getNumImages() const;
private:
    cv::Size srcSize, dstSize;
    std::vector<iocl::UMat> xmaps, ymaps;
    IntelMemoryPool pool;
    typedef ForceWaitRealTimeQueue<std::pair<iocl::UMat, long long int> > RealTimeQueue;
    typedef BoundedCompleteQueue<std::pair<iocl::UMat, long long int> > CompleteQueue;
    RealTimeQueue rtQueue;
    CompleteQueue cpQueue;
    int highQualityBlend;
    int completeQueue;
    std::vector<iocl::UMat> weights;
    int numImages;
    int success;
};
#endif

#if COMPILE_INTEGRATED_OPENCL
#include "IntelOpenCLInterface.h"
class IOclPanoramaRender
{
public:
    IOclPanoramaRender() : success(0) {};
    ~IOclPanoramaRender() { clear(); };
    bool prepare(const std::string& path, int highQualityBlend, const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const std::vector<iocl::UMat>& src, iocl::UMat& dst);
    void clear();
    int getNumImages() const;
private:
    cv::Size srcSize, dstSize;
    std::vector<iocl::UMat> xmaps, ymaps;
    std::vector<iocl::UMat> reprojImages;
    int highQualityBlend;
    IOclTilingMultibandBlendFast mbBlender;
    std::vector<iocl::UMat> weights;
    iocl::UMat accum;
    int numImages;
    int success;
};
#endif

#if COMPILE_DISCRETE_OPENCL
#include "OpenCLAccel/basic.hpp"
#include "DiscreteOpenCL/DiscreteOpenCLInterface.h"
class DOclPanoramaRender
{
public:
    DOclPanoramaRender() : success(0) {};
    ~DOclPanoramaRender() { clear(); };
    bool prepare(const std::string& path, int highQualityBlend, const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const std::vector<docl::HostMem>& src, docl::GpuMat& dst);
    void clear();
    int getNumImages() const;
private:
    cv::Size srcSize, dstSize;
    std::vector<docl::GpuMat> xmaps, ymaps;
    std::vector<docl::GpuMat> images;
    std::vector<docl::GpuMat> reprojImages;
    int highQualityBlend;
    DOclTilingMultibandBlendFast mbBlender;
    std::vector<docl::GpuMat> weights;
    docl::GpuMat accum;
    int numImages;
    int success;

    std::vector<std::unique_ptr<OpenCLProgramOneKernel> > reprojKernels;
    std::vector<std::unique_ptr<OpenCLQueue> > queues;
};
#endif

class ImageVisualCorrect
{
public:
    ImageVisualCorrect() : numImages(0), srcWidth(0), srcHeight(0), equiRectWidth(0), equiRectHeight(0), success(0) {};
    virtual ~ImageVisualCorrect() {}
    virtual bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize);
    virtual bool correct(const std::vector<cv::Mat>& images, std::vector<double>& exposures);
    virtual void clear();

protected:
    ExposureColorCorrect corrector;
    std::vector<cv::Mat> maps;
    int numImages;
    int srcWidth, srcHeight;
    int equiRectWidth, equiRectHeight;
    int success;
};

class RicohImageVisualCorrect : public ImageVisualCorrect
{
public:
    RicohImageVisualCorrect() {};
    virtual ~RicohImageVisualCorrect() {}
    virtual bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize);
    virtual bool correct(const std::vector<cv::Mat>& images, std::vector<double>& exposures);
    virtual void clear();
};