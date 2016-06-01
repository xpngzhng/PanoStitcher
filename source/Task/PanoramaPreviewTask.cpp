#include "PanoramaTask.h"
#include "PanoramaTaskUtil.h"
#include "ZReproject.h"
#include "ZBlend.h"
#include "RicohUtil.h"

struct CPUPanoramaPreviewTask::Impl
{
    Impl();
    ~Impl();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile,
        int dstWidth, int dstHeight);
    bool reset(const std::string& cameraParamFile);
    bool seek(const std::vector<long long int>& timeStamps);
    bool stitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst, int frameIncrement);

    bool getMasks(std::vector<cv::Mat>& masks);
    bool readNextAndReprojectForAll(std::vector<cv::Mat>& images);
    bool readNextAndReprojectForOne(int index, cv::Mat& image);
    bool readPrevAndReprojectForOne(int index, cv::Mat& image);

    void clear();

    int numVideos;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader> readers;
    std::vector<cv::Mat> dstSrcMaps, dstMasks;
    TilingMultibandBlendFast blender;
    std::vector<cv::Mat> images, reprojImages;
    std::vector<avp::AudioVideoFrame> frames;
    cv::Mat blendImage;
    bool initSuccess;
};

CPUPanoramaPreviewTask::Impl::Impl()
{
    clear();
}

CPUPanoramaPreviewTask::Impl::~Impl()
{
    clear();
}

bool CPUPanoramaPreviewTask::Impl::init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile,
    int dstWidth, int dstHeight)
{
    clear();

    if (srcVideoFiles.empty())
    {
        ptlprintf("Error in %s, size of srcVideoFiles empty\n", __FUNCTION__);
        return false;
    }

    numVideos = srcVideoFiles.size();

    ptlprintf("Info in %s, open videos and set to the correct frames\n", __FUNCTION__);
    bool ok = false;
    int validFrameCount;
    int audioIndex;
    ok = prepareSrcVideos(srcVideoFiles, true, std::vector<int>(), -1, readers, audioIndex, srcSize, validFrameCount);
    if (!ok)
    {
        ptlprintf("Error in %s, could not open video file(s)\n", __FUNCTION__);
        return false;
    }
    ptlprintf("Info in %s, open videos done\n", __FUNCTION__);
    
    ptlprintf("Info in %s, prepare for reproject and blend\n", __FUNCTION__);
    dstSize.width = dstWidth;
    dstSize.height = dstHeight;
    std::vector<PhotoParam> params;
    if (!loadPhotoParams(cameraParamFile, params))
    {
        ptlprintf("Error in %s, failed to load params\n", __FUNCTION__);
        return false;
    }
    if (params.size() < numVideos)
    {
        ptlprintf("Error in %s, params.size() < numVideos\n", __FUNCTION__);
        return false;
    }
    else if (params.size() > numVideos)
    {
        ptlprintf("Warning in %s, params.size() > numVideos\n", __FUNCTION__);
    }
    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, dstMasks);
    ok = blender.prepare(dstMasks, 16, 2);
    //ok = blender.prepare(dstMasks, 50);
    if (!ok)
    {
        ptlprintf("Error in %s, blender prepare failed\n", __FUNCTION__);
        return false;
    }
    ptlprintf("Info in %s, prepare finish\n", __FUNCTION__);

    initSuccess = true;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::reset(const std::string& cameraParamFile)
{
    if (!initSuccess)
        return false;

    std::vector<PhotoParam> params;
    if (!loadPhotoParams(cameraParamFile, params))
    {
        ptlprintf("Error in %s, failed to load params\n", __FUNCTION__);
        return false;
    }
    if (params.size() < numVideos)
    {
        ptlprintf("Error in %s, params.size() < numVideos\n", __FUNCTION__);
        return false;
    }
    else if (params.size() > numVideos)
    {
        ptlprintf("Warning in %s, params.size() > numVideos\n", __FUNCTION__);
    }
    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, dstMasks);

    bool ok = blender.prepare(dstMasks, 16, 2);
    //ok = blender.prepare(dstMasks, 50);
    if (!ok)
    {
        ptlprintf("Error in %s, blender prepare failed\n", __FUNCTION__);
        return false;
    }

    return true;
}

bool CPUPanoramaPreviewTask::Impl::seek(const std::vector<long long int>& timeStamps)
{
    if (!initSuccess)
        return false;

    if (timeStamps.size() != numVideos)
        return false;

    bool ok = true;
    for (int i = 0; i < numVideos; i++)
    {
        if (!readers[i].seek(timeStamps[i], avp::VIDEO))
        {
            ok = false;
            break;
        }
    }
    return ok;
}

bool CPUPanoramaPreviewTask::Impl::stitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst, int frameIncrement)
{
    if (!initSuccess)
        return false;

    if (frameIncrement <= 0 || frameIncrement > 10)
        frameIncrement = 1;

    //ptlprintf("In %s, begin read frame\n", __FUNCTION__);
    frames.resize(numVideos);
    images.resize(numVideos);
    timeStamps.resize(numVideos);
    bool ok = true;
    for (int i = 0; i < numVideos; i++)
    {
        for (int j = 0; j < frameIncrement; j++)
        {
            if (!readers[i].read(frames[i]))
            {
                ok = false;
                break;
            }
        }
        if (!ok)
            break;

        images[i] = cv::Mat(frames[i].height, frames[i].width, CV_8UC3, frames[i].data, frames[i].step);
        timeStamps[i] = frames[i].timeStamp;
    }
    if (!ok)
        return false;

    //ptlprintf("In %s, read frame success\n", __FUNCTION__);
    reprojectParallel(images, reprojImages, dstSrcMaps);
    //ptlprintf("In %s, reproject success\n", __FUNCTION__);
    blender.blend(reprojImages, blendImage);
    src = images;
    dst = blendImage;
    //ptlprintf("In %s, stitch success\n", __FUNCTION__);
    return true;
}

bool CPUPanoramaPreviewTask::Impl::getMasks(std::vector<cv::Mat>& masks)
{
    if (!initSuccess)
        return false;

    masks = dstMasks;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::readNextAndReprojectForAll(std::vector<cv::Mat>& dst)
{
    if (!initSuccess)
        return false;

    frames.resize(numVideos);
    images.resize(numVideos);
    bool ok = true;
    for (int i = 0; i < numVideos; i++)
    {
        if (!readers[i].read(frames[i]))
        {
            ok = false;
            break;
        }
        images[i] = cv::Mat(frames[i].height, frames[i].width, CV_8UC3, frames[i].data, frames[i].step);
    }
    if (!ok)
        return false;

    //ptlprintf("In %s, read frame success\n", __FUNCTION__);
    reprojectParallel(images, reprojImages, dstSrcMaps);
    dst = reprojImages;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::readNextAndReprojectForOne(int index, cv::Mat& image)
{
    if (!initSuccess)
        return false;

    if (index < 0 || index >= numVideos)
        return false;

    if (images.size() != numVideos)
        return false;

    for (int i = 0; i < numVideos; i++)
    {
        if (!images[i].data || images[i].size() != srcSize)
            return false;
    }

    if (!readers[index].read(frames[index]))
        return false;

    images[index] = cv::Mat(frames[index].height, frames[index].width, CV_8UC3, frames[index].data, frames[index].step);
    reprojectParallel(images[index], reprojImages[index], dstSrcMaps[index]);
    image = reprojImages[index];
    return true;
}

bool CPUPanoramaPreviewTask::Impl::readPrevAndReprojectForOne(int index, cv::Mat& image)
{
    if (!initSuccess)
        return false;

    if (index < 0 || index >= numVideos)
        return false;

    if (images.size() != numVideos)
        return false;

    for (int i = 0; i < numVideos; i++)
    {
        if (!images[i].data || images[i].size() != srcSize)
            return false;
    }

    long long int timeIncUnit = 1000000 / readers[index].getVideoFps() + 0.5;
    if (!readers[index].seek(frames[index].timeStamp - timeIncUnit, avp::VIDEO))
        return false;

    if (!readers[index].read(frames[index]))
        return false;

    images[index] = cv::Mat(frames[index].height, frames[index].width, CV_8UC3, frames[index].data, frames[index].step);
    reprojectParallel(images[index], reprojImages[index], dstSrcMaps[index]);
    image = reprojImages[index];
    return true;
}

void CPUPanoramaPreviewTask::Impl::clear()
{
    numVideos = 0;
    readers.clear();
    initSuccess = false;
}

CPUPanoramaPreviewTask::CPUPanoramaPreviewTask()
{
    ptrImpl.reset(new Impl);
}

CPUPanoramaPreviewTask::~CPUPanoramaPreviewTask()
{

}

bool CPUPanoramaPreviewTask::init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile,
    int dstWidth, int dstHeight)
{
    return ptrImpl->init(srcVideoFiles, cameraParamFile, dstWidth, dstHeight);
}

bool CPUPanoramaPreviewTask::reset(const std::string& cameraParamFile)
{
    return ptrImpl->reset(cameraParamFile);
}

bool CPUPanoramaPreviewTask::seek(const std::vector<long long int>& timeStamps)
{
    return ptrImpl->seek(timeStamps);
}

bool CPUPanoramaPreviewTask::stitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst, int frameIncrement)
{
    return ptrImpl->stitch(src, timeStamps, dst, frameIncrement);
}

bool CPUPanoramaPreviewTask::getMasks(std::vector<cv::Mat>& masks)
{
    return ptrImpl->getMasks(masks);
}

bool CPUPanoramaPreviewTask::readNextAndReprojectForAll(std::vector<cv::Mat>& images)
{
    return ptrImpl->readNextAndReprojectForAll(images);
}

bool CPUPanoramaPreviewTask::readNextAndReprojectForOne(int index, cv::Mat& image)
{
    return ptrImpl->readNextAndReprojectForOne(index, image);
}

bool CPUPanoramaPreviewTask::readPrevAndReprojectForOne(int index, cv::Mat& image)
{
    return ptrImpl->readPrevAndReprojectForOne(index, image);
}

struct CudaPanoramaPreviewTask::Impl
{
    Impl();
    ~Impl();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile,
        int dstWidth, int dstHeight);
    bool reset(const std::string& cameraParamFile);
    bool seek(const std::vector<long long int>& timeStamps);
    bool stitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst, int frameIncrement = 1);

    void clear();

    int numVideos;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader> readers;
    std::unique_ptr<PanoramaRender> ptrRender;
    cv::Mat blendImage;
    bool initSuccess;
};

CudaPanoramaPreviewTask::Impl::Impl()
{
    clear();
}

CudaPanoramaPreviewTask::Impl::~Impl()
{
    clear();
}

bool CudaPanoramaPreviewTask::Impl::init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile,
    int dstWidth, int dstHeight)
{
    clear();

    if (srcVideoFiles.empty())
    {
        ptlprintf("Error in %s, size of srcVideoFiles empty\n", __FUNCTION__);
        return false;
    }
    numVideos = srcVideoFiles.size();

    ptlprintf("Info in %s, open videos and set to the correct frames\n", __FUNCTION__);
    bool ok = false;
    int validFrameCount;
    int audioIndex;
    ok = prepareSrcVideos(srcVideoFiles, false, std::vector<int>(), -1, readers, audioIndex, srcSize, validFrameCount);
    if (!ok)
    {
        ptlprintf("Error in %s, could not open video file(s)\n", __FUNCTION__);
        return false;
    }
    ptlprintf("Info in %s, open videos done\n", __FUNCTION__);

    ptlprintf("Info in %s, prepare for reproject and blend\n", __FUNCTION__);
    dstSize.width = dstWidth;
    dstSize.height = dstHeight;
    ptrRender.reset(new CudaMultiCameraPanoramaRender);
    ok = ptrRender->prepare(cameraParamFile, PanoramaRender::BlendTypeMultiband, srcSize, dstSize);
    if (!ok)
    {
        ptlprintf("Error in %s, prepare failed\n", __FUNCTION__);
        return false;
    }
    ptlprintf("Info in %s, prepare finish\n", __FUNCTION__);

    initSuccess = true;
    return true;
}

bool CudaPanoramaPreviewTask::Impl::reset(const std::string& cameraParamFile)
{
    if (!initSuccess)
        return false;

    bool ok = ptrRender->prepare(cameraParamFile, PanoramaRender::BlendTypeMultiband, srcSize, dstSize);
    if (!ok)
    {
        ptlprintf("Error in %s, prepare failed\n", __FUNCTION__);
        return false;
    }

    return true;
}

bool CudaPanoramaPreviewTask::Impl::seek(const std::vector<long long int>& timeStamps)
{
    if (!initSuccess)
        return false;

    if (timeStamps.size() != numVideos)
        return false;

    bool ok = true;
    for (int i = 0; i < numVideos; i++)
    {
        if (!readers[i].seek(timeStamps[i], avp::VIDEO))
        {
            ok = false;
            break;
        }
    }
    return ok;
}

bool CudaPanoramaPreviewTask::Impl::stitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst, int frameIncrement)
{
    if (!initSuccess)
        return false;

    if (frameIncrement <= 0 || frameIncrement > 10)
        frameIncrement = 1;

    std::vector<cv::Mat> images(numVideos);
    timeStamps.resize(numVideos);
    bool ok = true;
    for (int i = 0; i < numVideos; i++)
    {
        avp::AudioVideoFrame frame;
        for (int j = 0; j < frameIncrement; j++)
        {
            if (!readers[i].read(frame))
            {
                ok = false;
                break;
            }
        }
        if (!ok)
            break;

        images[i] = cv::Mat(frame.height, frame.width, CV_8UC4, frame.data, frame.step);
        timeStamps[i] = frame.timeStamp;
    }
    if (!ok)
        return false;

   ok = ptrRender->render(images, blendImage);
   if (!ok)
       return false;
   src = images;
   dst = blendImage;
   return true;
}

void CudaPanoramaPreviewTask::Impl::clear()
{
    numVideos = 0;
    readers.clear();
    ptrRender.reset(0);
    initSuccess = false;
}

CudaPanoramaPreviewTask::CudaPanoramaPreviewTask()
{
    ptrImpl.reset(new Impl);
}

CudaPanoramaPreviewTask::~CudaPanoramaPreviewTask()
{

}

bool CudaPanoramaPreviewTask::init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile,
    int dstWidth, int dstHeight)
{
    return ptrImpl->init(srcVideoFiles, cameraParamFile, dstWidth, dstHeight);
}

bool CudaPanoramaPreviewTask::reset(const std::string& cameraParamFile)
{
    return ptrImpl->reset(cameraParamFile);
}

bool CudaPanoramaPreviewTask::seek(const std::vector<long long int>& timeStamps)
{
    return ptrImpl->seek(timeStamps);
}

bool CudaPanoramaPreviewTask::stitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst, int frameIncrement)
{
    return ptrImpl->stitch(src, timeStamps, dst, frameIncrement);
}