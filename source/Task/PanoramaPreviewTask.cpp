#include "PanoramaTask.h"
#include "PanoramaTaskUtil.h"
#include "ZReproject.h"
#include "ZBlend.h"
#include "RicohUtil.h"
#include "opencv2/highgui.hpp"

struct CPUPanoramaPreviewTask::Impl
{
    Impl();
    ~Impl();

    bool init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile,
        int dstWidth, int dstHeight);
    bool reset(const std::string& cameraParamFile);
    bool seek(const std::vector<long long int>& timeStamps);
    bool stitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst, int frameIncrement);
    bool restitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst);

    bool isValid() const;
    int getNumSourceVideos() const;
    double getVideoFrameRate() const;
    bool getMasks(std::vector<cv::Mat>& masks) const;
    bool getUniqueMasks(std::vector<cv::Mat>& masks) const;

    bool getCurrReprojectForAll(std::vector<cv::Mat>& images, std::vector<long long int>& timeStamps) const;
    bool reReprojectForAll(std::vector<cv::Mat>& images, std::vector<long long int>& timeStamps);
    bool readNextAndReprojectForAll(std::vector<cv::Mat>& images, std::vector<long long int>& timeStamps);
    bool readNextAndReprojectForOne(int index, cv::Mat& image, long long int& timeStamp);
    bool readPrevAndReprojectForOne(int index, cv::Mat& image, long long int& timeStamp);

    bool setCustomMaskForOne(int index, long long int begInc, long long int endExc, const cv::Mat& mask);
    void eraseCustomMaskForOne(int index, long long int begInc, long long int endExc, long long int precision = 1000);
    void eraseAllMasksForOne(int index);

    bool getCustomMaskIfHasOrUniqueMaskForOne(int index, long long int timeStamp, cv::Mat& mask) const;
    bool getCustomMasksIfHaveOrUniqueMasksForAll(const std::vector<long long int>& timeStamps, std::vector<cv::Mat>& masks) const;
    bool getAllCustomMasksForOne(int index, std::vector<long long int>& begIncs, std::vector<long long int>& endExcs,
        std::vector<cv::Mat>& masks);

    void clear();

    int numVideos;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader3> readers;
    std::vector<cv::Mat> dstSrcMaps, dstMasks, dstUniqueMasks, currMasks;
    std::vector<CustomIntervaledMasks> customMasks;
    TilingMultibandBlendFast blender;
    std::vector<cv::Mat> images, reprojImages;
    std::vector<avp::AudioVideoFrame2> frames;
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

    customMasks.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
        customMasks[i].init(dstSize.width, dstSize.height);

    ok = blender.prepare(dstMasks, 16, 2);
    //ok = blender.prepare(dstMasks, 50);
    blender.getUniqueMasks(dstUniqueMasks);
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

        images[i] = cv::Mat(frames[i].height, frames[i].width, CV_8UC3, frames[i].data[0], frames[i].steps[0]);
        timeStamps[i] = frames[i].timeStamp;
    }
    if (!ok)
        return false;

    reprojectParallel(images, reprojImages, dstSrcMaps);

    bool useCustomMask = false;
    currMasks.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        if (customMasks[i].getMask(frames[i].timeStamp, currMasks[i]))
            useCustomMask = true;
        else
            currMasks[i] = dstUniqueMasks[i];
    }

    if (useCustomMask)
        blender.blend(reprojImages, currMasks, blendImage);
    else
        blender.blend(reprojImages, blendImage);
    src = images;
    dst = blendImage;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::restitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst)
{
    if (!initSuccess)
        return false;

    if (images.size() != numVideos)
        return false;

    reprojectParallel(images, reprojImages, dstSrcMaps);

    bool useCustomMask = false;
    currMasks.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        if (customMasks[i].getMask(frames[i].timeStamp, currMasks[i]))
            useCustomMask = true;
        else
            currMasks[i] = dstUniqueMasks[i];
    }

    if (useCustomMask)
        blender.blend(reprojImages, currMasks, blendImage);
    else
        blender.blend(reprojImages, blendImage);
    src = images;
    dst = blendImage;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::isValid() const
{
    return initSuccess != 0;
}

int CPUPanoramaPreviewTask::Impl::getNumSourceVideos() const
{
    if (!initSuccess)
        return 0;
    return numVideos;
}

double CPUPanoramaPreviewTask::Impl::getVideoFrameRate() const
{
    if (!initSuccess)
        return 0;
    return readers[0].getVideoFrameRate();
}

bool CPUPanoramaPreviewTask::Impl::getMasks(std::vector<cv::Mat>& masks) const
{
    if (!initSuccess)
        return false;

    masks = dstMasks;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::getUniqueMasks(std::vector<cv::Mat>& masks) const
{
    if (!initSuccess)
        return false;

    blender.getUniqueMasks(masks);
    return masks.size() == numVideos;
}

bool CPUPanoramaPreviewTask::Impl::getCurrReprojectForAll(std::vector<cv::Mat>& dst, std::vector<long long int>& timeStamps) const
{
    if (!initSuccess)
        return false;

    if (images.empty() || frames.empty() || reprojImages.empty())
        return false;

    if (reprojImages.size() != numVideos)
        return false;

    dst = reprojImages;
    timeStamps.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
        timeStamps[i] = frames[i].timeStamp;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::reReprojectForAll(std::vector<cv::Mat>& dst, std::vector<long long int>& timeStamps)
{
    if (!initSuccess)
        return false;

    if (images.empty() || frames.empty() || reprojImages.empty())
        return false;

    reprojectParallel(images, reprojImages, dstSrcMaps);

    bool useCustomMask = false;
    currMasks.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        if (customMasks[i].getMask(frames[i].timeStamp, currMasks[i]))
            useCustomMask = true;
        else
            currMasks[i] = dstUniqueMasks[i];
    }

    if (useCustomMask)
        blender.blend(reprojImages, currMasks, blendImage);
    else
        blender.blend(reprojImages, blendImage);
    dst = reprojImages;
    timeStamps.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
        timeStamps[i] = frames[i].timeStamp;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::readNextAndReprojectForAll(std::vector<cv::Mat>& dst, std::vector<long long int>& timeStamps)
{
    if (!initSuccess)
        return false;

    frames.resize(numVideos);
    images.resize(numVideos);
    timeStamps.resize(numVideos);
    bool ok = true;
    for (int i = 0; i < numVideos; i++)
    {
        if (!readers[i].read(frames[i]))
        {
            ok = false;
            break;
        }
        images[i] = cv::Mat(frames[i].height, frames[i].width, CV_8UC3, frames[i].data[0], frames[i].steps[0]);
        timeStamps[i] = frames[i].timeStamp;
    }
    if (!ok)
        return false;

    reprojectParallel(images, reprojImages, dstSrcMaps);
    dst = reprojImages;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::readNextAndReprojectForOne(int index, cv::Mat& image, long long int& timeStamp)
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

    images[index] = cv::Mat(frames[index].height, frames[index].width, CV_8UC3, frames[index].data[0], frames[index].steps[0]);
    reprojectParallel(images[index], reprojImages[index], dstSrcMaps[index]);
    image = reprojImages[index];
    timeStamp = frames[index].timeStamp;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::readPrevAndReprojectForOne(int index, cv::Mat& image, long long int& timeStamp)
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

    long long int timeIncUnit = 1000000 / readers[index].getVideoFrameRate() + 0.5;
    if (!readers[index].seek(frames[index].timeStamp - timeIncUnit, avp::VIDEO))
        return false;

    if (!readers[index].read(frames[index]))
        return false;

    images[index] = cv::Mat(frames[index].height, frames[index].width, CV_8UC3, frames[index].data[0], frames[index].steps[0]);
    reprojectParallel(images[index], reprojImages[index], dstSrcMaps[index]);
    image = reprojImages[index];
    timeStamp = frames[index].timeStamp;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::setCustomMaskForOne(int index, long long int begInc, long long int endExc, const cv::Mat& mask)
{
    if (!initSuccess)
        return false;

    if (index < 0 || index >= numVideos)
        return false;

    return customMasks[index].addMask(begInc, endExc, mask);
}

void CPUPanoramaPreviewTask::Impl::eraseCustomMaskForOne(int index, long long int begInc, long long int endExc, long long int precision)
{
    if (!initSuccess)
        return;

    if (index < 0 || index >= numVideos)
        return;

    customMasks[index].clearMask(begInc, endExc, precision);
}

void CPUPanoramaPreviewTask::Impl::eraseAllMasksForOne(int index)
{
    if (!initSuccess)
        return;

    if (index < 0 || index >= numVideos)
        return;

    customMasks[index].clearAllMasks();
}

bool CPUPanoramaPreviewTask::Impl::getCustomMaskIfHasOrUniqueMaskForOne(int index, long long int timeStamp, cv::Mat& mask) const
{
    if (!initSuccess)
        return false;

    if (index < 0 || index >= numVideos)
        return false;

    if (!customMasks[index].getMask(timeStamp, mask))
        mask = dstUniqueMasks[index];
    return true;
}

bool CPUPanoramaPreviewTask::Impl::getCustomMasksIfHaveOrUniqueMasksForAll(const std::vector<long long int>& timeStamps, std::vector<cv::Mat>& masks) const
{
    if (!initSuccess)
        return false;

    if (timeStamps.size() != numVideos)
        return false;

    masks.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        if (!customMasks[i].getMask(timeStamps[i], masks[i]))
            masks[i] = dstUniqueMasks[i];
    }
    return true;
}

bool CPUPanoramaPreviewTask::Impl::getAllCustomMasksForOne(int index, std::vector<long long int>& begIncs, 
    std::vector<long long int>& endExcs, std::vector<cv::Mat>& masks)
{
    begIncs.clear();
    endExcs.clear();
    masks.clear();

    if (!initSuccess)
        return false;

    if (index < 0 || index >= numVideos)
        return false;

    int size = customMasks[index].masks.size();
    begIncs.resize(size);
    endExcs.resize(size);
    masks.resize(size);
    for (int i = 0; i < size; i++)
    {
        begIncs[i] = customMasks[index].masks[i].begInc;
        endExcs[i] = customMasks[index].masks[i].endExc;
        masks[i] = customMasks[index].masks[i].mask;
    }
    return true;
}

void CPUPanoramaPreviewTask::Impl::clear()
{
    numVideos = 0;
    readers.clear();
    initSuccess = false;

    dstSrcMaps.clear(); 
    dstMasks.clear();
    dstUniqueMasks.clear();
    currMasks.clear();
    customMasks.clear();
    images.clear();
    reprojImages.clear();
    frames.clear();
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

bool CPUPanoramaPreviewTask::restitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst)
{
    return ptrImpl->restitch(src, timeStamps, dst);
}

bool CPUPanoramaPreviewTask::isValid() const
{
    return ptrImpl->isValid();
}

int CPUPanoramaPreviewTask::getNumSourceVideos() const
{
    return ptrImpl->getNumSourceVideos();
}

double CPUPanoramaPreviewTask::getVideoFrameRate() const
{
    return ptrImpl->getVideoFrameRate();
}

bool CPUPanoramaPreviewTask::getMasks(std::vector<cv::Mat>& masks) const
{
    return ptrImpl->getMasks(masks);
}

bool CPUPanoramaPreviewTask::getUniqueMasks(std::vector<cv::Mat>& masks) const
{
    return ptrImpl->getUniqueMasks(masks);
}

bool CPUPanoramaPreviewTask::getCurrReprojectForAll(std::vector<cv::Mat>& images, std::vector<long long int>& timeStamps) const
{
    return ptrImpl->getCurrReprojectForAll(images, timeStamps);
}

bool CPUPanoramaPreviewTask::reReprojectForAll(std::vector<cv::Mat>& images, std::vector<long long int>& timeStamps)
{
    return ptrImpl->reReprojectForAll(images, timeStamps);
}

bool CPUPanoramaPreviewTask::readNextAndReprojectForAll(std::vector<cv::Mat>& images, std::vector<long long int>& timeStamps)
{
    return ptrImpl->readNextAndReprojectForAll(images, timeStamps);
}

bool CPUPanoramaPreviewTask::readNextAndReprojectForOne(int index, cv::Mat& image, long long int& timeStamp)
{
    return ptrImpl->readNextAndReprojectForOne(index, image, timeStamp);
}

bool CPUPanoramaPreviewTask::readPrevAndReprojectForOne(int index, cv::Mat& image, long long int& timeStamp)
{
    return ptrImpl->readPrevAndReprojectForOne(index, image, timeStamp);
}

bool CPUPanoramaPreviewTask::setCustomMaskForOne(int index, long long int begInc, long long int endExc, const cv::Mat& mask)
{
    return ptrImpl->setCustomMaskForOne(index, begInc, endExc, mask);
}

void CPUPanoramaPreviewTask::eraseCustomMaskForOne(int index, long long int begInc, long long int endExc, long long int precision)
{
    ptrImpl->eraseCustomMaskForOne(index, begInc, endExc, precision);
}

void CPUPanoramaPreviewTask::eraseAllMasksForOne(int index)
{
    ptrImpl->eraseAllMasksForOne(index);
}

bool CPUPanoramaPreviewTask::getCustomMaskIfHasOrUniqueMaskForOne(int index, long long int timeStamp, cv::Mat& mask) const
{
    return ptrImpl->getCustomMaskIfHasOrUniqueMaskForOne(index, timeStamp, mask);
}

bool CPUPanoramaPreviewTask::getCustomMasksIfHaveOrUniqueMasksForAll(const std::vector<long long int>& timeStamps, std::vector<cv::Mat>& masks) const
{
    return ptrImpl->getCustomMasksIfHaveOrUniqueMasksForAll(timeStamps, masks);
}

bool CPUPanoramaPreviewTask::getAllCustomMasksForOne(int index, std::vector<long long int>& begIncs, std::vector<long long int>& endExcs,
    std::vector<cv::Mat>& masks) const
{
    return ptrImpl->getAllCustomMasksForOne(index, begIncs, endExcs, masks);
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
    std::vector<avp::AudioVideoReader3> readers;
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
        avp::AudioVideoFrame2 frame;
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

        images[i] = cv::Mat(frame.height, frame.width, CV_8UC4, frame.data[0], frame.steps[0]);
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