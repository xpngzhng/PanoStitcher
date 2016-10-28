#include "PanoramaTask.h"
#include "PanoramaTaskUtil.h"
#include "RicohUtil.h"
#include "Warp/ZReproject.h"
#include "Blend/ZBlend.h"
#include "Tool/Print.h"
#include "opencv2/highgui.hpp"

struct CPUPanoramaPreviewTask::Impl
{
    Impl();
    ~Impl();

    bool init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile,
        int dstWidth, int dstHeight, int activateMbBlend, int mbBlendNumLevels, int lBlendRadius);
    bool reset(const std::string& cameraParamFile);

    bool setBlendType(bool multibandBlend);
    bool setMultibandBlendParam(int numLevels);
    bool setLinearBlendParam(int radius);
    bool getBlendType(bool& multibandBlend) const;
    bool getMultibandBlendParam(int& numLevels) const;
    bool getLinearBlendParam(int& radius) const;

    bool isValid() const;
    int getNumSourceVideos() const;
    double getVideoFrameRate() const;
    bool getStichSize(int& width, int& height) const;
    bool getMasks(std::vector<cv::Mat>& masks) const;
    bool getUniqueMasks(std::vector<cv::Mat>& masks) const;

    bool seek(const std::vector<int>& indexes);
    bool stitch(std::vector<cv::Mat>& src, std::vector<int>& indexes, cv::Mat& dst, int frameIncrement);
    bool restitch(std::vector<cv::Mat>& src, std::vector<int>& indexes, cv::Mat& dst);
    bool getCurrStitch(std::vector<cv::Mat>& src, std::vector<int>& indexes, cv::Mat& dst);

    bool getCurrReprojectForAll(std::vector<cv::Mat>& images, std::vector<int>& indexes) const;
    bool reReprojectForAll(std::vector<cv::Mat>& images, std::vector<int>& indexes);
    bool readNextAndReprojectForAll(std::vector<cv::Mat>& images, std::vector<int>& indexes);
    bool readNextAndReprojectForOne(int videoIndex, cv::Mat& image, int& frameIndex);
    bool readPrevAndReprojectForOne(int videoIndex, cv::Mat& image, int& frameIndex);

    bool setCustomMaskForOne(int videoIndex, int begFrameIndexInc, int endFrameIndexExc, const cv::Mat& mask);
    void eraseCustomMaskForOne(int videoIndex, int begFrameIndexInc, int endFrameIndexExc);
    void eraseAllMasksForOne(int index);

    bool getCustomMaskIfHasOrUniqueMaskForOne(int videoIndex, int frameIndex, cv::Mat& mask) const;
    bool getCustomMasksIfHaveOrUniqueMasksForAll(const std::vector<int>& indexes, std::vector<cv::Mat>& masks) const;
    bool getAllCustomMasksForOne(int videoIndex, std::vector<int>& begFrameIndexesInc, std::vector<int>& endFrameIndexesInc,
        std::vector<cv::Mat>& masks) const;

    bool correctExposureWhiteBalance(bool whiteBalance, std::vector<double>& exposures,
        std::vector<double>& redRatios, std::vector<double>& blueRatios);
    bool getExposureWhiteBalance(std::vector<double>& exposures,
        std::vector<double>& redRatios, std::vector<double>& blueRatios);
    bool setExposureWhiteBalance(const std::vector<double>& exposures,
        const std::vector<double>& redRatios, const std::vector<double>& blueRatios);

    void clear();

    int numVideos;
    double frameIntervalInMicroSec;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader3> readers;
    ImageVisualCorrect2 visualCorrect;
    std::vector<double> es, rs, bs;
    std::vector<std::vector<std::vector<unsigned char> > > luts;
    std::vector<cv::Mat> dstSrcMaps, dstMasks, dstUniqueMasks, currMasks;
    std::vector<CustomIntervaledMasks> customMasks;
    TilingMultibandBlendFast mbBlender;
    TilingLinearBlend lBlender;
    std::vector<cv::Mat> images, correctImages, reprojImages;
    std::vector<avp::AudioVideoFrame2> frames;
    cv::Mat blendImage;
    int blendNumLevels;
    int blendRadius;
    bool isMultibandBlend;
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
    int dstWidth, int dstHeight, int activateMbBlend, int mbBlendNumLevels, int lBlendRadius)
{
    clear();

    if (srcVideoFiles.empty())
    {
        ztool::lprintf("Error in %s, size of srcVideoFiles empty\n", __FUNCTION__);
        return false;
    }

    numVideos = srcVideoFiles.size();

    ztool::lprintf("Info in %s, open videos and set to the correct frames\n", __FUNCTION__);
    bool ok = false;
    int validFrameCount;
    int audioIndex;
    ok = prepareSrcVideos(srcVideoFiles, avp::PixelTypeBGR24, std::vector<int>(), -1, readers, audioIndex, srcSize, validFrameCount);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not open video file(s)\n", __FUNCTION__);
        return false;
    }
    ztool::lprintf("Info in %s, open videos done\n", __FUNCTION__);
    
    ztool::lprintf("Info in %s, prepare for reproject and blend\n", __FUNCTION__);
    dstSize.width = dstWidth;
    dstSize.height = dstHeight;
    std::vector<PhotoParam> params;
    if (!loadPhotoParams(cameraParamFile, params))
    {
        ztool::lprintf("Error in %s, failed to load params\n", __FUNCTION__);
        return false;
    }
    if (params.size() < numVideos)
    {
        ztool::lprintf("Error in %s, params.size() < numVideos\n", __FUNCTION__);
        return false;
    }
    else if (params.size() > numVideos)
    {
        ztool::lprintf("Warning in %s, params.size() > numVideos\n", __FUNCTION__);
    }
    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, dstMasks);

    if (!visualCorrect.prepare(cameraParamFile))
    {
        ztool::lprintf("Error in %s, visual correct prepare failed\n", __FUNCTION__);
        return false;
    }
    es.resize(numVideos);
    rs.resize(numVideos);
    bs.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        es[i] = 1;
        rs[i] = 1;
        bs[i] = 1;
    }

    customMasks.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
        customMasks[i].init(dstSize.width, dstSize.height);

    blendNumLevels = mbBlendNumLevels;
    blendRadius = lBlendRadius;
    ok = mbBlender.prepare(dstMasks, blendNumLevels, 2);
    if (!ok)
    {
        ztool::lprintf("Error in %s, multiband blender prepare failed\n", __FUNCTION__);
        return false;
    }
    mbBlender.getUniqueMasks(dstUniqueMasks);
    ok = lBlender.prepare(dstMasks, blendRadius);
    if (!ok)
    {
        ztool::lprintf("Error in %s, linear blender prepare failed\n", __FUNCTION__);
        return false;
    }

    isMultibandBlend = activateMbBlend;

    ztool::lprintf("Info in %s, prepare finish\n", __FUNCTION__);

    frameIntervalInMicroSec = 1000000.0 / readers[0].getVideoFrameRate();

    initSuccess = true;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::reset(const std::string& cameraParamFile)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not reset\n", __FUNCTION__);
        return false;
    }

    std::vector<PhotoParam> params;
    if (!loadPhotoParams(cameraParamFile, params))
    {
        ztool::lprintf("Error in %s, failed to load params\n", __FUNCTION__);
        return false;
    }
    if (params.size() < numVideos)
    {
        ztool::lprintf("Error in %s, params.size() < numVideos\n", __FUNCTION__);
        return false;
    }
    else if (params.size() > numVideos)
    {
        ztool::lprintf("Warning in %s, params.size() > numVideos\n", __FUNCTION__);
    }
    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, dstMasks);

    bool ok = false;
    ok = mbBlender.prepare(dstMasks, blendNumLevels, 2);
    if (!ok)
    {
        ztool::lprintf("Error in %s, multiband blender prepare failed\n", __FUNCTION__);
        return false;
    }
    mbBlender.getUniqueMasks(dstUniqueMasks);
    ok = lBlender.prepare(dstMasks, blendRadius);
    if (!ok)
    {
        ztool::lprintf("Error in %s, linear blender prepare failed\n", __FUNCTION__);
        return false;
    }

    return true;
}

bool CPUPanoramaPreviewTask::Impl::setBlendType(bool multibandBlend)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    isMultibandBlend = multibandBlend;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::setMultibandBlendParam(int param)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    bool ok = false;
        ok = mbBlender.prepare(dstMasks, param, 2);
    if (!ok)
    {
        ztool::lprintf("Error in %s, reconfig multiband blender failed, param = %d\n",
            __FUNCTION__, param);
    }
    blendNumLevels = param;
    return ok;
}

bool CPUPanoramaPreviewTask::Impl::setLinearBlendParam(int param)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    bool ok = false;
    ok = lBlender.prepare(dstMasks, param);
    if (!ok)
    {
        ztool::lprintf("Error in %s, reconfig linear blender failed, param = %d\n",
            __FUNCTION__, param);
    }
    blendRadius = param;
    return ok;
}

bool CPUPanoramaPreviewTask::Impl::getBlendType(bool& multibandBlend) const
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }
    multibandBlend = isMultibandBlend;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::getMultibandBlendParam(int& param) const
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }
    param = blendNumLevels;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::getLinearBlendParam(int& param) const
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }
    param = blendRadius;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::seek(const std::vector<int>& indexes)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not seek\n", __FUNCTION__);
        return false;
    }

    if (indexes.size() != numVideos)
    {
        ztool::lprintf("Error in %s, indexes.size() = %d, require %d, unmatched, could not seek\n", 
            __FUNCTION__, indexes.size(), numVideos);
        return false;
    }

    bool ok = true;
    for (int i = 0; i < numVideos; i++)
    {
        if (!readers[i].seekByIndex(indexes[i], avp::VIDEO))
        {
            ok = false;
            break;
        }
    }
    return ok;
}

bool CPUPanoramaPreviewTask::Impl::stitch(std::vector<cv::Mat>& src, std::vector<int>& indexes, cv::Mat& dst, int frameIncrement)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not stitch\n", __FUNCTION__);
        return false;
    }

    if (frameIncrement <= 0 || frameIncrement > 10)
        frameIncrement = 1;

    //ztool::lprintf("In %s, begin read frame\n", __FUNCTION__);
    frames.resize(numVideos);
    images.resize(numVideos);
    correctImages.resize(numVideos);
    indexes.resize(numVideos);
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
        indexes[i] = frames[i].frameIndex;
    }
    if (!ok)
    {
        ztool::lprintf("Info in %s, read frame failed, perhaps went to the end of files\n", __FUNCTION__);
        return false;
    }

    if (luts.empty())
        reprojectParallel(images, reprojImages, dstSrcMaps);
    else
    {
        for (int i = 0; i < numVideos; i++)
            transform(images[i], correctImages[i], luts[i]);
        reprojectParallel(correctImages, reprojImages, dstSrcMaps);
    }

    bool useCustomMask = false;
    currMasks.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        if (customMasks[i].getMask2(frames[i].frameIndex, currMasks[i]))
            useCustomMask = true;
        else
            currMasks[i] = dstUniqueMasks[i];
    }

    if (isMultibandBlend)
    {
        if (useCustomMask)
            mbBlender.blend(reprojImages, currMasks, blendImage);
        else
            mbBlender.blend(reprojImages, blendImage);
    }
    else
        lBlender.blend(reprojImages, blendImage);
    src = images;
    dst = blendImage;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::restitch(std::vector<cv::Mat>& src, std::vector<int>& indexes, cv::Mat& dst)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not stitch\n", __FUNCTION__);
        return false;
    }

    if (images.size() != numVideos)
    {
        ztool::lprintf("Error in %s, restitch can be called after stitch at least once\n", __FUNCTION__);
        return false;
    }

    if (luts.empty())
        reprojectParallel(images, reprojImages, dstSrcMaps);
    else
    {
        for (int i = 0; i < numVideos; i++)
            transform(images[i], correctImages[i], luts[i]);
        reprojectParallel(correctImages, reprojImages, dstSrcMaps);
    }

    bool useCustomMask = false;
    currMasks.resize(numVideos);
    indexes.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        if (customMasks[i].getMask2(frames[i].frameIndex, currMasks[i]))
            useCustomMask = true;
        else
            currMasks[i] = dstUniqueMasks[i];
        indexes[i] = frames[i].frameIndex;
    }

    if (isMultibandBlend)
    {
        if (useCustomMask)
            mbBlender.blend(reprojImages, currMasks, blendImage);
        else
            mbBlender.blend(reprojImages, blendImage);
    }
    else
        lBlender.blend(reprojImages, blendImage);
    src = images;
    dst = blendImage;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::getCurrStitch(std::vector<cv::Mat>& src, std::vector<int>& indexes, cv::Mat& dst)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not stitch\n", __FUNCTION__);
        return false;
    }

    if (images.size() != numVideos)
    {
        ztool::lprintf("Error in %s, get stitch can be called after stitch at least once\n", __FUNCTION__);
        return false;
    }

    indexes.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
        indexes[i] = frames[i].frameIndex;
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

bool CPUPanoramaPreviewTask::Impl::getStichSize(int& width, int& height) const
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not get stitch size\n", __FUNCTION__);
        return false;
    }

    width = dstSize.width;
    height = dstSize.height;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::getMasks(std::vector<cv::Mat>& masks) const
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not get masks\n", __FUNCTION__);
        return false;
    }

    masks = dstMasks;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::getUniqueMasks(std::vector<cv::Mat>& masks) const
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not get unique masks\n", __FUNCTION__);
        return false;
    }

    mbBlender.getUniqueMasks(masks);
    return masks.size() == numVideos;
}

bool CPUPanoramaPreviewTask::Impl::getCurrReprojectForAll(std::vector<cv::Mat>& dst, std::vector<int>& indexes) const
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    if (images.empty() || frames.empty() || reprojImages.empty())
    {
        ztool::lprintf("Error in %s, at least one of images, frames and reprojImages empty, stitch function had not been called. "
            "This function can run correctly only after stitch has been call once\n", __FUNCTION__);
        return false;
    }

    if (reprojImages.size() != numVideos)
    {
        ztool::lprintf("Error in %s, reprojImages.size() != numVideos\n", __FUNCTION__);
        return false;
    }

    dst = reprojImages;
    indexes.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
        indexes[i] = frames[i].frameIndex;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::reReprojectForAll(std::vector<cv::Mat>& dst, std::vector<int>& indexes)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    if (images.empty() || frames.empty() || reprojImages.empty())
    {
        ztool::lprintf("Error in %s, at least one of images, frames and reprojImages empty, stitch function had not been called. "
            "This function can run correctly only after stitch has been call once\n", __FUNCTION__);
        return false;
    }

    if (luts.empty())
        reprojectParallel(images, reprojImages, dstSrcMaps);
    else
    {
        for (int i = 0; i < numVideos; i++)
            transform(images[i], correctImages[i], luts[i]);
        reprojectParallel(correctImages, reprojImages, dstSrcMaps);
    }

    bool useCustomMask = false;
    currMasks.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        if (customMasks[i].getMask2(frames[i].frameIndex, currMasks[i]))
            useCustomMask = true;
        else
            currMasks[i] = dstUniqueMasks[i];
    }

    if (isMultibandBlend)
    {
        if (useCustomMask)
            mbBlender.blend(reprojImages, currMasks, blendImage);
        else
            mbBlender.blend(reprojImages, blendImage);
    }
    else
        lBlender.blend(reprojImages, blendImage);
    dst = reprojImages;
    indexes.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
        indexes[i] = frames[i].frameIndex;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::readNextAndReprojectForAll(std::vector<cv::Mat>& dst, std::vector<int>& indexes)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    frames.resize(numVideos);
    images.resize(numVideos);
    indexes.resize(numVideos);
    bool ok = true;
    for (int i = 0; i < numVideos; i++)
    {
        if (!readers[i].read(frames[i]))
        {
            ok = false;
            break;
        }
        images[i] = cv::Mat(frames[i].height, frames[i].width, CV_8UC3, frames[i].data[0], frames[i].steps[0]);
        indexes[i] = frames[i].frameIndex;
    }
    if (!ok)
    {
        ztool::lprintf("Info in %s, read frame failed, perhaps went to the end of files\n", __FUNCTION__);
        return false;
    }

    if (luts.empty())
        reprojectParallel(images, reprojImages, dstSrcMaps);
    else
    {
        for (int i = 0; i < numVideos; i++)
            transform(images[i], correctImages[i], luts[i]);
        reprojectParallel(correctImages, reprojImages, dstSrcMaps);
    }
    dst = reprojImages;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::readNextAndReprojectForOne(int videoIndex, cv::Mat& image, int& frameIndex)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    if (videoIndex < 0 || videoIndex >= numVideos)
    {
        ztool::lprintf("Error in %s, index out of bound\n", __FUNCTION__);
        return false;
    }

    if (!readers[videoIndex].read(frames[videoIndex]))
    {
        printf("Error in %s, could not read frames from video source indexed %d, perhaps went to the end\n", __FUNCTION__, videoIndex);
        return false;
    }

    images[videoIndex] = cv::Mat(frames[videoIndex].height, frames[videoIndex].width, CV_8UC3, 
        frames[videoIndex].data[0], frames[videoIndex].steps[0]);
    if (luts.empty())
        reprojectParallel(images[videoIndex], reprojImages[videoIndex], dstSrcMaps[videoIndex]);
    else
    {
        transform(images[videoIndex], correctImages[videoIndex], luts[videoIndex]);
        reprojectParallel(correctImages[videoIndex], reprojImages[videoIndex], dstSrcMaps[videoIndex]);
    }
    image = reprojImages[videoIndex];
    frameIndex = frames[videoIndex].timeStamp;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::readPrevAndReprojectForOne(int videoIndex, cv::Mat& image, int& frameIndex)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    if (videoIndex < 0 || videoIndex >= numVideos)
    {
        ztool::lprintf("Error in %s, index out of bound\n", __FUNCTION__);
        return false;
    }

    long long int timeIncUnit = 1000000 / readers[videoIndex].getVideoFrameRate() + 0.5;
    if (!readers[videoIndex].seek(frames[videoIndex].timeStamp - timeIncUnit, avp::VIDEO))
    {
        ztool::lprintf("Error in %s, could not seek to the prev frame in video source indexed %d\n", __FUNCTION__, videoIndex);
        return false;
    }

    if (!readers[videoIndex].read(frames[videoIndex]))
    {
        ztool::lprintf("Error in %s, could not read frame in video source indexed %d\n", __FUNCTION__, videoIndex);
        return false;
    }

    images[videoIndex] = cv::Mat(frames[videoIndex].height, frames[videoIndex].width, CV_8UC3, 
        frames[videoIndex].data[0], frames[videoIndex].steps[0]);
    if (luts.empty())
        reprojectParallel(images[videoIndex], reprojImages[videoIndex], dstSrcMaps[videoIndex]);
    else
    {
        transform(images[videoIndex], correctImages[videoIndex], luts[videoIndex]);
        reprojectParallel(correctImages[videoIndex], reprojImages[videoIndex], dstSrcMaps[videoIndex]);
    }
    image = reprojImages[videoIndex];
    frameIndex = frames[videoIndex].timeStamp;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::setCustomMaskForOne(int videoIndex, int begFrameIndexInc, int endFrameIndexInc, const cv::Mat& mask)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    if (videoIndex < 0 || videoIndex >= numVideos)
    {
        ztool::lprintf("Error in %s, index out of bound\n", __FUNCTION__);
        return false;
    }

    return customMasks[videoIndex].addMask2(begFrameIndexInc, endFrameIndexInc, mask);
}

void CPUPanoramaPreviewTask::Impl::eraseCustomMaskForOne(int videoIndex, int begFrameIndexInc, int endFrameIndexExc)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return;
    }

    if (videoIndex < 0 || videoIndex >= numVideos)
    {
        ztool::lprintf("Error in %s, index out of bound\n", __FUNCTION__);
        return;
    }

    customMasks[videoIndex].clearMask2(begFrameIndexInc, endFrameIndexExc);
}

void CPUPanoramaPreviewTask::Impl::eraseAllMasksForOne(int index)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return;
    }

    if (index < 0 || index >= numVideos)
    {
        ztool::lprintf("Error in %s, index out of bound\n", __FUNCTION__);
        return;
    }

    customMasks[index].clearAllMasks();
}

bool CPUPanoramaPreviewTask::Impl::getCustomMaskIfHasOrUniqueMaskForOne(int videoIndex, int frameIndex, cv::Mat& mask) const
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    if (videoIndex < 0 || videoIndex >= numVideos)
    {
        ztool::lprintf("Error in %s, index out of bound\n", __FUNCTION__);
        return false;
    }

    if (!customMasks[videoIndex].getMask2(frameIndex, mask))
        mask = dstUniqueMasks[videoIndex];
    return true;
}

bool CPUPanoramaPreviewTask::Impl::getCustomMasksIfHaveOrUniqueMasksForAll(const std::vector<int>& indexes, std::vector<cv::Mat>& masks) const
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    if (indexes.size() != numVideos)
    {
        ztool::lprintf("Error in %s, indexes.size() = %d, required = %d, unmatch\n", __FUNCTION__, indexes.size(), numVideos);
        return false;
    }

    masks.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        if (!customMasks[i].getMask2(indexes[i], masks[i]))
            masks[i] = dstUniqueMasks[i];
    }
    return true;
}

bool CPUPanoramaPreviewTask::Impl::getAllCustomMasksForOne(int videoIndex, std::vector<int>& begFrameIndexesInc, 
    std::vector<int>& endFrameIndexesInc, std::vector<cv::Mat>& masks) const
{
    begFrameIndexesInc.clear();
    endFrameIndexesInc.clear();
    masks.clear();

    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    if (videoIndex < 0 || videoIndex >= numVideos)
    {
        ztool::lprintf("Error in %s, index out of bound\n", __FUNCTION__);
        return false;
    }

    int size = customMasks[videoIndex].masks.size();
    begFrameIndexesInc.resize(size);
    endFrameIndexesInc.resize(size);
    masks.resize(size);
    for (int i = 0; i < size; i++)
    {
        begFrameIndexesInc[i] = customMasks[videoIndex].masks[i].begIndexInc;
        endFrameIndexesInc[i] = customMasks[videoIndex].masks[i].endIndexInc;
        masks[i] = customMasks[videoIndex].masks[i].mask;
    }
    return true;
}

bool CPUPanoramaPreviewTask::Impl::correctExposureWhiteBalance(bool whiteBalance, std::vector<double>& exposures,
    std::vector<double>& redRatios, std::vector<double>& blueRatios)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    if (images.size() != numVideos)
    {
        ztool::lprintf("Error in %s, this function can be called after stitch at least once\n", __FUNCTION__);
        return false;
    }

    bool ret = true;
    if (whiteBalance)
        ret = visualCorrect.correct(images, exposures, redRatios, blueRatios);
    else
    {
        redRatios.resize(numVideos);
        blueRatios.resize(numVideos);
        for (int i = 0; i < numVideos; i++)
        {
            redRatios[i] = 1;
            blueRatios[i] = 1;
        }
        ret = visualCorrect.correct(images, exposures);
    }
    return ret;
}

bool CPUPanoramaPreviewTask::Impl::getExposureWhiteBalance(std::vector<double>& exposures,
    std::vector<double>& redRatios, std::vector<double>& blueRatios)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    exposures = es;
    redRatios = rs;
    blueRatios = bs;
    return true;
}

bool CPUPanoramaPreviewTask::Impl::setExposureWhiteBalance(const std::vector<double>& exposures,
    const std::vector<double>& redRatios, const std::vector<double>& blueRatios)
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not run this function\n", __FUNCTION__);
        return false;
    }

    if (exposures.size() != numVideos ||
        redRatios.size() != numVideos ||
        blueRatios.size() != numVideos)
    {
        ztool::lprintf("Error in %s, exposures, redRatios or blueRatios's size not equal to %d\n",
            __FUNCTION__, numVideos);
        return false;
    }

    es = exposures;
    rs = redRatios;
    bs = blueRatios;
    if (needCorrectExposureWhiteBalance(es, rs, bs))
        visualCorrect.getLUTs(es, rs, bs, luts);
    else
        luts.clear();

    return true;
}

void CPUPanoramaPreviewTask::Impl::clear()
{
    frameIntervalInMicroSec = 0;
    numVideos = 0;
    readers.clear();
    initSuccess = false;

    es.clear();
    rs.clear();
    bs.clear();
    luts.clear();

    dstSrcMaps.clear(); 
    dstMasks.clear();
    dstUniqueMasks.clear();
    currMasks.clear();
    customMasks.clear();
    images.clear();
    correctImages.clear();
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
    int dstWidth, int dstHeight, int activateMbBlend, int mbBlendNumLevels, int lBlendRadius)
{
    return ptrImpl->init(srcVideoFiles, cameraParamFile, dstWidth, dstHeight, 
        activateMbBlend, mbBlendNumLevels, lBlendRadius);
}

bool CPUPanoramaPreviewTask::reset(const std::string& cameraParamFile)
{
    return ptrImpl->reset(cameraParamFile);
}

bool CPUPanoramaPreviewTask::setBlendType(bool multibandBlend)
{
    return ptrImpl->setBlendType(multibandBlend);
}

bool CPUPanoramaPreviewTask::setMultibandBlendParam(int numLevels)
{
    return ptrImpl->setMultibandBlendParam(numLevels);
}

bool CPUPanoramaPreviewTask::setLinearBlendParam(int radius)
{
    return ptrImpl->setLinearBlendParam(radius);
}

bool CPUPanoramaPreviewTask::getBlendType(bool& multibandBlend) const
{
    return ptrImpl->getBlendType(multibandBlend);
}

bool CPUPanoramaPreviewTask::getMultibandBlendParam(int& numLevels) const
{
    return ptrImpl->getMultibandBlendParam(numLevels);
}

bool CPUPanoramaPreviewTask::getLinearBlendParam(int& radius) const
{
    return ptrImpl->getLinearBlendParam(radius);
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

bool CPUPanoramaPreviewTask::getStichSize(int& width, int& height) const
{
    return ptrImpl->getStichSize(width, height);
}

bool CPUPanoramaPreviewTask::getMasks(std::vector<cv::Mat>& masks) const
{
    return ptrImpl->getMasks(masks);
}

bool CPUPanoramaPreviewTask::getUniqueMasks(std::vector<cv::Mat>& masks) const
{
    return ptrImpl->getUniqueMasks(masks);
}

bool CPUPanoramaPreviewTask::seek(const std::vector<int>& indexes)
{
    return ptrImpl->seek(indexes);
}

bool CPUPanoramaPreviewTask::stitch(std::vector<cv::Mat>& src, std::vector<int>& indexes, cv::Mat& dst, int frameIncrement)
{
    return ptrImpl->stitch(src, indexes, dst, frameIncrement);
}

bool CPUPanoramaPreviewTask::restitch(std::vector<cv::Mat>& src, std::vector<int>& indexes, cv::Mat& dst)
{
    return ptrImpl->restitch(src, indexes, dst);
}

bool CPUPanoramaPreviewTask::getCurrStitch(std::vector<cv::Mat>& src, std::vector<int>& indexes, cv::Mat& dst) const
{
    return ptrImpl->getCurrStitch(src, indexes, dst);
}

bool CPUPanoramaPreviewTask::getCurrReprojectForAll(std::vector<cv::Mat>& images, std::vector<int>& indexes) const
{
    return ptrImpl->getCurrReprojectForAll(images, indexes);
}

bool CPUPanoramaPreviewTask::reReprojectForAll(std::vector<cv::Mat>& images, std::vector<int>& indexes)
{
    return ptrImpl->reReprojectForAll(images, indexes);
}

bool CPUPanoramaPreviewTask::readNextAndReprojectForAll(std::vector<cv::Mat>& images, std::vector<int>& indexes)
{
    return ptrImpl->readNextAndReprojectForAll(images, indexes);
}

bool CPUPanoramaPreviewTask::readNextAndReprojectForOne(int videoIndex, cv::Mat& image, int& frameIndex)
{
    return ptrImpl->readNextAndReprojectForOne(videoIndex, image, frameIndex);
}

bool CPUPanoramaPreviewTask::readPrevAndReprojectForOne(int videoIndex, cv::Mat& image, int& frameIndex)
{
    return ptrImpl->readPrevAndReprojectForOne(videoIndex, image, frameIndex);
}

bool CPUPanoramaPreviewTask::setCustomMaskForOne(int videoIndex, int begFrameIndexInc, int endFrameIndexExc, const cv::Mat& mask)
{
    return ptrImpl->setCustomMaskForOne(videoIndex, begFrameIndexInc, endFrameIndexExc, mask);
}

void CPUPanoramaPreviewTask::eraseCustomMaskForOne(int videoIndex, int begFrameIndexInc, int endFrameIndexExc)
{
    return ptrImpl->eraseCustomMaskForOne(videoIndex, begFrameIndexInc, endFrameIndexExc);
}

void CPUPanoramaPreviewTask::eraseAllMasksForOne(int index)
{
    ptrImpl->eraseAllMasksForOne(index);
}

bool CPUPanoramaPreviewTask::getCustomMaskIfHasOrUniqueMaskForOne(int videoIndex, int frameIndex, cv::Mat& mask) const
{
    return ptrImpl->getCustomMaskIfHasOrUniqueMaskForOne(videoIndex, frameIndex, mask);
}

bool CPUPanoramaPreviewTask::getCustomMasksIfHaveOrUniqueMasksForAll(const std::vector<int>& indexes, std::vector<cv::Mat>& masks) const
{
    return ptrImpl->getCustomMasksIfHaveOrUniqueMasksForAll(indexes, masks);
}

bool CPUPanoramaPreviewTask::getAllCustomMasksForOne(int videoIndex, std::vector<int>& begFrameIndexesInc, 
    std::vector<int>& endFrameIndexesInc, std::vector<cv::Mat>& masks) const
{
    return ptrImpl->getAllCustomMasksForOne(videoIndex, begFrameIndexesInc, endFrameIndexesInc, masks);
}

bool CPUPanoramaPreviewTask::correctExposureWhiteBalance(bool whiteBalance, std::vector<double>& exposures,
    std::vector<double>& redRatios, std::vector<double>& blueRatios)
{
    return ptrImpl->correctExposureWhiteBalance(whiteBalance, exposures, redRatios, blueRatios);
}

bool CPUPanoramaPreviewTask::getExposureWhiteBalance(std::vector<double>& exposures,
    std::vector<double>& redRatios, std::vector<double>& blueRatios)
{
    return ptrImpl->getExposureWhiteBalance(exposures, redRatios, blueRatios);
}

bool CPUPanoramaPreviewTask::setExposureWhiteBalance(const std::vector<double>& exposures,
    const std::vector<double>& redRatios, const std::vector<double>& blueRatios)
{
    return ptrImpl->setExposureWhiteBalance(exposures, redRatios, blueRatios);
}

bool CPUPanoramaPreviewTask::seek(const std::vector<long long int>& timeStamps)
{
    int num = timeStamps.size();
    double scale = ptrImpl->frameIntervalInMicroSec;
    if (scale == 0)
        scale = 1;
    std::vector<int> indexes(num);
    for (int i = 0; i < num; i++)
        indexes[i] = timeStamps[i] / scale + 0.5;
    return ptrImpl->seek(indexes);
}

#define RUN_AND_CONVERT_VECTOR(x) \
std::vector<int> indexes; \
    if (!(x)) \
    return false; \
int num = indexes.size(); \
double scale = ptrImpl->frameIntervalInMicroSec; \
timeStamps.resize(num); \
for (int i = 0; i < num; i++) \
    timeStamps[i] = indexes[i] * scale; \
return true

bool CPUPanoramaPreviewTask::stitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst, int frameIncrement)
{
    RUN_AND_CONVERT_VECTOR(ptrImpl->stitch(src, indexes, dst, frameIncrement));
}

bool CPUPanoramaPreviewTask::restitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst)
{
    RUN_AND_CONVERT_VECTOR(ptrImpl->restitch(src, indexes, dst));
}

bool CPUPanoramaPreviewTask::getCurrReprojectForAll(std::vector<cv::Mat>& images, std::vector<long long int>& timeStamps) const
{
    RUN_AND_CONVERT_VECTOR(ptrImpl->getCurrReprojectForAll(images, indexes));
}

bool CPUPanoramaPreviewTask::reReprojectForAll(std::vector<cv::Mat>& images, std::vector<long long int>& timeStamps)
{
    RUN_AND_CONVERT_VECTOR(ptrImpl->reReprojectForAll(images, indexes));
}

bool CPUPanoramaPreviewTask::readNextAndReprojectForAll(std::vector<cv::Mat>& images, std::vector<long long int>& timeStamps)
{
    RUN_AND_CONVERT_VECTOR(ptrImpl->readNextAndReprojectForAll(images, indexes));
}

bool CPUPanoramaPreviewTask::readNextAndReprojectForOne(int index, cv::Mat& image, long long int& timeStamp)
{
    int frameIndex;
    if (!ptrImpl->readNextAndReprojectForOne(index, image, frameIndex))
        return false;
    timeStamp = frameIndex * ptrImpl->frameIntervalInMicroSec;
    return true;
}

bool CPUPanoramaPreviewTask::readPrevAndReprojectForOne(int index, cv::Mat& image, long long int& timeStamp)
{
    int frameIndex;
    if (!ptrImpl->readPrevAndReprojectForOne(index, image, frameIndex))
        return false;
    timeStamp = frameIndex * ptrImpl->frameIntervalInMicroSec;
    return true;
}

bool CPUPanoramaPreviewTask::setCustomMaskForOne(int index, long long int begInc, long long int endExc, const cv::Mat& mask)
{
    double scale = ptrImpl->frameIntervalInMicroSec;
    if (scale == 0)
        scale = 1;
    return ptrImpl->setCustomMaskForOne(index, begInc / scale + 0.5, endExc / scale + 0.5, mask);
}

void CPUPanoramaPreviewTask::eraseCustomMaskForOne(int index, long long int begInc, long long int endExc, long long int precision)
{
    double scale = ptrImpl->frameIntervalInMicroSec;
    if (scale == 0)
        scale = 1;
    ptrImpl->eraseCustomMaskForOne(index, begInc / scale + 0.5, endExc / scale + 0.5);
}

bool CPUPanoramaPreviewTask::getCustomMaskIfHasOrUniqueMaskForOne(int index, long long int timeStamp, cv::Mat& mask) const
{
    double scale = ptrImpl->frameIntervalInMicroSec;
    if (scale == 0)
        scale = 1;
    return ptrImpl->getCustomMaskIfHasOrUniqueMaskForOne(index, timeStamp / scale + 0.5, mask);
}

bool CPUPanoramaPreviewTask::getCustomMasksIfHaveOrUniqueMasksForAll(const std::vector<long long int>& timeStamps, std::vector<cv::Mat>& masks) const
{
    double scale = ptrImpl->frameIntervalInMicroSec;
    if (scale == 0)
        scale = 1;
    int num = timeStamps.size();
    std::vector<int> indexes(num);
    for (int i = 0; i < num; i++)
        indexes[i] = timeStamps[i] / scale + 0.5;
    return ptrImpl->getCustomMasksIfHaveOrUniqueMasksForAll(indexes, masks);
}

bool CPUPanoramaPreviewTask::getAllCustomMasksForOne(int index, std::vector<long long int>& begIncs, std::vector<long long int>& endExcs,
    std::vector<cv::Mat>& masks) const
{
    double scale = ptrImpl->frameIntervalInMicroSec;
    if (scale == 0)
        scale = 1;
    std::vector<int> begIndexes, endIndexes;
    if (!ptrImpl->getAllCustomMasksForOne(index, begIndexes, endIndexes, masks))
        return false;
    int num = begIndexes.size();
    begIncs.resize(num);
    endExcs.resize(num);
    for (int i = 0; i < num; i++)
    {
        begIncs[i] = begIndexes[i] * scale;
        endExcs[i] = endIndexes[i] * scale + 500;
    }
    return true;
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
        ztool::lprintf("Error in %s, size of srcVideoFiles empty\n", __FUNCTION__);
        return false;
    }
    numVideos = srcVideoFiles.size();

    ztool::lprintf("Info in %s, open videos and set to the correct frames\n", __FUNCTION__);
    bool ok = false;
    int validFrameCount;
    int audioIndex;
    ok = prepareSrcVideos(srcVideoFiles, avp::PixelTypeBGR32, std::vector<int>(), -1, readers, audioIndex, srcSize, validFrameCount);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not open video file(s)\n", __FUNCTION__);
        return false;
    }
    ztool::lprintf("Info in %s, open videos done\n", __FUNCTION__);

    ztool::lprintf("Info in %s, prepare for reproject and blend\n", __FUNCTION__);
    dstSize.width = dstWidth;
    dstSize.height = dstHeight;
    ptrRender.reset(new CudaMultiCameraPanoramaRender);
    ok = ptrRender->prepare(cameraParamFile, PanoramaRender::BlendTypeMultiband, srcSize, dstSize);
    if (!ok)
    {
        ztool::lprintf("Error in %s, prepare failed\n", __FUNCTION__);
        return false;
    }
    ztool::lprintf("Info in %s, prepare finish\n", __FUNCTION__);

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
        ztool::lprintf("Error in %s, prepare failed\n", __FUNCTION__);
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
    int dstWidth, int dstHeight, int activateMbBlend, int mbBlendNumLevels, int lBlendRadius)
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