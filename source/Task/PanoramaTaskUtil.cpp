#include "PanoramaTaskUtil.h"
#include "CudaPanoramaTaskUtil.h"
#include "Image.h"
#include "Text.h"
#include "Warp/ZReproject.h"
#include "Tool/Print.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <stdarg.h>

void setLanguage(bool isChinese)
{
    setTextLanguage(isChinese);
    setWatermarkLanguage(isChinese);
}

PanoTaskLogCallbackFunc setPanoTaskLogCallback(PanoTaskLogCallbackFunc func)
{
    return ztool::setPrintfCallback(func);
}

const char* getPanoStitchTypeString(int type)
{
    switch (type)
    {
    case PanoStitchTypeMISO:
        return "PanoStitchTypeMISO";
    case PanoStitchTypeRicoh:
        return "PanoStitchTypeRicoh";
    default:
        return "UnkownPanoStitchType";
    }
}

const char* getPanoProjectTypeString(int type)
{
    switch (type)
    {
    case PanoTypeEquiRect:
        return "PanoTypeEquiRect";
    case PanoTypeCube6x1:
        return "PanoTypeCube6x1";
    case PanoTypeCube3x2:
        return "PanoTypeCube3x2";
    case PanoTypeCube180:
        return "PanoTypeCube180";
    default:
        return "UnkownPanoType";
    }
}

bool prepareSrcVideos(const std::vector<std::string>& srcVideoFiles, avp::PixelType pixelType, const std::vector<int>& offsets,
    int tryAudioIndex, std::vector<avp::AudioVideoReader3>& readers, int& audioIndex, cv::Size& srcSize, int& validFrameCount)
{
    readers.clear();
    srcSize = cv::Size();
    validFrameCount = 0;

    if (srcVideoFiles.empty())
    {
        ztool::lprintf("Error in %s, srcVideoFiles empty\n", __FUNCTION__);
        return false;
    }

    int numVideos = srcVideoFiles.size();
    bool hasOffsets = !offsets.empty();
    if (hasOffsets && offsets.size() != numVideos)
    {
        ztool::lprintf("Error in %s, offsets size = %d, num videos = %d, not match\n", 
            __FUNCTION__, offsets.size(), numVideos);
        return false;
    }

    readers.resize(numVideos);

    if (tryAudioIndex < 0 || tryAudioIndex >= numVideos)
    {
        ztool::lprintf("Info in %s, no audio will be opened\n", __FUNCTION__);
        audioIndex = -1;
    }

    bool ok = false;
    double fps = -1;
    for (int i = 0; i < numVideos; i++)
    {
        if (i == tryAudioIndex)
        {
            ok = readers[i].open(srcVideoFiles[i], true, avp::SampleTypeUnknown, true, pixelType);
            if (ok)
                audioIndex = tryAudioIndex;
            else
            {
                ztool::lprintf("Warning in %s, failed to open file %s with audio, index %s, open video only instead\n",
                    __FUNCTION__, srcVideoFiles[i].c_str(), i);
                ok = readers[i].open(srcVideoFiles[i], false, avp::SampleTypeUnknown, true, pixelType);
                audioIndex = -1;
            }
        }
        else
            ok = readers[i].open(srcVideoFiles[i], false, avp::SampleTypeUnknown, true, pixelType);
        if (!ok)
        {
            ztool::lprintf("Error in %s, could not open file %s, index %d\n", __FUNCTION__, srcVideoFiles[i].c_str(), i);
            break;
        }
        ztool::lprintf("Info in %s, file %s opened, index %d, video width = %d, video height = %d, video frame rate = %f",
            __FUNCTION__, srcVideoFiles[i].c_str(), i, readers[i].getVideoWidth(), readers[i].getVideoHeight(),
            readers[i].getVideoFrameRate());
        if (tryAudioIndex == i)
            ztool::lprintf(", audio sample rate = %d\n", readers[i].getAudioSampleRate());
        else
            ztool::lprintf("\n");

        if (srcSize == cv::Size())
        {
            srcSize.width = readers[i].getVideoWidth();
            srcSize.height = readers[i].getVideoHeight();
        }
        if (srcSize.width != readers[i].getVideoWidth() ||
            srcSize.height != readers[i].getVideoHeight())
        {
            ok = false;
            ztool::lprintf("Error in %s, video size unmatch\n", __FUNCTION__);
            break;
        }

        if (fps < 0)
            fps = readers[i].getVideoFrameRate();
        if (abs(fps - readers[i].getVideoFrameRate()) > 0.1)
        {
            ztool::lprintf("Error in %s, video fps not consistent\n", __FUNCTION__);
            ok = false;
            break;
        }

        int count = hasOffsets ? offsets[i] : 0;
        int currValidFrameCount = readers[i].getVideoNumFrames();
        if (currValidFrameCount <= 0)
            validFrameCount = -1;
        else
        {
            currValidFrameCount -= count;
            if (currValidFrameCount <= 0)
            {
                ztool::lprintf("Error in %s, video at index %d has only %d frames, "
                    "should be saught to frame indexed %d, video not long enough\n", 
                    __FUNCTION__, i, readers[i].getVideoNumFrames(), offsets[i]);
                ok = false;
                break;
            }
        }

        if (validFrameCount == 0)
            validFrameCount = currValidFrameCount;
        if (validFrameCount > 0)
            validFrameCount = validFrameCount > currValidFrameCount ? currValidFrameCount : validFrameCount;

        if (hasOffsets)
        {
            if (!readers[i].seekByIndex(count, avp::VIDEO))
            {
                ztool::lprintf("Error in %s, video at index %d cannot be saught to frame indexed %d\n", 
                    __FUNCTION__, i, count);
                ok = false;
                break;
            }
            if (!ok)
                break;
        }
    }

    if (!ok)
    {
        readers.clear();
        audioIndex = -1;
        srcSize = cv::Size();
        validFrameCount = 0;
    }

    return ok;
}

static void alphaBlend(cv::Mat& image, const cv::Mat& logo)
{
    CV_Assert(image.data && (image.type() == CV_8UC3 || image.type() == CV_8UC4) &&
        logo.data && logo.type() == CV_8UC4 && image.size() == logo.size());

    int rows = image.rows, cols = image.cols, channels = image.channels();
    for (int i = 0; i < rows; i++)
    {
        unsigned char* ptrImage = image.ptr<unsigned char>(i);
        const unsigned char* ptrLogo = logo.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrLogo[3])
            {
                int val = ptrLogo[3];
                int comp = 255 - ptrLogo[3];
                ptrImage[0] = (comp * ptrImage[0] + val * ptrLogo[0] + 254) / 255;
                ptrImage[1] = (comp * ptrImage[1] + val * ptrLogo[1] + 254) / 255;
                ptrImage[2] = (comp * ptrImage[2] + val * ptrLogo[2] + 254) / 255;
            }
            ptrImage += channels;
            ptrLogo += 4;
        }
    }
}

static const int blockWidth = 512;
static const int blockHeight = 256;

bool WatermarkFilter::init(int width_, int height_, int type_)
{
    initSuccess = false;
    clear();

    if (width_ < 0 || height_ < 0 || (type_ != CV_8UC3 && type_ != CV_8UC4))
    {
        ztool::lprintf("Error in %s, width(%d) height(%d) type(%d) not satisfied\n",
            __FUNCTION__, width_, height_, type_);
        return false;
    }

    width = width_;
    height = height_;
    type = type_;

    cv::Mat origLogo(watermarkHeight, watermarkWidth, CV_8UC4, watermarkData);

    rects.clear();
    if (width < watermarkWidth || height < watermarkHeight)
    {
        cv::Rect logoRect(watermarkWidth / 2 - width / 2, watermarkHeight / 2 - height / 2, width, height);
        logo = origLogo(logoRect);
        rects.push_back(cv::Rect(0, 0, width, height));
    }
    else
    {
        logo = origLogo;
        int w = (width + blockWidth - 1) / blockWidth, h = (height + blockHeight - 1) / blockHeight;
        cv::Rect full(0, 0, width, height);
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                cv::Rect thisRect = cv::Rect(j * blockWidth + blockWidth / 2 - watermarkWidth / 2,
                                             i * blockHeight + blockHeight / 2 - watermarkHeight / 2,
                                             watermarkWidth, watermarkHeight) &
                                    full;
                if (thisRect.area())
                    rects.push_back(thisRect);
            }
        }
    }

    initSuccess = true;
    return true;
}

bool WatermarkFilter::addWatermark(cv::Mat& image) const
{
    if (!initSuccess || !image.data || image.rows != height || image.cols != width || image.type() != type)
    {
        ztool::lprintf("Error in %s, initSuccess(%d), image.data(%p), image.rows(%d), image.cols(%d), image.type()(%d) unsatisfied, "
            "require initSuccess = 1, image.data not NULL, image.rows = %d, image.cols = %d, image.type() = %d\n",
            __FUNCTION__, initSuccess, image.data, image.rows, image.cols, image.type(), height, width, type);
        return false;
        return false;
    }

    int size = rects.size();
    for (int i = 0; i < size; i++)
    {
        cv::Mat imagePart(image, rects[i]);
        cv::Mat logoPart(logo, cv::Rect(0, 0, rects[i].width, rects[i].height));
        alphaBlend(imagePart, logoPart);
    }

    return true;
}

void WatermarkFilter::clear()
{
    initSuccess = false;
    width = 0;
    height = 0;
    rects.clear();
    logo.release();
}

bool LogoFilter::init(const std::string& logoFileName, int hFov, int width_, int height_)
{
    initSuccess = false;
    clear();

    if (width_ <= 0 || height_ <= 0 || width_ != height_ * 2)
    {
        ztool::lprintf("Error in %s, width(%d) or height(%d) not satisfied\n", __FUNCTION__, width_, height_);
        return false;
    }

    if (hFov <= 0 || hFov > 180)
    {
        ztool::lprintf("Error in %s, hFov(%d) not satisfied, shoul be in (0, 180]", __FUNCTION__, hFov);
        return false;
    }

    cv::Mat origLogo = cv::imread(logoFileName, -1);
    if (!origLogo.data)
    {
        ztool::lprintf("Error in %s, could not open file %s\n", __FUNCTION__, logoFileName.c_str());
        return false;
    }
    if (origLogo.type() != CV_8UC3 && origLogo.type() != CV_8UC4)
    {
        ztool::lprintf("Error in %s, logo image should be of type CV_8UC3(%d) or CV_8UC4(%d), "
            "but now type %d, depth %d, channels %d\n", __FUNCTION__, CV_8UC3, CV_8UC4, 
            origLogo.type(), origLogo.depth(), origLogo.channels());
        return false;
    }

    PhotoParam param;
    param.imageType = PhotoParam::ImageTypeFullFrameFishEye;
    param.hfov = hFov;
    param.pitch = -90;
    param.cropWidth = origLogo.cols;
    param.cropHeight = origLogo.rows;
    cv::Mat map, mask;
    getReprojectMapAndMask(param, origLogo.size(), cv::Size(width_, height_), map, mask);
    cv::Mat logoReproj;
    reprojectParallel(origLogo, logoReproj, map);

    if (origLogo.type() == CV_8UC3)
    {
        logo.create(cv::Size(width_, height_), CV_8UC4);
        int fromTo[] = { 0, 0, 1, 1, 2, 2, 3, 3 };
        cv::Mat src[] = { logoReproj, mask };
        cv::mixChannels(src, 2, &logo, 1, fromTo, 4);
    }
    else
        logo = logoReproj;

    width = width_;
    height = height_;
    initSuccess = true;
    return true;
}

bool LogoFilter::init(const cv::Mat& origLogo, int hFov, int width_, int height_)
{
    clear();

    if (width_ <= 0 || height_ <= 0 || width_ != height_ * 2)
    {
        ztool::lprintf("Error in %s, width(%d) or height(%d) not satisfied\n", __FUNCTION__, width_, height_);
        return false;
    }

    if (hFov <= 0 || hFov > 180)
    {
        ztool::lprintf("Error in %s, hFov(%d) not satisfied, shoul be in (0, 180]", __FUNCTION__, hFov);
        return false;
    }

    if (!origLogo.data)
    {
        ztool::lprintf("Error in %s, orig logo image empty\n", __FUNCTION__);
        return false;
    }
    if (origLogo.type() != CV_8UC3 && origLogo.type() != CV_8UC4)
    {
        ztool::lprintf("Error in %s, logo image should be of type CV_8UC3(%d) or CV_8UC4(%d), "
            "but now type %d, depth %d, channels %d\n", __FUNCTION__, CV_8UC3, CV_8UC4,
            origLogo.type(), origLogo.depth(), origLogo.channels());
        return false;
    }

    PhotoParam param;
    param.imageType = PhotoParam::ImageTypeFullFrameFishEye;
    param.hfov = hFov;
    param.pitch = -90;
    param.cropWidth = origLogo.cols;
    param.cropHeight = origLogo.rows;
    cv::Mat map, mask;
    getReprojectMapAndMask(param, origLogo.size(), cv::Size(width_, height_), map, mask);
    cv::Mat logoReproj;
    reprojectParallel(origLogo, logoReproj, map);

    if (origLogo.type() == CV_8UC3)
    {
        logo.create(cv::Size(width_, height_), CV_8UC4);
        int fromTo[] = { 0, 0, 1, 1, 2, 2, 3, 3 };
        cv::Mat src[] = { logoReproj, mask };
        cv::mixChannels(src, 2, &logo, 1, fromTo, 4);
    }
    else
        logo = logoReproj;

    width = width_;
    height = height_;
    initSuccess = true;
    return true;
}

bool LogoFilter::addLogo(cv::Mat& image) const
{
    if (!initSuccess || !image.data || image.rows != height || image.cols != width ||
        (image.type() != CV_8UC3 && image.type() != CV_8UC4))
    {
        ztool::lprintf("Error in %s, initSuccess(%d), image.data(%p), image.rows(%d), image.cols(%d), image.type()(%d) unsatisfied, "
            "require initSuccess = 1, image.data not NULL, image.rows = %d, image.cols = %d, image.type() = %d or %d\n",
            __FUNCTION__, initSuccess, image.data, image.rows, image.cols, image.type(), height, width, CV_8UC3, CV_8UC4);
        return false;
    }

    alphaBlend(image, logo);
    return true;
}

void LogoFilter::clear()
{
    initSuccess = false;
    width = 0;
    height = 0;
    logo.release();
}

bool CudaWatermarkFilter::init(int width_, int height_)
{
    initSuccess = false;
    clear();

    if (width_ < 0 || height_ < 0 || width_ != height_ * 2)
    {
        ztool::lprintf("Error in %s, width(%d) or height(%d) not satisfied\n", __FUNCTION__, width_, height_);
        return false;
    }

    width = width_;
    height = height_;

    cv::Mat origLogo(watermarkHeight, watermarkWidth, CV_8UC4, watermarkData);
    cv::Mat fullLogo;

    if (width < watermarkWidth || height < watermarkHeight)
    {
        cv::Rect logoRect(watermarkWidth / 2 - width / 2, watermarkHeight / 2 - height / 2, width, height);
        fullLogo = origLogo(logoRect);
    }
    else
    {
        fullLogo = cv::Mat::zeros(height, width, CV_8UC4);
        int w = (width + blockWidth - 1) / blockWidth, h = (height + blockHeight - 1) / blockHeight;
        cv::Rect full(0, 0, width, height);
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                cv::Rect thisRect = cv::Rect(j * blockWidth + blockWidth / 2 - watermarkWidth / 2,
                    i * blockHeight + blockHeight / 2 - watermarkHeight / 2,
                    watermarkWidth, watermarkHeight) &
                    full;
                cv::Mat fullLogoPart = fullLogo(thisRect);
                cv::Mat origLogoPart = origLogo(cv::Rect(0, 0, thisRect.width, thisRect.height));
                origLogoPart.copyTo(fullLogoPart);
            }
        }
    }
    logo.upload(fullLogo);

    initSuccess = true;
    return true;
}

bool CudaWatermarkFilter::addWatermark(cv::cuda::GpuMat& image) const
{
    if (!initSuccess || !image.data || image.rows != height || image.cols != width || image.type() != CV_8UC4)
    {
        ztool::lprintf("Error in %s, initSuccess(%d), image.data(%p), image.rows(%d), image.cols(%d), image.type()(%d) not satisfied, "
            "requre initSuccess = 1, image.data not NULL, image.rows = %d, image.cols = %d, image.type() = %d\n",
            __FUNCTION__, initSuccess, image.data, image.rows, image.cols, image.type(), height, width, CV_8UC4);
        return false;
    }

    alphaBlend8UC4(image, logo);
    return true;
}

void CudaWatermarkFilter::clear()
{
    initSuccess = false;
    width = 0;
    height = 0;
    logo.release();
}

bool CudaLogoFilter::init(const std::string& logoFileName, int hFov, int width_, int height_)
{
    initSuccess = false;

    if (width_ <= 0 || height_ <= 0 || width_ != height_ * 2)
    {
        ztool::lprintf("Error in %s, width(%d) or height(%d) not satisfied\n", __FUNCTION__, width_, height_);
        return false;
    }

    if (hFov <= 0 || hFov > 180)
    {
        ztool::lprintf("Error in %s, hFov(%d) not satisfied, shoul be in (0, 180]", __FUNCTION__, hFov);
        return false;
    }

    cv::Mat origLogo = cv::imread(logoFileName, -1);
    if (!origLogo.data)
    {
        ztool::lprintf("Error in %s, could not open file %s\n", __FUNCTION__, logoFileName.c_str());
        return false;
    }
    if (origLogo.type() != CV_8UC3 && origLogo.type() != CV_8UC4)
    {
        ztool::lprintf("Error in %s, logo image should be of type CV_8UC3(%d) or CV_8UC4(%d), "
            "but now type %d, depth %d, channels %d\n", __FUNCTION__, CV_8UC3, CV_8UC4,
            origLogo.type(), origLogo.depth(), origLogo.channels());
        return false;
    }

    PhotoParam param;
    param.imageType = PhotoParam::ImageTypeFullFrameFishEye;
    param.hfov = hFov;
    param.pitch = -90;
    param.cropWidth = origLogo.cols;
    param.cropHeight = origLogo.rows;
    cv::Mat map, mask;
    getReprojectMapAndMask(param, origLogo.size(), cv::Size(width_, height_), map, mask);
    cv::Mat logoReproj;
    reprojectParallel(origLogo, logoReproj, map);

    cv::Mat logoCpu;
    if (origLogo.type() == CV_8UC3)
    {
        logoCpu.create(cv::Size(width_, height_), CV_8UC4);
        int fromTo[] = { 0, 0, 1, 1, 2, 2, 3, 3 };
        cv::Mat src[] = { logoReproj, mask };
        cv::mixChannels(src, 2, &logoCpu, 1, fromTo, 4);
    }
    else
        logoCpu = logoReproj;

    logo.upload(logoCpu);

    width = width_;
    height = height_;
    initSuccess = true;
    return true;
}

bool CudaLogoFilter::addLogo(cv::cuda::GpuMat& image) const
{
    if (!initSuccess || !image.data || image.rows != height || image.cols != width || image.type() != CV_8UC4)
    {
        ztool::lprintf("Error in %s, initSuccess(%d), image.data(%p), image.rows(%d), image.cols(%d), image.type()(%d) unsatisfied, "
            "require initSuccess = 1, image.data not NULL, image.rows = %d, image.cols = %d, image.type() = %d\n",
            __FUNCTION__, initSuccess, image.data, image.rows, image.cols, image.type(), height, width, CV_8UC4);
        return false;
    }

    alphaBlend8UC4(image, logo);
    return true;
}

void CudaLogoFilter::clear()
{
    initSuccess = false;
    width = 0;
    height = 0;
    logo.release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CompileControl.h"
#if COMPILE_DISCRETE_OPENCL

#include "DiscreteOpenCL/DiscreteOpenCLInterface.h"
#include "DOclPanoramaTaskUtil.h"

bool DOclWatermarkFilter::init(int width_, int height_)
{
    initSuccess = false;
    clear();

    if (width_ < 0 || height_ < 0 || width_ != height_ * 2)
    {
        ztool::lprintf("Error in %s, width(%d) or height(%d) not satisfied\n", __FUNCTION__, width_, height_);
        return false;
    }

    width = width_;
    height = height_;

    cv::Mat origLogo(watermarkHeight, watermarkWidth, CV_8UC4, watermarkData);
    cv::Mat fullLogo;

    if (width < watermarkWidth || height < watermarkHeight)
    {
        cv::Rect logoRect(watermarkWidth / 2 - width / 2, watermarkHeight / 2 - height / 2, width, height);
        fullLogo = origLogo(logoRect);
    }
    else
    {
        fullLogo = cv::Mat::zeros(height, width, CV_8UC4);
        int w = (width + blockWidth - 1) / blockWidth, h = (height + blockHeight - 1) / blockHeight;
        cv::Rect full(0, 0, width, height);
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                cv::Rect thisRect = cv::Rect(j * blockWidth + blockWidth / 2 - watermarkWidth / 2,
                    i * blockHeight + blockHeight / 2 - watermarkHeight / 2,
                    watermarkWidth, watermarkHeight) &
                    full;
                cv::Mat fullLogoPart = fullLogo(thisRect);
                cv::Mat origLogoPart = origLogo(cv::Rect(0, 0, thisRect.width, thisRect.height));
                origLogoPart.copyTo(fullLogoPart);
            }
        }
    }
    logo.upload(fullLogo);

    initSuccess = true;
    return true;
}

bool DOclWatermarkFilter::addWatermark(docl::GpuMat& image) const
{
    if (!initSuccess || !image.data || image.rows != height || image.cols != width || image.type != CV_8UC4)
    {
        ztool::lprintf("Error in %s, initSuccess(%d), image.data(%p), image.rows(%d), image.cols(%d), image.type()(%d) not satisfied, "
            "requre initSuccess = 1, image.data not NULL, image.rows = %d, image.cols = %d, image.type() = %d\n",
            __FUNCTION__, initSuccess, image.data, image.rows, image.cols, image.type, height, width, CV_8UC4);
        return false;
    }

    alphaBlend8UC4(image, logo);
    return true;
}

void DOclWatermarkFilter::clear()
{
    initSuccess = false;
    width = 0;
    height = 0;
    logo.release();
}

bool DOclLogoFilter::init(const std::string& logoFileName, int hFov, int width_, int height_)
{
    initSuccess = false;

    if (width_ <= 0 || height_ <= 0 || width_ != height_ * 2)
    {
        ztool::lprintf("Error in %s, width(%d) or height(%d) not satisfied\n", __FUNCTION__, width_, height_);
        return false;
    }

    if (hFov <= 0 || hFov > 180)
    {
        ztool::lprintf("Error in %s, hFov(%d) not satisfied, shoul be in (0, 180]", __FUNCTION__, hFov);
        return false;
    }

    cv::Mat origLogo = cv::imread(logoFileName, -1);
    if (!origLogo.data)
    {
        ztool::lprintf("Error in %s, could not open file %s\n", __FUNCTION__, logoFileName.c_str());
        return false;
    }
    if (origLogo.type() != CV_8UC3 && origLogo.type() != CV_8UC4)
    {
        ztool::lprintf("Error in %s, logo image should be of type CV_8UC3(%d) or CV_8UC4(%d), "
            "but now type %d, depth %d, channels %d\n", __FUNCTION__, CV_8UC3, CV_8UC4,
            origLogo.type(), origLogo.depth(), origLogo.channels());
        return false;
    }

    PhotoParam param;
    param.imageType = PhotoParam::ImageTypeFullFrameFishEye;
    param.hfov = hFov;
    param.pitch = -90;
    param.cropWidth = origLogo.cols;
    param.cropHeight = origLogo.rows;
    cv::Mat map, mask;
    getReprojectMapAndMask(param, origLogo.size(), cv::Size(width_, height_), map, mask);
    cv::Mat logoReproj;
    reprojectParallel(origLogo, logoReproj, map);

    cv::Mat logoCpu;
    if (origLogo.type() == CV_8UC3)
    {
        logoCpu.create(cv::Size(width_, height_), CV_8UC4);
        int fromTo[] = { 0, 0, 1, 1, 2, 2, 3, 3 };
        cv::Mat src[] = { logoReproj, mask };
        cv::mixChannels(src, 2, &logoCpu, 1, fromTo, 4);
    }
    else
        logoCpu = logoReproj;

    logo.upload(logoCpu);

    width = width_;
    height = height_;
    initSuccess = true;
    return true;
}

bool DOclLogoFilter::addLogo(docl::GpuMat& image) const
{
    if (!initSuccess || !image.data || image.rows != height || image.cols != width || image.type != CV_8UC4)
    {
        ztool::lprintf("Error in %s, initSuccess(%d), image.data(%p), image.rows(%d), image.cols(%d), image.type()(%d) unsatisfied, "
            "require initSuccess = 1, image.data not NULL, image.rows = %d, image.cols = %d, image.type() = %d\n",
            __FUNCTION__, initSuccess, image.data, image.rows, image.cols, image.type, height, width, CV_8UC4);
        return false;
    }

    alphaBlend8UC4(image, logo);
    return true;
}

void DOclLogoFilter::clear()
{
    initSuccess = false;
    width = 0;
    height = 0;
    logo.release();
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
bool setIntervaledContoursToPreviewTask(const std::vector<std::vector<IntervaledContour> >& contours,
    CPUPanoramaPreviewTask& task)
{
    std::vector<cv::Mat> boundedMasks;
    if (!task.getMasks(boundedMasks))
        return false;

    if (contours.size() != boundedMasks.size())
        return false;

    int size = boundedMasks.size();
    bool success = true;
    IntervaledMask currItvMask;
    for (int i = 0; i < size; i++)
    {
        int num = contours[i].size();
        for (int j = 0; j < num; j++)
        {
            if (!cvtContourToMask(contours[i][j], boundedMasks[i], currItvMask))
            {
                success = false;
                break;
            }
            task.setCustomMaskForOne(i, currItvMask.begInc, currItvMask.endExc, currItvMask.mask);
        }
        if (!success)
            break;
    }

    return success;
}

bool getIntervaledContoursFromPreviewTask(const CPUPanoramaPreviewTask& task,
    std::vector<std::vector<IntervaledContour> >& contours)
{
    contours.clear();
    if (!task.isValid())
        return false;

    int size = task.getNumSourceVideos();
    contours.resize(size);
    std::vector<long long int> begIncs, endExcs;
    std::vector<cv::Mat> masks;
    bool success = true;
    for (int i = 0; i < size; i++)
    {
        if (task.getAllCustomMasksForOne(i, begIncs, endExcs, masks))
        {
            int len = begIncs.size();
            contours[i].resize(len);
            for (int j = 0; j < len; j++)
            {
                if (!cvtMaskToContour(IntervaledMask(begIncs[j], endExcs[j], masks[j]), contours[i][j]))
                {
                    success = false;
                    break;
                }
            }
            if (!success)
                break;
        }
        else
        {
            success = false;
            break;
        }
    }
    if (!success)
        contours.clear();
    return success;
}
*/

bool setIntervaledContoursToPreviewTask(const std::vector<std::vector<IntervaledContour> >& contours,
    CPUPanoramaPreviewTask& task)
{
    std::vector<cv::Mat> boundedMasks;
    if (!task.getMasks(boundedMasks))
        return false;

    int numIntervals = contours.size();
    int numVideos = boundedMasks.size();
    bool success = true;
    IntervaledMask currItvMask;
    for (int i = 0; i < numIntervals; i++)
    {
        int num = contours[i].size();
        for (int j = 0; j < num; j++)
        {
            int videoIndex = contours[i][j].videoIndex;
            if (videoIndex < 0 || videoIndex >= numVideos)
            {
                success = false;
                break;
            }
            if (!cvtContourToMask(contours[i][j], boundedMasks[videoIndex], currItvMask))
            {
                success = false;
                break;
            }
            if (!task.setCustomMaskForOne(videoIndex, currItvMask.begIndexInc, currItvMask.endIndexInc, currItvMask.mask))
            {
                success = false;
                break;
            }
        }
        if (!success)
            break;
    }

    return success;
}

bool getIntervaledContoursFromPreviewTask(const CPUPanoramaPreviewTask& task, const std::vector<int>& offsets,
    std::vector<std::vector<IntervaledContour> >& contours)
{
    contours.clear();
    if (!task.isValid())
        return false;

    int numVideos = task.getNumSourceVideos();
    if (numVideos != offsets.size())
        return false;

    std::vector<std::vector<IntervaledContour> > contoursNumVideosMajor;
    contoursNumVideosMajor.resize(numVideos);
    std::vector<int> begIndexesInc, endIndexesInc;
    std::vector<cv::Mat> masks;
    bool success = true;
    for (int i = 0; i < numVideos; i++)
    {
        if (task.getAllCustomMasksForOne(i, begIndexesInc, endIndexesInc, masks))
        {
            int len = begIndexesInc.size();
            contoursNumVideosMajor[i].resize(len);
            for (int j = 0; j < len; j++)
            {
                if (!cvtMaskToContour(IntervaledMask(i, begIndexesInc[j], endIndexesInc[j], masks[j]), contoursNumVideosMajor[i][j]))
                {
                    success = false;
                    break;
                }
            }
            if (!success)
                break;
        }
        else
        {
            success = false;
            break;
        }
    }
    if (!success)
        contours.clear();

    for (int i = 0; i < numVideos; i++)
    {
        int numIntervals = contoursNumVideosMajor[i].size();
        for (int j = 0; j < numIntervals; j++)
        {
            contoursNumVideosMajor[i][j].begIndexInc -= offsets[i];
            contoursNumVideosMajor[i][j].endIndexInc -= offsets[i];
        }
    }

    typedef std::vector<IntervaledContour>::iterator Iterator;
    for (int i = 0; i < numVideos; i++)
    {
        for (Iterator itrI = contoursNumVideosMajor[i].begin(); itrI != contoursNumVideosMajor[i].end();)
        {
            std::vector<IntervaledContour> currContours;
            for (int j = 0; j < numVideos; j++)
            {
                if (i == j)
                    continue;
                for (Iterator itrJ = contoursNumVideosMajor[j].begin(); itrJ != contoursNumVideosMajor[j].end();)
                {
                    if (itrI->begIndexInc == itrJ->begIndexInc &&
                        itrI->endIndexInc == itrJ->endIndexInc)
                    {
                        if (currContours.empty())
                            currContours.push_back(*itrI);
                        currContours.push_back(*itrJ);
                        itrJ = contoursNumVideosMajor[j].erase(itrJ);
                    }
                    else
                        ++itrJ;
                }
            }
            contours.push_back(currContours);
            itrI = contoursNumVideosMajor[i].erase(itrI);
        }
    }
    return success;
}

#include "ticpp.h"

using ticpp::Element;
using ticpp::Document;

#include <sstream>
bool cvtStringToPoints(const std::string& text, std::vector<cv::Point>& points)
{
    points.clear();
    if (text.empty())
        return false;

    std::istringstream strm(text);
    bool success = true;
    while (!strm.eof())
    {
        int x, y;
        strm >> x;
        // Failing to satisty following if condition cannot set success to false
        // because the final charactor may be space
        if (strm.fail() || strm.bad())
            break;
        strm >> y;
        if (strm.fail() || strm.bad())
        {
            success = false;
            break;
        }
        points.push_back(cv::Point(x, y));
    }
    if (!success)
        points.clear();
    return success;
}

bool loadVideoFileNamesAndOffset(const std::string& fileName, std::vector<std::string>& videoNames, std::vector<int>& offsets)
{
    videoNames.clear();
    offsets.clear();

    Document doc;
    try
    {
        doc.LoadFile(fileName);
    }
    catch (...)
    {
        return false;
    }

    Element* ptrRoot = doc.FirstChildElement("Root", false);
    if (ptrRoot == NULL)
        return false;

    Element* ptrPos = ptrRoot->FirstChildElement("VIDEO", false);
    if (ptrPos == NULL)
        return false;

    bool success = true;
    for (ticpp::Iterator<ticpp::Element> itrVideo(ptrPos, "VIDEO"); itrVideo != itrVideo.end(); itrVideo++)
    {
        Element* ptrVideoName = itrVideo->FirstChildElement("VIDEONAME");
        Element* ptrSyncFrame = itrVideo->FirstChildElement("SYNCFRAME");
        std::string name;
        int offset;
        try
        {
            name = ptrVideoName->GetText();
            ptrSyncFrame->GetText(&offset);
        }
        catch (...)
        {
            success = false;
            break;
        }
        videoNames.push_back(name);
        offsets.push_back(offset);
    }
    if (!success)
    {
        videoNames.clear();
        offsets.clear();
    }
    return success;
}

bool loadExposureWhiteBalance(const std::string& fileName, std::vector<double>& exposures,
    std::vector<double>& redRatios, std::vector<double>& blueRatios)
{
    exposures.clear();
    redRatios.clear();
    blueRatios.clear();

    Document doc;
    try
    {
        doc.LoadFile(fileName);
    }
    catch (...)
    {
        return false;
    }

    Element* ptrRoot = doc.FirstChildElement("Root", false);
    if (ptrRoot == NULL)
        return false;

    Element* ptrPos = ptrRoot->FirstChildElement("VIDEO", false);
    if (ptrPos == NULL)
        return false;

    bool success = true;
    for (ticpp::Iterator<ticpp::Element> itrVideo(ptrPos, "VIDEO"); itrVideo != itrVideo.end(); itrVideo++)
    {
        Element* ptrExposure = itrVideo->FirstChildElement("EXPOSURE", false);
        Element* ptrRed = itrVideo->FirstChildElement("RED", false);
        Element* ptrBlue = itrVideo->FirstChildElement("BLUE", false);
        if (!ptrExposure || !ptrRed || !ptrBlue)
        {
            success = false;
            break;
        }

        double e, r, b;
        try
        {
            ptrExposure->GetText(&e);
            ptrRed->GetText(&r);
            ptrBlue->GetText(&b);
        }
        catch (...)
        {
            success = false;
            break;
        }
        exposures.push_back(e);
        redRatios.push_back(r);
        blueRatios.push_back(b);
    }
    if (!success)
    {
        exposures.clear();
        redRatios.clear();
        blueRatios.clear();
    }
    return success;
}

bool needCorrectExposureWhiteBalance(const std::vector<double>& exposures,
    const std::vector<double>& redRatios, const std::vector<double>& blueRatios)
{
    int size = exposures.size();
    if (!size || size != redRatios.size() || size != blueRatios.size())
        return false;

    for (int i = 0; i < size; i++)
    {
        if (abs(exposures[i] - 1) > 0.05 ||
            abs(redRatios[i] - 1) > 0.05 ||
            abs(blueRatios[i] - 1) > 0.05)
            return true;
    }
    return false;
}

bool loadOutputConfig(const std::string& fileName, int& audioIndex, int& panoStitchType,
    std::string& logoFile, int& logoFOV, int& highQualityBlend, int& blendParam,
    std::string& dstVideoFile, int& dstWidth, int& dstHeight, int& dstVideoBitRate,
    std::string& dstVideoEncoder, std::string& dstVideoPreset, 
    int& startFrameIndex, int& dstVideoMaxFrameCount)
{
    Document doc;
    try
    {
        doc.LoadFile(fileName);
    }
    catch (...)
    {
        return false;
    }

    Element* ptrRoot = doc.FirstChildElement("Root", false);
    if (ptrRoot == NULL)
        return false;

    Element* ptrOutput = ptrRoot->FirstChildElement("Output", false);
    if (ptrOutput == NULL)
        return false;

    Element* ptrElem = NULL;

    try
    {
        ptrElem = ptrOutput->FirstChildElement("AudioIndex", false);
        if (ptrElem == NULL)
            return false;
        ptrElem->GetText(&audioIndex);

        ptrElem = ptrOutput->FirstChildElement("PanoStitchType", false);
        if (ptrElem == NULL)
            return false;
        ptrElem->GetText(&panoStitchType);

        ptrElem = ptrOutput->FirstChildElement("LogoFile", false);
        if (ptrElem == NULL)
            return false;
        ptrElem->GetTextOrDefault(&logoFile, std::string());

        ptrElem = ptrOutput->FirstChildElement("LogoFOV", false);
        if (ptrElem == NULL)
            return false;
        ptrElem->GetText(&logoFOV);

        ptrElem = ptrOutput->FirstChildElement("HighQualityBlend", false);
        if (ptrElem == NULL)
            return false;
        ptrElem->GetText(&highQualityBlend);

        ptrElem = ptrOutput->FirstChildElement("BlendParam", false);
        if (ptrElem == NULL)
            return false;
        ptrElem->GetText(&blendParam);

        ptrElem = ptrOutput->FirstChildElement("VideoFile", false);
        if (ptrElem == NULL)
            return false;
        ptrElem->GetText(&dstVideoFile);

        ptrElem = ptrOutput->FirstChildElement("VideoWidth", false);
        if (ptrElem == NULL)
            return false;
        ptrElem->GetText(&dstWidth);

        ptrElem = ptrOutput->FirstChildElement("VideoHeight", false);
        if (ptrElem == NULL)
            return false;
        ptrElem->GetText(&dstHeight);

        ptrElem = ptrOutput->FirstChildElement("VideoBitRate", false);
        if (ptrElem == NULL)
            return false;
        ptrElem->GetText(&dstVideoBitRate);

        ptrElem = ptrOutput->FirstChildElement("VideoEncoder", false);
        if (ptrElem == NULL)
            return false;
        dstVideoEncoder = ptrElem->GetText();

        ptrElem = ptrOutput->FirstChildElement("VideoPreset", false);
        if (ptrElem == NULL)
            return false;
        dstVideoPreset = ptrElem->GetText();

        ptrElem = ptrOutput->FirstChildElement("StartFrameIndex", false);
        if (ptrElem == NULL)
            return false;
        ptrElem->GetText(&startFrameIndex);

        ptrElem = ptrOutput->FirstChildElement("MaxFrameCount", false);
        if (ptrElem == NULL)
            return false;
        ptrElem->GetText(&dstVideoMaxFrameCount);
    }
    catch (...)
    {
        return false;
    }

    return true;
}

/*
bool loadIntervaledContours(const std::string& fileName, std::vector<std::vector<IntervaledContour> >& contours)
{
    contours.clear();

    Document doc;
    try
    {
        doc.LoadFile(fileName);
    }
    catch (...)
    {
        return false;
    }

    Element* ptrRoot = doc.FirstChildElement("Root", false);
    if (ptrRoot == NULL)
        return false;

    Element* ptrPos = ptrRoot->FirstChildElement("VIDEO", false);
    if (ptrPos == NULL)
        return false;

    bool success = true;
    for (ticpp::Iterator<ticpp::Element> itrVideo(ptrPos, "VIDEO"); itrVideo != itrVideo.end(); itrVideo++)
    {
        Element* ptrContours = NULL;
        std::vector<IntervaledContour> currContours;
        ptrContours = itrVideo->FirstChildElement("Contours", false);
        if (ptrContours == NULL)
        {
            contours.push_back(currContours);
            continue;
        }
        Element* ptrContour = ptrContours->FirstChildElement("Contour", false);
        if (ptrContour == NULL)
        {
            contours.push_back(currContours);
            continue;
        }

        for (ticpp::Iterator<ticpp::Element> itrContour(ptrContour, "Contour"); itrContour != itrContour.end(); ++itrContour)
        {
            IntervaledContour currContour;
            try
            {
                Element* ptrElement = NULL;
                ptrElement = itrContour->FirstChildElement("Width");
                ptrElement->GetText(&currContour.width);
                ptrElement = itrContour->FirstChildElement("Height");
                ptrElement->GetText(&currContour.height);
                ptrElement = itrContour->FirstChildElement("Begin");
                ptrElement->GetText(&currContour.begIndexInc);
                ptrElement = itrContour->FirstChildElement("End");
                ptrElement->GetText(&currContour.endIndexInc);

                Element* ptrPoints = itrContour->FirstChildElement("Points", false);
                if (ptrPoints == NULL)
                    continue;
                for (ticpp::Iterator<ticpp::Element> itrPoints(ptrPoints, "Points"); itrPoints != itrPoints.end(); ++itrPoints)
                {
                    std::string text = itrPoints->GetText(false);
                    if (text.empty())
                        continue;
                    currContour.contours.resize(currContour.contours.size() + 1);
                    if (!cvtStringToPoints(text, currContour.contours.back()))
                    {
                        success = false;
                        break;
                    }
                }
                if (!success)
                    break;
            }
            catch (...)
            {
                success = false;
                break;
            }

            currContours.push_back(currContour);
        }
        if (!success)
            break;

        contours.push_back(currContours);
    }
    if (!success)
        contours.clear();
    return success;
}
*/

bool loadIntervaledContours(const std::string& fileName, std::vector<std::vector<IntervaledContour> >& contours)
{
    contours.clear();

    Document doc;
    try
    {
        doc.LoadFile(fileName);
    }
    catch (...)
    {
        return false;
    }

    Element* ptrRoot = doc.FirstChildElement("Root", false);
    if (ptrRoot == NULL)
        return false;

    Element* ptrContours = ptrRoot->FirstChildElement("Contours", false);
    if (ptrContours == NULL)
        return true;

    Element* ptrContour = ptrContours->FirstChildElement("Contour", false);
    if (ptrContour == NULL)
        return true;

    bool success = true;
    for (ticpp::Iterator<ticpp::Element> itrContour(ptrContour, "Contour"); itrContour != itrContour.end(); itrContour++)
    {
        std::string attrib;
        int width, height, begIndexInc, endIndexInc;
        attrib = itrContour->GetAttributeOrDefault("Width", "-1");
        width = atoi(attrib.c_str());
        attrib = itrContour->GetAttributeOrDefault("Height", "-1");
        height = atoi(attrib.c_str());
        attrib = itrContour->GetAttributeOrDefault("Begin", "-1");
        begIndexInc = atoi(attrib.c_str());
        attrib = itrContour->GetAttributeOrDefault("End", "-1");
        endIndexInc = atoi(attrib.c_str());
        if (width <= 0 || height <= 0 || begIndexInc < 0 || endIndexInc < 0)
        {
            success = false;
            break;
        }

        std::vector<IntervaledContour> currContours;
        Element* ptrVideo = itrContour->FirstChildElement("Video", false);
        if (ptrVideo == NULL)
            continue;

        for (ticpp::Iterator<ticpp::Element> itrVideo(ptrContour, "Video"); itrContour != itrContour.end(); ++itrContour)
        {
            std::string attrib = itrVideo->GetAttributeOrDefault("ID", "-1");
            int videoIndex = atoi(attrib.c_str());
            if (videoIndex < 0)
            {
                success = false;
                break;
            }

            IntervaledContour currContour;
            try
            {
                Element* ptrPoints = itrVideo->FirstChildElement("Points", false);
                if (ptrPoints == NULL)
                    continue;
                for (ticpp::Iterator<ticpp::Element> itrPoints(ptrPoints, "Points"); itrPoints != itrPoints.end(); ++itrPoints)
                {
                    std::string text = itrPoints->GetText(false);
                    if (text.empty())
                        continue;
                    currContour.contours.resize(currContour.contours.size() + 1);
                    if (!cvtStringToPoints(text, currContour.contours.back()))
                    {
                        success = false;
                        break;
                    }
                }
                if (!success)
                    break;
            }
            catch (...)
            {
                success = false;
                break;
            }

            currContour.videoIndex = videoIndex;
            currContour.width = width;
            currContour.height = height;
            currContour.begIndexInc = begIndexInc;
            currContour.endIndexInc = endIndexInc;
            currContours.push_back(currContour);
        }
        if (!success)
            break;

        contours.push_back(currContours);
    }
    if (!success)
        contours.clear();
    return success;
}

