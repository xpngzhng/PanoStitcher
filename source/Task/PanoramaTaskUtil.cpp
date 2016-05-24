#include "PanoramaTaskUtil.h"
#include "Image.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <stdarg.h>

bool prepareSrcVideos(const std::vector<std::string>& srcVideoFiles, bool bgr24, const std::vector<int>& offsets,
    int tryAudioIndex, std::vector<avp::AudioVideoReader>& readers, int& audioIndex, cv::Size& srcSize, int& validFrameCount)
{
    readers.clear();
    srcSize = cv::Size();
    validFrameCount = 0;

    if (srcVideoFiles.empty())
        return false;

    int numVideos = srcVideoFiles.size();
    bool hasOffsets = !offsets.empty();
    if (hasOffsets && offsets.size() != numVideos)
        return false;

    readers.resize(numVideos);

    if (tryAudioIndex < 0 || tryAudioIndex >= numVideos)
    {
        ptlprintf("Info in %s, no audio will be opened\n", __FUNCTION__);
        audioIndex = -1;
    }

    bool ok = false;
    double fps = -1;
    for (int i = 0; i < numVideos; i++)
    {
        if (i == tryAudioIndex)
        {
            ok = readers[i].open(srcVideoFiles[i], true, true, bgr24 ? avp::PixelTypeBGR24 : avp::PixelTypeBGR32);
            if (ok)
                audioIndex = tryAudioIndex;
            else
            {
                ok = readers[i].open(srcVideoFiles[i], false, true, bgr24 ? avp::PixelTypeBGR24 : avp::PixelTypeBGR32);
                audioIndex = -1;
            }
        }
        else
            ok = readers[i].open(srcVideoFiles[i], false, true, bgr24 ? avp::PixelTypeBGR24 : avp::PixelTypeBGR32);
        if (!ok)
            break;

        if (srcSize == cv::Size())
        {
            srcSize.width = readers[i].getVideoWidth();
            srcSize.height = readers[i].getVideoHeight();
        }
        if (srcSize.width != readers[i].getVideoWidth() ||
            srcSize.height != readers[i].getVideoHeight())
        {
            ok = false;
            ptlprintf("Error in %s, video size unmatch\n", __FUNCTION__);
            break;
        }

        if (fps < 0)
            fps = readers[i].getVideoFps();
        if (abs(fps - readers[i].getVideoFps()) > 0.1)
        {
            ptlprintf("Error in %s, video fps not consistent\n", __FUNCTION__);
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
                ptlprintf("Error in %s, video not long enough\n", __FUNCTION__);
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
            if (!readers[i].seek(1000000.0 * count / fps + 0.5, avp::VIDEO))
            {
                ptlprintf("Error in %s, cannot seek to target frame\n", __FUNCTION__);
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

bool LogoFilter::init(int width_, int height_, int type_)
{
    initSuccess = false;
    if (width_ < 0 || height_ < 0 || (type_ != CV_8UC3 && type_ != CV_8UC4))
        return false;

    width = width_;
    height = height_;
    type = type_;

    cv::Mat origLogo(logoHeight, logoWidth, CV_8UC4, logoData);

    int blockWidth = 512, blockHeight = 512;
    rects.clear();
    if (width < logoWidth || height < logoHeight)
    {
        cv::Rect logoRect(logoWidth / 2 - width / 2, logoHeight / 2 - height / 2, width, height);
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
                cv::Rect thisRect = cv::Rect(j * blockWidth + blockWidth / 2 - logoWidth / 2, 
                                             i * blockHeight + blockHeight / 2 - logoHeight / 2, 
                                             logoWidth, logoHeight) & 
                                    full;
                if (thisRect.area())
                    rects.push_back(thisRect);
            }
        }
    }

    initSuccess = true;
    return true;
}

bool LogoFilter::addLogo(cv::Mat& image)
{
    if (!initSuccess || !image.data || image.rows != height || image.cols != width || image.type() != type)
        return false;

    int size = rects.size();
    for (int i = 0; i < size; i++)
    {
        cv::Mat imagePart(image, rects[i]);
        cv::Mat logoPart(logo, cv::Rect(0, 0, rects[i].width, rects[i].height));
        alphaBlend(imagePart, logoPart);
    }

    return true;
}

void ptLogDefaultCallback(const char* format, va_list vl)
{
    vprintf(format, vl);
}

PanoTaskLogCallbackFunc ptLogCallback = ptLogDefaultCallback;

void ptlprintf(const char* format, ...)
{
    if (ptLogCallback)
    {
        va_list vl;
        va_start(vl, format);
        ptLogCallback(format, vl);
        va_end(vl);
    }    
}

PanoTaskLogCallbackFunc setPanoTaskLogCallback(PanoTaskLogCallbackFunc func)
{
    PanoTaskLogCallbackFunc oldFunc = ptLogCallback;
    ptLogCallback = func;
    return oldFunc;
}