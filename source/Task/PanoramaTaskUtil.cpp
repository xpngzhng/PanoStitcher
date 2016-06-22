#include "PanoramaTaskUtil.h"
#include "CudaPanoramaTaskUtil.h"
#include "Image.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <stdarg.h>

bool prepareSrcVideos(const std::vector<std::string>& srcVideoFiles, avp::PixelType pixelType, const std::vector<int>& offsets,
    int tryAudioIndex, std::vector<avp::AudioVideoReader3>& readers, int& audioIndex, cv::Size& srcSize, int& validFrameCount)
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
            ok = readers[i].open(srcVideoFiles[i], true, avp::SampleTypeUnknown, true, pixelType);
            if (ok)
                audioIndex = tryAudioIndex;
            else
            {
                ok = readers[i].open(srcVideoFiles[i], false, avp::SampleTypeUnknown, true, pixelType);
                audioIndex = -1;
            }
        }
        else
            ok = readers[i].open(srcVideoFiles[i], false, avp::SampleTypeUnknown, true, pixelType);
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
            fps = readers[i].getVideoFrameRate();
        if (abs(fps - readers[i].getVideoFrameRate()) > 0.1)
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

bool LogoFilter::addLogo(cv::Mat& image) const
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

bool CudaLogoFilter::init(int width_, int height_)
{
    initSuccess = false;
    if (width_ < 0 || height_ < 0)
        return false;

    width = width_;
    height = height_;

    cv::Mat origLogo(logoHeight, logoWidth, CV_8UC4, logoData);
    cv::Mat fullLogo;

    int blockWidth = 512, blockHeight = 512;
    if (width < logoWidth || height < logoHeight)
    {
        cv::Rect logoRect(logoWidth / 2 - width / 2, logoHeight / 2 - height / 2, width, height);
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
                cv::Rect thisRect = cv::Rect(j * blockWidth + blockWidth / 2 - logoWidth / 2,
                    i * blockHeight + blockHeight / 2 - logoHeight / 2,
                    logoWidth, logoHeight) &
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

bool CudaLogoFilter::addLogo(cv::cuda::GpuMat& image) const
{
    if (!initSuccess)
        return false;

    if (!image.data || image.rows != height || image.cols != width || image.type() != CV_8UC4)
        return false;

    alphaBlend8UC4(image, logo);
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
                ptrElement->GetText(&currContour.begIncInMilliSec);
                ptrElement = itrContour->FirstChildElement("End");
                ptrElement->GetText(&currContour.endExcInMilliSec);

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
