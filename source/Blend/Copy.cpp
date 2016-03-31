#include "ZBlendAlgo.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

void copyIfIntersect(const cv::Mat& src, cv::Mat& dst, 
    const cv::Rect& srcRect, const cv::Rect& dstRect)
{
    CV_Assert(src.data && dst.data && src.type() == dst.type());
    CV_Assert(src.size() == srcRect.size() && dst.size() == dstRect.size());

    cv::Rect intersectRect = srcRect & dstRect;
    if (intersectRect.area() == 0)
        return;

    cv::Rect insideSrcRect = intersectRect - srcRect.tl();
    cv::Rect insideDstRect = intersectRect - dstRect.tl();
    cv::Mat dstROI(dst, insideDstRect);
    src(insideSrcRect).copyTo(dstROI);
}

void horiCircularRepeat(cv::Mat& image, int xBegin, int period)
{
    CV_Assert(image.data);
    
    if (xBegin >= 0 || xBegin + image.cols <= period)
    {
        //printf("nothing to do in copyIfHoriRepeat\n");
        return;
    }

    cv::Rect rect(xBegin, 0, image.cols, image.rows);
    cv::Point offset(xBegin, 0);
    int begin = (xBegin - period + 1) / period;
    int end = (xBegin + image.cols - 1) / period + 1;
    for (int i = begin; i < end; i++)
    {
        if (i == 0)
            continue;
        cv::Rect maxRect(i * period, 0, period, image.rows);
        cv::Rect currRect = rect & maxRect;
        cv::Rect srcRect = currRect - cv::Point(i * period, 0) - offset;
        cv::Rect dstRect = currRect - offset;
        //printf("%d, srcRect = (%d, %d, %d, %d), dstRect = (%d, %d, %d, %d)\n",
        //    i, srcRect.x, srcRect.y, srcRect.width, srcRect.height,
        //    dstRect.x, dstRect.y, dstRect.width, dstRect.height);
        cv::Mat dstROI(image, dstRect);
        image(srcRect).copyTo(dstROI);
    }
}

void horiCircularExpand(const cv::Mat& src, const cv::Rect& dstRect, cv::Mat& dst)
{
    CV_Assert(src.data);
    CV_Assert(dstRect.y >= 0 && dstRect.y + dstRect.height <= src.rows);

    dst.create(dstRect.size(), src.type());
    int period = src.cols;
    int begin = (dstRect.x - period + 1) / period;
    int end = (dstRect.x + dstRect.width - 1) / period + 1;
    for (int i = begin; i < end; i++)
    {
        cv::Rect maxRect(i * period, dstRect.y, period, dstRect.height);
        cv::Rect currRect = dstRect & maxRect;
        cv::Rect relativeSrcRect = currRect - cv::Point(i * period, 0);
        cv::Rect relativeDstRect = currRect - dstRect.tl();
        cv::Mat dstROI(dst, relativeDstRect);
        src(relativeSrcRect).copyTo(dstROI);
    }
}

void horiCircularFold(const cv::Mat& src, const cv::Rect& srcRect, cv::Mat& dst)
{
    CV_Assert(src.data && dst.data && src.type() == dst.type());
    CV_Assert(src.size() == srcRect.size());
    CV_Assert(srcRect.y >= 0 && srcRect.y + srcRect.height <= dst.rows);

    int period = dst.cols;
    int begin = (srcRect.x - period + 1) / period;
    int end = (srcRect.x + srcRect.width - 1) / period + 1;
    for (int i = begin; i < end; i++)
    {
        cv::Rect maxRect(i * period, srcRect.y, period, srcRect.height);
        cv::Rect currRect = srcRect & maxRect;
        cv::Rect relativeSrcRect = currRect - srcRect.tl();
        cv::Rect relativeDstRect = currRect - cv::Point(i * period, 0);
        cv::Mat dstROI(dst, relativeDstRect);
        src(relativeSrcRect).copyTo(dstROI);
    }
}

void horiCircularFold(const cv::Mat& src, const cv::Mat& srcMask, const cv::Rect& srcRect, cv::Mat& dst)
{
    CV_Assert(src.data && dst.data && src.type() == dst.type());
    CV_Assert(srcMask.data && srcMask.type() == CV_8UC1);
    CV_Assert(src.size() == srcMask.size() && src.size() == srcRect.size());
    CV_Assert(srcRect.y >= 0 && srcRect.y + srcRect.height <= dst.rows);

    int period = dst.cols;
    int begin = (srcRect.x - period + 1) / period;
    int end = (srcRect.x + srcRect.width - 1) / period + 1;
    for (int i = begin; i < end; i++)
    {
        cv::Rect maxRect(i * period, srcRect.y, period, srcRect.height);
        cv::Rect currRect = srcRect & maxRect;
        cv::Rect relativeSrcRect = currRect - srcRect.tl();
        cv::Rect relativeDstRect = currRect - cv::Point(i * period, 0);
        cv::Mat dstROI(dst, relativeDstRect), maskROI(srcMask, relativeSrcRect);
        src(relativeSrcRect).copyTo(dstROI, maskROI);
    }
}

void horiCircularFold(const cv::Rect& src, int period, std::vector<cv::Rect>& dst)
{
    dst.clear();
    int begin = (src.x - period + 1) / period;
    int end = (src.x + src.width - 1) / period + 1;
    dst.resize(end - begin);
    for (int i = begin; i < end; i++)
    {
        cv::Rect maxRect(i * period, src.y, period, src.height);
        dst[i - begin] = (src & maxRect) - cv::Point(i * period, 0);
    }
}