#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

static const int UNIT_SHIFT = 10;
static const int UNIT = 1 << UNIT_SHIFT;

void prepare(const cv::Mat& mask1, const cv::Mat& mask2,
    cv::Mat& from1, cv::Mat& from2, cv::Mat& intersect, cv::Mat& weight1, cv::Mat& weight2)
{
    cv::Mat dist1, dist2;
    cv::distanceTransform(mask1, dist1, CV_DIST_L1, 3);
    cv::distanceTransform(mask2, dist2, CV_DIST_L1, 3);

    cv::Mat newMask1 = dist1 > dist2;
    cv::Mat newMask2 = ~newMask1;
    newMask1 &= mask1;
    newMask2 &= mask2;

    int radius = 60;
    cv::Size kernSize(radius * 2 + 1, radius * 2 + 1);
    double sigma = radius / 3.0;
    cv::Mat blurMask1, blurMask2;
    cv::GaussianBlur(newMask1, blurMask1, kernSize, sigma, sigma);
    cv::GaussianBlur(newMask2, blurMask2, kernSize, sigma, sigma);
    //cv::imshow("orig blur mask 1", blurMask1);
    //cv::imshow("orig blur mask 2", blurMask2);
    blurMask1 &= mask1;
    blurMask2 &= mask2;
    //cv::imshow("mask 1", mask1);
    //cv::imshow("mask 2", mask2);
    //cv::imshow("blur mask 1", blurMask1);
    //cv::imshow("blur mask 2", blurMask2);
    //cv::waitKey(0);

    int rows = mask1.rows, cols = mask1.cols;

    from1.create(rows, cols, CV_8UC1);
    from1.setTo(0);
    from2.create(rows, cols, CV_8UC1);
    from2.setTo(0);
    intersect.create(rows, cols, CV_8UC1);
    intersect.setTo(0);
    weight1.create(rows, cols, CV_32SC1);
    weight1.setTo(0);
    weight2.create(rows, cols, CV_32SC1);
    weight2.setTo(0);

    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrMask1 = blurMask1.ptr<unsigned char>(i);
        const unsigned char* ptrMask2 = blurMask2.ptr<unsigned char>(i);
        unsigned char* ptrFrom1 = from1.ptr<unsigned char>(i);
        unsigned char* ptrFrom2 = from2.ptr<unsigned char>(i);
        unsigned char* ptrIntersect = intersect.ptr<unsigned char>(i);
        int* ptrWeight1 = weight1.ptr<int>(i);
        int* ptrWeight2 = weight2.ptr<int>(i);
        for (int j = 0; j < cols; j++)
        {
            double w1 = ptrMask1[j], w2 = ptrMask2[j];
            double wsum = w1 + w2;
            if (wsum < std::numeric_limits<double>::epsilon())
                continue;

            w1 /= wsum;
            w2 /= wsum;

            int iw1 = w1 * UNIT + 0.5;
            int iw2 = UNIT - iw1;

            if (iw1 == UNIT)
            {
                ptrFrom1[j] = 255;
            }
            else if (iw2 == UNIT)
            {
                ptrFrom2[j] = 255;
            }
            else
            {
                ptrIntersect[j] = 255;
            }

            ptrWeight1[j] = iw1;
            ptrWeight2[j] = iw2;
        }
    }
}

void prepareSmart(const cv::Mat& mask1, const cv::Mat& mask2, int initRadius,
    cv::Mat& from1, cv::Mat& from2, cv::Mat& intersect, cv::Mat& weight1, cv::Mat& weight2)
{
    cv::Mat dist1, dist2;
    cv::distanceTransform(mask1, dist1, CV_DIST_L1, 3);
    cv::distanceTransform(mask2, dist2, CV_DIST_L1, 3);

    cv::Mat region1 = dist1 > dist2;
    cv::Mat region2 = ~region1;
    cv::Mat newMask1 = region1 & mask1;
    cv::Mat newMask2 = region2 & mask2;
    region1 |= mask1;
    region2 |= mask2;
    region1 = ~region1;
    region2 = ~region2;

    int step = 2;
    cv::Mat blurMask1, blurMask2;
    cv::Mat binaryBlurMask1, binaryBlurMask2;
    cv::Mat outSide1, outSide2;
    for (int radius = initRadius; radius >= 1; radius -= step)
    {
        cv::Size kernSize(radius * 2 + 1, radius * 2 + 1);
        double sigma = radius / 3.0;
        cv::GaussianBlur(newMask1, blurMask1, kernSize, sigma, sigma);
        cv::GaussianBlur(newMask2, blurMask2, kernSize, sigma, sigma);
        binaryBlurMask1 = blurMask1 > 0;
        binaryBlurMask2 = blurMask2 > 0;
        int numInside1 = cv::countNonZero(binaryBlurMask1 & region1);
        int numInside2 = cv::countNonZero(binaryBlurMask2 & region2);
        if (numInside1 == 0 && numInside2 == 0)
        {
            printf("final radius = %d\n", radius);
            break;
        }
    }
    
    //cv::imshow("orig blur mask 1", blurMask1);
    //cv::imshow("orig blur mask 2", blurMask2);
    blurMask1 &= mask1;
    blurMask2 &= mask2;
    //cv::imshow("mask 1", mask1);
    //cv::imshow("mask 2", mask2);
    //cv::imshow("blur mask 1", blurMask1);
    //cv::imshow("blur mask 2", blurMask2);
    //cv::waitKey(0);

    int rows = mask1.rows, cols = mask1.cols;

    from1.create(rows, cols, CV_8UC1);
    from1.setTo(0);
    from2.create(rows, cols, CV_8UC1);
    from2.setTo(0);
    intersect.create(rows, cols, CV_8UC1);
    intersect.setTo(0);
    weight1.create(rows, cols, CV_32SC1);
    weight1.setTo(0);
    weight2.create(rows, cols, CV_32SC1);
    weight2.setTo(0);

    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrMask1 = blurMask1.ptr<unsigned char>(i);
        const unsigned char* ptrMask2 = blurMask2.ptr<unsigned char>(i);
        unsigned char* ptrFrom1 = from1.ptr<unsigned char>(i);
        unsigned char* ptrFrom2 = from2.ptr<unsigned char>(i);
        unsigned char* ptrIntersect = intersect.ptr<unsigned char>(i);
        int* ptrWeight1 = weight1.ptr<int>(i);
        int* ptrWeight2 = weight2.ptr<int>(i);
        for (int j = 0; j < cols; j++)
        {
            double w1 = ptrMask1[j], w2 = ptrMask2[j];
            double wsum = w1 + w2;
            if (wsum < std::numeric_limits<double>::epsilon())
                continue;

            w1 /= wsum;
            w2 /= wsum;

            int iw1 = w1 * UNIT + 0.5;
            int iw2 = UNIT - iw1;

            if (iw1 == UNIT)
            {
                ptrFrom1[j] = 255;
            }
            else if (iw2 == UNIT)
            {
                ptrFrom2[j] = 255;
            }
            else
            {
                ptrIntersect[j] = 255;
            }

            ptrWeight1[j] = iw1;
            ptrWeight2[j] = iw2;
        }
    }
}

void blend(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& from1, const cv::Mat& from2, 
    const cv::Mat& intersect, const cv::Mat& weight1, const cv::Mat& weight2, cv::Mat& blendImage)
{
    int rows = image1.rows, cols = image1.cols;
    blendImage.create(rows, cols, CV_8UC3);
    blendImage.setTo(0);
    image1.copyTo(blendImage, from1);
    image2.copyTo(blendImage, from2);
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrImage1 = image1.ptr<unsigned char>(i);
        const unsigned char* ptrImage2 = image2.ptr<unsigned char>(i);
        const unsigned char* ptrInt = intersect.ptr<unsigned char>(i);
        const int* ptrW1 = weight1.ptr<int>(i);
        const int* ptrW2 = weight2.ptr<int>(i);
        unsigned char* ptrB = blendImage.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrInt[j])
            {
                int w1 = ptrW1[j], w2 = ptrW2[j];
                ptrB[j * 3] = (ptrImage1[j * 3] * w1 + ptrImage2[j * 3] * w2) >> UNIT_SHIFT;
                ptrB[j * 3 + 1] = (ptrImage1[j * 3 + 1] * w1 + ptrImage2[j * 3 + 1] * w2) >> UNIT_SHIFT;
                ptrB[j * 3 + 2] = (ptrImage1[j * 3 + 2] * w1 + ptrImage2[j * 3 + 2] * w2) >> UNIT_SHIFT;
            }
        }
    }
}

inline void bilinearResampling(int width, int height, int step, const unsigned char* data,
    double x, double y, unsigned char rgb[3])
{
    int x0 = x, y0 = y, x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0) x0 = 0;
    if (x1 > width - 1) x1 = width - 1;
    if (y0 < 0) y0 = 0;
    if (y1 > height - 1) y1 = height - 1;
    double wx0 = x - x0, wx1 = 1 - wx0;
    double wy0 = y - y0, wy1 = 1 - wy0;
    double w00 = wx1 * wy1, w01 = wx0 * wy1;
    double w10 = wx1 * wy0, w11 = wx0 * wy0;

    double b = 0, g = 0, r = 0;
    const unsigned char* ptr;
    ptr = data + step * y0 + x0 * 3;
    b += *(ptr++) * w00;
    g += *(ptr++) * w00;
    r += *(ptr++) * w00;
    b += *(ptr++) * w01;
    g += *(ptr++) * w01;
    r += *(ptr++) * w01;
    ptr = data + step * y1 + x0 * 3;
    b += *(ptr++) * w10;
    g += *(ptr++) * w10;
    r += *(ptr++) * w10;
    b += *(ptr++) * w11;
    g += *(ptr++) * w11;
    r += *(ptr++) * w11;

    rgb[0] = b;
    rgb[1] = g;
    rgb[2] = r;
}

void reprojectAndBlend(const cv::Mat& src1, const cv::Mat& src2, 
    const cv::Mat& dstSrcMap1, const cv::Mat& dstSrcMap2,
    const cv::Mat& from1, const cv::Mat& from2, const cv::Mat& intersect, 
    const cv::Mat& weight1, const cv::Mat& weight2, cv::Mat& dst)
{
    /*CV_Assert(src1.data && src2.data);
    CV_Assert(src1.type() == CV_8UC3 && src2.type() == CV_8UC3);    
    CV_Assert(dstSrcMap1.data && dstSrcMap2.data && from1.data && from2.data &&
        intersect.data && weight1.data && weight2.data);
    CV_Assert(dstSrcMap1.type() == CV_64FC2 && dstSrcMap2.type() == CV_64FC2 &&
        from1.type() == CV_8UC1 && from2.type() == CV_8UC1 && intersect.type() == CV_8UC1 &&
        weight1.type() == CV_32SC1 && weight2.type() == CV_32SC1);
    cv::Size dstSize = intersect.size();
    CV_Assert(dstSrcMap1.size() == dstSize && dstSrcMap2.size() == dstSize &&
        from1.size() == dstSize && from2.size() == dstSize &&
        weight1.size() == dstSize && weight2.size() == dstSize);*/

    int rows = intersect.rows, cols = intersect.cols;
    dst.create(rows, cols, CV_8UC3);
    dst.setTo(0);

    int src1Width = src1.cols, src1Height = src1.rows, src1Step = src1.step;
    int src2Width = src2.cols, src2Height = src2.rows, src2Step = src2.step;
    const unsigned char* src1Data = src1.data;
    const unsigned char* src2Data = src2.data;
    for (int i = 0; i < rows; i++)
    {
        const cv::Point2d* ptrMap1 = dstSrcMap1.ptr<cv::Point2d>(i);
        const cv::Point2d* ptrMap2 = dstSrcMap2.ptr<cv::Point2d>(i);
        const unsigned char* ptrFrom1 = from1.ptr<unsigned char>(i);
        const unsigned char* ptrFrom2 = from2.ptr<unsigned char>(i);
        const unsigned char* ptrIntersect = intersect.ptr<unsigned char>(i);
        const int* ptrWeight1 = weight1.ptr<int>(i);
        const int* ptrWeight2 = weight2.ptr<int>(i);
        unsigned char* ptrDst = dst.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrFrom1[j])
            {
                cv::Point2d pt = ptrMap1[j];
                bilinearResampling(src1Width, src1Height, src1Step, src1Data, pt.x, pt.y, ptrDst + j * 3);
            }
            else if (ptrFrom2[j])
            {
                cv::Point2d pt = ptrMap2[j];
                bilinearResampling(src2Width, src2Height, src2Step, src2Data, pt.x, pt.y, ptrDst + j * 3);
            }
            else if (ptrIntersect[j])
            {
                cv::Point2d pt;
                unsigned char bgr1[3], bgr2[3];
                pt = ptrMap1[j];
                bilinearResampling(src1Width, src1Height, src1Step, src1Data, pt.x, pt.y, bgr1);
                pt = ptrMap2[j];
                bilinearResampling(src2Width, src2Height, src2Step, src2Data, pt.x, pt.y, bgr2);
                int w1 = ptrWeight1[j], w2 = ptrWeight2[j];
                ptrDst[j * 3] = (bgr1[0] * w1 + bgr2[0] * w2) >> UNIT_SHIFT;
                ptrDst[j * 3 + 1] = (bgr1[1] * w1 + bgr2[1] * w2) >> UNIT_SHIFT;
                ptrDst[j * 3 + 2] = (bgr1[2] * w1 + bgr2[2] * w2) >> UNIT_SHIFT;
            }
        }
    }
}

class ReprojectAndBlendLoop : public cv::ParallelLoopBody
{
public:
    ReprojectAndBlendLoop(const cv::Mat& src1_, const cv::Mat& src2_,
        const cv::Mat& dstSrcMap1_, const cv::Mat& dstSrcMap2_,
        const cv::Mat& from1_, const cv::Mat& from2_, const cv::Mat& intersect_,
        const cv::Mat& weight1_, const cv::Mat& weight2_, cv::Mat& dst_)
        : src1(src1_), src2(src2_), dstSrcMap1(dstSrcMap1_), dstSrcMap2(dstSrcMap2_),
          from1(from1_), from2(from2_), intersect(intersect_),
          weight1(weight1_), weight2(weight2_), dst(dst_)
    {
        src1Width = src1.cols, src1Height = src1.rows, src1Step = src1.step;
        src2Width = src2.cols, src2Height = src2.rows, src2Step = src2.step;
        src1Data = src1.data;
        src2Data = src2.data;
        dstWidth = dst.cols, dstHeight = dst.rows;
    }

    virtual ~ReprojectAndBlendLoop() {}

    virtual void operator()(const cv::Range& r) const
    {
        int start = r.start, end = std::min(r.end, dstHeight);
        for (int i = start; i < end; i++)
        {
            const cv::Point2d* ptrMap1 = dstSrcMap1.ptr<cv::Point2d>(i);
            const cv::Point2d* ptrMap2 = dstSrcMap2.ptr<cv::Point2d>(i);
            const unsigned char* ptrFrom1 = from1.ptr<unsigned char>(i);
            const unsigned char* ptrFrom2 = from2.ptr<unsigned char>(i);
            const unsigned char* ptrIntersect = intersect.ptr<unsigned char>(i);
            const int* ptrWeight1 = weight1.ptr<int>(i);
            const int* ptrWeight2 = weight2.ptr<int>(i);
            unsigned char* ptrDst = dst.ptr<unsigned char>(i);
            for (int j = 0; j < dstWidth; j++)
            {
                if (ptrFrom1[j])
                {
                    cv::Point2d pt = ptrMap1[j];
                    bilinearResampling(src1Width, src1Height, src1Step, src1Data, pt.x, pt.y, ptrDst + j * 3);
                }
                else if (ptrFrom2[j])
                {
                    cv::Point2d pt = ptrMap2[j];
                    bilinearResampling(src2Width, src2Height, src2Step, src2Data, pt.x, pt.y, ptrDst + j * 3);
                }
                else if (ptrIntersect[j])
                {
                    cv::Point2d pt;
                    unsigned char bgr1[3], bgr2[3];
                    pt = ptrMap1[j];
                    bilinearResampling(src1Width, src1Height, src1Step, src1Data, pt.x, pt.y, bgr1);
                    pt = ptrMap2[j];
                    bilinearResampling(src2Width, src2Height, src2Step, src2Data, pt.x, pt.y, bgr2);
                    int w1 = ptrWeight1[j], w2 = ptrWeight2[j];
                    ptrDst[j * 3] = (bgr1[0] * w1 + bgr2[0] * w2) >> UNIT_SHIFT;
                    ptrDst[j * 3 + 1] = (bgr1[1] * w1 + bgr2[1] * w2) >> UNIT_SHIFT;
                    ptrDst[j * 3 + 2] = (bgr1[2] * w1 + bgr2[2] * w2) >> UNIT_SHIFT;
                }
            }
        }
    }

    const cv::Mat& src1; const cv::Mat& src2;
    const cv::Mat& dstSrcMap1; const cv::Mat& dstSrcMap2;
    const cv::Mat& from1; const cv::Mat& from2; const cv::Mat& intersect;
    const cv::Mat& weight1; const cv::Mat& weight2; cv::Mat& dst;
    int src1Width, src1Height, src1Step;
    int src2Width, src2Height, src2Step;
    const unsigned char* src1Data;
    const unsigned char* src2Data;
    int dstWidth, dstHeight;
};

void reprojectAndBlendParallel(const cv::Mat& src1, const cv::Mat& src2,
    const cv::Mat& dstSrcMap1, const cv::Mat& dstSrcMap2,
    const cv::Mat& from1, const cv::Mat& from2, const cv::Mat& intersect,
    const cv::Mat& weight1, const cv::Mat& weight2, cv::Mat& dst)
{
    int rows = intersect.rows, cols = intersect.cols;
    dst.create(rows, cols, CV_8UC3);
    dst.setTo(0);

    ReprojectAndBlendLoop loop(src1, src2, dstSrcMap1, dstSrcMap2, from1, from2, intersect, weight1, weight2, dst);
    cv::parallel_for_(cv::Range(0, rows), loop);
}

#include "RicohUtil.h"
#include "ZReproject.h"

struct RicohPanoramaRender::Impl
{
    Impl() : success(0) {};
    bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize);
    void render(const cv::Mat& src, cv::Mat& dst);

    cv::Size srcFullSize;
    cv::Mat dstSrcMap1, dstSrcMap2;
    cv::Mat from1, from2, intersect;
    cv::Mat weight1, weight2;
    int success;
};

bool RicohPanoramaRender::Impl::prepare(const std::string& path, 
    const cv::Size& srcSize, const cv::Size& dstSize)
{
    success = 0;

    if (!((dstSize.width & 1) == 0 && (dstSize.height & 1) == 0 &&
        dstSize.height * 2 == dstSize.width))
        return false;

    srcFullSize = srcSize;

    std::vector<PhotoParam> params;
    loadPhotoParamFromXML(path, params);
    if (params.size() != 2)
        return false;

    cv::Mat mask1, mask2;
    getReprojectMapAndMask(params[0], srcSize, dstSize, dstSrcMap1, mask1);
    getReprojectMapAndMask(params[1], srcSize, dstSize, dstSrcMap2, mask2);

    ::prepare(mask1, mask2, from1, from2, intersect, weight1, weight2);

    success = 1;
    return true;
}

void RicohPanoramaRender::Impl::render(const cv::Mat& src, cv::Mat& dst)
{
    if (!success)
        return;

    CV_Assert(src.data && src.type() == CV_8UC3 && src.size() == srcFullSize);

    reprojectAndBlendParallel(src, src, dstSrcMap1, dstSrcMap2,
        from1, from2, intersect, weight1, weight2, dst);
}

RicohPanoramaRender::RicohPanoramaRender()
{
    ptrImpl.reset(new Impl);
}

bool RicohPanoramaRender::prepare(const std::string& path, 
    const cv::Size& srcSize, const cv::Size& dstSize)
{
    return ptrImpl->prepare(path, srcSize, dstSize);
}

void RicohPanoramaRender::render(const cv::Mat& src, cv::Mat& dst)
{
    ptrImpl->render(src, dst);
}

struct DetuPanoramaRender::Impl
{
    Impl() : success(0) {};
    bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize);
    void render(const cv::Mat& src, cv::Mat& dst);

    cv::Size srcFullSize;
    cv::Mat dstSrcMap;
    int success;
};

bool DetuPanoramaRender::Impl::prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize)
{
    success = 0;

    if (!((dstSize.width & 1) == 0 && (dstSize.height & 1) == 0 &&
        dstSize.height * 2 == dstSize.width))
        return false;

    srcFullSize = srcSize;

    PhotoParam param;
    loadPhotoParamFromXML(path, param);

    cv::Mat mask;
    getReprojectMapAndMask(param, srcSize, dstSize, dstSrcMap, mask);

    success = 1;
    return true;
}

void DetuPanoramaRender::Impl::render(const cv::Mat& src, cv::Mat& dst)
{
    if (!success)
        return;

    CV_Assert(src.size() == srcFullSize);
    
    dst.create(dstSrcMap.size(), CV_8UC3);
    
    reprojectParallel(src, dst, dstSrcMap);
}

DetuPanoramaRender::DetuPanoramaRender()
{
    ptrImpl.reset(new Impl);
}

bool DetuPanoramaRender::prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize)
{
    return ptrImpl->prepare(path, srcSize, dstSize);
}

void DetuPanoramaRender::render(const cv::Mat& src, cv::Mat& dst)
{
    ptrImpl->render(src, dst);
}

struct DualGoProPanoramaRender::Impl
{
    Impl() : success(0) {};
    bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst);
    bool getMasks(std::vector<cv::Mat>& masks) const;

    cv::Size srcFullSize;
    cv::Mat dstSrcMap1, dstSrcMap2;
    cv::Mat mask1, mask2;
    cv::Mat from1, from2, intersect;
    cv::Mat weight1, weight2;
    int success;
};

bool DualGoProPanoramaRender::Impl::prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize)
{
    success = 0;

    if (!((dstSize.width & 1) == 0 && (dstSize.height & 1) == 0 &&
        dstSize.height * 2 == dstSize.width))
        return false;

    srcFullSize = srcSize;

    std::vector<PhotoParam> params;
    loadPhotoParamFromPTS(path, params);
    if (params.size() != 2)
        return false;

    PhotoParam param1 = params[0], param2 = params[1];
    getReprojectMapAndMask(param1, srcSize, dstSize, dstSrcMap1, mask1);
    getReprojectMapAndMask(param2, srcSize, dstSize, dstSrcMap2, mask2);

    ::prepareSmart(mask1, mask2, 50, from1, from2, intersect, weight1, weight2);
    //::prepare(mask1, mask2, from1, from2, intersect, weight1, weight2);

    success = 1;
    return true;    
}

bool DualGoProPanoramaRender::Impl::render(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst)
{
    if (!success)
        return false;

    if (!(src1.data && src2.data && src1.type() == CV_8UC3 && src2.type() == CV_8UC3 &&
        src1.size() == srcFullSize && src2.size() == srcFullSize))
        return false;

    reprojectAndBlendParallel(src1, src2, dstSrcMap1, dstSrcMap2,
        from1, from2, intersect, weight1, weight2, dst);
    return true;
}

bool DualGoProPanoramaRender::Impl::getMasks(std::vector<cv::Mat>& masks) const
{
    if (!success)
        return false;

    masks.resize(2);
    mask1.copyTo(masks[0]);
    mask2.copyTo(masks[1]);
    return true;
}

DualGoProPanoramaRender::DualGoProPanoramaRender()
{
    ptrImpl.reset(new Impl);
}

bool DualGoProPanoramaRender::prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize)
{
    return ptrImpl->prepare(path, srcSize, dstSize);
}

bool DualGoProPanoramaRender::render(const std::vector<cv::Mat>& src, cv::Mat& dst)
{
    if (src.size() != 2)
        return false;

    return ptrImpl->render(src[0], src[1], dst);
}

#include "ZBlend.h"

struct CPUMultiCameraPanoramaRender::Impl
{
    Impl() : success(0) {};
    bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const std::vector<cv::Mat>& src, cv::Mat& dst);

    cv::Size srcFullSize;
    std::vector<cv::Mat> dstSrcMaps;
    std::vector<cv::Mat> masks;
    std::vector<cv::Mat> reprojImages;
    TilingMultibandBlendFastParallel blender;
    int numImages;
    int success;
};

bool CPUMultiCameraPanoramaRender::Impl::prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize)
{
    success = 0;

    if (!((dstSize.width & 1) == 0 && (dstSize.height & 1) == 0 &&
        dstSize.height * 2 == dstSize.width))
        return false;

    srcFullSize = srcSize;

    std::string::size_type length = path.length();
    std::string fileExt = path.substr(length - 3, 3);
    std::vector<PhotoParam> params;
    try
    {
        if (fileExt == "pts")
            loadPhotoParamFromPTS(path, params);
        else /*if (fileExt == "xml")*/
            loadPhotoParamFromXML(path, params);
        /*else
            return false;*/
    }
    catch (...)
    {
        printf("load file error\n");
        return false;
    }

    numImages = params.size();
    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, masks);
    if (!blender.prepare(masks, 20, 2))
        return false;

    success = 1;
    return true;
}

bool CPUMultiCameraPanoramaRender::Impl::render(const std::vector<cv::Mat>& src, cv::Mat& dst)
{
    if (!success)
        return false;

    if (src.size() != numImages)
        return false;

    for (int i = 0; i < numImages; i++)
    {
        if (src[i].size() != srcFullSize)
            return false;
    }

    reprojImages.resize(numImages);
    for (int i = 0; i < numImages; i++)
        reprojectParallel(src[i], reprojImages[i], dstSrcMaps[i]);

    blender.blend(reprojImages, dst);
    return true;
}

CPUMultiCameraPanoramaRender::CPUMultiCameraPanoramaRender()
{
    ptrImpl.reset(new Impl);
}

bool CPUMultiCameraPanoramaRender::prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize)
{
    return ptrImpl->prepare(path, srcSize, dstSize);
}

bool CPUMultiCameraPanoramaRender::render(const std::vector<cv::Mat>& src, cv::Mat& dst)
{
   return ptrImpl->render(src, dst);
}

struct CudaMultiCameraPanoramaRender::Impl
{
    Impl() : success(0) {};
    bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const std::vector<cv::Mat>& src, cv::Mat& dst);

    cv::Size srcFullSize;
    std::vector<cv::gpu::GpuMat> dstSrcXMapsGPU, dstSrcYMapsGPU;
    std::vector<cv::gpu::CudaMem> srcMems;
    std::vector<cv::Mat> srcImages;
    std::vector<cv::gpu::GpuMat> srcImagesGPU;
    std::vector<cv::gpu::GpuMat> reprojImagesGPU;
    cv::gpu::GpuMat blendImageGPU;
    cv::Mat blendImage;
    std::vector<cv::gpu::Stream> streams;
    CudaTilingMultibandBlendFast blender;
    int numImages;
    int success;
};

bool CudaMultiCameraPanoramaRender::Impl::prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize)
{
    success = 0;

    if (!((dstSize.width & 1) == 0 && (dstSize.height & 1) == 0 &&
        dstSize.height * 2 == dstSize.width))
        return false;

    srcFullSize = srcSize;

    std::string::size_type length = path.length();
    std::string fileExt = path.substr(length - 3, 3);
    std::vector<PhotoParam> params;
    try
    {
        if (fileExt == "pts")
            loadPhotoParamFromPTS(path, params);
        else /*if (fileExt == "xml")*/
            loadPhotoParamFromXML(path, params);
        /*else
            return false;*/
    }
    catch (...)
    {
        printf("load file error\n");
        return false;
    }

    numImages = params.size();
    std::vector<cv::Mat> masks, dstSrcMaps;
    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, masks);
    if (!blender.prepare(masks, 20, 2))
        return false;

    cudaGenerateReprojectMaps(params, srcSize, dstSize, dstSrcXMapsGPU, dstSrcYMapsGPU);
    srcMems.resize(numImages);
    srcImages.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        srcMems[i].create(srcFullSize, CV_8UC4);
        srcImages[i] = srcMems[i];
    }

    streams.resize(numImages);

    success = 1;
    return true;
}

bool CudaMultiCameraPanoramaRender::Impl::render(const std::vector<cv::Mat>& src, cv::Mat& dst)
{
    if (!success)
        return false;

    if (src.size() != numImages)
        return false;

    for (int i = 0; i < numImages; i++)
    {
        if (src[i].size() != srcFullSize)
            return false;
        if (src[i].type() != CV_8UC3 && src[i].type() != CV_8UC4)
            return false;
    }

    //double freq = cv::getTickFrequency();
    //long long int beg = cv::getTickCount();

    int fromTo[] = { 0, 0, 1, 1, 2, 2 };
    for (int i = 0; i < numImages; i++)
    {
        if (src[i].type() == CV_8UC4)
            src[i].copyTo(srcImages[i]);
        else
            cv::mixChannels(&src[i], 1, &srcImages[i], 1, fromTo, 3);
    }

    srcImagesGPU.resize(numImages);
    reprojImagesGPU.resize(numImages);
    for (int i = 0; i < numImages; i++)
        streams[i].enqueueUpload(srcImages[i], srcImagesGPU[i]);
    for (int i = 0; i < numImages; i++)
        cudaReprojectTo16S(srcImagesGPU[i], reprojImagesGPU[i], dstSrcXMapsGPU[i], dstSrcYMapsGPU[i], streams[i]);
    for (int i = 0; i < numImages; i++)
        streams[i].waitForCompletion();

    blender.blend(reprojImagesGPU, blendImageGPU);
    blendImageGPU.download(dst);
    return true;
}

CudaMultiCameraPanoramaRender::CudaMultiCameraPanoramaRender()
{
    ptrImpl.reset(new Impl);
}

bool CudaMultiCameraPanoramaRender::prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize)
{
    return ptrImpl->prepare(path, srcSize, dstSize);
}

bool CudaMultiCameraPanoramaRender::render(const std::vector<cv::Mat>& src, cv::Mat& dst)
{
    return ptrImpl->render(src, dst);
}