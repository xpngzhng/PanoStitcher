#include "PanoramaTaskUtil.h"
#include "Timer.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <exception>

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
            ptlprintf("final radius = %d\n", radius);
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

bool RicohPanoramaRender::prepare(const std::string& path_, 
    const cv::Size& srcSize_, const cv::Size& dstSize_)
{
    success = 0;

    if (!((dstSize_.width & 1) == 0 && (dstSize_.height & 1) == 0 &&
        dstSize_.height * 2 == dstSize_.width))
        return false;

    srcSize = srcSize_;
    dstSize = dstSize_;

    std::vector<PhotoParam> params;
    loadPhotoParamFromXML(path_, params);
    if (params.size() != 2)
        return false;

    cv::Mat mask1, mask2;
    getReprojectMapAndMask(params[0], srcSize, dstSize, dstSrcMap1, mask1);
    getReprojectMapAndMask(params[1], srcSize, dstSize, dstSrcMap2, mask2);

    ::prepare(mask1, mask2, from1, from2, intersect, weight1, weight2);

    success = 1;
    return true;
}

void RicohPanoramaRender::render(const cv::Mat& src, cv::Mat& dst)
{
    if (!success)
        return;

    CV_Assert(src.data && src.type() == CV_8UC3 && src.size() == srcSize);

    reprojectAndBlendParallel(src, src, dstSrcMap1, dstSrcMap2,
        from1, from2, intersect, weight1, weight2, dst);
}

bool DetuPanoramaRender::prepare(const std::string& path_, const cv::Size& srcSize_, const cv::Size& dstSize_)
{
    success = 0;

    if (!((dstSize_.width & 1) == 0 && (dstSize_.height & 1) == 0 &&
        dstSize_.height * 2 == dstSize_.width))
        return false;

    srcSize = srcSize_;
    dstSize = dstSize_;

    PhotoParam param;
    loadPhotoParamFromXML(path_, param);

    cv::Mat mask;
    getReprojectMapAndMask(param, srcSize, dstSize, dstSrcMap, mask);

    success = 1;
    return true;
}

void DetuPanoramaRender::render(const cv::Mat& src, cv::Mat& dst)
{
    if (!success)
        return;

    CV_Assert(src.size() == srcSize);
    
    dst.create(dstSrcMap.size(), CV_8UC3);
    
    reprojectParallel(src, dst, dstSrcMap);
}

bool DualGoProPanoramaRender::prepare(const std::string& path_, int blendType_, const cv::Size& srcSize_, const cv::Size& dstSize_)
{
    success = 0;

    if (!((dstSize_.width & 1) == 0 && (dstSize_.height & 1) == 0 &&
        dstSize_.height * 2 == dstSize_.width))
        return false;

    srcSize = srcSize_;
    dstSize = dstSize_;

    std::vector<PhotoParam> params;
    loadPhotoParamFromPTS(path_, params);
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

bool DualGoProPanoramaRender::render(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst)
{
    if (!success)
        return false;

    if (!(src1.data && src2.data && src1.type() == CV_8UC3 && src2.type() == CV_8UC3 &&
        src1.size() == srcSize && src2.size() == srcSize))
        return false;

    reprojectAndBlendParallel(src1, src2, dstSrcMap1, dstSrcMap2,
        from1, from2, intersect, weight1, weight2, dst);
    return true;
}

bool DualGoProPanoramaRender::render(const std::vector<cv::Mat>& src, cv::Mat& dst)
{
    if (src.size() != 2)
        return false;

    return render(src[0], src[1], dst);
}

bool CPUMultiCameraPanoramaRender::prepare(const std::string& path_, int blendType_, const cv::Size& srcSize_, const cv::Size& dstSize_)
{
    success = 0;

    if (!((dstSize_.width & 1) == 0 && (dstSize_.height & 1) == 0 &&
        dstSize_.height * 2 == dstSize_.width))
        return false;

    srcSize = srcSize_;
    dstSize = dstSize_;

    std::vector<PhotoParam> params;
    if (!loadPhotoParams(path_, params) || params.empty())
        return false;

    numImages = params.size();
    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, masks);
    if (!blender.prepare(masks, 20, 2))
        return false;

    success = 1;
    return true;
}

bool CPUMultiCameraPanoramaRender::render(const std::vector<cv::Mat>& src, cv::Mat& dst)
{
    if (!success)
        return false;

    if (src.size() != numImages)
        return false;

    for (int i = 0; i < numImages; i++)
    {
        if (src[i].size() != srcSize)
            return false;
    }

    reprojImages.resize(numImages);
    for (int i = 0; i < numImages; i++)
        reprojectParallel(src[i], reprojImages[i], dstSrcMaps[i]);

    blender.blend(reprojImages, dst);
    return true;
}

bool CudaMultiCameraPanoramaRender::prepare(const std::string& path_, int blendType_, const cv::Size& srcSize_, const cv::Size& dstSize_)
{
    success = 0;

    if (!((dstSize_.width & 1) == 0 && (dstSize_.height & 1) == 0 &&
        dstSize_.height * 2 == dstSize_.width))
        return false;

    srcSize = srcSize_;
    dstSize = dstSize_;

    std::vector<PhotoParam> params;
    if (!loadPhotoParams(path_, params) || params.empty())
        return false;

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
        srcMems[i].create(srcSize, CV_8UC4);
        srcImages[i] = srcMems[i].createMatHeader();
    }

    streams.resize(numImages);

    success = 1;
    return true;
}

bool CudaMultiCameraPanoramaRender::render(const std::vector<cv::Mat>& src, cv::Mat& dst)
{
    if (!success)
        return false;

    if (src.size() != numImages)
        return false;

    for (int i = 0; i < numImages; i++)
    {
        if (src[i].size() != srcSize)
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
        srcImagesGPU[i].upload(srcImages[i], streams[i]);    
    for (int i = 0; i < numImages; i++)
        cudaReprojectTo16S(srcImagesGPU[i], reprojImagesGPU[i], dstSrcXMapsGPU[i], dstSrcYMapsGPU[i], streams[i]);
    for (int i = 0; i < numImages; i++)
        streams[i].waitForCompletion();

    blender.blend(reprojImagesGPU, blendImageGPU);
    blendImageGPU.download(dst);
    return true;
}

bool CudaMultiCameraPanoramaRender2::prepare(const std::string& path_, int type_, const cv::Size& srcSize_, const cv::Size& dstSize_)
{
    success = 0;

    if (type_ != BlendTypeLinear && type_ != BlendTypeMultiband)
        return false;

    if (!((dstSize_.width & 1) == 0 && (dstSize_.height & 1) == 0 &&
        dstSize_.height * 2 == dstSize_.width))
        return false;

    blendType = type_;
    srcSize = srcSize_;
    dstSize = dstSize_;

    std::vector<PhotoParam> params;
    if (!loadPhotoParams(path_, params) || params.empty())
        return false;

    numImages = params.size();
    std::vector<cv::Mat> masks, dstSrcMaps;
    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, masks);
    if (blendType == BlendTypeLinear)
    {
        if (!lBlender.prepare(masks, 50))
            return false;
    }
    else if (blendType == BlendTypeMultiband)
    {
        if (!mbBlender.prepare(masks, 20, 2))
            return false;
    }
    else
        return false;

    blendImage = cv::cuda::HostMem(dstSize, CV_8UC4, cv::cuda::HostMem::SHARED);
    blendImageGPU = blendImage.createGpuMatHeader();

    cudaGenerateReprojectMaps(params, srcSize, dstSize, dstSrcXMapsGPU, dstSrcYMapsGPU);
    streams.resize(numImages);

    success = 1;
    return true;
}

bool CudaMultiCameraPanoramaRender2::render(const std::vector<cv::Mat>& src, cv::Mat& dst)
{
    if (!success)
        return false;

    if (src.size() != numImages)
        return false;

    for (int i = 0; i < numImages; i++)
    {
        if (src[i].size() != srcSize)
            return false;
        if (src[i].type() != CV_8UC4)
            return false;
    }

    if (blendType == BlendTypeLinear)
    {
        srcImagesGPU.resize(numImages);
        reprojImagesGPU.resize(numImages);
        for (int i = 0; i < numImages; i++)
            srcImagesGPU[i].upload(src[i], streams[i]);
        for (int i = 0; i < numImages; i++)
            cudaReproject(srcImagesGPU[i], reprojImagesGPU[i], dstSrcXMapsGPU[i], dstSrcYMapsGPU[i], streams[i]);
        for (int i = 0; i < numImages; i++)
            streams[i].waitForCompletion();
        lBlender.blend(reprojImagesGPU, blendImageGPU);
        //blendImageGPU.download(dst);
        cv::Mat temp = blendImage.createMatHeader();
        temp.copyTo(dst);
    }
    else if (blendType == BlendTypeMultiband)
    {
        srcImagesGPU.resize(numImages);
        reprojImagesGPU.resize(numImages);
        for (int i = 0; i < numImages; i++)
            srcImagesGPU[i].upload(src[i], streams[i]);
        for (int i = 0; i < numImages; i++)
            cudaReprojectTo16S(srcImagesGPU[i], reprojImagesGPU[i], dstSrcXMapsGPU[i], dstSrcYMapsGPU[i], streams[i]);
        for (int i = 0; i < numImages; i++)
            streams[i].waitForCompletion();
        mbBlender.blend(reprojImagesGPU, blendImageGPU);
        //blendImageGPU.download(dst);
        cv::Mat temp = blendImage.createMatHeader();
        temp.copyTo(dst);
    }
    
    return true;
}

bool CudaMultiCameraPanoramaRender3::prepare(const std::string& path_, int type_, const cv::Size& srcSize_, const cv::Size& dstSize_)
{
    success = 0;

    if (!((dstSize_.width & 1) == 0 && (dstSize_.height & 1) == 0 &&
        dstSize_.height * 2 == dstSize_.width))
        return false;

    srcSize = srcSize_;
    dstSize = dstSize_;

    std::vector<PhotoParam> params;
    if (!loadPhotoParams(path_, params) || params.empty())
        return false;

    numImages = params.size();
    std::vector<cv::Mat> masks, dstSrcMaps;
    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, masks);

    if (!adjuster.prepare(masks, 50))
        return false;

    if (!blender.prepare(masks, 50))
        return false;

    cudaGenerateReprojectMaps(params, srcSize, dstSize, dstSrcXMapsGPU, dstSrcYMapsGPU);
    streams.resize(numImages);

    success = 1;
    return true;
}

bool CudaMultiCameraPanoramaRender3::render(const std::vector<cv::Mat>& src, cv::Mat& dst)
{
    if (!success)
        return false;

    if (src.size() != numImages)
        return false;

    for (int i = 0; i < numImages; i++)
    {
        if (src[i].size() != srcSize)
            return false;
        if (src[i].type() != CV_8UC4)
            return false;
    }

    srcImagesGPU.resize(numImages);
    reprojImagesGPU.resize(numImages);
    for (int i = 0; i < numImages; i++)
        srcImagesGPU[i].upload(src[i], streams[i]);
    for (int i = 0; i < numImages; i++)
        cudaReproject(srcImagesGPU[i], reprojImagesGPU[i], dstSrcXMapsGPU[i], dstSrcYMapsGPU[i], streams[i]);
    for (int i = 0; i < numImages; i++)
        streams[i].waitForCompletion();
    if (luts.empty())
    {
        cv::Mat imageC4;
        std::vector<cv::Mat> imagesC3(numImages);
        int fromTo[] = { 0, 0, 1, 1, 2, 2 };
        for (int i = 0; i < numImages; i++)
        {
            reprojImagesGPU[i].download(imageC4);
            imagesC3[i].create(reprojImagesGPU[i].size(), CV_8UC3);
            cv::mixChannels(&imageC4, 1, &imagesC3[i], 1, fromTo, 3);
        }
        adjuster.calcGain(imagesC3, luts);
    }
    for (int i = 0; i < numImages; i++)
        cudaTransform(reprojImagesGPU[i], reprojImagesGPU[i], luts[i]);
    blender.blend(reprojImagesGPU, blendImageGPU);
    blendImageGPU.download(dst);

    return true;
}

void getWeightsLinearBlend32F(const std::vector<cv::Mat>& masks, int radius, std::vector<cv::Mat>& weights);

bool CudaPanoramaRender::prepare(const std::string& path_, const std::string& customMaskPath_, int highQualityBlend_, int completeQueue_, 
    const cv::Size& srcSize_, const cv::Size& dstSize_)
{
    clear();

    success = 0;

    if (!((dstSize_.width & 1) == 0 && (dstSize_.height & 1) == 0 &&
        dstSize_.height * 2 == dstSize_.width))
    {
        ptlprintf("Error in %s, dstSize not qualified\n", __FUNCTION__);
        return false;
    }

    std::vector<PhotoParam> params;
    bool ok = loadPhotoParams(path_, params);
    if (!ok || params.empty())
    {
        ptlprintf("Error in %s, load photo params failed\n", __FUNCTION__);
        return false;
    }

    highQualityBlend = highQualityBlend_;
    completeQueue = completeQueue_;
    srcSize = srcSize_;
    dstSize = dstSize_;

    std::vector<cv::Mat> masks, dstSrcMaps;
    try
    {
        numImages = params.size();        
        getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, masks);
        if (highQualityBlend)
        {
            if (!mbBlender.prepare(masks, 20, 2))
                return false;
        }
        else
        {
            std::vector<cv::Mat> weights;
            getWeightsLinearBlend32F(masks, 50, weights);
            weightsGPU.resize(numImages);
            for (int i = 0; i < numImages; i++)
                weightsGPU[i].upload(weights[i]);
            accumGPU.create(dstSize, CV_32FC4);
        }

        cudaGenerateReprojectMaps(params, srcSize, dstSize, dstSrcXMapsGPU, dstSrcYMapsGPU);
        streams.resize(numImages);

        pool.init(dstSize.height, dstSize.width, CV_8UC4, cv::cuda::HostMem::SHARED);

        if (completeQueue)
            cpQueue.setMaxSize(4);
    }
    catch (std::exception& e)
    {
        ptlprintf("Error in %s, exception caught: %s\n", __FUNCTION__, e.what());
        return false;
    }

    useCustomMasks = 0;
    if (customMaskPath_.size())
    {
        if (highQualityBlend)
        {
            std::vector<std::vector<IntervaledContour> > contours;
            ok = loadIntervaledContours(customMaskPath_, contours);
            if (!ok)
            {
                ptlprintf("Error in %s, load custom masks failed\n", __FUNCTION__);
                return false;
            }
            if (contours.size() != numImages)
            {
                ptlprintf("Error in %s, loaded contours.size() != numVideos\n", __FUNCTION__);
                return false;
            }
            if (!cvtContoursToCudaMasks(contours, masks, customMasks))
            {
                ptlprintf("Error in %s, convert contours to customMasks failed\n", __FUNCTION__);
                return false;
            }
            mbBlender.getUniqueMasks(dstUniqueMasksGPU);
            useCustomMasks = 1;
        }
        else
            ptlprintf("Warning in %s, non high quality blend, i.e. linear blend, does not support custom masks\n", __FUNCTION__);
    }

    success = 1;
    return true;
}

bool CudaPanoramaRender::render(const std::vector<cv::Mat>& src, const std::vector<long long int> timeStamps)
{
    if (!success)
    {
        ptlprintf("Error in %s, have not prepared or prepare failed before\n", __FUNCTION__);
        return false;
    }

    if (src.size() != numImages || timeStamps.size() != numImages)
    {
        ptlprintf("Error in %s, size not equal\n", __FUNCTION__);
        return false;
    }

    for (int i = 0; i < numImages; i++)
    {
        if (src[i].size() != srcSize)
        {
            ptlprintf("Error in %s, src[%d] size (%d, %d), not equal to (%d, %d)\n",
                __FUNCTION__, i, src[i].size().width, src[i].size().height, 
                srcSize.width, srcSize.height);
            return false;
        }
            
        if (src[i].type() != CV_8UC4)
        {
            ptlprintf("Error in %s, type %d not equal to %d\n", __FUNCTION__, src[i].type(), CV_8UC4);
            return false;
        }
            
    }

    try
    {
        cv::cuda::HostMem blendImageMem;
        if (!pool.get(blendImageMem))
            return false;

        cv::cuda::GpuMat blendImageGPU = blendImageMem.createGpuMatHeader();
        if (!highQualityBlend)
        {
            accumGPU.setTo(0);
            srcImagesGPU.resize(numImages);
            reprojImagesGPU.resize(numImages);
            for (int i = 0; i < numImages; i++)
                srcImagesGPU[i].upload(src[i], streams[i]);
            for (int i = 0; i < numImages; i++)
                cudaReprojectWeightedAccumulateTo32F(srcImagesGPU[i], accumGPU, dstSrcXMapsGPU[i], dstSrcYMapsGPU[i], weightsGPU[i], streams[i]);
            for (int i = 0; i < numImages; i++)
                streams[i].waitForCompletion();
            accumGPU.convertTo(blendImageGPU, CV_8U);
        }
        else
        {
            srcImagesGPU.resize(numImages);
            reprojImagesGPU.resize(numImages);
            // Add the following two lines to prevent exception if dstSize is around (1200, 600)
            //for (int i = 0; i < numImages; i++)
            //    reprojImagesGPU[i].create(dstSize, CV_16SC4);
            // Further test shows that the above two lines cannot prevent cuda runtime
            // from throwing exception, so they are commented.
            // It seems that the only way to avoid exception is to call 
            // cudaReproject instead of cudaReprojectTo16S, but then CudaTilingMultibandBlend::blend
            // will perform data conversion from type CV_8UC4 to CV_16SC4, more time consumed.
            for (int i = 0; i < numImages; i++)
                srcImagesGPU[i].upload(src[i], streams[i]);
            for (int i = 0; i < numImages; i++)
                cudaReprojectTo16S(srcImagesGPU[i], reprojImagesGPU[i], dstSrcXMapsGPU[i], dstSrcYMapsGPU[i], streams[i]);
            for (int i = 0; i < numImages; i++)
                streams[i].waitForCompletion();

            if (useCustomMasks)
            {
                bool custom = false;
                currMasksGPU.resize(numImages);
                for (int i = 0; i < numImages; i++)
                {
                    if (customMasks[i].getMask(timeStamps[i], currMasksGPU[i]))
                        custom = true;
                    else
                        currMasksGPU[i] = dstUniqueMasksGPU[i];
                }

                if (custom)
                {
                    printf("custom masks\n");
                    mbBlender.blend(reprojImagesGPU, currMasksGPU, blendImageGPU);
                }                    
                else
                    mbBlender.blend(reprojImagesGPU, blendImageGPU);
            }
            else
                mbBlender.blend(reprojImagesGPU, blendImageGPU);
        }
        if (completeQueue)
            cpQueue.push(std::make_pair(blendImageMem, timeStamps[0]));
        else
            rtQueue.push(std::make_pair(blendImageMem, timeStamps[0]));
    }
    catch (std::exception& e)
    {
        ptlprintf("Error in %s, exception caught: %s\n", __FUNCTION__, e.what());
        return false;
    }

    return true;
}

bool CudaPanoramaRender::getResult(cv::Mat& dst, long long int& timeStamp)
{
    std::pair<cv::cuda::HostMem, long long int> item;
    bool ret = completeQueue ? cpQueue.pull(item) : rtQueue.pull(item);
    if (ret)
    {
        cv::Mat temp = item.first.createMatHeader();
        temp.copyTo(dst);
        timeStamp = item.second;
    }        
    return ret;
}

void CudaPanoramaRender::stop()
{
    rtQueue.stop();
    cpQueue.stop();
}

void CudaPanoramaRender::resume()
{
    rtQueue.resume();
    cpQueue.resume();
}

void CudaPanoramaRender::waitForCompletion()
{
    if (completeQueue)
    {
        while (cpQueue.size())
            std::this_thread::sleep_for(std::chrono::microseconds(25));
    }
    else
    {
        while (rtQueue.size())
            std::this_thread::sleep_for(std::chrono::microseconds(25));
    }
}

void CudaPanoramaRender::clear()
{
    dstUniqueMasksGPU.clear();
    currMasksGPU.clear();
    useCustomMasks = 0;
    customMasks.clear();
    dstSrcXMapsGPU.clear();
    dstSrcYMapsGPU.clear();
    srcImagesGPU.clear();
    reprojImagesGPU.clear();
    weightsGPU.clear();
    rtQueue.stop();
    cpQueue.stop();
    pool.clear();
    rtQueue.clear();
    cpQueue.clear();
    streams.clear();
    highQualityBlend = 0;
    completeQueue = 0;
    numImages = 0;
    success = 0;
}

int CudaPanoramaRender::getNumImages() const
{
    return success ? numImages : 0;
}

bool CPUPanoramaRender::prepare(const std::string& path_, int highQualityBlend_, int completeQueue_,
    const cv::Size& srcSize_, const cv::Size& dstSize_)
{
    clear();

    success = 0;

    if (!((dstSize_.width & 1) == 0 && (dstSize_.height & 1) == 0 &&
        dstSize_.height * 2 == dstSize_.width))
    {
        ptlprintf("Error in %s, dstSize not qualified\n", __FUNCTION__);
        return false;
    }

    std::vector<PhotoParam> params;
    bool ok = loadPhotoParams(path_, params);
    if (!ok || params.empty())
    {
        ptlprintf("Error in %s, load photo params failed\n", __FUNCTION__);
        return false;
    }

    highQualityBlend = highQualityBlend_;
    completeQueue = completeQueue_;
    srcSize = srcSize_;
    dstSize = dstSize_;

    try
    {
        numImages = params.size();
        std::vector<cv::Mat> masks;
        getReprojectMapsAndMasks(params, srcSize, dstSize, maps, masks);
        if (highQualityBlend)
        {
            if (!mbBlender.prepare(masks, 20, 2))
                return false;
        }
        else
        {
            getWeightsLinearBlend32F(masks, 50, weights);
            accum.create(dstSize, CV_32FC3);
        }

        pool.init(dstSize.height, dstSize.width, CV_8UC3);

        if (completeQueue)
            cpQueue.setMaxSize(4);
    }
    catch (std::exception& e)
    {
        ptlprintf("Error in %s, exception caught: %s\n", __FUNCTION__, e.what());
        return false;
    }

    success = 1;
    return true;
}

bool CPUPanoramaRender::render(const std::vector<cv::Mat>& src, long long int timeStamp)
{
    if (!success)
    {
        ptlprintf("Error in %s, have not prepared or prepare failed before\n", __FUNCTION__);
        return false;
    }

    if (src.size() != numImages)
    {
        ptlprintf("Error in %s, size not equal\n", __FUNCTION__);
        return false;
    }

    for (int i = 0; i < numImages; i++)
    {
        if (src[i].size() != srcSize)
        {
            ptlprintf("Error in %s, src[%d] size (%d, %d), not equal to (%d, %d)\n",
                __FUNCTION__, i, src[i].size().width, src[i].size().height,
                srcSize.width, srcSize.height);
            return false;
        }

        if (src[i].type() != CV_8UC3)
        {
            ptlprintf("Error in %s, type %d not equal to %d\n", __FUNCTION__, src[i].type(), CV_8UC3);
            return false;
        }

    }

    try
    {
        cv::Mat blendImage;
        if (!pool.get(blendImage))
            return false;

        if (!highQualityBlend)
        {
            accum.setTo(0);
            for (int i = 0; i < numImages; i++)
                reprojectWeightedAccumulateParallelTo32F(src[i], accum, maps[i], weights[i]);
            accum.convertTo(blendImage, CV_8U);
        }
        else
        {
            reprojImages.resize(numImages);
            for (int i = 0; i < numImages; i++)
                reprojectParallelTo16S(src[i], reprojImages[i], maps[i]);
            mbBlender.blend(reprojImages, blendImage);
        }
        if (completeQueue)
            cpQueue.push(std::make_pair(blendImage, timeStamp));
        else
            rtQueue.push(std::make_pair(blendImage, timeStamp));
    }
    catch (std::exception& e)
    {
        ptlprintf("Error in %s, exception caught: %s\n", __FUNCTION__, e.what());
        return false;
    }

    return true;
}

bool CPUPanoramaRender::getResult(cv::Mat& dst, long long int& timeStamp)
{
    std::pair<cv::Mat, long long int> item;
    bool ret = completeQueue ? cpQueue.pull(item) : rtQueue.pull(item);
    if (ret)
    {
        item.first.copyTo(dst);
        timeStamp = item.second;
    }
    return ret;
}

void CPUPanoramaRender::stop()
{
    rtQueue.stop();
    cpQueue.stop();
}

void CPUPanoramaRender::resume()
{
    rtQueue.resume();
    cpQueue.resume();
}

void CPUPanoramaRender::waitForCompletion()
{
    if (completeQueue)
    {
        while (cpQueue.size())
            std::this_thread::sleep_for(std::chrono::microseconds(25));
    }
    else
    {
        while (rtQueue.size())
            std::this_thread::sleep_for(std::chrono::microseconds(25));
    }
}

void CPUPanoramaRender::clear()
{
    maps.clear();
    reprojImages.clear();
    weights.clear();
    rtQueue.stop();
    cpQueue.stop();
    pool.clear();
    rtQueue.clear();
    cpQueue.clear();
}

int CPUPanoramaRender::getNumImages() const
{
    return success ? numImages : 0;
}

#include "CompileControl.h"

#if COMPILE_INTEL_OPENCL
#include "IntelOpenCLInterface.h"

bool IOclPanoramaRender::prepare(const std::string& path_, int highQualityBlend_, int completeQueue_,
    const cv::Size& srcSize_, const cv::Size& dstSize_, OpenCLBasic* ocl_)
{
    clear();

    success = 0;

    if (!((dstSize_.width & 1) == 0 && (dstSize_.height & 1) == 0 &&
        dstSize_.height * 2 == dstSize_.width))
    {
        ptlprintf("Error in %s, dstSize not qualified\n", __FUNCTION__);
        return false;
    }

    std::vector<PhotoParam> params;
    bool ok = loadPhotoParams(path_, params);
    if (!ok || params.empty())
    {
        ptlprintf("Error in %s, load photo params failed\n", __FUNCTION__);
        return false;
    }

    highQualityBlend = highQualityBlend_;
    completeQueue = completeQueue_;
    srcSize = srcSize_;
    dstSize = dstSize_;
    ocl = ocl_;

    try
    {
        numImages = params.size();
        std::vector<cv::Mat> masks, xmaps32F, ymaps32F;
        getReprojectMaps32FAndMasks(params, srcSize, dstSize, xmaps32F, ymaps32F, masks);
        xmaps.resize(numImages);
        ymaps.resize(numImages);
        cv::Mat map32F;
        for (int i = 0; i < numImages; i++)
        {
            xmaps[i].create(dstSize, CV_32FC1, ocl->context);
            ymaps[i].create(dstSize, CV_32FC1, ocl->context);
            cv::Mat headx = xmaps[i].toOpenCVMat();
            cv::Mat heady = ymaps[i].toOpenCVMat();
            xmaps32F[i].copyTo(headx);
            ymaps32F[i].copyTo(heady);
        }
        //if (highQualityBlend)
        //{
        //}
        //else
        {
            std::vector<cv::Mat> ws;
            getWeightsLinearBlend32F(masks, 50, ws);
            weights.resize(numImages);
            for (int i = 0; i < numImages; i++)
            {
                weights[i].create(dstSize, CV_32FC1, ocl->context);
                cv::Mat header = weights[i].toOpenCVMat();
                ws[i].copyTo(header);
            }
        }

        pool.init(dstSize.height, dstSize.width, CV_32FC4, ocl->context);

        setZeroKern.reset(new OpenCLProgramOneKernel(*ocl, L"MatOp.txt", "", "setZeroKernel"));
        rprjKern.reset(new OpenCLProgramOneKernel(*ocl, L"Reproject.txt", "", "reprojectWeighedAccumulateTo32FKernel"));

        if (completeQueue)
            cpQueue.setMaxSize(4);
    }
    catch (std::exception& e)
    {
        ptlprintf("Error in %s, exception caught: %s\n", __FUNCTION__, e.what());
        return false;
    }

    success = 1;
    return true;
}

bool IOclPanoramaRender::render(const std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps)
{
    ztool::Timer t, tt;;
    if (!success)
    {
        ptlprintf("Error in %s, have not prepared or prepare failed before\n", __FUNCTION__);
        return false;
    }

    if (src.size() != numImages)
    {
        ptlprintf("Error in %s, size not equal\n", __FUNCTION__);
        return false;
    }

    for (int i = 0; i < numImages; i++)
    {
        if (src[i].size() != srcSize)
        {
            ptlprintf("Error in %s, src[%d] size (%d, %d), not equal to (%d, %d)\n",
                __FUNCTION__, i, src[i].size().width, src[i].size().height, 
                srcSize.width, srcSize.height);
            return false;
        }

        if (src[i].type() != CV_8UC4)
        {
            ptlprintf("Error in %s, type %d not equal to %d\n", __FUNCTION__, src[i].type(), CV_8UC4);
            return false;
        }

    }

    try
    {
        IOclMat blendImage;
        if (!pool.get(blendImage))
            return false;

        //if (!highQualityBlend)
        {
            tt.start();
            ioclSetZero(blendImage, *ocl, *setZeroKern);
            for (int i = 0; i < numImages; i++)
            {
                IOclMat temp(src[i].size(), CV_8UC4, src[i].data, src[i].step, ocl->context);
                ioclReprojectAccumulateWeightedTo32F(temp, blendImage, xmaps[i], ymaps[i], weights[i], *ocl, *rprjKern);
            }
            tt.end();
        }
        //else
        //{
        //}
        if (completeQueue)
            cpQueue.push(std::make_pair(blendImage, timeStamps[0]));
        else
            rtQueue.push(std::make_pair(blendImage, timeStamps[0]));

        t.end();
        //ptlprintf("t = %f, tt = %f\n", t.elapse(), tt.elapse());
    }
    catch (std::exception& e)
    {
        ptlprintf("Error in %s, exception caught: %s\n", __FUNCTION__, e.what());
        return false;
    }
    return true;
}

bool IOclPanoramaRender::getResult(cv::Mat& dst, long long int& timeStamp)
{
    std::pair<IOclMat, long long int> item;
    bool ret = completeQueue ? cpQueue.pull(item) : rtQueue.pull(item);
    if (ret)
    {
        cv::Mat header = item.first.toOpenCVMat();
        header.convertTo(dst, CV_8U);
        timeStamp = item.second;
    }
    return ret;
}

void IOclPanoramaRender::stop()
{
    rtQueue.stop();
    cpQueue.stop();
}

void IOclPanoramaRender::resume()
{
    rtQueue.resume();
    cpQueue.resume();
}

void IOclPanoramaRender::waitForCompletion()
{
    if (completeQueue)
    {
        while (cpQueue.size())
            std::this_thread::sleep_for(std::chrono::microseconds(25));
    }
    else
    {
        while (rtQueue.size())
            std::this_thread::sleep_for(std::chrono::microseconds(25));
    }
}

void IOclPanoramaRender::clear()
{
    xmaps.clear();
    ymaps.clear();
    weights.clear();
    rtQueue.stop();
    cpQueue.stop();
    pool.clear();
    rtQueue.clear();
    cpQueue.clear();
}

int IOclPanoramaRender::getNumImages() const
{
    return success ? numImages : 0;
}

#endif