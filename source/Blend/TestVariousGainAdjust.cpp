#include "ZBlend.h"
#include "ZBlendAlgo.h"
#include "ZReproject.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <thread>
#include <memory>
#include <algorithm>

//static void getExtendedMasks(const std::vector<cv::Mat>& masks, int radius, std::vector<cv::Mat>& extendedMasks)
//{
//    int numImages = masks.size();
//    std::vector<cv::Mat> uniqueMasks(numImages);
//    getNonIntersectingMasks(masks, uniqueMasks);
//
//    std::vector<cv::Mat> compMasks(numImages);
//    for (int i = 0; i < numImages; i++)
//        cv::bitwise_not(masks[i], compMasks[i]);
//
//    std::vector<cv::Mat> blurMasks(numImages);
//    cv::Mat intersect;
//    int validCount, r;
//    for (r = radius; r > 0; r = r - 2)
//    {
//        cv::Size blurSize(r * 2 + 1, r * 2 + 1);
//        double sigma = r / 3.0;
//        validCount = 0;
//        for (int i = 0; i < numImages; i++)
//        {
//            cv::GaussianBlur(uniqueMasks[i], blurMasks[i], blurSize, sigma, sigma);
//            cv::bitwise_and(blurMasks[i], compMasks[i], intersect);
//            if (cv::countNonZero(intersect) == 0)
//                validCount++;
//        }
//        if (validCount == numImages)
//            break;
//    }
//
//    extendedMasks.resize(numImages);
//    for (int i = 0; i < numImages; i++)
//        extendedMasks[i] = (blurMasks[i] != 0);
//}

static void calcHist(const cv::Mat& image, const cv::Mat& mask, std::vector<int>& hist)
{
    CV_Assert(image.data && image.type() == CV_8UC1 &&
        mask.data && mask.type() == CV_8UC1 && image.size() == mask.size());

    hist.resize(256, 0);
    int rows = image.rows, cols = image.cols;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptr = image.ptr<unsigned char>(i);
        const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrMask[j])
                hist[ptr[j]] += 1;
        }
    }
}

static void calcAccumHist(const cv::Mat& image, const cv::Mat& mask, std::vector<double>& hist)
{
    CV_Assert(image.data && image.type() == CV_8UC1 &&
        mask.data && mask.type() == CV_8UC1 && image.size() == mask.size());

    hist.resize(256, 0);
    std::vector<int> tempHist(256, 0);
    int rows = image.rows, cols = image.cols;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptr = image.ptr<unsigned char>(i);
        const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrMask[j])
                tempHist[ptr[j]] += 1;
        }

    }
    for (int i = 1; i < 256; i++)
        tempHist[i] += tempHist[i - 1];
    double scale = 1.0 / tempHist[255];
    for (int i = 0; i < 256; i++)
        hist[i] = tempHist[i] * scale;
}

static void histSpecification(std::vector<double>& src, std::vector<double>& dst, std::vector<unsigned char>& lut)
{
    CV_Assert(src.size() == 256 && dst.size() == 256);

    lut.resize(256);
    for (int i = 0; i < 256; i++)
    {
        double val = src[i];
        double minDiff = fabs(val - dst[0]);
        int index = 0;
        for (int j = 1; j < 256; j++)
        {
            double currDiff = fabs(val - dst[j]);
            if (currDiff < minDiff)
            {
                index = j;
                minDiff = currDiff;
            }
        }
        lut[i] = index;
    }
}

static void calcMaxGrad(const cv::Mat& image, const cv::Mat& mask, std::vector<int>& maxGrad)
{
    CV_Assert(image.data && image.type() == CV_8UC1 &&
        mask.data && mask.type() == CV_8UC1 && image.size() == mask.size());

    maxGrad.resize(256, 0);
    int rows = image.rows, cols = image.cols;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptr = image.ptr<unsigned char>(i);
        const unsigned char* ptrTop = image.ptr<unsigned char>(i > 0 ? i - 1 : 0);
        const unsigned char* ptrBot = image.ptr<unsigned char>(i < rows - 1 ? i + 1 : rows - 1);
        const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrMask[j])
            {
                int currVal = ptr[j];
                int topVal = ptrTop[j];
                int botVal = ptrBot[j];
                int leftVal = ptr[j > 0 ? j - 1 : 0];
                int rightVal = ptr[j < cols - 1 ? j + 1 : cols - 1];
                int diff = abs(currVal - topVal);
                diff = std::max(diff, abs(currVal - botVal));
                diff = std::max(diff, abs(currVal - leftVal));
                diff = std::max(diff, abs(currVal - rightVal));
                maxGrad[currVal] = std::max(diff, maxGrad[currVal]);
            }
        }
    }
}

inline int clamp0255(int val)
{
    return val < 0 ? 0 : (val > 255 ? 255 : val);
}

static void lineParamToLUT(double k, double b, std::vector<unsigned char>& lut)
{
    lut.resize(256);
    for (int i = 0; i < 256; i++)
        lut[i] = clamp0255(k * i + b);
}

static void showHistogram(const std::string& winName, const std::vector<double>& lut)
{
    CV_Assert(lut.size() == 256);

    cv::Mat image = cv::Mat::zeros(256, 256, CV_8UC1);
    for (int i = 0; i < 255; i++)
    {
        cv::line(image, cv::Point(i, 255 - lut[i] * 255), cv::Point(i + 1, 255 - lut[i + 1] * 255), cv::Scalar(255));
    }
    cv::imshow(winName, image);
}

static void showLUT(const std::string& winName, const std::vector<unsigned char>& lut)
{
    cv::Mat image = cv::Mat::zeros(256, 256, CV_8UC1);
    for (int i = 0; i < 255; i++)
    {
        cv::line(image, cv::Point(i, 255 - lut[i]), cv::Point(i + 1, 255 - lut[i + 1]), cv::Scalar(255));
    }
    cv::imshow(winName, image);
}

static void calcHistSpecLUT(const cv::Mat& src, const cv::Mat& srcMask,
    const cv::Mat& dst, const cv::Mat& dstMask, std::vector<unsigned char>& lutSrcToDst)
{
    CV_Assert(src.data && src.type() == CV_8UC1 && srcMask.data && srcMask.type() == CV_8UC1 &&
        dst.data && dst.type() == CV_8UC1 && dstMask.data && dstMask.type() == CV_8UC1 &&
        src.size() == srcMask.size() && dst.size() == dstMask.size() && src.size() == dst.size());

    cv::Mat intersect = srcMask & dstMask;
    cv::imshow("intersect", intersect);
    cv::imshow("src i", src & intersect);
    cv::imshow("dst i", dst & intersect);
    cv::waitKey(0);
    cv::Mat blurSrc, blurDst;
    cv::blur(src, blurSrc, cv::Size(9, 9));
    cv::blur(dst, blurDst, cv::Size(9, 9));
    cv::Mat diffSmall = (((blurSrc - blurDst) < 40) & ((blurDst - blurSrc) < 40));
    cv::imshow("diff small", diffSmall);
    cv::waitKey(0);
    intersect &= diffSmall;
    std::vector<double> srcAccumHist, dstAccumHist;
    calcAccumHist(src, intersect, srcAccumHist);
    calcAccumHist(dst, intersect, dstAccumHist);
    std::vector<unsigned char> lut;
    histSpecification(srcAccumHist, dstAccumHist, lut);
    showHistogram("src accum hist", srcAccumHist);
    showHistogram("dst accum hist", dstAccumHist);
    showLUT("raw lut", lut);
    cv::waitKey(0);
    lutSrcToDst = lut;
    //std::vector<cv::Point2f> pts(256);
    //for (int i = 0; i < 256; i++)
    //{
    //    pts[i].x = i;
    //    pts[i].y = lut[i];
    //}
    //cv::Vec4f param;
    //cv::fitLine(pts, param, cv::DIST_L2, 0, 0.01, 0.01);
    //printf("fit line out: %f %f %f %f\n", param[0], param[1], param[2], param[3]);
    //double k = param[1] / param[0];
    //double b = param[3] - k * param[2];
    //printf("line param: %f, %f\n", k, b);
    //lineParamToLUT(k, b, lutSrcToDst);
}

int getLineRANSAC(const std::vector<cv::Point>& points, cv::Point2d& p, cv::Point2d& dir);
void calcHist2D(const std::vector<cv::Point>& points, cv::Mat& hist);
void normalizeAndConvert(const cv::Mat& hist, cv::Mat& image);

static void logNormalizeAndConvert(const cv::Mat& hist, cv::Mat& result)
{
    cv::Mat image(256, 256, CV_8UC1);
    double minVal, maxVal;
    cv::minMaxLoc(hist, &minVal, &maxVal);
    double scale = 255.0 / (log(1 + maxVal) - log(1 + minVal));
    double logMinVal = log(1 + minVal);
    for (int i = 0; i < 256; i++)
    {
        const int* ptrHist = hist.ptr<int>(i);
        unsigned char* ptrImage = image.ptr<unsigned char>(i);
        for (int j = 0; j < 256; j++)
        {
            ptrImage[j] = (log(1 + ptrHist[j]) - logMinVal) * scale + 0.5;
        }
    }
    cv::flip(image, result, 0);
}

static void calcLUT(const cv::Mat& src, const cv::Mat& srcMask,
    const cv::Mat& dst, const cv::Mat& dstMask, std::vector<unsigned char>& lutSrcToDst)
{
    CV_Assert(src.data && src.type() == CV_8UC1 && srcMask.data && srcMask.type() == CV_8UC1 &&
        dst.data && dst.type() == CV_8UC1 && dstMask.data && dstMask.type() == CV_8UC1 &&
        src.size() == srcMask.size() && dst.size() == dstMask.size() && src.size() == dst.size());

    cv::Mat intersect = srcMask & dstMask;
    //cv::imshow("intersect", intersect);
    //cv::imshow("src i", src & intersect);
    //cv::imshow("dst i", dst & intersect);
    //cv::waitKey(0);
    cv::Mat blurSrc, blurDst;
    cv::blur(src, blurSrc, cv::Size(9, 9));
    cv::blur(dst, blurDst, cv::Size(9, 9));
    cv::Mat diffSmall = (((blurSrc - blurDst) < 40) & ((blurDst - blurSrc) < 40));
    //cv::imshow("diff small", diffSmall);
    //cv::waitKey(0);
    intersect &= diffSmall;

    int rows = src.rows, cols = src.cols;
    std::vector<cv::Point> pts;
    pts.reserve(cv::countNonZero(intersect));
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrSrc = src.ptr<unsigned char>(i);
        const unsigned char* ptrDst = dst.ptr<unsigned char>(i);
        const unsigned char* ptrIts = intersect.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrIts[j])
                pts.push_back(cv::Point(ptrSrc[j], ptrDst[j]));
        }
    }
    cv::Point2d p, dir;
    getLineRANSAC(pts, p, dir);
    double k = dir.y / dir.x;
    double b = p.y - k * p.x;
    lineParamToLUT(k, b, lutSrcToDst);

    cv::Mat hist2D, histShow;
    calcHist2D(pts, hist2D);
    logNormalizeAndConvert(hist2D, histShow);
    cv::imshow("hist", histShow);
    cv::waitKey(0);
}

static void calcScale(const cv::Size& size, double minScale, cv::Mat& scale)
{
    double alpha = 4.0 * (1.0 - minScale) / (size.width * size.width + size.height * size.height);
    scale.create(size, CV_64FC1);
    int halfHeight = size.height / 2, halfWidth = size.width / 2;
    for (int i = 0; i < size.height; i++)
    {
        double* ptr = scale.ptr<double>(i);
        for (int j = 0; j < size.width; j++)
        {
            int sqrDiff = (i - halfHeight) * (i - halfHeight) + (j - halfWidth) * (j - halfWidth);
            ptr[j] = 1.0 / (1 - alpha * sqrDiff);
        }
    }
}

static void mulScale(cv::Mat& image, const cv::Mat& scale)
{
    CV_Assert(image.data && (image.type() == CV_8UC1 || image.type() == CV_8UC3) &&
        scale.data && scale.type() == CV_64FC1 && image.size() == scale.size());
    int rows = image.rows, cols = image.cols;
    if (image.type() == CV_8UC1)
    {
        for (int i = 0; i < rows; i++)
        {
            unsigned char* ptrImage = image.ptr<unsigned char>(i);
            const double* ptrScale = scale.ptr<double>(i);
            for (int j = 0; j < cols; j++)
                ptrImage[j] = clamp0255(ptrImage[j] * ptrScale[j] + 0.5);
        }
    }
    else
    {
        for (int i = 0; i < rows; i++)
        {
            unsigned char* ptrImage = image.ptr<unsigned char>(i);
            const double* ptrScale = scale.ptr<double>(i);
            for (int j = 0; j < cols; j++)
            {
                ptrImage[j * 3] = clamp0255(ptrImage[j * 3] * ptrScale[j] + 0.5);
                ptrImage[j * 3 + 1] = clamp0255(ptrImage[j * 3 + 1] * ptrScale[j] + 0.5);
                ptrImage[j * 3 + 2] = clamp0255(ptrImage[j * 3 + 2] * ptrScale[j] + 0.5);
            }
        }
    }
}

// main1
int main1()
{
    std::vector<std::string> imagePaths;
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-00.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-01.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-02.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-03.jpg");

    //std::vector<std::string> imagePaths;
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\0.jpg");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\1.jpg");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\2.jpg");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\3.jpg");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\4.jpg");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\5.jpg");

    int numImages = imagePaths.size();
    std::vector<cv::Mat> origImages(numImages), origReprojImages(numImages), images(numImages), grayImages(numImages);
    std::vector<cv::Mat> maps(numImages), origMasks(numImages), masks(numImages);

    for (int i = 0; i < numImages; i++)
        origImages[i] = cv::imread(imagePaths[i]);

    //cv::Mat scaleImage;
    //calcScale(origImages[0].size(), 0.4, scaleImage);
    //for (int k = 0; k < numImages; k++)
    //    mulScale(origImages[k], scaleImage);
    //mulScale(origImages[2], scaleImage);

    //for (int i = 0; i < numImages; i++)
    //{
    //    cv::imshow("orig image", origImages[i]);
    //    cv::waitKey(0);
    //}

    std::vector<PhotoParam> params;
    loadPhotoParamFromPTS("F:\\panoimage\\detuoffice\\4p.pts", params);
    //loadPhotoParamFromXML("F:\\panoimage\\zhanxiang\\zhanxiang.xml", params);
    getReprojectMapsAndMasks(params, origImages[0].size(), cv::Size(2048, 1024), maps, origMasks);
    reprojectParallel(origImages, origReprojImages, maps);

    //getExtendedMasks(origMasks, 100, masks);
    masks = origMasks;

    for (int i = 0; i < numImages; i++)
    {
        origReprojImages[i].copyTo(images[i]);
        cv::cvtColor(images[i], grayImages[i], CV_BGR2GRAY);
    }
        

    int maxGrayMeanIndex = 0;
    double maxGrayMean = cv::mean(grayImages[0], masks[0])[0];
    for (int i = 1; i < numImages; i++)
    {
        double currMean = cv::mean(grayImages[i], masks[i])[0];
        if (currMean > maxGrayMean)
        {
            maxGrayMeanIndex = i;
            maxGrayMean = currMean;
        }
    }
    printf("max gray index = %d\n", maxGrayMeanIndex);

    std::vector<std::vector<unsigned char> > luts(numImages);

    //calcHistSpecLUT(grayImages[2], masks[2], grayImages[1], masks[1], luts[0]);
    //calcHistSpecLUT(grayImages[2], masks[2], grayImages[3], masks[3], luts[1]);
    //showLUT("lut0", luts[0]);
    //showLUT("lut1", luts[1]);
    //cv::waitKey(0);
    //return 0;

    //maxGrayMeanIndex = 1;
    std::vector<int> workIndexes, adoptIndexes, remainIndexes;
    std::vector<cv::Mat> workImages;
    cv::Mat refImage, refGrayImage, refMask;
    for (int i = 0; i < numImages; i++)
    {
        if (i == maxGrayMeanIndex)
        {
            refImage = images[i].clone();
            refGrayImage = grayImages[i].clone();
            refMask = masks[i].clone();
            refImage.setTo(0, ~refMask);
            refGrayImage.setTo(0, ~refMask);
            luts[i].resize(256);
            for (int j = 0; j < 256; j++)
                luts[i][j] = j;
        }
        else
        {
            workIndexes.push_back(i);
            workImages.push_back(grayImages[i]);
        }
    }
    while (true)
    {
        adoptIndexes.clear();
        remainIndexes.clear();
        for (int i = 0; i < workIndexes.size(); i++)
        {
            printf("work index = %d\n", workIndexes[i]);
            if (cv::countNonZero(refMask & masks[workIndexes[i]]))
            {
                adoptIndexes.push_back(workIndexes[i]);
                calcHistSpecLUT(grayImages[workIndexes[i]], masks[workIndexes[i]], refGrayImage, refMask, luts[workIndexes[i]]);
                std::vector<unsigned char> localLUT;
                calcLUT(grayImages[workIndexes[i]], masks[workIndexes[i]], refGrayImage, refMask, localLUT);
                transform(images[workIndexes[i]], images[workIndexes[i]], luts[workIndexes[i]], masks[workIndexes[i]]);
                printf("lut index = %d\n", workIndexes[i]);
                for (int j = 0; j < 256; j++)
                {
                    printf("%3d ", luts[workIndexes[i]][j]);
                    if (j % 16 == 15)
                        printf("\n");
                }
                showLUT("lut", luts[workIndexes[i]]);
                std::vector<int> maxGrad;
                calcMaxGrad(grayImages[workIndexes[i]], masks[workIndexes[i]], maxGrad);
                for (int j = 0; j < 256; j++)
                {
                    printf("%3d ", maxGrad[j]);
                    if (j % 16 == 15)
                        printf("\n");
                }
                cv::imshow("transformed image", images[workIndexes[i]]);
                cv::waitKey(0);
                /*char buf[256];
                sprintf(buf, "adjust%d.bmp", workIndexes[i]);
                cv::imwrite(buf, images[workIndexes[i]]);*/
            }
            else
                remainIndexes.push_back(workIndexes[i]);
        }
        if (remainIndexes.empty())
            break;

        std::vector<cv::Mat> srcImages, srcMasks;
        srcImages.push_back(refImage);
        srcMasks.push_back(refMask);
        for (int i = 0; i < adoptIndexes.size(); i++)
        {
            srcImages.push_back(images[adoptIndexes[i]]);
            srcMasks.push_back(masks[adoptIndexes[i]]);
        }

        for (int i = 0; i < srcImages.size(); i++)
        {
            cv::imshow("src", srcImages[i]);
            cv::waitKey(0);
        }

        //TilingMultibandBlendFast blender;
        //blender.prepare(srcMasks, 20, 2);
        //blender.blend(srcImages, refImage);
        //for (int i = 0; i < adoptIndexes.size(); i++)
        //    refMask |= masks[adoptIndexes[i]];
        //refImage.setTo(0, ~refMask);
        for (int i = 0; i < adoptIndexes.size(); i++)
        {
            images[adoptIndexes[i]].copyTo(refImage, masks[adoptIndexes[i]]);
            refMask |= masks[adoptIndexes[i]];
        }
        cv::cvtColor(refImage, refGrayImage, CV_BGR2GRAY);
        cv::imshow("ref image", refImage);
        cv::imshow("ref mask", refMask);
        cv::waitKey(0);

        workIndexes = remainIndexes;
        workImages.clear();
        for (int i = 0; i < workIndexes.size(); i++)
            workImages.push_back(images[workIndexes[i]]);
    }

    for (int i = 0; i < numImages; i++)
        transform(origReprojImages[i], images[i], luts[i], masks[i]);

    TilingLinearBlend blender;
    blender.prepare(origMasks, 100);
    cv::Mat result;
    blender.blend(images, result);
    cv::imshow("result", result);
    cv::waitKey(0);

    return 0;
}

struct MaskIntersection
{
    int i, j;
    cv::Mat mask;
    int numNonZero;
};

void calcMaskIntersections(const std::vector<cv::Mat>& masks, std::vector<MaskIntersection>& intersects)
{
    intersects.clear();
    if (masks.empty())
        return;

    int size = masks.size();
    int rows = masks[0].rows, cols = masks[0].cols;
    for (int i = 0; i < size; i++)
    {
        CV_Assert(masks[i].data && masks[i].type() == CV_8UC1 &&
            masks[i].rows == rows && masks[i].cols == cols);
    }

    for (int i = 0; i < size - 1; i++)
    {
        for (int j = i + 1; j < size; j++)
        {
            cv::Mat mask = masks[i] & masks[j];
            int numNonZero = cv::countNonZero(mask);
            if (numNonZero)
            {
                intersects.push_back(MaskIntersection());
                MaskIntersection& intersect = intersects.back();
                intersect.i = i;
                intersect.j = j;
                intersect.mask = mask;
                intersect.numNonZero = numNonZero;
            }
            
        }
    }
}

double calcSqrDiff(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& mask)
{
    CV_Assert(image1.data && image1.type() == CV_8UC1 &&
        image2.data && image2.type() == CV_8UC1 && mask.data && mask.type() == CV_8UC1 &&
        image1.size() == mask.size() && image2.size() == mask.size());

    double val = 0;
    int rows = image1.rows, cols = image1.cols;
    int minDiff = -1, maxDiff = -1;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptr1 = image1.ptr<unsigned char>(i);
        const unsigned char* ptr2 = image2.ptr<unsigned char>(i);
        const unsigned char* ptrm = mask.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrm[j])
            {
                int diff = ptr1[j] - ptr2[j];
                val += diff * diff;
                if (diff < 0)
                    diff = -diff;
                if (minDiff < 0 || minDiff > diff)
                    minDiff = diff;
                if (maxDiff < 0 || maxDiff < diff)
                    maxDiff = diff;
            }
        }
    }
    printf("min diff = %d, maxDiff = %d\n");

    return val;
}

struct ScalesAndError
{
    ScalesAndError() : error(0), errorB(0), errorG(0), errorR(0) {}
    std::vector<double> scales;
    double error;
    double errorB, errorG, errorR;
};

void getScalesAndErrorVector(double minScale, double maxScale, int numSteps, int numItems, std::vector<ScalesAndError>& infos)
{
    /*CV_Assert(minScale >= 0 && minScale <= 1 && maxScale >= 0 &&
        maxScale <= 1 && minScale < maxScale && numSteps > 0 && numItems > 0);*/

    double scaleStep = (maxScale - minScale) / numSteps;

    numSteps += 1;
    std::vector<int> indexes(numItems);
    int numInfos = pow(numSteps, numItems);
    infos.resize(numInfos);
    for (int i = 0; i < numInfos; i++)
    {
        int val = i;
        for (int j = 0; j < numItems; j++)
        {
            indexes[j] = val % numSteps;
            val -= indexes[j];
            val /= numSteps;
        }

        //for (int j = 0; j < numItems; j++)
        //    printf("%d ", indexes[j]);
        //printf("\n");

        ScalesAndError& info = infos[i];
        info.scales.resize(numItems);
        for (int j = 0; j < numItems; j++)
            info.scales[j] = minScale + scaleStep * indexes[j];
    }
}

void getScales(const std::vector<double>& scales, const cv::Size& imageSize, std::vector<cv::Mat>& scaleImages)
{
    int numImages = scales.size();
    scaleImages.resize(numImages);
    for (int i = 0; i < numImages; i++)
        calcScale(imageSize, scales[i], scaleImages[i]);
}

void mulScales(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& scales, std::vector<cv::Mat>& dst)
{
    CV_Assert(src.size() == scales.size());

    int size = src.size();
    dst.resize(size);
    for (int i = 0; i < size; i++)
    {
        src[i].copyTo(dst[i]);
        mulScale(dst[i], scales[i]);
    }
}

double calcTotalError(const std::vector<cv::Mat>& images, const std::vector<MaskIntersection>& intersects)
{
    int numInts = intersects.size();
    double val = 0;
    for (int i = 0; i < numInts; i++)
    {
        const MaskIntersection& currInt = intersects[i];
        val += calcSqrDiff(images[currInt.i], images[currInt.j], currInt.mask);
    }
    return val;
}

void calcErrors(const std::vector<cv::Mat>& origImages, const std::vector<cv::Mat>& maps, 
    std::vector<MaskIntersection>& intersects, std::vector<ScalesAndError>& infos)
{
    int numInfos = infos.size();
    int numImages = origImages.size();
    std::vector<cv::Mat> reprojImages, grayImages, scaleImages, scaledImages;
    std::vector<std::vector<cv::Mat> > bgrImages;
    std::vector<cv::Mat> rImages, gImages, bImages;
    for (int i = 0; i < numInfos; i++)
    {
        ScalesAndError& info = infos[i];
        getScales(info.scales, origImages[0].size(), scaleImages);
        mulScales(origImages, scaleImages, scaledImages);
        reprojectParallel(scaledImages, reprojImages, maps);

        grayImages.resize(numImages);
        for (int j = 0; j < numImages; j++)
            cv::cvtColor(reprojImages[j], grayImages[j], CV_BGR2GRAY);        
        info.error = calcTotalError(grayImages, intersects);

        //bgrImages.resize(numImages);
        //for (int j = 0; j < numImages; j++)
        //{
        //    bgrImages[j].resize(3);
        //    cv::split(reprojImages[j], bgrImages[j]);
        //}        
        //bImages.resize(numImages);
        //gImages.resize(numImages);
        //rImages.resize(numImages);
        //for (int j = 0; j < numImages; j++)
        //{
        //    bImages[j] = bgrImages[j][0];
        //    gImages[j] = bgrImages[j][1];
        //    rImages[j] = bgrImages[j][2];
        //}
        //info.errorB = calcTotalError(bImages, intersects);
        //info.errorG = calcTotalError(gImages, intersects);
        //info.errorR = calcTotalError(rImages, intersects);

        if (i % 100 == 0)
        {
            printf("%d/%d, %f%% finish\n", i, numInfos, double(i) / numInfos);
        }
    }
}

void enumErrors(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& maps, const std::vector<cv::Mat>& masks,
    double minScale, double maxScale, int numSteps, std::vector<ScalesAndError>& infos)
{
    std::vector<MaskIntersection> intersects;
    calcMaskIntersections(masks, intersects);
    getScalesAndErrorVector(minScale, maxScale, numSteps, images.size(), infos);
    calcErrors(images, maps, intersects, infos);
}

void calcSqrDistToCenterImage(const cv::Size& size, cv::Mat& image)
{
    image.create(size, CV_64FC1);
    int rows = size.height, cols = size.width;
    double cx = cols * 0.5, cy = rows * 0.5;
    double scale = 1.0 / (cx * cx + cy * cy);
    for (int i = 0; i < rows; i++)
    {
        double* ptr = image.ptr<double>(i);
        for (int j = 0; j < cols; j++)
        {
            double diffx = j - cx, diffy = i - cy;
            ptr[j] = (diffx * diffx + diffy * diffy) * scale;
        }
    }
}

void reproject64FC1(const cv::Mat& src, cv::Mat& dst, const cv::Mat& map)
{
    CV_Assert(src.data && src.type() == CV_64FC1 &&
        map.data && map.type() == CV_64FC2);

    int rows = map.rows, cols = map.cols;
    dst.create(rows, cols, CV_64FC1);
    double minVal, maxVal;
    cv::minMaxLoc(src, &minVal, &maxVal);

    int srcRows = src.rows, srcCols = src.cols;
    for (int i = 0; i < rows; i++)
    {
        double* ptrDst = dst.ptr<double>(i);
        const cv::Point2d* ptrMap = map.ptr<cv::Point2d>(i);
        for (int j = 0; j < cols; j++)
        {
            cv::Point2d pt = ptrMap[j];
            if (pt.x >= 0 && pt.x < srcCols && pt.y >= 0 && pt.y < srcRows)
            {
                int x0 = pt.x, y0 = pt.y;
                int x1 = x0 + 1, y1 = y0 + 1;
                if (x1 >= srcCols) x1 = srcCols - 1;
                if (y1 >= srcRows) y1 = srcRows - 1;
                double wx0 = pt.x - x0, wx1 = 1 - wx0;
                double wy0 = pt.y - y0, wy1 = 1 - wy0;
                ptrDst[j] = wx1 * wy1 * src.at<double>(y0, x0) +
                    wx1 * wy0 * src.at<double>(y1, x0) +
                    wx0 * wy1 * src.at<double>(y0, x1) +
                    wx0 * wy0 * src.at<double>(y1, x1);
            }
            else
                ptrDst[j] = maxVal;
        }
    }
}

void rescaleImage(const cv::Mat& src, const cv::Mat& mask, const cv::Mat& dist, double scale, cv::Mat& dst)
{
    CV_Assert(src.data && (src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
        mask.data && mask.type() == CV_8UC1 && dist.data && dist.type() == CV_64FC1 &&
        src.size() == mask.size() && src.size() == dist.size());

    double alpha = 1.0 - scale;
    int rows = src.rows, cols = src.cols;
    dst.create(rows, cols, src.type());
    if (src.type() == CV_8UC1)
    {
        for (int i = 0; i < rows; i++)
        {
            const unsigned char* ptrSrc = src.ptr<unsigned char>(i);
            const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
            const double* ptrDist = dist.ptr<double>(i);
            unsigned char* ptrDst = dst.ptr<unsigned char>(i);
            for (int j = 0; j < cols; j++)
            {
                if (ptrMask[j])
                    ptrDst[j] = clamp0255(double(ptrSrc[j]) / (1.0 - alpha * ptrDist[j]));
                else
                    ptrDst[j] = ptrSrc[j];
            }
        }
    }
    else
    {
        for (int i = 0; i < rows; i++)
        {
            const unsigned char* ptrSrc = src.ptr<unsigned char>(i);
            const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
            const double* ptrDist = dist.ptr<double>(i);
            unsigned char* ptrDst = dst.ptr<unsigned char>(i);
            for (int j = 0; j < cols; j++)
            {
                if (ptrMask[j])
                {
                    ptrDst[j * 3] = clamp0255(double(ptrSrc[j * 3]) / (1.0 - alpha * ptrDist[j]));
                    ptrDst[j * 3 + 1] = clamp0255(double(ptrSrc[j * 3 + 1]) / (1.0 - alpha * ptrDist[j]));
                    ptrDst[j * 3 + 2] = clamp0255(double(ptrSrc[j * 3 + 2]) / (1.0 - alpha * ptrDist[j]));
                }                    
                else
                {
                    ptrDst[j * 3] = ptrSrc[j * 3];
                    ptrDst[j * 3 + 1] = ptrSrc[j * 3 + 1];
                    ptrDst[j * 3 + 2] = ptrSrc[j * 3 + 2];
                }
            }
        }
    }
}

void rescaleImages(const std::vector<cv::Mat>& srcs, const std::vector<cv::Mat>& masks,
    const std::vector<cv::Mat>& dists, const std::vector<double>& scales, std::vector<cv::Mat>& dsts)
{
    int size = srcs.size();
    dsts.resize(size);
    for (int i = 0; i < size; i++)
        rescaleImage(srcs[i], masks[i], dists[i], scales[i], dsts[i]);
}

void calcErrors(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, const std::vector<cv::Mat>& dists,
    const const std::vector<MaskIntersection>& intersects, std::vector<ScalesAndError>& infos)
{
    int numInfos = infos.size();
    int numImages = images.size();
    std::vector<cv::Mat> scaledImages;
    for (int i = 0; i < numInfos; i++)
    {
        ScalesAndError& info = infos[i];
        rescaleImages(images, masks, dists, info.scales, scaledImages);
        info.error = calcTotalError(scaledImages, intersects);

        if (i % 100 == 0)
        {
            printf("%d/%d, %f%% finish\n", i, numInfos, double(i) / numInfos);
        }
    }
}

void calcErrorsPartial(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, const std::vector<cv::Mat>& dists,
    const std::vector<MaskIntersection>& intersects, std::vector<ScalesAndError>& infos, int beg, int end)
{
    size_t id = std::this_thread::get_id().hash();
    end = end > infos.size() ? infos.size() : end;
    std::vector<cv::Mat> scaledImages;
    int count = 0;
    int totalCount = end - beg;
    for (int i = beg; i < end; i++)
    {
        ScalesAndError& info = infos[i];
        rescaleImages(images, masks, dists, info.scales, scaledImages);
        info.error = calcTotalError(scaledImages, intersects);
        count++;
        if (count % 100 == 0)
        {
            printf("[%16X] %d/%d, %f%% finish\n", id, count, totalCount, double(count) / totalCount * 100);
        }
    }
}

void calcErrorsParallel(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, const std::vector<cv::Mat>& dists,
    const const std::vector<MaskIntersection>& intersects, std::vector<ScalesAndError>& infos)
{
    int numCPUs = cv::getNumberOfCPUs();
    if (numCPUs > 1)
        numCPUs -= 1;
    int size = infos.size();
    int grain = (size + numCPUs - 1) / numCPUs;
    std::vector<std::thread> threads;
    for (int i = 0; i < numCPUs; i++)
    {
        threads.emplace_back(std::thread(calcErrorsPartial, std::ref(images), std::ref(masks),
            std::ref(dists), std::ref(intersects), std::ref(infos), i * grain, (i + 1) * grain));
    }
    for (int i = 0; i < numCPUs; i++)
        threads[i].join();
}

void listErrors(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, const std::vector<cv::Mat>& dists,
    double minScale, double maxScale, int numSteps, std::vector<ScalesAndError>& infos)
{
    std::vector<MaskIntersection> intersects;
    calcMaskIntersections(masks, intersects);
    getScalesAndErrorVector(minScale, maxScale, numSteps, images.size(), infos);
    calcErrorsParallel(images, masks, dists, intersects, infos);
}

// main2
int main2()
{
    //std::vector<std::string> imagePaths;
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\input-00.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\input-01.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\input-02.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\input-03.jpg");

    std::vector<std::string> imagePaths;
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\0.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\1.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\2.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\3.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\4.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\5.jpg");

    int numImages = imagePaths.size();
    std::vector<cv::Mat> origImages(numImages), images(numImages), maps(numImages), masks(numImages), grayImages(numImages);

    for (int i = 0; i < numImages; i++)
        origImages[i] = cv::imread(imagePaths[i]);

    std::vector<PhotoParam> params;
    //loadPhotoParamFromPTS("F:\\panoimage\\detuoffice\\4p.pts", params);
    loadPhotoParamFromXML("F:\\panoimage\\zhanxiang\\zhanxiang.xml", params);
    getReprojectMapsAndMasks(params, origImages[0].size(), cv::Size(2048, 1024), maps, masks);

    reprojectParallel(origImages, images, maps);
    for (int i = 0; i < numImages; i++)
        cv::cvtColor(images[i], grayImages[i], CV_BGR2GRAY);

    cv::Mat origDist;
    calcSqrDistToCenterImage(origImages[0].size(), origDist);
    std::vector<cv::Mat> dists(numImages);
    for (int i = 0; i < numImages; i++)
        reproject64FC1(origDist, dists[i], maps[i]);

    //std::vector<cv::Mat> eMasks;
    //getExtendedMasks(masks, 100, eMasks);

    std::vector<ScalesAndError> infos;
    //enumErrors(origImages, maps, masks, 0.5, 1.5, 10, infos);
    listErrors(grayImages, masks, dists, 0.9, 1.0, 2, infos);
    std::sort(infos.begin(), infos.end(),
        [](const ScalesAndError& lhs, const ScalesAndError& rhs){return lhs.error < rhs.error; });
    printf("result\n");
    for (int i = 0; i < 10; i++)
    {
        ScalesAndError& info = infos[i];
        for (int j = 0; j < info.scales.size(); j++)
            printf("%8.4f ", info.scales[j]);
        printf("Error: %f\n", info.error);
    }
    printf("result\n");
    for (int i = 0; i < 10; i++)
    {
        ScalesAndError& info = infos[infos.size() - 1 - i];
        for (int j = 0; j < info.scales.size(); j++)
            printf("%8.4f ", info.scales[j]);
        printf("Error: %f\n", info.error);
    }
    printf("\n");

    //std::vector<cv::Mat> scaleImages, scaledImages;
    //getScales(infos[0].scales, origImages[0].size(), scaleImages);
    //mulScales(origImages, scaleImages, scaledImages);
    //reprojectParallel(scaledImages, images, maps);

    rescaleImages(images, masks, dists, infos[0].scales, images);

    TilingLinearBlend blender;
    cv::Mat result;
    blender.prepare(masks, 100);
    blender.blend(images, result);
    cv::imshow("result", result);
    cv::waitKey(0);

    return 0;
}

// Minimize mutual difference
void getQuadSystemVals(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& dist1, const cv::Mat& dist2,
    const cv::Mat& mask, double& A1, double& A2, double& A12, double& A21, double& B1, double& B2)
{
    double a1 = 0, a2 = 0, a12 = 0, b1 = 0, b2 = 0;
    int rows = mask.rows, cols = mask.cols;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrImage1 = image1.ptr<unsigned char>(i);
        const unsigned char* ptrImage2 = image2.ptr<unsigned char>(i);
        const double* ptrDist1 = dist1.ptr<double>(i);
        const double* ptrDist2 = dist2.ptr<double>(i);
        const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrMask[j])
            {
                double lambda = 1.0 - 0.5 * sqrt(ptrDist1[j] * ptrDist2[j]);
                a1 += lambda * ptrDist1[j] * ptrDist1[j] * double(ptrImage1[j]) * double(ptrImage1[j]);
                a2 += lambda * ptrDist2[j] * ptrDist2[j] * double(ptrImage2[j]) * double(ptrImage2[j]);
                a12 += lambda * -2 * ptrDist1[j] * ptrDist2[j] * double(ptrImage1[j]) * double(ptrImage2[j]);
                b1 += lambda * 2 * (double(ptrImage1[j]) - double(ptrImage2[j])) * double(ptrImage1[j]) * ptrDist1[j];
                b2 += lambda * -2 * (double(ptrImage1[j]) - double(ptrImage2[j])) * double(ptrImage2[j]) * ptrDist2[j];
            }
        }
    }
    A1 = a1;
    A2 = a2;
    A12 = a12 / 2;
    A21 = a12 / 2;
    B1 = b1;
    B2 = b2;
}

// Mininize mutual difference
// y = x^T A x + B^T x
void getQuadSystem(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& dists,
    const std::vector<MaskIntersection>& intersects, cv::Mat& A, cv::Mat& B)
{
    int size = images.size();
    A.create(size, size, CV_64FC1);
    B.create(size, 1, CV_64FC1);
    A.setTo(0);
    B.setTo(0);

    int numInts = intersects.size();
    for (int i = 0; i < numInts; i++)
    {
        const MaskIntersection& mi = intersects[i];
        //cv::imshow("i mask", mi.mask);
        //cv::waitKey(0);
        double ai, aj, aij, aji, bi, bj;
        getQuadSystemVals(images[mi.i], images[mi.j], dists[mi.i], dists[mi.j], mi.mask,
            ai, aj, aij, aji, bi, bj);
        A.at<double>(mi.i, mi.i) += ai;
        A.at<double>(mi.j, mi.j) += aj;
        A.at<double>(mi.i, mi.j) += aij;
        A.at<double>(mi.j, mi.i) += aji;
        B.at<double>(mi.i) += bi;
        B.at<double>(mi.j) += bj;
    }
}

void solveScales(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& dists,
    const std::vector<cv::Mat>& masks, std::vector<double>& scales)
{
    int size = images.size();
    cv::Mat A, B, X;
    std::vector<MaskIntersection> intersects;
    calcMaskIntersections(masks, intersects);
    getQuadSystem(images, dists, intersects, A, B);
    B *= -0.5;
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    cv::solve(A, B, X);
    scales.resize(size);
    for (int i = 0; i < size; i++)
        scales[i] = X.at<double>(i);
}

void getQuadSystemVals(const cv::Mat& image, const cv::Mat& dist, const cv::Mat& mask, 
    double expectMean, double& A, double& B)
{
    double a = 0, b = 0;
    int rows = mask.rows, cols = mask.cols;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrImage = image.ptr<unsigned char>(i);
        const double* ptrDist = dist.ptr<double>(i);
        const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrMask[j])
            {
                a += double(ptrImage[j]) * double(ptrImage[j]) * ptrDist[j] * ptrDist[j];
                b += 2 * (ptrImage[j] - expectMean) * ptrImage[j] * ptrDist[j];
            }
        }
    }
    A = a;
    B = b;
}

void getQuadSystem(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& dists,
    const std::vector<cv::Mat>& masks, double expectMean, cv::Mat& A, cv::Mat& B)
{
    int size = images.size();
    A.create(size, size, CV_64FC1);
    B.create(size, 1, CV_64FC1);
    A.setTo(0);
    B.setTo(0);
    for (int i = 0; i < size; i++)
    {
        double a, b;
        getQuadSystemVals(images[i], dists[i], masks[i], expectMean, a, b);
        A.at<double>(i, i) += a;
        B.at<double>(i) += b;
    }
}

void solveScales2(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& dists,
    const std::vector<cv::Mat>& masks, std::vector<double>& scales)
{
    int size = images.size();
    cv::Mat A1, B1, A2, B2, A, B, X;

    std::vector<MaskIntersection> intersects;
    calcMaskIntersections(masks, intersects);

    getQuadSystem(images, dists, intersects, A1, B1);
    B1 *= -0.5;
    int count1 = 0;
    for (int i = 0, numInts = intersects.size(); i < numInts; i++)
        count1 += intersects[i].numNonZero;

    int maxMeanIndex = 0;
    double maxMean = cv::mean(images[0], masks[0])[0];
    for (int i = 1; i < size; i++)
    {
        double currMean = cv::mean(images[i], masks[i])[0];
        if (currMean > maxMean)
        {
            maxMean = currMean;
            maxMeanIndex = i;
        }
    }

    getQuadSystem(images, dists, masks, maxMean, A2, B2);
    B2 *= -0.5;
    int count2 = 0;
    for (int i = 0; i < size; i++)
        count2 += cv::countNonZero(masks[i]);

    double scale1 = 1.0 / count1, scale2 = 1.0 / count2;
    A = scale1 * A1 + scale2 * A2;
    B = scale1 * B1 + scale2 * B2;
    std::cout << A << "\n";
    std::cout << B << "\n";

    cv::solve(A, B, X);
    scales.resize(size);
    for (int i = 0; i < size; i++)
        scales[i] = X.at<double>(i);
}

void show64FC1(const std::string& winName, cv::Mat& mat)
{
    CV_Assert(mat.data && mat.type() == CV_64FC1);
    double minVal, maxVal;
    cv::minMaxLoc(mat, &minVal, &maxVal);
    cv::Mat forShow = mat * (1.0 / maxVal);
    cv::imshow(winName, forShow);
}

static void calcParabollaScale(const cv::Size& size, double alpha, cv::Mat& scale)
{
    scale.create(size, CV_64FC1);
    int halfHeight = size.height / 2, halfWidth = size.width / 2;
    for (int i = 0; i < size.height; i++)
    {
        double* ptr = scale.ptr<double>(i);
        for (int j = 0; j < size.width; j++)
        {
            int sqrDiff = (i - halfHeight / 2) * (i - halfHeight / 2) + (j - halfWidth) * (j - halfWidth);
            ptr[j] = 1.0 + alpha * sqrDiff;
        }
    }
}

void rescaleImage2(const cv::Mat& src, const cv::Mat& mask, const cv::Mat& dist, double scale, cv::Mat& dst)
{
    CV_Assert(src.data && (src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
        mask.data && mask.type() == CV_8UC1 && dist.data && dist.type() == CV_64FC1 &&
        src.size() == mask.size() && src.size() == dist.size());

    int rows = src.rows, cols = src.cols;
    dst.create(rows, cols, src.type());
    if (src.type() == CV_8UC1)
    {
        for (int i = 0; i < rows; i++)
        {
            const unsigned char* ptrSrc = src.ptr<unsigned char>(i);
            const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
            const double* ptrDist = dist.ptr<double>(i);
            unsigned char* ptrDst = dst.ptr<unsigned char>(i);
            for (int j = 0; j < cols; j++)
            {
                if (ptrMask[j])
                    ptrDst[j] = clamp0255(double(ptrSrc[j]) * (1.0 + scale * ptrDist[j]));
                else
                    ptrDst[j] = ptrSrc[j];
            }
        }
    }
    else
    {
        for (int i = 0; i < rows; i++)
        {
            const unsigned char* ptrSrc = src.ptr<unsigned char>(i);
            const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
            const double* ptrDist = dist.ptr<double>(i);
            unsigned char* ptrDst = dst.ptr<unsigned char>(i);
            for (int j = 0; j < cols; j++)
            {
                if (ptrMask[j])
                {
                    ptrDst[j * 3] = clamp0255(double(ptrSrc[j * 3]) * (1.0 + scale * ptrDist[j]));
                    ptrDst[j * 3 + 1] = clamp0255(double(ptrSrc[j * 3 + 1]) * (1.0 + scale * ptrDist[j]));
                    ptrDst[j * 3 + 2] = clamp0255(double(ptrSrc[j * 3 + 2]) * (1.0 + scale * ptrDist[j]));
                }
                else
                {
                    ptrDst[j * 3] = ptrSrc[j * 3];
                    ptrDst[j * 3 + 1] = ptrSrc[j * 3 + 1];
                    ptrDst[j * 3 + 2] = ptrSrc[j * 3 + 2];
                }
            }
        }
    }
}

void rescaleImages2(const std::vector<cv::Mat>& srcs, const std::vector<cv::Mat>& masks,
    const std::vector<cv::Mat>& dists, const std::vector<double>& scales, std::vector<cv::Mat>& dsts)
{
    int size = srcs.size();
    dsts.resize(size);
    for (int i = 0; i < size; i++)
        rescaleImage2(srcs[i], masks[i], dists[i], scales[i], dsts[i]);
}

// main3
int main3()
{
    std::vector<std::string> imagePaths;
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-00.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-01.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-02.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-03.jpg");

    //std::vector<std::string> imagePaths;
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\0.jpg");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\1.jpg");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\2.jpg");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\3.jpg");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\4.jpg");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\5.jpg");

    int numImages = imagePaths.size();
    std::vector<cv::Mat> origImages(numImages), images(numImages), maps(numImages), masks(numImages), grayImages(numImages);

    for (int i = 0; i < numImages; i++)
        origImages[i] = cv::imread(imagePaths[i]);

    std::vector<PhotoParam> params;
    loadPhotoParamFromPTS("F:\\panoimage\\detuoffice\\4p.pts", params);
    //loadPhotoParamFromXML("F:\\panoimage\\zhanxiang\\zhanxiang.xml", params);
    getReprojectMapsAndMasks(params, origImages[0].size(), cv::Size(2048, 1024), maps, masks);
    reprojectParallel(origImages, images, maps);
    for (int i = 0; i < numImages; i++)
        cv::cvtColor(images[i], grayImages[i], CV_BGR2GRAY);

    cv::Mat origDist;
    calcSqrDistToCenterImage(origImages[0].size(), origDist);
    std::vector<cv::Mat> dists(numImages);
    for (int i = 0; i < numImages; i++)
        reproject64FC1(origDist, dists[i], maps[i]);

    //for (int i = 0; i < numImages; i++)
    //{
    //    show64FC1("dist", dists[i]);
    //    cv::imshow("gray", grayImages[i]);
    //    cv::imshow("color", images[i]);
    //    cv::imshow("mask", masks[i]);
    //    cv::waitKey(0);
    //}

    std::vector<double> scales;
    solveScales2(grayImages, dists, masks, scales);
    for (int i = 0; i < numImages; i++)
        printf("%f ", scales[i]);
    printf("\n");

    rescaleImages2(images, masks, dists, scales, images);
    for (int i = 0; i < numImages; i++)
    {
        cv::imshow("image", images[i]);
        cv::waitKey(0);
    }

    TilingLinearBlend blender;
    blender.prepare(masks, 100);
    cv::Mat result;
    blender.blend(images, result);
    cv::imshow("result", result);
    cv::waitKey(0);

    return 0;
}

inline void cvtPDirToKH(const cv::Point2d& p, const cv::Point2d& dir, double& k, double& h)
{
    k = dir.y / dir.x;
    h = p.y - k * p.x;
}


static void iterativeGainAdjust(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results)
{
    std::vector<cv::Mat> extendedMasks;
    getExtendedMasks(masks, 50, extendedMasks);

    TilingMultibandBlendFast blender;
    blender.prepare(masks, 20, 2);

    cv::Mat blendImage;
    int numImages = src.size();
    std::vector<double> kvals(numImages), hvals(numImages);
    std::vector<int> darkImageIndexes;
    darkImageIndexes.reserve(numImages);

    std::vector<cv::Mat> images(numImages);
    for (int i = 0; i < numImages; i++)
        src[i].copyTo(images[i]);

    while (true)
    {
        blender.blend(images, blendImage);
        cv::imshow("blend image", blendImage);
        cv::waitKey(0);

        darkImageIndexes.clear();
        for (int k = 0; k < numImages; k++)
        {
            const cv::Mat& image = images[k];
            const cv::Mat& mask = extendedMasks[k];

            int count = cv::countNonZero(mask);
            std::vector<cv::Point> valPairs(count);
            int rows = mask.rows, cols = mask.cols;
            cv::Mat blendGray, imageGray;
            cv::cvtColor(blendImage, blendGray, CV_BGR2GRAY);
            cv::cvtColor(image, imageGray, CV_BGR2GRAY);
            int index = 0;
            for (int i = 0; i < rows; i++)
            {
                const unsigned char* ptrBlend = blendGray.ptr<unsigned char>(i);
                const unsigned char* ptrImage = imageGray.ptr<unsigned char>(i);
                const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
                for (int j = 0; j < cols; j++)
                {
                    if (ptrMask[j])
                        valPairs[index++] = cv::Point(ptrImage[j], ptrBlend[j]);
                }
            }
            cv::Point2d p, dir;
            getLineRANSAC(valPairs, p, dir);

            //cv::Mat hist2D, histShow;
            //calcHist2D(valPairs, hist2D);
            //normalizeAndConvert(hist2D, histShow);
            //cv::imshow("hist", histShow);
            //cv::imwrite("hist.bmp", histShow);

            //double r = 500;
            //cv::Point2d p0(p.x + dir.x * r, p.y + dir.y * r), p1(p.x - dir.x * r, p.y - dir.y * r);
            //cv::Mat lineShow = cv::Mat::zeros(256, 256, CV_8UC1);
            //cv::line(lineShow, p0, p1, cv::Scalar(255));
            //cv::imshow("line", lineShow);
            //cv::waitKey(0);

            cvtPDirToKH(p, dir, kvals[k], hvals[k]);

            printf("[%d] k = %f, b = %f\n", k, kvals[k], hvals[k]);
            /*if (kvals[k] > 1.05 || hvals[k] > 10)
            {
                darkImageIndexes.push_back(k);
                std::vector<unsigned char> lut;
                lineParamToLUT(kvals[k], hvals[k], lut);
                transform(images[k], images[k], lut, masks[k]);
            }*/
        }
        int maxHValIndex = 0;
        double maxHVal = hvals[0];
        for (int i = 0; i < numImages; i++)
        {
            if (hvals[i] > maxHVal)
            {
                maxHValIndex = i;
                maxHVal = hvals[i];
            }
        }
        if (kvals[maxHValIndex] > 1.05 || hvals[maxHValIndex] > 10)
        {
            darkImageIndexes.push_back(maxHValIndex);
            std::vector<unsigned char> lut;
            lineParamToLUT(kvals[maxHValIndex], hvals[maxHValIndex], lut);
            transform(images[maxHValIndex], images[maxHValIndex], lut, masks[maxHValIndex]);
        }
        break;
        //if (darkImageIndexes.empty())
        //    break;
    }

    blender.blend(images, blendImage);
    cv::imshow("blend image", blendImage);
    cv::waitKey(0);

    TilingLinearBlend lBlender;
    lBlender.prepare(masks, 100);
    lBlender.blend(images, blendImage);
    cv::imshow("linear blend image", blendImage);
    cv::waitKey(0);
}

// main4
int main4()
{
    //std::vector<std::string> imagePaths;
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\input-00.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\input-01.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\input-02.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\input-03.jpg");

    std::vector<std::string> imagePaths;
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\0.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\1.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\2.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\3.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\4.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\5.jpg");

    int numImages = imagePaths.size();
    std::vector<cv::Mat> origImages(numImages), images(numImages), maps(numImages), masks(numImages);

    for (int i = 0; i < numImages; i++)
        origImages[i] = cv::imread(imagePaths[i]);

    std::vector<PhotoParam> params;
    //loadPhotoParamFromPTS("F:\\panoimage\\detuoffice\\4p.pts", params);
    loadPhotoParamFromXML("F:\\panoimage\\zhanxiang\\zhanxiang.xml", params);
    getReprojectMapsAndMasks(params, origImages[0].size(), cv::Size(2048, 1024), maps, masks);
    reprojectParallel(origImages, images, maps);

    std::vector<cv::Mat> results;
    iterativeGainAdjust(images, masks, results);

    return 0;
}

void calcRatio(const cv::Mat& image, const cv::Mat& mask, cv::Mat& bgRatio, cv::Mat& rgRatio)
{
    CV_Assert(image.data && image.type() == CV_8UC3 &&
        mask.data && mask.type() == CV_8UC1 && mask.size() == image.size());

    int rows = image.rows, cols = image.cols;
    bgRatio.create(rows, cols, CV_32FC1);
    rgRatio.create(rows, cols, CV_32FC1);
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrImage = image.ptr<unsigned char>(i);
        const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        float* ptrBG = bgRatio.ptr<float>(i);
        float* ptrRG = rgRatio.ptr<float>(i);
        for (int j = 0; j < cols; j++)
        {
            if (*ptrMask)
            {
                float b = ptrImage[0];
                float g = ptrImage[1];
                float r = ptrImage[2];
                if (g == 0)
                {
                    *ptrBG = 0;
                    *ptrRG = 0;
                }
                else
                {
                    *ptrBG = b / g;
                    *ptrRG = r / g;
                }
            }
            else
            {
                *ptrBG = 0;
                *ptrRG = 0;
            }
            ptrImage += 3;
            ptrMask++;
            ptrBG++;
            ptrRG++;
        }
    }
}

void isDiffSmall(const cv::Mat& lhs, const cv::Mat& rhs, int thresh, cv::Mat& result)
{
    CV_Assert(lhs.data && lhs.type() == CV_8UC3 &&
        rhs.data && rhs.type() == CV_8UC3 && lhs.size() == rhs.size());

    int rows = lhs.rows, cols = rhs.cols;
    result.create(rows, cols, CV_8UC1);
    cv::Mat blhs, brhs;
    cv::Size sz(9, 9);
    cv::boxFilter(lhs, blhs, CV_8U, sz);
    cv::boxFilter(rhs, brhs, CV_8U, sz);
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrl = blhs.ptr<unsigned char>(i);
        const unsigned char* ptrr = brhs.ptr<unsigned char>(i);
        unsigned char* ptr = result.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (abs(int(ptrl[0]) - ptrr[0]) < thresh &&
                abs(int(ptrl[1]) - ptrr[1]) < thresh &&
                abs(int(ptrl[2]) - ptrr[2]) < thresh)
                *ptr = 255;
            else
                *ptr = 0;
            ptr++;
            ptrl += 3;
            ptrr += 3;
        }
    }
}

void scale(const cv::Mat& src, const cv::Mat& mask, double rRatio, double bRatio, cv::Mat& dst)
{
    CV_Assert(src.data && src.type() == CV_8UC3 &&
        mask.data && mask.type() == CV_8UC1 && mask.size() == src.size());
    int rows = src.rows, cols = src.cols;
    dst.create(rows, cols, CV_8UC3);
    unsigned char rlut[256], blut[256];
    for (int i = 0; i < 256; i++)
    {
        rlut[i] = clamp0255(i * rRatio);
        blut[i] = clamp0255(i * bRatio);
    }
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrSrc = src.ptr<unsigned char>(i);
        const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        unsigned char* ptrDst = dst.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (*ptrMask)
            {
                ptrDst[0] = blut[ptrSrc[0]];
                ptrDst[1] = ptrSrc[1];
                ptrDst[2] = rlut[ptrSrc[2]];
            }
            else
            {
                ptrDst[0] = 0;
                ptrDst[1] = 0;
                ptrDst[2] = 0;
            }
            ptrSrc += 3;
            ptrMask++;
            ptrDst += 3;
        }
    }
}

static void getLinearTransforms(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    std::vector<double>& rgRatioGains, std::vector<double>& bgRatioGains)
{
    int numImages = images.size();

    cv::Mat_<double> N(numImages, numImages), rgI(numImages, numImages), bgI(numImages, numImages);
    for (int i = 0; i < numImages; i++)
    {
        for (int j = 0; j < numImages; j++)
        {
            if (i == j)
            {
                N(i, i) = cv::countNonZero(masks[i]);
                cv::Scalar meanVal = cv::mean(images[i], masks[i]);
                rgI(i, i) = meanVal[2] / meanVal[1];
                bgI(i, i) = meanVal[0] / meanVal[1];
            }
            else
            {
                cv::Mat intersect = masks[i] & masks[j];
                int numNonZero = cv::countNonZero(intersect);
                N(i, j) = numNonZero;
                if (numNonZero)
                {
                    cv::Scalar meanVal = cv::mean(images[i], intersect);
                    rgI(i, j) = meanVal[2] / meanVal[1];
                    bgI(i, j) = meanVal[0] / meanVal[1];
                }
                else
                {
                    rgI(i, j) = 0;
                    bgI(i, j) = 0;
                }
            }
        }
    }
    //std::cout << N << "\n" << rgI << "\n" << bgI << "\n";

    double invSigmaNSqr = 1;
    double invSigmaGSqr = 0.05;

    bool success;

    cv::Mat_<double> A(numImages, numImages);
    cv::Mat_<double> b(numImages, 1);
    cv::Mat_<double> gains(numImages, 1);

    A.setTo(0);
    b.setTo(0);
    for (int i = 0; i < numImages; ++i)
    {
        for (int j = 0; j < numImages; ++j)
        {
            A(i, i) += N[i][j] * (rgI[i][j] * rgI[i][j] * invSigmaNSqr + invSigmaGSqr);
            A(j, j) += N[i][j] * (rgI[j][i] * rgI[j][i] * invSigmaNSqr);
            A(i, j) -= 2 * N[i][j] * (rgI[i][j] * rgI[j][i] * invSigmaNSqr);
            b(i) += N[i][j] * invSigmaGSqr;
        }
    }

    //std::cout << A << "\n" << b << "\n";
    success = cv::solve(A, b, gains);
    //std::cout << gains << "\n";
    if (!success)
        gains.setTo(1);

    rgRatioGains.resize(numImages);
    for (int i = 0; i < numImages; i++)
        rgRatioGains[i] = gains(i);

    A.setTo(0);
    b.setTo(0);
    for (int i = 0; i < numImages; ++i)
    {
        for (int j = 0; j < numImages; ++j)
        {
            A(i, i) += N[i][j] * (bgI[i][j] * bgI[i][j] * invSigmaNSqr + invSigmaGSqr);
            A(j, j) += N[i][j] * (bgI[j][i] * bgI[j][i] * invSigmaNSqr);
            A(i, j) -= 2 * N[i][j] * (bgI[i][j] * bgI[j][i] * invSigmaNSqr);
            b(i) += N[i][j] * invSigmaGSqr;
        }
    }

    //std::cout << A << "\n" << b << "\n";
    success = cv::solve(A, b, gains);
    //std::cout << gains << "\n";
    if (!success)
        gains.setTo(1);

    bgRatioGains.resize(numImages);
    for (int i = 0; i < numImages; i++)
        bgRatioGains[i] = gains(i);
}

void tintAdjust(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results)
{
    std::vector<double> rgGains, bgGains;
    getLinearTransforms(images, masks, rgGains, bgGains);
    int numImages = images.size();
    results.resize(numImages);
    for (int i = 0; i < numImages; i++)
        scale(images[i], masks[i], rgGains[i], bgGains[i], results[i]);
}

// main5
int main5()
{
    //std::vector<std::string> imagePaths;
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\image0.bmp");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\image1.bmp");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\image2.bmp");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\image3.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\detuoffice\\mask0.bmp");
    //maskPaths.push_back("F:\\panoimage\\detuoffice\\mask1.bmp");
    //maskPaths.push_back("F:\\panoimage\\detuoffice\\mask2.bmp");
    //maskPaths.push_back("F:\\panoimage\\detuoffice\\mask3.bmp");

    std::vector<std::string> imagePaths;
    imagePaths.push_back("F:\\panoimage\\919-4\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\image3.bmp");
    std::vector<std::string> maskPaths;
    maskPaths.push_back("F:\\panoimage\\919-4\\mask0.bmp");
    maskPaths.push_back("F:\\panoimage\\919-4\\mask1.bmp");
    maskPaths.push_back("F:\\panoimage\\919-4\\mask2.bmp");
    maskPaths.push_back("F:\\panoimage\\919-4\\mask3.bmp");

    //std::vector<std::string> imagePaths;
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\0.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\1.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\2.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\3.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\4.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\5.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\0mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\1mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\2mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\3mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\4mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\5mask.bmp");

    //std::vector<std::string> imagePaths;
    //imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage0.bmp");
    //imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage1.bmp");
    //imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage2.bmp");
    //imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage3.bmp");
    //imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage4.bmp");
    //imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage5.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask0.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask1.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask2.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask3.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask4.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask5.bmp");

    int numImages = imagePaths.size();
    std::vector<cv::Mat> images(numImages), masks(numImages);
    for (int i = 0; i < numImages; i++)
    {
        images[i] = cv::imread(imagePaths[i]);
        masks[i] = cv::imread(maskPaths[i], -1);
    }

    int indexL = 0;
    int indexR = 1;
    cv::Mat intersectMask = masks[indexL] & masks[indexR];
    cv::Mat diffSmall;
    isDiffSmall(images[indexL], images[indexR], 40, diffSmall);
    std::vector<cv::Mat> rgRatios(numImages), bgRatios(numImages);
    calcRatio(images[indexL], intersectMask, bgRatios[indexL], rgRatios[indexL]);
    calcRatio(images[indexR], intersectMask, bgRatios[indexR], rgRatios[indexR]);
    cv::Mat bgDiff = bgRatios[indexR] - bgRatios[indexL];
    cv::Mat rgDiff = rgRatios[indexR] - rgRatios[indexL];
    cv::Scalar bgDiffMean, bgDiffStdDev, rgDiffMean, rgDiffStdDev;
    intersectMask &= diffSmall;
    cv::meanStdDev(bgDiff, bgDiffMean, bgDiffStdDev, intersectMask);
    cv::meanStdDev(rgDiff, rgDiffMean, rgDiffStdDev, intersectMask);
    cv::Mat newImageR;
    scale(images[indexR], masks[indexR], 1.0 - rgDiffMean[indexL] * 0.8, 1.0 - bgDiffMean[indexL] * 0.8, newImageR);
    cv::imshow("L", images[indexL]);
    cv::imshow("old R", images[indexR]);
    cv::imshow("new R", newImageR);
    cv::waitKey(0);

    cv::Scalar meanL, meanR;
    meanL = cv::mean(images[indexL], intersectMask);
    meanR = cv::mean(images[indexR], intersectMask);
    double ratioRGL = meanL[2] / meanL[1], ratioBGL = meanL[0] / meanL[1];
    double ratioRGR = meanR[2] / meanR[1], ratioBGR = meanR[0] / meanR[1];
    scale(images[indexR], masks[indexR], 1.0 + (ratioRGL - ratioRGR), 1.0 + (ratioBGL - ratioBGR), newImageR);
    cv::imshow("L", images[indexL]);
    cv::imshow("old R", images[indexR]);
    cv::imshow("new R", newImageR);
    cv::waitKey(0);

    std::vector<cv::Mat> compResults;
    //compensate(images, masks, compResults);  
    MultibandBlendGainAdjust gainAdjust;
    gainAdjust.prepare(masks, 50);
    std::vector<std::vector<unsigned char> > luts;
    gainAdjust.calcGain(images, luts);
    compResults.resize(numImages);
    for (int i = 0; i < numImages; i++)
        transform(images[i], compResults[i], luts[i]);

    std::vector<cv::Mat> tintResults(numImages);
    tintAdjust(compResults, masks, tintResults);

    for (int i = 0; i < numImages; i++)
    {
        cv::imshow("orig", images[i]);
        cv::imshow("comp", compResults[i]);
        cv::imshow("tint", tintResults[i]);
        cv::waitKey(0);
    }

    TilingLinearBlend blend;
    blend.prepare(masks, 100);
    cv::Mat result;
    blend.blend(images, result);
    cv::imshow("old blend", result);
    blend.blend(compResults, result);
    cv::imshow("gain blend", result);
    blend.blend(tintResults, result);
    cv::imshow("gain tint blend", result);
    //cv::imwrite("GainTintBlend2.bmp", result);
    cv::waitKey(0);

    return 0;
}

void calcCompDistToCenterImage(const cv::Size& size, cv::Mat& image)
{
    image.create(size, CV_32FC1);
    int rows = size.height, cols = size.width;
    double cx = cols * 0.5, cy = rows * 0.5;
    double r = sqrt(cx * cx + cy * cy);
    for (int i = 0; i < rows; i++)
    {
        float* ptr = image.ptr<float>(i);
        for (int j = 0; j < cols; j++)
        {
            double diffx = j - cx, diffy = i - cy;
            ptr[j] = r - sqrt(diffx * diffx + diffy * diffy);
        }
    }
}

void reproject32FC1(const cv::Mat& src, cv::Mat& dst, const cv::Mat& map)
{
    CV_Assert(src.data && src.type() == CV_32FC1 &&
        map.data && map.type() == CV_64FC2);

    int rows = map.rows, cols = map.cols;
    dst.create(rows, cols, CV_32FC1);
    double minVal, maxVal;
    cv::minMaxLoc(src, &minVal, &maxVal);

    int srcRows = src.rows, srcCols = src.cols;
    for (int i = 0; i < rows; i++)
    {
        float* ptrDst = dst.ptr<float>(i);
        const cv::Point2d* ptrMap = map.ptr<cv::Point2d>(i);
        for (int j = 0; j < cols; j++)
        {
            cv::Point2d pt = ptrMap[j];
            if (pt.x >= 0 && pt.x < srcCols && pt.y >= 0 && pt.y < srcRows)
            {
                int x0 = pt.x, y0 = pt.y;
                int x1 = x0 + 1, y1 = y0 + 1;
                if (x1 >= srcCols) x1 = srcCols - 1;
                if (y1 >= srcRows) y1 = srcRows - 1;
                float wx0 = pt.x - x0, wx1 = 1 - wx0;
                float wy0 = pt.y - y0, wy1 = 1 - wy0;
                ptrDst[j] = wx1 * wy1 * src.at<float>(y0, x0) +
                    wx1 * wy0 * src.at<float>(y1, x0) +
                    wx0 * wy1 * src.at<float>(y0, x1) +
                    wx0 * wy0 * src.at<float>(y1, x1);
            }
            else
                ptrDst[j] = maxVal;
        }
    }
}

void calcWeights32F(const std::vector<cv::Mat>& dists, std::vector<cv::Mat>& weights)
{
    int numImages = dists.size();
    int rows = dists[0].rows, cols = dists[0].cols;

    weights.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        weights[i].create(rows, cols, CV_32FC1);
        weights[i].setTo(0);
    }

    std::vector<const float*> ptrDistVector(numImages);
    const float** ptrDist = &ptrDistVector[0];
    std::vector<float*> ptrWeightVector(numImages);
    float** ptrWeight = &ptrWeightVector[0];
    for (int i = 0; i < rows; i++)
    {
        for (int k = 0; k < numImages; k++)
        {
            ptrDist[k] = dists[k].ptr<float>(i);
            ptrWeight[k] = weights[k].ptr<float>(i);
        }
        for (int j = 0; j < cols; j++)
        {
            float sum = 0;
            int nonZeroCount = 0;
            int nonZeroIndex = 0;
            for (int k = 0; k < numImages; k++)
            {
                sum += ptrDist[k][j];
                if (ptrDist[k][j])
                {
                    nonZeroCount++;
                    nonZeroIndex = k;
                }
            }
            if (nonZeroCount > 1)
            {
                sum = fabs(sum) <= FLT_MIN ? 0 : 1.0F / sum;
                for (int k = 0; k < numImages; k++)
                    ptrWeight[k][j] = ptrDist[k][j] * sum;
            }
            else if (nonZeroCount == 1)
                ptrWeight[nonZeroIndex][j] = 1.0;

            //float sum = 0;
            //for (int k = 0; k < numImages; k++)
            //    sum += ptrDist[k][j];
            //sum = fabs(sum) <= FLT_MIN ? 0 : 1.0F / sum;
            //int intSum = 0;
            //for (int k = 0; k < numImages; k++)
            //{
            //    ;
            //    ptrWeight[k][j] = ptrDist[k][j] * sum;
            //}
        }
    }
}

// main6
int main6()
{
    std::vector<std::string> paths;
    paths.push_back("F:\\panoimage\\919-4\\snapshot0(2).bmp");
    paths.push_back("F:\\panoimage\\919-4\\snapshot1(2).bmp");
    paths.push_back("F:\\panoimage\\919-4\\snapshot2(2).bmp");
    paths.push_back("F:\\panoimage\\919-4\\snapshot3(2).bmp");

    int numImages = paths.size();
    std::vector<cv::Mat> src(numImages);
    for (int i = 0; i < numImages; i++)
        src[i] = cv::imread(paths[i]);

    std::vector<PhotoParam> params;
    //loadPhotoParams("E:\\Projects\\GitRepo\\panoLive\\PanoLive\\PanoLive\\PanoLive\\201603260848.vrdl", params);
    loadPhotoParamFromXML("F:\\panoimage\\919-4\\vrdl201606231708.xml", params);

    cv::Size dstSize(2048, 1024);
    std::vector<cv::Mat> maps, masks;
    getReprojectMapsAndMasks(params, src[0].size(), dstSize, maps, masks);
    
    cv::Mat dist, compMask;
    calcCompDistToCenterImage(dstSize, dist);
    std::vector<cv::Mat> dists(numImages);
    for (int i = 0; i < numImages; i++)
    {
        reproject32FC1(dist, dists[i], maps[i]);
        compMask = ~masks[i];
        dists[i].setTo(0, compMask);
    }

    std::vector<cv::Mat> dst;
    reproject(src, dst, maps);

    std::vector<cv::Mat> weights;
    calcWeights32F(dists, weights);

    std::vector<cv::Mat> compResults;
    MultibandBlendGainAdjust gainAdjust;
    gainAdjust.prepare(masks, 50);
    std::vector<std::vector<unsigned char> > luts;
    gainAdjust.calcGain(dst, luts);
    compResults.resize(numImages);
    for (int i = 0; i < numImages; i++)
        transform(src[i], compResults[i], luts[i]);
    
    cv::Mat result32F = cv::Mat::zeros(dstSize, CV_32FC3);
    for (int i = 0; i < numImages; i++)
        reprojectWeightedAccumulateTo32F(compResults[i], result32F, maps[i], weights[i]);
    cv::Mat result;
    result32F.convertTo(result, CV_8U);
    cv::imshow("result", result);
    cv::waitKey(0);

    return 0;
}

void getExtendedMasks(const std::vector<cv::Mat>& masks, int radius, std::vector<cv::Mat>& extendedMasks);

void cvtPDirToKH(const cv::Point2d& p, const cv::Point2d& dir, double& k, double& h);

static void getLUT(std::vector<unsigned char>& lut, double k, double b)
{
    lut.resize(256);
    for (int i = 0; i < 256; i++)
        lut[i] = cv::saturate_cast<unsigned char>(i * k + b);
}

static void normalizeHist(const std::vector<int>& src, std::vector<double>& dst)
{
    if (src.empty() || src.size() != 256)
    {
        dst.resize(256, 0);
        return;
    }

    int count = 0;
    for (int i = 0; i < 256; i++)
        count += src[i];
    if (count == 0)
    {
        dst.resize(256, 0);
        return;
    }
    double scale = 10.0 / count;
    dst.resize(256);
    for (int i = 0; i < 256; i++)
        dst[i] = src[i] * scale;
}

void fitParabola(const std::vector<cv::Point>& pts, double& a, double& b, double& c)
{
    int numPoints = pts.size();
    CV_Assert(numPoints >= 3);

    //cv::Mat_<double> A(numPoints, 3);
    //cv::Mat_<double> B(numPoints, 1);
    //cv::Mat_<double> X;
    //for (int i = 0; i < numPoints; i++)
    //{
    //    double val = pts[i].x;
    //    A(i, 0) = 1;
    //    A(i, 1) = val;
    //    val *= val;
    //    A(i, 2) = val;
    //    B(i) = pts[i].y;
    //}
    //bool success = cv::solve(A, B, X, cv::DECOMP_NORMAL);
    //printf("result of fitting parabola:\n");
    //std::cout << X << "\n";
    //if (success)
    //{
    //    a = X(2);
    //    b = X(1);
    //    c = X(0);
    //}
    //else
    //{
    //    a = 0;
    //    b = 1;
    //    c = 0;
    //}

    double A = 0, B = 0;
    for (int i = 0; i < numPoints; i++)
    {
        double x = pts[i].x, y = pts[i].y;
        double temp1 = x * x - 255 * x;
        double temp2 = x - y;
        A += temp1 * temp1;
        B += temp1 * temp2;
    }
    if (abs(A) < 0.001)
    {
        a = 0; 
        b = 1;
        c = 0;
    }
    else
    {
        a = -B / A;
        b = 1 - 255 * a;
        c = 0;
    }
}

void getLUT(std::vector<unsigned char>& lut, double a, double b, double c)
{
    lut.resize(256);
    for (int i = 0; i < 256; i++)
        lut[i] = cv::saturate_cast<unsigned char>(a * i * i + b * i + c);
}

void calcTransform(const cv::Mat& image, const cv::Mat& imageMask, const cv::Mat& base, const cv::Mat& baseMask,
    std::vector<unsigned char>& lut)
{
    CV_Assert(image.data && image.type() == CV_8UC3 &&
        base.data && base.type() == CV_8UC3 && imageMask.data && imageMask.type() == CV_8UC1 &&
        baseMask.data && baseMask.type() == CV_8UC1);
    int rows = image.rows, cols = image.cols;
    CV_Assert(imageMask.rows == rows && imageMask.cols == cols &&
        base.rows == rows && base.cols == cols && baseMask.rows == rows && baseMask.cols == cols);

    cv::Mat intersectMask = imageMask & baseMask;

    int count = cv::countNonZero(intersectMask);
    std::vector<cv::Point> valPairs(count);
    cv::Mat baseGray, imageGray;
    cv::cvtColor(base, baseGray, CV_BGR2GRAY);
    cv::cvtColor(image, imageGray, CV_BGR2GRAY);

    int index = 0;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrBase = baseGray.ptr<unsigned char>(i);
        const unsigned char* ptrImage = imageGray.ptr<unsigned char>(i);
        const unsigned char* ptrMask = intersectMask.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            int baseVal = ptrBase[j];
            int imageVal = ptrImage[j];
            if (ptrMask[j] && baseVal > 15 && baseVal < 240)
            {
                valPairs[index++] = cv::Point(imageVal, baseVal);
            }
        }
    }
    valPairs.resize(index);

    double a, b, c;
    fitParabola(valPairs, a, b, c);
    getLUT(lut, a, b, c);
    //showLUT("parabola lut", lut);
    //cv::waitKey(0);
}

// main7
int main7()
{
    //std::vector<std::string> imagePaths;
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\image0.bmp");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\image1.bmp");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\image2.bmp");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\image3.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\detuoffice\\mask0.bmp");
    //maskPaths.push_back("F:\\panoimage\\detuoffice\\mask1.bmp");
    //maskPaths.push_back("F:\\panoimage\\detuoffice\\mask2.bmp");
    //maskPaths.push_back("F:\\panoimage\\detuoffice\\mask3.bmp");

    std::vector<std::string> imagePaths;
    imagePaths.push_back("F:\\panoimage\\919-4\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\image3.bmp");
    std::vector<std::string> maskPaths;
    maskPaths.push_back("F:\\panoimage\\919-4\\mask0.bmp");
    maskPaths.push_back("F:\\panoimage\\919-4\\mask1.bmp");
    maskPaths.push_back("F:\\panoimage\\919-4\\mask2.bmp");
    maskPaths.push_back("F:\\panoimage\\919-4\\mask3.bmp");

    //std::vector<std::string> imagePaths;
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\0.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\1.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\2.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\3.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\4.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\5.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\0mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\1mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\2mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\3mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\4mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\5mask.bmp");

    //std::vector<std::string> imagePaths;
    //imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage0.bmp");
    //imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage1.bmp");
    //imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage2.bmp");
    //imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage3.bmp");
    //imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage4.bmp");
    //imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage5.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask0.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask1.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask2.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask3.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask4.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask5.bmp");

    int numImages = imagePaths.size();
    std::vector<cv::Mat> images(numImages), masks(numImages);
    for (int i = 0; i < numImages; i++)
    {
        images[i] = cv::imread(imagePaths[i]);
        masks[i] = cv::imread(maskPaths[i], -1);
    }

    BlendConfig blendConfig;
    blendConfig.setSeamDistanceTransform();
    blendConfig.setBlendMultiBand();
    cv::Mat blend, mask = cv::Mat::zeros(masks[0].size(), CV_8UC1);
    std::vector<cv::Mat> imageGroup(3), maskGroup(3);
    for (int i = 0; i < 3; i++)
    {
        imageGroup[i] = images[i];
        maskGroup[i] = masks[i];
        mask |= masks[i];
    }
    parallelBlend(blendConfig, imageGroup, maskGroup, blend);
    cv::imshow("blend", blend);
    cv::waitKey(0);

    std::vector<unsigned char> lut;
    calcTransform(images[3], masks[3], blend, mask, lut);
    cv::Mat result;
    transform(images[3], result, lut, masks[3]);
    cv::imshow("result", result);
    cv::waitKey(0);

    images[3] = result;
    TilingLinearBlend blender;
    blender.prepare(masks, 100);
    cv::Mat blendResult;
    blender.blend(images, blendResult);
    cv::imshow("blend result", blendResult);
    cv::waitKey(0);

    return 0;
}

void valuesInRange8UC1(const cv::Mat& image, int begInc, int endExc, cv::Mat& mask)
{
    CV_Assert(image.data && image.type() == CV_8UC1);
    int rows = image.rows, cols = image.cols;
    mask.create(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrImage = image.ptr<unsigned char>(i);
        unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            int val = *(ptrImage++);
            *(ptrMask++) = (val >= begInc && val < endExc) ? 255 : 0;
        }
    }
}

struct ImageInfo
{
    ImageInfo() : i(0), fullMean(0), mainMean(0), seamMean(0) {};
    int i;
    cv::Mat fullMask;
    cv::Mat mainMask;
    cv::Mat seamMask;
    cv::Mat gray;
    double fullMean;
    double mainMean;
    double seamMean;
};

struct IntersectionInfo
{
    IntersectionInfo() :
        i(0), j(0),
        numFullNonZero(0), numMainNonZero(0), numSeamNonZero(0),
        iFullMean(0), jFullMean(0), iMainMean(0), jMainMean(0), iSeamMean(0), jSeamMean(0)
    {}
    int i, j;
    cv::Mat fullMask;
    cv::Mat mainMask;
    cv::Mat seamMask;
    int numFullNonZero;
    int numMainNonZero;
    int numSeamNonZero;
    double iFullMean, jFullMean;
    double iMainMean, jMainMean;
    double iSeamMean, jSeamMean;
};

void calcInfo(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, 
    std::vector<ImageInfo>& imageInfos, std::vector<IntersectionInfo>& intersectInfos)
{
    imageInfos.clear();
    intersectInfos.clear();
    if (masks.empty())
        return;

    int size = masks.size();
    int rows = masks[0].rows, cols = masks[0].cols;
    for (int i = 0; i < size; i++)
    {
        CV_Assert(masks[i].data && masks[i].type() == CV_8UC1 &&
            masks[i].rows == rows && masks[i].cols == cols);
        CV_Assert(images[i].data && images[i].type() == CV_8UC3 &&
            images[i].rows == rows && images[i].cols == cols);
    }

    std::vector<cv::Mat> seamMasks;
    getExtendedMasks(masks, 100, seamMasks);

    imageInfos.resize(size);
    for (int i = 0; i < size; i++)
    {
        ImageInfo& imageInfo = imageInfos[i];
        imageInfo.i = i;
        imageInfo.fullMask = masks[i].clone();
        cv::cvtColor(images[i], imageInfo.gray, CV_BGR2GRAY);
        valuesInRange8UC1(imageInfo.gray, 16, 240, imageInfo.mainMask);
        imageInfo.seamMask = seamMasks[i];
        imageInfo.fullMean = cv::mean(imageInfo.gray, imageInfo.fullMask)[0];
        imageInfo.mainMean = cv::mean(imageInfo.gray, imageInfo.mainMask)[0];
        imageInfo.seamMean = cv::mean(imageInfo.gray, imageInfo.seamMask)[0];
    }

    for (int i = 0; i < size - 1; i++)
    {
        for (int j = i + 1; j < size; j++)
        {
            cv::Mat mask = masks[i] & masks[j];
            int numNonZero = cv::countNonZero(mask);
            if (numNonZero)
            {
                intersectInfos.push_back(IntersectionInfo());
                IntersectionInfo& intersect = intersectInfos.back();
                intersect.i = i;
                intersect.j = j;

                intersect.fullMask = mask.clone();
                intersect.numFullNonZero = numNonZero;
                intersect.iFullMean = cv::mean(imageInfos[i].gray, mask)[0];
                intersect.jFullMean = cv::mean(imageInfos[j].gray, mask)[0];

                mask = imageInfos[i].mainMask & imageInfos[j].mainMask;
                intersect.mainMask = mask.clone();
                intersect.numMainNonZero = cv::countNonZero(mask);
                intersect.iMainMean = cv::mean(imageInfos[i].gray, mask)[0];
                intersect.jMainMean = cv::mean(imageInfos[j].gray, mask)[0];

                mask = imageInfos[i].seamMask & imageInfos[j].seamMask;
                intersect.seamMask = mask.clone();
                intersect.numSeamNonZero = cv::countNonZero(mask);
                intersect.iSeamMean = cv::mean(imageInfos[i].gray, mask)[0];
                intersect.jSeamMean = cv::mean(imageInfos[j].gray, mask)[0];
            }
        }
    }
}

void pickAlwaysLargeOrSmall(const std::vector<IntersectionInfo>& intersectInfos, double thresh, 
    std::vector<int>& alwaysSmallIndexes, std::vector<int>& alwaysLargeIndexes)
{
    alwaysSmallIndexes.clear();
    alwaysLargeIndexes.clear();
    int intersectSize = intersectInfos.size();
    if (!intersectSize)
        return;

    int numImages = 0;
    for (int i = 0; i < intersectSize; i++)
    {
        const IntersectionInfo& info = intersectInfos[i];
        numImages = std::max(numImages, std::max(info.i, info.j));
    }
    numImages++;

    std::vector<int> smallCount(numImages, 0);
    std::vector<int> largeCount(numImages, 0);
    std::vector<int> totalCount(numImages, 0);
    for (int i = 0; i < intersectSize; i++)
    {
        const IntersectionInfo& info = intersectInfos[i];
        if (!info.numSeamNonZero)
            continue;

        totalCount[info.i]++;
        totalCount[info.j]++;
        if (info.iSeamMean > info.jSeamMean + thresh)
        {
            smallCount[info.j]++;
            largeCount[info.i]++;
        }
        if (info.jSeamMean > info.iSeamMean + thresh)
        {
            smallCount[info.i]++;
            largeCount[info.j]++;
        }
    }

    for (int i = 0; i < numImages; i++)
    {
        if (smallCount[i] == totalCount[i])
            alwaysSmallIndexes.push_back(i);
        if (largeCount[i] == totalCount[i])
            alwaysLargeIndexes.push_back(i);
    }
}

// main8
int main()
{
    //std::vector<std::string> imagePaths;
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\image0.bmp");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\image1.bmp");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\image2.bmp");
    //imagePaths.push_back("F:\\panoimage\\detuoffice\\image3.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\detuoffice\\mask0.bmp");
    //maskPaths.push_back("F:\\panoimage\\detuoffice\\mask1.bmp");
    //maskPaths.push_back("F:\\panoimage\\detuoffice\\mask2.bmp");
    //maskPaths.push_back("F:\\panoimage\\detuoffice\\mask3.bmp");

    //std::vector<std::string> imagePaths;
    //imagePaths.push_back("F:\\panoimage\\919-4\\image0.bmp");
    //imagePaths.push_back("F:\\panoimage\\919-4\\image1.bmp");
    //imagePaths.push_back("F:\\panoimage\\919-4\\image2.bmp");
    //imagePaths.push_back("F:\\panoimage\\919-4\\image3.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\919-4\\mask0.bmp");
    //maskPaths.push_back("F:\\panoimage\\919-4\\mask1.bmp");
    //maskPaths.push_back("F:\\panoimage\\919-4\\mask2.bmp");
    //maskPaths.push_back("F:\\panoimage\\919-4\\mask3.bmp");

    //std::vector<std::string> imagePaths;
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\0.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\1.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\2.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\3.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\4.bmp");
    //imagePaths.push_back("F:\\panoimage\\zhanxiang\\5.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\0mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\1mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\2mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\3mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\4mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\5mask.bmp");

    std::vector<std::string> imagePaths;
    imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage0.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage1.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage2.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage3.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage4.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\reprojimage5.bmp");
    std::vector<std::string> maskPaths;
    maskPaths.push_back("F:\\panoimage\\changtai\\mask0.bmp");
    maskPaths.push_back("F:\\panoimage\\changtai\\mask1.bmp");
    maskPaths.push_back("F:\\panoimage\\changtai\\mask2.bmp");
    maskPaths.push_back("F:\\panoimage\\changtai\\mask3.bmp");
    maskPaths.push_back("F:\\panoimage\\changtai\\mask4.bmp");
    maskPaths.push_back("F:\\panoimage\\changtai\\mask5.bmp");

    int numImages = imagePaths.size();
    std::vector<cv::Mat> images(numImages), masks(numImages);
    for (int i = 0; i < numImages; i++)
    {
        images[i] = cv::imread(imagePaths[i]);
        masks[i] = cv::imread(maskPaths[i], -1);
    }

    std::vector<ImageInfo> imageInfos;
    std::vector<IntersectionInfo> intersectInfos;
    calcInfo(images, masks, imageInfos, intersectInfos);
    
    std::vector<int> alwaysSmallIndexes, alwaysLargeIndexes;
    pickAlwaysLargeOrSmall(intersectInfos, 10, alwaysSmallIndexes, alwaysLargeIndexes);
    int numSmall = alwaysSmallIndexes.size();
    int numLarge = alwaysLargeIndexes.size();

    std::vector<int> mainIndexes;
    for (int i = 0; i < numImages; i++)
    {
        bool isMain = true;
        for (int j = 0; j < numSmall; j++)
        {
            if (alwaysSmallIndexes[j] == i)
            {
                isMain = false;
                break;
            }
        }
        if (!isMain)
            continue;
        for (int j = 0; j < numLarge; j++)
        {
            if (alwaysLargeIndexes[j] == i)
            {
                isMain = false;
                break;
            }
        }
        if (!isMain)
            continue;
        mainIndexes.push_back(i);
    }

    int numMain = mainIndexes.size();
    std::vector<cv::Mat> mainImages, mainMasks;
    cv::Mat mainMask = cv::Mat::zeros(masks[0].size(), CV_8UC1);
    for (int i = 0; i < numMain; i++)
    {
        mainImages.push_back(images[mainIndexes[i]]);
        mainMasks.push_back(masks[mainIndexes[i]]);
        mainMask |= masks[mainIndexes[i]];
    }

    BlendConfig blendConfig;
    blendConfig.setSeamDistanceTransform();
    blendConfig.setBlendMultiBand();
    cv::Mat mainBlend;
    parallelBlend(blendConfig, mainImages, mainMasks, mainBlend);
    cv::imshow("blend", mainBlend);
    cv::waitKey(0);

    std::vector<cv::Mat> adjustLargeImages(numLarge), adjustSmallImages(numSmall);
    std::vector<unsigned char> lut;
    for (int i = 0; i < numLarge; i++)
    {
        calcTransform(images[alwaysLargeIndexes[i]], masks[alwaysLargeIndexes[i]], mainBlend, mainMask, lut);
        transform(images[alwaysLargeIndexes[i]], adjustLargeImages[i], lut, masks[alwaysLargeIndexes[i]]);
    }
    for (int i = 0; i < numSmall; i++)
    {
        calcTransform(images[alwaysSmallIndexes[i]], masks[alwaysSmallIndexes[i]], mainBlend, mainMask, lut);
        transform(images[alwaysSmallIndexes[i]], adjustSmallImages[i], lut, masks[alwaysSmallIndexes[i]]);
    }

    std::vector<cv::Mat> adjustImages(numImages);
    for (int i = 0; i < numImages; i++)
    {
        int largeIndex = -1, smallIndex = -1;
        for (int k = 0; k < numLarge; k++)
        {
            if (alwaysLargeIndexes[k] == i)
            {
                largeIndex = k;
                break;
            }
        }
        for (int k = 0; k < numSmall; k++)
        {
            if (alwaysSmallIndexes[k] == i)
            {
                smallIndex = k;
                break;
            }
        }
        if (largeIndex >= 0)
            adjustImages[i] = adjustLargeImages[largeIndex];
        else if (smallIndex >= 0)
            adjustImages[i] = adjustSmallImages[smallIndex];
        else
            adjustImages[i] = images[i];
    }

    cv::Mat result;
    TilingLinearBlend blender;
    blender.prepare(masks, 100);    
    blender.blend(adjustImages, result);
    cv::imshow("linear blend", result);

    parallelBlend(blendConfig, adjustImages, masks, result);
    cv::imshow("multiband blend", result);
    cv::waitKey(0);

    std::vector<cv::Mat> tintAdjustImages;
    tintAdjust(adjustImages, masks, tintAdjustImages);
    for (int i = 0; i < numImages; i++)
    {
        cv::imshow("tint", tintAdjustImages[i]);
        cv::waitKey(0);
    }

    blender.blend(tintAdjustImages, result);
    cv::imshow("tint linear blend", result);
    cv::waitKey(0);

    return 0;
}