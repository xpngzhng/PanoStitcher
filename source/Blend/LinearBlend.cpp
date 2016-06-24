#include "ZBlendAlgo.h"
#include "ZBlend.h"
#include "Timer.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>

#define NEED_MAIN 0

static const int UNIT_SHIFT = 16;
static const int UNIT = 1 << UNIT_SHIFT;
static const float eps = 1.0F / UNIT;

static void calcWeights(const std::vector<cv::Mat>& dists, std::vector<cv::Mat>& weights)
{
    int numImages = dists.size();
    int rows = dists[0].rows, cols = dists[0].cols;

    weights.resize(numImages);
    for (int i = 0; i < numImages; i++)
        weights[i].create(rows, cols, CV_32SC1);

    std::vector<const unsigned char*> ptrDistVector(numImages);
    const unsigned char** ptrDist = &ptrDistVector[0];
    std::vector<int*> ptrWeightVector(numImages);
    int** ptrWeight = &ptrWeightVector[0];
    for (int i = 0; i < rows; i++)
    {
        for (int k = 0; k < numImages; k++)
        {
            ptrDist[k] = dists[k].ptr<unsigned char>(i);
            ptrWeight[k] = weights[k].ptr<int>(i);
        }
        for (int j = 0; j < cols; j++)
        {
            float sum = 0;
            for (int k = 0; k < numImages; k++)
                sum += ptrDist[k][j];
            sum = fabs(sum) <= FLT_MIN ? 0 : 1.0F / sum;
            int intSum = 0;
            for (int k = 0; k < numImages; k++)
            {
                // WE MUST ENSURE THAT intSum >= UNIT, 
                // SO WE ADD 1 AFTER SCALING.
                int weight = ptrDist[k][j] * sum * UNIT + 1;
                intSum += weight;
                ptrWeight[k][j] = weight;
            }
        }
    }
}

void getWeightsLinearBlend(const std::vector<cv::Mat>& masks, int radius, std::vector<cv::Mat>& weights)
{
    int numImages = masks.size();
    std::vector<cv::Mat> uniqueMasks(numImages);
    getNonIntersectingMasks(masks, uniqueMasks);

    std::vector<cv::Mat> dists(numImages);
    cv::Size blurSize(radius * 2 + 1, radius * 2 + 1);
    double sigma = radius / 3.0;
    for (int i = 0; i < numImages; i++)
    {
        cv::GaussianBlur(uniqueMasks[i], dists[i], blurSize, sigma, sigma);
        cv::bitwise_and(dists[i], masks[i], dists[i]);
        //cv::imshow("dist", dists[i]);
        //cv::waitKey(0);
    }

    calcWeights(dists, weights);
}

static void calcWeights32F(const std::vector<cv::Mat>& dists, std::vector<cv::Mat>& weights)
{
    int numImages = dists.size();
    int rows = dists[0].rows, cols = dists[0].cols;

    weights.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        weights[i].create(rows, cols, CV_32FC1);
        weights[i].setTo(0);
    }        

    std::vector<const unsigned char*> ptrDistVector(numImages);
    const unsigned char** ptrDist = &ptrDistVector[0];
    std::vector<float*> ptrWeightVector(numImages);
    float** ptrWeight = &ptrWeightVector[0];
    for (int i = 0; i < rows; i++)
    {
        for (int k = 0; k < numImages; k++)
        {
            ptrDist[k] = dists[k].ptr<unsigned char>(i);
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

void getWeightsLinearBlend32F(const std::vector<cv::Mat>& masks, int radius, std::vector<cv::Mat>& weights)
{
    int numImages = masks.size();
    std::vector<cv::Mat> uniqueMasks(numImages);
    getNonIntersectingMasks(masks, uniqueMasks);

    std::vector<cv::Mat> dists(numImages);
    cv::Size blurSize(radius * 2 + 1, radius * 2 + 1);
    double sigma = radius / 3.0;
    for (int i = 0; i < numImages; i++)
    {
        cv::GaussianBlur(uniqueMasks[i], dists[i], blurSize, sigma, sigma);
        cv::bitwise_and(dists[i], masks[i], dists[i]);
        //cv::imshow("dist", dists[i]);
        //cv::waitKey(0);
    }

    calcWeights32F(dists, weights);
}

static bool findExternContours(const cv::Mat& mask, std::vector<std::vector<cv::Point> >& contours)
{
    if (!mask.data || mask.type() != CV_8UC1)
    {
        contours.clear();
        return false;
    }

    int rows = mask.rows, cols = mask.cols;
    int pad = 4;
    if (cv::countNonZero(mask.row(0)) || cv::countNonZero(mask.row(rows - 1)) ||
        cv::countNonZero(mask.col(0)) || cv::countNonZero(mask.col(cols - 1)))
    {
        cv::Mat extendMask(rows + 2 * pad, cols + 2 * pad, CV_8UC1);
        extendMask.setTo(0);
        cv::Mat roi = extendMask(cv::Rect(pad, pad, cols, rows));
        mask.copyTo(roi);
        cv::findContours(extendMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        int numCountors = contours.size();
        cv::Point offset(pad, pad);
        for (int i = 0; i < numCountors; i++)
        {
            int len = contours[i].size();
            for (int j = 0; j < len; j++)
                contours[i][j] -= offset;
        }
    }
    else
    {
        cv::Mat nonExtendMask;
        mask.copyTo(nonExtendMask);
        cv::findContours(nonExtendMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    }

    return true;
}

int getMaxRadius(const std::vector<cv::Mat>& masks, const std::vector<cv::Mat>& uniqueMasks, 
    const std::vector<cv::Mat>& dists, int distBound)
{
    if (!checkSize(masks) || !checkSize(uniqueMasks) || !checkSize(dists) ||
        !checkType(masks, CV_8UC1) || !checkType(uniqueMasks, CV_8UC1) || !checkType(dists, CV_32FC1))
        return -1;

    int numImages = masks.size();
    if (uniqueMasks.size() != numImages || dists.size() != numImages)
        return -1;

    //for (int i = 0; i < numImages; i++)
    //{
    //    cv::imshow("mask", masks[i]);
    //    cv::imshow("umask", uniqueMasks[i]);
    //    cv::imshow("diff", masks[i] - uniqueMasks[i]);
    //    cv::waitKey(0);
    //}

    float maxRadius = std::max(masks[0].rows, masks[0].cols);
    std::vector<std::vector<std::vector<cv::Point> > > contours(numImages);
    for (int i = 0; i < numImages; i++)
    {
        findExternContours(uniqueMasks[i], contours[i]);
    }
    int u;
    for (u = distBound; u > 1; u -= 3)
    {
        bool success = true;
        for (int i = 0; i < numImages; i++)
        {
            float currMaxRadius = 0;
            int numContours = contours[i].size();
            for (int j = 0; j < numContours; j++)
            {
                int numPts = contours[i][j].size();
                for (int k = 0; k < numPts; k++)
                {
                    cv::Point p = contours[i][j][k];
                    if (abs(p.x) + abs(p.y) > u && dists[i].at<float>(p) < u)
                    {
                        success = false;
                        //printf("fail at [%d][%d][%d] x = %d, y = %d, dist = %f\n",
                        //    i, j, k, p.x, p.y, dists[i].at<float>(p)); 
                        break;
                    }
                }
                if (!success)
                    break;
            }
            if (!success)
                break;
        }
        if (success)
            break;
    }
    return u;
}

void getWeightsLinearBlendBoundedRadius(const std::vector<cv::Mat>& masks, int maxRadius, std::vector<cv::Mat>& weights)
{
    int numImages = masks.size();
    std::vector<cv::Mat> uniqueMasks(numImages), dists(numImages);
    getNonIntersectingMasks(masks, uniqueMasks);
    for (int i = 0; i < numImages; i++)
        cv::distanceTransform(masks[i], dists[i], CV_DIST_L2, 3);

    int radius = getMaxRadius(masks, uniqueMasks, dists, maxRadius);
    if (radius <= 1)
        radius = 1;
    else
        radius -= 1;
    //printf("radius = %d\n", radius);
    cv::Size blurSize(radius * 2 + 1, radius * 2 + 1);
    double sigma = radius / 3.0;
    for (int i = 0; i < numImages; i++)
    {
        cv::GaussianBlur(uniqueMasks[i], dists[i], blurSize, sigma, sigma);
        cv::bitwise_and(dists[i], masks[i], dists[i]);
        //cv::imshow("dist", dists[i]);
        //cv::waitKey(0);
    }

    calcWeights(dists, weights);
}

void getWeightsLinearBlendBoundedRadius32F(const std::vector<cv::Mat>& masks, int maxRadius, std::vector<cv::Mat>& weights)
{
    int numImages = masks.size();
    std::vector<cv::Mat> uniqueMasks(numImages), dists(numImages);
    getNonIntersectingMasks(masks, uniqueMasks);
    for (int i = 0; i < numImages; i++)
        cv::distanceTransform(masks[i], dists[i], CV_DIST_L2, 3);

    int radius = getMaxRadius(masks, uniqueMasks, dists, maxRadius);
    if (radius <= 1)
        radius = 1;
    else
        radius -= 1;
    //printf("radius = %d\n", radius);
    cv::Size blurSize(radius * 2 + 1, radius * 2 + 1);
    double sigma = radius / 3.0;
    for (int i = 0; i < numImages; i++)
    {
        cv::GaussianBlur(uniqueMasks[i], dists[i], blurSize, sigma, sigma);
        cv::bitwise_and(dists[i], masks[i], dists[i]);
        //cv::imshow("dist", dists[i]);
        //cv::waitKey(0);
    }

    calcWeights32F(dists, weights);
}

static void accumulate(const cv::Mat& image, const cv::Mat& weight, cv::Mat& accumImage)
{
    CV_Assert(image.data && image.type() == CV_8UC3 &&
        weight.data && weight.type() == CV_32SC1 &&
        accumImage.data && accumImage.type() == CV_32SC3);

    cv::Size size = image.size();
    CV_Assert(weight.size() == size && accumImage.size() == size);

    int rows = image.rows, cols = image.cols;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrImage = image.ptr<unsigned char>(i);
        const int* ptrWeight = weight.ptr<int>(i);
        int* ptrAccumImage = accumImage.ptr<int>(i);
        for (int j = 0; j < cols; j++)
        {
            int weight = ptrWeight[0];
            ptrAccumImage[0] += ptrImage[0] * weight;
            ptrAccumImage[1] += ptrImage[1] * weight;
            ptrAccumImage[2] += ptrImage[2] * weight;

            ptrImage += 3;
            ptrAccumImage += 3;
            ptrWeight++;
        }
    }
}

static void normalize(cv::Mat& accumImage)
{
    CV_Assert(accumImage.data && accumImage.type() == CV_32SC3);

    int rows = accumImage.rows, cols = accumImage.cols;
    for (int i = 0; i < rows; i++)
    {
        int* ptrAccumImage = accumImage.ptr<int>(i);
        for (int j = 0; j < cols; j++)
        {
            ptrAccumImage[0] >>= UNIT_SHIFT;
            ptrAccumImage[1] >>= UNIT_SHIFT;
            ptrAccumImage[2] >>= UNIT_SHIFT;
            ptrAccumImage += 3;
        }
    }
}

inline int clamp0255(int val)
{
    return val < 0 ? 0 : (val < 255 ? val : 255);
}

static void divide(const cv::Mat& src, const cv::Mat& alpha, cv::Mat& dst)
{
    CV_Assert(src.data && src.type() == CV_32FC3 &&
        alpha.data && alpha.type() == CV_32FC1 &&
        src.size() == alpha.size());

    int rows = src.rows, cols = src.cols;
    dst.create(rows, cols, CV_8UC3);
    for (int i = 0; i < rows; i++)
    {
        const float* ptrSrc = src.ptr<float>(i);
        const float* ptrAlp = alpha.ptr<float>(i);
        unsigned char* ptrDst = dst.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            float alp = ptrAlp[j];
            if (alp < std::numeric_limits<float>::epsilon())
            {
                ptrDst[j * 3] = 0;
                ptrDst[j * 3 + 1] = 0;
                ptrDst[j * 3 + 2] = 0;
            }
            else
            {
                ptrDst[j * 3] = clamp0255(ptrSrc[j * 3] / alp * 255);
                ptrDst[j * 3 + 1] = clamp0255(ptrSrc[j * 3 + 1] / alp * 255);
                ptrDst[j * 3 + 2] = clamp0255(ptrSrc[j * 3 + 2] / alp * 255);
            }
        }
    }
}

void linearBlend(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& alpha1, const cv::Mat& alpha2,
    cv::Mat& mask1, const cv::Mat& mask2, int radius, cv::Mat& result)
{
    CV_Assert(image1.data && image1.type() == CV_8UC3 &&
        image2.data && image2.type() == CV_8UC3 &&
        alpha1.data && alpha1.type() == CV_8UC1 &&
        alpha2.data && alpha2.type() == CV_8UC1 &&
        mask1.data && mask1.type() == CV_8UC1 &&
        mask2.data && mask2.type() == CV_8UC1);
    cv::Size size = image1.size();
    CV_Assert(size == image2.size() && size == mask1.size() && size == mask2.size());

    std::vector<cv::Mat> dists(2);
    cv::Size blurSize(radius * 2 + 1, radius * 2 + 1);
    double sigma = radius / 3.0;
    cv::GaussianBlur(mask1, dists[0], blurSize, sigma, sigma);
    cv::GaussianBlur(mask2, dists[1], blurSize, sigma, sigma);
    cv::bitwise_and(dists[0], alpha1, dists[0]);
    cv::bitwise_and(dists[1], alpha2, dists[1]);
    //cv::imshow("mask0", mask1);
    //cv::imshow("mask1", mask2);
    //cv::imshow("dist0", dists[0]);
    //cv::imshow("dist1", dists[1]);
    //cv::waitKey(0);

    //cv::Mat image1Flt, image2Flt, alpha1Flt, alpha2Flt;
    //cv::Mat image1BlurFlt, image2BlurFlt, alpha1BlurFlt, alpha2BlurFlt;
    //image1.convertTo(image1Flt, CV_32F);
    //image2.convertTo(image2Flt, CV_32F);
    //alpha1.convertTo(alpha1Flt, CV_32F);
    //alpha2.convertTo(alpha2Flt, CV_32F);
    //cv::GaussianBlur(image1Flt, image1BlurFlt, blurSize, sigma, sigma);
    //cv::GaussianBlur(image2Flt, image2BlurFlt, blurSize, sigma, sigma);
    //cv::GaussianBlur(alpha1Flt, alpha1BlurFlt, blurSize, sigma, sigma);
    //cv::GaussianBlur(alpha2Flt, alpha2BlurFlt, blurSize, sigma, sigma);

    //cv::Mat image1New, image2New;
    //divide(image1BlurFlt, alpha1BlurFlt, image1New);
    //divide(image2BlurFlt, alpha2BlurFlt, image2New);
    //image1.copyTo(image1New, alpha1);
    //image2.copyTo(image2New, alpha2);
    //cv::imshow("image1", image1New);
    //cv::imshow("image2", image2New);
    //cv::waitKey(0);

    std::vector<cv::Mat> weights;
    calcWeights(dists, weights);
    dists.clear();

    cv::Mat accumImage = cv::Mat::zeros(size, CV_32SC3);
    //accumulate(image1New, weights[0], accumImage);
    //accumulate(image2New, weights[1], accumImage);
    accumulate(image1, weights[0], accumImage);
    accumulate(image2, weights[1], accumImage);
    normalize(accumImage);
    accumImage.convertTo(result, CV_8U);    
}

void linearBlend(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& alphas,
    const std::vector<cv::Mat>& masks, int radius, cv::Mat& result)
{
    CV_Assert(checkSize(images) && checkSize(alphas) && checkSize(masks));
    CV_Assert(images[0].size() == alphas[0].size() && images[0].size() == masks[0].size());
    CV_Assert(checkType(images, CV_8UC3) && checkType(alphas, CV_8UC1) && checkType(masks, CV_8UC1));

    int numImages = images.size();
    std::vector<cv::Mat> dists(numImages);
    cv::Size blurSize(radius * 2 + 1, radius * 2 + 1);
    double sigma = radius / 3.0;
    for (int i = 0; i < numImages; i++)
    {
        cv::GaussianBlur(masks[i], dists[i], blurSize, sigma, sigma);
        cv::bitwise_and(dists[i], alphas[i], dists[i]);
    }

    std::vector<cv::Mat> weights;
    calcWeights(dists, weights);
    dists.clear();

    cv::Mat accumImage = cv::Mat::zeros(images[0].size(), CV_32SC3);
    for (int i = 0; i < numImages; i++)
        accumulate(images[i], weights[i], accumImage);
    normalize(accumImage);
    accumImage.convertTo(result, CV_8U);
}

bool TilingLinearBlend::prepare(const std::vector<cv::Mat>& masks, int radius)
{
    success = false;
    if (masks.empty())
        return false;
    if (radius < 1)
        return false;

    int currNumMasks = masks.size();
    int currRows = masks[0].rows, currCols = masks[0].cols;
    for (int i = 0; i < currNumMasks; i++)
    {
        if (!masks[i].data || masks[i].type() != CV_8UC1 ||
            masks[i].rows != currRows || masks[i].cols != currCols)
            return false;
    }
    numImages = currNumMasks;
    rows = currRows;
    cols = currCols;

    getWeightsLinearBlend(masks, radius, weights);
    
    success = true;
    return true;
}

void TilingLinearBlend::blend(const std::vector<cv::Mat>& images, cv::Mat& blendImage) const
{
    if (!success)
        return;

    CV_Assert(images.size() == numImages);

    cv::Mat accumImage = cv::Mat::zeros(rows, cols, CV_32SC3);
    for (int i = 0; i < numImages; i++)
        accumulate(images[i], weights[i], accumImage);
    normalize(accumImage);
    accumImage.convertTo(blendImage, CV_8U);
}

#if NEED_MAIN

void compensate(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results);
void compensate3(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results);

int main()
{
    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\panoimage\\changtai\\1.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\2.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\3.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\4.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\5.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\6.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_1.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_2.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_3.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_4.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_5.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_6.bmp");

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\pano0000.tif");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\pano0001.tif");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\pano0002.tif");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\panomask0000.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\panomask0001.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\panomask0002.bmp");

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\0.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\1.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\2.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\0mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\1mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\2mask.bmp");

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\0.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\1.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\2.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\3.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\4.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\5.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\6.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\7.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\8mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\9mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\10mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\11mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\12mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\13mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\14mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\15mask.bmp");

    std::vector<std::string> contentPaths;
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\0.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\1.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\2.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\3.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\4.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\5.bmp");
    std::vector<std::string> maskPaths;
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\0mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\1mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\2mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\3mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\4mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\5mask.bmp");

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0000.tif");
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0001.tif");
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0002.tif");
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0003.tif");
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0004.tif");
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0005.tif");
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0006.tif");
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0007.tif");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0000.bmp");
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0001.bmp");
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0002.bmp");
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0003.bmp");
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0004.bmp");
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0005.bmp");
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0006.bmp");
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0007.bmp");

    ztool::Timer timer;

    int numImages = contentPaths.size();
    std::vector<cv::Mat> images, masks;
    cv::Size imageSize;
    getImagesAndMasks(contentPaths, maskPaths, imageSize, images, masks);

    //for (int i = 0; i < numImages; i++)
    //{
    //    cv::Scalar meanVal = cv::mean(images[i], masks[i]);
    //    printf("mean b = %f, g = %f, r = %f, gray = %f\n", 
    //        meanVal[0], meanVal[1], meanVal[2], 0.114 * meanVal[0] + 0.587 * meanVal[1] + 0.299 * meanVal[2]);
    //}

    std::vector<cv::Mat> results;
    compensate3(images, masks, results);
    for (int i = 0; i < numImages; i++)
    {
        cv::imshow("image", results[i]);
        //char buf[128];
        //sprintf(buf, "imagecompensate%d.bmp", i);
        //cv::imwrite(buf, results[i]);
        cv::waitKey(0);
    }
    //return 0;

    //compensate(images, masks, results);

    //for (int i = 0; i < numImages; i++)
    //{
    //    cv::Mat dist;
    //    timer.start();
    //    cv::distanceTransform(masks[i], dist, CV_DIST_L1, 3);
    //    timer.end();
    //    cv::imshow("mask", masks[i]);
    //    showFloatMat("dist", dist);
    //    cv::waitKey(0);
    //}
    //images[4] *= 1.5;
    cv::Mat result, resultMask;
    //timer.start();
    //featherBlend(results, masks, result);
    //cv::imshow("tiling result", result);
    //cv::waitKey(0);
    //featherBlendProgressively(results, masks, result, resultMask);
    //timer.end();
    //cv::imwrite("blend1.bmp", result);
    //printf("time = %f\n", timer.elapse());
    //cv::imshow("result", result);
    TilingLinearBlend blender;
    timer.start();
    blender.prepare(masks, 150);
    timer.end();
    printf("prepare time = %f\n", timer.elapse());
    timer.start();
    blender.blend(images, result);
    timer.end();
    printf("blend time = %f\n", timer.elapse());
    cv::imshow("orig image result", result);
    timer.start();
    blender.blend(results, result);
    timer.end();
    printf("blend time = %f\n", timer.elapse());
    cv::imshow("compensate image result", result);
    //cv::imwrite("compensateresult.bmp", result);
    cv::waitKey(0);

    return 0;
}

#endif

#undef NEED_MAIN