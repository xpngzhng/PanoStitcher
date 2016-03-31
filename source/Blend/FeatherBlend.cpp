#include "ZBlendAlgo.h"
#include "ZBlend.h"
#include "Timer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>

#define NEED_MAIN 0

static void showFloatMat(const std::string& winName, const cv::Mat& image)
{
    CV_Assert(image.data && image.type() == CV_32FC1);
    
    double minVal, maxVal;
    cv::minMaxLoc(image, &minVal, &maxVal);
    printf("%s, minVal = %f, maxVal = %f\n", winName.c_str(), minVal, maxVal);
    cv::imshow(winName, image / maxVal);
    //cv::Mat img;
    //cv::Mat fltImg = image * (255.F / maxVal);
    //fltImg.convertTo(img, CV_8U);
    //cv::imwrite("weightsum.bmp", img);
}

static void accumulate(const cv::Mat& image, const cv::Mat& weight, cv::Mat& accumImage, cv::Mat& accumWeight)
{
    CV_Assert(image.data && image.type() == CV_8UC3 &&
        weight.data && weight.type() == CV_32FC1 && 
        accumImage.data && accumImage.type() == CV_32FC3 &&
        accumWeight.data && accumWeight.type() == CV_32FC1);

    cv::Size size = image.size();
    CV_Assert(weight.size() == size && accumImage.size() == size && accumWeight.size() == size);

    int rows = image.rows, cols = image.cols;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrImage = image.ptr<unsigned char>(i);
        const float* ptrWeight = weight.ptr<float>(i);
        float* ptrAccumImage = accumImage.ptr<float>(i);
        float* ptrAccumWeight = accumWeight.ptr<float>(i);
        for (int j = 0; j < cols; j++)
        {
            float weight = ptrWeight[0];
            ptrAccumImage[0] += ptrImage[0] * weight;
            ptrAccumImage[1] += ptrImage[1] * weight;
            ptrAccumImage[2] += ptrImage[2] * weight;
            ptrAccumWeight[0] += weight;

            ptrImage += 3;
            ptrAccumImage += 3;
            ptrWeight++;
            ptrAccumWeight++;
        }
    }
}

static void normalize(cv::Mat& accumImage, const cv::Mat& accumWeight)
{
    CV_Assert(accumImage.data && accumImage.type() == CV_32FC3 &&
        accumWeight.data && accumWeight.type() == CV_32FC1 &&
        accumImage.size() == accumWeight.size());

    int rows = accumImage.rows, cols = accumImage.cols;
    for (int i = 0; i < rows; i++)
    {
        float* ptrAccumImage = accumImage.ptr<float>(i);
        const float* ptrAccumWeight = accumWeight.ptr<float>(i);
        for (int j = 0; j < cols; j++)
        {
            float weight = 1.0F / (ptrAccumWeight[0] + FLT_MIN);
            ptrAccumImage[0] *= weight;
            ptrAccumImage[1] *= weight;
            ptrAccumImage[2] *= weight;
            ptrAccumImage += 3;
            ptrAccumWeight++;
        }
    }
}

void distanceTransformFeatherBlend(const cv::Mat& mask, cv::Mat& dist)
{
    CV_Assert(mask.data && mask.type() == CV_8UC1);

    int width = mask.cols, height = mask.rows;
    cv::Mat largeMask = cv::Mat::zeros(height, width * 2, CV_8UC1), largeDist;
    cv::Mat largeMaskROI(largeMask, cv::Rect(width / 2, 0, width, height));
    mask.copyTo(largeMaskROI);
    horiCircularRepeat(largeMask, -width / 2, width);
    cv::distanceTransform(largeMask, largeDist, CV_DIST_L1, 3);
    largeDist(cv::Rect(width / 2, 0, width, height)).copyTo(dist);
}

static void asymptoticExpSaturateScale(cv::Mat& mat, float halfLife, float maxVal)
{
    CV_Assert(mat.data && mat.type() == CV_32FC1);
    float scale = -log(2.0F) / halfLife;
    printf("scale = %f\n", scale);
    int rows = mat.rows, cols = mat.cols;
    float actualMaxVal = 0;
    for (int i = 0; i < rows; i++)
    {
        float* ptr = mat.ptr<float>(i);
        for (int j = 0; j < cols; j++)
        {
            *ptr = (1.0F - exp(*ptr * scale)) * maxVal;
            actualMaxVal = std::max(*ptr, actualMaxVal);
            ptr++;
        }
    }
    printf("actual max val = %f\n", actualMaxVal);
}

void featherBlend(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, cv::Mat& result)
{
    CV_Assert(!images.empty() && images.size() == masks.size());

    int size = images.size();
    cv::Mat accumImage = cv::Mat::zeros(images[0].size(), CV_32FC3);
    cv::Mat accumWeight = cv::Mat::zeros(images[0].size(), CV_32FC1);
    cv::Mat dist;
    ztool::InterruptTimer timer;
    for (int i = 0; i < size; i++)
    {
        //cv::distanceTransform(masks[i], dist, CV_DIST_L1, 3);
        timer.start();
        distanceTransformFeatherBlend(masks[i], dist);
        timer.end();
        //cv::pow(dist, 2, dist);
        //cv::threshold(dist, dist, 50, 50, cv::THRESH_TRUNC);
        //asymptoticExpSaturateScale(dist, 25, 50);
        double minVal, maxVal;
        cv::minMaxLoc(dist, &minVal, &maxVal);
        dist *= (1.0 / maxVal);
        accumulate(images[i], dist, accumImage, accumWeight);
    }
    //printf("dist trans time = %f\n", timer.elapse());
    normalize(accumImage, accumWeight);
    //showFloatMat("accum weight", accumWeight);
    accumImage.convertTo(result, CV_8U);
    //cv::imshow("result", result);
}

void featherBlend(const cv::Mat& image, const cv::Mat& mask, cv::Mat& blendImage, cv::Mat& blendMask)
{
    CV_Assert(image.data && image.type() == CV_8UC3 &&
        mask.data && mask.type() == CV_8UC1 &&
        blendImage.data && blendImage.type() == CV_8UC3 &&
        blendMask.data && blendMask.type() == CV_8UC1);

    cv::Size imageSize = image.size();
    CV_Assert(mask.size() == imageSize && 
        blendImage.size() == imageSize && 
        blendMask.size() == imageSize);

    cv::Rect imageRect(0, 0, imageSize.width, imageSize.height);
    // If blendMask is all zero, blendRect is invalid
    cv::Rect blendRect = getNonZeroBoundingRect(blendMask);
    
    cv::Mat intersectMask = mask & blendMask;
    cv::Rect intersectRect = getNonZeroBoundingRect(intersectMask);
    int intersectMaskNonZero = cv::countNonZero(intersectMask);

    if (intersectMaskNonZero == 0)
    {
#if WRITE_CONSOLE
        printf("curr mask does not intersect blend mask, copy curr image and mask\n");
#endif
        cv::Rect currNonZeroRect = getNonZeroBoundingRect(mask);
        cv::Mat currNonZeroMask(mask, currNonZeroRect);
        cv::Mat blendImageROI(blendImage, currNonZeroRect);
        image(currNonZeroRect).copyTo(blendImageROI, currNonZeroMask);
        currNonZeroMask.copyTo(blendMask(currNonZeroRect), currNonZeroMask);
        return;
    }

    int blendMaskNonZero = cv::countNonZero(blendMask);
    int currMaskNonZero = cv::countNonZero(mask);    
    if (intersectMaskNonZero == currMaskNonZero)
    {
#if WRITE_CONSOLE
        printf("curr mask totally inside blend mask, return\n");
#endif
        return;
    }
    if (intersectMaskNonZero == blendMaskNonZero)
    {
        if (currMaskNonZero == blendMaskNonZero)
        {
#if WRITE_CONSOLE
            printf("blend mask equals curr mask, return");
#endif
        }
        else
        {
#if WRITE_CONSOLE
            printf("blend mask totally inside curr mask, copy curr image and mask, then return");
#endif
            cv::Rect currNonZeroRect = getNonZeroBoundingRect(mask);
            cv::Mat currNonZeroMask(mask, currNonZeroRect);
            cv::Mat blendImageROI(blendImage, currNonZeroRect);
            image(currNonZeroRect).copyTo(blendImageROI, currNonZeroMask);
            cv::Mat blendMaskROI(blendMask, currNonZeroRect);
            currNonZeroMask.copyTo(blendMaskROI, currNonZeroMask);
        }
        return;
    }

    std::vector<cv::Mat> images(2), masks(2);
    images[0] = image;
    images[1] = blendImage;
    masks[0] = mask;
    masks[1] = blendMask;    
#if WRITE_CONSOLE
    timer.start();
#endif
    featherBlend(images, masks, blendImage);
#if WRITE_CONSOLE
    timer.end();
    printf("findSeam time elapse: %f\n", timer.elapse());
#endif
    blendMask |= mask;
    cv::waitKey(0);
}

void featherBlendProgressively(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, 
    cv::Mat& blendImage, cv::Mat& blendMask)
{
    if (!checkSize(images, masks))
        return;   

    cv::Size imageSize = images[0].size();
    blendImage.create(imageSize, CV_8UC3);
    blendImage.setTo(0);
    blendMask.create(imageSize, CV_8UC1);
    blendMask.setTo(0);

    int numImages = images.size();
    for (int i = 0; i < numImages; i++)
    {
        //printf("i = %d\n", i);
        featherBlend(images[i], masks[i], blendImage, blendMask);
    }
}

static const int UNIT_SHIFT = 16;
static const int UNIT = 1 << UNIT_SHIFT;
static const float eps = 1.0F / UNIT;

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

void calcWeightsFeatherBlend(const std::vector<cv::Mat>& dists, std::vector<cv::Mat>& weights)
{
    int numImages = dists.size();
    int rows = dists[0].rows, cols = dists[0].cols;

    weights.resize(numImages);
    for (int i = 0; i < numImages; i++)
        weights[i].create(rows, cols, CV_32SC1);

    std::vector<const float*> ptrDistVector(numImages);
    const float** ptrDist = &ptrDistVector[0];
    std::vector<int*> ptrWeightVector(numImages);
    int** ptrWeight = &ptrWeightVector[0];
    for (int i = 0; i < rows; i++)
    {
        for (int k = 0; k < numImages; k++)
        {
            ptrDist[k] = dists[k].ptr<float>(i);
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

class TilingFeatherBlend
{
public:
    TilingFeatherBlend() : numImages(0), rows(0), cols(0), success(false) {}
    bool prepare(const std::vector<cv::Mat>& masks);
    bool prepareWithDist(const std::vector<cv::Mat>& masks);
    void blend(const std::vector<cv::Mat>& images, cv::Mat& blendImage) const;
private:
    std::vector<cv::Mat> weights;
    int numImages;
    int rows, cols;
    bool success;
};

bool TilingFeatherBlend::prepare(const std::vector<cv::Mat>& masks)
{
    success = false;
    if (masks.empty())
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

    ztool::Timer timer;
    std::vector<cv::Mat> dists(numImages);
    for (int i = 0; i < numImages; i++)
        distanceTransformFeatherBlend(masks[i], dists[i]);
    timer.end();
    //printf("dist trans time = %f\n", timer.elapse());

    calcWeightsFeatherBlend(dists, weights);

    success = true;
    return true;
}

bool TilingFeatherBlend::prepareWithDist(const std::vector<cv::Mat>& masks)
{
    success = false;
    if (masks.empty())
        return false;

    int currNumMasks = masks.size();
    int currRows = masks[0].rows, currCols = masks[0].cols;
    for (int i = 0; i < currNumMasks; i++)
    {
        if (!masks[i].data || masks[i].type() != CV_32FC1 ||
            masks[i].rows != currRows || masks[i].cols != currCols)
            return false;
    }
    numImages = currNumMasks;
    rows = currRows;
    cols = currCols;

    ztool::Timer timer;
    std::vector<cv::Mat> dists(numImages);
    for (int i = 0; i < numImages; i++)
    {
        dists[i] = masks[i].clone();
        asymptoticExpSaturateScale(dists[i], 10000, 100);
        showFloatMat("scaled", dists[i]);
        cv::waitKey(0);
    }
    timer.end();
    //printf("dist trans time = %f\n", timer.elapse());

    calcWeightsFeatherBlend(dists, weights);
    success = true;
    return true;
}

void TilingFeatherBlend::blend(const std::vector<cv::Mat>& images, cv::Mat& blendImage) const
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

int main()
{
    //const int width = 16, height = 16;
    //unsigned char maskData[height][width] = {0};
    //float distData[height][width];
    //for (int i = 1; i < height - 1; i++)
    //{
    //    for (int j = 1; j < width - 1; j++)
    //    {
    //        maskData[i][j] = 255;
    //    }
    //}
    //cv::Mat mask(height, width, CV_8UC1, maskData);
    //cv::Mat dist(height, width, CV_32FC1, distData);
    //cv::distanceTransform(mask, dist, CV_DIST_L1, 3);
    //std::cout << dist << std::endl;

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

    //std::vector<cv::Mat> origMasks(numImages);
    //for (int i = 0; i < numImages; i++)
    //    masks[i].copyTo(origMasks[i]);
    //cv::Mat dist, currDist;
    //cv::distanceTransform(masks[0], dist, CV_DIST_L1, 3);
    //cv::Mat mask = masks[0].clone();    
    //for (int i = 1; i < numImages; i++)
    //{
    //    cv::distanceTransform(mask, dist, CV_DIST_L1, 3);
    //    cv::distanceTransform(masks[i], currDist, CV_DIST_L1, 3);
    //    masks[i] = currDist > dist;
    //    cv::imshow("curr mask", masks[i]);
    //    cv::waitKey(0);
    //    for (int j = 0; j < i; j++)
    //        masks[j].setTo(0, masks[i]);
    //    mask |= masks[i];
    //}
    //cv::Mat ex = cv::Mat::zeros(imageSize, CV_8UC1);
    //for (int i = 0; i < numImages; i++)
    //{
    //    cv::bitwise_xor(ex, masks[i], ex);
    //    cv::Mat diff(imageSize, CV_8UC1);
    //    diff.setTo(0);
    //    diff.setTo(128, origMasks[i]);
    //    diff.setTo(255, masks[i]);
    //    cv::imshow("diff", diff);
    //    cv::imshow("mask", masks[i]);
    //    cv::waitKey(0);
    //}
    //cv::imshow("ex", ex);
    //cv::waitKey(0);
    //return 0;

    //for (int i = 0; i < numImages; i++)
    //{
    //    cv::Scalar meanVal = cv::mean(images[i], masks[i]);
    //    printf("mean b = %f, g = %f, r = %f, gray = %f\n", 
    //        meanVal[0], meanVal[1], meanVal[2], 0.114 * meanVal[0] + 0.587 * meanVal[1] + 0.299 * meanVal[2]);
    //}

    std::vector<cv::Mat> nonIntMasks;
    getNonIntersectingMasks(masks, nonIntMasks);
    //for (int i = 0; i < numImages; i++)
    //{
    //    cv::Mat show = cv::Mat::zeros(imageSize, CV_8UC1);
    //    show.setTo(128, masks[i]);
    //    show.setTo(255, nonIntMasks[i]);
    //    cv::imshow("show", show);
    //    cv::waitKey(0);
    //}

    std::vector<cv::Mat> results;
    //compensate(images, masks, results);
    //compensate3(images, masks, results);
    //compensateGray(images, masks, results);
    compensateLightAndSaturation(images, masks, results);
    for (int i = 0; i < numImages; i++)
    {
        cv::imshow("comp image", results[i]);
        cv::waitKey(0);
    }

    std::vector<cv::Mat> newMasks;
    findSeams(results, masks, newMasks, 8, 1, 1, true);
    for (int i = 0; i < numImages; i++)
    {
        cv::imshow("new mask", newMasks[i]);
        cv::waitKey(0);
    }
    
    cv::Mat blendResult;
    multibandBlendAnyMask(results, masks, newMasks, 20, 2, blendResult);
    cv::imshow("blend result", blendResult);
    cv::waitKey(0);

    return 0;

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
    TilingFeatherBlend blender;
    timer.start();
    blender.prepare(masks);
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