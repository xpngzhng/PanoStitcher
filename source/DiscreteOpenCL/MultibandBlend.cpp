#include "DiscreteOpenCLInterface.h"
#include "RunTimeObjects.h"
#include "MatOp.h"
#include "Pyramid.h"
#include "../Blend/ZBlendAlgo.h"
#include "opencv2/highgui.hpp"
//#include <iostream>
//#include <fstream>

//static void show16S(const std::string& winName, cv::Mat& image)
//{
//    CV_Assert(image.data && image.depth() == CV_16S);
//    cv::Mat temp;
//    image.convertTo(temp, CV_8U, 0.5, 128);
//    cv::imshow(winName, temp);
//}
//
//static void show32S(const std::string& winName, cv::Mat& image)
//{
//    CV_Assert(image.data && image.depth() == CV_32S);
//    cv::Mat temp;
//    image.convertTo(temp, CV_8U, 0.5 / 256, 127);
//    cv::imshow(winName, temp);
//}
//static void show32SUnscale(const std::string& winName, cv::Mat& image)
//{
//    CV_Assert(image.data && image.depth() == CV_32S);
//    cv::Mat temp;
//    image.convertTo(temp, CV_8U);
//    cv::imshow(winName, temp);
//}

static void getPyramidLevelSizes(std::vector<cv::Size>& sizes, int rows, int cols, int numLevels)
{
    sizes.resize(numLevels + 1);
    sizes[0] = cv::Size(cols, rows);
    for (int i = 1; i <= numLevels; i++)
        sizes[i] = cv::Size((sizes[i - 1].width + 1) / 2, (sizes[i - 1].height + 1) / 2);
}

static void getStepsOfImageUpPyr(const std::vector<cv::Size>& sizes, std::vector<int>& steps)
{
    int numLevels = sizes.size() - 1;
    steps.resize(numLevels + 1);
    for (int i = 0; i <= numLevels; i++)
    {
        docl::GpuMat tmp(2, sizes[i].width, CV_16SC4);
        steps[i] = tmp.step;
    }
}

static void getStepsOfResultUpPyr(const std::vector<cv::Size>& sizes, std::vector<int>& steps)
{
    int numLevels = sizes.size() - 1;
    steps.resize(numLevels + 1);
    for (int i = 0; i <= numLevels; i++)
    {
        docl::GpuMat tmp(2, sizes[i].width, CV_32SC4);
        steps[i] = tmp.step;
    }
}

static void allocMemoryForUpPyrs(const std::vector<cv::Size>& sizes,
    const std::vector<int>& stepsImageUpPyr, const std::vector<int>& stepsResultUpPyr,
    std::vector<docl::GpuMat>& imageUpPyr, std::vector<docl::GpuMat>& resultUpPyr)
{
    int numLevels = sizes.size() - 1;
    docl::GpuMat mem(sizes[0], CV_32SC4);

    imageUpPyr.resize(numLevels + 1);
    for (int i = 0; i < numLevels; i++)
        imageUpPyr[i] = docl::GpuMat(sizes[i], CV_16SC4, mem.data, stepsImageUpPyr[i]);

    resultUpPyr.resize(numLevels + 1);
    resultUpPyr[0] = mem;
    for (int i = 1; i < numLevels; i++)
        resultUpPyr[i] = docl::GpuMat(sizes[i], CV_32SC4, mem.data, stepsResultUpPyr[i]);
}

static void allocMemoryForResultPyr(const std::vector<cv::Size>& sizes, std::vector<docl::GpuMat>& resultPyr)
{
    int numLevels = sizes.size() - 1;
    resultPyr.resize(numLevels + 1);
    for (int i = 0; i <= numLevels; i++)
        resultPyr[i].create(sizes[i], CV_32SC4);
}

static void accumulateWeight(const std::vector<docl::GpuMat>& src, std::vector<docl::GpuMat>& dst)
{
    CV_Assert(src.size() == dst.size());
    int size = src.size();
    for (int i = 0; i < size; i++)
        accumulate16SC1To32SC1(src[i], dst[i]);
}

static void accumulate(const std::vector<docl::GpuMat>& imagePyr, const std::vector<docl::GpuMat>& weightPyr,
    std::vector<docl::GpuMat>& resultPyr)
{
    CV_Assert(imagePyr.size() == weightPyr.size() &&
        imagePyr.size() == resultPyr.size());
    int size = imagePyr.size();
    for (int i = 0; i < size; i++)
        accumulate16SC4To32SC4(imagePyr[i], weightPyr[i], resultPyr[i]);
}

static void normalize(std::vector<docl::GpuMat>& pyr)
{
    int size = pyr.size();
    for (int i = 0; i < size; i++)
        normalize32SC4(pyr[i]);
}

static void normalize(std::vector<docl::GpuMat>& pyr, const std::vector<docl::GpuMat>& weightPyr)
{
    int size = pyr.size();
    for (int i = 0; i < size; i++)
        normalize32SC4(pyr[i], weightPyr[i]);
}

static void restoreImageFromLaplacePyramid(std::vector<docl::GpuMat>& pyr,
    std::vector<docl::GpuMat>& upPyr)
{
    if (pyr.empty())
        return;
    upPyr.resize(pyr.size());
    for (size_t i = pyr.size() - 1; i > 0; --i)
    {
        pyramidUp32SC4To32SC4(pyr[i], upPyr[i - 1], pyr[i - 1].size());
        add32SC4(upPyr[i - 1], pyr[i - 1], pyr[i - 1]);
    }
}

bool DOclTilingMultibandBlendFast::prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength)
{
    success = false;
    if (masks.empty())
        return false;

    int currNumMasks = masks.size();
    if (currNumMasks > 255)
        return false;

    int currRows = masks[0].rows, currCols = masks[0].cols;
    for (int i = 0; i < currNumMasks; i++)
    {
        if (!masks[i].data || masks[i].type() != CV_8UC1 ||
            masks[i].rows != currRows || masks[i].cols != currCols)
            return false;
    }
    rows = currRows;
    cols = currCols;
    numImages = currNumMasks;

    std::vector<cv::Mat> uniqueMasksCpu;
    getNonIntersectingMasks(masks, uniqueMasksCpu);

    uniqueMasks.resize(numImages);
    for (int i = 0; i < numImages; i++)
        uniqueMasks[i].upload(uniqueMasksCpu[i]);

    numLevels = getTrueNumLevels(cols, rows, maxLevels, minLength);

    std::vector<docl::GpuMat> masksGpu(numImages);
    for (int i = 0; i < numImages; i++)
        masksGpu[i].upload(masks[i]);

    docl::GpuMat aux16S(rows, cols, CV_16SC1);

    std::vector<docl::GpuMat> tempAlphaPyr(numLevels + 1);
    alphaPyrs.resize(numImages);
    weightPyrs.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        alphaPyrs[i].resize(numLevels + 1);
        weightPyrs[i].resize(numLevels + 1);
        setVal16SC1(aux16S, 0);
        setVal16SC1Mask8UC1(aux16S, 256, masksGpu[i]);
        tempAlphaPyr[0] = aux16S.clone();
        setVal16SC1(aux16S, 0);
        setVal16SC1Mask8UC1(aux16S, 256, uniqueMasks[i]);
        weightPyrs[i][0] = aux16S.clone();
        for (int j = 0; j < numLevels; j++)
        {
            pyramidDown16SC1To32SC1(tempAlphaPyr[j], alphaPyrs[i][j + 1], cv::Size());
            tempAlphaPyr[j + 1].create(alphaPyrs[i][j + 1].size(), CV_16SC1);
            scaledSet16SC1Mask32SC1(tempAlphaPyr[j + 1], 256, alphaPyrs[i][j + 1]);
            pyramidDown16SC1To16SC1(weightPyrs[i][j], weightPyrs[i][j + 1], cv::Size());
        }
    }

    std::vector<cv::Size> sizes;
    getPyramidLevelSizes(sizes, rows, cols, numLevels);

    std::vector<int> stepsImageUpPyr, stepsResultUpPyr;
    getStepsOfImageUpPyr(sizes, stepsImageUpPyr);
    getStepsOfResultUpPyr(sizes, stepsResultUpPyr);
    allocMemoryForUpPyrs(sizes, stepsImageUpPyr, stepsResultUpPyr, imageUpPyr, resultUpPyr);

    allocMemoryForResultPyr(sizes, resultPyr);

    cv::Mat mask = cv::Mat::zeros(rows, cols, CV_8UC1);
    for (int i = 0; i < numImages; i++)
        mask |= masks[i];
    fullMask = cv::countNonZero(mask) == (rows * cols);
    if (fullMask)
    {
        resultWeightPyr.clear();
        maskNot.release();
    }
    else
    {
        resultWeightPyr.resize(numLevels + 1);
        for (int i = 0; i < numLevels + 1; i++)
        {
            resultWeightPyr[i].create(sizes[i], CV_32SC1);
            setZero(resultWeightPyr[i]);
        }
        for (int i = 0; i < numImages; i++)
            accumulateWeight(weightPyrs[i], resultWeightPyr);
        mask = ~mask;
        maskNot.upload(mask);
    }

    success = true;
    return true;
}

void DOclTilingMultibandBlendFast::blend(const std::vector<docl::GpuMat>& images, docl::GpuMat& blendImage)
{
    if (!success)
        return;

    CV_Assert(images.size() == numImages);

    for (int i = 0; i <= numLevels; i++)
        setZero(resultPyr[i]);

    imagePyr.resize(numLevels + 1);
    imageUpPyr.resize(numLevels + 1);
    for (int i = 0; i < numImages; i++)
    {
        //if (images[i].type() == CV_8UC4)
        //    images[i].convertTo(imagePyr[0], CV_16S);
        //else if (images[i].type() == CV_16SC4)
        CV_Assert(images[i].type == CV_16SC4);
            images[i].copyTo(imagePyr[0]);
        for (int j = 0; j < numLevels; j++)
            pyramidDown16SC4To16SC4(imagePyr[j], alphaPyrs[i][j + 1], imagePyr[j + 1]);
        for (int j = 0; j < numLevels; j++)
        {
            pyramidUp16SC4To16SC4(imagePyr[j + 1], imageUpPyr[j], imagePyr[j].size());
            subtract16SC4(imagePyr[j], imageUpPyr[j], imagePyr[j]);
        }
        accumulate(imagePyr, weightPyrs[i], resultPyr);
    }
    if (fullMask)
        normalize(resultPyr);
    else
        normalize(resultPyr, resultWeightPyr);
    restoreImageFromLaplacePyramid(resultPyr, resultUpPyr);
    convert32SC4To8UC4(resultPyr[0], blendImage);
    if (!fullMask)
        setZero8UC4Mask8UC1(blendImage, maskNot);
}

void DOclTilingMultibandBlendFast::getUniqueMasks(std::vector<docl::GpuMat>& masks) const
{
    if (success)
        masks = uniqueMasks;
    else
        masks.clear();
}