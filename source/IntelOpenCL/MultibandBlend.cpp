#include "IntelOpenCLInterface.h"
#include "RunTimeObjects.h"
#include "MatOp.h"
#include "Pyramid.h"
#include "../Blend/ZBlendAlgo.h"
#include "opencv2/highgui.hpp"
#include <iostream>

static void show16S(const std::string& winName, cv::Mat& image)
{
    CV_Assert(image.data && image.depth() == CV_16S);
    cv::Mat temp;
    image.convertTo(temp, CV_8U, 0.5, 128);
    cv::imshow(winName, temp);
}

static void show32S(const std::string& winName, cv::Mat& image)
{
    CV_Assert(image.data && image.depth() == CV_32S);
    cv::Mat temp;
    image.convertTo(temp, CV_8U, 0.5 / 258, 127);
    cv::imshow(winName, temp);
}

static void getPyramidLevelSizes(std::vector<cv::Size>& sizes, int rows, int cols, int numLevels)
{
    sizes.resize(numLevels + 1);
    sizes[0] = cv::Size(cols, rows);
    for (int i = 1; i <= numLevels; i++)
        sizes[i] = cv::Size((sizes[i - 1].width + 1) / 2, (sizes[i - 1].height + 1) / 2);
}

static void getStepsOfImage32SPyr(const std::vector<cv::Size>& sizes, std::vector<int>& steps)
{
    int numLevels = sizes.size() - 1;
    steps.resize(numLevels + 1);
    for (int i = 0; i <= numLevels; i++)
    {
        IOclMat tmp(2, sizes[i].width, CV_32SC4, iocl::ocl->context);
        steps[i] = tmp.step;

    }
}

static void getStepsOfImageUpPyr(const std::vector<cv::Size>& sizes, std::vector<int>& steps)
{
    int numLevels = sizes.size() - 1;
    steps.resize(numLevels + 1);
    for (int i = 0; i <= numLevels; i++)
    {
        IOclMat tmp(2, sizes[i].width, CV_16SC4, iocl::ocl->context);
        steps[i] = tmp.step;
    }
}

static void getStepsOfResultUpPyr(const std::vector<cv::Size>& sizes, std::vector<int>& steps)
{
    int numLevels = sizes.size() - 1;
    steps.resize(numLevels + 1);
    for (int i = 0; i <= numLevels; i++)
    {
        IOclMat tmp(2, sizes[i].width, CV_32SC4, iocl::ocl->context);
        steps[i] = tmp.step;
    }
}

static void allocMemoryForImage32SPyrAndImageUpPyr(const std::vector<cv::Size>& sizes,
    const std::vector<int>& stepsImage32SPyr, const std::vector<int>& stepsImageUpPyr,
    std::vector<IOclMat>& image32SPyr, std::vector<IOclMat>& imageUpPyr)
{
    int numLevels = sizes.size() - 1;
    IOclMat mem(sizes[0], CV_16SC4, iocl::ocl->context);

    imageUpPyr.resize(numLevels + 1);
    imageUpPyr[0] = mem;
    for (int i = 1; i < numLevels; i++)
        imageUpPyr[i] = IOclMat(sizes[i], CV_16SC4, mem.data, stepsImageUpPyr[i], iocl::ocl->context);

    image32SPyr.resize(numLevels + 1);
    for (int i = 1; i <= numLevels; i++)
        image32SPyr[i] = IOclMat(sizes[i], CV_32SC4, mem.data, stepsImage32SPyr[i], iocl::ocl->context);
}

static void allocMemoryForResultPyrAndResultUpPyr(const std::vector<cv::Size>& sizes,
    const std::vector<int>& stepsResultUpPyr, std::vector<IOclMat>& resultPyr,
    std::vector<IOclMat>& resultUpPyr)
{
    int numLevels = sizes.size() - 1;

    resultPyr.resize(numLevels + 1);
    for (int i = 0; i <= numLevels; i++)
        resultPyr[i].create(sizes[i], CV_32SC4, iocl::ocl->context);

    IOclMat mem(sizes[0], CV_32SC4, iocl::ocl->context);
    resultUpPyr.resize(numLevels + 1);
    resultUpPyr[0] = mem;
    for (int i = 1; i < numLevels; i++)
        resultUpPyr[i] = IOclMat(sizes[i], CV_32SC4, mem.data, stepsResultUpPyr[i], iocl::ocl->context);
}

static void accumulateWeight(const std::vector<IOclMat>& src, std::vector<IOclMat>& dst)
{
    CV_Assert(src.size() == dst.size());
    int size = src.size();
    for (int i = 0; i < size; i++)
        accumulate16SC1To32SC1(src[i], dst[i]);
}

static void accumulate(const std::vector<IOclMat>& imagePyr, const std::vector<IOclMat>& weightPyr,
    std::vector<IOclMat>& resultPyr)
{
    CV_Assert(imagePyr.size() == weightPyr.size() &&
        imagePyr.size() == resultPyr.size());
    int size = imagePyr.size();
    for (int i = 0; i < size; i++)
    {
        accumulate16SC4To32SC4(imagePyr[i], weightPyr[i], resultPyr[i]);
        //if (i == size - 1)
        //    std::cout << resultPyr[i].toOpenCVMat() << std::endl;
        //show16S("image", imagePyr[i].toOpenCVMat());
        //show16S("weight", weightPyr[i].toOpenCVMat());
        //show32S("result", resultPyr[i].toOpenCVMat());
        //cv::waitKey(0);
    }
}

static void normalize(std::vector<IOclMat>& pyr)
{
    int size = pyr.size();
    for (int i = 0; i < size; i++)
        normalize32SC4(pyr[i]);
}

static void normalize(std::vector<IOclMat>& pyr, const std::vector<IOclMat>& weightPyr)
{
    int size = pyr.size();
    for (int i = 0; i < size; i++)
        normalize32SC4(pyr[i], weightPyr[i]);
}

static void restoreImageFromLaplacePyramid(std::vector<IOclMat>& pyr,
    std::vector<IOclMat>& upPyr)
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

int getTrueNumLevels(int width, int height, int maxLevels, int minLength);

bool IOclTilingMultibandBlendFast::prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength)
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
        uniqueMasks[i].upload(uniqueMasksCpu[i], iocl::ocl->context);

    numLevels = getTrueNumLevels(cols, rows, maxLevels, minLength);

    std::vector<IOclMat> masksGpu(numImages);
    for (int i = 0; i < numImages; i++)
        masksGpu[i].upload(masks[i], iocl::ocl->context);

    IOclMat aux16S(rows, cols, CV_16SC1, iocl::ocl->context);

    std::vector<IOclMat> tempAlphaPyr(numLevels + 1);
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
            tempAlphaPyr[j + 1].create(alphaPyrs[i][j + 1].size(), CV_16SC1, iocl::ocl->context);
            scaledSet16SC1Mask32SC1(tempAlphaPyr[j + 1], 256, alphaPyrs[i][j + 1]);
            pyramidDown16SC1To16SC1(weightPyrs[i][j], weightPyrs[i][j + 1], cv::Size());
        }
    }

    //for (int i = 0; i < numImages; i++)
    //{
    //    char buf[128];
    //    for (int j = 0; j < numLevels + 1; j++)
    //    {
    //        cv::Mat image;
    //        cv::Mat header = weightPyrs[i][j].toOpenCVMat();
    //        header.convertTo(image, CV_8U);
    //        sprintf(buf, "level %d", j);
    //        cv::imshow(buf, image);
    //        cv::waitKey(0);
    //    }
    //}

    //for (int i = 0; i < numImages; i++)
    //{
    //    char buf[128];
    //    for (int j = 1; j < numLevels + 1; j++)
    //    {
    //        cv::Mat image;
    //        cv::Mat header = alphaPyrs[i][j].toOpenCVMat();
    //        header.convertTo(image, CV_8U, 1.0 / 256);
    //        sprintf(buf, "level %d", j);
    //        cv::imshow(buf, image);
    //        cv::waitKey(0);
    //    }
    //}

    std::vector<cv::Size> sizes;
    getPyramidLevelSizes(sizes, rows, cols, numLevels);

    std::vector<int> stepsImage32SPyr, stepsImageUpPyr;
    getStepsOfImage32SPyr(sizes, stepsImage32SPyr);
    getStepsOfImageUpPyr(sizes, stepsImageUpPyr);
    allocMemoryForImage32SPyrAndImageUpPyr(sizes, stepsImage32SPyr, stepsImageUpPyr, image32SPyr, imageUpPyr);

    // The above-mentioned fact also applies to resultUpPyr.
    std::vector<int> stepsResultUpPyr;
    getStepsOfResultUpPyr(sizes, stepsResultUpPyr);
    allocMemoryForResultPyrAndResultUpPyr(sizes, stepsResultUpPyr, resultPyr, resultUpPyr);

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
            resultWeightPyr[i].create(sizes[i], CV_32SC1, iocl::ocl->context);
            setZero(resultWeightPyr[i]);
        }
        for (int i = 0; i < numImages; i++)
            accumulateWeight(weightPyrs[i], resultWeightPyr);
        mask = ~mask;
        maskNot.upload(mask, iocl::ocl->context);
    }

    success = true;
    return true;
}

void IOclTilingMultibandBlendFast::blend(const std::vector<IOclMat>& images, IOclMat& blendImage)
{
    if (!success)
        return;

    CV_Assert(images.size() == numImages);

    for (int i = 0; i <= numLevels; i++)
        setZero(resultPyr[i]);

    imagePyr.resize(numLevels + 1);
    image32SPyr.resize(numLevels + 1);
    imageUpPyr.resize(numLevels + 1);
    for (int i = 0; i < numImages; i++)
    {
        //if (images[i].type() == CV_8UC4)
        //    images[i].convertTo(imagePyr[0], CV_16S);
        //else if (images[i].type() == CV_16SC4)
        CV_Assert(images[i].type == CV_16SC4);
            images[i].copyTo(imagePyr[0]);
        for (int j = 0; j < numLevels; j++)
        {
            //pyramidDown16SC4To16SC4(imagePyr[j], imagePyr[j + 1], cv::Size(), true);
            pyramidDown16SC4To16SC4(imagePyr[j], alphaPyrs[i][j + 1], imagePyr[j + 1]);
            //show16S("level", imagePyr[j + 1].toOpenCVMat());
            //cv::waitKey(0);
        }
        for (int j = 0; j < numLevels; j++)
        {
            pyramidUp16SC4To16SC4(imagePyr[j + 1], imageUpPyr[j], imagePyr[j].size());
            subtract16SC4(imagePyr[j], imageUpPyr[j], imagePyr[j]);
            //show16S("level", imagePyr[j].toOpenCVMat());
            //cv::waitKey(0);
        }
        //show16S("level", imagePyr[numLevels].toOpenCVMat());
        //cv::waitKey(0);
        accumulate(imagePyr, weightPyrs[i], resultPyr);
    }
    for (int i = 0; i < numLevels + 1; i++)
    {
        show32S("level", resultPyr[i].toOpenCVMat());
        cv::waitKey(0);
    }
        
    if (fullMask)
        normalize(resultPyr);
    else
        normalize(resultPyr, resultWeightPyr);
    restoreImageFromLaplacePyramid(resultPyr, resultUpPyr);
    //resultPyr[0].convertTo(blendImage, CV_8U);
    convert32SC4To8UC4(resultPyr[0], blendImage);
    if (!fullMask)
        //blendImage.setTo(0, maskNot);
        setZero8UC4Mask8UC1(blendImage, maskNot);
}

void IOclTilingMultibandBlendFast::getUniqueMasks(std::vector<IOclMat>& masks) const
{
    if (success)
        masks = uniqueMasks;
    else
        masks.clear();
}