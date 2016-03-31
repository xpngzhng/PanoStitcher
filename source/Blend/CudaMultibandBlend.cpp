#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/device/common.hpp>
#include "Timer.h"
#include "ZBlendAlgo.h"
#include "ZBlend.h"
#include "Pyramid.h"
#include "CudaPyramid.h"
#include <iostream>
#include <cuda_runtime.h>

int getTrueNumLevels(int width, int height, int maxLevels, int minLength);

static void inspect(const cv::gpu::GpuMat& image)
{
    cv::Mat cpuImage;
    image.download(cpuImage);
    cv::Mat image8U;
    cpuImage.convertTo(image8U, CV_8U);
    cv::imshow("image", image8U);
    cv::waitKey(0);
}

static void inspect32S(cv::gpu::GpuMat& image)
{
    cv::Mat cpuImage;
    image.download(cpuImage);
    //cpuImage += cv::Scalar::all(255);
    //cpuImage *= (0.5);
    cpuImage += cv::Scalar::all(128 * 255);
    cpuImage *= (1.0 / 256);
    cv::Mat image8U;
    cpuImage.convertTo(image8U, CV_8U);
    cv::imshow("image", image8U);
    cv::waitKey(0);
}

static void createGaussPyramid(const cv::gpu::GpuMat& image, int numLevels, bool horiWrap, std::vector<cv::gpu::GpuMat>& pyr)
{
    CV_Assert(image.data && image.type() == CV_16SC1);
    pyr.resize(numLevels + 1);
    pyr[0] = image;
    for (int i = 0; i < numLevels; i++)
        pyramidDown16SC1To16SC1(pyr[i], pyr[i + 1], cv::Size(), horiWrap);
}

static void restoreImageFromLaplacePyramid(std::vector<cv::gpu::GpuMat>& pyr, bool horiWrap)
{
    if (pyr.empty())
        return;
    cv::gpu::GpuMat tmp;
    for (size_t i = pyr.size() - 1; i > 0; --i)
    {
        pyramidUp32SC4To32SC4(pyr[i], tmp, pyr[i - 1].size(), horiWrap);
        cv::gpu::add(tmp, pyr[i - 1], pyr[i - 1]);
    }
}

static void accumulate(const std::vector<cv::gpu::GpuMat>& imagePyr, const std::vector<cv::gpu::GpuMat>& weightPyr, 
    std::vector<cv::gpu::GpuMat>& resultPyr)
{
    CV_Assert(imagePyr.size() == weightPyr.size() && 
              imagePyr.size() == resultPyr.size());
    int size = imagePyr.size();
    for (int i = 0; i < size; i++)
        accumulate16SC4To32SC4(imagePyr[i], weightPyr[i], resultPyr[i]);
}

static void normalize(std::vector<cv::gpu::GpuMat>& pyr)
{
    int size = pyr.size();
    for (int i = 0; i < size; i++)
        normalize32SC4(pyr[i]);
}

static void createLaplacePyramidPrecise(const cv::gpu::GpuMat& image, const cv::gpu::GpuMat& alpha, 
    int numLevels, bool horiWrap, std::vector<cv::gpu::GpuMat>& pyr)
{
    CV_Assert(image.data && image.type() == CV_16SC4 &&
        alpha.data && alpha.type() == CV_16SC1);
    pyr.resize(numLevels + 1);
    pyr[0] = image;
    cv::gpu::GpuMat currAlpha;
    currAlpha = alpha;
    for (int i = 0; i < numLevels; ++i)
    {
        cv::gpu::GpuMat newAlpha;
        pyramidDown16SC4To16SC4(pyr[i], currAlpha, horiWrap, pyr[i + 1], newAlpha);
        currAlpha = newAlpha;
    }
    currAlpha.release();

    cv::gpu::GpuMat tmp;
    for (int i = 0; i < numLevels; ++i)
    {
        pyramidUp16SC4To16SC4(pyr[i + 1], tmp, pyr[i].size(), horiWrap);
        cv::gpu::subtract(pyr[i], tmp, pyr[i]);
    }
}

void cudaMultibandBlend(const cv::gpu::GpuMat& image1, const cv::gpu::GpuMat& image2, 
    const cv::gpu::GpuMat& alpha1, const cv::gpu::GpuMat& alpha2,
    cv::gpu::GpuMat& mask1, const cv::gpu::GpuMat& mask2, 
    bool horiWrap, int maxLevels, int minLength, cv::gpu::GpuMat& result)
{
    CV_Assert(image1.data && image1.type() == CV_8UC4 &&
        image2.data && image2.type() == CV_8UC4 &&
        alpha1.data && alpha1.type() == CV_8UC1 &&
        alpha2.data && alpha2.type() == CV_8UC1 &&
        mask1.data && mask1.type() == CV_8UC1 &&
        mask2.data && mask2.type() == CV_8UC1);
    cv::Size size = image1.size();
    CV_Assert(size == image2.size() && size == mask1.size() && size == mask2.size());    

    int numLevels = getTrueNumLevels(size.width, size.height, maxLevels, minLength);
    //printf("numLevels = %d\n", numLevels);
    if (numLevels == 0)
    {
        image1.copyTo(result, mask1);
        image2.copyTo(result, mask2);
        return;
    }
    std::vector<cv::gpu::GpuMat> resultPyr(numLevels + 1);
    resultPyr[0].create(image1.size(), CV_32SC4);
    resultPyr[0].setTo(0);
    for (int i = 1; i <= numLevels; i++)
    {
        resultPyr[i].create((resultPyr[i - 1].rows + 1) / 2, (resultPyr[i - 1].cols + 1) / 2, CV_32SC4);
        resultPyr[i].setTo(0);
    }

    cv::gpu::GpuMat image(size, CV_16SC4), mask(size, CV_16SC1);
    std::vector<cv::gpu::GpuMat> imagePyr, weightPyr;

    //static int count = 0;
    //count++;
    //char name[256];

    image1.convertTo(image, CV_16S);
    mask.setTo(0);
    mask.setTo(256, alpha1);
    createLaplacePyramidPrecise(image, mask, numLevels, horiWrap, imagePyr);
    mask.setTo(0);
    mask.setTo(256, mask1);
    createGaussPyramid(mask, numLevels, horiWrap, weightPyr);
    accumulate(imagePyr, weightPyr, resultPyr);

    //sprintf_s(name, "image1_number%d_level", count);
    //savePyramid(imagePyr, name, "tif");
    //sprintf_s(name, "mask1_number%d_level", count);
    //savePyramid(weightPyr, name, "tif");

    image2.convertTo(image, CV_16S);
    mask.setTo(0);
    mask.setTo(256, alpha2);
    createLaplacePyramidPrecise(image, mask, numLevels, horiWrap, imagePyr);
    mask.setTo(0);
    mask.setTo(256, mask2);
    createGaussPyramid(mask, numLevels, horiWrap, weightPyr);
    accumulate(imagePyr, weightPyr, resultPyr);

    //sprintf_s(name, "image2_number%d_level", count);
    //savePyramid(imagePyr, name, "tif");
    //sprintf_s(name, "mask2_number%d_level", count);
    //savePyramid(weightPyr, name, "tif");

    normalize(resultPyr);
    restoreImageFromLaplacePyramid(resultPyr, horiWrap);
    resultPyr[0].convertTo(result, CV_8U);
}

void createLaplacePyramidPrecise(const cv::gpu::GpuMat& image, const cv::gpu::GpuMat& alpha, 
    int numLevels, bool horiWrap, 
    std::vector<cv::gpu::GpuMat>& imagePyr, std::vector<cv::gpu::GpuMat>& image32SPyr, 
    std::vector<cv::gpu::GpuMat>& alphaPyr, std::vector<cv::gpu::GpuMat>& alpha32SPyr,
    std::vector<cv::gpu::GpuMat>& imageUpPyr)
{
    CV_Assert(image.data && image.type() == CV_16SC4 &&
        alpha.data && alpha.type() == CV_16SC1);
    imagePyr.resize(numLevels + 1);
    imagePyr[0] = image;
    alphaPyr.resize(numLevels + 1);
    alphaPyr[0] = alpha;
    alpha32SPyr.resize(numLevels + 1);    
    image32SPyr.resize(numLevels + 1);
    imageUpPyr.resize(numLevels + 1);
    for (int i = 0; i < numLevels; ++i)
    {
        pyramidDown16SC4To32SC4(imagePyr[i], image32SPyr[i + 1], cv::Size(), horiWrap);
        pyramidDown16SC1To32SC1(alphaPyr[i], alpha32SPyr[i + 1], cv::Size(), horiWrap);
        divide32SC4To16SC4(image32SPyr[i + 1], alpha32SPyr[i + 1], imagePyr[i + 1], alphaPyr[i + 1]);
    }

    for (int i = 0; i < numLevels; ++i)
    {
        pyramidUp16SC4To16SC4(imagePyr[i + 1], imageUpPyr[i], imagePyr[i].size(), horiWrap);
        cv::gpu::subtract(imagePyr[i], imageUpPyr[i], imagePyr[i]);
    }
}

void restoreImageFromLaplacePyramid(std::vector<cv::gpu::GpuMat>& pyr, bool horiWrap, 
    std::vector<cv::gpu::GpuMat>& upPyr)
{
    if (pyr.empty())
        return;
    upPyr.resize(pyr.size());
    for (size_t i = pyr.size() - 1; i > 0; --i)
    {
        pyramidUp32SC4To32SC4(pyr[i], upPyr[i - 1], pyr[i - 1].size(), horiWrap);
        cv::gpu::add(upPyr[i - 1], pyr[i - 1], pyr[i - 1]);
    }
}

bool CudaTilingMultibandBlend::prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength)
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
    uniqueMasksCpu.clear();
    //std::vector<cv::Mat>().swap(uniqueMasksCpu);

    numLevels = getTrueNumLevels(cols, rows, maxLevels, minLength);

    resultPyr.resize(numLevels + 1);
    resultPyr[0].create(rows, cols, CV_32SC4);
    resultPyr[0].setTo(0);
    for (int i = 1; i <= numLevels; i++)
    {
        resultPyr[i].create((resultPyr[i - 1].rows + 1) / 2, (resultPyr[i - 1].cols + 1) / 2, CV_32SC4);
        resultPyr[i].setTo(0);
    }

    success = true;
    return true;
}
/*
void CudaTilingMultibandBlend::tile(const cv::gpu::GpuMat& image, const cv::gpu::GpuMat& mask, int index)
{
    if (!success)
        return;

    CV_Assert(image.data && image.type() == CV_8UC4 && image.rows == rows && image.cols == cols &&
        mask.data && mask.type() == CV_8UC1 && mask.rows == rows && mask.cols == cols &&
        index >= 0 && index < numImages);

    cv::gpu::GpuMat image16S;
    image.convertTo(image16S, CV_16S);
    std::vector<cv::gpu::GpuMat> imagePyr, weightPyr;
    cv::gpu::GpuMat aux(rows, cols, CV_16SC1);
    aux.setTo(0);
    aux.setTo(256, mask);
    createLaplacePyramidPrecise(image16S, aux, numLevels, true, imagePyr);
    aux.setTo(0);
    aux.setTo(256, uniqueMasks[index]);
    createGaussPyramid(aux, numLevels, true, weightPyr);
    accumulate(imagePyr, weightPyr, resultPyr);
}

void CudaTilingMultibandBlend::composite(cv::gpu::GpuMat& blendImage)
{
    if (!success)
        return;

    normalize(resultPyr);
    restoreImageFromLaplacePyramid(resultPyr, true);
    resultPyr[0].convertTo(blendImage, CV_8U);

    // Remember to set resultPyr to zero, 
    // otherwise next round of tile and composite will produce incorrect result!!!!!
    for (int i = 0; i <= numLevels; i++)
        resultPyr[i].setTo(0);
}
*/

void CudaTilingMultibandBlend::tile(const cv::gpu::GpuMat& image, const cv::gpu::GpuMat& mask, int index)
{
    if (!success)
        return;

    CV_Assert(image.data && (image.type() == CV_8UC4 || image.type() == CV_16SC4) && 
        image.rows == rows && image.cols == cols &&
        mask.data && mask.type() == CV_8UC1 && mask.rows == rows && mask.cols == cols &&
        index >= 0 && index < numImages);

    if (image.type() == CV_8UC4)
        image.convertTo(image16S, CV_16S);
    else
        image16S = image;
    aux16S.create(rows, cols, CV_16SC1);
    aux16S.setTo(0);
    aux16S.setTo(256, mask);
    createLaplacePyramidPrecise(image16S, aux16S, numLevels, true, 
        imagePyr, image32SPyr, alphaPyr, alpha32SPyr, imageUpPyr);
    aux16S.setTo(0);
    aux16S.setTo(256, uniqueMasks[index]);
    createGaussPyramid(aux16S, numLevels, true, weightPyr);
    accumulate(imagePyr, weightPyr, resultPyr);
}

void CudaTilingMultibandBlend::composite(cv::gpu::GpuMat& blendImage)
{
    if (!success)
        return;

    normalize(resultPyr);
    restoreImageFromLaplacePyramid(resultPyr, true, resultUpPyr);
    resultPyr[0].convertTo(blendImage, CV_8U);

    // Remember to set resultPyr to zero, 
    // otherwise next round of tile and composite will produce incorrect result!!!!!
    for (int i = 0; i <= numLevels; i++)
        resultPyr[i].setTo(0);
}

void CudaTilingMultibandBlend::blend(const std::vector<cv::gpu::GpuMat>& images, 
    const std::vector<cv::gpu::GpuMat>& masks, cv::gpu::GpuMat& blendImage)
{
    if (!success)
        return;

    CV_Assert(images.size() == numImages && masks.size() == numImages);

    for (int i = 0; i < numImages; i++)
        tile(images[i], masks[i], i);
    composite(blendImage);
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
        cv::gpu::GpuMat tmp(2, sizes[i].width, CV_32SC4);
        steps[i] = tmp.step;
    }
}

static void getStepsOfImageUpPyr(const std::vector<cv::Size>& sizes, std::vector<int>& steps)
{
    int numLevels = sizes.size() - 1;
    steps.resize(numLevels + 1);
    for (int i = 0; i <= numLevels; i++)
    {
        cv::gpu::GpuMat tmp(2, sizes[i].width, CV_16SC4);
        steps[i] = tmp.step;
    }
}

static void getStepsOfResultUpPyr(const std::vector<cv::Size>& sizes, std::vector<int>& steps)
{
    int numLevels = sizes.size() - 1;
    steps.resize(numLevels + 1);
    for (int i = 0; i <= numLevels; i++)
    {
        cv::gpu::GpuMat tmp(2, sizes[i].width, CV_32SC4);
        steps[i] = tmp.step;
    }
}

static void allocMemoryForImage32SPyrAndImageUpPyr(const std::vector<cv::Size>& sizes, 
    const std::vector<int>& stepsImage32SPyr, const std::vector<int>& stepsImageUpPyr,
    std::vector<cv::gpu::GpuMat>& image32SPyr, std::vector<cv::gpu::GpuMat>& imageUpPyr)
{
    int numLevels = sizes.size() - 1;
    cv::gpu::GpuMat mem(sizes[0], CV_16SC4);

    imageUpPyr.resize(numLevels + 1);
    imageUpPyr[0] = mem;
    for (int i = 1; i < numLevels; i++)
        imageUpPyr[i] = cv::gpu::GpuMat(sizes[i], CV_16SC4, mem.data, stepsImageUpPyr[i]);

    image32SPyr.resize(numLevels + 1);
    for (int i = 1; i <= numLevels; i++)
        image32SPyr[i] = cv::gpu::GpuMat(sizes[i], CV_32SC4, mem.data, stepsImage32SPyr[i]);
}

static void allocMemoryForResultPyrAndResultUpPyr(const std::vector<cv::Size>& sizes,
    const std::vector<int>& stepsResultUpPyr, std::vector<cv::gpu::GpuMat>& resultPyr,
    std::vector<cv::gpu::GpuMat>& resultUpPyr)
{
    int numLevels = sizes.size() - 1;

    resultPyr.resize(numLevels + 1);
    for (int i = 0; i <= numLevels; i++)
        resultPyr[i].create(sizes[i], CV_32SC4);

    cv::gpu::GpuMat mem(sizes[0], CV_32SC4);
    resultUpPyr.resize(numLevels + 1);
    resultUpPyr[0] = mem;
    for (int i = 1; i < numLevels; i++)
        resultUpPyr[i] = cv::gpu::GpuMat(sizes[i], CV_32SC4, mem.data, stepsResultUpPyr[i]);
}

bool CudaTilingMultibandBlendFast::prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength)
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

    std::vector<cv::Mat> uniqueMasks;
    getNonIntersectingMasks(masks, uniqueMasks);

    std::vector<cv::gpu::GpuMat> uniqueMasksGpu(numImages);
    for (int i = 0; i < numImages; i++)
        uniqueMasksGpu[i].upload(uniqueMasks[i]);
    uniqueMasks.clear();

    numLevels = getTrueNumLevels(cols, rows, maxLevels, minLength);

    std::vector<cv::gpu::GpuMat> masksGpu(numImages);
    for (int i = 0; i < numImages; i++)
        masksGpu[i].upload(masks[i]);

    cv::gpu::GpuMat aux16S(rows, cols, CV_16SC1);

    std::vector<cv::gpu::GpuMat> tempAlphaPyr(numLevels + 1);
    alphaPyrs.resize(numImages);
    weightPyrs.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        alphaPyrs[i].resize(numLevels + 1);
        weightPyrs[i].resize(numLevels + 1);
        aux16S.setTo(0);
        aux16S.setTo(256, masksGpu[i]);
        tempAlphaPyr[0] = aux16S.clone();
        aux16S.setTo(0);
        aux16S.setTo(256, uniqueMasksGpu[i]);
        weightPyrs[i][0] = aux16S.clone();
        for (int j = 0; j < numLevels; j++)
        {
            pyramidDown16SC1To32SC1(tempAlphaPyr[j], alphaPyrs[i][j + 1], cv::Size(), true);
            tempAlphaPyr[j + 1].create(alphaPyrs[i][j + 1].size(), CV_16SC1);
            scaledSet16SC1Mask32SC1(tempAlphaPyr[j + 1], 256, alphaPyrs[i][j + 1]);
            pyramidDown16SC1To16SC1(weightPyrs[i][j], weightPyrs[i][j + 1], cv::Size(), true);
        }
    }

    // IMPORTANT NOTE!!!!
    // The following lines of code is not necessary but can help reduce memory consumption.
    // image32SPyr and imageUpPyr are temporary image pyramids for building the final imagePyrs, 
    // athe use of image32SPyr[j] is independent of the use of image32SPyr[k], 
    // which means that for a certain image, the processing of different levels of 32SPyr do not interfere each other,
    // they can actually SHARE THE SAME MEMORY.
    // The SAME HAPPENS to imageUpPyrs.
    // Moreover, the use of image32SPyr and imageUpPyr happens in different stage of the whole process,
    // which means that the two pyramids can also SHARE THE SAME MEMORY.
    // If we build a (numLevels + 1) levels pyramid, with each level indexed by 0, 1, 2, ..., numLevels,
    // image32SPyr only use the levels indexed by 1, 2, 3, ..., numLevels,
    // and imageUpPyr only use the levels indexed by 0, 1, 2, ..., numLevels - 1.
    // By some computation we can determine that the largest amount of memory lies in imageUpPyr[0], 
    // so we just allocate the amount of memory fot that level,
    // and imageUpPyr[i], i = 1, 2, ..., numLevels - 1 and image32SPyr[i], i = 1, 2, 3, ..., numLevels - 1
    // can reused that alloced memory.
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

    success = true;
    return true;
}

void CudaTilingMultibandBlendFast::blend(const std::vector<cv::gpu::GpuMat>& images, cv::gpu::GpuMat& blendImage)
{
    if (!success)
        return;

    CV_Assert(images.size() == numImages);

    for (int i = 0; i <= numLevels; i++)
        resultPyr[i].setTo(0);

    imagePyr.resize(numLevels + 1);
    image32SPyr.resize(numLevels + 1);
    imageUpPyr.resize(numLevels + 1);
    for (int i = 0; i < numImages; i++)
    {
        if (images[i].type() == CV_8UC4)
            images[i].convertTo(imagePyr[0], CV_16S);
        else if (images[i].type() == CV_16SC4)
            imagePyr[0] = images[i];
        for (int j = 0; j < numLevels; j++)
        {
            pyramidDown16SC4To32SC4(imagePyr[j], image32SPyr[j + 1], cv::Size(), true);
            divide32SC4To16SC4(image32SPyr[j + 1], alphaPyrs[i][j + 1], imagePyr[j + 1]);
        }
        for (int j = 0; j < numLevels; j++)
        {
            pyramidUp16SC4To16SC4(imagePyr[j + 1], imageUpPyr[j], imagePyr[j].size(), true);
            subtract16SC4(imagePyr[j], imageUpPyr[j], imagePyr[j]);
        }
        accumulate(imagePyr, weightPyrs[i], resultPyr);
    }
    normalize(resultPyr);
    restoreImageFromLaplacePyramid(resultPyr, true, resultUpPyr);
    resultPyr[0].convertTo(blendImage, CV_8U);
}

static void calcAlphasAndWeights(const std::vector<cv::gpu::GpuMat>& masks, const std::vector<cv::gpu::GpuMat>& uniqueMasks, int numLevels,
    std::vector<std::vector<cv::gpu::GpuMat> >& alphaPyrs, std::vector<std::vector<cv::gpu::GpuMat> >& weightPyrs)
{
    int numImages = masks.size();
    int rows = masks[0].rows, cols = masks[0].cols;

    cv::gpu::GpuMat aux16S(rows, cols, CV_16SC1);

    std::vector<cv::gpu::GpuMat> tempAlphaPyr(numLevels + 1);
    alphaPyrs.resize(numImages);
    weightPyrs.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        alphaPyrs[i].resize(numLevels + 1);
        weightPyrs[i].resize(numLevels + 1);
        aux16S.setTo(0);
        aux16S.setTo(256, masks[i]);
        tempAlphaPyr[0] = aux16S.clone();
        aux16S.setTo(0);
        aux16S.setTo(256, uniqueMasks[i]);
        weightPyrs[i][0] = aux16S.clone();
        for (int j = 0; j < numLevels; j++)
        {
            pyramidDown16SC1To32SC1(tempAlphaPyr[j], alphaPyrs[i][j + 1], cv::Size(), true);
            tempAlphaPyr[j + 1].create(alphaPyrs[i][j + 1].size(), CV_16SC1);
            scaledSet16SC1Mask32SC1(tempAlphaPyr[j + 1], 256, alphaPyrs[i][j + 1]);
            pyramidDown16SC1To16SC1(weightPyrs[i][j], weightPyrs[i][j + 1], cv::Size(), true);
        }
    }
}

void prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength,
    std::vector<std::vector<cv::gpu::GpuMat> >& alphaPyrs, std::vector<std::vector<cv::gpu::GpuMat> >& weightPyrs,
    std::vector<cv::gpu::GpuMat>& resultPyr, std::vector<std::vector<cv::gpu::GpuMat> >& image32SPyrs,
    std::vector<std::vector<cv::gpu::GpuMat> >& imageUpPyrs, std::vector<cv::gpu::GpuMat>& resultUpPyr)
{
    int numImages = masks.size();
    int rows = masks[0].rows, cols = masks[0].cols;
    std::vector<cv::Mat> uniqueMasks;
    getNonIntersectingMasks(masks, uniqueMasks);
    std::vector<cv::gpu::GpuMat> masksGpu(numImages), uniqueMasksGpu(numImages);
    for (int i = 0; i < numImages; i++)
    {
        masksGpu[i].upload(masks[i]);
        uniqueMasksGpu[i].upload(uniqueMasks[i]);
    }
    uniqueMasks.clear();

    int numLevels = getTrueNumLevels(cols, rows, maxLevels, minLength);

    calcAlphasAndWeights(masksGpu, uniqueMasksGpu, numLevels, alphaPyrs, weightPyrs);

    std::vector<cv::Size> sizes;
    getPyramidLevelSizes(sizes, rows, cols, numLevels);

    std::vector<int> stepsImage32SPyr, stepsImageUpPyr;
    getStepsOfImage32SPyr(sizes, stepsImage32SPyr);
    getStepsOfImageUpPyr(sizes, stepsImageUpPyr);

    image32SPyrs.resize(numImages);
    imageUpPyrs.resize(numImages);
    for (int i = 0; i < numImages; i++)
        allocMemoryForImage32SPyrAndImageUpPyr(sizes, stepsImage32SPyr, stepsImageUpPyr, image32SPyrs[i], imageUpPyrs[i]);

    std::vector<int> stepsResultUpPyr;
    getStepsOfResultUpPyr(sizes, stepsResultUpPyr);
    allocMemoryForResultPyrAndResultUpPyr(sizes, stepsResultUpPyr, resultPyr, resultUpPyr);
}

void calcImagePyramid(const cv::gpu::GpuMat& image, const std::vector<cv::gpu::GpuMat>& alphaPyr,
    std::vector<cv::gpu::GpuMat>& imagePyr, cv::gpu::Stream& stream,
    std::vector<cv::gpu::GpuMat>& image32SPyr, std::vector<cv::gpu::GpuMat>& imageUpPyr)
{
    CV_Assert(image.type() == CV_16SC4 || image.type() == CV_8UC4);
    int numLevels = alphaPyr.size() - 1;
    imagePyr.resize(numLevels + 1);
    image32SPyr.resize(numLevels + 1);
    imageUpPyr.resize(numLevels + 1);
    if (image.type() == CV_8UC4)
        //image.convertTo(imagePyr[0], CV_16S);
        stream.enqueueConvert(image, imagePyr[0], CV_16S);
    else
        imagePyr[0] = image;
    for (int j = 0; j < numLevels; j++)
    {
        pyramidDown16SC4To32SC4(imagePyr[j], image32SPyr[j + 1], cv::Size(), true, stream);
        divide32SC4To16SC4(image32SPyr[j + 1], alphaPyr[j + 1], imagePyr[j + 1], stream);
    }
    for (int j = 0; j < numLevels; j++)
    {
        pyramidUp16SC4To16SC4(imagePyr[j + 1], imageUpPyr[j], imagePyr[j].size(), true, stream);
        subtract16SC4(imagePyr[j], imageUpPyr[j], imagePyr[j], stream);
    }
}

void calcResult(const std::vector<std::vector<cv::gpu::GpuMat> >& imagePyrs, 
    const std::vector<std::vector<cv::gpu::GpuMat> >& weightPyrs, cv::gpu::GpuMat& result,
    std::vector<cv::gpu::GpuMat>& resultPyr, std::vector<cv::gpu::GpuMat>& resultUpPyr)
{
    int size = resultPyr.size();
    for (int i = 0; i < size; i++)
        resultPyr[i].setTo(0);

    int numImages = imagePyrs.size();
    for (int i = 0; i < numImages; i++)
        accumulate(imagePyrs[i], weightPyrs[i], resultPyr);

    normalize(resultPyr);
    restoreImageFromLaplacePyramid(resultPyr, true, resultUpPyr);
    resultPyr[0].convertTo(result, CV_8U);
}