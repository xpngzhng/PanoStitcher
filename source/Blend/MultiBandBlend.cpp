#include "ZBlendAlgo.h"
#include "ZBlend.h"
#include "Pyramid.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

void convert16STo16U(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(src.data && src.depth() == CV_16S);
    int rows = src.rows, cols = src.cols, channels = src.channels();
    int numElemRow = cols * channels;
    dst.create(rows, cols, CV_MAKETYPE(CV_16U, channels));
    for (int i = 0; i < rows; i++)
    {
        const short* ptrSrc = src.ptr<short>(i);
        unsigned short* ptrDst = dst.ptr<unsigned short>(i);
        for (int j = 0; j < numElemRow; j++)
        {
            int val = *ptrSrc + 255;
            if (val < 0)
                printf("val = %d, underflow\n", val);
            else if (val > 512)
                printf("val = %d, overflow\n", val);
            *ptrDst = (*ptrSrc + 255) << 7;
            ptrSrc++;
            ptrDst++;
        }
    }
}

void savePyramid(const std::vector<cv::Mat>& pyr, const std::string& prefix, const std::string& ext)
{
    if (pyr.empty())
        return;
    
    int size = pyr.size();
    for (int i = 0; i < size; i++)
    {
        CV_Assert(pyr[i].data && pyr[i].depth() == CV_16S);
        cv::Mat temp;
        convert16STo16U(pyr[i], temp);
        char name[256];
        sprintf_s(name, "%s%d.%s", prefix.c_str(), i, ext.c_str());
        cv::imwrite(name, temp);
    }
}

void createGaussPyramid(const cv::Mat& image, int numLevels, bool horiWrap, std::vector<cv::Mat>& pyr)
{
    pyr.resize(numLevels + 1);
    if (image.depth() == CV_16S)
        //image.copyTo(pyr[0]);
        pyr[0] = image;
    else
        image.convertTo(pyr[0], CV_16S);
    for (int i = 0; i < numLevels; i++)
        pyramidDown(pyr[i], pyr[i + 1], cv::Size(), horiWrap ? cv::BORDER_WRAP : cv::BORDER_DEFAULT);
}

void createLaplacePyramid(const cv::Mat& image, int numLevels, bool horiWrap, std::vector<cv::Mat>& pyr)
{    
    pyr.resize(numLevels + 1);
    if (image.depth() == CV_16S)
        //image.copyTo(pyr[0]);
        pyr[0] = image;
    else
        image.convertTo(pyr[0], CV_16S);
    for (int i = 0; i < numLevels; ++i)
        pyramidDown(pyr[i], pyr[i + 1]);
    cv::Mat tmp;
    for (int i = 0; i < numLevels; ++i)
    {
        pyramidUp(pyr[i + 1], tmp, pyr[i].size(), horiWrap ? cv::BORDER_WRAP : cv::BORDER_DEFAULT);
        cv::subtract(pyr[i], tmp, pyr[i]);
    }
}

void restoreImageFromLaplacePyramid(std::vector<cv::Mat>& pyr, bool horiWrap)
{
    if (pyr.empty())
        return;
    cv::Mat tmp;
    for (size_t i = pyr.size() - 1; i > 0; --i)
    {
        pyramidUp(pyr[i], tmp, pyr[i - 1].size(), horiWrap ? cv::BORDER_WRAP : cv::BORDER_DEFAULT);
        cv::add(tmp, pyr[i - 1], pyr[i - 1]);
    }
}

void accumulate(const cv::Mat& src, const cv::Mat& weight, cv::Mat& dst)
{
    CV_Assert(src.data && src.type() == CV_16SC3 &&
              weight.data && weight.type() == CV_16SC1 &&
              dst.data && dst.type() == CV_32SC3 &&
              src.size() == weight.size() && src.size() == dst.size());

    int rows = src.rows, cols = src.cols;
    for (int i = 0; i < rows; i++)
    {
        const short* ptrSrcRow = src.ptr<short>(i);
        const short* ptrWeightRow = weight.ptr<short>(i);
        int* ptrDstRow = dst.ptr<int>(i);
        for (int j = 0; j < cols; j++)
        {
            ptrDstRow[0] += ptrSrcRow[0] * ptrWeightRow[0];
            ptrDstRow[1] += ptrSrcRow[1] * ptrWeightRow[0];
            ptrDstRow[2] += ptrSrcRow[2] * ptrWeightRow[0];
            ptrDstRow += 3;
            ptrSrcRow += 3;
            ptrWeightRow++;
        }
    }
}

void accumulate(const std::vector<cv::Mat>& imagePyr, const std::vector<cv::Mat>& weightPyr, 
    std::vector<cv::Mat>& resultPyr)
{
    CV_Assert(imagePyr.size() == weightPyr.size() && 
              imagePyr.size() == resultPyr.size());
    int size = imagePyr.size();
    for (int i = 0; i < size; i++)
        ::accumulate(imagePyr[i], weightPyr[i], resultPyr[i]);
}

void normalize(cv::Mat& image)
{
    CV_Assert(image.data && image.type() == CV_32SC3);
    int rows = image.rows, cols = image.cols * 3;
    for (int i = 0; i < rows; i++)
    {
        int* ptr = image.ptr<int>(i);
        for (int j = 0; j < cols; j++)
        {
            *ptr = (*ptr + 128) >> 8;
            ptr++;
        }
    }
}

void normalize(std::vector<cv::Mat>& pyr)
{
    int size = pyr.size();
    for (int i = 0; i < size; i++)
        normalize(pyr[i]);
}

void accumulate(const cv::Mat& src, const cv::Mat& srcWeight, cv::Mat& dst, cv::Mat& dstWeight)
{
    CV_Assert(src.data && src.type() == CV_16SC3 &&
              srcWeight.data && srcWeight.type() == CV_16SC1 &&
              dst.data && dst.type() == CV_32SC3 &&
              dstWeight.data && dstWeight.type() == CV_32SC1);
    cv::Size size = src.size();
    CV_Assert(srcWeight.size() == size && dst.size() == size && dstWeight.size() == size);

    int rows = src.rows, cols = src.cols;
    for (int i = 0; i < rows; i++)
    {
        const short* ptrSrcRow = src.ptr<short>(i);
        const short* ptrSrcWeightRow = srcWeight.ptr<short>(i);
        int* ptrDstRow = dst.ptr<int>(i);
        int* ptrDstWeightRow = dstWeight.ptr<int>(i);
        for (int j = 0; j < cols; j++)
        {
            ptrDstRow[0] += ptrSrcRow[0] * ptrSrcWeightRow[0];
            ptrDstRow[1] += ptrSrcRow[1] * ptrSrcWeightRow[0];
            ptrDstRow[2] += ptrSrcRow[2] * ptrSrcWeightRow[0];
            ptrDstWeightRow[0] += ptrSrcWeightRow[0];
            ptrDstRow += 3;
            ptrSrcRow += 3;
            ptrSrcWeightRow++;
            ptrDstWeightRow++;
        }
    }
}

void accumulate(const std::vector<cv::Mat>& srcPyr, const std::vector<cv::Mat>& srcWeightPyr, 
    std::vector<cv::Mat>& dstPyr, std::vector<cv::Mat>& dstWeightPyr)
{
    int size = srcPyr.size();
    CV_Assert(srcWeightPyr.size() == size && dstPyr.size() == size && dstWeightPyr.size() == size);
    
    for (int i = 0; i < size; i++)
        ::accumulate(srcPyr[i], srcWeightPyr[i], dstPyr[i], dstWeightPyr[i]);
}

void normalize(cv::Mat& image, const cv::Mat& weight)
{
    CV_Assert(image.data && image.type() == CV_32SC3 &&
        weight.data && weight.type() == CV_32SC1 &&
        image.size() == weight.size());
    int rows = image.rows, cols = image.cols;
    for (int i = 0; i < rows; i++)
    {
        int* ptr = image.ptr<int>(i);
        const int* ptrWeight = weight.ptr<int>(i);
        for (int j = 0; j < cols; j++)
        {
            // WARNING: remember to add 1 to avoid DIVIDE BY ZERO!!!
            int w = ptrWeight[0];
            if (!w) w++;
            ptr[0] = ptr[0] / w;
            ptr[1] = ptr[1] / w;
            ptr[2] = ptr[2] / w;            
            //int wh = w >> 1;
            //int val0 = ptr[0];
            //int val1 = ptr[1];
            //int val2 = ptr[2];
            // WARNING: expressiong "ptr[0] = (val0 + wh) / w" and so on may cause severe rounding error!!!
            //ptr[0] = (val0 >= 0) ? (val0 + wh) / w : (val0 - wh) / w;
            //ptr[1] = (val1 >= 0) ? (val1 + wh) / w : (val1 - wh) / w;
            //ptr[2] = (val2 >= 0) ? (val2 + wh) / w : (val2 - wh) / w;
            ptr += 3;
            ptrWeight++;
        }
    }
}

void normalize(std::vector<cv::Mat>& pyr, const std::vector<cv::Mat>& weightPyr)
{
    CV_Assert(pyr.size() == weightPyr.size());
    int size = pyr.size();
    for (int i = 0; i < size; i++)
        normalize(pyr[i], weightPyr[i]);
}

void calcDstImageAndAlpha(const cv::Mat& srcImage32S, const cv::Mat& srcAlpha32S,
    cv::Mat& dstImage, cv::Mat& dstAlpha)
{
    CV_Assert(srcImage32S.data && srcImage32S.type() == CV_32SC3 &&
        srcAlpha32S.data && srcAlpha32S.type() == CV_32SC1 &&
        srcImage32S.size() == srcAlpha32S.size());
    int rows = srcImage32S.rows, cols = srcImage32S.cols;
    dstImage.create(rows, cols, CV_16SC3);
    dstAlpha.create(rows, cols, CV_16SC1);
    for (int i = 0; i < rows; i++)
    {
        const int* ptrSrcImage = srcImage32S.ptr<int>(i);
        const int* ptrSrcAlpha = srcAlpha32S.ptr<int>(i);
        short* ptrDstImage = dstImage.ptr<short>(i);
        short* ptrDstAlpha = dstAlpha.ptr<short>(i);
        for (int j = 0; j < cols; j++)
        {
            if (*ptrSrcAlpha)
            {
                int alpha = *ptrSrcAlpha, halfAlpha = alpha >> 1;                
                int val0, val1, val2;
                val0 = ((ptrSrcImage[0] << 8) - ptrSrcImage[0]/* + halfAlpha*/) / alpha;
                val1 = ((ptrSrcImage[1] << 8) - ptrSrcImage[1]/* + halfAlpha*/) / alpha;
                val2 = ((ptrSrcImage[2] << 8) - ptrSrcImage[2]/* + halfAlpha*/) / alpha;
                ptrDstImage[0] = val0;
                ptrDstImage[1] = val1;
                ptrDstImage[2] = val2;
                *ptrDstAlpha = 256;
            }
            else
            {
                ptrDstImage[0] = 0;
                ptrDstImage[1] = 0;
                ptrDstImage[2] = 0;
                *ptrDstAlpha = 0;
            }
            ptrSrcImage += 3;
            ptrSrcAlpha++;
            ptrDstImage += 3;
            ptrDstAlpha++;
        }
    }
}

void pyrDownPrecise(const cv::Mat& srcImage, const cv::Mat& srcAlpha, bool horiWrap, cv::Mat& dstImage, cv::Mat& dstAlpha)
{
    CV_Assert(srcImage.data && srcImage.type() == CV_16SC3 &&
        srcAlpha.data && srcAlpha.type() == CV_16SC1 && srcImage.size() == srcAlpha.size());
    cv::Mat dstImage32S, dstAlpha32S;
    pyramidDownTo32S(srcImage, dstImage32S, cv::Size(), horiWrap ? cv::BORDER_WRAP : cv::BORDER_DEFAULT);
    pyramidDownTo32S(srcAlpha, dstAlpha32S, cv::Size(), horiWrap ? cv::BORDER_WRAP : cv::BORDER_DEFAULT);
    calcDstImageAndAlpha(dstImage32S, dstAlpha32S, dstImage, dstAlpha);
}

void createLaplacePyramidPrecise(const cv::Mat& image, const cv::Mat& alpha, int numLevels, bool horiWrap, std::vector<cv::Mat>& pyr)
{    
    pyr.resize(numLevels + 1);
    if (image.depth() == CV_16S)
        //image.copyTo(pyr[0]);
        pyr[0] = image;
    else
        image.convertTo(pyr[0], CV_16S);
    cv::Mat currAlpha;
    if (alpha.depth() == CV_16S)
        currAlpha = alpha;
    else
        alpha.convertTo(currAlpha, CV_16S);
    for (int i = 0; i < numLevels; ++i)
    {
        cv::Mat newAlpha;
        pyrDownPrecise(pyr[i], currAlpha, horiWrap, pyr[i + 1], newAlpha);
        currAlpha = newAlpha;
    }

    cv::Mat tmp;
    for (int i = 0; i < numLevels; ++i)
    {
        pyramidUp(pyr[i + 1], tmp, pyr[i].size(), horiWrap ? cv::BORDER_WRAP : cv::BORDER_DEFAULT);
        cv::subtract(pyr[i], tmp, pyr[i]);
    }
}

static const int NUM_LEVELS = 16;
static const int MIN_DEFAULT_LENGTH = 2;
static const int MAX_DEFAULT_LENGTH = 64;

int getTrueNumLevels(int width, int height, int maxLevels, int minLength)
{
    int numLevels = maxLevels < 0 ? NUM_LEVELS : (maxLevels > NUM_LEVELS ? NUM_LEVELS : maxLevels);
    int sideLength = minLength < 0 ? MIN_DEFAULT_LENGTH : (minLength > MAX_DEFAULT_LENGTH ? MAX_DEFAULT_LENGTH : minLength);
    int numLevelsWidth = log(double(width) / sideLength) / log(2.0);
    int numLevelsHeight = log(double(height) / sideLength) / log(2.0);
    numLevels = std::min(numLevels, std::min(numLevelsWidth, numLevelsHeight));
    return numLevels;
}

void multibandBlend(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& alpha1, const cv::Mat& alpha2,
    cv::Mat& mask1, const cv::Mat& mask2, bool horiWrap, int maxLevels, int minLength, cv::Mat& result)
{
    CV_Assert(image1.data && image1.type() == CV_8UC3 &&
        image2.data && image2.type() == CV_8UC3 &&
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
    std::vector<cv::Mat> resultPyr(numLevels + 1);
    resultPyr[0] = cv::Mat::zeros(image1.size(), CV_32SC3);
    for (int i = 1; i <= numLevels; i++)
        resultPyr[i] = cv::Mat::zeros((resultPyr[i - 1].rows + 1) / 2, (resultPyr[i - 1].cols + 1) / 2, CV_32SC3);

    cv::Mat mask(size, CV_16SC1);
    std::vector<cv::Mat> imagePyr, weightPyr;

    //static int count = 0;
    //count++;
    //char name[256];

    mask.setTo(0);
    mask.setTo(256, alpha1);
    createLaplacePyramidPrecise(image1, mask, numLevels, horiWrap, imagePyr);    
    mask.setTo(0);
    mask.setTo(256, mask1);
    createGaussPyramid(mask, numLevels, horiWrap, weightPyr);
    accumulate(imagePyr, weightPyr, resultPyr);
    //sprintf_s(name, "image1_number%d_level", count);
    //savePyramid(imagePyr, name, "tif");
    //sprintf_s(name, "mask1_number%d_level", count);
    //savePyramid(weightPyr, name, "tif");

    mask.setTo(0);
    mask.setTo(256, alpha2);
    createLaplacePyramidPrecise(image2, mask, numLevels, horiWrap, imagePyr);
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

void multibandBlendAnyMask(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& alpha1, const cv::Mat& alpha2,
    cv::Mat& mask1, const cv::Mat& mask2, bool horiWrap, int maxLevels, int minLength, cv::Mat& result)
{
    CV_Assert(image1.data && image1.type() == CV_8UC3 &&
        image2.data && image2.type() == CV_8UC3 &&
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
    std::vector<cv::Mat> resultPyr(numLevels + 1);
    resultPyr[0] = cv::Mat::zeros(image1.size(), CV_32SC3);
    for (int i = 1; i <= numLevels; i++)
        resultPyr[i] = cv::Mat::zeros((resultPyr[i - 1].rows + 1) / 2, (resultPyr[i - 1].cols + 1) / 2, CV_32SC3);
    std::vector<cv::Mat> resultWeightPyr(numLevels + 1);
    resultWeightPyr[0] = cv::Mat::zeros(image1.size(), CV_32SC1);
    for (int i = 1; i <= numLevels; i++)
        resultWeightPyr[i] = cv::Mat::zeros((resultWeightPyr[i - 1].rows + 1) / 2, (resultWeightPyr[i - 1].cols + 1) / 2, CV_32SC1);

    cv::Mat mask(size, CV_16SC1);
    std::vector<cv::Mat> imagePyr, weightPyr;

    //static int count = 0;
    //count++;
    //char name[256];

    mask.setTo(0);
    mask.setTo(256, alpha1);
    createLaplacePyramidPrecise(image1, mask, numLevels, horiWrap, imagePyr);    
    mask.setTo(0);
    mask.setTo(256, mask1);
    createGaussPyramid(mask, numLevels, horiWrap, weightPyr);
    accumulate(imagePyr, weightPyr, resultPyr, resultWeightPyr);
    //sprintf_s(name, "image1_number%d_level", count);
    //savePyramid(imagePyr, name, "tif");
    //sprintf_s(name, "mask1_number%d_level", count);
    //savePyramid(weightPyr, name, "tif");

    mask.setTo(0);
    mask.setTo(256, alpha2);
    createLaplacePyramidPrecise(image2, mask, numLevels, horiWrap, imagePyr);
    mask.setTo(0);
    mask.setTo(256, mask2);
    createGaussPyramid(mask, numLevels, horiWrap, weightPyr);
    accumulate(imagePyr, weightPyr, resultPyr, resultWeightPyr);
    //sprintf_s(name, "image2_number%d_level", count);
    //savePyramid(imagePyr, name, "tif");
    //sprintf_s(name, "mask2_number%d_level", count);
    //savePyramid(weightPyr, name, "tif");

    normalize(resultPyr, resultWeightPyr);
    restoreImageFromLaplacePyramid(resultPyr, horiWrap);
    resultPyr[0].convertTo(result, CV_8U);
}

void multibandBlend(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& alphas,
    const std::vector<cv::Mat>& masks, int maxLevels, int minLength, cv::Mat& result)
{
    CV_Assert(checkSize(images) && checkSize(alphas) && checkSize(masks));
    CV_Assert(images[0].size() == alphas[0].size() && images[0].size() == masks[0].size());
    CV_Assert(checkType(images, CV_8UC3) && checkType(alphas, CV_8UC1) && checkType(masks, CV_8UC1));

    int numImages = images.size();
    int width = images[0].cols, height = images[0].rows;
    cv::Size size = images[0].size();
    int numLevels = getTrueNumLevels(width, height, maxLevels, minLength);

    std::vector<cv::Mat> resultPyr(numLevels + 1);
    resultPyr[0] = cv::Mat::zeros(size, CV_32SC3);
    for (int i = 1; i <= numLevels; i++)
        resultPyr[i] = cv::Mat::zeros((resultPyr[i - 1].rows + 1) / 2, (resultPyr[i - 1].cols + 1) / 2, CV_32SC3);

    std::vector<cv::Mat> imagePyr, weightPyr;
    cv::Mat aux(height, width, CV_16SC1);
    for (int i = 0; i < numImages; i++)
    {
        aux.setTo(0);
        aux.setTo(256, alphas[i]);
        createLaplacePyramidPrecise(images[i], aux, numLevels, true, imagePyr);
        aux.setTo(0);
        aux.setTo(256, masks[i]);
        createGaussPyramid(aux, numLevels, true, weightPyr);
        accumulate(imagePyr, weightPyr, resultPyr);
    }
    normalize(resultPyr);
    restoreImageFromLaplacePyramid(resultPyr, true);
    resultPyr[0].convertTo(result, CV_8U);
}

void multibandBlendAnyMask(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& alphas,
    const std::vector<cv::Mat>& masks, int maxLevels, int minLength, cv::Mat& result)
{
    CV_Assert(checkSize(images) && checkSize(alphas) && checkSize(masks));
    CV_Assert(images[0].size() == alphas[0].size() && images[0].size() == masks[0].size());
    CV_Assert(checkType(images, CV_8UC3) && checkType(alphas, CV_8UC1) && checkType(masks, CV_8UC1));

    int numImages = images.size();
    int width = images[0].cols, height = images[0].rows;
    cv::Size size = images[0].size();
    int numLevels = getTrueNumLevels(width, height, maxLevels, minLength);

    std::vector<cv::Mat> resultPyr(numLevels + 1);
    resultPyr[0] = cv::Mat::zeros(size, CV_32SC3);
    for (int i = 1; i <= numLevels; i++)
        resultPyr[i] = cv::Mat::zeros((resultPyr[i - 1].rows + 1) / 2, (resultPyr[i - 1].cols + 1) / 2, CV_32SC3);
    std::vector<cv::Mat> resultWeightPyr(numLevels + 1);
    resultWeightPyr[0] = cv::Mat::zeros(size, CV_32SC1);
    for (int i = 1; i <= numLevels; i++)
        resultWeightPyr[i] = cv::Mat::zeros((resultWeightPyr[i - 1].rows + 1) / 2, (resultWeightPyr[i - 1].cols + 1) / 2, CV_32SC1);

    std::vector<cv::Mat> imagePyr, weightPyr;
    cv::Mat aux(height, width, CV_16SC1);
    cv::Mat alpha = cv::Mat::zeros(height, width, CV_8UC1);
    for (int i = 0; i < numImages; i++)
    {        
        aux.setTo(0);
        aux.setTo(256, alphas[i]);
        createLaplacePyramidPrecise(images[i], aux, numLevels, true, imagePyr);
        aux.setTo(0);
        aux.setTo(256, masks[i]);
        createGaussPyramid(aux, numLevels, true, weightPyr);
        accumulate(imagePyr, weightPyr, resultPyr, resultWeightPyr);
        alpha |= alphas[i];
    }
    normalize(resultPyr, resultWeightPyr);
    restoreImageFromLaplacePyramid(resultPyr, true);
    resultPyr[0].convertTo(result, CV_8U);
    cv::bitwise_not(alpha, alpha);
    result.setTo(0, alpha);
}

bool TilingMultibandBlend::prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength)
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
    rows = currRows;
    cols = currCols;
    numImages = currNumMasks;

    getNonIntersectingMasks(masks, uniqueMasks);

    //cv::Mat result(rows, cols, CV_8UC3);
    //cv::Scalar colors[] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
    //                       cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255)};
    //for (int i = 0; i < numImages; i++)
    //{
    //    cv::Mat show = cv::Mat::zeros(rows, cols, CV_8UC1);
    //    show.setTo(128, masks[i]);
    //    show.setTo(255, uniqueMasks[i]);
    //    result.setTo(colors[i], uniqueMasks[i]);
    //    cv::imshow("mask", show);
    //    cv::waitKey(0);
    //}
    //cv::imshow("color", result);
    //cv::waitKey(0);

    numLevels = getTrueNumLevels(cols, rows, maxLevels, minLength);

    resultPyr.resize(numLevels + 1);
    resultPyr[0] = cv::Mat::zeros(rows, cols, CV_32SC3);
    for (int i = 1; i <= numLevels; i++)
        resultPyr[i] = cv::Mat::zeros((resultPyr[i - 1].rows + 1) / 2, (resultPyr[i - 1].cols + 1) / 2, CV_32SC3);

    resultWeightPyr.resize(numLevels + 1);
    resultWeightPyr[0] = cv::Mat::zeros(rows, cols, CV_32SC1);
    for (int i = 1; i <= numLevels; i++)
        resultWeightPyr[i] = cv::Mat::zeros((resultWeightPyr[i - 1].rows + 1) / 2, (resultWeightPyr[i - 1].cols + 1) / 2, CV_32SC1);

    success = true;
    return true;
}

void TilingMultibandBlend::tile(const cv::Mat& image, const cv::Mat& mask, int index)
{
    if (!success)
        return;

    CV_Assert(image.data && (image.type() == CV_8UC3 || image.type() == CV_16SC3) && 
        image.rows == rows && image.cols == cols &&
        mask.data && mask.type() == CV_8UC1 && 
        mask.rows == rows && mask.cols == cols &&
        index >= 0 && index < numImages);

    std::vector<cv::Mat> imagePyr, weightPyr;
    cv::Mat aux(rows, cols, CV_16SC1);
    aux.setTo(0);
    aux.setTo(256, mask);
    createLaplacePyramidPrecise(image, aux, numLevels, true, imagePyr);
    aux.setTo(0);
    aux.setTo(256, uniqueMasks[index]);
    createGaussPyramid(aux, numLevels, true, weightPyr);
    accumulate(imagePyr, weightPyr, resultPyr, resultWeightPyr);
}

void TilingMultibandBlend::composite(cv::Mat& blendImage)
{
    if (!success)
        return;

    normalize(resultPyr, resultWeightPyr);
    restoreImageFromLaplacePyramid(resultPyr, true);
    resultPyr[0].convertTo(blendImage, CV_8U);

    // Remember to set resultPyr to zero, 
    // otherwise next round of tile and composite will produce incorrect result!!!!!
    for (int i = 0; i <= numLevels; i++)
    {
        resultPyr[i].setTo(0);
        resultWeightPyr[i].setTo(0);
    }
}

void TilingMultibandBlend::blend(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, cv::Mat& blendImage)
{
    if (!success)
        return;

    CV_Assert(images.size() == numImages && masks.size() == numImages);

    for (int i = 0; i < numImages; i++)
        tile(images[i], masks[i], i);
    composite(blendImage);
}

void TilingMultibandBlend::blendAndCompensate(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, cv::Mat& blendImage)
{
    if (!success)
        return;

    CV_Assert(images.size() == numImages && masks.size() == numImages);
    for (int i = 0; i < numImages; i++)
    {
        CV_Assert(images[i].data && (images[i].type() == CV_8UC3 || images[i].type() == CV_16SC3) &&
            images[i].rows == rows && images[i].cols == cols &&
            masks[i].data && masks[i].type() == CV_8UC1 &&
            masks[i].rows == rows && masks[i].cols == cols);
    }

    cv::Mat remain = cv::Mat::zeros(rows, cols, CV_8UC1);
    std::vector<cv::Mat> adjustMasks(numImages);
    for (int i = 0; i < numImages; i++)
    {
        adjustMasks[i] = masks[i] & uniqueMasks[i];
        remain |= adjustMasks[i];
    }
    cv::bitwise_not(remain, remain);

    cv::Mat matchArea;
    for (int i = 0; i < numImages; i++)
    {
        matchArea = masks[i] & remain;
        adjustMasks[i] |= matchArea;
        remain -= matchArea;
    }
    remain.release();
    matchArea.release();

    std::vector<cv::Mat> imagePyr, weightPyr;
    cv::Mat aux(rows, cols, CV_16SC1);
    for (int i = 0; i < numImages; i++)
    {
        aux.setTo(0);
        aux.setTo(256, masks[i]);
        createLaplacePyramidPrecise(images[i], aux, numLevels, true, imagePyr);
        aux.setTo(0);
        aux.setTo(256, adjustMasks[i]);
        createGaussPyramid(aux, numLevels, true, weightPyr);
        accumulate(imagePyr, weightPyr, resultPyr, resultWeightPyr);
    }

    normalize(resultPyr, resultWeightPyr);
    restoreImageFromLaplacePyramid(resultPyr, true);
    resultPyr[0].convertTo(blendImage, CV_8U);

    // Remember to set resultPyr to zero, 
    // otherwise next round of tile and composite will produce incorrect result!!!!!
    for (int i = 0; i <= numLevels; i++)
    {
        resultPyr[i].setTo(0);
        resultWeightPyr[i].setTo(0);
    }
        
}

static void setAlpha16SAccordingToAlpha32S(const cv::Mat& alpha32S, cv::Mat& alpha16S)
{
    CV_Assert(alpha32S.data && alpha32S.type() == CV_32SC1);
    int rows = alpha32S.rows, cols = alpha32S.cols;
    alpha16S.create(rows, cols, CV_16SC1);
    alpha16S.setTo(0);
    for (int i = 0; i < rows; i++)
    {
        const int* ptrSrc = alpha32S.ptr<int>(i);
        short* ptrDst = alpha16S.ptr<short>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrSrc[j])
                ptrDst[j] = 256;
        }
    }
}

static void calcDstImage(const cv::Mat& srcImage32S, const cv::Mat& srcAlpha32S, cv::Mat& dstImage)
{
    CV_Assert(srcImage32S.data && srcImage32S.type() == CV_32SC3 &&
        srcAlpha32S.data && srcAlpha32S.type() == CV_32SC1 &&
        srcImage32S.size() == srcAlpha32S.size());
    int rows = srcImage32S.rows, cols = srcImage32S.cols;
    dstImage.create(rows, cols, CV_16SC3);
    for (int i = 0; i < rows; i++)
    {
        const int* ptrSrcImage = srcImage32S.ptr<int>(i);
        const int* ptrSrcAlpha = srcAlpha32S.ptr<int>(i);
        short* ptrDstImage = dstImage.ptr<short>(i);
        for (int j = 0; j < cols; j++)
        {
            if (*ptrSrcAlpha)
            {
                int alpha = *ptrSrcAlpha, halfAlpha = alpha >> 1;
                int val0, val1, val2;
                val0 = ((ptrSrcImage[0] << 8) - ptrSrcImage[0]/* + halfAlpha*/) / alpha;
                val1 = ((ptrSrcImage[1] << 8) - ptrSrcImage[1]/* + halfAlpha*/) / alpha;
                val2 = ((ptrSrcImage[2] << 8) - ptrSrcImage[2]/* + halfAlpha*/) / alpha;
                ptrDstImage[0] = val0;
                ptrDstImage[1] = val1;
                ptrDstImage[2] = val2;
            }
            else
            {
                ptrDstImage[0] = 0;
                ptrDstImage[1] = 0;
                ptrDstImage[2] = 0;
            }
            ptrSrcImage += 3;
            ptrSrcAlpha++;
            ptrDstImage += 3;
        }
    }
}

void restoreImageFromLaplacePyramid(std::vector<cv::Mat>& pyr, bool horiWrap, std::vector<cv::Mat>& upPyr)
{
    if (pyr.empty())
        return;
    upPyr.resize(pyr.size());
    for (size_t i = pyr.size() - 1; i > 0; --i)
    {
        pyramidUp(pyr[i], upPyr[i - 1], pyr[i - 1].size(), horiWrap ? cv::BORDER_WRAP : cv::BORDER_DEFAULT);
        cv::add(upPyr[i - 1], pyr[i - 1], pyr[i - 1]);
    }
}

void restoreImageFromLaplacePyramid(std::vector<cv::Mat>& pyr, bool horiWrap, std::vector<cv::Mat>& upPyr, 
    std::vector<unsigned char>& aux1, std::vector<unsigned char>& aux2)
{
    if (pyr.empty())
        return;
    upPyr.resize(pyr.size());
    for (size_t i = pyr.size() - 1; i > 0; --i)
    {
        pyramidUp(pyr[i], upPyr[i - 1], aux1, aux2, pyr[i - 1].size(), horiWrap ? cv::BORDER_WRAP : cv::BORDER_DEFAULT);
        cv::add(upPyr[i - 1], pyr[i - 1], pyr[i - 1]);
    }
}

static void getPyramidLevelSizes(std::vector<cv::Size>& sizes, int rows, int cols, int numLevels)
{
    sizes.resize(numLevels + 1);
    sizes[0] = cv::Size(cols, rows);
    for (int i = 1; i <= numLevels; i++)
        sizes[i] = cv::Size((sizes[i - 1].width + 1) / 2, (sizes[i - 1].height + 1) / 2);
}

static void allocMemoryForImage32SPyrAndImageUpPyr(const std::vector<cv::Size>& sizes,
    std::vector<cv::Mat>& image32SPyr, std::vector<cv::Mat>& imageUpPyr)
{
    int numLevels = sizes.size() - 1;
    cv::Mat mem(sizes[0], CV_16SC3);

    imageUpPyr.resize(numLevels + 1);
    imageUpPyr[0] = mem;
    for (int i = 1; i < numLevels; i++)
        imageUpPyr[i] = cv::Mat(sizes[i], CV_16SC3, mem.data);

    image32SPyr.resize(numLevels + 1);
    for (int i = 1; i <= numLevels; i++)
        image32SPyr[i] = cv::Mat(sizes[i], CV_32SC3, mem.data);
}

static void allocMemoryForResultPyrAndResultUpPyr(const std::vector<cv::Size>& sizes,
    std::vector<cv::Mat>& resultPyr, std::vector<cv::Mat>& resultUpPyr)
{
    int numLevels = sizes.size() - 1;

    resultPyr.resize(numLevels + 1);
    for (int i = 0; i <= numLevels; i++)
        resultPyr[i].create(sizes[i], CV_32SC3);

    cv::Mat mem(sizes[0], CV_32SC3);
    resultUpPyr.resize(numLevels + 1);
    resultUpPyr[0] = mem;
    for (int i = 1; i < numLevels; i++)
        resultUpPyr[i] = cv::Mat(sizes[i], CV_32SC3, mem.data);
}

void multiply(const cv::Mat& src, const cv::Mat& weight, cv::Mat& dst)
{
    CV_Assert(src.data && src.type() == CV_16SC3 &&
        weight.data && weight.type() == CV_16SC1 &&
        src.size() == weight.size());

    int rows = src.rows, cols = src.cols;
    dst.create(rows, cols, CV_32SC3);
    for (int i = 0; i < rows; i++)
    {
        const short* ptrSrcRow = src.ptr<short>(i);
        const short* ptrWeightRow = weight.ptr<short>(i);
        int* ptrDstRow = dst.ptr<int>(i);
        for (int j = 0; j < cols; j++)
        {
            ptrDstRow[0] = ptrSrcRow[0] * ptrWeightRow[0];
            ptrDstRow[1] = ptrSrcRow[1] * ptrWeightRow[0];
            ptrDstRow[2] = ptrSrcRow[2] * ptrWeightRow[0];
            ptrDstRow += 3;
            ptrSrcRow += 3;
            ptrWeightRow++;
        }
    }
}

void multiply(const std::vector<cv::Mat>& pyr, const std::vector<cv::Mat>& weightPyr, std::vector<cv::Mat>& weightedPyr)
{
    CV_Assert(pyr.size() == weightPyr.size());

    int size = pyr.size();
    weightedPyr.resize(size);
    for (int i = 0; i < size; i++)
        multiply(pyr[i], weightPyr[i], weightedPyr[i]);
}

void add(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst)
{
    CV_Assert(src.size() == dst.size());

    int size = src.size();
    for (int i = 0; i < size; i++)
        cv::add(src[i], dst[i], dst[i]);
}

void accumulateWeight(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(src.data && src.type() == CV_16SC1 &&
        dst.data && dst.type() == CV_32SC1 && src.size() == dst.size());

    int rows = src.rows, cols = src.cols;
    for (int i = 0; i < rows; i++)
    {
        const short* ptrSrc = src.ptr<short>(i);
        int* ptrDst = dst.ptr<int>(i);
        for (int j = 0; j < cols; j++)
        {
            *ptrDst += *ptrSrc;
            ptrSrc++;
            ptrDst++;
        }
    }
}

void accumulateWeight(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst)
{
    CV_Assert(src.size() == dst.size());
    int size = src.size();
    for (int i = 0; i < size; i++)
        accumulateWeight(src[i], dst[i]);
}

bool TilingMultibandBlendFast::prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength)
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
    rows = currRows;
    cols = currCols;
    numImages = currNumMasks;

    getNonIntersectingMasks(masks, uniqueMasks);

    numLevels = getTrueNumLevels(cols, rows, maxLevels, minLength);

    std::vector<cv::Size> sizes;
    getPyramidLevelSizes(sizes, rows, cols, numLevels);
    allocMemoryForImage32SPyrAndImageUpPyr(sizes, image32SPyr, imageUpPyr);
    allocMemoryForResultPyrAndResultUpPyr(sizes, resultPyr, resultUpPyr);

    cv::Mat aux(rows, cols, CV_16SC1);

    weightPyrs.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        aux.setTo(0);
        aux.setTo(256, uniqueMasks[i]);
        cv::Mat base = aux.clone();
        createGaussPyramid(base, numLevels, true, weightPyrs[i]);
    }

    alphaPyrs.resize(numImages);
    std::vector<cv::Mat> tempAlphaPyr(numLevels + 1);
    for (int i = 0; i < numImages; i++)
    {
        alphaPyrs[i].resize(numLevels + 1);
        aux.setTo(0);
        aux.setTo(256, masks[i]);
        tempAlphaPyr[0] = aux.clone();
        for (int j = 0; j < numLevels; j++)
        {
            pyramidDownTo32S(tempAlphaPyr[j], alphaPyrs[i][j + 1], cv::Size(), cv::BORDER_WRAP);
            tempAlphaPyr[j + 1].create(alphaPyrs[i][j + 1].size(), CV_16SC1);
            setAlpha16SAccordingToAlpha32S(alphaPyrs[i][j + 1], tempAlphaPyr[j + 1]);
        }
    }

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
            resultWeightPyr[i].setTo(0);
        }
        for (int i = 0; i < numImages; i++)
            accumulateWeight(weightPyrs[i], resultWeightPyr);
        maskNot = ~mask;
    }

    success = true;
    return true;
}

void TilingMultibandBlendFast::blend(const std::vector<cv::Mat>& images, cv::Mat& blendImage)
{
    if (!success)
        return;

    CV_Assert(images.size() == numImages);

    for (int i = 0; i < numImages; i++)
    {
        CV_Assert(images[i].data && (images[i].type() == CV_8UC3 || images[i].type() == CV_16SC3) &&
            images[i].rows == rows && images[i].cols == cols);
    }

    for (int i = 0; i <= numLevels; i++)
        resultPyr[i].setTo(0);

    imagePyr.resize(numLevels + 1);
    for (int i = 0; i < numImages; i++)
    {
        if (images[i].type() == CV_8UC3)
            images[i].convertTo(imagePyr[0], CV_16S);
        else if (images[i].type() == CV_16SC3)
            imagePyr[0] = images[i];
        for (int j = 0; j < numLevels; j++)
        {
            pyramidDownTo32S(imagePyr[j], image32SPyr[j + 1], cv::Size(), cv::BORDER_WRAP);
            calcDstImage(image32SPyr[j + 1], alphaPyrs[i][j + 1], imagePyr[j + 1]);
        }
        for (int j = 0; j < numLevels; j++)
        {
            pyramidUp(imagePyr[j + 1], imageUpPyr[j], imagePyr[j].size(), cv::BORDER_WRAP);
            cv::subtract(imagePyr[j], imageUpPyr[j], imagePyr[j]);
        }
        accumulate(imagePyr, weightPyrs[i], resultPyr);
    }

    if (fullMask)
        normalize(resultPyr);
    else
        normalize(resultPyr, resultWeightPyr);
    restoreImageFromLaplacePyramid(resultPyr, true, resultUpPyr);
    resultPyr[0].convertTo(blendImage, CV_8U);
    if (!fullMask)
        blendImage.setTo(0, maskNot);
}

void TilingMultibandBlendFast::blend(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, cv::Mat& blendImage)
{
    if (!success)
        return;

    CV_Assert(images.size() == numImages && masks.size() == numImages);

    for (int i = 0; i < numImages; i++)
    {
        CV_Assert(images[i].data && (images[i].type() == CV_8UC3 || images[i].type() == CV_16SC3) &&
            images[i].rows == rows && images[i].cols == cols);
        CV_Assert(masks[i].data && masks[i].type() == CV_8UC1 &&
            masks[i].rows == rows && masks[i].cols == cols);
    }

    customMaskNot.create(rows, cols, CV_8UC1);
    customMaskNot.setTo(0);
    for (int i = 0; i < numImages; i++)
        customMaskNot |= masks[i];
    bool customFullMask = cv::countNonZero(customMaskNot) == rows * cols;

    customAux.create(rows, cols, CV_16SC1);
    customWeightPyrs.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        customAux.setTo(0);
        customAux.setTo(256, masks[i]);
        cv::Mat base = customAux.clone();
        createGaussPyramid(base, numLevels, true, customWeightPyrs[i]);
    }

    customResultWeightPyr.resize(numLevels + 1);
    for (int i = 0; i < numLevels + 1; i++)
    {
        customResultWeightPyr[i].create(customWeightPyrs[0][i].size(), CV_32SC1);
        customResultWeightPyr[i].setTo(0);
    }
    for (int i = 0; i < numImages; i++)
        accumulateWeight(customWeightPyrs[i], customResultWeightPyr);
    cv::bitwise_not(customMaskNot, customMaskNot);

    for (int i = 0; i <= numLevels; i++)
        resultPyr[i].setTo(0);

    imagePyr.resize(numLevels + 1);
    for (int i = 0; i < numImages; i++)
    {
        if (images[i].type() == CV_8UC3)
            images[i].convertTo(imagePyr[0], CV_16S);
        else if (images[i].type() == CV_16SC3)
            imagePyr[0] = images[i];
        for (int j = 0; j < numLevels; j++)
        {
            pyramidDownTo32S(imagePyr[j], image32SPyr[j + 1], cv::Size(), cv::BORDER_WRAP);
            calcDstImage(image32SPyr[j + 1], alphaPyrs[i][j + 1], imagePyr[j + 1]);
        }
        for (int j = 0; j < numLevels; j++)
        {
            pyramidUp(imagePyr[j + 1], imageUpPyr[j], imagePyr[j].size(), cv::BORDER_WRAP);
            cv::subtract(imagePyr[j], imageUpPyr[j], imagePyr[j]);
        }
        accumulate(imagePyr, customWeightPyrs[i], resultPyr);
    }

    normalize(resultPyr, customResultWeightPyr);
    restoreImageFromLaplacePyramid(resultPyr, true, resultUpPyr);
    resultPyr[0].convertTo(blendImage, CV_8U);
    if (!customFullMask)
        blendImage.setTo(0, maskNot);
}

void TilingMultibandBlendFast::getUniqueMasks(std::vector<cv::Mat>& masks)
{
    if (success)
        masks = uniqueMasks;
    else
        masks.clear();
}

TilingMultibandBlendFastParallel::~TilingMultibandBlendFastParallel()
{
    endThreads();
}

bool TilingMultibandBlendFastParallel::prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength)
{
    endThreads();
    init();

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
    rows = currRows;
    cols = currCols;
    numImages = currNumMasks;

    std::vector<cv::Mat> uniqueMasks;
    getNonIntersectingMasks(masks, uniqueMasks);

    numLevels = getTrueNumLevels(cols, rows, maxLevels, minLength);

    std::vector<cv::Size> sizes;
    getPyramidLevelSizes(sizes, rows, cols, numLevels);
    image32SPyrs.resize(numImages);
    imageUpPyrs.resize(numImages);
    for (int i = 0; i < numImages; i++)
        allocMemoryForImage32SPyrAndImageUpPyr(sizes, image32SPyrs[i], imageUpPyrs[i]);
    allocMemoryForResultPyrAndResultUpPyr(sizes, resultPyr, resultUpPyr);

    cv::Mat aux(rows, cols, CV_16SC1);

    weightPyrs.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        aux.setTo(0);
        aux.setTo(256, uniqueMasks[i]);
        cv::Mat base = aux.clone();
        createGaussPyramid(base, numLevels, true, weightPyrs[i]);
    }

    alphaPyrs.resize(numImages);
    std::vector<cv::Mat> tempAlphaPyr(numLevels + 1);
    for (int i = 0; i < numImages; i++)
    {
        alphaPyrs[i].resize(numLevels + 1);
        aux.setTo(0);
        aux.setTo(256, masks[i]);
        tempAlphaPyr[0] = aux.clone();
        for (int j = 0; j < numLevels; j++)
        {
            pyramidDownTo32S(tempAlphaPyr[j], alphaPyrs[i][j + 1], cv::Size(), cv::BORDER_WRAP);
            tempAlphaPyr[j + 1].create(alphaPyrs[i][j + 1].size(), CV_16SC1);
            setAlpha16SAccordingToAlpha32S(alphaPyrs[i][j + 1], tempAlphaPyr[j + 1]);
        }
    }

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
            resultWeightPyr[i].setTo(0);
        }
        for (int i = 0; i < numImages; i++)
            accumulateWeight(weightPyrs[i], resultWeightPyr);
        maskNot = ~mask;
    }

    imageHeaders.resize(numImages);
    imagePyrs.resize(numImages);

    rowBuffers.resize(numImages);
    tabBuffers.resize(numImages);

    threadEnd = false;
    threads.clear();
    threads.resize(numImages);
    for (int i = 0; i < numImages; i++)
        threads[i].reset(new std::thread(&TilingMultibandBlendFastParallel::buildPyramid, this, i));

    success = true;
    return true;
}

void TilingMultibandBlendFastParallel::blend(const std::vector<cv::Mat>& images, cv::Mat& blendImage)
{
    if (!success)
        return;

    CV_Assert(images.size() == numImages);

    for (int i = 0; i < numImages; i++)
    {
        CV_Assert(images[i].data && (images[i].type() == CV_8UC3 || images[i].type() == CV_16SC3) &&
            images[i].rows == rows && images[i].cols == cols);
    }

    for (int i = 0; i < numImages; i++)
        imageHeaders[i] = images[i];

    buildCount.store(0);
    cvBuildPyr.notify_all();

    // It seems that unconditional waiting causes deadlock if this library
    // runs on a less powerful CPU.
    // After I used conditional wait, deadlock no longer happened.
    // The cause of deadlock has not been really discovered.
    // I do not recommend using TilingMultibandBlendFastParallel,
    // it runs just a little faster than TilingMultibandBlendFast but 
    // consumes more memory.
    // I want to mark this class as DEPRECATED.
    {
        std::unique_lock<std::mutex> lk(mtxAccum);
        cvAccum.wait(lk, [this] { return buildCount.load() == numImages; });
    }

    for (int i = 0; i <= numLevels; i++)
        resultPyr[i].setTo(0);
    // Acording to run time test, weighting imagrPyr to a weightedPyr in respective thread 
    // and then add all weightedPyrs in the main thread will slow down the speed.
    for (int i = 0; i < numImages; i++)
        accumulate(imagePyrs[i], weightPyrs[i], resultPyr);
    if (fullMask)
        normalize(resultPyr);
    else
        normalize(resultPyr, resultWeightPyr);
    restoreImageFromLaplacePyramid(resultPyr, true, resultUpPyr, restoreRowBuffer, restoreTabBuffer);
    resultPyr[0].convertTo(blendImage, CV_8U);
    if (!fullMask)
        blendImage.setTo(0, maskNot);
}

void TilingMultibandBlendFastParallel::buildPyramid(int index)
{
    const cv::Mat& image = imageHeaders[index];
    std::vector<cv::Mat>& imagePyr = imagePyrs[index];
    std::vector<cv::Mat>& image32SPyr = image32SPyrs[index];
    std::vector<cv::Mat>& imageUpPyr = imageUpPyrs[index];
    std::vector<cv::Mat>& alphaPyr = alphaPyrs[index];
    std::vector<cv::Mat>& weightPyr = weightPyrs[index];
    std::vector<unsigned char>& rowBuffer = rowBuffers[index];
    std::vector<unsigned char>& tabBuffer = tabBuffers[index];
    while (true)
    {
        {
            std::unique_lock<std::mutex> lk(mtxBuildPyr);
            cvBuildPyr.wait(lk);
        }
        if (threadEnd)
            break;

        imagePyr.resize(numLevels + 1);
        if (image.type() == CV_8UC3)
            image.convertTo(imagePyr[0], CV_16S);
        else if (image.type() == CV_16SC3)
            imagePyr[0] = image;
        for (int j = 0; j < numLevels; j++)
        {
            pyramidDownTo32S(imagePyr[j], image32SPyr[j + 1], rowBuffer, tabBuffer, cv::Size(), cv::BORDER_WRAP);
            calcDstImage(image32SPyr[j + 1], alphaPyr[j + 1], imagePyr[j + 1]);
        }
        for (int j = 0; j < numLevels; j++)
        {
            pyramidUp(imagePyr[j + 1], imageUpPyr[j], rowBuffer, tabBuffer, imagePyr[j].size(), cv::BORDER_WRAP);
            cv::subtract(imagePyr[j], imageUpPyr[j], imagePyr[j]);
        }

        buildCount.fetch_add(1);
        if (buildCount.load() == numImages)
            cvAccum.notify_one();
    }
}

void TilingMultibandBlendFastParallel::endThreads()
{
    if (!threadEnd)
    {
        threadEnd = true;
        cvBuildPyr.notify_all();
        for (int i = 0; i < numImages; i++)
        {
            if (threads[i]->joinable())
                threads[i]->join();
        }
    }
}

void TilingMultibandBlendFastParallel::init()
{
    numImages = 0;
    rows = 0;
    cols = 0;
    numLevels = 0;
    success = false;

    threads.clear();
    threadEnd = true;
    buildCount.store(0);
}