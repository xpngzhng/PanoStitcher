#include "ZBlend.h"
#include "ZBlendAlgo.h"
#include "ZReproject.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

static void getExtendedMasks(const std::vector<cv::Mat>& masks, int radius, std::vector<cv::Mat>& extendedMasks)
{
    int numImages = masks.size();
    std::vector<cv::Mat> uniqueMasks(numImages);
    getNonIntersectingMasks(masks, uniqueMasks);

    std::vector<cv::Mat> compMasks(numImages);
    for (int i = 0; i < numImages; i++)
        cv::bitwise_not(masks[i], compMasks[i]);

    std::vector<cv::Mat> blurMasks(numImages);
    cv::Mat intersect;
    int validCount, r;
    for (r = radius; r > 0; r = r - 2)
    {
        cv::Size blurSize(r * 2 + 1, r * 2 + 1);
        double sigma = r / 3.0;
        validCount = 0;
        for (int i = 0; i < numImages; i++)
        {
            cv::GaussianBlur(uniqueMasks[i], blurMasks[i], blurSize, sigma, sigma);
            cv::bitwise_and(blurMasks[i], compMasks[i], intersect);
            if (cv::countNonZero(intersect) == 0)
                validCount++;
        }
        if (validCount == numImages)
            break;
    }

    extendedMasks.resize(numImages);
    for (int i = 0; i < numImages; i++)
        extendedMasks[i] = (blurMasks[i] != 0);
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

static void calcHistSpecLUT(const cv::Mat& src, const cv::Mat& srcMask,
    const cv::Mat& dst, const cv::Mat& dstMask, std::vector<unsigned char>& lutSrcToDst)
{
    CV_Assert(src.data && src.type() == CV_8UC1 && srcMask.data && srcMask.type() == CV_8UC1 &&
        dst.data && dst.type() == CV_8UC1 && dstMask.data && dstMask.type() == CV_8UC1 &&
        src.size() == srcMask.size() && dst.size() == dstMask.size() && src.size() == dst.size());

    cv::Mat intersect = srcMask & dstMask;
    std::vector<double> srcAccumHist, dstAccumHist;
    calcAccumHist(src, intersect, srcAccumHist);
    calcAccumHist(dst, intersect, dstAccumHist);
    histSpecification(srcAccumHist, dstAccumHist, lutSrcToDst);
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
            int sqrDiff = (i - halfHeight / 2) * (i - halfHeight / 2) + (j - halfWidth) * (j - halfWidth);
            ptr[j] = 1.0 / (1 - alpha * sqrDiff);
        }
    }
}

inline int clamp0255(int val)
{
    return val < 0 ? 0 : (val > 255 ? 255 : val);
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

int main1()
{
    std::vector<std::string> imagePaths;
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-00.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-01.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-02.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-03.jpg");

    int numImages = imagePaths.size();
    std::vector<cv::Mat> origImages(numImages), images(numImages), maps(numImages), masks(numImages), grayImages(numImages);

    for (int i = 0; i < numImages; i++)
        origImages[i] = cv::imread(imagePaths[i]);

    cv::Mat scaleImage;
    calcScale(origImages[0].size(), 0.4, scaleImage);
    //for (int k = 0; k < numImages; k++)
    //    mulScale(origImages[k], scaleImage);
    mulScale(origImages[2], scaleImage);

    for (int i = 0; i < numImages; i++)
    {
        cv::imshow("orig image", origImages[i]);
        cv::waitKey(0);
    }

    std::vector<PhotoParam> params;
    loadPhotoParamFromPTS("F:\\panoimage\\detuoffice\\4p.pts", params);
    getReprojectMapsAndMasks(params, origImages[0].size(), cv::Size(2048, 1024), maps, masks);
    reprojectParallel(origImages, images, maps);

    getExtendedMasks(masks, 100, masks);

    for (int i = 0; i < numImages; i++)
        cv::cvtColor(images[i], grayImages[i], CV_BGR2GRAY);

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
            if (cv::countNonZero(refMask & masks[workIndexes[i]]))
            {
                adoptIndexes.push_back(workIndexes[i]);
                calcHistSpecLUT(grayImages[workIndexes[i]], masks[workIndexes[i]], refGrayImage, refMask, luts[workIndexes[i]]);
                transform(images[workIndexes[i]], images[workIndexes[i]], luts[workIndexes[i]], masks[workIndexes[i]]);
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

        TilingMultibandBlendFast blender;
        blender.prepare(srcMasks, 20, 2);
        blender.blend(srcImages, refImage);
        for (int i = 0; i < adoptIndexes.size(); i++)
            refMask |= masks[adoptIndexes[i]];
        refImage.setTo(0, ~refMask);
        cv::cvtColor(refImage, refGrayImage, CV_BGR2GRAY);
        cv::imshow("ref image", refImage);
        cv::imshow("ref mask", refMask);
        cv::waitKey(0);

        workIndexes = remainIndexes;
        workImages.clear();
        for (int i = 0; i < workIndexes.size(); i++)
            workImages.push_back(images[workIndexes[i]]);
    }

    TilingLinearBlend blender;
    blender.prepare(masks, 100);
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
            if (cv::countNonZero(mask))
            {
                intersects.push_back(MaskIntersection());
                MaskIntersection& intersect = intersects.back();
                intersect.i = i;
                intersect.j = j;
                intersect.mask = mask;
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
            }
        }
    }

    return val;
}

struct ScalesAndError
{
    ScalesAndError() : error(0) {}
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

double calcTotalError(const std::vector<cv::Mat>& images, std::vector<MaskIntersection>& intersects)
{
    int numInts = intersects.size();
    double val = 0;
    for (int i = 0; i < numInts; i++)
    {
        MaskIntersection& currInt = intersects[i];
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

        bgrImages.resize(numImages);
        for (int j = 0; j < numImages; j++)
        {
            bgrImages[j].resize(3);
            cv::split(reprojImages[j], bgrImages[j]);
        }        
        bImages.resize(numImages);
        gImages.resize(numImages);
        rImages.resize(numImages);
        for (int j = 0; j < numImages; j++)
        {
            bImages[j] = bgrImages[j][0];
            gImages[j] = bgrImages[j][1];
            rImages[j] = bgrImages[j][2];
        }
        info.errorB = calcTotalError(bImages, intersects);
        info.errorG = calcTotalError(gImages, intersects);
        info.errorR = calcTotalError(rImages, intersects);

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

int main()
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

    std::vector<ScalesAndError> infos;
    enumErrors(origImages, maps, masks, 0.5, 1.5, 10, infos);
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

    std::vector<cv::Mat> scaleImages, scaledImages;
    getScales(infos[0].scales, origImages[0].size(), scaleImages);
    mulScales(origImages, scaleImages, scaledImages);
    reprojectParallel(scaledImages, images, maps);

    TilingLinearBlend blender;
    cv::Mat result;
    blender.prepare(masks, 100);
    blender.blend(images, result);
    cv::imshow("result", result);
    cv::waitKey(0);

    std::sort(infos.begin(), infos.end(),
        [](const ScalesAndError& lhs, const ScalesAndError& rhs){return lhs.errorB < rhs.errorB; });
    printf("result\n");
    for (int i = 0; i < 10; i++)
    {
        ScalesAndError& info = infos[i];
        for (int j = 0; j < info.scales.size(); j++)
            printf("%8.4f ", info.scales[j]);
        printf("Error: %f\n", info.errorB);
    }
    printf("\n");
    std::vector<double> scalesB = infos[0].scales;

    std::sort(infos.begin(), infos.end(),
        [](const ScalesAndError& lhs, const ScalesAndError& rhs){return lhs.errorG < rhs.errorG; });
    printf("result\n");
    for (int i = 0; i < 10; i++)
    {
        ScalesAndError& info = infos[i];
        for (int j = 0; j < info.scales.size(); j++)
            printf("%8.4f ", info.scales[j]);
        printf("Error: %f\n", info.errorG);
    }
    printf("\n");
    std::vector<double> scalesG = infos[0].scales;

    std::sort(infos.begin(), infos.end(),
        [](const ScalesAndError& lhs, const ScalesAndError& rhs){return lhs.errorR < rhs.errorR; });
    printf("result\n");
    for (int i = 0; i < 10; i++)
    {
        ScalesAndError& info = infos[i];
        for (int j = 0; j < info.scales.size(); j++)
            printf("%8.4f ", info.scales[j]);
        printf("Error: %f\n", info.errorR);
    }
    printf("\n");
    std::vector<double> scalesR = infos[0].scales;

    std::vector<std::vector<cv::Mat> > bgrImages(numImages);
    std::vector<cv::Mat> bImages(numImages), gImages(numImages), rImages(numImages);
    for (int i = 0; i < numImages; i++)
    {
        cv::split(origImages[i], bgrImages[i]);
        bImages[i] = bgrImages[i][0];
        gImages[i] = bgrImages[i][1];
        rImages[i] = bgrImages[i][2];
    }
    getScales(scalesB, origImages[0].size(), scaleImages);
    mulScales(bImages, scaleImages, bImages);
    mulScales(gImages, scaleImages, gImages);
    mulScales(rImages, scaleImages, rImages);
    for (int i = 0; i < numImages; i++)
        cv::merge(bgrImages[i], scaledImages[i]);
    reprojectParallel(scaledImages, images, maps);

    return 0;
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
            ptr[j] = (diffx * diffx + diffy + diffy) * scale;
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
                a1 += ptrDist1[j] * ptrDist1[j] * double(ptrImage1[j]) * double(ptrImage1[j]);
                a2 += ptrDist2[j] * ptrDist2[j] * double(ptrImage2[j]) * double(ptrImage2[j]);
                a12 += -2 * ptrDist1[j] * ptrDist2[j] * double(ptrImage1[j]) * double(ptrImage2[j]);
                b1 += 2 * (double(ptrImage1[j]) - double(ptrImage2[j])) * double(ptrImage1[j]) * ptrDist1[j];
                b2 += -2 * (double(ptrImage1[j]) - double(ptrImage2[j])) * double(ptrImage2[j]) * ptrDist2[j];
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

// y = x^T A x + B^T x
void getQuadSystem(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& dists,
    const std::vector<cv::Mat>& masks, cv::Mat& A, cv::Mat& B)
{
    int size = images.size();
    A.create(size, size, CV_64FC1);
    B.create(size, 1, CV_64FC1);
    A.setTo(0);
    B.setTo(0);

    std::vector<MaskIntersection> intersects;
    calcMaskIntersections(masks, intersects);
    int numInts = intersects.size();
    for (int i = 0; i < numInts; i++)
    {
        const MaskIntersection& mi = intersects[i];
        cv::imshow("i mask", mi.mask);
        cv::waitKey(0);
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
    getQuadSystem(images, dists, masks, A, B);
    B *= -0.5;
    std::cout << A << std::endl;
    std::cout << B << std::endl;
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

    for (int i = 0; i < numImages; i++)
    {
        show64FC1("dist", dists[i]);
        cv::imshow("gray", grayImages[i]);
        cv::waitKey(0);
    }

    std::vector<double> scales;
    solveScales(grayImages, dists, masks, scales);
    for (int i = 0; i < numImages; i++)
        printf("%f ", scales[i]);
    printf("\n");

    return 0;
}