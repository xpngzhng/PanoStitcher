#include "ZBlend.h"
#include "ZBlendAlgo.h"
#include "Warp/ZReproject.h"
#include "Warp/ConvertCoordinate.h"
#include "Tool/Timer.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#define PRINT_AND_SHOW 1

static int getResizeTimes(int width, int height, int minWidth, int minHeight)
{
    if (width < minWidth || height < minHeight)
        return 0;
    int num = 0;
    while (true)
    {
        width /= 2;
        height /= 2;
        num++;
        if (width < minWidth || height < minHeight)
            break;
    }
    return num - 1;
}

inline cv::Vec3b toVec3b(const cv::Vec3d v)
{
    return cv::Vec3b(cv::saturate_cast<unsigned char>(v[0]),
        cv::saturate_cast<unsigned char>(v[1]),
        cv::saturate_cast<unsigned char>(v[2]));
}

inline cv::Vec3d toVec3d(const cv::Vec3b v)
{
    return cv::Vec3d(v[0], v[1], v[2]);
}

static void calcGradImage(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(src.type() == CV_8UC3);
    cv::Mat gray, blurred, grad;
    cv::cvtColor(src, gray, CV_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 1.0);
    cv::Laplacian(blurred, grad, CV_16S);
    //cv::Laplacian(gray, grad, CV_16S);
    cv::convertScaleAbs(grad, dst);
}

static void rescalePhotoParam(PhotoParam& param, double scale)
{
    param.shiftX *= scale;
    param.shiftY *= scale;
    param.shearX *= scale;
    param.shearY *= scale;

    param.cropX *= scale;
    param.cropY *= scale;
    param.cropWidth *= scale;
    param.cropHeight *= scale;
    param.circleX *= scale;
    param.circleY *= scale;
    param.circleR *= scale;
}

static void rescalePhotoParams(std::vector<PhotoParam>& params, double scale)
{
    int size = params.size();
    for (int i = 0; i < size; i++)
        rescalePhotoParam(params[i], scale);
}

struct ValuePair
{
    int i, j;
    cv::Vec3b iVal, jVal;
    cv::Vec3d iValD, jValD;
    cv::Point iPos, jPos;
    cv::Point equiRectPos;
};

static void printAndShowPairsInfo(const std::vector<cv::Mat>& images, bool reprojected, 
    const std::vector<ValuePair>& pairs, int erWidth, int erHeight)
{
    int numImages = images.size();
    std::vector<int> appearCount(numImages, 0);

    int numPairs = pairs.size();
    cv::Mat mask = cv::Mat::zeros(erHeight, erWidth, CV_8UC1);
    for (int i = 0; i < numPairs; i++)
    {
        mask.at<unsigned char>(pairs[i].equiRectPos) = 255;
        for (int k = 0; k < numImages; k++)
        {
            if (pairs[i].i == k)
                appearCount[k]++;
            if (pairs[i].j == k)
                appearCount[k]++;
        }
    }

    printf("num pairs found %d\n", numPairs);
    for (int i = 0; i < numImages; i++)
    {
        printf("%d appear %d times\n", i, appearCount[i]);
    }
    cv::imshow("mask", mask);
    cv::waitKey(0);

    std::vector<cv::Mat> show(numImages);
    for (int i = 0; i < numImages; i++)
        show[i] = images[i].clone();

    if (reprojected)
    {
        for (int i = 0; i < numPairs; i++)
        {
            for (int k = 0; k < numImages; k++)
            {
                if (pairs[i].i == k)
                    cv::circle(show[k], pairs[i].equiRectPos, 2, cv::Scalar(255), -1);
                if (pairs[i].j == k)
                    cv::circle(show[k], pairs[i].equiRectPos, 2, cv::Scalar(255), -1);
            }
        }
    }
    else
    {
        for (int i = 0; i < numPairs; i++)
        {
            for (int k = 0; k < numImages; k++)
            {
                if (pairs[i].i == k)
                    cv::circle(show[k], pairs[i].iPos, 2, cv::Scalar(255), -1);
                if (pairs[i].j == k)
                    cv::circle(show[k], pairs[i].jPos, 2, cv::Scalar(255), -1);
            }
        }
    }

    for (int i = 0; i < numImages; i++)
    {
        char buf[64];
        sprintf(buf, "%d", i);
        cv::imshow(buf, show[i]);
    }
    cv::waitKey(0);
}

static void getPointPairsRandom(const std::vector<cv::Mat>& src, const std::vector<PhotoParam>& photoParams, 
    int downSizeRatio, std::vector<ValuePair>& pairs)
{
    int numImages = src.size();
    CV_Assert(photoParams.size() == numImages);

    int erWidth = 256, erHeight = 128;
    std::vector<Remap> remaps(numImages);
    for (int i = 0; i < numImages; i++)
        remaps[i].init(photoParams[i], erWidth, erHeight, src[0].cols * downSizeRatio, src[0].rows * downSizeRatio);

    std::vector<cv::Mat> grads(numImages);
    for (int i = 0; i < numImages; i++)
        calcGradImage(src[i], grads[i]);

#if PRINT_AND_SHOW
    for (int i = 0; i < numImages; i++)
    {
        double maxVal, minVal;
        cv::minMaxLoc(grads[i], &minVal, &maxVal);
        printf("min %f, max %f\n", minVal, maxVal);
    }
#endif

    cv::Rect validRect(0, 0, src[0].cols, src[0].rows);

    pairs.clear();

    int minValThresh = 5, maxValThresh = 250;
    int gradThresh = 3;
    cv::RNG_MT19937 rng(cv::getTickCount()/*0xffffffff*/);
    int numTrials = 10000 * 50;
    int expectNumPairs = 100 * 5;
    int numPairs = 0;
    const double downSizeScale = 1.0 / downSizeRatio;
    const double normScale = 1.0 / 255.0;
    const double halfWidth = erWidth * 0.5;
    const double halfHeight = erHeight * 0.5;
    for (int t = 0; t < numTrials; t++)
    {
        //int erx = rng.uniform(0, erWidth);
        //int ery = rng.uniform(0, erHeight);
        double theta = rng.uniform(-1.0, 1.0) * PI;
        double u = rng.uniform(-1.0, 1.0);
        double v = sqrt(1 - u * u);
        cv::Point2d pd = sphereToEquirect(cv::Point3d(cos(theta) * v, sin(theta) * v, u), halfWidth, halfHeight);
        int erx = pd.x + 0.5;
        int ery = pd.y + 0.5;
        for (int i = 0; i < numImages; i++)
        {
            int getPair = 0;
            double srcxid, srcyid;
            remaps[i].remapImage(srcxid, srcyid, erx, ery);
            cv::Point pti(srcxid * downSizeScale, srcyid * downSizeScale);
            if (validRect.contains(pti))
            {
                if (photoParams[i].circleR > 0)
                {
                    double diffx = srcxid - photoParams[i].circleX;
                    double diffy = srcyid - photoParams[i].circleY;
                    if (diffx * diffx + diffy * diffy > photoParams[i].circleR * photoParams[i].circleR - 25)
                        continue;
                }
                cv::Vec3b valI = src[i].at<cv::Vec3b>(pti);
                int gradValI = grads[i].at<unsigned char>(pti);
                if (valI[0] > minValThresh && valI[0] < maxValThresh &&
                    valI[1] > minValThresh && valI[1] < maxValThresh &&
                    valI[2] > minValThresh && valI[2] < maxValThresh &&
                    gradValI < gradThresh)
                {
                    for (int j = 0; j < numImages; j++)
                    {
                        if (i == j)
                            continue;

                        double srcxjd, srcyjd;
                        remaps[j].remapImage(srcxjd, srcyjd, erx, ery);
                        cv::Point ptj(srcxjd * downSizeScale, srcyjd * downSizeScale);
                        if (validRect.contains(ptj))
                        {
                            if (photoParams[j].circleR > 0)
                            {
                                double diffx = srcxjd - photoParams[j].circleX;
                                double diffy = srcyjd - photoParams[j].circleY;
                                if (diffx * diffx + diffy * diffy > photoParams[j].circleR * photoParams[j].circleR - 25)
                                    continue;
                            }
                            cv::Vec3b valJ = src[j].at<cv::Vec3b>(ptj);
                            int gradValJ = grads[j].at<unsigned char>(ptj);
                            if (valJ[0] > minValThresh && valJ[0] < maxValThresh &&
                                valJ[1] > minValThresh && valJ[1] < maxValThresh &&
                                valJ[2] > minValThresh && valJ[2] < maxValThresh &&
                                gradValJ < gradThresh)
                            {
                                ValuePair pair;
                                pair.i = i;
                                pair.j = j;
                                pair.iPos = pti;
                                pair.jPos = ptj;
                                pair.iVal = valI;
                                pair.jVal = valJ;
                                pair.iValD = toVec3d(pair.iVal) * normScale;
                                pair.jValD = toVec3d(pair.jVal) * normScale;
                                pair.equiRectPos = cv::Point(erx, ery);
                                getPair = 1;
                                numPairs++;
                                pairs.push_back(pair);
                                break;
                            }
                        }
                    }
                }
            }
            if (getPair)
                break;
        }
        if (numPairs >= expectNumPairs)
            break;
    }

#if PRINT_AND_SHOW
    printAndShowPairsInfo(src, false, pairs, erWidth, erHeight);
#endif
}

static void getPointPairsAll(const std::vector<cv::Mat>& src, const std::vector<PhotoParam>& photoParams, 
    int downSizeRatio, std::vector<ValuePair>& pairs)
{
    int numImages = src.size();
    CV_Assert(photoParams.size() == numImages);

    int erWidth = 200, erHeight = 100;
    std::vector<Remap> remaps(numImages);
    for (int i = 0; i < numImages; i++)
        remaps[i].init(photoParams[i], erWidth, erHeight, src[0].cols * downSizeRatio, src[0].rows * downSizeRatio);

    std::vector<cv::Mat> grads(numImages);
    for (int i = 0; i < numImages; i++)
        calcGradImage(src[i], grads[i]);

#if PRINT_AND_SHOW
    for (int i = 0; i < numImages; i++)
    {
        double maxVal, minVal;
        cv::minMaxLoc(grads[i], &minVal, &maxVal);
        printf("min %f, max %f\n", minVal, maxVal);
    }
#endif

    cv::Rect validRect(0, 0, src[0].cols, src[0].rows);

    pairs.clear();

    int minValThresh = 5, maxValThresh = 250;
    int gradThresh = 3;
    cv::RNG_MT19937 rng(cv::getTickCount());
    int numTrials = 8000 * 50;
    int expectNumPairs = 1000 * 5;
    int numPairs = 0;
    const double downSizeScale = 1.0 / downSizeRatio;
    const double normScale = 1.0 / 255.0;
    const double halfWidth = erWidth * 0.5;
    const double halfHeight = erHeight * 0.5;
    const int gridSize = 200;
    //for (int ery = 0; ery < erHeight; ery++)
    //for (int erx = 0; erx < erWidth; erx++)
    for (int y = 0; y < gridSize; y++)
    for (int x = 0; x < gridSize; x++)
    {
        double theta = 2 * PI * ((x + 0.5) / gridSize - 0.5);
        double u = ((y + 0.5) / gridSize - 0.5) * 2;
        double v = sqrt(1 - u * u);
        cv::Point2d pd = sphereToEquirect(cv::Point3d(cos(theta) * v, sin(theta) * v, u), halfWidth, halfHeight);
        int erx = pd.x + 0.5;
        int ery = pd.y + 0.5;
        for (int i = 0; i < numImages; i++)
        {
            int getPair = 0;
            double srcxid, srcyid;
            remaps[i].remapImage(srcxid, srcyid, erx, ery);
            cv::Point pti(srcxid * downSizeScale, srcyid * downSizeScale);
            if (validRect.contains(pti))
            {
                if (photoParams[i].circleR > 0)
                {
                    double diffx = srcxid - photoParams[i].circleX;
                    double diffy = srcyid - photoParams[i].circleY;
                    if (diffx * diffx + diffy * diffy > photoParams[i].circleR * photoParams[i].circleR - 25)
                        continue;
                }
                cv::Vec3b valI = src[i].at<cv::Vec3b>(pti);
                int gradValI = grads[i].at<unsigned char>(pti);
                if (valI[0] > minValThresh && valI[0] < maxValThresh &&
                    valI[1] > minValThresh && valI[1] < maxValThresh &&
                    valI[2] > minValThresh && valI[2] < maxValThresh &&
                    gradValI < gradThresh)
                {
                    for (int j = 0; j < numImages; j++)
                    {
                        if (i == j)
                            continue;

                        double srcxjd, srcyjd;
                        remaps[j].remapImage(srcxjd, srcyjd, erx, ery);
                        cv::Point ptj(srcxjd * downSizeScale, srcyjd * downSizeScale);
                        if (validRect.contains(ptj))
                        {
                            if (photoParams[j].circleR > 0)
                            {
                                double diffx = srcxjd - photoParams[j].circleX;
                                double diffy = srcyjd - photoParams[j].circleY;
                                if (diffx * diffx + diffy * diffy > photoParams[j].circleR * photoParams[j].circleR - 25)
                                    continue;
                            }
                            if (pti.x < 20 || ptj.x < 20)
                            {
                                int a = 0;
                            }
                            cv::Vec3b valJ = src[j].at<cv::Vec3b>(ptj);
                            int gradValJ = grads[j].at<unsigned char>(ptj);
                            if (valJ[0] > minValThresh && valJ[0] < maxValThresh &&
                                valJ[1] > minValThresh && valJ[1] < maxValThresh &&
                                valJ[2] > minValThresh && valJ[2] < maxValThresh &&
                                gradValJ < gradThresh)
                            {
                                ValuePair pair;
                                pair.i = i;
                                pair.j = j;
                                pair.iPos = pti;
                                pair.jPos = ptj;
                                pair.iVal = valI;
                                pair.jVal = valJ;
                                pair.iValD = toVec3d(pair.iVal) * normScale;
                                pair.jValD = toVec3d(pair.jVal) * normScale;
                                pair.equiRectPos = cv::Point(erx, ery);
                                getPair = 1;
                                numPairs++;
                                pairs.push_back(pair);
                                //break;
                            }
                        }
                    }
                }
            }
            //if (getPair)
            //    break;
        }
        //if (numPairs >= expectNumPairs)
        //    break;
    }

#if PRINT_AND_SHOW
    printAndShowPairsInfo(src, false, pairs, erWidth, erHeight);
#endif
}

static void getPointPairsAll2(const std::vector<cv::Mat>& src, const std::vector<PhotoParam>& photoParams, 
    int downSizeRatio, std::vector<ValuePair>& pairs)
{
    int numImages = src.size();
    CV_Assert(photoParams.size() == numImages);

    std::vector<PhotoParam> params = photoParams;
    rescalePhotoParams(params, 1.0 / downSizeRatio);

    int erWidth = 200, erHeight = 100;
    std::vector<Remap> remaps(numImages);
    for (int i = 0; i < numImages; i++)
        remaps[i].init(params[i], erWidth, erHeight, src[0].cols, src[0].rows);

    std::vector<cv::Mat> grads(numImages);
    for (int i = 0; i < numImages; i++)
        calcGradImage(src[i], grads[i]);

#if PRINT_AND_SHOW
    for (int i = 0; i < numImages; i++)
    {
        double maxVal, minVal;
        cv::minMaxLoc(grads[i], &minVal, &maxVal);
        printf("min %f, max %f\n", minVal, maxVal);
    }
#endif

    cv::Rect validRect(0, 0, src[0].cols, src[0].rows);

    pairs.clear();

    int minValThresh = 5, maxValThresh = 250;
    int gradThresh = 3;
    cv::RNG_MT19937 rng(cv::getTickCount());
    int numTrials = 8000 * 5;
    int expectNumPairs = 1000 * 5;
    int numPairs = 0;
    const double normScale = 1.0 / 255.0;
    const double halfWidth = erWidth * 0.5;
    const double halfHeight = erHeight * 0.5;
    const int gridSize = 200;
    //for (int ery = 0; ery < erHeight; ery++)
    //for (int erx = 0; erx < erWidth; erx++)
    for (int y = 0; y < gridSize; y++)
    for (int x = 0; x < gridSize; x++)
    {
        double theta = 2 * PI * ((x + 0.5) / gridSize - 0.5);
        double u = ((y + 0.5) / gridSize - 0.5) * 2;
        double v = sqrt(1 - u * u);
        cv::Point2d pd = sphereToEquirect(cv::Point3d(cos(theta) * v, sin(theta) * v, u), halfWidth, halfHeight);
        int erx = pd.x + 0.5;
        int ery = pd.y + 0.5;
        for (int i = 0; i < numImages; i++)
        {
            int getPair = 0;
            double srcxid, srcyid;
            remaps[i].remapImage(srcxid, srcyid, erx, ery);
            cv::Point pti(srcxid, srcyid);
            if (validRect.contains(pti))
            {
                if (params[i].circleR > 0)
                {
                    double diffx = srcxid - params[i].circleX;
                    double diffy = srcyid - params[i].circleY;
                    if (diffx * diffx + diffy * diffy > params[i].circleR * params[i].circleR - 5)
                        continue;
                }
                cv::Vec3b valI = src[i].at<cv::Vec3b>(pti);
                int gradValI = grads[i].at<unsigned char>(pti);
                if (valI[0] > minValThresh && valI[0] < maxValThresh &&
                    valI[1] > minValThresh && valI[1] < maxValThresh &&
                    valI[2] > minValThresh && valI[2] < maxValThresh &&
                    gradValI < gradThresh)
                {
                    for (int j = 0; j < numImages; j++)
                    {
                        if (i == j)
                            continue;

                        double srcxjd, srcyjd;
                        remaps[j].remapImage(srcxjd, srcyjd, erx, ery);
                        cv::Point ptj(srcxjd, srcyjd);
                        if (validRect.contains(ptj))
                        {
                            if (params[j].circleR > 0)
                            {
                                double diffx = srcxjd - params[j].circleX;
                                double diffy = srcyjd - params[j].circleY;
                                if (diffx * diffx + diffy * diffy > params[j].circleR * params[j].circleR - 5)
                                    continue;
                            }
                            if (pti.x < 20 || ptj.x < 20)
                            {
                                int a = 0;
                            }
                            cv::Vec3b valJ = src[j].at<cv::Vec3b>(ptj);
                            int gradValJ = grads[j].at<unsigned char>(ptj);
                            if (valJ[0] > minValThresh && valJ[0] < maxValThresh &&
                                valJ[1] > minValThresh && valJ[1] < maxValThresh &&
                                valJ[2] > minValThresh && valJ[2] < maxValThresh &&
                                gradValJ < gradThresh)
                            {
                                ValuePair pair;
                                pair.i = i;
                                pair.j = j;
                                pair.iPos = pti;
                                pair.jPos = ptj;
                                pair.iVal = valI;
                                pair.jVal = valJ;
                                pair.iValD = toVec3d(pair.iVal) * normScale;
                                pair.jValD = toVec3d(pair.jVal) * normScale;
                                pair.equiRectPos = cv::Point(erx, ery);
                                getPair = 1;
                                numPairs++;
                                pairs.push_back(pair);
                                //break;
                            }
                        }
                    }
                }
            }
            //if (getPair)
            //    break;
        }
        //if (numPairs >= expectNumPairs)
        //    break;
    }

#if PRINT_AND_SHOW
    printAndShowPairsInfo(src, false, pairs, erWidth, erHeight);
#endif
}

static void getPointPairsAllReproject(const std::vector<cv::Mat>& src, const std::vector<PhotoParam>& photoParams,
    int downSizeRatio, std::vector<ValuePair>& pairs)
{
    int numImages = src.size();
    int erWidth = 200, erHeight = 100;

    std::vector<PhotoParam> params = photoParams;
    rescalePhotoParams(params, 1.0 / downSizeRatio);

    std::vector<cv::Mat> reprojImages, masks;
    reproject(src, reprojImages, masks, params, cv::Size(erWidth, erHeight));

    std::vector<cv::Mat> grads(numImages);
    for (int i = 0; i < numImages; i++)
        calcGradImage(reprojImages[i], grads[i]);

    cv::Mat intersect;
    int minValThresh = 5, maxValThresh = 250;
    int gradThresh = 3;
    double normScale = 1.0 / 255.0;

    int numPairs = 0;

    pairs.reserve(numImages * (numImages - 1) * 2 * 256);
    for (int i = 0; i < numImages - 1; i++)
    {
        for (int j = i + 1; j < numImages; j++)
        {
            cv::bitwise_and(masks[i], masks[j], intersect);
            //cv::bitwise_and(intersect, gradSmalls[i], intersect);
            //cv::bitwise_and(intersect, gradSmalls[j], intersect);
            if (cv::countNonZero(intersect) <= 0)
                continue;

            for (int y = 0; y < erHeight; y++)
            {
                const unsigned char* ptrI = reprojImages[i].ptr<unsigned char>(y);
                const unsigned char* ptrJ = reprojImages[j].ptr<unsigned char>(y);
                const unsigned char* ptrGradI = grads[i].ptr<unsigned char>(y);
                const unsigned char* ptrGradJ = grads[j].ptr<unsigned char>(y);
                for (int x = 0; x < erWidth; x++)
                {
                    if (ptrI[0] > minValThresh && ptrI[0] < maxValThresh &&
                        ptrI[1] > minValThresh && ptrI[1] < maxValThresh &&
                        ptrI[2] > minValThresh && ptrI[2] < maxValThresh &&
                        ptrJ[0] > minValThresh && ptrJ[0] < maxValThresh &&
                        ptrJ[1] > minValThresh && ptrJ[1] < maxValThresh &&
                        ptrJ[2] > minValThresh && ptrJ[2] < maxValThresh &&
                        ptrGradI[0] < gradThresh && ptrGradJ[0] < gradThresh)
                    {
                        ValuePair pair;
                        pair.i = i;
                        pair.j = j;
                        pair.iVal = cv::Vec3b(ptrI[0], ptrI[1], ptrI[2]);
                        pair.jVal = cv::Vec3b(ptrJ[0], ptrJ[1], ptrJ[2]);
                        pair.iValD = toVec3d(pair.iVal) * normScale;
                        pair.jValD = toVec3d(pair.jVal) * normScale;
                        pair.equiRectPos = cv::Point(x, y);
                        numPairs++;
                        pairs.push_back(pair);
                    }

                    ptrI += 3;
                    ptrJ += 3;
                    ptrGradI++;
                    ptrGradJ++;
                }
            }
        }
    }

#if PRINT_AND_SHOW
    printAndShowPairsInfo(src, true, pairs, erWidth, erHeight);
#endif
}

// use this downSizeRatio had better keep small
static void getPointPairsHistogram(const std::vector<cv::Mat>& src, const std::vector<PhotoParam>& photoParams,
    int downSizeRatio, std::vector<ValuePair>& pairs)
{
    int numImages = src.size();
    int erWidth = 1600, erHeight = 800;

    std::vector<PhotoParam> params = photoParams;
    rescalePhotoParams(params, 1.0 / downSizeRatio);

    std::vector<cv::Mat> reprojImages, masks;
    reproject(src, reprojImages, masks, params, cv::Size(erWidth, erHeight));

    int gradThresh = 5;
    std::vector<cv::Mat> grads(numImages), gradSmalls(numImages);
    for (int i = 0; i < numImages; i++)
    {
        calcGradImage(reprojImages[i], grads[i]);
        gradSmalls[i] = grads[i] <= gradThresh;
    }

    std::vector<unsigned char> lutI[3], lutJ[3];
    std::vector<double> accumHistI[3], accumHistJ[3];
    std::vector<unsigned char> transIToJ[3], transJToI[3];
    cv::Mat intersect;
    cv::Mat bgrI[3], bgrJ[3];

    int minVal = 5, maxVal = 250;
    double normScale = 1.0 / 255.0;

    pairs.reserve(numImages * (numImages - 1) * 2 * 256);
    for (int i = 0; i < numImages - 1; i++)
    {
        for (int j = i + 1; j < numImages; j++)
        {
            cv::bitwise_and(masks[i], masks[j], intersect);
            //cv::bitwise_and(intersect, gradSmalls[i], intersect);
            //cv::bitwise_and(intersect, gradSmalls[j], intersect);
            if (cv::countNonZero(intersect) <= 0)
                continue;

            cv::split(reprojImages[i], bgrI);
            cv::split(reprojImages[j], bgrJ);

            for (int k = 0; k < 3; k++)
            {
                calcAccumHist(bgrI[k], intersect, accumHistI[k]);
                calcAccumHist(bgrJ[k], intersect, accumHistJ[k]);
                histSpecification(accumHistI[k], accumHistJ[k], transIToJ[k]);
                histSpecification(accumHistJ[k], accumHistI[k], transJToI[k]);
            }

            for (int k = minVal; k < maxVal; k++)
            {
                if (transIToJ[0][k] > minVal && transIToJ[0][k] < maxVal &&
                    transIToJ[1][k] > minVal && transIToJ[1][k] < maxVal &&
                    transIToJ[2][k] > minVal && transIToJ[2][k] < maxVal)
                {
                    ValuePair pair;
                    pair.i = i;
                    pair.j = j;
                    pair.iVal = cv::Vec3b(k, k, k);
                    pair.iValD = toVec3d(pair.iVal) * normScale;
                    pair.jVal = cv::Vec3b(transIToJ[0][k], transIToJ[1][k], transIToJ[2][k]);
                    pair.jValD = toVec3d(pair.jVal) * normScale;
                    pairs.push_back(pair);
                }

                if (transJToI[0][k] > minVal && transJToI[0][k] < maxVal &&
                    transJToI[1][k] > minVal && transJToI[1][k] < maxVal &&
                    transJToI[2][k] > minVal && transJToI[2][k] < maxVal)
                {
                    ValuePair pair;
                    pair.i = j;
                    pair.j = i;
                    pair.iVal = cv::Vec3b(k, k, k);
                    pair.iValD = toVec3d(pair.iVal) * normScale;
                    pair.jVal = cv::Vec3b(transJToI[0][k], transJToI[1][k], transJToI[2][k]);
                    pair.jValD = toVec3d(pair.jVal) * normScale;
                    pairs.push_back(pair);
                }
            }
        }
    }
}

#include "VisualManip.h"

struct ImageInfo
{
    ImageInfo()
    {

    }

    ImageInfo(const cv::Size& size_)
    {
        exposure = 1;
        whiteBalanceRed = 1;
        whiteBalanceBlue = 1;
        size = size_;
    }

    int static getNumParams(int optimizeWhat)
    {
        int num = 0;
        if (optimizeWhat & EXPOSURE)
            num += 1;
        if (optimizeWhat & WHITE_BALANCE)
            num += 2;
        return num;
    }

    void fromOutside(const double* x, int optimizeWhat)
    {
        int index = 0;
        if (optimizeWhat & EXPOSURE)
        {
            exposure = x[index++];
        }
        if (optimizeWhat & WHITE_BALANCE)
        {
            whiteBalanceRed = x[index++];
            whiteBalanceBlue = x[index];
        }
    }

    void toOutside(double* x, int optimizeWhat) const
    {
        int index = 0;
        if (optimizeWhat & EXPOSURE)
        {
            x[index++] = exposure;
        }
        if (optimizeWhat & WHITE_BALANCE)
        {
            x[index++] = whiteBalanceRed;
            x[index] = whiteBalanceBlue;
        }
    }

    double exposure;
    double whiteBalanceRed;
    double whiteBalanceBlue;
    cv::Size size;
    cv::Vec3d meanVals;
};

inline bool contains(const std::vector<int>& arr, int test)
{
    int size = arr.size();
    for (int i = 0; i < size; i++)
    {
        if (arr[i] == test)
            return true;
    }
    return false;
}

static void readFrom(std::vector<ImageInfo>& infos, const double* x, const std::vector<int> anchorIndexes, int optimizeWhat)
{
    int numInfos = infos.size();
    int offset = 0;
    for (int i = 0; i < numInfos; i++)
    {
        if (!contains(anchorIndexes, i))
        {
            infos[i].fromOutside(x + offset, optimizeWhat);
            offset += infos[i].getNumParams(optimizeWhat);
        }
    }
}

static void writeTo(const std::vector<ImageInfo>& infos, double* x, const std::vector<int>& anchorIndexes, int optimizeWhat)
{
    int numInfos = infos.size();
    int offset = 0;
    for (int i = 0; i < numInfos; i++)
    {
        if (!contains(anchorIndexes, i))
        {
            infos[i].toOutside(x + offset, optimizeWhat);
            offset += infos[i].getNumParams(optimizeWhat);
        }
    }
}

struct Transform
{
    Transform()
    {

    }

    Transform(const ImageInfo& imageInfo)
    {
        exposure = imageInfo.exposure;
        whiteBalanceRed = imageInfo.whiteBalanceRed;
        whiteBalanceBlue = imageInfo.whiteBalanceBlue;
    }

    cv::Vec3d apply(const cv::Vec3d& val) const
    {
        double scale = exposure;
        double b = val[0] * scale * whiteBalanceBlue;
        double g = val[1] * scale;
        double r = val[2] * scale * whiteBalanceRed;
        return cv::Vec3d(LUT(b), LUT(g), LUT(r));
    }

    cv::Vec3d applyInverse(const cv::Vec3d& val) const
    {
        double scale = 1.0 / exposure;
        double b = invLUT(val[0]) * scale;
        double g = invLUT(val[1]) * scale;
        double r = invLUT(val[2]) * scale;
        r /= whiteBalanceRed;
        b /= whiteBalanceBlue;
        return cv::Vec3d(b, g, r);
    }

    cv::Vec3d apply(const cv::Point& p, const cv::Vec3d& val) const
    {
        double scale = exposure;
        double b = val[0] * scale * whiteBalanceBlue;
        double g = val[1] * scale;
        double r = val[2] * scale * whiteBalanceRed;
        return cv::Vec3d(LUT(b), LUT(g), LUT(r));
    }

    cv::Vec3d applyInverse(const cv::Point& p, const cv::Vec3d& val) const
    {
        double scale = 1.0 / exposure;
        double b = invLUT(val[0]) * scale;
        double g = invLUT(val[1]) * scale;
        double r = invLUT(val[2]) * scale;
        r /= whiteBalanceRed;
        b /= whiteBalanceBlue;
        return cv::Vec3d(b, g, r);
    }

    cv::Vec3d applyInverseExposureOnly(const cv::Point& p, const cv::Vec3d& val) const
    {
        double scale = 1.0 / exposure;
        double b = invLUT(val[0]) * scale;
        double g = invLUT(val[1]) * scale;
        double r = invLUT(val[2]) * scale;
        return cv::Vec3d(b, g, r);
    }

    cv::Vec3d correctExposureOnly(const cv::Point& p, const cv::Vec3d& val) const
    {
        double scale = 1.0 / exposure;
        double b = invLUT(val[0]) * scale;
        double g = invLUT(val[1]) * scale;
        double r = invLUT(val[2]) * scale;
        return cv::Vec3d(LUT(b), LUT(g), LUT(r));
    }

    double LUT(double val) const
    {
        return val;
    }

    double invLUT(double val) const
    {
        return val;
    }

    double exposure;
    double whiteBalanceRed;
    double whiteBalanceBlue;
};

struct ExternData
{
    ExternData(std::vector<ImageInfo>& infos_, const std::vector<ValuePair>& pairs_)
    : imageInfos(infos_), pairs(pairs_)
    {}
    std::vector<ImageInfo>& imageInfos;
    const std::vector<ValuePair>& pairs;
    double huberSigma;
    int errorFuncCallCount;
    int optimizeWhat;
    std::vector<int> anchoIndexes;
    std::vector<cv::Vec3d> meanVals;
};

inline double weightHuber(double x, double sigma)
{
    if (x > sigma)
    {
        return sqrt(sigma* (2 * x - sigma));
    }
    return x;
}

static void errorFunc(double* p, double* hx, int m, int n, void* data)
{
    ExternData* edata = (ExternData*)data;
    const std::vector<ImageInfo>& infos = edata->imageInfos;
    const std::vector<ValuePair>& pairs = edata->pairs;
    const std::vector<int>& anchorIndexes = edata->anchoIndexes;

    std::vector<double> pv(m);
    memcpy(pv.data(), p, m * 8);

    readFrom(edata->imageInfos, p, anchorIndexes, edata->optimizeWhat);
    int numImages = infos.size();

    std::vector<Transform> transforms(numImages);
    for (int i = 0; i < numImages; i++)
        transforms[i] = Transform(infos[i]);

    int index = 0;
    for (int i = 0; i < numImages; i++)
    {
        if (contains(anchorIndexes, i))
            continue;

        hx[index++] = abs(transforms[i].exposure - 1) * 2;
        hx[index++] = abs(transforms[i].whiteBalanceBlue - 1) * 2;
        hx[index++] = abs(transforms[i].whiteBalanceRed - 1) * 2;
    }

    double huberSigma = edata->huberSigma;

    double sqrErr = 0;
    int numPairs = pairs.size();
    for (int i = 0; i < numPairs; i++)
    {
        const ValuePair& pair = pairs[i];

        cv::Vec3d lightI = transforms[pair.i].applyInverse(pair.iPos, pair.iValD);
        cv::Vec3d valIInJ = transforms[pair.j].apply(pair.jPos, lightI);
        cv::Vec3d errI = pair.jValD - valIInJ;

        cv::Vec3d lightJ = transforms[pair.j].applyInverse(pair.jPos, pair.jValD);
        cv::Vec3d valJInI = transforms[pair.i].apply(pair.iPos, lightJ);
        cv::Vec3d errJ = pair.iValD - valJInI;

        for (int j = 0; j < 3; j++)
        {
            hx[index++] = weightHuber(abs(errI[j]), huberSigma);
            hx[index++] = weightHuber(abs(errJ[j]), huberSigma);
            //hx[index++] = errI[j] * errI[j];
            //hx[index++] = errJ[j] * errJ[j];
            //hx[index++] = errI[j];
            //hx[index++] = errJ[j];
        }

        sqrErr += errI.dot(errI);
        sqrErr += errJ.dot(errJ);
    }

    cv::Vec3d diff;
    for (int i = 0; i < numImages; i++)
    {
        diff += transforms[i].applyInverse(edata->meanVals[i]) - edata->meanVals[i];
    }
    hx[index++] = weightHuber(abs(diff[0] + diff[1] + diff[2]) / 3.0, huberSigma);

    edata->errorFuncCallCount++;

#if PRINT_AND_SHOW
    printf("call count %d, sqr err = %f, avg err %f\n", edata->errorFuncCallCount, sqrErr, sqrt(sqrErr / n));
#endif
}

#include "levmar.h"

static void optimize(const std::vector<ValuePair>& valuePairs, int numImages, std::vector<int> anchorIndexes,
    const cv::Size& imageSize, const std::vector<int>& optimizeOptions,
    std::vector<ImageInfo>& outImageInfos)
{
    std::vector<ImageInfo> imageInfos(numImages);
    for (int i = 0; i < numImages; i++)
    {
        ImageInfo info(imageSize);
        imageInfos[i] = info;
        imageInfos[i].meanVals = outImageInfos[i].meanVals;
    }
    int numAnchors = anchorIndexes.size();

    int ret;
    //double opts[LM_OPTS_SZ];
    double info[LM_INFO_SZ];

    // TODO: setup optimisation options with some good defaults.
    double optimOpts[5];

    optimOpts[0] = 1E-03;  // init mu
    // stop thresholds
    optimOpts[1] = 1e-5;   // ||J^T e||_inf
    optimOpts[2] = 1e-5;   // ||Dp||_2
    optimOpts[3] = 1e-1;   // ||e||_2
    // difference mode
    optimOpts[4] = LM_DIFF_DELTA;

    int maxIter = 500;

    for (int i = 0; i < optimizeOptions.size(); i++)
    {
        int option = optimizeOptions[i];
        int numParams = ImageInfo::getNumParams(option);

        // parameters
        int m = (numImages - numAnchors) * numParams;
        std::vector<double> p(m, 0.0);

        // vector for errors
        int n = 2 * 3 * valuePairs.size() + 3 * (numImages - numAnchors) + 1;
        std::vector<double> x(n, 0.0);

        writeTo(imageInfos, p.data(), anchorIndexes, option);

        // covariance matrix at solution
        cv::Mat cov(m, m, CV_64FC1);

        ExternData edata(imageInfos, valuePairs);
        edata.huberSigma = 5.0 / 255;
        edata.errorFuncCallCount = 0;
        edata.optimizeWhat = option;
        edata.anchoIndexes = anchorIndexes;
        edata.meanVals.resize(numImages);
        for (int i = 0; i < numImages; i++)
            edata.meanVals[i] = imageInfos[i].meanVals;

        ret = dlevmar_dif(&errorFunc, &(p[0]), &(x[0]), m, n, maxIter, optimOpts, info, NULL, (double*)cov.data, &edata);  // no jacobian
        // copy to source images (data.m_imgs)
        readFrom(imageInfos, p.data(), anchorIndexes, option);
    }

#if PRINT_AND_SHOW
    for (int i = 0; i < numImages; i++)
    {
        printf("[%d] e = %f, blue = %f, red = %f\n",
            i, imageInfos[i].exposure, imageInfos[i].whiteBalanceBlue, imageInfos[i].whiteBalanceRed);
    }
#endif

    outImageInfos = imageInfos;
}

void getLUT(std::vector<unsigned char> lut, double k)
{
    CV_Assert(k > 0);
    lut.resize(256);
    for (int i = 0; i < 256; i++)
        lut[i] = cv::saturate_cast<unsigned char>(k * i);
}

void exposureColorOptimize(const std::vector<cv::Mat>& images, const std::vector<PhotoParam>& params,
    const std::vector<int> anchorIndexes, const std::vector<int>& optimizeOptions,
    std::vector<double>& exposures, std::vector<double>& redRatios, std::vector<double>& blueRatios)
{
    int numImages = images.size();
    CV_Assert(numImages == params.size());
    CV_Assert(checkSize(images));
    CV_Assert(checkType(images, CV_8UC3));

    int minWidth = 100, minHeight = 100;
    int resizeTimes = getResizeTimes(images[0].cols, images[0].rows, minWidth, minHeight);

    std::vector<cv::Mat> testSrc(numImages);
    if (resizeTimes == 0)
    {
        testSrc = images;
    }
    else
    {
        for (int i = 0; i < numImages; i++)
        {
            cv::Mat large = images[i];
            cv::Mat small;
            for (int j = 0; j < resizeTimes; j++)
            {
                cv::resize(large, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
                large = small;
            }
            testSrc[i] = small;
        }
    }

    int downSizePower = pow(2, resizeTimes);
    std::vector<ValuePair> pairs;
    //getPointPairsRandom(testSrc, params, downSizePower, pairs);
    getPointPairsAll(testSrc, params, downSizePower, pairs);
    //getPointPairsAll2(testSrc, params, downSizePower, pairs);
    //getPointPairsAllReproject(testSrc, params, downSizePower, pairs);
    //getPointPairsHistogram(testSrc, params, downSizePower, pairs);

    std::vector<ImageInfo> imageInfos(numImages);
    for (int i = 0; i < numImages; i++)
    {
        cv::Scalar meanVals = cv::mean(testSrc[i]) / 255.0;
        imageInfos[i].meanVals = cv::Vec3d(meanVals[0], meanVals[1], meanVals[2]);
    }
    optimize(pairs, numImages, anchorIndexes, testSrc[0].size(), optimizeOptions, imageInfos);

    exposures.resize(numImages);
    redRatios.resize(numImages);
    blueRatios.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        exposures[i] = 1.0 / imageInfos[i].exposure;
        redRatios[i] = 1.0 / imageInfos[i].whiteBalanceRed;
        blueRatios[i] = 1.0 / imageInfos[i].whiteBalanceBlue;
    }
}

void getExposureColorOptimizeLUTs(const std::vector<double>& exposures, const std::vector<double>& redRatios,
    const std::vector<double>& blueRatios, std::vector<std::vector<std::vector<unsigned char> > >& luts)
{
    int size = exposures.size();
    CV_Assert(size > 0 && size == redRatios.size() && size == blueRatios.size());

    luts.resize(size);
    for (int i = 0; i < size; i++)
    {
        luts[i].resize(3);
        getLUT(luts[i][0], exposures[i] * blueRatios[i]);
        getLUT(luts[i][1], exposures[i]);
        getLUT(luts[i][2], exposures[i] * redRatios[i]);
    }
}

