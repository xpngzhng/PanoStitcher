#include "ZBlend.h"
#include "ZBlendAlgo.h"
#include "Warp/ZReproject.h"
#include "Warp/ConvertCoordinate.h"
#include "Tool/Timer.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

int getResizeTimes(int width, int height, int minWidth, int minHeight)
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

void calcGradImage(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(src.type() == CV_8UC3);
    cv::Mat gray, blurred, grad;
    cv::cvtColor(src, gray, CV_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 1.0);
    cv::Laplacian(blurred, grad, CV_16S);
    //cv::Laplacian(src, grad, CV_16S);
    cv::convertScaleAbs(grad, dst);
}

void rescalePhotoParam(PhotoParam& param, double scale)
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

void rescalePhotoParams(std::vector<PhotoParam>& params, double scale)
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

void getPointPairsRandom(const std::vector<cv::Mat>& src, const std::vector<PhotoParam>& photoParams, int downSizeRatio, std::vector<ValuePair>& pairs)
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

    for (int i = 0; i < numImages; i++)
    {
        double maxVal, minVal;
        cv::minMaxLoc(grads[i], &minVal, &maxVal);
        printf("min %f, max %f\n", minVal, maxVal);
    }

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

    std::vector<int> appearCount(numImages, 0);

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
        show[i] = src[i].clone();

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

    for (int i = 0; i < numImages; i++)
    {
        char buf[64];
        sprintf(buf, "%d", i);
        cv::imshow(buf, show[i]);
    }
    cv::waitKey(0);
}

void getPointPairsAll(const std::vector<cv::Mat>& src, const std::vector<PhotoParam>& photoParams, int downSizeRatio, std::vector<ValuePair>& pairs)
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

    for (int i = 0; i < numImages; i++)
    {
        double maxVal, minVal;
        cv::minMaxLoc(grads[i], &minVal, &maxVal);
        printf("min %f, max %f\n", minVal, maxVal);
    }

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

    std::vector<int> appearCount(numImages, 0);

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
    cv::imwrite("mask.bmp", mask);
    cv::imshow("mask", mask);
    cv::waitKey(0);

    std::vector<cv::Mat> show(numImages);
    for (int i = 0; i < numImages; i++)
        show[i] = src[i].clone();

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

    for (int i = 0; i < numImages; i++)
    {
        char buf[64];
        sprintf(buf, "%d", i);
        cv::imshow(buf, show[i]);
    }
    cv::waitKey(0);
}

void getPointPairsAll2(const std::vector<cv::Mat>& src, const std::vector<PhotoParam>& photoParams, int downSizeRatio, std::vector<ValuePair>& pairs)
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

    for (int i = 0; i < numImages; i++)
    {
        double maxVal, minVal;
        cv::minMaxLoc(grads[i], &minVal, &maxVal);
        printf("min %f, max %f\n", minVal, maxVal);
    }

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

    std::vector<int> appearCount(numImages, 0);

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
        show[i] = src[i].clone();

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

    for (int i = 0; i < numImages; i++)
    {
        char buf[64];
        sprintf(buf, "%d", i);
        cv::imshow(buf, show[i]);
    }
    cv::waitKey(0);
}

void getPointPairsAllReproject(const std::vector<cv::Mat>& src, const std::vector<PhotoParam>& photoParams,
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

    std::vector<int> appearCount(numImages, 0);

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
        show[i] = reprojImages[i].clone();

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

    for (int i = 0; i < numImages; i++)
    {
        char buf[64];
        sprintf(buf, "%d", i);
        cv::imshow(buf, show[i]);
    }
    cv::waitKey(0);
}

// use this downSizeRatio had better keep small
void getPointPairsHistogram(const std::vector<cv::Mat>& src, const std::vector<PhotoParam>& photoParams,
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

enum OptimizeParamType
{
    EXPOSURE = 1,
    WHITE_BALANCE = 4,
};

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

void readFrom(std::vector<ImageInfo>& infos, const double* x, int anchorIndex, int optimizeWhat)
{
    int numInfos = infos.size();
    int offset = 0;
    for (int i = 0; i < numInfos; i++)
    {
        if (i != anchorIndex)
        {
            infos[i].fromOutside(x + offset, optimizeWhat);
            offset += infos[i].getNumParams(optimizeWhat);
        }
    }
}

void writeTo(const std::vector<ImageInfo>& infos, double* x, int anchorIndex, int optimizeWhat)
{
    int numInfos = infos.size();
    int offset = 0;
    for (int i = 0; i < numInfos; i++)
    {
        if (i != anchorIndex)
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
    int anchoIndex;
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

void errorFunc(double* p, double* hx, int m, int n, void* data)
{
    ExternData* edata = (ExternData*)data;
    const std::vector<ImageInfo>& infos = edata->imageInfos;
    const std::vector<ValuePair>& pairs = edata->pairs;
    int anchorIndex = edata->anchoIndex;

    std::vector<double> pv(m);
    memcpy(pv.data(), p, m * 8);

    readFrom(edata->imageInfos, p, anchorIndex, edata->optimizeWhat);
    int numImages = infos.size();

    std::vector<Transform> transforms(numImages);
    for (int i = 0; i < numImages; i++)
        transforms[i] = Transform(infos[i]);

    int index = 0;
    for (int i = 0; i < numImages; i++)
    {
        if (i == anchorIndex)
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

    //std::vector<double> pp(m), hxhx(n);
    //memcpy(pp.data(), p, m * sizeof(double));
    //memcpy(hxhx.data(), hx, n * sizeof(double));

    printf("call count %d, sqr err = %f, avg err %f\n", edata->errorFuncCallCount, sqrErr, sqrt(sqrErr / n));
}

#include "levmar.h"

void optimize(const std::vector<ValuePair>& valuePairs, int numImages, int anchorIndex,
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
        int m = numImages * numParams;
        if (anchorIndex >= 0 && anchorIndex < numImages)
            m -= numParams;
        std::vector<double> p(m, 0.0);

        // vector for errors
        int n = 2 * 3 * valuePairs.size() + 3 * numImages + 1;
        if (anchorIndex >= 0 && anchorIndex < numImages)
            n -= 3;
        std::vector<double> x(n, 0.0);

        writeTo(imageInfos, p.data(), anchorIndex, option);

        // covariance matrix at solution
        cv::Mat cov(m, m, CV_64FC1);

        ExternData edata(imageInfos, valuePairs);
        edata.huberSigma = 5.0 / 255;
        edata.errorFuncCallCount = 0;
        edata.optimizeWhat = option;
        edata.anchoIndex = anchorIndex;
        edata.meanVals.resize(numImages);
        for (int i = 0; i < numImages; i++)
            edata.meanVals[i] = imageInfos[i].meanVals;

        ret = dlevmar_dif(&errorFunc, &(p[0]), &(x[0]), m, n, maxIter, optimOpts, info, NULL, (double*)cov.data, &edata);  // no jacobian
        // copy to source images (data.m_imgs)
        readFrom(imageInfos, p.data(), anchorIndex, option);
    }

    for (int i = 0; i < numImages; i++)
    {
        printf("[%d] e = %f, blue = %f, red = %f\n",
            i, imageInfos[i].exposure, imageInfos[i].whiteBalanceBlue, imageInfos[i].whiteBalanceRed);
    }
    //cv::waitKey(0);

    outImageInfos = imageInfos;
}

void getLUT(unsigned char lut[256], double k)
{
    CV_Assert(k > 0);
    for (int i = 0; i < 256; i++)
        lut[i] = cv::saturate_cast<unsigned char>(k * i);
}

double calcMaxScale(const std::vector<double>& es, const std::vector<double>& rs, const std::vector<double>& bs)
{
    int size = es.size();
    CV_Assert(size > 0 && rs.size() == size && bs.size() == size);

    double scale = 0;
    for (int i = 0; i < size; i++)
    {
        scale = es[i] > scale ? es[i] : scale;
        double s;
        s = es[i] * rs[i];
        scale = s > scale ? s : scale;
        s = es[i] * bs[i];
        scale = s > scale ? s : scale;
    }
    return scale;
}

void getLUTMaxScale(unsigned char LUT[256], double k, double maxK)
{
    CV_Assert(k > 0);
    if (maxK <= 1.05)
    {
        for (int i = 0; i < 256; i++)
            LUT[i] = cv::saturate_cast<unsigned char>(k * i);
        return;
    }
    LUT[0] = 0;
    for (int i = 1; i < 256; i++)
    {
        double val = i / 255.0 * k;
        cv::Point2d p0(0, 0), p1(1, 1), p2(maxK, 1);
        double a = p0.x + p2.x - 2 * p1.x, b = 2 * (p1.x - p0.x), c = p0.x - val;
        double m = -b / (2 * a), n = sqrt(b * b - 4 * a * c) / (2 * a);
        double t0 = m - n, t1 = m + n, t;
        if (t0 <= 1 && t0 >= 0)
            t = t0;
        else if (t1 <= 1 && t1 >= 0)
            t = t1;
        else
            //CV_Assert(0);
        {
            if (i < 2)
                t = 0;
            if (i > 253)
                t = 1;
        }
        double y = (1 - t) * (1 - t) * p0.y + 2 * (1 - t) * t * p1.y + t * t * p2.y;
        LUT[i] = cv::saturate_cast<unsigned char>(y * 255);
    }
}

void correct(const std::vector<cv::Mat>& src, const std::vector<double>& es, 
    const std::vector<double>& rs, const std::vector<double>& bs, std::vector<cv::Mat>& dst)
{
    int numImages = src.size();

    double maxE = 0;
    for (int i = 0; i < numImages; i++)
    {
        double e = es[i];
        maxE = e > maxE ? e : maxE;
    }
    double maxScale = calcMaxScale(es, rs, bs);

    dst.resize(numImages);
    char buf[64];
    unsigned char lutr[256], lutg[256], lutb[256];
    for (int i = 0; i < numImages; i++)
    {
        dst[i].create(src[i].size(), CV_8UC3);
        int rows = dst[i].rows, cols = dst[i].cols;

        double e = es[i];
        //e /= maxE;
        double r = rs[i];
        double b = bs[i];
        getLUT(lutr, e * r);
        getLUT(lutg, e);
        getLUT(lutb, e * b);
        //getLUTMaxScale(lutr, e * r, maxScale);
        //getLUTMaxScale(lutg, e, maxScale);
        //getLUTMaxScale(lutb, e * b, maxScale);
        for (int y = 0; y < rows; y++)
        {
            const unsigned char* ptrSrc = src[i].ptr<unsigned char>(y);
            unsigned char* ptrDst = dst[i].ptr<unsigned char>(y);
            for (int x = 0; x < cols; x++)
            {
                //ptrDst[0] = cv::saturate_cast<unsigned char>(ptrSrc[0] * e * b);
                //ptrDst[1] = cv::saturate_cast<unsigned char>(ptrSrc[1] * e);
                //ptrDst[2] = cv::saturate_cast<unsigned char>(ptrSrc[2] * e * r);

                ptrDst[0] = lutb[ptrSrc[0]];
                ptrDst[1] = lutg[ptrSrc[1]];
                ptrDst[2] = lutr[ptrSrc[2]];

                //ptrDst[0] = pow(ptrSrc[0] / 255.0 * e, 1.0 / 2.2) * 255;
                //ptrDst[1] = pow(ptrSrc[1] / 255.0 * e, 1.0 / 2.2) * 255;
                //ptrDst[2] = pow(ptrSrc[2] / 255.0 * e, 1.0 / 2.2) * 255;

                ptrSrc += 3;
                ptrDst += 3;
            }
        }

        sprintf(buf, "dst image %d", i);
        cv::Mat show;
        cv::resize(dst[i], show, cv::Size(), 0.5, 0.5);
        cv::imshow(buf, show);
    }
    cv::waitKey(0);
}

void exposureColorOptimize(const std::vector<cv::Mat>& images, const std::vector<PhotoParam>& params,
    const std::vector<int> anchorIndexes, const std::vector<int>& optimizeOptions,
    std::vector<double>& exposures, std::vector<double>& redRatios, std::vector<double>& blueRatios)
{
    int numImages = images.size();
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
    optimize(pairs, numImages, numImages - 2, testSrc[0].size(), optimizeOptions, imageInfos);

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

void run(const std::vector<cv::Mat>& images, const std::vector<PhotoParam>& params,
    const std::vector<int>& optimizeOptions)
{
    int numImages = images.size();
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
    optimize(pairs, numImages, numImages - 2, testSrc[0].size(), optimizeOptions, imageInfos);

    std::vector<double> exposures, redRatios, blueRatios;
    exposures.resize(numImages);
    redRatios.resize(numImages);
    blueRatios.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        exposures[i] = 1.0 / imageInfos[i].exposure;
        redRatios[i] = 1.0 / imageInfos[i].whiteBalanceRed;
        blueRatios[i] = 1.0 / imageInfos[i].whiteBalanceBlue;
    }

    std::vector<cv::Mat> dstImages;
    correct(images, exposures, redRatios, blueRatios, dstImages);

    cv::Size dstSize(1200, 600);
    std::vector<cv::Mat> maps, masks, weights;
    getReprojectMapsAndMasks(params, images[0].size(), dstSize, maps, masks);

    std::vector<cv::Mat> reprojImages;
    //ztool::Timer t;
    //for (int i = 0; i < 100; i++)
    reprojectParallel(dstImages, reprojImages, maps);
    //t.end();
    //printf("t = %f\n", t.elapse());

    cv::Mat blendImage;

    TilingLinearBlend blender;
    blender.prepare(masks, 50);
    blender.blend(reprojImages, blendImage);
    cv::imshow("blend", blendImage);
    cv::imwrite("out.bmp", blendImage);

    TilingMultibandBlendFast mbBlender;
    mbBlender.prepare(masks, 10, 8);
    mbBlender.blend(reprojImages, blendImage);
    cv::imshow("mb blend", blendImage);

    cv::waitKey(0);
}

void loadImages(const std::vector<std::string>& imagePaths, std::vector<cv::Mat>& images)
{
    int numImages = imagePaths.size();
    images.resize(numImages);
    for (int i = 0; i < numImages; i++)
        images[i] = cv::imread(imagePaths[i]);
}

int main()
{
    double PI = 3.1415926;

    std::vector<std::string> imagePaths;
    std::vector<PhotoParam> params;
    std::vector<cv::Mat> srcImages;

    std::vector<int> opts;
    opts.push_back(EXPOSURE | WHITE_BALANCE);
    //opts.push_back(WHITE_BALANCE);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-00.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-01.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-02.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-03.jpg");
    loadPhotoParams("F:\\panoimage\\detuoffice\\detuoffice.xml", params);
    loadImages(imagePaths, srcImages);
    run(srcImages, params, opts);

    //xxxx
    //imagePaths.clear();
    //imagePaths.push_back("F:\\panoimage\\detuoffice2\\input-00.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice2\\input-01.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice2\\input-02.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice2\\input-03.jpg");
    //loadPhotoParamFromXML("F:\\panoimage\\detuoffice2\\detu.xml", params);
    //run(imagePaths, params, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\919-4\\snapshot0(2).bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\snapshot1(2).bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\snapshot2(2).bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\snapshot3(2).bmp");
    loadPhotoParamFromXML("F:\\panoimage\\919-4\\vrdl4.xml", params);
    loadImages(imagePaths, srcImages);
    run(srcImages, params, opts);

    //xxxx
    //imagePaths.clear();
    //imagePaths.push_back("F:\\panovideo\\ricoh m15\\image2-128.bmp");
    //imagePaths.push_back("F:\\panovideo\\ricoh m15\\image2-128.bmp");
    //loadPhotoParamFromXML("F:\\panovideo\\ricoh m15\\parambestcircle.xml", params);

    //xxxx
    //imagePaths.clear();
    //imagePaths.push_back("F:\\panoimage\\vrdlc\\2016_1011_153743_001.JPG");
    //imagePaths.push_back("F:\\panoimage\\vrdlc\\2016_1011_153743_001.JPG");
    //loadPhotoParamFromXML("F:\\panoimage\\vrdlc\\vrdl-201610112019.xml", params);
    //run(imagePaths, params, opts);

    //xxxx
    //imagePaths.clear();
    //imagePaths.push_back("F:\\panoimage\\vrdlc\\QQÍ¼Æ¬20161014101159.png");
    //imagePaths.push_back("F:\\panoimage\\vrdlc\\QQÍ¼Æ¬20161014101159.png");
    //loadPhotoParamFromXML("F:\\panoimage\\vrdlc\\vrdl-201610112019small.xml", params);
    //run(imagePaths, params, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\919-4-1\\snapshot0.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4-1\\snapshot1.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4-1\\snapshot2.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4-1\\snapshot3.bmp");
    loadPhotoParamFromXML("F:\\panoimage\\919-4-1\\vrdl(4).xml", params);
    loadImages(imagePaths, srcImages);
    run(srcImages, params, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\0.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\1.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\2.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\3.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\4.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\5.jpg");
    loadPhotoParamFromXML("F:\\panoimage\\zhanxiang\\zhanxiang.xml", params);
    rotateCameras(params, 0, 35.264 / 180 * PI, PI / 4);
    loadImages(imagePaths, srcImages);
    run(srcImages, params, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image5.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\test6\\proj.pvs", params);
    loadImages(imagePaths, srcImages);
    run(srcImages, params, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image5.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\test6\\proj.pvs", params);
    loadImages(imagePaths, srcImages);
    run(srcImages, params, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image5.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\test6\\proj.pvs", params);
    loadImages(imagePaths, srcImages);
    run(srcImages, params, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image5.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\test6\\proj.pvs", params);
    loadImages(imagePaths, srcImages);
    run(srcImages, params, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\2\\1\\1.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\2.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\3.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\4.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\5.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\6.jpg");
    loadPhotoParamFromXML("F:\\panoimage\\2\\1\\distortnew.xml", params);
    rotateCameras(params, 0, -35.264 / 180 * PI, -PI / 4);
    loadImages(imagePaths, srcImages);
    run(srcImages, params, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\changtai\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image5.bmp");
    loadPhotoParamFromXML("F:\\panoimage\\changtai\\test_test5_cam_param.xml", params);
    loadImages(imagePaths, srcImages);
    run(srcImages, params, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\1.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\2.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\3.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\4.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\5.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\6.MP4.jpg");
    loadPhotoParamFromXML("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\proj.pvs", params);
    loadImages(imagePaths, srcImages);
    run(srcImages, params, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image0.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image1.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image2.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image3.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image4.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image5.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image6.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\chengdu\\1\\proj.pvs", params);
    loadImages(imagePaths, srcImages);
    run(srcImages, params, opts);

    return 0;
}
