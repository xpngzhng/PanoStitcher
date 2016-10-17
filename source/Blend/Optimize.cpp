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

struct UniformToSphereConverter
{
    void init(int radius)
    {
        length = radius * 2;
        std::vector<double> temp(length, 0);
        double r2 = radius * radius;
        for (int i = 0; i < length; i++)
        {
            double diff = i - radius;
            double val = sqrt(r2 - diff * diff);
            temp[i] = val;
        }
        for (int i = 1; i < length; i++)
            temp[i] += temp[i - 1];
        double scale = length / temp[length - 1];
        for (int i = 0; i < length; i++)
            temp[i] *= scale;

        lut.resize(length);
        for (int i = 0; i < length; i++)
        {
            int lowIndex = 0, highIndex = length - 1;
            for (int j = 0; j < length - 1; j++)
            {
                if (temp[j] <= i && temp[j + 1] >= i)
                {
                    lowIndex = j;
                    break;
                }
            }
            for (int j = length - 1; j > 0; j--)
            {
                if (temp[j - 1] <= i && temp[j] >= i)
                {
                    highIndex = j;
                    break;
                }
            }
            if (lowIndex == highIndex)
            {
                lut[i] = lowIndex;
                continue;
            }
            double diff = highIndex - lowIndex;
            double lambda = (i - lowIndex) / diff;
            lut[i] = lambda * lowIndex + (1 - lambda) * highIndex + 0.5;
        }
    }
    int transform(int i) const
    {
        if (i < 0)
            return 0;
        if (i >= length)
            return length - 1;
        return lut[i];
    }
    std::vector<int> lut;
    int length;
};

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

    UniformToSphereConverter cvtUToS;
    cvtUToS.init(erHeight / 2);

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

enum { OPTIMIZE_VAL_NUM = 8 };
enum { EMOR_COEFF_LENGTH = 5 };
enum { VIGNETT_COEFF_LENGTH = 4 };

enum OptimizeParamType
{
    EXPOSURE = 1,
    RESPONSE_CURVE = 2,
    WHITE_BALANCE = 4,
    VIGNETTE = 8
};

enum ResponseCurveType
{
    IDENTITY = 0,
    EMOR = 1,
    GAMMA = 2
};

struct ImageInfo
{
    ImageInfo()
    {

    }

    ImageInfo(const cv::Size& size_, int responseCurveType_)
    {
        memset(emorCoeffs, 0, sizeof(emorCoeffs));
        memset(radialVignettCoeffs, 0, sizeof(radialVignettCoeffs));
        radialVignettCoeffs[0] = 1;
        //exposureExponent = 0;
        exposureExponent = 1;
        gamma = 1;
        whiteBalanceRed = 1;
        whiteBalanceBlue = 1;
        size = size_;
        responseCurveType = responseCurveType_;
    }

    double getExposure() const
    {
        //return 1.0 / pow(2.0, exposureExponent);
        return exposureExponent;
    }

    void setExposure(double e)
    {
        //exposureExponent = log2(1 / e);
        exposureExponent = e;
    }

    int static getNumParams(int optimizeWhat, int responseCurveType)
    {
        int num = 0;
        if (optimizeWhat & EXPOSURE)
            num += 1;
        if (optimizeWhat & RESPONSE_CURVE)
        {
            if (responseCurveType == EMOR)
                num += EMOR_COEFF_LENGTH;
            else if (responseCurveType == GAMMA)
                num += 1;
        }
        if (optimizeWhat & VIGNETTE)
            num += VIGNETT_COEFF_LENGTH;
        if (optimizeWhat & WHITE_BALANCE)
            num += 2;
        return num;
    }

    int getNumParams(int optimizeWhat) const
    {
        return getNumParams(optimizeWhat, responseCurveType);
    }

    void fromOutside(const double* x, int optimizeWhat)
    {
        int index = 0;
        if (optimizeWhat & RESPONSE_CURVE)
        {
            if (responseCurveType == EMOR)
            {
                for (; index < EMOR_COEFF_LENGTH; index++)
                    emorCoeffs[index] = x[index];
            }
            else if (responseCurveType == GAMMA)
                gamma = x[index++];
        }
        if (optimizeWhat & VIGNETTE)
        {
            int lastLength = index;
            for (; index < lastLength + VIGNETT_COEFF_LENGTH; index++)
                radialVignettCoeffs[index - EMOR_COEFF_LENGTH] = x[index];
        }
        if (optimizeWhat & EXPOSURE)
        {
            setExposure(x[index++]);
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
        if (optimizeWhat & RESPONSE_CURVE)
        {
            if (responseCurveType == EMOR)
            {
                for (; index < EMOR_COEFF_LENGTH; index++)
                    x[index] = emorCoeffs[index];
            }
            else if (responseCurveType == GAMMA)
                x[index++] = gamma;
            
        }
        if (optimizeWhat & VIGNETTE)
        {
            int lastLength = index;
            for (; index < lastLength + VIGNETT_COEFF_LENGTH; index++)
                x[index] = radialVignettCoeffs[index - EMOR_COEFF_LENGTH];
        }
        if (optimizeWhat & EXPOSURE)
        {
            x[index++] = getExposure();
        }
        if (optimizeWhat & WHITE_BALANCE)
        {
            x[index++] = whiteBalanceRed;
            x[index] = whiteBalanceBlue;
        }
    }
    
    double emorCoeffs[EMOR_COEFF_LENGTH];
    double radialVignettCoeffs[VIGNETT_COEFF_LENGTH];
    double exposureExponent;
    double whiteBalanceRed;
    double whiteBalanceBlue;
    double gamma;
    cv::Size size;
    int responseCurveType;
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

#include "emor.h"
struct Transform
{
    Transform()
    {

    }

    Transform(const ImageInfo& imageInfo)
    {
        responseCurveType = imageInfo.responseCurveType;
        gamma = imageInfo.gamma;

        memset(lut, 0, sizeof(lut));
        if (responseCurveType == EMOR)
        {
            for (int i = 0; i < LUT_LENGTH; i++)
            {
                double t = EMoR::f0[i];
                for (int k = 0; k < EMOR_COEFF_LENGTH; k++)
                    t += EMoR::h[k][i] * imageInfo.emorCoeffs[k];
                lut[i] = t;
            }
        }
        else if (responseCurveType == GAMMA)
        {
            double gamma = imageInfo.gamma;
            for (int i = 0; i < LUT_LENGTH; i++)
            {
                double s = double(i) / (LUT_LENGTH - 1);
                lut[i] = pow(s, gamma);
            }
        }
        else
        {
            for (int i = 0; i < LUT_LENGTH; i++)
                lut[i] = double(i) / (LUT_LENGTH - 1);
        }

        vigCenterX = imageInfo.size.width / 2;
        vigCenterY = imageInfo.size.height / 2;
        memcpy(vigCoeffs, imageInfo.radialVignettCoeffs, sizeof(vigCoeffs));
        radiusScale = 1.0 / sqrt(vigCenterX * vigCenterX + vigCenterY * vigCenterY);

        exposure = imageInfo.getExposure();

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
        double scale = /*calcVigFactor(p) **/ exposure;
        double b = val[0] * scale * whiteBalanceBlue;
        double g = val[1] * scale;
        double r = val[2] * scale * whiteBalanceRed;
        return cv::Vec3d(LUT(b), LUT(g), LUT(r));
    }

    cv::Vec3d applyInverse(const cv::Point& p, const cv::Vec3d& val) const
    {
        double scale = 1.0 / (/*calcVigFactor(p) **/ exposure);
        double b = invLUT(val[0]) * scale;
        double g = invLUT(val[1]) * scale;
        double r = invLUT(val[2]) * scale;
        r /= whiteBalanceRed;
        b /= whiteBalanceBlue;
        return cv::Vec3d(b, g, r);
    }

    cv::Vec3d applyInverseExposureOnly(const cv::Point& p, const cv::Vec3d& val) const
    {
        double scale = 1.0 / (/*calcVigFactor(p) **/ exposure);
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

    double calcVigFactor(const cv::Point& p) const
    {
        double diffx = (p.x - vigCenterX) * radiusScale;
        double diffy = (p.y - vigCenterY) * radiusScale;
        double vig = vigCoeffs[0];
        double r2 = diffx * diffx + diffy * diffy;
        double r = r2;
        for (int i = 1; i < VIGNETT_COEFF_LENGTH; i++)
        {
            vig += vigCoeffs[i] * r;
            r *= r2;
        }
        return vig;
    }

    double LUT(double val) const
    {
        if (responseCurveType == EMOR/*1*/)
        {
            if (val <= 0)
                return lut[0];
            if (val >= 1)
                return lut[LUT_LENGTH - 1];

            return lut[int(val * LUT_LENGTH)];
        }
        else if (responseCurveType == GAMMA)
            return /*val <= 0 ? 0 : val > 1 ? 1 : */pow(val, gamma);
        else
            return /*val <= 0 ? 0 : val > 1 ? 1 : */val;
    }

    double invLUT(double val) const
    {
        if (responseCurveType == EMOR/*1*/)
        {
            if (val <= 0)
                return 0;
            if (val >= 1)
                return 1;

            int lowIdx = 0, upIdx = LUT_LENGTH - 1;
            for (int i = 0; i < LUT_LENGTH - 1; i++)
            {
                if (lut[i] <= val && lut[i + 1] >= val)
                {
                    lowIdx = i;
                    break;
                }
            }
            for (int i = LUT_LENGTH - 1; i > 0; i--)
            {
                if (lut[i - 1] <= val && lut[i] >= val)
                {
                    upIdx = i;
                    break;
                }
            }
            if (lowIdx == upIdx)
                return double(lowIdx) / (LUT_LENGTH - 1);

            double diff = lut[upIdx] - lut[lowIdx];
            double lambda = (val - lut[lowIdx]) / diff;
            return (lowIdx * (1 - lambda) + upIdx * lambda) / (LUT_LENGTH - 1);
        }
        else if (responseCurveType == GAMMA)
            return /*val <= 0 ? 0 : val > 1 ? 1 : */pow(val, 1.0 / gamma);
        else
            return /*val <= 0 ? 0 : val > 1 ? 1 : */val;
    }

    void enforceMonotonicity()
    {
        double val = lut[LUT_LENGTH - 1];
        for (int i = 0; i < LUT_LENGTH - 1; i++)
        {
            if (lut[i] > val)
                lut[i] = val;
            if (lut[i + 1] < lut[i])
                lut[i + 1] = lut[i];
        }
    }

    void showLUT(const std::string& winName)
    {
        cv::Mat image = cv::Mat::zeros(LUT_LENGTH, LUT_LENGTH, CV_8UC1);
        for (int i = 0; i < LUT_LENGTH - 1; i++)
        {
            cv::line(image, cv::Point(i, LUT_LENGTH * (1 - lut[i])), 
                            cv::Point(i + 1, LUT_LENGTH * (1 - lut[i + 1])), cv::Scalar(255));
        }
        cv::imshow(winName, image);
    }

    enum { LUT_LENGTH = 1024 };
    double lut[LUT_LENGTH];
    double vigCenterX, vigCenterY;
    double vigCoeffs[VIGNETT_COEFF_LENGTH];
    double radiusScale;
    double exposure;
    double whiteBalanceRed;
    double whiteBalanceBlue;
    double gamma;
    int responseCurveType;
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

    //for (int i = 0; i < numImages; i++)
    //{
    //    printf("[%d] e = %f, gamma = %f, blue = %f, red = %f\n",
    //        i, infos[i].getExposure(), infos[i].gamma, infos[i].whiteBalanceBlue, infos[i].whiteBalanceRed);
    //}

    std::vector<Transform> transforms(numImages);
    for (int i = 0; i < numImages; i++)
        transforms[i] = Transform(infos[i]);

    int index = 0;
    for (int i = 0; i < numImages; i++)
    {
        if (i == anchorIndex)
            continue;

        Transform& trans = transforms[i];
        double err = 0;
        for (int j = 0; j < Transform::LUT_LENGTH; j++)
        {
            if (trans.lut[j] > trans.lut[j + 1])
            {
                double diff = trans.lut[j] - trans.lut[j + 1];
                err += diff * diff * 256;
            }
        }
        //printf("%f ", trans.lut[Transform::LUT_LENGTH - 1]);
        trans.enforceMonotonicity();
        //hx[index++] = err;
        hx[index++] = abs(transforms[i].exposure - 1) * 2;
        hx[index++] = abs(transforms[i].whiteBalanceBlue - 1) * 2;
        hx[index++] = abs(transforms[i].whiteBalanceRed - 1) * 2;
    }
    //printf("\n");

    double huberSigma = edata->huberSigma;

    double sqrErr = 0;
    int numPairs = pairs.size();
    double rdiff = 0, gdiff = 0, bdiff = 0;
    for (int i = 0; i < numPairs; i++)
    {
        const ValuePair& pair = pairs[i];
        
        if ((double(pair.iVal[0]) - pair.jVal[0]) > 10 ||
            (double(pair.iVal[1]) - pair.jVal[1]) > 10 ||
            (double(pair.iVal[2]) - pair.jVal[2]) > 10)
        {
            //printf("diff large\n");
            int a = 0;
        }

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

        //hx[index++] = weightHuber(abs(lightI[0] - pair.iValD[0]) * 0.05, huberSigma);
        //hx[index++] = weightHuber(abs(lightI[1] - pair.iValD[1]) * 0.05, huberSigma);
        //hx[index++] = weightHuber(abs(lightI[2] - pair.iValD[2]) * 0.05, huberSigma);

        //hx[index++] = weightHuber(abs(lightJ[0] - pair.jValD[0]) * 0.05, huberSigma);
        //hx[index++] = weightHuber(abs(lightJ[1] - pair.jValD[1]) * 0.05, huberSigma);
        //hx[index++] = weightHuber(abs(lightJ[2] - pair.jValD[2]) * 0.05, huberSigma);

        //printf("err %d: %f %f %f %f %f %f\n", i, errI[0], errI[1], errI[2], errJ[0], errJ[1], errJ[2]);

        bdiff += lightI[0] - pair.iValD[0];
        gdiff += lightI[1] - pair.iValD[1];
        rdiff += lightI[2] - pair.iValD[2];

        bdiff += lightJ[0] - pair.jValD[0];
        gdiff += lightJ[1] - pair.jValD[1];
        rdiff += lightJ[2] - pair.jValD[2];

        sqrErr += errI.dot(errI);
        sqrErr += errJ.dot(errJ);
    }

    //hx[index++] = weightHuber(abs(bdiff + gdiff + rdiff) * 1.0 / numPairs, huberSigma);
    //hx[index++] = weightHuber(abs(gdiff), huberSigma);
    //hx[index++] = weightHuber(abs(rdiff), huberSigma);

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
    const cv::Size& imageSize, int responseCurveType, const std::vector<int>& optimizeOptions,
    std::vector<ImageInfo>& outImageInfos)
{
    std::vector<ImageInfo> imageInfos(numImages);
    for (int i = 0; i < numImages; i++)
    {
        ImageInfo info(imageSize, responseCurveType);
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
        //int option = i == 0 ? EXPOSURE | RESPONSE_CURVE/* | WHITE_BALANCE*/ : (WHITE_BALANCE);
        int option = optimizeOptions[i];
        int numParams = ImageInfo::getNumParams(option, responseCurveType);

        // parameters
        int m = numImages * numParams;
        if (anchorIndex >= 0 && anchorIndex < numImages)
            m -= numParams;
        std::vector<double> p(m, 0.0);

        // vector for errors
        int n = /*2 **/ 2 * 3 * valuePairs.size() + 3 * numImages + 1/*0*/;
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
        printf("[%d] e = %f, gamma = %f, blue = %f, red = %f\n",
            i, imageInfos[i].getExposure(), imageInfos[i].gamma, imageInfos[i].whiteBalanceBlue, imageInfos[i].whiteBalanceRed);
        char buf[256];
        sprintf(buf, "emor lut %d", i);
        Transform t(imageInfos[i]);
        t.enforceMonotonicity();
        //t.showLUT(buf);
    }
    //cv::waitKey(0);

    outImageInfos = imageInfos;
}

void getLUT(unsigned char lut[256], double k)
{
    CV_Assert(k > 0);
    if (/*abs(k - 1) < 0.02*/1)
    {
        for (int i = 0; i < 256; i++)
            lut[i] = cv::saturate_cast<unsigned char>(i * k);
    }
    else
    {
        cv::Point2d p0(0, 0), p1 = k > 1 ? cv::Point(255 / k, 255) : cv::Point(255, k * 255), p2(255, 255);
        lut[0] = 0;
        lut[255] = 255;
        for (int i = 1; i < 255; i++)
        {
            double a = p0.x + p2.x - 2 * p1.x, b = 2 * (p1.x - p0.x), c = p0.x - i;
            double m = -b / (2 * a), n = sqrt(b * b - 4 * a * c) / (2 * a);
            double t0 = m - n, t1 = m + n, t;
            if (t0 < 1 && t0 > 0)
                t = t0;
            else if (t1 < 1 && t1 > 0)
                t = t1;
            else
                CV_Assert(0);
            double y = (1 - t) * (1 - t) * p0.y + 2 * (1 - t) * t * p1.y + t * t * p2.y + 0.5;
            y = y < 0 ? 0 : (y > 255 ? 255 : y);
            lut[i] = y;
        }
    }
}

void getLUT(unsigned char lut[256], double k, double gamma)
{
    CV_Assert(k > 0);
    for (int i = 0; i < 256; i++)
        lut[i] = cv::saturate_cast<unsigned char>(pow((double)i / 255, gamma) * k * 255);
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

void correct(const std::vector<cv::Mat>& src, const std::vector<ImageInfo>& infos, std::vector<cv::Mat>& dst)
{
    int numImages = src.size();

    std::vector<Transform> transforms(numImages);
    for (int i = 0; i < numImages; i++)
        transforms[i] = Transform(infos[i]);

    double maxE = 0;
    for (int i = 0; i < numImages; i++)
    {
        double e = 1.0 / infos[i].getExposure();
        maxE = e > maxE ? e : maxE;
    }

    std::vector<double> es(numImages), bs(numImages), rs(numImages);
    for (int i = 0; i < numImages; i++)
    {
        es[i] = 1.0 / infos[i].getExposure();
        rs[i] = 1.0 / infos[i].whiteBalanceRed;
        bs[i] = 1.0 / infos[i].whiteBalanceBlue;
    }
    double maxScale = calcMaxScale(es, rs, bs);

    dst.resize(numImages);
    char buf[64];
    unsigned char lutr[256], lutg[256], lutb[256];
    for (int i = 0; i < numImages; i++)
    {
        Transform& trans = transforms[i];
        dst[i].create(src[i].size(), CV_8UC3);
        int rows = dst[i].rows, cols = dst[i].cols;
        
        /*
        double e = infos[i].getExposure();
        for (int y = 0; y < rows; y++)
        {
            const unsigned char* ptrSrc = src[i].ptr<unsigned char>(y);
            unsigned char* ptrDst = dst[i].ptr<unsigned char>(y);
            for (int x = 0; x < cols; x++)
            {
                double b = ptrSrc[0] / 255.0, g = ptrSrc[1] / 255.0, r = ptrSrc[2] / 255.0;
                cv::Vec3d d = trans.applyInverse(cv::Point(x, y), cv::Vec3d(b, g, r));
                //ptrDst[0] = cv::saturate_cast<unsigned char>(d[0] * 255);
                //ptrDst[1] = cv::saturate_cast<unsigned char>(d[1] * 255);
                //ptrDst[2] = cv::saturate_cast<unsigned char>(d[2] * 255);
                ptrDst[0] = cv::saturate_cast<unsigned char>(trans.LUT(d[0]) * 255);
                ptrDst[1] = cv::saturate_cast<unsigned char>(trans.LUT(d[1]) * 255);
                ptrDst[2] = cv::saturate_cast<unsigned char>(trans.LUT(d[2]) * 255);
                ptrSrc += 3;
                ptrDst += 3;
            }
        }
        */
        
        double e = 1.0 / infos[i].getExposure();
        //e /= maxE;
        double r = 1.0 / infos[i].whiteBalanceRed;
        double b = 1.0 / infos[i].whiteBalanceBlue;
        double gamma = 1.0 / infos[i].gamma;
        getLUT(lutr, e * r, gamma);
        getLUT(lutg, e, gamma);
        getLUT(lutb, e * b, gamma);
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

void huginCorrect(const std::vector<cv::Mat>& src, const std::vector<PhotoParam>& params,
    int responseCurveType, std::vector<std::vector<std::vector<unsigned char> > >& luts)
{
    int numImages = src.size();

    int resizeTimes = 0;
    int minWidth = 80, minHeight = 60;
    resizeTimes = getResizeTimes(src[0].cols, src[0].rows, minWidth, minHeight);

    std::vector<cv::Mat> testSrc(numImages);
    if (resizeTimes == 0)
    {
        testSrc = src;
    }
    else
    {
        for (int i = 0; i < numImages; i++)
        {
            cv::Mat large = src[i];
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
    getPointPairsRandom(testSrc, params, downSizePower, pairs);

    std::vector<int> opts;
    opts.push_back(EXPOSURE);
    std::vector<ImageInfo> infos;
    optimize(pairs, numImages, -1, testSrc[0].size(), responseCurveType, opts, infos);

    std::vector<Transform> transforms(numImages);
    for (int i = 0; i < numImages; i++)
        transforms[i] = Transform(infos[i]);

    luts.resize(numImages);
    char buf[64];
    for (int i = 0; i < numImages; i++)
    {
        Transform& trans = transforms[i];
        double e = 1.0 / infos[i].getExposure();
        double r = 1.0 / infos[i].whiteBalanceRed;
        double b = 1.0 / infos[i].whiteBalanceBlue;
        luts[i].resize(3);
        luts[i][0].resize(256);
        luts[i][1].resize(256);
        luts[i][2].resize(256);
        getLUT(luts[i][2].data(), e * r);
        getLUT(luts[i][1].data(), e);
        getLUT(luts[i][0].data(), e * b);
    }
}

void run(const std::vector<std::string>& imagePaths, const std::vector<PhotoParam>& params,
    int responseCurveType, const std::vector<int>& optimizeOptions)
{
    int numImages = imagePaths.size();
    std::vector<cv::Mat> src(numImages);
    for (int i = 0; i < numImages; i++)
        src[i] = cv::imread(imagePaths[i]);

    int resizeTimes = 0;
    int minWidth = 100, minHeight = 100;
    resizeTimes = getResizeTimes(src[0].cols, src[0].rows, minWidth, minHeight);

    std::vector<cv::Mat> testSrc(numImages);
    if (resizeTimes == 0)
    {
        testSrc = src;
    }
    else
    {
        for (int i = 0; i < numImages; i++)
        {
            cv::Mat large = src[i];
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
    optimize(pairs, numImages, numImages -2, testSrc[0].size(), responseCurveType, optimizeOptions, imageInfos);

    std::vector<cv::Mat> dstImages;
    correct(src, imageInfos, dstImages);

    cv::Size dstSize(1200, 600);
    std::vector<cv::Mat> maps, masks, weights;
    getReprojectMapsAndMasks(params, src[0].size(), dstSize, maps, masks);

    std::vector<cv::Mat> images;
    //ztool::Timer t;
    //for (int i = 0; i < 100; i++)
    reprojectParallel(dstImages, images, maps);
    //t.end();
    //printf("t = %f\n", t.elapse());

    cv::Mat blendImage;

    TilingLinearBlend blender;
    blender.prepare(masks, 50);
    blender.blend(images, blendImage);
    cv::imshow("blend", blendImage);
    cv::imwrite("out.bmp", blendImage);

    TilingMultibandBlendFast mbBlender;
    mbBlender.prepare(masks, 10, 8);
    mbBlender.blend(images, blendImage);
    cv::imshow("mb blend", blendImage);

    cv::waitKey(0);
}

int main()
{
    double PI = 3.1415926;

    std::vector<std::string> imagePaths;
    std::vector<PhotoParam> params;

    int respCurveType = GAMMA;
    std::vector<int> opts;
    opts.push_back(EXPOSURE | WHITE_BALANCE);
    //opts.push_back(WHITE_BALANCE);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-00.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-01.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-02.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-03.jpg");
    loadPhotoParams("F:\\panoimage\\detuoffice\\detuoffice.xml", params);
    run(imagePaths, params, respCurveType, opts);

    //xxxx
    //imagePaths.clear();
    //imagePaths.push_back("F:\\panoimage\\detuoffice2\\input-00.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice2\\input-01.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice2\\input-02.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice2\\input-03.jpg");
    //loadPhotoParamFromXML("F:\\panoimage\\detuoffice2\\detu.xml", params);
    //run(imagePaths, params, respCurveType, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\919-4\\snapshot0(2).bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\snapshot1(2).bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\snapshot2(2).bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\snapshot3(2).bmp");
    loadPhotoParamFromXML("F:\\panoimage\\919-4\\vrdl4.xml", params);
    run(imagePaths, params, respCurveType, opts);

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
    //run(imagePaths, params, respCurveType, opts);

    //xxxx
    //imagePaths.clear();
    //imagePaths.push_back("F:\\panoimage\\vrdlc\\QQÍ¼Æ¬20161014101159.png");
    //imagePaths.push_back("F:\\panoimage\\vrdlc\\QQÍ¼Æ¬20161014101159.png");
    //loadPhotoParamFromXML("F:\\panoimage\\vrdlc\\vrdl-201610112019small.xml", params);
    //run(imagePaths, params, respCurveType, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\919-4-1\\snapshot0.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4-1\\snapshot1.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4-1\\snapshot2.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4-1\\snapshot3.bmp");
    loadPhotoParamFromXML("F:\\panoimage\\919-4-1\\vrdl(4).xml", params);
    run(imagePaths, params, respCurveType, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\0.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\1.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\2.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\3.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\4.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\5.jpg");
    loadPhotoParamFromXML("F:\\panoimage\\zhanxiang\\zhanxiang.xml", params);
    rotateCameras(params, 0, 35.264 / 180 * PI, PI / 4);
    run(imagePaths, params, respCurveType, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image5.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\test6\\proj.pvs", params);
    run(imagePaths, params, respCurveType, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image5.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\test6\\proj.pvs", params);
    run(imagePaths, params, respCurveType, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image5.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\test6\\proj.pvs", params);
    run(imagePaths, params, respCurveType, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image5.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\test6\\proj.pvs", params);
    run(imagePaths, params, respCurveType, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\2\\1\\1.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\2.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\3.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\4.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\5.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\6.jpg");
    loadPhotoParamFromXML("F:\\panoimage\\2\\1\\distortnew.xml", params);
    rotateCameras(params, 0, -35.264 / 180 * PI, -PI / 4);
    run(imagePaths, params, respCurveType, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\changtai\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image5.bmp");
    loadPhotoParamFromXML("F:\\panoimage\\changtai\\test_test5_cam_param.xml", params);
    run(imagePaths, params, respCurveType, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\1.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\2.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\3.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\4.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\5.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\6.MP4.jpg");
    loadPhotoParamFromXML("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\proj.pvs", params);
    run(imagePaths, params, respCurveType, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image0.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image1.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image2.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image3.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image4.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image5.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image6.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\chengdu\\1\\proj.pvs", params);
    run(imagePaths, params, respCurveType, opts);

    return 0;

    int numImages = imagePaths.size();
    std::vector<cv::Mat> src(numImages);
    for (int i = 0; i < numImages; i++)
        src[i] = cv::imread(imagePaths[i]);

    int resizeTimes = 0;
    int minWidth = 120, minHeight = 90;
    resizeTimes = getResizeTimes(src[0].cols, src[0].rows, minWidth, minHeight);
    
    std::vector<cv::Mat> testSrc(numImages);
    if (resizeTimes == 0)
    {
        testSrc = src;
    }
    else
    {
        for (int i = 0; i < numImages; i++)
        {
            cv::Mat large = src[i];
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
    //getPointPairsHistogram(testSrc, params, pairs);

    std::vector<int> options;
    opts.push_back(EXPOSURE);
    std::vector<ImageInfo> imageInfos;
    optimize(pairs, numImages, -1, testSrc[0].size(), GAMMA, options, imageInfos);

    std::vector<cv::Mat> dstImages;
    correct(src, imageInfos, dstImages);

    cv::Size dstSize(1600, 800);
    std::vector<cv::Mat> maps, masks, weights;
    getReprojectMapsAndMasks(params, src[0].size(), dstSize, maps, masks);

    std::vector<cv::Mat> images;
    //ztool::Timer t;
    //for (int i = 0; i < 100; i++)
    reprojectParallel(dstImages, images, maps);
    //t.end();
    //printf("t = %f\n", t.elapse());

    cv::Mat blendImage;

    TilingLinearBlend blender;
    blender.prepare(masks, 50);
    blender.blend(images, blendImage);
    cv::imshow("blend", blendImage);
    cv::imwrite("out.bmp", blendImage);

    TilingMultibandBlendFast mbBlender;
    mbBlender.prepare(masks, 10, 8);
    mbBlender.blend(images, blendImage);
    cv::imshow("mb blend", blendImage);
    
    cv::waitKey(0);
    return 0;
}

struct Data
{
    double* ptrIn;
    double* ptrOut;
    int count;
};

void compute(double *p, double *x, int m, int n, void *data)
{
    Data* ptrData = (Data*)data;
    double* ptrIn = ptrData->ptrIn;
    double* ptrOut = ptrData->ptrOut;
    for (int i = 0; i < n; i++)
        x[i] = ptrOut[i] - exp(p[0] * ptrIn[i * 2] * ptrIn[i * 2] + p[1] * ptrIn[i * 2 + 1] * ptrIn[i * 2 + 1]);
    ptrData->count++;
    printf("%d: a = %f, b = %f\n", ptrData->count, p[0], p[1]);
}

void jacob(double *p, double *x, int m, int n, void *data)
{
    Data* ptrData = (Data*)data;
    double* ptrIn = ptrData->ptrIn;
    double* ptrOut = ptrData->ptrOut;
    for (int i = 0; i < n; i++)
    {
        double y = exp(p[0] * ptrIn[i * 2] * ptrIn[i * 2] + p[1] * ptrIn[i * 2 + 1] * ptrIn[i * 2 + 1]);
        x[i * 2] = -y * ptrIn[i * 2] * ptrIn[i * 2];
        x[i * 2 + 1] = -y * ptrIn[i * 2 + 1] * ptrIn[i * 2 + 1];
    }
}

int mainy()
{
    int num = 1000;
    double beg = -3, end = 3;
    double a = -1, b = -1;
    cv::Mat input(num, 2, CV_64FC1);
    cv::Mat output(num, 1, CV_64FC1);
    cv::Mat noise(num, 1, CV_64FC1);
    cv::RNG rng;
    rng.fill(input, cv::RNG::UNIFORM, beg, end);
    rng.fill(noise, cv::RNG::NORMAL, 0, 0.01);
    double* ptrIn = (double*)input.data;
    double* ptrOut = (double*)output.data;
    double* ptrNoise = (double*)noise.data;
    for (int i = 0; i < num; i++)
        ptrOut[i] = exp(a * ptrIn[i * 2] * ptrIn[i * 2] + b * ptrIn[i * 2 + 1] * ptrIn[i * 2 + 1]) + ptrNoise[i];

    int ret;
    //double opts[LM_OPTS_SZ];
    double info[LM_INFO_SZ];

    // parameters
    std::vector<double> p(2, 0.0);
    p[0] = 1, p[1] = 1;

    // vector for mesurements
    std::vector<double> x(num, 0.0);

    // covariance matrix at solution
    cv::Mat cov(2, 2, CV_64FC1);
    // TODO: setup optimisation options with some good defaults.
    double optimOpts[5];

    optimOpts[0] = 1E-03;  // init mu
    // stop thresholds
    optimOpts[1] = 1e-5;   // ||J^T e||_inf
    optimOpts[2] = 1e-5;   // ||Dp||_2
    optimOpts[3] = 1e-1;   // ||e||_2
    // difference mode
    optimOpts[4] = LM_DIFF_DELTA;

    int maxIter = 300;

    Data data;
    data.ptrIn = ptrIn;
    data.ptrOut = ptrOut;
    data.count = 0;

    ret = dlevmar_dif(&compute, &(p[0]), &(x[0]), 2, num, maxIter, optimOpts, info, NULL, (double*)cov.data, &data);  // no jacobian
    //ret = dlevmar_der(&compute, &jacob, &(p[0]), &(x[0]), 2, num, maxIter, optimOpts, info, NULL, (double*)cov.data, &data);  // with jacobian
    // copy to source images (data.m_imgs)

    int kkka = 0;
    return 0;
}