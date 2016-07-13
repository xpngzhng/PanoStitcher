#include "ZBlend.h"
#include "ZBlendAlgo.h"
#include "opencv2/imgproc.hpp"
#include <iostream>

void getLUT(std::vector<unsigned char>& lut, double k);
void isGradSmall(const cv::Mat& image, int thresh, cv::Mat& mask, cv::Mat& blurred, cv::Mat& grad16S);

void calcTintTransform(const cv::Mat& image, const cv::Mat& imageMask, const cv::Mat& base, const cv::Mat& baseMask,
    std::vector<std::vector<unsigned char> >& luts)
{
    CV_Assert(image.data && image.type() == CV_8UC3 &&
        base.data && base.type() == CV_8UC3 && imageMask.data && imageMask.type() == CV_8UC1 &&
        baseMask.data && baseMask.type() == CV_8UC1);
    int rows = image.rows, cols = image.cols;
    CV_Assert(imageMask.rows == rows && imageMask.cols == cols &&
        base.rows == rows && base.cols == cols && baseMask.rows == rows && baseMask.cols == cols);

    cv::Mat intersectMask = imageMask & baseMask;
    cv::Scalar baseMean = cv::mean(base, intersectMask);
    cv::Scalar imageMean = cv::mean(image, intersectMask);
    double baseRGRatio = baseMean[2] / baseMean[1], baseBGRatio = baseMean[0] / baseMean[1];
    double imageRGRatio = imageMean[2] / imageMean[1], imageBGRatio = imageMean[0] / imageMean[1];
    double rRatio = baseRGRatio / imageRGRatio, bRatio = baseBGRatio / imageBGRatio;
    double ratios[3] = { bRatio, 1, rRatio };
    printf("base r/g = %f, b/g = %f, image r/g = %f, b/g = %f\n",
        baseRGRatio, baseBGRatio, imageRGRatio, imageBGRatio);

    luts.resize(3);
    for (int i = 0; i < 3; i++)
    {
        luts[i].resize(256);
        getLUT(luts[i], ratios[i]);
        //for (int j = 0; j < 256; j++)
            //luts[i][j] = cv::saturate_cast<unsigned char>(j * ratios[i]);
    }
}

void tintCorrect(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    const std::vector<int>& correct, std::vector<std::vector<std::vector<unsigned char> > >& luts)
{
    CV_Assert(checkSize(images) && checkSize(masks) &&
        checkType(images, CV_8UC3) && checkType(masks, CV_8UC1));

    int numImages = images.size();
    CV_Assert(correct.size() == numImages);

    std::vector<cv::Mat> mainImages, mainMasks;
    cv::Mat mainMask = cv::Mat::zeros(images[0].size(), CV_8UC1);
    for (int i = 0; i < numImages; i++)
    {
        if (!correct[i])
        {
            mainImages.push_back(images[i]);
            mainMasks.push_back(masks[i]);
            mainMask |= masks[i];
        }
    }

    BlendConfig blendConfig;
    blendConfig.setSeamDistanceTransform();
    blendConfig.setBlendMultiBand();
    cv::Mat mainBlend;
    parallelBlend(blendConfig, mainImages, mainMasks, mainBlend);

    std::vector<unsigned char> identityLut(256);
    for (int i = 0; i < 256; i++)
        identityLut[i] = i;

    luts.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        if (correct[i])
        {
            calcTintTransform(images[i], masks[i], mainBlend, mainMask, luts[i]);
        }
        else
        {
            luts[i].resize(3);
            luts[i][0] = identityLut;
            luts[i][1] = identityLut;
            luts[i][2] = identityLut;
        }
    }
}

void scale(const cv::Mat& src, const cv::Mat& mask, double rRatio, double bRatio, cv::Mat& dst)
{
    CV_Assert(src.data && src.type() == CV_8UC3 &&
        mask.data && mask.type() == CV_8UC1 && mask.size() == src.size());
    int rows = src.rows, cols = src.cols;
    dst.create(rows, cols, CV_8UC3);
    std::vector<unsigned char> rLUT(256), bLUT(256);
    const unsigned char* rlut = rLUT.data(), * blut = bLUT.data();
    getLUT(rLUT, rRatio);
    getLUT(bLUT, bRatio);
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrSrc = src.ptr<unsigned char>(i);
        const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        unsigned char* ptrDst = dst.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (*ptrMask)
            {
                ptrDst[0] = blut[ptrSrc[0]];
                ptrDst[1] = ptrSrc[1];
                ptrDst[2] = rlut[ptrSrc[2]];
            }
            else
            {
                ptrDst[0] = 0;
                ptrDst[1] = 0;
                ptrDst[2] = 0;
            }
            ptrSrc += 3;
            ptrMask++;
            ptrDst += 3;
        }
    }
}

void getTintTransformsMeanApproxMimicSiftPanoPaper(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    std::vector<double>& rgRatioGains, std::vector<double>& bgRatioGains)
{
    int numImages = images.size();

    cv::Mat_<double> N(numImages, numImages), rgI(numImages, numImages), bgI(numImages, numImages);
    for (int i = 0; i < numImages; i++)
    {
        for (int j = 0; j < numImages; j++)
        {
            if (i == j)
            {
                N(i, i) = cv::countNonZero(masks[i]);
                cv::Scalar meanVal = cv::mean(images[i], masks[i]);
                rgI(i, i) = meanVal[2] / meanVal[1];
                bgI(i, i) = meanVal[0] / meanVal[1];
            }
            else
            {
                cv::Mat intersect = masks[i] & masks[j];
                int numNonZero = cv::countNonZero(intersect);
                N(i, j) = numNonZero;
                if (numNonZero)
                {
                    cv::Scalar meanVal = cv::mean(images[i], intersect);
                    rgI(i, j) = meanVal[2] / meanVal[1];
                    bgI(i, j) = meanVal[0] / meanVal[1];
                }
                else
                {
                    rgI(i, j) = 0;
                    bgI(i, j) = 0;
                }
            }
        }
    }
    //std::cout << N << "\n" << rgI << "\n" << bgI << "\n";

    double invSigmaNSqr = 1;
    double invSigmaGSqr = 0.05;

    bool success;

    cv::Mat_<double> A(numImages, numImages);
    cv::Mat_<double> b(numImages, 1);
    cv::Mat_<double> gains(numImages, 1);

    A.setTo(0);
    b.setTo(0);
    for (int i = 0; i < numImages; ++i)
    {
        for (int j = 0; j < numImages; ++j)
        {
            A(i, i) += N[i][j] * (rgI[i][j] * rgI[i][j] * invSigmaNSqr + invSigmaGSqr);
            A(j, j) += N[i][j] * (rgI[j][i] * rgI[j][i] * invSigmaNSqr);
            A(i, j) -= 2 * N[i][j] * (rgI[i][j] * rgI[j][i] * invSigmaNSqr);
            b(i) += N[i][j] * invSigmaGSqr;
        }
    }

    //std::cout << A << "\n" << b << "\n";
    success = cv::solve(A, b, gains);
    std::cout << gains << "\n";
    if (!success)
        gains.setTo(1);

    rgRatioGains.resize(numImages);
    for (int i = 0; i < numImages; i++)
        rgRatioGains[i] = gains(i);

    A.setTo(0);
    b.setTo(0);
    for (int i = 0; i < numImages; ++i)
    {
        for (int j = 0; j < numImages; ++j)
        {
            A(i, i) += N[i][j] * (bgI[i][j] * bgI[i][j] * invSigmaNSqr + invSigmaGSqr);
            A(j, j) += N[i][j] * (bgI[j][i] * bgI[j][i] * invSigmaNSqr);
            A(i, j) -= 2 * N[i][j] * (bgI[i][j] * bgI[j][i] * invSigmaNSqr);
            b(i) += N[i][j] * invSigmaGSqr;
        }
    }

    //std::cout << A << "\n" << b << "\n";
    success = cv::solve(A, b, gains);
    std::cout << gains << "\n";
    if (!success)
        gains.setTo(1);

    bgRatioGains.resize(numImages);
    for (int i = 0; i < numImages; i++)
        bgRatioGains[i] = gains(i);
}

void getTintTransformsPairWiseMimicSiftPanoPaper(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    std::vector<double>& rgRatioGains, std::vector<double>& bgRatioGains)
{
    int numImages = images.size();

    double invSigmaNSqr = 1;
    double invSigmaGSqr = 0.1;

    cv::Mat_<double> rgA(numImages, numImages), bgA(numImages, numImages);
    cv::Mat_<double> rgB(numImages, 1), bgB(numImages, 1);
    cv::Mat_<double> rgGains(numImages, 1), bgGains(numImages, 1);

    rgA.setTo(0), bgA.setTo(0);
    rgB.setTo(0), bgB.setTo(0);
    cv::Mat intersect;
    int rows = images[0].rows, cols = images[0].cols;
    for (int i = 0; i < numImages; i++)
    {
        for (int j = 0; j < numImages; j++)
        {
            if (i == j)
                continue;

            intersect = masks[i] & masks[j];
            if (cv::countNonZero(intersect) == 0)
                continue;

            for (int u = 0; u < rows; u++)
            {
                const unsigned char* ptri = images[i].ptr<unsigned char>(u);
                const unsigned char* ptrj = images[j].ptr<unsigned char>(u);
                const unsigned char* ptrm = intersect.ptr<unsigned char>(u);
                for (int v = 0; v < cols; v++)
                {
                    if (*(ptrm++))
                    {
                        double bi = *(ptri++), gi = *(ptri++), ri = *(ptri++);
                        double bj = *(ptrj++), gj = *(ptrj++), rj = *(ptrj++);
                        if (gi > 15 && gi < 240 && gj > 15 && gj < 240)
                        //if (gi > 1 && gj > 1)
                        {
                            double bgi = bi / gi, rgi = ri / gi;
                            double bgj = bj / gj, rgj = rj / gj;

                            if (bgi > 0.8 && bgi < 1.25 && bgj > 0.8 && bgj < 1.25)
                            {
                                bgA(i, i) += bgi * bgi * invSigmaNSqr + invSigmaGSqr;
                                bgA(j, j) += bgj * bgj * invSigmaNSqr;
                                bgA(i, j) -= 2 * bgi * bgj * invSigmaNSqr;
                                bgB(i) += invSigmaGSqr;
                            }

                            if (rgi > 0.8 && rgi < 1.25 && rgj > 0.8 && rgj < 1.25)
                            {
                                rgA(i, i) += rgi * rgi * invSigmaNSqr + invSigmaGSqr;
                                rgA(j, j) += rgj * rgj * invSigmaNSqr;
                                rgA(i, j) -= 2 * rgi * rgj * invSigmaNSqr;
                                rgB(i) += invSigmaGSqr;
                            }
                        }
                    }
                    else
                    {
                        ptri += 3;
                        ptrj += 3;
                    }
                }
            }
        }
    }

    bool success;

    //std::cout << A << "\n" << b << "\n";
    success = cv::solve(rgA, rgB, rgGains);
    //std::cout << rgGains << "\n";
    if (!success)
        rgGains.setTo(1);

    rgRatioGains.resize(numImages);
    for (int i = 0; i < numImages; i++)
        rgRatioGains[i] = rgGains(i);

    //std::cout << A << "\n" << b << "\n";
    success = cv::solve(bgA, bgB, bgGains);
    //std::cout << bgGains << "\n";
    if (!success)
        bgGains.setTo(1);

    bgRatioGains.resize(numImages);
    for (int i = 0; i < numImages; i++)
        bgRatioGains[i] = bgGains(i);
}

void tintAdjust(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results)
{
    int numImages = images.size();

    std::vector<cv::Mat> grayImages(numImages);
    for (int i = 0; i < numImages; i++)
        cv::cvtColor(images[i], grayImages[i], CV_BGR2GRAY);

    //std::vector<cv::Mat> outMasks(numImages);
    //cv::Mat gradSmallMask, blur, grad;
    //for (int i = 0; i < numImages; i++)
    //{
    //    isGradSmall(grayImages[i], 2, gradSmallMask, blur, grad);
    //    outMasks[i] = masks[i] & gradSmallMask;
    //}

    std::vector<double> rgGains, bgGains;
    getTintTransformsPairWiseMimicSiftPanoPaper(images, masks/*outMasks*/, rgGains, bgGains);

    std::vector<double> diff(numImages);
    for (int i = 0; i < numImages; i++)
    {
        cv::Scalar mean = cv::mean(images[i], masks[i]);
        diff[i] = abs(1 - mean[0] / mean[1]) + abs(1 - mean[2] / mean[1]);
    }

    int anchorIndex = 0;
    int minDiff = diff[0];
    for (int i = 0; i < numImages; i++)
    {
        if (minDiff > diff[i])
        {
            minDiff = diff[i];
            anchorIndex = i;
        }
    }

    //printf("anchor = %d\n", anchorIndex);

    double rgScale = 1.0 / rgGains[anchorIndex], bgScale = 1.0 / bgGains[anchorIndex];

    for (int i = 0; i < numImages; i++)
    {
        rgGains[i] *= rgScale;
        bgGains[i] *= bgScale;
    }
    results.resize(numImages);
    for (int i = 0; i < numImages; i++)
        scale(images[i], masks[i], rgGains[i], bgGains[i], results[i]);
}