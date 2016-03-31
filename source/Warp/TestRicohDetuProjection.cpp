#include "Reprojection.h"
#include "Timer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/gpumat.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
    cv::Size dstSize = cv::Size(2048, 1024);

    {
        cv::Size srcSize = cv::Size(1440, 1080);

        ReprojectParam param;
        param.LoadConfig("F:\\panovideo\\detu\\param.xml");
        param.SetPanoSize(dstSize);
        param.rotateCamera(0, 3.14 / 2, 0);

        std::vector<cv::Mat> dstMask;
        std::vector<cv::Mat> dstSrcMap;
        getReprojectMapsAndMasks2(param, srcSize, dstSrcMap, dstMask);

        cv::Mat origImage = cv::imread("F:\\panovideo\\detu\\aab.png");
        std::vector<cv::Mat> src;
        src.push_back(origImage);

        std::vector<cv::Mat> dst;
        reproject(src, dst, dstSrcMap);
        cv::imshow("dst", dst[0]);
        cv::waitKey(0);
    }

    return 0;

    cv::Mat dstLeft1, dstLeft2, dstRight1, dstRight2;
    cv::Mat maskLeft1, maskLeft2, maskRight1, maskRight2;
    cv::Mat mapLeft1, mapLeft2, mapRight1, mapRight2;
    {
        cv::Size srcSizeLeft = cv::Size(890, 890);
        cv::Size srcSizeRight = cv::Size(890, 890);
        cv::Rect srcRectLeft(25, 37, 890, 890);
        cv::Rect srcRectRight(45, 33, 890, 890);

        ReprojectParam paramLeft, paramRight;
        paramLeft.LoadConfig("F:\\panovideo\\ricoh\\1builtinleft.xml");
        paramLeft.SetPanoSize(dstSize);
        paramRight.LoadConfig("F:\\panovideo\\ricoh\\1builtinright.xml");
        paramRight.SetPanoSize(dstSize);

        std::vector<cv::Mat> dstMaskLeft, dstMaskRight;
        std::vector<cv::Mat> dstSrcMapLeft, dstSrcMapRight;
        getReprojectMapsAndMasks(paramLeft, srcSizeLeft, dstSrcMapLeft, dstMaskLeft);
        getReprojectMapsAndMasks(paramRight, srcSizeRight, dstSrcMapRight, dstMaskRight);

        cv::Mat origImage = cv::imread("F:\\panovideo\\ricoh\\image1.bmp");
        std::vector<cv::Mat> leftImage, rightImage;
        leftImage.push_back(origImage(cv::Rect(0, 0, 960, 1080))(srcRectLeft));
        rightImage.push_back(origImage(cv::Rect(960, 0, 960, 1080))(srcRectRight));

        std::vector<cv::Mat> dstLeft, dstRight;
        reproject(leftImage, dstLeft, dstSrcMapLeft);
        reproject(rightImage, dstRight, dstSrcMapRight);
        cv::imshow("dst left", dstLeft[0]);
        cv::imshow("dst right", dstRight[0]);
        cv::waitKey(0);

        mapLeft1 = dstSrcMapLeft[0];
        mapRight1 = dstSrcMapRight[0];
        maskLeft1 = dstMaskLeft[0];
        maskRight1 = dstMaskRight[0];
        dstLeft1 = dstLeft[0];
        dstRight1 = dstRight[0];
    }

    {
        cv::Size srcSizeLeft = cv::Size(960, 1080);
        cv::Size srcSizeRight = cv::Size(960, 1080);

        ReprojectParam paramLeft, paramRight;
        paramLeft.LoadConfig("F:\\panovideo\\ricoh\\2builtinleft.xml");
        paramLeft.SetPanoSize(dstSize);
        paramRight.LoadConfig("F:\\panovideo\\ricoh\\2builtinright.xml");
        paramRight.SetPanoSize(dstSize);

        std::vector<cv::Mat> dstMaskLeft, dstMaskRight;
        std::vector<cv::Mat> dstSrcMapLeft, dstSrcMapRight;
        getReprojectMapsAndMasks2(paramLeft, srcSizeLeft, dstSrcMapLeft, dstMaskLeft);
        getReprojectMapsAndMasks2(paramRight, srcSizeRight, dstSrcMapRight, dstMaskRight);

        cv::Mat origImage = cv::imread("F:\\panovideo\\ricoh\\image1.bmp");
        std::vector<cv::Mat> leftImage, rightImage;
        leftImage.push_back(origImage(cv::Rect(0, 0, 960, 1080)));
        rightImage.push_back(origImage(cv::Rect(960, 0, 960, 1080)));

        std::vector<cv::Mat> dstLeft, dstRight;
        reproject(leftImage, dstLeft, dstSrcMapLeft);
        reproject(rightImage, dstRight, dstSrcMapRight);
        cv::imshow("dst left", dstLeft[0]);
        cv::imshow("dst right", dstRight[0]);
        cv::waitKey(0);

        mapLeft2 = dstSrcMapLeft[0];
        mapRight2 = dstSrcMapRight[0];
        maskLeft2 = dstMaskLeft[0];
        maskRight2 = dstMaskRight[0];
        dstLeft2 = dstLeft[0];
        dstRight2 = dstRight[0];
    }

    cv::Mat leftMapDiff = mapLeft1 - mapLeft2, rightMapDiff = mapRight1 - mapRight2;
    cv::Scalar dm1 = cv::mean(leftMapDiff, maskLeft1), dm2 = cv::mean(rightMapDiff, maskRight1);
    printf("%f, %f, %f, %f\n", dm1[0], dm1[1], dm2[0], dm2[1]);

    cv::imshow("mask diff1", maskLeft1 ^ maskLeft2);
    cv::imshow("mask diff2", maskRight1 ^ maskRight2);

    cv::Mat dstLeftShadow1(dstSize.height, dstSize.width * 3, CV_8UC1, dstLeft1.data);
    cv::Mat dstRightShadow1(dstSize.height, dstSize.width * 3, CV_8UC1, dstRight1.data);
    cv::Mat dstLeftShadow2(dstSize.height, dstSize.width * 3, CV_8UC1, dstLeft2.data);
    cv::Mat dstRightShadow2(dstSize.height, dstSize.width * 3, CV_8UC1, dstRight2.data);
    cv::Mat result1, result2;
    cv::bitwise_xor(dstLeftShadow1, dstLeftShadow2, result1);
    cv::bitwise_xor(dstRightShadow1, dstRightShadow2, result2);
    cv::Mat result1Shadow(dstSize, CV_8UC3, result1.data);
    cv::Mat result2Shadow(dstSize, CV_8UC3, result2.data);
    cv::imshow("diff1", result1Shadow);
    cv::imshow("diff2", result2Shadow);
    cv::waitKey(0);
    printf("%d, %d\n", cv::countNonZero(result1), cv::countNonZero(result2));
    cv::imwrite("diff1.bmp", result1Shadow);
    cv::imwrite("diff2.bmp", result2Shadow);
}