#include "ConvertCoordinate.h"
#include "RotateImage.h"
#include "ZReproject.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

const double radOverDegree = 3.1415926535898 / 180;
int main1()
{
    int srcWidth = 720, srcHeight = 360;
    int dstWidth = 720, dstHeight = 720;
    double hfov = 60.0 * radOverDegree, horiOffset = 0 * radOverDegree, vertOffset = 0 * radOverDegree;
    FishEyeBackToEquiRect fishEye(srcWidth, srcHeight, dstWidth, dstHeight, hfov, horiOffset, vertOffset);
    RectLinearBackToEquiRect rectLinear(srcWidth, srcHeight, dstWidth, dstHeight, hfov, horiOffset, vertOffset);
    cv::Mat mask(srcHeight, srcWidth, CV_8UC1);

    mask.setTo(0);
    for (int i = 0; i < dstHeight; i++)
    {
        for (int j = 0; j < dstWidth; j++)
        {
            cv::Point srcPt = fishEye(j, i);
            int x = cvFloor(srcPt.x), y = cvFloor(srcPt.y);
            if (x >= 0 && x < srcWidth && y >= 0 && y < srcHeight)
                mask.at<unsigned char>(y, x) = 255;
        }
    }
    cv::imshow("fisheye", mask);

    mask.setTo(0);
    for (int i = 0; i < dstHeight; i++)
    {
        for (int j = 0; j < dstWidth; j++)
        {
            cv::Point srcPt = rectLinear(j, i);
            int x = cvFloor(srcPt.x), y = cvFloor(srcPt.y);
            if (x >= 0 && x < srcWidth && y >= 0 && y < srcHeight)
                mask.at<unsigned char>(y, x) = 255;
            //printf("(%d, %d)\n", x, y);
        }
    }
    cv::imshow("rectlinear", mask);

    cv::waitKey(0);

    return 0;
}

int main2()
{
    cv::Mat src = cv::imread("F:\\panoimage\\detuoffice\\blendmultiband.bmp");
    double hfov = 175.0 * radOverDegree, horiOffset = 120 * radOverDegree, vertOffset = 50 * radOverDegree;
    cv::Size dstSize(720, 720);
    //FishEyeBackToEquiRect fishEye(src.cols, src.rows, dstSize.width, dstSize.height, hfov, horiOffset, vertOffset);
    //RectLinearBackToEquiRect rectLinear(src.cols, src.rows, dstSize.width, dstSize.height, hfov, horiOffset, vertOffset);
    cv::Mat dstFishEye, dstRectLinear;
    mapNearestNeighbor(src, dstFishEye, dstSize, hfov, horiOffset, vertOffset, false);
    mapNearestNeighbor(src, dstRectLinear, dstSize, hfov, horiOffset, vertOffset, true);
    cv::imshow("fisheye", dstFishEye);
    cv::imshow("rectlinear", dstRectLinear);
    cv::waitKey(0);
    return 0;
}

cv::Mat image, show;
cv::Point beg, end;
int accumX, accumY;
double scale = 3.1415926 / 2000;
cv::Size dstSize(720, 720);
double hfov = 90 * 3.1415927 / 180;
void onMouse(int event, int x, int y, int flags, void*)
{
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        beg = cv::Point(x, y);
    }
    else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
    {
        end = cv::Point(x, y);
        accumX += (end.x - beg.x);
        accumY += (end.y - beg.y);
        mapNearestNeighbor(image, show, dstSize, hfov, accumX * scale, accumY * scale, true);
        cv::imshow("image", show);
    }
    else if (event == CV_EVENT_LBUTTONUP)
    {

        cv::imshow("image", show);
    }
}

static void alphaBlend(cv::Mat& image, const cv::Mat& logo)
{
    CV_Assert(image.data && (image.type() == CV_8UC3 || image.type() == CV_8UC4) &&
        logo.data && logo.type() == CV_8UC4 && image.size() == logo.size());

    int rows = image.rows, cols = image.cols, channels = image.channels();
    for (int i = 0; i < rows; i++)
    {
        unsigned char* ptrImage = image.ptr<unsigned char>(i);
        const unsigned char* ptrLogo = logo.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrLogo[3])
            {
                int val = ptrLogo[3];
                int comp = 255 - ptrLogo[3];
                ptrImage[0] = (comp * ptrImage[0] + val * ptrLogo[0] + 254) / 255;
                ptrImage[1] = (comp * ptrImage[1] + val * ptrLogo[1] + 254) / 255;
                ptrImage[2] = (comp * ptrImage[2] + val * ptrLogo[2] + 254) / 255;
            }
            ptrImage += channels;
            ptrLogo += 4;
        }
    }
}

int main()
{
    const char* controlWinName = "image";
    cv::namedWindow(controlWinName);
    cv::setMouseCallback(controlWinName, onMouse);
    image = cv::imread("F:\\panoimage\\2\\1\\1 - 6 small.jpg");

    cv::Mat logo = cv::imread("F:\\image\\Earth_global.png", -1);
    PhotoParam param;
    param.imageType = PhotoParam::ImageTypeFullFrameFishEye;
    param.hfov = 45;
    param.pitch = -90;
    param.cropWidth = logo.cols;
    param.cropHeight = logo.rows;
    cv::Mat map, mask;
    getReprojectMapAndMask(param, logo.size(), image.size(), map, mask);
    cv::imshow("mask", mask);
    cv::Mat logoReproj;
    reprojectParallel(logo, logoReproj, map);
    cv::imshow("logo reproj", logoReproj);
    alphaBlend(image, logoReproj);
    cv::imshow("logoed image", image);
    cv::waitKey(0);

    accumX = 0;
    accumY = 0;
    mapNearestNeighborParallel(image, show, dstSize, hfov, accumX * scale, accumY * scale, true);
    cv::imshow("image", show);
    cv::waitKey(0);
    return 0;
}