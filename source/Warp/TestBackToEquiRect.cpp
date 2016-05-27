#include "ConvertCoordinate.h"
#include "RotateImage.h"
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
cv::Size dstSize(720, 360);
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

int main()
{
    const char* controlWinName = "image";
    cv::namedWindow(controlWinName);
    cv::setMouseCallback(controlWinName, onMouse);
    image = cv::imread("F:\\panoimage\\2\\1\\1 - 6 small.jpg");
    accumX = 0;
    accumY = 0;
    mapNearestNeighbor(image, show, dstSize, hfov, accumX * scale, accumY * scale, true);
    cv::imshow("image", show);
    cv::waitKey(0);
    return 0;
}