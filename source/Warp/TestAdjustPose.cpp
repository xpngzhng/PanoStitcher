#include "AdjustPose.h"
#include "RotateImage.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include <iostream>
#include <utility>

const double PI = 3.1415926535898;

void drawGrid(cv::Mat image, int numSteps, double horiStep, double vertStep)
{
    int width = image.cols, height = image.rows;
    for (int i = 1; i < numSteps; i++)
    {
        cv::line(image, cv::Point(i * horiStep, 0), cv::Point(i * horiStep, height), cv::Scalar::all(128));
        cv::line(image, cv::Point(0, i * vertStep), cv::Point(width, i * vertStep), cv::Scalar::all(128));
    }
}

int width, height;
double horiStep, vertStep;
int step = 18;
cv::Mat image, dst, show;
cv::Point beg, end;
AdjustPose adjustPose;
void onMouse(int event, int x, int y, int flags, void*)
{
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        beg = cv::Point(x, y);
        adjustPose.buttonDown(beg);
    }
    else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
    {
        end = cv::Point(x, y);
        cv::Matx33d rot = adjustPose.mouseMove(end);
        mapNearestNeighborParallel(image, dst, rot);
        dst.copyTo(show);
        drawGrid(show, step, horiStep, vertStep);
        cv::line(show, beg, end, cv::Scalar(0, 0, 255));
        cv::imshow("image", show);
    }
    else if (event == CV_EVENT_LBUTTONUP)
    {
        cv::Vec3d ypr = adjustPose.buttonUp();
        printf("rotation: yaw = %f(%f), pitch = %f(%f), roll = %f(%f)\n",
            ypr[0], ypr[0] * 180 / PI, ypr[1], ypr[1] * 180 / PI, ypr[2], ypr[2] * 180 / PI);
        dst.copyTo(show);
        drawGrid(show, step, horiStep, vertStep);
        cv::imshow("image", show);
    }
}

int main()
{
    const char* controlWinName = "image";
    cv::namedWindow(controlWinName);
    cv::setMouseCallback(controlWinName, onMouse);
    cv::Mat orig = cv::imread("F:\\panoimage\\2\\1\\fps_tmp_pano.jpg");
    cv::resize(orig, image, cv::Size(), 0.1875, 0.1875); 
    width = image.cols;
    height = image.rows;
    horiStep = double(width) / step;
    vertStep = double(height) / step;
    dst = image.clone();
    show = image.clone();
    drawGrid(show, step, horiStep, vertStep);
    //totalRotation = cv::Matx33d::eye();
    cv::imshow("image", show);
    adjustPose.init(image.size());
    cv::waitKey(0); 
    return 0;
}