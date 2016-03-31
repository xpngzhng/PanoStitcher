#include "Rotation.h"
#include "ConvertCoordinate.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include <iostream>
#include <utility>

void mapBilinear(const cv::Mat& src, cv::Mat& dst, const cv::Matx33d& rot);
void mapNearestNeighbor(const cv::Mat& src, cv::Mat& dst, const cv::Matx33d& rot);
void mapBilinearParallel(const cv::Mat& src, cv::Mat& dst, const cv::Matx33d& rot);
void mapNearestNeighborParallel(const cv::Mat& src, cv::Mat& dst, const cv::Matx33d& rot);

int width, height;
double halfWidth, halfHeight;
double horiStep, vertStep;
int step = 18;
cv::Mat image, dst, show;
cv::Point beg, end;
bool action = false;
// initialization of totalRotation failed!!!
static cv::Matx33d totalRotation = cv::Matx33d::eye(), currRotation;
void onMouse(int event, int x, int y, int flags, void*)
{
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        action = true;
        beg = cv::Point(x, y);
        currRotation = cv::Matx33d::eye();
    }
    else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
    {
        end = cv::Point(x, y);
        if (abs(beg.x - end.x) + abs(beg.y - end.y) < 1)
            return;
        cv::Point3d begSphere = equirectToSphere(beg, halfWidth, halfHeight);
        cv::Point3d endSphere = equirectToSphere(end, halfWidth, halfHeight);
        cv::Matx33d rot;
        setRotationThroughPointPair(rot, begSphere, endSphere);
        currRotation = rot * totalRotation;
        mapNearestNeighborParallel(image, dst, currRotation);
        //mapBilinearParallel(image, dst, currRotation);
        //mapBilinear(image, dst, currRotation);
        dst.copyTo(show);
        for (int i = 1; i < step; i++)
        {
            cv::line(show, cv::Point(i * horiStep, 0), cv::Point(i * horiStep, height), cv::Scalar::all(128));
            cv::line(show, cv::Point(0, i * vertStep), cv::Point(width, i * vertStep), cv::Scalar::all(128));
        }
        cv::line(show, beg, end, cv::Scalar(0, 0, 255));
        cv::imshow("image", show);
    }
    else if (event == CV_EVENT_LBUTTONUP)
    {
        end = cv::Point(x, y);
        if (abs(beg.x - end.x) + abs(beg.y - end.y) < 1)
            return;
        totalRotation = currRotation;
        double yaw, pitch, roll;
        getRotationRM(totalRotation, yaw, pitch, roll);
        printf("rotation: yaw = %f(%f), pitch = %f(%f), roll = %f(%f)\n",
            yaw, yaw * 180 / PI, pitch, pitch * 180 / PI, roll, roll * 180 / PI);
        dst.copyTo(show);
        for (int i = 1; i < step; i++)
        {
            cv::line(show, cv::Point(i * horiStep, 0), cv::Point(i * horiStep, height), cv::Scalar::all(128));
            cv::line(show, cv::Point(0, i * vertStep), cv::Point(width, i * vertStep), cv::Scalar::all(128));
        }
        cv::imshow("image", show);
    }
}

int main()
{
    //{
    //    int width = 20, height = 10;
    //    double halfWidth = width * 0.5, halfHeight = 10 * 0.5;
    //    for (int i = 0; i < height; i++)
    //    {
    //        for (int j = 0; j < width; j++)
    //        {
    //            cv::Point3d pt = equirectToSphere(cv::Point(j, i), halfWidth, halfHeight);
    //            printf("(%5.2f, %5.2f, %5.2f) ", pt.x, pt.y, pt.z);
    //        }
    //        printf("\n");
    //    }
    //}

    const char* controlWinName = "image";
    cv::namedWindow(controlWinName);
    cv::setMouseCallback(controlWinName, onMouse);
    //image = cv::imread("F:\\panoimage\\2\\1\\panoup.bmp");
    cv::Mat orig = cv::imread("F:\\panoimage\\2\\1\\fps_tmp_pano.jpg");
    cv::resize(orig, image, cv::Size(), 0.25, 0.25); 
    width = image.cols;
    height = image.rows;
    halfWidth = width * 0.5;
    halfHeight = height * 0.5;
    horiStep = double(width) / step;
    vertStep = double(height) / step;
    show = image.clone();
    for (int i = 1; i < step; i++)
    {
        cv::line(show, cv::Point(i * horiStep, 0), cv::Point(i * horiStep, height), cv::Scalar::all(128));
        cv::line(show, cv::Point(0, i * vertStep), cv::Point(width, i * vertStep), cv::Scalar::all(128));
    }
    //totalRotation = cv::Matx33d::eye();
    cv::imshow("image", show);
    cv::waitKey(0); 
    return 0;
}