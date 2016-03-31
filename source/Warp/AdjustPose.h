#pragma once

#include "opencv2/core.hpp"

class AdjustPose
{
public:
    AdjustPose();
    void init(cv::Size size);
    void buttonDown(cv::Point pt);
    cv::Matx33d mouseMove(cv::Point pt);
    cv::Vec3d buttonUp();
private:
    cv::Point beg;
    cv::Rect rect;
    cv::Matx33d currRotation, totalRotation;
    double halfWidth, halfHeight;
    bool allow;
};
