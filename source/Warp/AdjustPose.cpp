#include "AdjustPose.h"
#include "Rotation.h"
#include "ConvertCoordinate.h"

AdjustPose::AdjustPose()
    : beg(0, 0), currRotation(cv::Matx33d::eye()), totalRotation(cv::Matx33d::eye()), allow(false) {}

void AdjustPose::init(cv::Size size)
{
    if (size.width <= 0 || size.height <= 0)
        return;

    beg = cv::Point(0, 0);
    rect = cv::Rect(0, 0, size.width, size.height);
    currRotation = cv::Matx33d::eye();
    totalRotation = cv::Matx33d::eye();
    halfWidth = size.width * 0.5;
    halfHeight = size .height * 0.5;
    allow = true;
}

void AdjustPose::buttonDown(cv::Point pt)
{
    if (!allow)
        return;

    if (!rect.contains(pt))
        return;

    beg = pt;
}

cv::Matx33d AdjustPose::mouseMove(cv::Point pt)
{
    if (!allow)
        return cv::Matx33d::eye();

    if (!rect.contains(pt) || abs(beg.x - pt.y) + abs(beg.y - pt.y) < 1)
        return currRotation;

    /*
    cv::Point3d begSphere = equirectToSphere(beg, halfWidth, halfHeight);
    cv::Point3d endSphere = equirectToSphere(pt, halfWidth, halfHeight);
    cv::Matx33d rot;
    setRotationThroughPointPair(rot, begSphere, endSphere);
    currRotation = rot * totalRotation;
    */
    cv::Point3d begSphere = equirectToSphere(cv::Point(pt.x, beg.y), halfWidth, halfHeight);
    cv::Point3d endSphere = equirectToSphere(pt, halfWidth, halfHeight);
    cv::Matx33d rot1, rot2;
    setRotationThroughPointPair(rot1, begSphere, endSphere);
    double yaw = (pt.x - beg.x) / halfWidth * PI;
    setRotationRM(rot2, yaw, 0, 0);
    currRotation = rot2 * rot1 * totalRotation;
    return currRotation;
}

cv::Vec3d AdjustPose::buttonUp()
{
    if (!allow)
        return cv::Vec3d(0, 0, 0);

    totalRotation = currRotation;
    cv::Vec3d ret;
    getRotationRM(totalRotation, ret[0], ret[1], ret[2]);
    return ret;
}