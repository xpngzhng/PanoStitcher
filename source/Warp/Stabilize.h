#pragma once

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include <vector>

#define FAST_ANGLE_CALC 1

void filterMatches(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
    std::vector<cv::DMatch>& matches1To2);

void extractMatchPoints(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch>& matches1To2, std::vector<cv::Point2d>& points1, std::vector<cv::Point2d>& points2);

void drawDirection(const std::vector<cv::Point2d>& points1, const std::vector<cv::Point2d>& points2, cv::Mat& image);

void solveRotation(const std::vector<cv::Point3d>& src, const std::vector<cv::Point3d>& dst, cv::Matx33d& rot, 
    double& yaw, double& pitch, double& roll);

void getRotation(const std::vector<cv::Point3d>& src, const std::vector<cv::Point3d>& dst, cv::Matx33d& rot, 
    double& yaw, double& pitch, double& roll);

void refineRotation(const std::vector<cv::Point3d>& src, const std::vector<cv::Point3d>& dst, cv::Matx33d& rot, 
    double& yaw, double& pitch, double& roll);

void getRigidTransform(const std::vector<cv::Point3d>& src, const std::vector<cv::Point3d>& dst, 
    cv::Matx33d& R, cv::Point3d& T);

int getRigidTransformRANSAC(const std::vector<cv::Point3d>& src, const std::vector<cv::Point3d>& dst, 
    cv::Matx33d& R, cv::Point3d& T, std::vector<unsigned char>& mask);

void checkMatchedPointsDist(const std::vector<cv::Point3d>& src, const std::vector<cv::Point3d>& dst);

void checkPrecision(const std::vector<cv::Point2d>& srcEquirect, const std::vector<cv::Point2d>& dstEquirect, cv::Size& imageSize, 
    const std::vector<cv::Point3d>& srcSphere, const std::vector<cv::Point3d>& dstSphere, const cv::Matx33d& rot);

void checkPrecision(const std::vector<cv::Point2d>& srcEquirect, const std::vector<cv::Point2d>& dstEquirect, cv::Size& imageSize, 
    const std::vector<cv::Point3d>& srcSphere, const std::vector<cv::Point3d>& dstSphere, const cv::Matx33d& R, const cv::Point3d& T);

void smooth(const std::vector<cv::Vec3d>& src, int radius, std::vector<cv::Vec3d>& dst);

void accumulate(const std::vector<cv::Vec3d>& src, std::vector<cv::Vec3d>& dst);

void draw(const std::vector<cv::Vec3d>& src, const cv::Scalar color[3], cv::Mat& image);