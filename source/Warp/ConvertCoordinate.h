#pragma once

#include "MathConstant.h"
#include "Rotation.h"
#include "opencv2/core.hpp"
#include <vector>

// Image horizontal axis is x, vertical axis is y.
// Origin is at the topleft corner.
// The positive directions of x and y axes are right and down.
// x corresponds to phi angle, y corresponds to theta angle.
// Image horizontal length is width, vertical length is height.
// 
// For hugin style,
// theta increases from -pi to pi as x increases from 0 to width,
// phi decreases from pi to zero as y increases from 0 to height.
// Then theta and phi are used to get sphere coordinate in the sphere 3D coordinate system. 
// X axis directs left, Y directs down, Z directs to yourself.
// Let (Xs, Ys, Zs) be the sphere point 
// corresponding to (x, y) represented by (theta, phi) on the image, then
// Xs = sin(theta) * sin(phi), Ys = cos(theta), Zs = sin(theta) * cos(phi).
// theta is the angle from Y axis positive direction to current point.
// Draw a plane alpha which is through current point and parallel to ZOX plane, then 
// phi is the angle on alpha from the direction parallel to Z axis positive direction to current point, 
// phi rotates around Y axis, direction of phi follows right hand side rule.
//
// For pt style
// theta decreases from pi to -pi as x increases from 0 to width,
// phi decreases from pi / 2 to -pi / 2 as y increases from 0 to height.
// Then theta and phi are used to get sphere coordinate in the sphere 3D coordinate system. 
// X axis directs yourself, Y directs right, Z directs up.
// Let (Xs, Ys, Zs) be the sphere point 
// corresponding to (x, y) represented by (theta, phi) on the image, then
// Xs = cos(theta) * cos(phi), Ys = cos(theta) * sin(phi), Zs = sin(theta)
// theta is the angle from XOY plane to current point.
// theta is positive above the XOY plane and is negative below.
// Draw a plane alpha which is through current point and parallel to XOY plane,
// phi is the angle on alpha from the direction parallel to X axis positive direction to current point,
// phi rotates around Z axis, direction of phi follows right hand side rule.
//
// For my style
// theta decreases from pi to -pi as x increases from 0 to width,
// phi decreases from pi / 2 to -pi / 2 as y increases from 0 to height.
// Then theta and phi are used to get sphere coordinate in the sphere 3D coordinate system. 
// X axis directs right, Y directs up, Z directs yourself.
// Let (Xs, Ys, Zs) be the sphere point 
// corresponding to (x, y) represented by (theta, phi) on the image, then
// Xs = cos(theta) * sin(phi), Ys = sin(theta), Zs = cos(theta) * cos(phi)
// theta is the angle from ZOX plane to current point.
// theta is positive above the ZOX plane and is negative below.
// Draw a plane alpha which is through current point and parallel to ZOX plane,
// phi is the angle on alpha from the direction parallel to Z axis positive direction to current point,
// phi rotates around Y axis, direction of phi follows right hand side rule.

#define HUGIN_REMAP 1
#define PT_STYLE 2
#define MY_STYLE 3
#define STYLE 1

#if STYLE == MY_STYLE
inline cv::Point3d equirectToSphere(const cv::Point& pt, double halfWidth, double halfHeight)
{
    double theta = (halfHeight - (pt.y + 0.5)) / halfHeight * HALF_PI;
    double phi = (halfWidth - (pt.x + 0.5)) / halfWidth * PI;
    return cv::Point3d(cos(theta) * sin(phi), sin(theta), cos(theta) * cos(phi));
}
#elif STYLE == PT_STYLE
inline cv::Point3d equirectToSphere(const cv::Point& pt, double halfWidth, double halfHeight)
{
    double theta = (halfHeight - (pt.y + 0.5)) / halfHeight * HALF_PI;
    double phi = (halfWidth - (pt.x + 0.5)) / halfWidth * PI;
    return cv::Point3d(cos(theta) * cos(phi), cos(theta) * sin(phi), sin(theta));
}
#else
// hugin remap style used
inline cv::Point3d equirectToSphere(const cv::Point& pt, double halfWidth, double halfHeight)
{
    double theta = PI - (pt.y + 0.5) / halfHeight * HALF_PI;
    double phi = ((pt.x + 0.5) - halfWidth) / halfWidth * PI;
    return cv::Point3d(sin(theta) * sin(phi), cos(theta), sin(theta) * cos(phi));
}
#endif

#if STYLE == MY_STYLE
inline cv::Point2d sphereToEquirect(const cv::Point3d& pt, double halfWidth, double halfHeight)
{
    double theta = asin(pt.y) / HALF_PI;
    double phi = atan2(pt.x, pt.z) / PI;
    //double theta = asin(pt.z) / HALF_PI;
    //double phi = atan2(pt.y, pt.x) / PI;
    return cv::Point2d(halfWidth - phi * halfWidth - 0.5, halfHeight - theta * halfHeight - 0.5);
}
#elif STYLE == PT_STYLE
inline cv::Point2d sphereToEquirect(const cv::Point3d& pt, double halfWidth, double halfHeight)
{
    double theta = asin(pt.z) / HALF_PI;
    double phi = atan2(pt.y, pt.x) / PI;
    return cv::Point2d(halfWidth - phi * halfWidth - 0.5, halfHeight - theta * halfHeight - 0.5);
}
#else
// hugin remap style used
inline cv::Point2d sphereToEquirect(const cv::Point3d& pt, double halfWidth, double halfHeight)
{
    double theta = acos(pt.y) / HALF_PI;
    double phi = atan2(pt.x, pt.z) / PI;
    return cv::Point2d(halfWidth + phi * halfWidth - 0.5, halfHeight * (2 - theta) - 0.5);
}
#endif

#if STYLE == MY_STYLE
inline void equirectToSphere(const std::vector<cv::Point2d>& src, double width, double height, std::vector<cv::Point3d>& dst)
{
    int size = src.size();
    dst.resize(size);
    double halfWidth = width / 2;
    double halfHeight = height / 2;
    for (int i = 0; i < size; i++)
    {
        double theta = (halfHeight - (src[i].y + 0.5)) / halfHeight * HALF_PI;
        double phi = (halfWidth - (src[i].x + 0.5)) / halfWidth * PI;
        dst[i].x = cos(theta) * sin(phi);
        dst[i].y = sin(theta);
        dst[i].z = cos(theta) * cos(phi);
    }
}
#elif STYLE == PT_STYLE
inline void equirectToSphere(const std::vector<cv::Point2d>& src, double width, double height, std::vector<cv::Point3d>& dst)
{
    int size = src.size();
    dst.resize(size);
    double halfWidth = width / 2;
    double halfHeight = height / 2;
    for (int i = 0; i < size; i++)
    {
        double theta = (halfHeight - (src[i].y + 0.5)) / halfHeight * HALF_PI;
        double phi = (halfWidth - (src[i].x + 0.5)) / halfWidth * PI;
        dst[i].x = cos(theta) * cos(phi);
        dst[i].y = cos(theta) * sin(phi);
        dst[i].z = sin(theta);
    }
}
#else
// hugin remap style used
inline void equirectToSphere(const std::vector<cv::Point2d>& src, double width, double height, std::vector<cv::Point3d>& dst)
{
    int size = src.size();
    dst.resize(size);
    double halfWidth = width / 2;
    double halfHeight = height / 2;
    for (int i = 0; i < size; i++)
    {
        double theta = PI - (src[i].y + 0.5) / halfHeight * HALF_PI;
        double phi = ((src[i].x + 0.5) - halfWidth) / halfWidth * PI;
        dst[i].x = sin(theta) * sin(phi);
        dst[i].y = cos(theta);
        dst[i].z = sin(theta) * cos(phi);
    }
}
#endif

#if STYLE == MY_STYLE
inline cv::Point2d findRotateEquiRectangularSrc(const cv::Point& dst, double halfWidth, double halfHeight, const cv::Matx33d& invRot)
{
    double theta = (halfHeight - (dst.y + 0.5)) / halfHeight * HALF_PI;
    double phi = (halfWidth - (dst.x + 0.5)) / halfWidth * PI;
    cv::Point3d pt(cos(theta) * sin(phi), sin(theta), cos(theta) * cos(phi));
    pt = invRot * pt;
    theta = asin(pt.y) / HALF_PI;
    phi = atan2(pt.x, pt.z) / PI;
    return cv::Point2d(halfWidth - phi * halfWidth - 0.5, halfHeight - theta * halfHeight - 0.5);
}
#elif STYLE == PT_STYLE
inline cv::Point2d findRotateEquiRectangularSrc(const cv::Point& dst, double halfWidth, double halfHeight, const cv::Matx33d& invRot)
{
    double theta = (halfHeight - (dst.y + 0.5)) / halfHeight * HALF_PI;
    double phi = (halfWidth - (dst.x + 0.5)) / halfWidth * PI;
    cv::Point3d pt(cos(theta) * cos(phi), cos(theta) * sin(phi), sin(theta));
    pt = invRot * pt;
    theta = asin(pt.z) / HALF_PI;
    phi = atan2(pt.y, pt.x) / PI;
    return cv::Point2d(halfWidth - phi * halfWidth - 0.5, halfHeight - theta * halfHeight - 0.5);
}
#else
// hugin remap style used
inline cv::Point2d findRotateEquiRectangularSrc(const cv::Point& dst, double halfWidth, double halfHeight, const cv::Matx33d& invRot)
{
    double theta = PI - (dst.y + 0.5) / halfHeight * HALF_PI;
    double phi = ((dst.x + 0.5) - halfWidth) / halfWidth * PI;
    cv::Point3d pt(sin(theta) * sin(phi), cos(theta), sin(theta) * cos(phi));
    pt = invRot * pt;
    theta = acos(pt.y) / HALF_PI;
    phi = atan2(pt.x, pt.z) / PI;
    return cv::Point2d(halfWidth + phi * halfWidth - 0.5, halfHeight * (2 - theta) - 0.5);
}
#endif

inline cv::Point2d findTransEquiRectangularSrc(const cv::Point& dst, double halfWidth, double halfHeight, const cv::Point3d& negTrans)
{
    double theta = PI - (dst.y + 0.5) / halfHeight * HALF_PI;
    double phi = ((dst.x + 0.5) - halfWidth) / halfWidth * PI;
    cv::Point3d pt(sin(theta) * sin(phi), cos(theta), sin(theta) * cos(phi));
    double R = 1.01 + norm(negTrans);
    cv::Point3d trans = -negTrans;
    double a, i2a, b, c, d, e, p, q, x1, x2, y1, y2, z1, z2;
    a = R * R;
    i2a = 1.0 / (2 * a);

    b = 2 * R * ((trans.y * pt.y + trans.z * pt.z) * pt.x - trans.x * (1 - pt.x * pt.x));
    d = trans.y * pt.x - trans.x * pt.y;
    e = trans.z * pt.x - trans.x * pt.z;
    c = d * d + e * e - pt.x * pt.x * a;
    p = -b * i2a;
    q = sqrt(b * b - 4 * a * c) * i2a;
    x1 = p + q;
    x2 = p - q;

    b = 2 * R * ((trans.x * pt.x + trans.z * pt.z) * pt.y - trans.y * (1 - pt.y * pt.y));
    d = trans.x * pt.y - trans.y * pt.x;
    e = trans.z * pt.y - trans.y * pt.z;
    c = d * d + e * e - pt.y * pt.y * a;
    p = -b * i2a;
    q = sqrt(b * b - 4 * a * c) * i2a;
    y1 = p + q;
    y2 = p - q;

    b = 2 * R * ((trans.y * pt.y + trans.x * pt.x) * pt.z - trans.z * (1 - pt.z * pt.z));
    d = trans.y * pt.z - trans.z * pt.y;
    e = trans.x * pt.z - trans.z * pt.x;
    c = d * d + e * e - pt.z * pt.z * a;
    p = -b * i2a;
    q = sqrt(b * b - 4 * a * c) * i2a;
    z1 = p + q;
    z2 = p - q;

    //pt.x = inRange(x1, -1, 1) ? x1 : x2;
    //pt.y = inRange(y1, -1, 1) ? y1 : y2;
    //pt.z = inRange(z1, -1, 1) ? z1 : z2;
    pt.x = (x1 * R - trans.x) * pt.x > 0 ? x1 : x2;
    pt.y = (y1 * R - trans.y) * pt.y > 0 ? y1 : y2;
    pt.z = (z1 * R - trans.z) * pt.z > 0 ? z1 : z2;
    theta = acos(pt.y) / HALF_PI;
    phi = atan2(pt.x, pt.z) / PI;
    return cv::Point2d(halfWidth + phi * halfWidth - 0.5, halfHeight * (2 - theta) - 0.5);
}