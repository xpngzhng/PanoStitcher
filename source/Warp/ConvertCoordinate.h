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
inline cv::Point3d equirectToSphere(const cv::Point2d& pt, double halfWidth, double halfHeight)
{
    double theta = PI - pt.y / halfHeight * HALF_PI;
    double phi = (pt.x - halfWidth) / halfWidth * PI;
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

#if STYLE == HUGIN_REMAP
namespace cts
{
enum
{
    RIGHT,
    LEFT,
    TOP,
    BOTTOM,
    FRONT, 
    BACK
};

const double P0[] = { -0.5, -0.5, -0.5 };
const double P1[] = { 0.5, -0.5, -0.5 };
const double P4[] = { -0.5, -0.5, 0.5 };
const double P5[] = { 0.5, -0.5, 0.5 };
const double P6[] = { -0.5, 0.5, 0.5 };

const double PX[] = { 1.0, 0.0, 0.0 };
const double PY[] = { 0.0, 1.0, 0.0 };
const double PZ[] = { 0.0, 0.0, 1.0 };
const double NX[] = { -1.0, 0.0, 0.0 };
const double NZ[] = { 0.0, 0.0, -1.0 };
}

inline cv::Point3d cubeToSphere(const cv::Point& pt, double cubeLength, int face)
{
    double x = (pt.x + 0.5) / cubeLength;
    double y = 1.0 - (pt.y + 0.5) / cubeLength;
    const double* p, * vx, * vy;
    switch (face)
    {
    case cts::RIGHT:   p = cts::P5; vx = cts::NZ; vy = cts::PY; break;
    case cts::LEFT:    p = cts::P0; vx = cts::PZ; vy = cts::PY; break;
    case cts::TOP:     p = cts::P6; vx = cts::PX; vy = cts::NZ; break;
    case cts::BOTTOM:  p = cts::P0; vx = cts::PX; vy = cts::PZ; break;
    case cts::FRONT:   p = cts::P4; vx = cts::PX; vy = cts::PY; break;
    case cts::BACK:    p = cts::P1; vx = cts::NX; vy = cts::PY; break;
    }
    double qx = p[0] + vx[0] * x + vy[0] * y;
    double qy = p[1] + vx[1] * x + vy[1] * y;
    double qz = p[2] + vx[2] * x + vy[2] * y;
    double scale = 1.0 / sqrt(qx * qx + qy * qy + qz * qz);
    return cv::Point3d(qx * scale, -qy * scale, qz * scale);
}

inline cv::Point3d cubeToSphere(const cv::Point2d& pt, double cubeLength)
{
    double width = cubeLength * 3, height = cubeLength * 2;
    double x = (pt.x) / width;
    double y = 1.0 - (pt.y) / height;
    int vface = (int)(y * 2);
    int hface = (int)(x * 3);
    x = x * 3 - hface;
    y = y * 2 - vface;
    int face = hface + (1 - vface) * 3;
    const double* p, *vx, *vy;
    switch (face)
    {
    case cts::RIGHT:   p = cts::P5; vx = cts::NZ; vy = cts::PY; break;
    case cts::LEFT:    p = cts::P0; vx = cts::PZ; vy = cts::PY; break;
    case cts::TOP:     p = cts::P6; vx = cts::PX; vy = cts::NZ; break;
    case cts::BOTTOM:  p = cts::P0; vx = cts::PX; vy = cts::PZ; break;
    case cts::FRONT:   p = cts::P4; vx = cts::PX; vy = cts::PY; break;
    case cts::BACK:    p = cts::P1; vx = cts::NX; vy = cts::PY; break;
    }
    double qx = p[0] + vx[0] * x + vy[0] * y;
    double qy = p[1] + vx[1] * x + vy[1] * y;
    double qz = p[2] + vx[2] * x + vy[2] * y;
    double scale = 1.0 / sqrt(qx * qx + qy * qy + qz * qz);
    return cv::Point3d(qx * scale, -qy * scale, qz * scale);
}

template<typename PointType>
inline void cubeToSphere(const std::vector<PointType>& src, double cubeLength, int face, std::vector<cv::Point3d>& dst)
{
    int size = src.size();
    dst.resize(size);
    for (int i = 0; i < size; i++)
    {
        double x = (src[i].x) / cubeLength;
        double y = 1.0 - (src[i].y) / cubeLength;
        const double* p, *vx, *vy;
        switch (face)
        {
        case cts::RIGHT:   p = cts::P5; vx = cts::NZ; vy = cts::PY; break;
        case cts::LEFT:    p = cts::P0; vx = cts::PZ; vy = cts::PY; break;
        case cts::TOP:     p = cts::P6; vx = cts::PX; vy = cts::NZ; break;
        case cts::BOTTOM:  p = cts::P0; vx = cts::PX; vy = cts::PZ; break;
        case cts::FRONT:   p = cts::P4; vx = cts::PX; vy = cts::PY; break;
        case cts::BACK:    p = cts::P1; vx = cts::NX; vy = cts::PY; break;
        }
        double qx = p[0] + vx[0] * x + vy[0] * y;
        double qy = p[1] + vx[1] * x + vy[1] * y;
        double qz = p[2] + vx[2] * x + vy[2] * y;
        double scale = 1.0 / sqrt(qx * qx + qy * qy + qz * qz);
        dst[i] = cv::Point3d(qx * scale, -qy * scale, qz * scale);
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

inline cv::Point2d findRotateEquiRectangularSrc(const cv::Point2d& dst, double halfWidth, double halfHeight, const cv::Matx33d& invRot)
{
    double theta = PI - (dst.y) / halfHeight * HALF_PI;
    double phi = ((dst.x) - halfWidth) / halfWidth * PI;
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

// Rotate pitch for the following two structs
// In the following two structs, we use traditional image coordinate system,
// in which x axis directs to left and y axis directs to bottom.
// In addition, in the following two structs, alphaX increases from -pi to pi
// and alphaY increases from -pi / 2 to pi / 2.
// To map equirectangular image to sphere, we here build a 3-D system following the 
// image coordinate system. In the 3-D system, x axis directs to left, y axis directs to bottom,
// and z axis directs front. To rotate a point on unit sphere around x axis,
// we call this function.
inline void rotatePitch(double srcAngleX, double srcAngleY, double& dstAngleX, double& dstAngleY, double angle)
{
    double x = cos(srcAngleY) * sin(srcAngleX);
    double y = sin(srcAngleY);
    double z = cos(srcAngleY) * cos(srcAngleX);
    double yy = cos(angle) * y - sin(angle) * z;
    double zz = sin(angle) * y + cos(angle) * z;
    dstAngleX = atan2(zz, x);
    dstAngleY = asin(yy);
}

// src equirect, dst fisheye
struct FishEyeBackToEquiRect
{
    FishEyeBackToEquiRect(int srcWidth, int srcHeight, int dstWidth, int dstHeight,
        double dstHFov, double srcDeltaAngleX, double srcDeltaAngleY)
    { 
        fullSrcWidth = srcWidth;
        fullSrcHeight = srcHeight;
        halfSrcWidth = srcWidth * 0.5;
        halfSrcHeight = srcHeight * 0.5;
        halfDstWidth = dstWidth * 0.5;
        halfDstHeight = dstHeight * 0.5;
        halfDstHFov = dstHFov * 0.5;
        srcR = halfSrcHeight;
        dstR = dstWidth / (2 * sin(halfDstHFov));
        dstRSqr = dstR * dstR;
        srcDeltaAX = -srcDeltaAngleX;
        srcDeltaAY = -srcDeltaAngleY;
        setRotationRM(invRot, 0, srcDeltaAngleY, 0);
        //invRot = invRot.t();
    }

    cv::Point2d operator()(double dstx, double dsty) const
    {
        double dstxx = dstx - halfDstWidth;
        double dstyy = dsty - halfDstHeight;
        double dstxxSqr = dstxx * dstxx;
        double dstyySqr = dstyy * dstyy;
        if (dstxxSqr + dstyySqr > dstRSqr)
            return cv::Point2d(-1, -1);
        double horiR = sqrt(dstRSqr - dstyySqr);
        double alphaX = asin(dstxx / horiR);
        double alphaY = asin(dstyy / dstR);
        rotatePitch(alphaX, alphaY, alphaX, alphaY, srcDeltaAY);
        double srcx = (alphaX) / PI * halfSrcWidth + halfSrcWidth;
        double srcy = (alphaY) / PI * fullSrcHeight + halfSrcHeight;
        //cv::Point2d dd = findRotateEquiRectangularSrc(cv::Point2d(srcx, srcy), halfSrcWidth, halfSrcHeight, invRot);
        //srcx = dd.x;
        //srcy = dd.y;
        srcx += srcDeltaAX / PI * halfSrcWidth;
        if (srcy < 0)
        {
            srcy = -srcy;
            srcx += halfSrcWidth;
        }
        if (srcy >= fullSrcHeight)
        {
            srcy = 2 * fullSrcHeight - srcy;
            srcx += halfSrcWidth;
        }
        while (srcx < 0)
            srcx += fullSrcWidth;
        while (srcx >= fullSrcWidth)
            srcx -= fullSrcWidth;
        return cv::Point2d(srcx, srcy);
    }

    double fullSrcWidth, fullSrcHeight;
    double halfSrcWidth, halfSrcHeight, halfDstWidth, halfDstHeight, halfDstHFov;
    double srcR, dstR, dstRSqr;
    double srcDeltaAX, srcDeltaAY;
    cv::Matx33d invRot;
};

// src equirect, dst rectlinear
struct RectLinearBackToEquiRect
{
    RectLinearBackToEquiRect(int srcWidth, int srcHeight, int dstWidth, int dstHeight,
        double dstHFov, double srcDeltaAngleX, double srcDeltaAngleY)
    {
        fullSrcWidth = srcWidth;
        fullSrcHeight = srcHeight;
        halfSrcWidth = srcWidth * 0.5;
        halfSrcHeight = srcHeight * 0.5;
        halfDstWidth = dstWidth * 0.5;
        halfDstHeight = dstHeight * 0.5;
        halfDstHFov = dstHFov * 0.5;
        srcR = halfSrcHeight;
        dstR = dstWidth / (2 * tan(halfDstHFov));
        dstRSqr = dstR * dstR;
        srcDeltaAX = -srcDeltaAngleX;
        srcDeltaAY = -srcDeltaAngleY;
        setRotationRM(invRot, 0, srcDeltaAngleY, 0);
        //invRot = invRot.t();
    }

    cv::Point2d operator()(double dstx, double dsty) const
    {
        double dstxx = dstx - halfDstWidth;
        double dstyy = dsty - halfDstHeight;
        double d = sqrt(dstxx * dstxx + dstyy * dstyy + dstRSqr);
        double alphaX = atan2(dstxx, dstR);
        double alphaY = asin(dstyy / d);
        rotatePitch(alphaX, alphaY, alphaX, alphaY, srcDeltaAY);
        double srcx = (alphaX) / PI * halfSrcWidth + halfSrcWidth;
        double srcy = (alphaY) / PI * fullSrcHeight + halfSrcHeight;
        //cv::Point2d dd = findRotateEquiRectangularSrc(cv::Point2d(srcx, srcy), halfSrcWidth, halfSrcHeight, invRot);
        //srcx = dd.x;
        //srcy = dd.y;
        srcx += srcDeltaAX / PI * halfSrcWidth;
        if (srcy < 0)
        {
            srcy = -srcy;
            srcx += halfSrcWidth;
        }
        if (srcy >= fullSrcHeight)
        {
            srcy = 2 * fullSrcHeight - srcy;
            srcx += halfSrcWidth;
        }
        while (srcx < 0)
            srcx += fullSrcWidth;
        while (srcx >= fullSrcWidth)
            srcx -= fullSrcWidth;        
        return cv::Point2d(srcx, srcy);
    }

    double fullSrcWidth, fullSrcHeight;
    double halfSrcWidth, halfSrcHeight, halfDstWidth, halfDstHeight, halfDstHFov;
    double srcR, dstR, dstRSqr;
    double srcDeltaAX, srcDeltaAY;
    cv::Matx33d invRot;
};