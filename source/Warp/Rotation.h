#pragma once
#include "MathConstant.h"
#include "opencv2/core.hpp"
#include <cmath>

inline bool inRange(double val, double lowExc, double highExc)
{
    return (val > lowExc) && (val < highExc);
}

// NOTICE: No matter how you organize the direction of X, Y and Z axes,
// yaw, pitch and roll directions are the same.

// obtain a rotation matrix
// x directs right, z directs yourself, y directs up
// roll rotates around z, right hand side
// pitch rotates around x, left hand side
// yaw rotates around y, left hand side
// R = R-y * R-p * Rr
inline void setRotationMY(cv::Matx33d& mat, double yaw, double pitch, double roll)
{
    double cy = cos(yaw), sy = -sin(yaw);
    double cp = cos(pitch), sp = -sin(pitch);
    double cr = cos(roll), sr = sin(roll);
    mat(0, 0) = cy * cr + sy * sp * sr, mat(0, 1) = sy * sp * cr - cy * sr, mat(0, 2) = sy * cp;
    mat(1, 0) = cp * sr,                mat(1, 1) = cp * cr,                mat(1, 2) = -sp;
    mat(2, 0) = cy * sp * sr - sy * cr, mat(2, 1) = cy * sp * cr + sy * sr, mat(2, 2) = cy * cp;
}

inline void getRotationMY(const cv::Matx33d& mat, double& yaw, double& pitch, double& roll)
{
    if (inRange(mat(1, 2), -1, 1))
    {
        pitch = asin(-mat(1, 2));
        yaw = atan2(mat(0, 2), mat(2, 2));
        roll = atan2(mat(1, 0), mat(1, 1));
    }
    else
    {
        roll = 0;
        yaw = atan2(-mat(2, 0), mat(0, 0));
        pitch = mat(1, 2) > 0 ? MINUS_HALF_PI : HALF_PI;
    }
    yaw = -yaw;
    pitch = -pitch;
}

// obtain a rotation matrix
// x directs yourself, y directs right, z directs up
// roll rotates around x, right hand side
// pitch rotates around y, left hand side
// yaw rotates around z, left hand side
// R = R-y * R-p * Rr
inline void setRotationPT(cv::Matx33d& mat, double yaw, double pitch, double roll)
{
    double cy = cos(yaw), sy = -sin(yaw);
    double cp = cos(pitch), sp = -sin(pitch);
    double cr = cos(roll), sr = sin(roll);
    mat(0, 0) = cy * cp, mat(0, 1) = cy * sp * sr - sy * cr, mat(0, 2) = sy * sr + cy * sp * cr;
    mat(1, 0) = sy * cp, mat(1, 1) = cy * cr + sy * sp * sr, mat(1, 2) = sy * sp * cr - cy * sr;
    mat(2, 0) = -sp,     mat(2, 1) = cp * sr,                mat(2, 2) = cp * cr;
}

inline void getRotationPT(const cv::Matx33d& mat, double& yaw, double& pitch, double& roll)
{
    if (inRange(mat(2, 0), -1, 1))
    {
        pitch = asin(-mat(2, 0));
        roll = atan2(mat(2, 1), mat(2, 2));
        yaw = atan2(mat(1, 0), mat(0, 0));
    }
    else
    {
        roll = 0;
        yaw = atan2(-mat(0, 1), mat(1, 1));
        pitch = mat(2, 0) > 0 ? MINUS_HALF_PI : HALF_PI;
    }
    yaw = -yaw;
    pitch = -pitch;
}

// obtain a rotation matrix
// x directs left, z directs yourself, y directs down
// roll rotates around z, right hand side
// pitch rotates around x, right hand side
// yaw rotates around y, right hand side
// R = Ry * Rp * Rr
inline void setRotationRM(cv::Matx33d& mat, double yaw, double pitch, double roll)
{
    double cy = cos(yaw), sy = sin(yaw);
    double cp = cos(pitch), sp = sin(pitch);
    double cr = cos(roll), sr = sin(roll);
    mat(0, 0) = cy * cr + sy * sp * sr, mat(0, 1) = sy * sp * cr - cy * sr, mat(0, 2) = sy * cp;
    mat(1, 0) = cp * sr,                mat(1, 1) = cp * cr,                mat(1, 2) = -sp;
    mat(2, 0) = cy * sp * sr - sy * cr, mat(2, 1) = cy * sp * cr + sy * sr, mat(2, 2) = cy * cp;
}

inline void getRotationRM(const cv::Matx33d& mat, double& yaw, double& pitch, double& roll)
{
    if (inRange(mat(1, 2), -1, 1))
    {
        pitch = asin(-mat(1, 2));
        yaw = atan2(mat(0, 2), mat(2, 2));
        roll = atan2(mat(1, 0), mat(1, 1));
    }
    else
    {
        roll = 0;
        yaw = atan2(-mat(2, 0), mat(0, 0));
        pitch = mat(1, 2) > 0 ? MINUS_HALF_PI : HALF_PI;
    }
}

inline void setRotationAroundU(cv::Matx33d& mat, const cv::Vec3d& axis, double angle)
{
    double n = cv::norm(axis);
    if (n < 0.00001) 
    {
        mat = cv::Matx33d::eye();
        return;
    }
    cv::Point3d u = axis * (1.0 / n);
	double cs, ss, ux2, uy2, uz2, uxy, uxz, uyz;	
	cs = cos(angle);
	ss = sin(angle);
	ux2 = u.x * u.x;
	uy2 = u.y * u.y;
	uz2 = u.z * u.z;
	uxy = u.x * u.y;
	uxz = u.x * u.z;
	uyz = u.y * u.z;
    mat(0, 0) = ux2 + cs * (1 - ux2);
    mat(0, 1) = uxy * (1 - cs) - u.z * ss;
    mat(0, 2) = uxz * (1 - cs) + u.y * ss;
    mat(1, 0) = uxy * (1 - cs) + u.z * ss;
    mat(1, 1) = uy2 + cs * (1 - uy2);
    mat(1, 2) = uyz * (1 - cs) - u.x * ss;
    mat(2, 0) = uxz * (1 - cs) - u.y * ss;
    mat(2, 1) = uyz * (1 - cs) + u.x * ss;
    mat(2, 2) = uz2 + cs * (1 - uz2);
}

inline void getRotationAroundU(const cv::Matx33d& mat, cv::Vec3d& axis, double& angle)
{
    cv::Matx33d m = (mat - mat.t()) * 0.5;
    double s = m(0, 0) * m(0, 0) + m(0, 1) * m(0, 1) + m(0, 2) * m(0, 2) +
               m(1, 0) * m(1, 0) + m(1, 1) * m(1, 1) + m(1, 2) * m(1, 2) +
               m(2, 0) * m(2, 0) + m(2, 1) * m(2, 1) + m(2, 2) * m(2, 2);
    double sa = sqrt(s * 0.5);
    if (abs(sa) > DBL_EPSILON)
    {
        double isa = 1.0 / sa;
        angle = asin(sa);
        axis[0] = m(2, 1) * isa;
        axis[1] = m(0, 2) * isa;
        axis[2] = m(1, 0) * isa;        
        return;
    }
    angle = 0;
    axis = cv::Vec3d(1, 0, 0);
}

// point from and to should be on the sphere whose center is origin and whose radius is one
inline void setRotationThroughPointPair(cv::Matx33d& mat, const cv::Point3d& from, const cv::Point3d& to)
{
    double l1 = cv::norm(from);
    double l2 = cv::norm(to);
    double dot = from.dot(to);
    double alpha = acos(dot / (l1 * l2));
    cv::Point3d axis = from.cross(to);
    setRotationAroundU(mat, axis, alpha);
}