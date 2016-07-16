#include "Rotation.h"
#include "Stabilize.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include <iostream>

void filterMatches(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
    std::vector<cv::DMatch>& matches1To2)
{
    if (matches1To2.empty())
        return;

    int size = matches1To2.size();
    std::vector<char> keep(size, 0);
    for (int i = 0; i < size; i++)
    {
        cv::Point pt1 = keypoints1[matches1To2[i].queryIdx].pt;
        cv::Point pt2 = keypoints2[matches1To2[i].trainIdx].pt;
        cv::Point diff = pt1 - pt2;
        if (sqrt(double(diff.dot(diff))) < 20)
            keep[i] = 1;
    }
    int numKeep = 0;
    for (int i = 0; i < size; i++)
    {
        if (keep[i])
            matches1To2[numKeep++] = matches1To2[i];
    }
    matches1To2.resize(numKeep);
    //printf("mathed keypoints, before = %d, after = %d\n", size, numKeep);
}

void extractMatchPoints(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch>& matches1To2, std::vector<cv::Point2d>& points1, std::vector<cv::Point2d>& points2)
{
    points1.clear();
    points2.clear();
    if (matches1To2.empty())
        return;
    int size = matches1To2.size();
    points1.resize(size);
    points2.resize(size);
    for (int i = 0; i < size; i++)
    {
        points1[i] = keypoints1[matches1To2[i].queryIdx].pt;
        points2[i] = keypoints2[matches1To2[i].trainIdx].pt;
    }
}

void drawDirection(const std::vector<cv::Point2d>& points1, const std::vector<cv::Point2d>& points2, cv::Mat& image)
{
    int size = points1.size();
    for (int i = 0; i < size; i++)
    {
        cv::circle(image, points1[i], 2, cv::Scalar(255), 1);
        cv::line(image, points1[i], points2[i], cv::Scalar(255), 1);
    }
}

void solveRotation(const std::vector<cv::Point3d>& src, const std::vector<cv::Point3d>& dst, cv::Matx33d& rot, 
    double& yaw, double& pitch, double& roll)
{
    CV_Assert(src.size() == dst.size() && src.size() > 3);
    int size = src.size();
    cv::Mat y(size * 3, 1, CV_64FC1);
    cv::Mat A(size * 3, 9, CV_64FC1);
    for (int i = 0; i < size; i++)
    {
        double* row1 = A.ptr<double>(i * 3);
        double* row2 = A.ptr<double>(i * 3 + 1);
        double* row3 = A.ptr<double>(i * 3 + 2);
        memset(row1, 0, 9 * sizeof(double));
        memset(row2, 0, 9 * sizeof(double));
        memset(row3, 0, 9 * sizeof(double));
        row1[0] = row2[3] = row3[6] = src[i].x;
        row1[1] = row2[4] = row3[7] = src[i].y;
        row1[2] = row2[5] = row3[8] = src[i].z;
        double* py = &y.at<double>(i * 3);
        py[0] = dst[i].x;
        py[1] = dst[i].y;
        py[2] = dst[i].z;
    }
    cv::Mat x;
    cv::solve(A, y, x, cv::DECOMP_QR);

    // NOTICE HERE!!! rot is actually NOT A ROTATION MATRIX!!!
    memcpy(rot.val, x.data, 9 * sizeof(double));
    getRotationPT(rot, yaw, pitch, roll);
    //printf("yaw = %f, pitch = %f, roll = %f\n", yaw, pitch, roll);

    //cv::Matx33d prod = rot * rot.inv();
    //double* pp = prod.val;
    //printf("[%8.4f, %8.4f, %8.4f\n %8.4f, %8.4f, %8.4f\n %8.4f, %8.4f, %8.4f]\n", 
    //    pp[0], pp[1], pp[2], pp[3], pp[4], pp[5], pp[6], pp[7], pp[8]);
    //printf("\n");

    //cv::Mat srcMat(size, 3, CV_64FC1, (void*)src.data());
    //cv::Mat dstMat(size, 3, CV_64FC1, (void*)dst.data());
    //cv::Mat rotSrcToDst, inliers;
    //cv::estimateAffine3D(srcMat, dstMat, rotSrcToDst, inliers);
    //std::cout << rotSrcToDst << std::endl;
}

inline cv::Point3d avg(const std::vector<cv::Point3d>& p)
{
    if (p.empty()) return cv::Point3d(0, 0, 0);
    int size = p.size();
    cv::Point3d sum(0, 0, 0);
    for (int i = 0; i < size; i++)
        sum += p[i];
    return sum * (1.0 / size);
}

inline void subtract(const std::vector<cv::Point3d>& src, const cv::Point3d& p, std::vector<cv::Point3d>& dst)
{
    dst.clear();
    if (src.empty()) return;
    int size = src.size();
    dst.resize(size);
    for (int i = 0; i < size; i++)
        dst[i] = src[i] - p;
}

inline void mulUVt(const cv::Point3d& u, const cv::Point3d& v, cv::Matx33d& mat)
{
    mat(0, 0) = u.x * v.x, mat(0, 1) = u.x * v.y, mat(0, 2) = u.x * v.z;
    mat(1, 0) = u.y * v.x, mat(1, 1) = u.y * v.y, mat(1, 2) = u.y * v.z;
    mat(2, 0) = u.z * v.x, mat(2, 1) = u.z * v.y, mat(2, 2) = u.z * v.z;
}

inline void accumulate(const std::vector<cv::Point3d>& u, const std::vector<cv::Point3d>& v, cv::Matx33d& mat)
{
    int size = u.size();
    cv::Matx33d sum = cv::Matx33d::zeros(), prod;
    for (int i = 0; i < size; i++)
    {
        mulUVt(u[i], v[i], prod);
        sum += prod;
    }
    mat = sum;
}

void getRotation(const std::vector<cv::Point3d>& src, const std::vector<cv::Point3d>& dst, cv::Matx33d& rot, 
    double& yaw, double& pitch, double& roll)
{
    CV_Assert(src.size() == dst.size() && src.size() > 3);
    int size = src.size();

    cv::Point3d srcAvg = avg(src), dstAvg = avg(dst);
    std::vector<cv::Point3d> srcDiff, dstDiff;
    subtract(src, srcAvg, srcDiff);
    subtract(dst, dstAvg, dstDiff);

    cv::Matx33d H;
    accumulate(srcDiff, dstDiff, H);

    cv::Matx33d U, VT;
    cv::Matx31d L;
    cv::SVD::compute(H, L, U, VT);
    rot = VT.t() * U.t();

    getRotationPT(rot, yaw, pitch, roll);
    //printf("yaw = %f, pitch = %f, roll = %f\n", yaw, pitch, roll);
    //cv::Point3d T = dstAvg - rot * srcAvg;
    //printf("t = %f, %f, %f\n", T.x, T.y, T.z);
}

void refineRotation(const std::vector<cv::Point3d>& src, const std::vector<cv::Point3d>& dst, cv::Matx33d& rot, 
    double& yaw, double& pitch, double& roll)
{
    CV_Assert(src.size() == dst.size() && src.size() > 3);
    int size = src.size();

    std::vector<double> dist(size);
    double accumDist = 0, minDist = DBL_MAX, maxDist = DBL_MIN;
    for (int i = 0; i < size; i++)
    {
        double currDist = cv::norm(dst[i] - rot * src[i]);
        accumDist += currDist;
        minDist = std::min(currDist, minDist);
        maxDist = std::max(currDist, maxDist);
        dist[i] = currDist;
    }

    double avgDist = accumDist / size;
    double accumSqrDiff = 0;
    for (int i = 0; i < size; i++)
    {
        double diff = avgDist - dist[i];
        accumSqrDiff += diff * diff;
    }
    double stdDevDiff = sqrt(accumSqrDiff / size);
    double thresh = avgDist + 2.5 * stdDevDiff;

    std::vector<cv::Point3d> filterSrc, filterDst;
    filterSrc.reserve(size);
    filterDst.reserve(size);
    for (int i = 0; i < size; i++)
    {
        if (dist[i] < thresh)
        {
            filterSrc.push_back(src[i]);
            filterDst.push_back(dst[i]);
        }
    }
    getRotation(filterSrc, filterDst, rot, yaw, pitch, roll);
}

void getRigidTransform(const std::vector<cv::Point3d>& src, const std::vector<cv::Point3d>& dst, 
    cv::Matx33d& rot, cv::Point3d& tsl)
{
    CV_Assert(src.size() == dst.size() && src.size() >= 3);
    int size = src.size();

    cv::Point3d srcAvg = avg(src), dstAvg = avg(dst);
    std::vector<cv::Point3d> srcDiff, dstDiff;
    subtract(src, srcAvg, srcDiff);
    subtract(dst, dstAvg, dstDiff);

    cv::Matx33d H;
    accumulate(srcDiff, dstDiff, H);

    cv::Matx33d U, VT;
    cv::Matx31d L;
    cv::SVD::compute(H, L, U, VT);
    rot = VT.t() * U.t();

    double yaw, pitch, roll;
    getRotationPT(rot, yaw, pitch, roll);
    //printf("yaw = %f, pitch = %f, roll = %f\n", yaw, pitch, roll);
    tsl = dstAvg - rot * srcAvg;
    //printf("t = %f, %f, %f\n", tsl.x, tsl.y, tsl.z);
}

void randomPermute(int size, int count, std::vector<int>& arr)
{
    arr.clear();
    CV_Assert(size > 0 && count > 0 && count <= size);    
    cv::RNG rng(cv::getTickCount());
    std::vector<int> total(size);
    for (int i = 0; i < size; i++)
        total[i] = i;
    for (int i = 0; i < count; i++)
    {
        int index = rng.uniform(i, size);
        std::swap(total[i], total[index]);
    }
    arr.resize(count);
    for (int i = 0; i < count; i++)
        arr[i] = total[i];
}

void select(const std::vector<cv::Point3d>& total, const std::vector<int>& index, std::vector<cv::Point3d>& subset)
{
    subset.clear();
    int size = index.size();
    subset.resize(size);
    for (int i = 0; i < size; i++)
        subset[i] = total[index[i]];
}

void select(const std::vector<cv::Point3d>& total, const std::vector<unsigned char>& mask, std::vector<cv::Point3d>& subset)
{
    subset.clear();
    CV_Assert(total.size() == mask.size());
    int size = total.size();
    subset.reserve(size);
    for (int i = 0; i < size; i++)
    {
        if (mask[i])
            subset.push_back(total[i]);
    }
}

int getInliers(const std::vector<cv::Point3d> src, const std::vector<cv::Point3d>& dst, 
    const cv::Matx33d& R, const cv::Point3d& T, double error, std::vector<unsigned char>& mask)
{
    mask.clear();
    if (src.empty())
        return 0;

    int size = src.size();
    CV_Assert(size == dst.size());
    mask.resize(size, 0);
    int count = 0;
    double minError = DBL_MAX, maxError = 0;
    for (int i = 0; i < size; i++)
    {
        double currError = cv::norm(dst[i] - R * src[i] - T);
        minError = MIN(currError, minError);
        maxError = MAX(currError, maxError);
        if (currError < error)
        {
            mask[i] = 255;
            count++;
        }
    }
    //printf("in getInliers, minError = %f, maxError = %f\n", minError, maxError);
    return count;
}

int updateNumIters(double p, double ep, int modelPoints, int maxIters)
{
    CV_Assert(modelPoints > 0);

    p = MAX(p, 0.);
    p = MIN(p, 1.);
    ep = MAX(ep, 0.);
    ep = MIN(ep, 1.);

    // avoid inf's & nan's
    double num = MAX(1. - p, DBL_MIN);
    double denom = 1. - pow(1. - ep, modelPoints);
    if (denom < DBL_MIN)
        return 0;

    num = log(num);
    denom = log(denom);

    return denom >= 0 || -num >= maxIters*(-denom) ?
        maxIters : cvRound(num / denom);
}

int getRigidTransformRANSAC(const std::vector<cv::Point3d>& src, const std::vector<cv::Point3d>& dst, 
    cv::Matx33d& R, cv::Point3d& T, std::vector<unsigned char>& mask)
{
    CV_Assert(src.size() == dst.size() && src.size() > 3);
    int size = src.size();
    //printf("RANSAC:\n");
    //printf("size = %d\n", size);

    int numModelPoints = 3;
    int numMaxIters = 1000;
    int numIters = numMaxIters;
    double confidence = 0.995;
    double projError = 0.03;
    std::vector<unsigned char> currMask, bestMask;
    std::vector<int> currIndex;
    std::vector<cv::Point3d> currSrc, currDst;
    int numMaxInliers = 0;
    for (int i = 0; i < numIters; i++)
    {
        //printf("iter = %d\n", i);
        randomPermute(size, numModelPoints, currIndex);
        select(src, currIndex, currSrc);
        select(dst, currIndex, currDst);
        cv::Matx33d currR;
        cv::Point3d currT;
        getRigidTransform(currSrc, currDst, currR, currT);
        int currNumInliers = getInliers(src, dst, currR, currT, projError, currMask);
        //printf("currNumInliers = %d\n", currNumInliers);
        if (currNumInliers > MAX(numMaxInliers, numModelPoints - 1))
        {
            numIters = updateNumIters(confidence, double(size - currNumInliers) / size, numModelPoints, numMaxIters);
            //int currNumIters = updateNumIters(confidence, double(size - currNumInliers) / size, numModelPoints, numMaxIters);
            //numIters = MIN(numMaxIters, i > 2 * currNumIters ? currNumIters : currNumIters + i); 
            //printf("numIters changed to %d\n", numIters);
            numMaxInliers = currNumInliers;
            bestMask = currMask;
        }
    }
    mask = bestMask;
    select(src, bestMask, currSrc);
    select(dst, bestMask, currDst);
    getRigidTransform(currSrc, currDst, R, T);
    return numMaxInliers;
}

void checkPrecision(const std::vector<cv::Point3d>& src, const std::vector<cv::Point3d>& dst, const cv::Matx33d& rot)
{
    CV_Assert(src.size() == dst.size());
    if (src.empty())
        return;
    double accumDist = 0, minDist = DBL_MAX, maxDist = DBL_MIN;
    int size = src.size();
    for (int i = 0; i < size; i++)
    {
        double currDist = cv::norm(dst[i] - rot * src[i]);
        accumDist += currDist;
        minDist = std::min(currDist, minDist);
        maxDist = std::max(currDist, maxDist);
    }
    printf("rot error: avg = %f, min = %f, max = %f\n", accumDist / size, minDist, maxDist);
}

void calcHist(const std::vector<double>& arr, double low, double high, int numBins, std::vector<int>& hist)
{
    hist.clear();
    if (arr.empty())
        return;
     int size = arr.size();
    double irange = 1.0 / (high - low);
    hist.resize(numBins, 0);
    for (int i = 0; i < size; i++)
    {
        hist[int((arr[i] - low) * irange * numBins)]++;
    }
}

void drawHist(const std::vector<int>& hist, cv::Mat& image)
{
    int size = hist.size();
    int maxVal = 0;
    for (int i = 0; i < size; i++)
    {
        maxVal = std::max(maxVal, hist[i]);
    }
    //printf("max = %d\n", maxVal);
    int height = 400, step = 10, width = size * step;
    image.create(height, width, CV_8UC3);
    image.setTo(0);
    for (int i = 0; i < size; i++)
    {
        int top = height - double(hist[i]) / maxVal * height;
        cv::rectangle(image, cv::Point(i * step, top), cv::Point((i + 1) * step, height), cv::Scalar(255, 255, 255), -1);
    }
}

void checkMatchedPointsDist(const std::vector<cv::Point3d>& src, const std::vector<cv::Point3d>& dst)
{
    CV_Assert(src.size() == dst.size() && src.size() > 3);
    int size = src.size();

    std::vector<double> dist(size);
    double accumDist = 0, minDist = DBL_MAX, maxDist = DBL_MIN;
    for (int i = 0; i < size; i++)
    {
        double currDist = cv::norm(dst[i] - src[i]);
        accumDist += currDist;
        minDist = std::min(currDist, minDist);
        maxDist = std::max(currDist, maxDist);
        dist[i] = currDist;
    }

    double avgDist = accumDist / size;
    double accumSqrDiff = 0;
    for (int i = 0; i < size; i++)
    {
        double diff = avgDist - dist[i];
        accumSqrDiff += diff * diff;
    }
    double stdDevDiff = sqrt(accumSqrDiff / size);

    printf("src dst dist: avg = %f, min = %f, max = %f, stdDev = %f\n",
        avgDist, minDist, maxDist, stdDevDiff);
}

void checkPrecision(const std::vector<cv::Point2d>& srcEquirect, const std::vector<cv::Point2d>& dstEquirect, cv::Size& imageSize, 
    const std::vector<cv::Point3d>& srcSphere, const std::vector<cv::Point3d>& dstSphere, const cv::Matx33d& rot)
{
    int size = srcEquirect.size();
    if (size == 0 || dstEquirect.size() != size ||
        srcSphere.size() != size || dstSphere.size() != size)
        return;

    std::vector<double> dist(size);
    double accumDist = 0, minDist = DBL_MAX, maxDist = DBL_MIN;
    for (int i = 0; i < size; i++)
    {
        double currDist = cv::norm(dstSphere[i] - rot * srcSphere[i]);
        accumDist += currDist;
        minDist = std::min(currDist, minDist);
        maxDist = std::max(currDist, maxDist);
        dist[i] = currDist;
    }

    double avgDist = accumDist / size;
    double accumSqrDiff = 0;
    for (int i = 0; i < size; i++)
    {
        double diff = avgDist - dist[i];
        accumSqrDiff += diff * diff;
    }
    double stdDev = sqrt(accumSqrDiff / size);

    double low1 = avgDist -     stdDev, high1 = avgDist +     stdDev;
    double low2 = avgDist - 2 * stdDev, high2 = avgDist + 2 * stdDev;
    double low3 = avgDist - 3 * stdDev, high3 = avgDist + 3 * stdDev;
    cv::Mat check = cv::Mat::zeros(imageSize, CV_8UC3);
    int numInRange1 = 0, numInRange2 = 0, numInRange3 = 0;
    for (int i = 0; i < size; i++)
    {
        bool inRange1 = inRange(dist[i], low1, high1);
        bool inRange2 = inRange(dist[i], low2, high2);
        bool inRange3 = inRange(dist[i], low3, high3);
        if (inRange1) numInRange1++;
        if (inRange2) numInRange2++;
        if (inRange3) numInRange3++;
        if (inRange1)
        {            
            cv::line(check, srcEquirect[i], dstEquirect[i], cv::Scalar(255, 0, 0));
            cv::circle(check, srcEquirect[i], 2, cv::Scalar(255, 0, 0));
        }
        else if (inRange2) 
        {
            cv::line(check, srcEquirect[i], dstEquirect[i], cv::Scalar(0, 255, 0));
            cv::circle(check, srcEquirect[i], 2, cv::Scalar(0, 255, 0));
        }
        else if (inRange3) 
        {
            cv::line(check, srcEquirect[i], dstEquirect[i], cv::Scalar(0, 0, 255));
            cv::circle(check, srcEquirect[i], 2, cv::Scalar(0, 0, 255));
        }
        else
        {
            cv::line(check, srcEquirect[i], dstEquirect[i], cv::Scalar(255, 255, 255));
            cv::circle(check, srcEquirect[i], 2, cv::Scalar(255, 255, 255));
        }
    }
    printf("only ROTATION used\n");
    printf("dist: avg = %f, min = %f, max = %f, stdDev = %f\n", avgDist, minDist, maxDist, stdDev);
    printf("num = %d, in range1 = %d(%f), in range2 = %d(%f), in range3 = %d(%f)\n",
        size, numInRange1, double(numInRange1) / size,
              numInRange2, double(numInRange2) / size,
              numInRange3, double(numInRange3) / size);
    cv::imshow("rotation check", check);

    std::vector<int> hist;
    cv::Mat histImage;
    calcHist(dist, minDist, maxDist + 0.001, 40, hist);
    drawHist(hist, histImage);
    cv::imshow("rotation hist", histImage);
}

void checkPrecision(const std::vector<cv::Point2d>& srcEquirect, const std::vector<cv::Point2d>& dstEquirect, cv::Size& imageSize, 
    const std::vector<cv::Point3d>& srcSphere, const std::vector<cv::Point3d>& dstSphere, const cv::Matx33d& rot, const cv::Point3d& trans)
{
    int size = srcEquirect.size();
    if (size == 0 || dstEquirect.size() != size ||
        srcSphere.size() != size || dstSphere.size() != size)
        return;

    std::vector<double> dist(size);
    double accumDist = 0, minDist = DBL_MAX, maxDist = DBL_MIN;
    for (int i = 0; i < size; i++)
    {
        double currDist = cv::norm(dstSphere[i] - rot * srcSphere[i] - trans);
        accumDist += currDist;
        minDist = std::min(currDist, minDist);
        maxDist = std::max(currDist, maxDist);
        dist[i] = currDist;
    }

    double avgDist = accumDist / size;
    double accumSqrDiff = 0;
    for (int i = 0; i < size; i++)
    {
        double diff = avgDist - dist[i];
        accumSqrDiff += diff * diff;
    }
    double stdDev = sqrt(accumSqrDiff / size);

    double low1 = avgDist -     stdDev, high1 = avgDist +     stdDev;
    double low2 = avgDist - 2 * stdDev, high2 = avgDist + 2 * stdDev;
    double low3 = avgDist - 3 * stdDev, high3 = avgDist + 3 * stdDev;
    cv::Mat check = cv::Mat::zeros(imageSize, CV_8UC3);
    int numInRange1 = 0, numInRange2 = 0, numInRange3 = 0;
    for (int i = 0; i < size; i++)
    {
        bool inRange1 = inRange(dist[i], low1, high1);
        bool inRange2 = inRange(dist[i], low2, high2);
        bool inRange3 = inRange(dist[i], low3, high3);
        if (inRange1) numInRange1++;
        if (inRange2) numInRange2++;
        if (inRange3) numInRange3++;
        if (inRange1)
        {            
            cv::line(check, srcEquirect[i], dstEquirect[i], cv::Scalar(255, 0, 0));
            cv::circle(check, srcEquirect[i], 2, cv::Scalar(255, 0, 0));
        }
        else if (inRange2) 
        {
            cv::line(check, srcEquirect[i], dstEquirect[i], cv::Scalar(0, 255, 0));
            cv::circle(check, srcEquirect[i], 2, cv::Scalar(0, 255, 0));
        }
        else if (inRange3) 
        {
            cv::line(check, srcEquirect[i], dstEquirect[i], cv::Scalar(0, 0, 255));
            cv::circle(check, srcEquirect[i], 2, cv::Scalar(0, 0, 255));
        }
        else
        {
            cv::line(check, srcEquirect[i], dstEquirect[i], cv::Scalar(255, 255, 255));
            cv::circle(check, srcEquirect[i], 2, cv::Scalar(255, 255, 255));
        }
    }
    printf("ROTATION and TRANSLATION used\n");
    printf("dist: avg = %f, min = %f, max = %f, stdDev = %f\n", avgDist, minDist, maxDist, stdDev);
    printf("num = %d, in range1 = %d(%f), in range2 = %d(%f), in range3 = %d(%f)\n",
        size, numInRange1, double(numInRange1) / size,
              numInRange2, double(numInRange2) / size,
              numInRange3, double(numInRange3) / size);
    cv::imshow("rotation translation check", check);

    std::vector<int> hist;
    cv::Mat histImage;
    calcHist(dist, minDist, maxDist + 0.001, 40, hist);
    drawHist(hist, histImage);
    cv::imshow("rotation translation hist", histImage);
}

#if FAST_ANGLE_CALC
void smooth(const std::vector<cv::Vec3d>& src, int radius, std::vector<cv::Vec3d>& dst)
{
    dst.clear();
    if (src.empty())
        return;
    int size = src.size();
    dst.resize(size);
    for (int i = 0; i < size; i++)
    {
        int beg = std::max(0, i - radius);
        int end = std::min(i + radius, size - 1);
        cv::Vec3d sum(0, 0, 0);
        for (int j = beg; j <= end; j++)
            sum += src[j];
        dst[i] = sum *= (1.0 / (end + 1 - beg));
    }
}
#else
void smooth(const std::vector<cv::Vec3d>& src, int radius, std::vector<cv::Vec3d>& dst)
{
    dst.clear();
    if (src.empty())
        return;
    int size = src.size();
    dst.resize(size);
    for (int i = 0; i < size; i++)
    {
        int beg = std::max(0, i - radius);
        int end = std::min(i + radius, size - 1);
        cv::Matx33d prod = cv::Matx33d::eye();
        cv::Matx33d curr;
        for (int j = beg; j <= end; j++)
        {
            setRotationPT(curr, src[j][0], src[j][1], src[j][2]);
            prod = curr * prod;
        }
        double scale = 1.0 / (end + 1 - beg);
        double yaw, pitch, roll;
        getRotationPT(prod, yaw, pitch, roll);
        dst[i][0] = yaw * scale;
        dst[i][1] = pitch * scale;
        dst[i][2] = roll * scale;
    }
}
#endif

void accumulate(const std::vector<cv::Vec3d>& src, std::vector<cv::Vec3d>& dst)
{
    dst.clear();
    if (src.empty())
        return;

    int size = src.size();
    cv::Vec3d accum(0, 0, 0);
    dst.resize(size);
    for (int i = 0; i < size; i++)
    {
        accum += src[i];
        dst[i] = accum;
    }
}

void draw(const std::vector<cv::Vec3d>& src, const cv::Scalar color[3], cv::Mat& image)
{
    if (src.empty())
        return;

    int size = src.size();
    double minVal = std::min(std::min(src[0][0], src[0][1]), src[0][2]);
    double maxVal = std::max(std::max(src[0][0], src[0][1]), src[0][2]);
    for (int i = 1; i < size; i++)
    {
        minVal = std::min(minVal, std::min(std::min(src[i][0], src[i][1]), src[i][2]));
        maxVal = std::max(maxVal, std::max(std::max(src[i][0], src[i][1]), src[i][2]));
    }
    double absMinVal = std::abs(minVal);
    double absMaxVal = std::abs(maxVal);
    double rawRadius = std::max(absMinVal, absMaxVal);
    int pad = 8;
    int width = size + 2 * pad;
    int rawHalfHeight = 100;
    int height = rawHalfHeight * 2 + pad * 2;
    image.create(height, width, CV_8UC3);
    image.setTo(0);
    double scale = rawHalfHeight / rawRadius;
    for (int i = 1; i < size; i++)
    {
        int begx = i - 1 + pad, endx = begx + 1;
        int begy, endy;
        for (int j = 0; j < 3; j++)
        {
            begy = pad + rawHalfHeight - src[i - 1][j] * scale;
            endy = pad + rawHalfHeight - src[i][j] * scale;
            cv::line(image, cv::Point(begx, begy), cv::Point(endx, endy), color[j]);
        }
    }
}