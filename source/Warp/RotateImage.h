#pragma once

#include <opencv2/core/core.hpp>

void mapNearestNeighbor(const cv::Mat& src, cv::Mat& dst, const cv::Matx33d& rot);

void mapNearestNeighborParallel(const cv::Mat& src, cv::Mat& dst, const cv::Matx33d& rot);

void mapBilinear(const cv::Mat& src, cv::Mat& dst, const cv::Matx33d& rot);

void mapBilinearParallel(const cv::Mat& src, cv::Mat& dst, const cv::Matx33d& rot);