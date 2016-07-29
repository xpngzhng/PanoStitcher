#pragma once

#include "DiscreteOpenCLMat.h"

void pyramidDown8UC1To8UC1(const docl::GpuMat& src, docl::GpuMat& dst, cv::Size dstSize = cv::Size());

void pyramidDown8UC4To8UC4(const docl::GpuMat& src, docl::GpuMat& dst, cv::Size dstSize = cv::Size());

void pyramidDown8UC4To32SC4(const docl::GpuMat& src, docl::GpuMat& dst, cv::Size dstSize = cv::Size());

void pyramidDown32FC1(const docl::GpuMat& src, docl::GpuMat& dst, cv::Size dstSize = cv::Size());

void pyramidDown32FC4(const docl::GpuMat& src, docl::GpuMat& dst, cv::Size dstSize = cv::Size());

void pyramidDown16SC1To16SC1(const docl::GpuMat& src, docl::GpuMat& dst, cv::Size dstSize = cv::Size());

void pyramidDown16SC1To32SC1(const docl::GpuMat& src, docl::GpuMat& dst, cv::Size dstSize = cv::Size());

void pyramidDown16SC4To16SC4(const docl::GpuMat& src, const docl::GpuMat& scale, docl::GpuMat& dst);

void pyramidUp8UC4To8UC4(const docl::GpuMat& src, docl::GpuMat& dst, cv::Size dstSize = cv::Size());

void pyramidUp16SC4To16SC4(const docl::GpuMat& src, docl::GpuMat& dst, cv::Size dstSize = cv::Size());

void pyramidUp32SC4To32SC4(const docl::GpuMat& src, docl::GpuMat& dst, cv::Size dstSize = cv::Size());
