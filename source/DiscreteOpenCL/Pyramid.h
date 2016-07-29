#pragma once

#include "DiscreteOpenCLMat.h"

void pyramidDown8UC1To8UC1(const IOclMat& src, IOclMat& dst, cv::Size dstSize);

void pyramidDown8UC4To8UC4(const IOclMat& src, IOclMat& dst, cv::Size dstSize);

void pyramidDown8UC4To32SC4(const IOclMat& src, IOclMat& dst, cv::Size dstSize);

void pyramidDown32FC1(const IOclMat& src, IOclMat& dst, cv::Size dstSize);

void pyramidDown32FC4(const IOclMat& src, IOclMat& dst, cv::Size dstSize);

void pyramidDown16SC1To16SC1(const IOclMat& src, IOclMat& dst, cv::Size dstSize = cv::Size());

void pyramidDown16SC1To32SC1(const IOclMat& src, IOclMat& dst, cv::Size dstSize = cv::Size());

void pyramidDown16SC4To16SC4(const IOclMat& src, const IOclMat& scale, IOclMat& dst);

void pyramidUp8UC4To8UC4(const IOclMat& src, IOclMat& dst, cv::Size dstSize = cv::Size());

void pyramidUp16SC4To16SC4(const IOclMat& src, IOclMat& dst, cv::Size dstSize = cv::Size());

void pyramidUp32SC4To32SC4(const IOclMat& src, IOclMat& dst, cv::Size dstSize = cv::Size());
