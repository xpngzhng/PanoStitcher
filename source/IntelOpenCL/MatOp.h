#pragma once

#include "IntelOpenCLMat.h"

void convert32SC4To8UC4(const iocl::UMat& src, iocl::UMat& dst);

void convert32FC4To8UC4(const iocl::UMat& src, iocl::UMat& dst);

void setZero(iocl::UMat& mat);

void setZero8UC4Mask8UC1(iocl::UMat& mat, const iocl::UMat& mask);

void setVal16SC1(iocl::UMat& mat, short val);

void setVal16SC1Mask8UC1(iocl::UMat& mat, short val, const iocl::UMat& mask);

void scaledSet16SC1Mask32SC1(iocl::UMat& image, short val, const iocl::UMat& mask);

void subtract16SC4(const iocl::UMat& a, const iocl::UMat& b, iocl::UMat& c);

void add32SC4(const iocl::UMat& a, const iocl::UMat& b, iocl::UMat& c);

void accumulate16SC1To32SC1(const iocl::UMat& src, iocl::UMat& dst);

void accumulate16SC4To32SC4(const iocl::UMat& src, const iocl::UMat& weight, iocl::UMat& dst);

void normalize32SC4(iocl::UMat& mat);

void normalize32SC4(iocl::UMat& mat, const iocl::UMat& weight);