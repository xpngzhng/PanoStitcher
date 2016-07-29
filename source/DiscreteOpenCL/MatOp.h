#pragma once

#include "DiscreteOpenCLMat.h"

void convert32SC4To8UC4(const docl::GpuMat& src, docl::GpuMat& dst);

void convert32FC4To8UC4(const docl::GpuMat& src, docl::GpuMat& dst);

void setZero(docl::GpuMat& mat);

void setZero8UC4Mask8UC1(docl::GpuMat& mat, const docl::GpuMat& mask);

void setVal16SC1(docl::GpuMat& mat, short val);

void setVal16SC1Mask8UC1(docl::GpuMat& mat, short val, const docl::GpuMat& mask);

void scaledSet16SC1Mask32SC1(docl::GpuMat& image, short val, const docl::GpuMat& mask);

void subtract16SC4(const docl::GpuMat& a, const docl::GpuMat& b, docl::GpuMat& c);

void add32SC4(const docl::GpuMat& a, const docl::GpuMat& b, docl::GpuMat& c);

void accumulate16SC1To32SC1(const docl::GpuMat& src, docl::GpuMat& dst);

void accumulate16SC4To32SC4(const docl::GpuMat& src, const docl::GpuMat& weight, docl::GpuMat& dst);

void normalize32SC4(docl::GpuMat& mat);

void normalize32SC4(docl::GpuMat& mat, const docl::GpuMat& weight);