#pragma once

#include "DiscreteOpenCLMat.h"

void convert32SC4To8UC4(const IOclMat& src, IOclMat& dst);

void convert32FC4To8UC4(const IOclMat& src, IOclMat& dst);

void setZero(IOclMat& mat);

void setZero8UC4Mask8UC1(IOclMat& mat, const IOclMat& mask);

void setVal16SC1(IOclMat& mat, short val);

void setVal16SC1Mask8UC1(IOclMat& mat, short val, const IOclMat& mask);

void scaledSet16SC1Mask32SC1(IOclMat& image, short val, const IOclMat& mask);

void subtract16SC4(const IOclMat& a, const IOclMat& b, IOclMat& c);

void add32SC4(const IOclMat& a, const IOclMat& b, IOclMat& c);

void accumulate16SC1To32SC1(const IOclMat& src, IOclMat& dst);

void accumulate16SC4To32SC4(const IOclMat& src, const IOclMat& weight, IOclMat& dst);

void normalize32SC4(IOclMat& mat);

void normalize32SC4(IOclMat& mat, const IOclMat& weight);