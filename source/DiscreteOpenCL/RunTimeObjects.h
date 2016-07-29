#pragma once

#include "oclobject.hpp"

namespace docl
{

bool init();

extern OpenCLBasic* ocl;

extern OpenCLProgramOneKernel* convert32SC4To8UC4;
extern OpenCLProgramOneKernel* convert32FC4To8UC4;

extern OpenCLProgramOneKernel* setZero;
extern OpenCLProgramOneKernel* setZero8UC4Mask8UC1;
extern OpenCLProgramOneKernel* setVal16SC1;
extern OpenCLProgramOneKernel* setVal16SC1Mask8UC1;
extern OpenCLProgramOneKernel* scaledSet16SC1Mask32SC1;
extern OpenCLProgramOneKernel* subtract16SC4;
extern OpenCLProgramOneKernel* add32SC4;
extern OpenCLProgramOneKernel* accumulate16SC1To32SC1;
extern OpenCLProgramOneKernel* accumulate16SC4To32SC4;
extern OpenCLProgramOneKernel* normalizeByShift32SC4;
extern OpenCLProgramOneKernel* normalizeByDivide32SC4;

extern OpenCLProgramOneKernel* reproject;
extern OpenCLProgramOneKernel* reprojectTo16S;
extern OpenCLProgramOneKernel* reprojectWeightedAccumulateTo32F;

extern OpenCLProgramOneKernel* pyrDown8UC1To8UC1;
extern OpenCLProgramOneKernel* pyrDown8UC4To8UC4;
extern OpenCLProgramOneKernel* pyrDown8UC4To32SC4;

extern OpenCLProgramOneKernel* pyrDown16SC1To16SC1;
extern OpenCLProgramOneKernel* pyrDown16SC1To32SC1;

extern OpenCLProgramOneKernel* pyrDown16SC4ScaleTo16SC4;

extern OpenCLProgramOneKernel* pyrDown32FC1;
extern OpenCLProgramOneKernel* pyrDown32FC4;

extern OpenCLProgramOneKernel* pyrUp8UC4To8UC4;
extern OpenCLProgramOneKernel* pyrUp16SC4To16SC4;
extern OpenCLProgramOneKernel* pyrUp32SC4To32SC4;

}

//namespace docl
//{
//
//bool init();
//
//extern OpenCLBasic* ocl;
//}