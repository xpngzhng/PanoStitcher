#pragma once

#include "oclobject.hpp"

#define PYR_UP_OPENCV 0

namespace iocl
{

bool init();

extern OpenCLBasic* ocl;

extern OpenCLProgramOneKernel* setZero;

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