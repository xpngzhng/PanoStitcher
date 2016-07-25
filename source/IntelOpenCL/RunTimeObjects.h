#pragma once

#include "oclobject.hpp"

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

}