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

}