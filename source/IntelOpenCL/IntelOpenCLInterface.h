#pragma once

#include "IntelOpenCLMat.h"
#include "oclobject.hpp"
#include "CL/cl.h"

void ioclSetZero(IOclMat& mat, OpenCLBasic& ocl, OpenCLProgramOneKernel& kern);

void ioclReproject(const IOclMat& src, IOclMat& dst, const IOclMat& xmap, const IOclMat& ymap,
    OpenCLBasic& ocl, OpenCLProgramOneKernel& kern);

void ioclReprojectAccumulateWeightedTo32F(const IOclMat& src, IOclMat& dst, const IOclMat& xmap, const IOclMat& ymap,
    const IOclMat& weight, OpenCLBasic& ocl, OpenCLProgramOneKernel& kern);

void ioclReprojectAccumulateWeightedTo32F2(const IOclMat& src, IOclMat& dst, const IOclMat& map, 
    const IOclMat& interpWeight, const IOclMat& blendWeight, OpenCLBasic& ocl, OpenCLProgramOneKernel& kern);
