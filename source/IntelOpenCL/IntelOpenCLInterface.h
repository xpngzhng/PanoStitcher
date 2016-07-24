#pragma once

#include "IntelOpenCLMat.h"
#include "oclobject.hpp"
#include "CL/cl.h"

void ioclSetZero(IOclMat& mat, OpenCLBasic& ocl, OpenCLProgramOneKernel& kern);

void ioclReproject(const IOclMat& src, IOclMat& dst, const IOclMat& xmap, const IOclMat& ymap,
    OpenCLBasic& ocl, OpenCLProgramOneKernel& kern);

void ioclReprojectAccumulateWeightedTo32F(const IOclMat& src, IOclMat& dst, const IOclMat& xmap, const IOclMat& ymap,
    const IOclMat& weight, OpenCLBasic& ocl, OpenCLProgramOneKernel& kern);

void ioclSetZero(IOclMat& mat);

void ioclReproject(const IOclMat& src, IOclMat& dst, const IOclMat& xmap, const IOclMat& ymap);

void ioclReprojectAccumulateWeightedTo32F(const IOclMat& src, IOclMat& dst, 
    const IOclMat& xmap, const IOclMat& ymap, const IOclMat& weight);


