#pragma once

#include "IntelOpenCLMat.h"
#include "oclobject.hpp"
#include "CL/cl.h"

void ioclSetZero(IOclMat& mat, OpenCLBasic& ocl, OpenCLProgramOneKernel& kern);

void ioclReproject(const IOclMat& src, IOclMat& dst, const IOclMat& xmap, const IOclMat& ymap,
    OpenCLBasic& ocl, OpenCLProgramOneKernel& kern);

void ioclReprojectAccumulateWeightedTo32F(const IOclMat& src, IOclMat& dst, const IOclMat& xmap, const IOclMat& ymap,
    const IOclMat& weight, OpenCLBasic& ocl, OpenCLProgramOneKernel& kern);

bool ioclInit();

void ioclSetZero(IOclMat& mat);

void ioclReproject(const IOclMat& src, IOclMat& dst, const IOclMat& xmap, const IOclMat& ymap);

void ioclReprojectTo16S(const IOclMat& src, IOclMat& dst, const IOclMat& xmap, const IOclMat& ymap);

void ioclReprojectWeightedAccumulateTo32F(const IOclMat& src, IOclMat& dst, 
    const IOclMat& xmap, const IOclMat& ymap, const IOclMat& weight);

void ioclPyramidDown8UC1To8UC1(const IOclMat& src, IOclMat& dst, cv::Size dstSize);

void ioclPyramidDown8UC4To8UC4(const IOclMat& src, IOclMat& dst, cv::Size dstSize);

void ioclPyramidDown8UC4To32SC4(const IOclMat& src, IOclMat& dst, cv::Size dstSize);

void ioclPyramidDown32FC1(const IOclMat& src, IOclMat& dst, cv::Size dstSize);

void ioclPyramidDown32FC4(const IOclMat& src, IOclMat& dst, cv::Size dstSize);
