#pragma once

#include "OpenCLAccel/oclobject.hpp"
#include "IntelOpenCLMat.h"

void ioclSetZero(iocl::UMat& mat, OpenCLBasic& ocl, OpenCLProgramOneKernel& kern);

void ioclReproject(const iocl::UMat& src, iocl::UMat& dst, const iocl::UMat& xmap, const iocl::UMat& ymap,
    OpenCLBasic& ocl, OpenCLProgramOneKernel& kern);

void ioclReprojectAccumulateWeightedTo32F(const iocl::UMat& src, iocl::UMat& dst, const iocl::UMat& xmap, const iocl::UMat& ymap,
    const iocl::UMat& weight, OpenCLBasic& ocl, OpenCLProgramOneKernel& kern);

bool ioclInit();

void ioclReproject(const iocl::UMat& src, iocl::UMat& dst, const iocl::UMat& xmap, const iocl::UMat& ymap);

void ioclReprojectTo16S(const iocl::UMat& src, iocl::UMat& dst, const iocl::UMat& xmap, const iocl::UMat& ymap);

void ioclReprojectWeightedAccumulateTo32F(const iocl::UMat& src, iocl::UMat& dst, 
    const iocl::UMat& xmap, const iocl::UMat& ymap, const iocl::UMat& weight);

class IOclTilingMultibandBlendFast
{
public:
    IOclTilingMultibandBlendFast() : numImages(0), rows(0), cols(0), numLevels(0), success(false) {}
    bool prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength);
    void blend(const std::vector<iocl::UMat>& images, iocl::UMat& blendImage);
    void getUniqueMasks(std::vector<iocl::UMat>& masks) const;

private:
    std::vector<iocl::UMat> uniqueMasks;
    std::vector<iocl::UMat> resultPyr, resultUpPyr, resultWeightPyr;
    std::vector<iocl::UMat> imagePyr, imageUpPyr;
    std::vector<std::vector<iocl::UMat> > alphaPyrs, weightPyrs;
    iocl::UMat maskNot;
    int numImages;
    int rows, cols;
    int numLevels;
    bool fullMask;
    bool success;
};
