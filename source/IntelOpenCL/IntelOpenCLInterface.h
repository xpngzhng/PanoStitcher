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

void ioclReproject(const IOclMat& src, IOclMat& dst, const IOclMat& xmap, const IOclMat& ymap);

void ioclReprojectTo16S(const IOclMat& src, IOclMat& dst, const IOclMat& xmap, const IOclMat& ymap);

void ioclReprojectWeightedAccumulateTo32F(const IOclMat& src, IOclMat& dst, 
    const IOclMat& xmap, const IOclMat& ymap, const IOclMat& weight);

class IOclTilingMultibandBlendFast
{
public:
    IOclTilingMultibandBlendFast() : numImages(0), rows(0), cols(0), numLevels(0), success(false) {}
    bool prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength);
    void blend(const std::vector<IOclMat>& images, IOclMat& blendImage);
    void getUniqueMasks(std::vector<IOclMat>& masks) const;

private:
    std::vector<IOclMat> uniqueMasks;
    std::vector<IOclMat> resultPyr, resultUpPyr, resultWeightPyr;
    std::vector<IOclMat> imagePyr, imageUpPyr;
    std::vector<std::vector<IOclMat> > alphaPyrs, weightPyrs;
    IOclMat maskNot;
    int numImages;
    int rows, cols;
    int numLevels;
    bool fullMask;
    bool success;
};
