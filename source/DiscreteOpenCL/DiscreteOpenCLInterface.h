#pragma once

#include "DiscreteOpenCLMat.h"
#include "oclobject.hpp"
#include "CL/cl.h"

bool doclInit();

void doclReproject(const docl::GpuMat& src, docl::GpuMat& dst, const docl::GpuMat& xmap, const docl::GpuMat& ymap);

void doclReprojectTo16S(const docl::GpuMat& src, docl::GpuMat& dst, const docl::GpuMat& xmap, const docl::GpuMat& ymap);

void doclReprojectWeightedAccumulateTo32F(const docl::GpuMat& src, docl::GpuMat& dst, 
    const docl::GpuMat& xmap, const docl::GpuMat& ymap, const docl::GpuMat& weight);

class DOclTilingMultibandBlendFast
{
public:
    DOclTilingMultibandBlendFast() : numImages(0), rows(0), cols(0), numLevels(0), success(false) {}
    bool prepare(const std::vector<cv::Mat>& masks, int maxLevels, int minLength);
    void blend(const std::vector<docl::GpuMat>& images, docl::GpuMat& blendImage);
    void getUniqueMasks(std::vector<docl::GpuMat>& masks) const;

private:
    std::vector<docl::GpuMat> uniqueMasks;
    std::vector<docl::GpuMat> resultPyr, resultUpPyr, resultWeightPyr;
    std::vector<docl::GpuMat> imagePyr, image32SPyr, imageUpPyr;
    std::vector<std::vector<docl::GpuMat> > alphaPyrs, weightPyrs;
    docl::GpuMat maskNot;
    int numImages;
    int rows, cols;
    int numLevels;
    bool fullMask;
    bool success;
};

void alphaBlend8UC4(docl::GpuMat& target, const docl::GpuMat& blender);

void cvtBGR32ToYUV420P(const docl::GpuMat& bgr32, docl::GpuMat& y, docl::GpuMat& u, docl::GpuMat& v);

void cvtBGR32ToNV12(const docl::GpuMat& bgr32, docl::GpuMat& y, docl::GpuMat& uv);
