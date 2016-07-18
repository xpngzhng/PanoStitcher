#pragma once
#include "PanoramaTaskUtil.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include <vector>

template <typename MatType>
struct IntervaledMaskTemplate
{
    IntervaledMaskTemplate() : begInc(-1L), endExc(-1L) {};
    IntervaledMaskTemplate(int videoIndex_, int begIndexInc_, int endIndexInc_, const MatType& mask_)
        : videoIndex(videoIndex_), begIndexInc(begIndexInc_), endIndexInc(endIndexInc_), mask(mask_) {};
    int videoIndex;
    int begIndexInc;
    int endIndexInc;
    MatType mask;
};

template <typename MatType>
struct CustomIntervaledMasksTemplate
{
    CustomIntervaledMasksTemplate() : width(0), height(0), initSuccess(0) {}

    void reset()
    {
        clearAllMasks();
        width = 0;
        height = 0;
        initSuccess = 0;
    }

    bool init(int width_, int height_)
    {
        clearAllMasks();

        initSuccess = 0;
        if (width_ < 0 || height_ < 0)
            return false;

        width = width_;
        height = height_;
        initSuccess = 1;
        return true;
    }

    bool getMask2(int index, MatType& mask) const
    {
        if (!initSuccess)
        {
            mask = MatType();
            return false;
        }

        int size = masks.size();
        for (int i = 0; i < size; i++)
        {
            const IntervaledMaskTemplate<MatType>& currMask = masks[i];
            if (index >= currMask.begIndexInc && index <= currMask.endIndexInc)
            {
                mask = currMask.mask;
                return true;
            }
        }
        mask = MatType();
        return false;
    }

    bool addMask2(int begIndexInc, int endIndexInc, const MatType& mask)
    {
        if (!initSuccess)
            return false;

        if (!mask.data || mask.type() != CV_8UC1 || mask.cols != width || mask.rows != height)
            return false;

        masks.push_back(IntervaledMaskTemplate<MatType>(-1, begIndexInc, endIndexInc, mask.clone()));
        return true;
    }

    void clearMask2(int begIndexInc, int endIndexInc)
    {
        if (precision < 0)
            precision = 0;
        for (std::vector<IntervaledMaskTemplate<MatType> >::iterator itr = masks.begin(); itr != masks.end();)
        {
            if (itr->begIndexInc == begIndexInc &&
                itr->endIndexInc == endIndexInc)
                itr = masks.erase(itr);
            else
                ++itr;
        }
    }

    void clearAllMasks()
    {
        masks.clear();
    }

    int width, height;
    std::vector<IntervaledMaskTemplate<MatType> > masks;
    int initSuccess;
};

typedef IntervaledMaskTemplate<cv::cuda::GpuMat> CudaIntervaledMask;
typedef CustomIntervaledMasksTemplate<cv::cuda::GpuMat> CudaCustomIntervaledMasks;

bool cvtContoursToCudaMasks(const std::vector<std::vector<IntervaledContour> >& contours,
    const std::vector<cv::Mat>& boundedMasks, std::vector<CudaCustomIntervaledMasks>& customMasks);