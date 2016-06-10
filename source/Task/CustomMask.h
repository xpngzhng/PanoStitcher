#pragma once
#include "PanoramaTaskUtil.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include <vector>

template <typename MatType>
struct IntervaledMaskTemplate
{
    IntervaledMaskTemplate() : begInc(-1LL), endExc(-1LL) {};
    IntervaledMaskTemplate(long long int begInc_, long long int endExc_, const MatType& mask_)
        : begInc(begInc_), endExc(endExc_), mask(mask_) {};
    long long int begInc;
    long long int endExc;
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

    bool getMask(long long int time, MatType& mask) const
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
            if (time >= currMask.begInc && time < currMask.endExc)
            {
                mask = currMask.mask;
                return true;
            }
        }
        mask = MatType();
        return false;
    }

    bool addMask(long long int begInc, long long int endExc, const MatType& mask)
    {
        if (!initSuccess)
            return false;

        if (!mask.data || mask.type() != CV_8UC1 || mask.cols != width || mask.rows != height)
            return false;

        masks.push_back(IntervaledMaskTemplate<MatType>(begInc, endExc, mask.clone()));
        return true;
    }

    void clearMask(long long int begInc, long long int endExc, long long int precision = 1000)
    {
        if (precision < 0)
            precision = 0;
        for (std::vector<IntervaledMaskTemplate<MatType> >::iterator itr = masks.begin(); itr != masks.end();)
        {
            if (abs(itr->begInc - begInc) <= precision &&
                abs(itr->endExc - endExc) <= precision)
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