#pragma once

#include "ZBlend.h"
#include "Warp/ZReproject.h"
#include "opencv2/core.hpp"
#include <vector>

void compensate(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results);

void compensateBGR(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results);

void tintAdjust(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results);

class MultibandBlendGainAdjust
{
public:
    MultibandBlendGainAdjust() : numImages(0), rows(0), cols(0), prepareSuccess(false), calcGainSuccess(false) {}
    bool prepare(const std::vector<cv::Mat>& masks, int radius);
    bool calcGain(const std::vector<cv::Mat>& images, std::vector<std::vector<unsigned char> >& luts);
    bool calcGain(const std::vector<cv::Mat>& images, std::vector<std::vector<std::vector<unsigned char> > >& luts);
private:
    int numImages;
    int rows, cols;
    bool prepareSuccess;
    bool calcGainSuccess;
    TilingMultibandBlendFast blender;
    cv::Mat blendImage;
    std::vector<cv::Mat> origMasks, extendedMasks;
    std::vector<std::vector<unsigned char> > luts;
};

class ExposureColorCorrect
{
public:
    ExposureColorCorrect() : numImages(0), rows(0), cols(0), prepareSuccess(0) {};
    bool prepare(const std::vector<cv::Mat>& masks);
    bool correctExposure(const std::vector<cv::Mat>& images, std::vector<double>& exposures);
    bool correctExposureAndWhiteBalance(const std::vector<cv::Mat>& images, std::vector<double>& exposures,
        std::vector<double>& redRatios, std::vector<double>& blueRatios);
    bool correctColorExposure(const std::vector<cv::Mat>& images, std::vector<std::vector<double> >& exposures);
    void clear();
    static bool getExposureLUTs(const std::vector<double>& exposures, std::vector<std::vector<unsigned char> >& luts);
    static bool getExposureAndWhiteBalanceLUTs(const std::vector<double>& exposures, const std::vector<double>& redRatios,
        const std::vector<double>& blueRatios, std::vector<std::vector<std::vector<unsigned char> > >& luts);
    static bool getColorExposureLUTs(const std::vector<std::vector<double> >& exposures,
        std::vector<std::vector<std::vector<unsigned char> > >& luts);
private:
    int numImages;
    int rows, cols;
    int prepareSuccess;
    std::vector<cv::Mat> origMasks, transImages;
};

enum OptimizeParamType
{
    EXPOSURE = 1,
    WHITE_BALANCE = 2,
};

enum GetPointPairsMethod
{
    RANDOM_SAMPLE,
    GRID_SAMPLE,
    HISTOGRAM,
    NUM_GET_POINT_PAIRS_METHODS
};

void exposureColorOptimize(const std::vector<cv::Mat>& images, const std::vector<PhotoParam>& params,
    const std::vector<int> anchorIndexes, int getPointPairsMethod, int optimizeWhat,
    std::vector<double>& exposures, std::vector<double>& redRatios, std::vector<double>& blueRatios);

void getExposureColorOptimizeLUTs(const std::vector<double>& exposures, const std::vector<double>& redRatios,
    const std::vector<double>& blueRatios, std::vector<std::vector<std::vector<unsigned char> > >& luts);

void transform(const cv::Mat& src, cv::Mat& dst, const std::vector<unsigned char>& lut, const cv::Mat& mask = cv::Mat());

void transform(const cv::Mat& src, cv::Mat& dst, const std::vector<std::vector<unsigned char> >& luts,
    const cv::Mat& mask = cv::Mat());
