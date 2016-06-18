#pragma once

#include "PanoramaTask.h"
#include "AudioVideoProcessor.h"
#include "opencv2/core.hpp"
#include <vector>

bool prepareSrcVideos(const std::vector<std::string>& srcVideoFiles, bool bgr24, const std::vector<int>& offsets,
    int tryAudioIndex, std::vector<avp::AudioVideoReader3>& readers, int& audioIndex, cv::Size& srcSize, int& validFrameCount);

struct LogoFilter
{
    LogoFilter() : initSuccess(false), width(0), height(0), type(0) {}
    bool init(int width, int height, int type);
    bool addLogo(cv::Mat& image) const;
    
    int width, height, type;
    cv::Mat logo;
    std::vector<cv::Rect> rects;
    bool initSuccess;
};

// PanoTask Log Printf
void ptlprintf(const char* format, ...);

struct IntervaledContour
{
    int width;
    int height;
    double begIncInMilliSec;
    double endExcInMilliSec;
    std::vector<std::vector<cv::Point> > contours;
};

struct IntervaledMask
{
    IntervaledMask() : begInc(-1LL), endExc(-1LL) {};
    IntervaledMask(long long int begInc_, long long int endExc_, const cv::Mat& mask_)
        : begInc(begInc_), endExc(endExc_), mask(mask_) {};
    long long int begInc;
    long long int endExc;
    cv::Mat mask;
};

bool cvtContourToMask(const IntervaledContour& contour, const cv::Mat& boundedMask, IntervaledMask& customMask);

bool cvtMaskToContour(const IntervaledMask& mask, IntervaledContour& contour);

struct CustomIntervaledMasks
{
    CustomIntervaledMasks() : width(0), height(0), initSuccess(0) {};
    void reset();
    bool init(int width, int height);
    bool getMask(long long int time, cv::Mat& mask) const;
    bool addMask(long long int begInc, long long int endExc, const cv::Mat& mask);
    void clearMask(long long int begInc, long long int endExc, long long int precision = 1000);
    void clearAllMasks();

    int width, height;
    std::vector<IntervaledMask> masks;
    int initSuccess;
};

bool setIntervaledContoursToPreviewTask(const std::vector<std::vector<IntervaledContour> >& contours,
    CPUPanoramaPreviewTask& task);

bool getIntervaledContoursFromPreviewTask(const CPUPanoramaPreviewTask& task,
    std::vector<std::vector<IntervaledContour> >& contours);

bool loadVideoFileNamesAndOffset(const std::string& fileName, std::vector<std::string>& videoFileNames, std::vector<int>& offsets);

bool loadIntervaledContours(const std::string& fileName, std::vector<std::vector<IntervaledContour> >& contours);

bool cvtContoursToMasks(const std::vector<std::vector<IntervaledContour> >& contours, 
    const std::vector<cv::Mat>& boundedMasks, std::vector<CustomIntervaledMasks>& customMasks);


