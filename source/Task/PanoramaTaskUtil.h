#pragma once

#include "PanoramaTask.h"
#include "AudioVideoProcessor.h"
#include "opencv2/core.hpp"
#include <vector>

bool prepareSrcVideos(const std::vector<std::string>& srcVideoFiles, avp::PixelType pixelType, const std::vector<int>& offsets,
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
    int videoIndex;
    int width;
    int height;
    int begIndexInc;
    int endIndexInc;
    std::vector<std::vector<cv::Point> > contours;
};

struct IntervaledMask
{
    IntervaledMask() : begIndexInc(-1), endIndexInc(-1) {};
    IntervaledMask(int index_, int begInc_, int endExc_, const cv::Mat& mask_)
        : videoIndex(index_), begIndexInc(begInc_), endIndexInc(endExc_), mask(mask_) {};
    int videoIndex;
    int begIndexInc;
    int endIndexInc;
    cv::Mat mask;
};

bool cvtContourToMask(const IntervaledContour& contour, const cv::Mat& boundedMask, IntervaledMask& customMask);

bool cvtMaskToContour(const IntervaledMask& mask, IntervaledContour& contour);

struct CustomIntervaledMasks
{
    CustomIntervaledMasks() : width(0), height(0), initSuccess(0) {};
    void reset();
    bool init(int width, int height);
    bool getMask2(int index, cv::Mat& mask) const;
    bool addMask2(int begIndexInc, int endIndexInc, const cv::Mat& mask);
    void clearMask2(int begIndexInc, int endIndexInc);
    void clearAllMasks();

    int width, height;
    std::vector<IntervaledMask> masks;
    int initSuccess;
};

struct GeneralMasks
{
    GeneralMasks() : width(0), height(0), initSuccess(0) {};
    void reset();
    bool init(const std::vector<cv::Mat>& masks);
    bool getMasks(const std::vector<int>& frameIndexes, std::vector<cv::Mat>& masks);
    bool addMasks(const std::vector<IntervaledMask>& masks);

    int width, height;
    int numVideos;
    std::vector<std::vector<IntervaledMask> > customMasks;
    std::vector<cv::Mat> defaultMasks;
    int initSuccess;
};

bool setIntervaledContoursToPreviewTask(const std::vector<std::vector<IntervaledContour> >& contours,
    CPUPanoramaPreviewTask& task);

bool getIntervaledContoursFromPreviewTask(const CPUPanoramaPreviewTask& task, const std::vector<int>& offsets, 
    std::vector<std::vector<IntervaledContour> >& contours);

bool loadVideoFileNamesAndOffset(const std::string& fileName, std::vector<std::string>& videoFileNames, std::vector<int>& offsets);

bool loadIntervaledContours(const std::string& fileName, std::vector<std::vector<IntervaledContour> >& contours);

bool cvtContoursToMasks(const std::vector<std::vector<IntervaledContour> >& contours, 
    const std::vector<cv::Mat>& boundedMasks, std::vector<CustomIntervaledMasks>& customMasks);
