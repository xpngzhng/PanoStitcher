#pragma once

#include "PanoramaTask.h"
#include "AudioVideoProcessor.h"
#include "opencv2/core.hpp"
#include <vector>

bool prepareSrcVideos(const std::vector<std::string>& srcVideoFiles, bool bgr24, const std::vector<int>& offsets,
    int tryAudioIndex, std::vector<avp::AudioVideoReader>& readers, int& audioIndex, cv::Size& srcSize, int& validFrameCount);

struct LogoFilter
{
    LogoFilter() : initSuccess(false), width(0), height(0), type(0) {}
    bool init(int width, int height, int type);
    bool addLogo(cv::Mat& image);
    
    int width, height, type;
    cv::Mat logo;
    std::vector<cv::Rect> rects;
    bool initSuccess;
};

// PanoTask Log Printf
void ptlprintf(const char* format, ...);

struct IntervaledMask
{
    IntervaledMask() : begInc(-1LL), endExc(-1LL) {};
    IntervaledMask(long long int begInc_, long long int endExc_, const cv::Mat& mask_)
        : begInc(begInc_), endExc(endExc_), mask(mask_) {};
    long long int begInc;
    long long int endExc;
    cv::Mat mask;
};

struct CustomIntervaledMasks
{
    CustomIntervaledMasks() : width(0), height(0), initSuccess(0) {};
    void reset();
    bool init(int width, int height);
    bool getMask(long long int time, cv::Mat& mask);
    bool addMask(long long int begInc, long long int endExc, const cv::Mat& mask);
    void clearMask(long long int begInc, long long int endExc, long long int precision = 1000);
    void clearAllMasks();

    int width, height;
    std::vector<IntervaledMask> masks;
    int initSuccess;
};

