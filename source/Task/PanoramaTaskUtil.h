#pragma once

#include "PanoramaTask.h"
#include "CustomMask.h"
#include "AudioVideoProcessor.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
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

bool loadVideoFileNamesAndOffset(const std::string& fileName, std::vector<std::string>& videoFileNames, std::vector<int>& offsets);

bool loadIntervaledContours(const std::string& fileName, std::vector<std::vector<IntervaledContour> >& contours);

bool cvtContoursToMasks(const std::vector<std::vector<IntervaledContour> >& contours, 
    const std::vector<cv::Mat>& boundedMasks, std::vector<CustomIntervaledMasks>& customMasks);

bool setIntervaledContoursToPreviewTask(const std::vector<std::vector<IntervaledContour> >& contours,
    CPUPanoramaPreviewTask& task);

bool getIntervaledContoursFromPreviewTask(const CPUPanoramaPreviewTask& task,
    std::vector<std::vector<IntervaledContour> >& contours);

bool cvtContoursToCudaMasks(const std::vector<std::vector<IntervaledContour> >& contours,
    const std::vector<cv::Mat>& boundedMasks, std::vector<CudaCustomIntervaledMasks>& customMasks);
