#pragma once

#include "AudioVideoProcessor.h"
#include "ZReproject.h"
#include "opencv2/core.hpp"
#include <vector>

bool loadPhotoParams(const std::string& cameraParamFile, std::vector<PhotoParam>& params);

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