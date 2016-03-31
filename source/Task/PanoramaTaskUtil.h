#pragma once

#include "AudioVideoProcessor.h"
#include "ZReproject.h"
#include "opencv2/core/core.hpp"
#include <vector>

bool loadPhotoParams(const std::string& cameraParamFile, std::vector<PhotoParam>& params);

bool prepareSrcVideos(const std::vector<std::string>& srcVideoFiles, bool bgr24, const std::vector<int>& offsets,
    int tryAudioIndex, std::vector<avp::AudioVideoReader>& readers, int& audioIndex, cv::Size& srcSize, int& validFrameCount);