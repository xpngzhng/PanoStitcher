#pragma once

#include "AudioVideoProcessor.h"
#include "ZReproject.h"
#include "opencv2/core/core.hpp"
#include <vector>

bool loadPhotoParams(const std::string& cameraParamFile, std::vector<PhotoParam>& params);

bool prepareSrcVideos(const std::vector<std::string>& srcVideoFiles, bool bgr24, const std::vector<int>& offsets,
    std::vector<avp::AudioVideoReader>& readers, cv::Size& srcSize, int& validFrameCount);