#include "PanoramaTaskUtil.h"

bool loadPhotoParams(const std::string& cameraParamFile, std::vector<PhotoParam>& params)
{
    std::string::size_type pos = cameraParamFile.find_last_of(".");
    if (pos == std::string::npos)
    {
        printf("Error in %s, file does not have extention\n", __FUNCTION__);
        return false;
    }
    std::string ext = cameraParamFile.substr(pos + 1);
    if (ext == "pts")
        loadPhotoParamFromPTS(cameraParamFile, params);
    else
        loadPhotoParamFromXML(cameraParamFile, params);
    return true;
}

bool prepareSrcVideos(const std::vector<std::string>& srcVideoFiles, bool bgr24, const std::vector<int>& offsets,
    int tryAudioIndex, std::vector<avp::AudioVideoReader>& readers, int& audioIndex, cv::Size& srcSize, int& validFrameCount)
{
    readers.clear();
    srcSize = cv::Size();
    validFrameCount = 0;

    if (srcVideoFiles.empty())
        return false;

    int numVideos = srcVideoFiles.size();
    bool hasOffsets = !offsets.empty();
    if (hasOffsets && offsets.size() != numVideos)
        return false;

    readers.resize(numVideos);

    if (tryAudioIndex < 0 || tryAudioIndex >= numVideos)
    {
        printf("Info in %s, no audio will be opened\n", __FUNCTION__);
        audioIndex = -1;
    }

    bool ok = false;
    double fps = -1;
    for (int i = 0; i < numVideos; i++)
    {
        if (i == tryAudioIndex)
        {
            ok = readers[i].open(srcVideoFiles[i], true, true, bgr24 ? avp::PixelTypeBGR24 : avp::PixelTypeBGR32);
            if (ok)
                audioIndex = tryAudioIndex;
            else
            {
                ok = readers[i].open(srcVideoFiles[i], false, true, bgr24 ? avp::PixelTypeBGR24 : avp::PixelTypeBGR32);
                audioIndex = -1;
            }
        }
        else
            ok = readers[i].open(srcVideoFiles[i], false, true, bgr24 ? avp::PixelTypeBGR24 : avp::PixelTypeBGR32);
        if (!ok)
            break;

        if (srcSize == cv::Size())
        {
            srcSize.width = readers[i].getVideoWidth();
            srcSize.height = readers[i].getVideoHeight();
        }
        if (srcSize.width != readers[i].getVideoWidth() ||
            srcSize.height != readers[i].getVideoHeight())
        {
            ok = false;
            printf("Error in %s, video size unmatch\n", __FUNCTION__);
            break;
        }

        if (fps < 0)
            fps = readers[i].getVideoFps();
        if (abs(fps - readers[i].getVideoFps()) > 0.1)
        {
            printf("Error in %s, video fps not consistent\n", __FUNCTION__);
            ok = false;
            break;
        }

        int count = hasOffsets ? offsets[i] : 0;
        int currValidFrameCount = readers[i].getVideoNumFrames();
        if (currValidFrameCount <= 0)
            validFrameCount = -1;
        else
        {
            currValidFrameCount -= count;
            if (currValidFrameCount <= 0)
            {
                printf("Error in %s, video not long enough\n", __FUNCTION__);
                ok = false;
                break;
            }
        }

        if (validFrameCount == 0)
            validFrameCount = currValidFrameCount;
        if (validFrameCount > 0)
            validFrameCount = validFrameCount > currValidFrameCount ? currValidFrameCount : validFrameCount;

        if (hasOffsets)
        {
            if (!readers[i].seek(1000000.0 * count / fps + 0.5, avp::VIDEO))
            {
                printf("Error in %s, cannot seek to target frame\n", __FUNCTION__);
                ok = false;
                break;
            }
            /*avp::AudioVideoFrame frame;
            for (int j = 0; j < count; j++)
            {
                ok = readers[i].read(frame);
                if (!ok)
                {
                    printf("Error in %s, cannot go to target frame\n", __FUNCTION__);
                    break;
                }
            }*/
            if (!ok)
                break;
        }
    }

    if (!ok)
    {
        readers.clear();
        audioIndex = -1;
        srcSize = cv::Size();
        validFrameCount = 0;
    }

    return ok;
}