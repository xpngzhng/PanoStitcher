#include "PanoramaTask.h"
#include "PanoramaTaskUtil.h"
#include "ZReproject.h"

CPUPanoramaPreviewTask::CPUPanoramaPreviewTask()
{
    clear();
}

CPUPanoramaPreviewTask::~CPUPanoramaPreviewTask()
{
    clear();
}

bool CPUPanoramaPreviewTask::init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile,
    int dstWidth, int dstHeight)
{
    clear();

    if (srcVideoFiles.empty())
    {
        printf("Error in %s, size of srcVideoFiles empty\n", __FUNCTION__);
        return false;
    }

    numVideos = srcVideoFiles.size();

    printf("Info in %s, open videos and set to the correct frames\n", __FUNCTION__);
    bool ok = false;
    int validFrameCount;
    int audioIndex;
    ok = prepareSrcVideos(srcVideoFiles, true, std::vector<int>(), -1, readers, audioIndex, srcSize, validFrameCount);
    if (!ok)
    {
        printf("Error in %s, could not open video file(s)\n", __FUNCTION__);
        return false;
    }
    printf("Info in %s, open videos done\n", __FUNCTION__);
    
    printf("Info in %s, prepare for reproject and blend\n", __FUNCTION__);
    dstSize.width = dstWidth;
    dstSize.height = dstHeight;
    std::vector<PhotoParam> params;
    if (!loadPhotoParams(cameraParamFile, params))
    {
        printf("Error in %s, failed to load params\n", __FUNCTION__);
        return false;
    }
    if (params.size() < numVideos)
    {
        printf("Error in %s, params.size() < numVideos\n", __FUNCTION__);
        return false;
    }
    else if (params.size() > numVideos)
    {
        printf("Warning in %s, params.size() > numVideos\n", __FUNCTION__);
    }
    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, dstMasks);
    ok = blender.prepare(dstMasks, 16, 2);
    //ok = blender.prepare(dstMasks, 50);
    if (!ok)
    {
        printf("Error in %s, blender prepare failed\n", __FUNCTION__);
        return false;
    }
    printf("Info in %s, prepare finish\n", __FUNCTION__);

    initSuccess = true;
    return true;
}

bool CPUPanoramaPreviewTask::reset(const std::string& cameraParamFile)
{
    if (!initSuccess)
        return false;

    std::vector<PhotoParam> params;
    if (!loadPhotoParams(cameraParamFile, params))
    {
        printf("Error in %s, failed to load params\n", __FUNCTION__);
        return false;
    }
    if (params.size() < numVideos)
    {
        printf("Error in %s, params.size() < numVideos\n", __FUNCTION__);
        return false;
    }
    else if (params.size() > numVideos)
    {
        printf("Warning in %s, params.size() > numVideos\n", __FUNCTION__);
    }
    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, dstMasks);

    bool ok = blender.prepare(dstMasks, 16, 2);
    //ok = blender.prepare(dstMasks, 50);
    if (!ok)
    {
        printf("Error in %s, blender prepare failed\n", __FUNCTION__);
        return false;
    }

    return true;
}

bool CPUPanoramaPreviewTask::seek(const std::vector<long long int>& timeStamps)
{
    if (!initSuccess)
        return false;

    if (timeStamps.size() != numVideos)
        return false;

    bool ok = true;
    for (int i = 0; i < numVideos; i++)
    {
        if (!readers[i].seek(timeStamps[i], avp::VIDEO))
        {
            ok = false;
            break;
        }
    }
    return ok;
}

bool CPUPanoramaPreviewTask::stitch(cv::Mat& result, std::vector<long long int>& timeStamps, int frameIncrement)
{
    if (!initSuccess)
        return false;

    if (frameIncrement <= 0 || frameIncrement > 10)
        frameIncrement = 1;

    //printf("In %s, begin read frame\n", __FUNCTION__);
    std::vector<cv::Mat> images(numVideos);
    timeStamps.resize(numVideos);
    bool ok = true;
    for (int i = 0; i < numVideos; i++)
    {
        avp::AudioVideoFrame frame;
        for (int j = 0; j < frameIncrement; j++)
        {
            if (!readers[i].read(frame))
            {
                ok = false;
                break;
            }
        }
        if (!ok)
            break;

        images[i] = cv::Mat(frame.height, frame.width, CV_8UC3, frame.data, frame.step);
        timeStamps[i] = frame.timeStamp;
    }
    if (!ok)
        return false;

    //printf("In %s, read frame success\n", __FUNCTION__);
    reprojectParallel(images, reprojImages, dstSrcMaps);
    //printf("In %s, reproject success\n", __FUNCTION__);
    blender.blend(reprojImages, blendImage);
    result = blendImage;
    //printf("In %s, stitch success\n", __FUNCTION__);
    return true;
}

void CPUPanoramaPreviewTask::clear()
{
    numVideos = 0;
    readers.clear();
    initSuccess = false;
}

CudaPanoramaPreviewTask::CudaPanoramaPreviewTask()
{
    clear();
}

CudaPanoramaPreviewTask::~CudaPanoramaPreviewTask()
{
    clear();
}

bool CudaPanoramaPreviewTask::init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile,
    int dstWidth, int dstHeight)
{
    clear();

    if (srcVideoFiles.empty())
    {
        printf("Error in %s, size of srcVideoFiles empty\n", __FUNCTION__);
        return false;
    }
    numVideos = srcVideoFiles.size();

    printf("Info in %s, open videos and set to the correct frames\n", __FUNCTION__);
    bool ok = false;
    int validFrameCount;
    int audioIndex;
    ok = prepareSrcVideos(srcVideoFiles, false, std::vector<int>(), -1, readers, audioIndex, srcSize, validFrameCount);
    if (!ok)
    {
        printf("Error in %s, could not open video file(s)\n", __FUNCTION__);
        return false;
    }
    printf("Info in %s, open videos done\n", __FUNCTION__);

    printf("Info in %s, prepare for reproject and blend\n", __FUNCTION__);
    dstSize.width = dstWidth;
    dstSize.height = dstHeight;
    ptrRender.reset(new CudaMultiCameraPanoramaRender);
    ok = ptrRender->prepare(cameraParamFile, PanoramaRender::BlendTypeMultiband, srcSize, dstSize);
    if (!ok)
    {
        printf("Error in %s, prepare failed\n", __FUNCTION__);
        return false;
    }
    printf("Info in %s, prepare finish\n", __FUNCTION__);

    initSuccess = true;
    return true;
}

bool CudaPanoramaPreviewTask::reset(const std::string& cameraParamFile)
{
    if (!initSuccess)
        return false;

    bool ok = ptrRender->prepare(cameraParamFile, PanoramaRender::BlendTypeMultiband, srcSize, dstSize);
    if (!ok)
    {
        printf("Error in %s, prepare failed\n", __FUNCTION__);
        return false;
    }

    return true;
}

bool CudaPanoramaPreviewTask::seek(const std::vector<long long int>& timeStamps)
{
    if (!initSuccess)
        return false;

    if (timeStamps.size() != numVideos)
        return false;

    bool ok = true;
    for (int i = 0; i < numVideos; i++)
    {
        if (!readers[i].seek(timeStamps[i], avp::VIDEO))
        {
            ok = false;
            break;
        }
    }
    return ok;
}

bool CudaPanoramaPreviewTask::stitch(cv::Mat& result, std::vector<long long int>& timeStamps, int frameIncrement)
{
    if (!initSuccess)
        return false;

    if (frameIncrement <= 0 || frameIncrement > 10)
        frameIncrement = 1;

    std::vector<cv::Mat> images(numVideos);
    timeStamps.resize(numVideos);
    bool ok = true;
    for (int i = 0; i < numVideos; i++)
    {
        avp::AudioVideoFrame frame;
        for (int j = 0; j < frameIncrement; j++)
        {
            if (!readers[i].read(frame))
            {
                ok = false;
                break;
            }
        }
        if (!ok)
            break;

        images[i] = cv::Mat(frame.height, frame.width, CV_8UC4, frame.data, frame.step);
        timeStamps[i] = frame.timeStamp;
    }
    if (!ok)
        return false;

   ok = ptrRender->render(images, blendImage);
   if (!ok)
       return false;
   result = blendImage;
   return true;
}

void CudaPanoramaPreviewTask::clear()
{
    numVideos = 0;
    readers.clear();
    ptrRender.reset(0);
    initSuccess = false;
}