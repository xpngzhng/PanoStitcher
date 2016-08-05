#include "CompileControl.h"
#include "PanoramaTask.h"
#include "ConcurrentQueue.h"
#include "PinnedMemoryFrameQueue.h"
#include "SharedAudioVideoFramePool.h"
#include "RicohUtil.h"
#include "PanoramaTaskUtil.h"
#include "CudaPanoramaTaskUtil.h"
#include "LiveStreamTaskUtil.h"
#include "Timer.h"
#include "Image.h"
#include "Text.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

typedef std::pair<int, std::string> AsyncErrorMessage;
typedef CompleteQueue<AsyncErrorMessage> AsyncErrorMessageQueue;

struct PanoramaLiveStreamTask2::Impl
{
    Impl();
    ~Impl();

    bool openAudioVideoSources(const std::vector<avp::Device>& devices, int width, int height, int frameRate,
        bool openAudio, const avp::Device& device, int sampleRate);
    bool openAudioVideoSources(const std::vector<std::string>& urls, bool openAudio, const std::string& url);
    void closeAudioVideoSources();

    bool beginVideoStitch(int panoStitchType, const std::string& configFileName, 
        int width, int height, bool highQualityBlend);
    void stopVideoStitch();

    bool openLiveStream(const std::string& name, int panoType, int width, int height, int videoBPS,
        const std::string& videoEncoder, const std::string& videoPreset, int audioBPS);
    void closeLiveStream();

    bool beginSaveToDisk(const std::string& dir, int panoType, int width, int height, int videoBPS,
        const std::string& videoEncoder, const std::string& videoPreset, int audioBPS, int fileDuration);
    void stopSaveToDisk();

    bool calcExposures(std::vector<double>& exposures);
    bool setExposures(const std::vector<double>& exposures);
    void resetExposures();

    bool getVideoSourceFrames(std::vector<avp::AudioVideoFrame2>& frames);
    bool getStitchedVideoFrame(avp::AudioVideoFrame2& frame);
    void cancelGetVideoSourceFrames();
    void cancelGetStitchedVideoFrame();

    double getVideoSourceFrameRate() const;
    double getStitchFrameRate() const;
    void getLastSyncErrorMessage(std::string& message) const;
    bool hasAsyncErrorMessage() const;
    void getLastAsyncErrorMessage(std::string& message, int& fromWhere);
    void getLog(std::string& logInfo);

    void initAll();
    void closeAll();
    bool hasFinished() const;

    std::unique_ptr<AudioVideoSource> audioVideoSource;
    cv::Size videoFrameSize;
    double videoFrameRate;
    int numVideos;
    int videoOpenSuccess;
    int audioSampleRate;
    int audioOpenSuccess;

    std::unique_ptr<CudaPanoramaRender2> render;
    std::string renderConfigName;
    cv::Size renderFrameSize;
    int renderPrepareSuccess;
    std::unique_ptr<std::thread> renderThread;
    int renderEndFlag;
    int renderThreadJoined;
    void procVideo();

    CudaWatermarkFilter watermarkFilter;

    avp::AudioVideoWriter3 streamWriter;
    std::string streamURL;
    int streamPanoType;
    cv::cuda::GpuMat streamXMap, streamYMap;
    cv::Size streamFrameSize;
    int streamVideoBitRate;
    std::string streamVideoEncodePreset;
    int streamAudioBitRate;
    int streamOpenSuccess;
    int streamIsLibX264;
    std::unique_ptr<std::thread> streamThread;
    int streamEndFlag;
    int streamThreadJoined;
    void streamSend();

    std::string fileDir;
    int filePanoType;
    cv::cuda::GpuMat fileXMap, fileYMap;
    cv::Size fileFrameSize;
    int fileVideoBitRate;
    std::string fileVideoEncoder;
    std::string fileVideoEncodePreset;
    int fileAudioBitRate;
    int fileDuration;
    int fileConfigSet;
    int fileIsLibX264;
    std::unique_ptr<std::thread> fileThread;
    int fileEndFlag;
    int fileThreadJoined;
    void fileSave();

    ImageVisualCorrect correct;
    std::vector<double> exposures;
    std::mutex mtxLuts;
    std::vector<std::vector<unsigned char> > luts;
    void setLUTs(const std::vector<double>& exposures);
    void clearLUTs();
    void getLuts(std::vector<std::vector<unsigned char> >& luts);

    double videoSourceFrameRate;
    double stitchVideoFrameRate;

    std::string syncErrorMessage;

    int hasAsyncError;
    void addAsyncErrorMessage(const std::string& message, int fromWhere);
    void clearAsyncErrorMessages();
    AsyncErrorMessageQueue asyncErrorMessageQueue;

    std::mutex mtxLog;
    std::string log;
    void appendLog(const std::string& message);
    void clearLog();

    int pixelType;
    int elemType;
    int finish;
    int allowGetSyncedFramesBufferForShow;
    ForShowFrameVectorQueue syncedFramesBufferForShow;
    BoundedPinnedMemoryFrameQueue syncedFramesBufferForProc;
    CudaHostMemVideoFrameMemoryPool procFramePool, sendFramePool, saveFramePool;
    ForShowMixedFrameQueue procFrameBufferForShow;
    ForceWaitMixedFrameQueue procFrameBufferForSend, procFrameBufferForSave;
};

PanoramaLiveStreamTask2::Impl::Impl()
{
    initAll();
}

PanoramaLiveStreamTask2::Impl::~Impl()
{
    closeAll();
}

bool PanoramaLiveStreamTask2::Impl::openAudioVideoSources(const std::vector<avp::Device>& devices, int width, int height, int frameRate,
    bool openAudio, const avp::Device& device, int sampleRate)
{
    if (audioVideoSource)
    {
        ptlprintf("Error in %s, audio video sources should be closed first before open again\n", __FUNCTION__);
        syncErrorMessage = getText(TI_AUDIO_VIDEO_SOURCE_RUNNING_CLOSE_BEFORE_LAUNCH_NEW);
        return false;
    }

    audioVideoSource.reset(new FFmpegAudioVideoSource(&syncedFramesBufferForShow, &syncedFramesBufferForProc, 1,
        &procFrameBufferForSend, &procFrameBufferForSave, &finish));
    bool ok = ((FFmpegAudioVideoSource*)audioVideoSource.get())->open(devices, width, height, frameRate, openAudio, device, sampleRate);
    if (!ok)
        return false;
    videoFrameSize.width = width;
    videoFrameSize.height = height;
    videoFrameRate = frameRate;
    numVideos = devices.size();
    audioSampleRate = sampleRate;
    videoOpenSuccess = 1;
    audioOpenSuccess = audioVideoSource->isAudioOpened();
    return true;
}

bool PanoramaLiveStreamTask2::Impl::openAudioVideoSources(const std::vector<std::string>& urls, bool openAudio, const std::string& url)
{
    if (audioVideoSource)
    {
        ptlprintf("Error in %s, audio video sources should be closed first before open again\n", __FUNCTION__);
        syncErrorMessage = getText(TI_AUDIO_VIDEO_SOURCE_RUNNING_CLOSE_BEFORE_LAUNCH_NEW);
        return false;
    }

    bool ok = false;
    if (areAllIPAdresses(urls))
    {
        audioVideoSource.reset(new JuJingAudioVideoSource(&syncedFramesBufferForShow, &syncedFramesBufferForProc, 1,
            &procFrameBufferForSend, &procFrameBufferForSave, &finish));
        ok = ((JuJingAudioVideoSource*)audioVideoSource.get())->open(urls);
    }
    else
    {
        audioVideoSource.reset(new FFmpegAudioVideoSource(&syncedFramesBufferForShow, &syncedFramesBufferForProc, 1,
            &procFrameBufferForSend, &procFrameBufferForSave, &finish));
        ok = ((FFmpegAudioVideoSource*)audioVideoSource.get())->open(urls);
    }
    if (!ok)
        return false;
    videoFrameSize.width = audioVideoSource->getVideoFrameWidth();
    videoFrameSize.height = audioVideoSource->getVideoFrameHeight();
    videoFrameRate = audioVideoSource->getVideoFrameRate();
    numVideos = urls.size();
    audioSampleRate = 0;
    videoOpenSuccess = 1;
    audioOpenSuccess = audioVideoSource->isAudioOpened();
    return true;
}

void PanoramaLiveStreamTask2::Impl::closeAudioVideoSources()
{
    if (audioVideoSource)
    {
        audioVideoSource->close();
        audioVideoSource.reset();

        videoOpenSuccess = 0;
        audioOpenSuccess = 0;
    } 
}

bool PanoramaLiveStreamTask2::Impl::beginVideoStitch(int panoStitchType, const std::string& configFileName, int width, int height, bool highQualityBlend)
{
    if (!videoOpenSuccess || !audioVideoSource)
    {
        ptlprintf("Error in %s, audio video sources not opened\n", __FUNCTION__);
        syncErrorMessage = getText(TI_SOURCE_NOT_OPENED_CANNOT_LAUNCH_STITCH);
        return false;
    }

    if (!renderThreadJoined)
    {
        ptlprintf("Error in %s, stitching running, stop before launching new stitching\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_RUNNING_CLOSE_BEFORE_LAUNCH_NEW);
        return false;
    }

    renderConfigName = configFileName;
    renderFrameSize.width = width;
    renderFrameSize.height = height;

    if (panoStitchType == PanoStitchTypeMISO)
        render.reset(new CudaPanoramaRender2);
    else if (panoStitchType == PanoStitchTypeRicoh)
        render.reset(new CudaRicohPanoramaRender);
    else
    {
        ptlprintf("Error in %s, unsupported pano stitch type %d, should be %d or %d\n",
            __FUNCTION__, panoStitchType, PanoStitchTypeMISO, PanoStitchTypeRicoh);
    }

    renderPrepareSuccess = render->prepare(renderConfigName, highQualityBlend, 
        videoFrameSize, renderFrameSize);
    if (!renderPrepareSuccess)
    {
        ptlprintf("Error in %s, could not prepare for video stitch\n", __FUNCTION__);
        appendLog(getText(TI_STITCH_INIT_FAIL) + "\n");
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (render->getNumImages() != numVideos)
    {
        ptlprintf("Error in %s, num images in render not equal to num videos\n", __FUNCTION__);
        appendLog(getText(TI_STITCH_INIT_FAIL) + "\n");
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    renderPrepareSuccess = procFramePool.init(pixelType, width, height);
    if (!renderPrepareSuccess)
    {
        ptlprintf("Error in %s, could not init proc frame pool\n", __FUNCTION__);
        appendLog(getText(TI_STITCH_INIT_FAIL) + "\n");
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (addWatermark)
        renderPrepareSuccess = watermarkFilter.init(width, height);
    if (!renderPrepareSuccess)
    {
        ptlprintf("Error in %s, could not init logo filter\n", __FUNCTION__);
        appendLog(getText(TI_STITCH_INIT_FAIL) + "\n");
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    renderPrepareSuccess = correct.prepare(renderConfigName, videoFrameSize, cv::Size(960, 480));
    if (!renderPrepareSuccess)
    {
        ptlprintf("Error in %s, could not prepare for exposure correct\n", __FUNCTION__);
        appendLog(getText(TI_STITCH_INIT_FAIL) + "\n");
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    ptlprintf("Info in %s, stitching param: width %d, height %d, high quality blend %d\n",
        __FUNCTION__, width, height, highQualityBlend);
    appendLog(getText(TI_STITCH_INIT_SUCCESS) + "\n");
    
    syncedFramesBufferForProc.clear();
    procFrameBufferForShow.clear();
    procFrameBufferForSave.clear();
    procFrameBufferForSend.clear();

    procFrameBufferForShow.setMaxSize(8);
    procFrameBufferForSend.setMaxSize(8);
    procFrameBufferForSave.setMaxSize(8);

    renderEndFlag = 0;
    renderThreadJoined = 0;
    renderThread.reset(new std::thread(&PanoramaLiveStreamTask2::Impl::procVideo, this));

    appendLog(getText(TI_STITCH_TASK_LAUNCH_SUCCESS) + "\n");

    return true;
}

void PanoramaLiveStreamTask2::Impl::stopVideoStitch()
{
    if (renderPrepareSuccess && !renderThreadJoined)
    {
        renderEndFlag = 1;
        syncedFramesBufferForProc.stop();
        renderThread->join();
        renderThread.reset(0);
        render.reset();
        correct.clear();
        renderPrepareSuccess = 0;
        renderThreadJoined = 1;
        procFramePool.clear();

        appendLog(getText(TI_STITCH_TASK_FINISH) + "\n");
    }
}

bool PanoramaLiveStreamTask2::Impl::getVideoSourceFrames(std::vector<avp::AudioVideoFrame2>& frames)
{
    if (allowGetSyncedFramesBufferForShow)
        return syncedFramesBufferForShow.pull(frames);
    else
        return false;
}

bool PanoramaLiveStreamTask2::Impl::getStitchedVideoFrame(avp::AudioVideoFrame2& frame)
{
    CudaMixedAudioVideoFrame mixedFrame;
    bool ok = procFrameBufferForShow.pull(mixedFrame);
    frame = mixedFrame.frame;
    return ok;
}

void PanoramaLiveStreamTask2::Impl::cancelGetVideoSourceFrames()
{
    //syncedFramesBufferForShow.stop();
}

void PanoramaLiveStreamTask2::Impl::cancelGetStitchedVideoFrame()
{
    //procFrameBufferForShow.stop();
}

bool PanoramaLiveStreamTask2::Impl::openLiveStream(const std::string& name, int panoType,
    int width, int height, int videoBPS, const std::string& videoEncoder, const std::string& videoPreset, int audioBPS)
{
    if (!videoOpenSuccess || !audioVideoSource)
    {
        ptlprintf("Error in %s, audio video sources not opened\n", __FUNCTION__);
        syncErrorMessage = getText(TI_SOURCE_NOT_OPENED_CANNOT_LAUNCH_LIVE);
        return false;
    }

    if (!renderPrepareSuccess || renderThreadJoined)
    {
        ptlprintf("Error in %s, render not running, cannot launch live streaming\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_NOT_RUNNING_CANNOT_LAUNCH_LIVE);
        return false;
    }

    if (!streamThreadJoined)
    {
        ptlprintf("Error in %s, live streaming running, stop before launching new live streaming\n", __FUNCTION__);
        syncErrorMessage = getText(TI_LIVE_RUNNING_CLOSE_BEFORE_LAUNCH_NEW);
        return false;
    }

    panoType = (panoType < 0 || panoType >= PanoTypeCount) ? PanoTypeEquiRect : panoType;
    if ((panoType == PanoTypeEquiRect && width != 2 * height) ||
        (panoType == PanoTypeCube6x1 && width != 6 * height) ||
        (panoType == PanoTypeCube3x2 && width * 2 != 3 * height) ||
        (panoType == PanoTypeCube180 && width * 3 != 5 * height))
    {
        ptlprintf("Error in %s, panoType(%d), width(%d), height(%d) not satisfy requirements\n",
            __FUNCTION__, panoType, width, height);
        syncErrorMessage = getText(TI_LIVE_PARAM_ERROR_CANNOT_LAUNCH_LIVE);
        return false;
    }

    if (panoType != PanoTypeEquiRect)
    {
        cv::Mat xmap, ymap;
        if (panoType == PanoTypeCube6x1)
            getEquiRectToCubeMap(xmap, ymap, renderFrameSize.height, height, CubeType6x1);
        else if (panoType == PanoTypeCube3x2)
            getEquiRectToCubeMap(xmap, ymap, renderFrameSize.height, height / 2, CubeType3x2);
        else if (panoType == PanoTypeCube180)
            getEquiRectToCubeMap(xmap, ymap, renderFrameSize.height, height * 2 / 3, CubeType180);
        
        streamXMap.upload(xmap);
        streamYMap.upload(ymap);
    }

    streamIsLibX264 = (videoEncoder == "h264" || videoEncoder == "libx264") ? 1 : 0;
    sendFramePool.init(streamIsLibX264 ? avp::PixelTypeYUV420P : avp::PixelTypeNV12, width, height);

    streamURL = name;
    streamPanoType = panoType;
    streamFrameSize.width = width;
    streamFrameSize.height = height;
    streamVideoBitRate = videoBPS;
    streamVideoEncodePreset = videoPreset;
    streamAudioBitRate = audioBPS;
    if (streamVideoEncodePreset != "ultrafast" && streamVideoEncodePreset != "superfast" &&
        streamVideoEncodePreset != "veryfast" && streamVideoEncodePreset != "faster" &&
        streamVideoEncodePreset != "fast" && streamVideoEncodePreset != "medium" && streamVideoEncodePreset != "slow" &&
        streamVideoEncodePreset != "slower" && streamVideoEncodePreset != "veryslow")
        streamVideoEncodePreset = "veryfast";

    std::vector<avp::Option> writerOpts;
    writerOpts.push_back(std::make_pair("preset", streamVideoEncodePreset));
    writerOpts.push_back(std::make_pair("profile", "main"));
    streamOpenSuccess = streamWriter.open(streamURL, streamURL.substr(0, 4) == "rtmp" ? "flv" : "rtsp", true,
        audioOpenSuccess, "aac", audioVideoSource->getAudioSampleType(),
        audioVideoSource->getAudioChannelLayout(), audioVideoSource->getAudioSampleRate(), streamAudioBitRate,
        true, (videoEncoder == "h264_qsv" || videoEncoder == "nvenc_h264") ? videoEncoder : "h264", 
        streamIsLibX264 ? avp::PixelTypeYUV420P : avp::PixelTypeNV12, streamFrameSize.width, streamFrameSize.height,
        videoFrameRate, streamVideoBitRate, writerOpts);
    if (!streamOpenSuccess)
    {
        ptlprintf("Error in %s, Could not open streaming url with frame rate = %f and bit rate = %d\n", 
            __FUNCTION__, videoFrameRate, streamVideoBitRate);
        appendLog(getText(TI_SERVER_CONNECT_FAIL) + "\n");
        syncErrorMessage = getText(TI_SERVER_CONNECT_FAIL);
        return false;
    }

    ptlprintf("Info in %s, live stream params: url %s, pano type %d, width %d, height %d\n",
        __FUNCTION__, streamURL.c_str(), streamPanoType, streamFrameSize.width, streamFrameSize.height);
    appendLog(getText(TI_SERVER_CONNECT_SUCCESS) + "\n");

    procFrameBufferForSend.resume();
    streamEndFlag = 0;
    streamThreadJoined = 0;
    streamThread.reset(new std::thread(&PanoramaLiveStreamTask2::Impl::streamSend, this));

    appendLog(getText(TI_LIVE_TASK_LAUNCH_SUCCESS) + "\n");

    return true;
}

void PanoramaLiveStreamTask2::Impl::closeLiveStream()
{
    if (streamOpenSuccess && !streamThreadJoined)
    {
        streamEndFlag = 1;
        procFrameBufferForSend.stop();
        streamThread->join();
        streamThread.reset(0);
        streamThreadJoined = 1;
        streamOpenSuccess = 0;
        sendFramePool.clear();

        appendLog(getText(TI_LIVE_TASK_FINISH) + "\n");
        appendLog(getText(TI_SERVER_DISCONNECT) + "\n");
    }
}

bool PanoramaLiveStreamTask2::Impl::beginSaveToDisk(const std::string& dir, int panoType, int width, int height, int videoBPS,
    const std::string& videoEncoder, const std::string& videoPreset, int audioBPS, int fileDurationInSeconds)
{
    if (!videoOpenSuccess || !audioVideoSource)
    {
        ptlprintf("Error in %s, audio video sources not opened\n", __FUNCTION__);
        syncErrorMessage = getText(TI_SOURCE_NOT_OPENED_CANNOT_LAUNCH_WRITE);
        return false;
    }

    if (!renderPrepareSuccess || renderThreadJoined)
    {
        ptlprintf("Error in %s, render not running, cannot launch saving to disk\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_NOT_RUNNING_CANNOT_LAUNCH_WRITE);
        return false;
    }

    if (!fileThreadJoined)
    {
        ptlprintf("Error in %s, saving to disk running, stop before launching new saving to disk\n", __FUNCTION__);
        syncErrorMessage = getText(TI_WRITE_RUNNING_CLOSE_BEFORE_LAUNCH_NEW);
        return false;
    }

    panoType = (panoType < 0 || panoType >= PanoTypeCount) ? PanoTypeEquiRect : panoType;
    if ((panoType == PanoTypeEquiRect && width != 2 * height) ||
        (panoType == PanoTypeCube6x1 && width != 6 * height) ||
        (panoType == PanoTypeCube3x2 && width * 2 != 3 * height) ||
        (panoType == PanoTypeCube180 && width * 3 != 5 * height))
    {
        ptlprintf("Error in %s, panoType(%d), width(%d), height(%d) not satisfy requirements\n",
            __FUNCTION__, panoType, width, height);
        syncErrorMessage = getText(TI_WRITE_PARAM_ERROR_CANNOT_LAUNCH_WRITE);
        return false;
    }

    if (panoType != PanoTypeEquiRect)
    {
        cv::Mat xmap, ymap;
        if (panoType == PanoTypeCube6x1)
            getEquiRectToCubeMap(xmap, ymap, renderFrameSize.height, height, CubeType6x1);
        else if (panoType == PanoTypeCube3x2)
            getEquiRectToCubeMap(xmap, ymap, renderFrameSize.height, height / 2, CubeType3x2);
        else if (panoType == PanoTypeCube180)
            getEquiRectToCubeMap(xmap, ymap, renderFrameSize.height, height * 2 / 3, CubeType180);

        fileXMap.upload(xmap);
        fileYMap.upload(ymap);
    }

    fileIsLibX264 = (fileVideoEncoder == "h264" || fileVideoEncoder == "libx264") ? 1 : 0;
    saveFramePool.init(fileIsLibX264 ? avp::PixelTypeYUV420P : avp::PixelTypeNV12, width, height);

    fileDir = dir;
    if (fileDir.back() != '\\' && fileDir.back() != '/')
        fileDir.append("/");
    filePanoType = panoType;
    fileFrameSize.width = width;
    fileFrameSize.height = height;
    fileVideoBitRate = videoBPS;
    fileVideoEncoder = videoEncoder;
    fileVideoEncodePreset = videoPreset;
    fileAudioBitRate = audioBPS;
    fileDuration = fileDurationInSeconds;
    if (fileVideoEncoder != "h264" && fileVideoEncoder != "h264_qsv" && fileVideoEncoder != "nvenc_h264")
        fileVideoEncoder = "h264";
    if (fileVideoEncodePreset != "ultrafast" && fileVideoEncodePreset != "superfast" &&
        fileVideoEncodePreset != "veryfast" && fileVideoEncodePreset != "faster" &&
        fileVideoEncodePreset != "fast" && fileVideoEncodePreset != "medium" && fileVideoEncodePreset != "slow" &&
        fileVideoEncodePreset != "slower" && fileVideoEncodePreset != "veryslow")
        fileVideoEncodePreset = "veryfast";
    fileConfigSet = 1;

    ptlprintf("Info in %s, save to disk params: dir %s, pano type %d, width %d, height %d\n",
        __FUNCTION__, fileDir.c_str(), filePanoType, fileFrameSize.width, fileFrameSize.height);

    procFrameBufferForSave.resume();
    fileEndFlag = 0;
    fileThreadJoined = 0;
    fileThread.reset(new std::thread(&PanoramaLiveStreamTask2::Impl::fileSave, this));

    appendLog(getText(TI_WRITE_LAUNCH) + "\n");

    return true;
}

void PanoramaLiveStreamTask2::Impl::stopSaveToDisk()
{
    if (fileConfigSet && !fileThreadJoined)
    {
        fileEndFlag = 1;
        procFrameBufferForSave.stop();
        fileThread->join();
        fileThread.reset(0);
        fileThreadJoined = 1;
        fileConfigSet = 0;
        saveFramePool.clear();

        appendLog(getText(TI_WRITE_FINISH) + "\n");
    }
}

bool PanoramaLiveStreamTask2::Impl::calcExposures(std::vector<double>& expos)
{
    expos.clear();

    if (!videoOpenSuccess || !audioVideoSource)
    {
        ptlprintf("Error in %s, audio video sources not opened\n", __FUNCTION__);
        syncErrorMessage = getText(TI_SOURCE_NOT_OPENED_CANNOT_CORRECT);
        return false;
    }

    if (!renderPrepareSuccess || renderThreadJoined)
    {
        ptlprintf("Error in %s, render not running, cannot apply exposure correct\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_NOT_RUNNING_CANNOT_CORRECT);
        return false;
    }

    if (!streamThreadJoined)
    {
        ptlprintf("Error in %s, live streaming running, stop before applying exposure correct\n", __FUNCTION__);
        syncErrorMessage = getText(TI_LIVE_RUNNING_CLOSE_BEFORE_CORRECT);
        return false;
    }

    if (!fileThreadJoined)
    {
        ptlprintf("Error in %s, saving to disk running, stop before applying exposure correct\n", __FUNCTION__);
        syncErrorMessage = getText(TI_WRITE_RUNNING_CLOSE_BEFORE_CORRECT);
        return false;
    }

    // Notice, we use the frames in syncedFramesBufferForShow,
    // so we first disable other threads from getting frames from this buffer,
    // and then try to get frames from it for several times
    allowGetSyncedFramesBufferForShow = 0;
    std::vector<avp::AudioVideoFrame2> frames;
    int maxPullFrameTimes = 32;
    bool success = false;
    for (int i = 0; i < maxPullFrameTimes; i++)
    {
        if (syncedFramesBufferForShow.pull(frames))
        {
            success = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    allowGetSyncedFramesBufferForShow = 1;
    if (!success)
    {
        ptlprintf("Error in %s, could not get frames from synced frames buffer for show, "
            "so exposure correct failed\n", __FUNCTION__);
        syncErrorMessage = getText(TI_CORRECT_FAIL);
        return false;
    }

    std::vector<cv::Mat> images(numVideos);
    for (int i = 0; i < numVideos; i++)
        images[i] = cv::Mat(frames[i].height, frames[i].width, elemType, frames[i].data[0], frames[i].steps[0]);

    bool ok = correct.correct(images, exposures);
    if (!ok)
    {
        syncErrorMessage = getText(TI_CORRECT_FAIL);
        resetExposures();
        ptlprintf("Error in %s, exposure correct failed\n", __FUNCTION__);
        return false;
    }

    expos = exposures;
    setLUTs(exposures);
    return true;
}

bool PanoramaLiveStreamTask2::Impl::setExposures(const std::vector<double>& expos)
{
    if (expos.size() != numVideos)
    {
        ptlprintf("Error in %s, input exposures vector length invalid\n", __FUNCTION__);
        return false;
    }

    exposures = expos;
    setLUTs(exposures);
    return true;
}

void PanoramaLiveStreamTask2::Impl::resetExposures()
{
    exposures.clear();
    clearLUTs();
}

double PanoramaLiveStreamTask2::Impl::getVideoSourceFrameRate() const
{
    return videoSourceFrameRate;
}

double PanoramaLiveStreamTask2::Impl::getStitchFrameRate() const
{
    return stitchVideoFrameRate;
}

void PanoramaLiveStreamTask2::Impl::getLastSyncErrorMessage(std::string& message) const
{
    message = syncErrorMessage;
}

bool PanoramaLiveStreamTask2::Impl::hasAsyncErrorMessage() const
{
    return hasAsyncError;
}

void PanoramaLiveStreamTask2::Impl::getLastAsyncErrorMessage(std::string& message, int& fromWhere)
{
    if (hasAsyncError)
    {
        AsyncErrorMessage msg;
        asyncErrorMessageQueue.pull(msg);
        fromWhere = msg.first;
        message = msg.second;
        if (!asyncErrorMessageQueue.size())
            hasAsyncError = 0;
    }
    else
    {
        message.clear();
        fromWhere = ErrorNone;
    }
}

void PanoramaLiveStreamTask2::Impl::getLog(std::string& logInfo)
{
    std::lock_guard<std::mutex> lg(mtxLog);
    if (log.empty())
    {
        logInfo.clear();
        return;
    }
    logInfo = log.substr(0, log.size() - 1);
    log.clear();
}

void PanoramaLiveStreamTask2::Impl::initAll()
{
    videoSourceFrameRate = 0;
    stitchVideoFrameRate = 0;
    syncErrorMessage.clear();
    clearAsyncErrorMessages();
    clearLog();

    videoFrameRate = 0;
    numVideos = 0;
    videoOpenSuccess = 0;

    audioSampleRate = 0;
    audioOpenSuccess = 0;

    renderPrepareSuccess = 0;
    renderEndFlag = 0;
    renderThreadJoined = 1;

    streamVideoBitRate = 0;
    streamAudioBitRate = 0;
    streamOpenSuccess = 0;
    streamEndFlag = 0;
    streamThreadJoined = 1;

    fileVideoBitRate = 0;
    fileAudioBitRate = 0;
    fileDuration = 0;
    fileConfigSet = 0;
    fileEndFlag = 0;
    fileThreadJoined = 1;

    finish = 0;
    allowGetSyncedFramesBufferForShow = 1;
    pixelType = avp::PixelTypeBGR32;
    elemType = CV_8UC4;
}

void PanoramaLiveStreamTask2::Impl::closeAll()
{
    closeAudioVideoSources();
    stopVideoStitch();
    closeLiveStream();
    stopSaveToDisk();

    finish = 1;

    //ptlprintf("Live stream task's all threads closed\n");

    syncedFramesBufferForShow.clear();
    syncedFramesBufferForProc.clear();
    procFramePool.clear();
    procFrameBufferForShow.clear();
    procFrameBufferForSend.clear();
    procFrameBufferForSave.clear();

    //ptlprintf("Live stream task's all buffer cleared\n");
}

bool PanoramaLiveStreamTask2::Impl::hasFinished() const
{
    return finish;
}

void PanoramaLiveStreamTask2::Impl::procVideo()
{
    size_t id = std::this_thread::get_id().hash();
    ptlprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    std::vector<cv::cuda::HostMem> mems;
    std::vector<long long int> timeStamps;
    std::vector<cv::Mat> src(numVideos);
    cv::cuda::GpuMat bgr32;
    std::vector<std::vector<unsigned char> > localLookUpTables;
    CudaMixedAudioVideoFrame renderFrame, sendFrame, saveFrame;
    cv::cuda::GpuMat bgr1, y1, u1, v1, uv1;
    cv::cuda::GpuMat bgr2, y2, u2, v2, uv2;
    bool ok;
    int roundedFrameRate = videoFrameRate + 0.5;
    int count = -1;
    ztool::Timer timer;
    while (true)
    {
        ztool::Timer localTimer, procTimer;
        if (finish || renderEndFlag)
            break;
        //ptlprintf("show\n");
        if (!syncedFramesBufferForProc.pull(mems, timeStamps))
        {
            //std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        //ptlprintf("ts %lld\n", timeStamp);
        //ptlprintf("before check size\n");
        // NOTICE: it would be better to check frames's pixelType and other properties.
        if (mems.size() == numVideos)
        {
            //ztool::Timer localTimer, procTimer;
            if (count < 0)
            {
                count = 0;
                timer.start();
            }
            else
            {
                count++;
                timer.end();
                double elapse = timer.elapse();
                if ((elapse >= 1 && count >= 2) || count == roundedFrameRate)
                {
                    double r = count / elapse;
                    printf("%d  %f, %f\n", count, elapse, r);
                    timer.start();
                    count = 0;
                    stitchVideoFrameRate = r;
                }
            }

            for (int i = 0; i < numVideos; i++)
                src[i] = mems[i].createMatHeader();
            if (luts.size())
            {
                getLuts(localLookUpTables);
                //procTimer.start();
                ok = render->render(src, bgr32, localLookUpTables);
                //procTimer.end();
            }
            else
                ok = render->render(src, bgr32);
            if (!ok)
            {
                ptlprintf("Error in %s [%8x], render failed\n", __FUNCTION__, id);
                addAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE), ErrorFromStitch);
                finish = 1;
                break;
            }

            if (addWatermark)
            {
                ok = watermarkFilter.addWatermark(bgr32);
                if (!ok)
                {
                    ptlprintf("Error in %s [%8x], render failed\n", __FUNCTION__, id);
                    addAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE), ErrorFromStitch);
                    finish = 1;
                    break;
                }
            }

            ok = procFramePool.get(renderFrame);
            if (!ok)
            {
                ptlprintf("Error in %s [%8x], could not get cpu memory to copy from gpu\n", __FUNCTION__, id);
                addAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE), ErrorFromStitch);
                finish = 1;
                break;
            }
            if (renderFrame.frame.width != renderFrameSize.width ||
                renderFrame.frame.height != renderFrameSize.height ||
                renderFrame.frame.pixelType != avp::PixelTypeBGR32)
            {
                ptlprintf("Error in %s [%8x], render frame obtained from proc frame pool not satisfied, "
                    "width = %d, height = %d, pixel type = %d, requires width = %d, height = %d, pixel type = %d\n",
                    __FUNCTION__, id, renderFrame.frame.width, renderFrame.frame.height, renderFrame.frame.pixelType,
                    renderFrameSize.width, renderFrameSize.height, avp::PixelTypeBGR32);
                addAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE), ErrorFromStitch);
                finish = 1;
                break;
            }
            cv::Mat bgr = renderFrame.planes[0].createMatHeader();
            bgr32.download(bgr);
            renderFrame.frame.timeStamp = timeStamps[0];
            procFrameBufferForShow.push(renderFrame);

            if (streamOpenSuccess && sendFramePool.get(sendFrame))
            {
                if (sendFrame.frame.width != streamFrameSize.width ||
                    sendFrame.frame.height != streamFrameSize.height ||
                    (streamIsLibX264 ? (sendFrame.frame.pixelType != avp::PixelTypeYUV420P) :
                                       (sendFrame.frame.pixelType != avp::PixelTypeNV12)))
                {
                    ptlprintf("Error in %s [%8x], send frame obtained from send frame pool not satisfied, "
                        "width = %d, height = %d, pixel type = %d, requires width = %d, height = %d, pixel type = %d\n",
                        __FUNCTION__, id, sendFrame.frame.width, sendFrame.frame.height, sendFrame.frame.pixelType,
                        streamFrameSize.width, streamFrameSize.height, streamIsLibX264 ? avp::PixelTypeYUV420P : avp::PixelTypeNV12);

                    addAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE), ErrorFromStitch);
                    finish = 1;
                    break;
                }

                if (streamPanoType == PanoTypeEquiRect)
                {
                    if (streamFrameSize == renderFrameSize)
                        bgr1 = bgr32;
                    else
                        resize8UC4(bgr32, bgr1, streamFrameSize);
                }
                else
                    cudaReproject(bgr32, bgr1, streamXMap, streamYMap);

                if (streamIsLibX264)
                {
                    cvtBGR32ToYUV420P(bgr1, y1, u1, v1);
                    cv::Mat yy = sendFrame.planes[0].createMatHeader();
                    cv::Mat uu = sendFrame.planes[1].createMatHeader();
                    cv::Mat vv = sendFrame.planes[2].createMatHeader();
                    y1.download(yy);
                    u1.download(uu);
                    v1.download(vv);
                }
                else
                {
                    cvtBGR32ToNV12(bgr1, y1, uv1);
                    cv::Mat yy = sendFrame.planes[0].createMatHeader();
                    cv::Mat uvuv = sendFrame.planes[1].createMatHeader();
                    y1.download(yy);
                    uv1.download(uvuv);
                }
                sendFrame.frame.timeStamp = timeStamps[0];
                procFrameBufferForSend.push(sendFrame);
            }

            if (fileConfigSet && saveFramePool.get(saveFrame))
            {
                if (saveFrame.frame.width != fileFrameSize.width ||
                    saveFrame.frame.height != fileFrameSize.height ||
                    (fileIsLibX264 ? (saveFrame.frame.pixelType != avp::PixelTypeYUV420P) :
                                     (saveFrame.frame.pixelType != avp::PixelTypeNV12)))
                {
                    ptlprintf("Error in %s [%8x], save frame obtained from save frame pool not satisfied, "
                        "width = %d, height = %d, pixel type = %d, requires width = %d, height = %d, pixel type = %d\n",
                        __FUNCTION__, id, saveFrame.frame.width, saveFrame.frame.height, saveFrame.frame.pixelType,
                        fileFrameSize.width, fileFrameSize.height, fileIsLibX264 ? avp::PixelTypeYUV420P : avp::PixelTypeNV12);

                    addAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE), ErrorFromStitch);
                    finish = 1;
                    break;
                }

                if (filePanoType == PanoTypeEquiRect)
                {
                    if (fileFrameSize == renderFrameSize)
                        bgr2 = bgr32;
                    else
                        resize8UC4(bgr32, bgr2, fileFrameSize);
                }
                else
                    cudaReproject(bgr32, bgr2, fileXMap, fileYMap);

                if (fileIsLibX264)
                {
                    cvtBGR32ToYUV420P(bgr2, y2, u2, v2);
                    cv::Mat yy = saveFrame.planes[0].createMatHeader();
                    cv::Mat uu = saveFrame.planes[1].createMatHeader();
                    cv::Mat vv = saveFrame.planes[2].createMatHeader();
                    y2.download(yy);
                    u2.download(uu);
                    v2.download(vv);
                }
                else
                {
                    cvtBGR32ToNV12(bgr2, y2, uv2);
                    cv::Mat yy = saveFrame.planes[0].createMatHeader();
                    cv::Mat uvuv = saveFrame.planes[1].createMatHeader();
                    y2.download(yy);
                    uv2.download(uvuv);
                }
                saveFrame.frame.timeStamp = timeStamps[0];
                procFrameBufferForSave.push(saveFrame);
            }

            localTimer.end();
            //ptlprintf("%f, %f\n", procTimer.elapse(), localTimer.elapse());
        }
    }

    procFrameBufferForSend.stop();
    procFrameBufferForSave.stop();

    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask2::Impl::streamSend()
{
    size_t id = std::this_thread::get_id().hash();
    ptlprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    cv::Mat dstMat;
    CudaMixedAudioVideoFrame frame;
    while (true)
    {
        if (finish || streamEndFlag)
            break;
        procFrameBufferForSend.pull(frame);
        // IMPORTANT NOTICE!!!
        // We should check the frame width and height, frame size may change from one live stream task to another.
        // But I DO NOT CLEAR THE BUFFER in the procVideo thread, since it is very difficult.
        // Examining the frame size before sending it to the writer is much more easy.
        // The frame which size does not match is discarded.
        if (frame.frame.data[0] && 
            (frame.frame.mediaType == avp::AUDIO || 
             (frame.frame.mediaType == avp::VIDEO && 
              (streamIsLibX264 ? (frame.frame.pixelType == avp::PixelTypeYUV420P) : (frame.frame.pixelType == avp::PixelTypeNV12)) &&
              frame.frame.width == streamFrameSize.width && frame.frame.height == streamFrameSize.height)))
        {
            bool ok = streamWriter.write(frame.frame);
            if (!ok)
            {
                ptlprintf("Error in %s [%8x], cannot write frame\n", __FUNCTION__, id);
                addAsyncErrorMessage(getText(TI_LIVE_FAIL_TASK_TERMINATE), ErrorFromLiveStream);
                finish = 1;
                break;
            }
        }
    }
    streamWriter.close();

    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask2::Impl::fileSave()
{
    if (!fileConfigSet)
        return;

    size_t id = std::this_thread::get_id().hash();
    ptlprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    time_t rawTime;    
    tm* ptrTm;

    char buf[1024], shortNameBuf[1024];
    int count = 0;
    cv::Mat dstMat;
    CudaMixedAudioVideoFrame frame;
    avp::AudioVideoWriter3 writer;
    std::vector<avp::Option> writerOpts;
    writerOpts.push_back(std::make_pair("preset", fileVideoEncodePreset));
    time(&rawTime);
    ptrTm = localtime(&rawTime);
    strftime(shortNameBuf, 1024, "%Y-%m-%d-%H-%M-%S.mp4", ptrTm);
    sprintf(buf, "%s%s", fileDir.c_str(), shortNameBuf);
    bool ok = writer.open(buf, "mp4", true,
        audioOpenSuccess, "aac", audioVideoSource->getAudioSampleType(),
        audioVideoSource->getAudioChannelLayout(), audioVideoSource->getAudioSampleRate(), fileAudioBitRate,
        true, fileVideoEncoder, fileIsLibX264 ? avp::PixelTypeYUV420P : avp::PixelTypeNV12, 
        fileFrameSize.width, fileFrameSize.height, videoFrameRate, fileVideoBitRate, writerOpts);
    if (!ok)
    {
        ptlprintf("Error in %s [%8x], could not save current audio video\n", __FUNCTION__, id);
        appendLog(std::string(buf) + " " + getText(TI_FILE_OPEN_FAIL_TASK_TERMINATE) + "\n");
        addAsyncErrorMessage(std::string(buf) + " " + getText(TI_FILE_OPEN_FAIL_TASK_TERMINATE), ErrorFromSaveToDisk);
        return;
    }
    else
        appendLog(std::string(buf) + " " + getText(TI_BEGIN_WRITE) + "\n");
    long long int fileFirstTimeStamp = -1;
    while (true)
    {
        if (finish || fileEndFlag)
            break;
        procFrameBufferForSave.pull(frame);
        //printf("pass pull frame\n");
        // IMPORTANT NOTICE!!!
        // We should check the frame width and height, frame size may change from one saving to disk task to another.
        // But I DO NOT CLEAR THE BUFFER in the procVideo thread, since it is very difficult.
        // Examining the frame size before sending it to the writer is much more easy.
        // The frame which size does not match is discarded.
        if (frame.frame.data[0] && 
            (frame.frame.mediaType == avp::AUDIO || 
             (frame.frame.mediaType == avp::VIDEO && 
              (fileIsLibX264 ? (frame.frame.pixelType == avp::PixelTypeYUV420P) : (frame.frame.pixelType == avp::PixelTypeNV12)) &&
              frame.frame.width == fileFrameSize.width && frame.frame.height == fileFrameSize.height)))
        {
            if (fileFirstTimeStamp < 0)
                fileFirstTimeStamp = frame.frame.timeStamp;

            if (frame.frame.timeStamp - fileFirstTimeStamp > fileDuration * 1000000LL)
            {
                writer.close();
                appendLog(std::string(buf) + " " + getText(TI_END_WRITE) + "\n");

                time(&rawTime);
                ptrTm = localtime(&rawTime);
                strftime(shortNameBuf, 1024, "%Y-%m-%d-%H-%M-%S.mp4", ptrTm);
                sprintf(buf, "%s%s", fileDir.c_str(), shortNameBuf);
                ok = writer.open(buf, "mp4", true,
                    audioOpenSuccess, "aac", audioVideoSource->getAudioSampleType(),
                    audioVideoSource->getAudioChannelLayout(), audioVideoSource->getAudioSampleRate(), fileAudioBitRate,
                    true, fileVideoEncoder, fileIsLibX264 ? avp::PixelTypeYUV420P : avp::PixelTypeNV12, 
                    fileFrameSize.width, fileFrameSize.height, videoFrameRate, fileVideoBitRate, writerOpts);
                if (!ok)
                {
                    ptlprintf("Error in %s [%8x], could not save current audio video\n", __FUNCTION__, id);
                    appendLog(std::string(buf) + " " + getText(TI_FILE_OPEN_FAIL_TASK_TERMINATE) + "\n");
                    break;
                }
                else
                    appendLog(std::string(buf) + " " + getText(TI_BEGIN_WRITE) + "\n");
                fileFirstTimeStamp = frame.frame.timeStamp;
            }

            ok = writer.write(frame.frame);
            if (!ok)
            {
                ptlprintf("Error in %s [%8x], could not write current frame\n", __FUNCTION__, id);
                addAsyncErrorMessage(std::string(buf) + " " + getText(TI_WRITE_FAIL_TASK_TERMINATE), ErrorFromSaveToDisk);
                break;
            }
        }
        else
        {
            //ptlprintf("Frame error\n");
        }
    }
    if (ok)
        appendLog(std::string(buf) + " " + getText(TI_END_WRITE) + "\n");
    writer.close();

    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask2::Impl::setLUTs(const std::vector<double>& exposures)
{
    std::lock_guard<std::mutex> lg(mtxLuts);
    ExposureColorCorrect::getExposureLUTs(exposures, luts);
}

void PanoramaLiveStreamTask2::Impl::clearLUTs()
{
    std::lock_guard<std::mutex> lg(mtxLuts);
    luts.clear();
}

void PanoramaLiveStreamTask2::Impl::getLuts(std::vector<std::vector<unsigned char> >& LUTs)
{
    std::lock_guard<std::mutex> lg(mtxLuts);
    LUTs = luts;
}

void PanoramaLiveStreamTask2::Impl::addAsyncErrorMessage(const std::string& message, int fromWhere)
{
    hasAsyncError = 1;
    asyncErrorMessageQueue.push(std::make_pair(fromWhere, message));
}

void PanoramaLiveStreamTask2::Impl::clearAsyncErrorMessages()
{
    hasAsyncError = 0;
    asyncErrorMessageQueue.clear();
}

void PanoramaLiveStreamTask2::Impl::appendLog(const std::string& msg)
{
    std::lock_guard<std::mutex> lg(mtxLog);
    log.append(msg);
}

void PanoramaLiveStreamTask2::Impl::clearLog()
{
    std::lock_guard<std::mutex> lg(mtxLog);
    log.clear();
}

PanoramaLiveStreamTask2::PanoramaLiveStreamTask2()
{
    ptrImpl.reset(new Impl);
}

PanoramaLiveStreamTask2::~PanoramaLiveStreamTask2()
{

}

bool PanoramaLiveStreamTask2::openAudioVideoSources(const std::vector<avp::Device>& devices, int width, int height, int frameRate,
    bool openAudio, const avp::Device& device, int sampleRate)
{
    return ptrImpl->openAudioVideoSources(devices, width, height, frameRate,
        openAudio, device, sampleRate);
}

bool PanoramaLiveStreamTask2::openAudioVideoSources(const std::vector<std::string>& urls, bool openAudio, const std::string& url)
{
    return ptrImpl->openAudioVideoSources(urls, openAudio, url);
}

void PanoramaLiveStreamTask2::closeAudioVideoSources()
{
    ptrImpl->closeAudioVideoSources();
}

bool PanoramaLiveStreamTask2::beginVideoStitch(int panoStitchType, const std::string& configFileName, 
    int width, int height, bool highQualityBlend)
{
    return ptrImpl->beginVideoStitch(panoStitchType, configFileName, width, height, highQualityBlend);
}

void PanoramaLiveStreamTask2::stopVideoStitch()
{
    ptrImpl->stopVideoStitch();
}

bool PanoramaLiveStreamTask2::openLiveStream(const std::string& name, int panoType, int width, int height, int videoBPS,
    const std::string& videoEncoder, const std::string& videoPreset, int audioBPS)
{
    return ptrImpl->openLiveStream(name, panoType, width, height, 
        videoBPS, videoEncoder, videoPreset, audioBPS);
}

void PanoramaLiveStreamTask2::closeLiveStream()
{
    ptrImpl->closeLiveStream();
}

bool PanoramaLiveStreamTask2::beginSaveToDisk(const std::string& dir, int panoType, int width, int height, int videoBPS,
    const std::string& videoEncoder, const std::string& videoPreset, int audioBPS, int fileDuration)
{
    return ptrImpl->beginSaveToDisk(dir, panoType, width, height, 
        videoBPS, videoEncoder, videoPreset, audioBPS, fileDuration);
}

void PanoramaLiveStreamTask2::stopSaveToDisk()
{
    ptrImpl->stopSaveToDisk();
}

bool PanoramaLiveStreamTask2::calcExposures(std::vector<double>& exposures)
{
    return ptrImpl->calcExposures(exposures);
}

bool PanoramaLiveStreamTask2::setExposures(const std::vector<double>& exposures)
{
    return ptrImpl->setExposures(exposures);
}

void PanoramaLiveStreamTask2::resetExposures()
{
    ptrImpl->resetExposures();
}

double PanoramaLiveStreamTask2::getVideoSourceFrameRate() const
{
    return ptrImpl->getVideoSourceFrameRate();
}

double PanoramaLiveStreamTask2::getStitchFrameRate() const
{
    return ptrImpl->getStitchFrameRate();
}

void PanoramaLiveStreamTask2::getLastSyncErrorMessage(std::string& message) const
{
    ptrImpl->getLastSyncErrorMessage(message);
}

bool PanoramaLiveStreamTask2::hasAsyncErrorMessage() const
{
    return ptrImpl->hasAsyncErrorMessage();
}

void PanoramaLiveStreamTask2::getLastAsyncErrorMessage(std::string& message, int& fromWhere)
{
    return ptrImpl->getLastAsyncErrorMessage(message, fromWhere);
}

void PanoramaLiveStreamTask2::getLog(std::string& logInfo)
{
    ptrImpl->getLog(logInfo);
}

bool PanoramaLiveStreamTask2::getVideoSourceFrames(std::vector<avp::AudioVideoFrame2>& frames)
{
    return ptrImpl->getVideoSourceFrames(frames);
}

bool PanoramaLiveStreamTask2::getStitchedVideoFrame(avp::AudioVideoFrame2& frame)
{
    return ptrImpl->getStitchedVideoFrame(frame);
}

void PanoramaLiveStreamTask2::cancelGetVideoSourceFrames()
{
    return ptrImpl->cancelGetVideoSourceFrames();
}

void PanoramaLiveStreamTask2::cancelGetStitchedVideoFrame()
{
    return ptrImpl->cancelGetStitchedVideoFrame();
}

void PanoramaLiveStreamTask2::initAll()
{
    ptrImpl->initAll();
}

void PanoramaLiveStreamTask2::closeAll()
{
    ptrImpl->closeAll();
}

bool PanoramaLiveStreamTask2::hasFinished() const
{
    return ptrImpl->hasFinished();
}

int PanoramaLiveStreamTask2::getNumVideos() const
{
    return ptrImpl->numVideos;
}

int PanoramaLiveStreamTask2::getVideoWidth() const
{
    return ptrImpl->videoFrameSize.width;
}

int PanoramaLiveStreamTask2::getVideoHeight() const
{
    return ptrImpl->videoFrameSize.height;
}

double PanoramaLiveStreamTask2::getVideoFrameRate() const
{
    return ptrImpl->videoFrameRate;
}

int PanoramaLiveStreamTask2::getAudioSampleRate() const
{
    return ptrImpl->audioSampleRate;
}