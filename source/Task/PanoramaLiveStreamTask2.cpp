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
#include "opencv2/cudawarping.hpp"

struct PanoramaLiveStreamTask2::Impl
{
    Impl();
    ~Impl();

    bool openAudioVideoSources(const std::vector<avp::Device>& devices, int width, int height, int frameRate,
        bool openAudio, const avp::Device& device, int sampleRate);
    bool openAudioVideoSources(const std::vector<std::string>& urls, bool openAudio, const std::string& url);
    void closeAudioVideoSources();

    bool beginVideoStitch(const std::string& configFileName, int width, int height, bool highQualityBlend);
    void stopVideoStitch();

    bool openLiveStream(const std::string& name, int panoType, int width, int height, int videoBPS,
        const std::string& videoEncoder, const std::string& videoPreset, int audioBPS);
    void closeLiveStream();

    bool beginSaveToDisk(const std::string& dir, int panoType, int width, int height, int videoBPS,
        const std::string& videoEncoder, const std::string& videoPreset, int audioBPS, int fileDuration);
    void stopSaveToDisk();

    bool getVideoSourceFrames(std::vector<avp::AudioVideoFrame2>& frames);
    bool getStitchedVideoFrame(avp::AudioVideoFrame2& frame);
    void cancelGetVideoSourceFrames();
    void cancelGetStitchedVideoFrame();

    double getVideoSourceFrameRate() const;
    double getStitchFrameRate() const;
    void getLastSyncErrorMessage(std::string& message) const;
    bool hasAsyncErrorMessage() const;
    void getLastAsyncErrorMessage(std::string& message);
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

    CudaPanoramaRender2 render;
    std::string renderConfigName;
    cv::Size renderFrameSize;
    int renderPrepareSuccess;
    std::unique_ptr<std::thread> renderThread;
    int renderEndFlag;
    int renderThreadJoined;
    void procVideo();

    CudaLogoFilter logoFilter;
    //std::unique_ptr<std::thread> postProcThread;
    //void postProc();

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

    double videoSourceFrameRate;
    double stitchVideoFrameRate;

    std::string syncErrorMessage;

    std::mutex mtxAsyncErrorMessage;
    std::string asyncErrorMessage;
    int hasAsyncError;
    void setAsyncErrorMessage(const std::string& message);
    void clearAsyncErrorMessage();

    std::mutex mtxLog;
    std::string log;
    void appendLog(const std::string& message);
    void clearLog();

    int pixelType;
    int elemType;
    int finish;
    ForShowFrameVectorQueue syncedFramesBufferForShow;
    BoundedPinnedMemoryFrameQueue syncedFramesBufferForProc;
    //AudioVideoFramePool procFramePool;
    CudaHostMemVideoFrameMemoryPool procFramePool, sendFramePool, saveFramePool;
    //ForShowFrameQueue procFrameBufferForShow;
    //ForceWaitFrameQueue procFrameBufferForSend, procFrameBufferForSave;
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
        //syncErrorMessage = "视频源任务正在运行中，先关闭当前运行的任务，再启动新的任务。";
        syncErrorMessage = getText(TI_AUDIO_VIDEO_SOURCE_RUNNING_CLOSE_BEFORE_LANCH_NEW);
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
        syncErrorMessage = getText(TI_AUDIO_VIDEO_SOURCE_RUNNING_CLOSE_BEFORE_LANCH_NEW);
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

bool PanoramaLiveStreamTask2::Impl::beginVideoStitch(const std::string& configFileName, int width, int height, bool highQualityBlend)
{
    if (!videoOpenSuccess || !audioVideoSource)
    {
        ptlprintf("Error in %s, audio video sources not opened\n", __FUNCTION__);
        syncErrorMessage = getText(TI_SOURCE_NOT_OPENED_CANNOT_LAUNCH_STITCH)/*"尚未打开音视频源，无法启动拼接任务。"*/;
        return false;
    }

    if (!renderThreadJoined)
    {
        ptlprintf("Error in %s, stitching running, stop before launching new stitching\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_RUNNING_CLOSE_BEFORE_LAUNCH_NEW)/*"视频拼接任务正在进行中，请先关闭正在执行的任务，再启动新的任务。"*/;
        return false;
    }

    renderConfigName = configFileName;
    renderFrameSize.width = width;
    renderFrameSize.height = height;

    renderPrepareSuccess = render.prepare(renderConfigName, "", highQualityBlend, 
        videoFrameSize, renderFrameSize);
    if (!renderPrepareSuccess)
    {
        ptlprintf("Error in %s, could not prepare for video stitch\n", __FUNCTION__);
        //appendLog("视频拼接初始化失败\n");
        //syncErrorMessage = "视频拼接初始化失败。";
        appendLog(getText(TI_STITCH_INIT_FAIL) + "\n");
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (!highQualityBlend)
    {
        std::vector<cv::cuda::HostMem> mems;
        std::vector<long long int> timeStamps;
        syncedFramesBufferForProc.pull(mems, timeStamps);
        std::vector<cv::Mat> images(numVideos);
        for (int i = 0; i < numVideos; i++)
            images[i] = mems[i].createMatHeader();
        renderPrepareSuccess = render.exposureCorrect(images);
        if (!renderPrepareSuccess)
        {
            ptlprintf("Error in %s, exposure correction failed\n", __FUNCTION__);
            //appendLog("视频拼接初始化失败\n");
            //syncErrorMessage = "视频拼接初始化失败。";
            appendLog(getText(TI_STITCH_INIT_FAIL) + "\n");
            syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
            return false;
        }
    }

    if (render.getNumImages() != numVideos)
    {
        ptlprintf("Error in %s, num images in render not equal to num videos\n", __FUNCTION__);
        //appendLog("视频拼接初始化失败\n");
        //syncErrorMessage = "视频拼接初始化失败。";
        appendLog(getText(TI_STITCH_INIT_FAIL) + "\n");
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    renderPrepareSuccess = procFramePool.init(pixelType, width, height);
    if (!renderPrepareSuccess)
    {
        ptlprintf("Error in %s, could not init proc frame pool\n", __FUNCTION__);
        //appendLog("视频拼接初始化失败\n");
        //syncErrorMessage = "视频拼接初始化失败。";
        appendLog(getText(TI_STITCH_INIT_FAIL) + "\n");
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (addLogo)
        renderPrepareSuccess = logoFilter.init(width, height);
    if (!renderPrepareSuccess)
    {
        ptlprintf("Error in %s, could not init logo filter\n", __FUNCTION__);
        //appendLog("视频拼接初始化失败\n");
        //syncErrorMessage = "视频拼接初始化失败。";
        appendLog(getText(TI_STITCH_INIT_FAIL) + "\n");
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    //appendLog("视频拼接初始化成功\n");
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
    //postProcThread.reset(new std::thread(&PanoramaLiveStreamTask2::Impl::postProc, this));

    //appendLog("视频拼接任务启动成功\n");
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
        render.clear();
        //postProcThread->join();
        //postProcThread.reset(0);
        renderPrepareSuccess = 0;
        renderThreadJoined = 1;
        procFramePool.clear();

        //appendLog("视频拼接任务结束\n");
        appendLog(getText(TI_STITCH_TASK_FINISH) + "\n");
    }
}

bool PanoramaLiveStreamTask2::Impl::getVideoSourceFrames(std::vector<avp::AudioVideoFrame2>& frames)
{
    return syncedFramesBufferForShow.pull(frames);
}

bool PanoramaLiveStreamTask2::Impl::getStitchedVideoFrame(avp::AudioVideoFrame2& frame)
{
    MixedAudioVideoFrame mixedFrame;
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
        syncErrorMessage = getText(TI_SOURCE_NOT_OPENED_CANNOT_LAUNCH_LIVE)/*"尚未打开音视频源，无法启动推流任务。"*/;
        return false;
    }

    if (!renderPrepareSuccess || renderThreadJoined)
    {
        ptlprintf("Error in %s, render not running, cannot launch live streaming\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_NOT_RUNNING_CANNOT_LAUNCH_LIVE)/*"尚未启动拼接任务，无法启动推流任务。"*/;
        return false;
    }

    if (!streamThreadJoined)
    {
        ptlprintf("Error in %s, live streaming running, stop before launching new live streaming\n", __FUNCTION__);
        syncErrorMessage = getText(TI_LIVE_RUNNING_CLOSE_BEFORE_LAUNCH_NEW)/*"推流任务正在进行中，请先关闭正在执行的任务，再启动新的任务。"*/;
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
        syncErrorMessage = getText(TI_LIVE_PARAM_ERROR_CANNOT_LAUNCH_LIVE)/*"参数错误，无法启动推流任务。"*/;
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
    if (streamVideoEncodePreset != "ultrafast" || streamVideoEncodePreset != "superfast" ||
        streamVideoEncodePreset != "veryfast" || streamVideoEncodePreset != "faster" ||
        streamVideoEncodePreset != "fast" || streamVideoEncodePreset != "medium" || streamVideoEncodePreset != "slow" ||
        streamVideoEncodePreset != "slower" || streamVideoEncodePreset != "veryslow")
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
        ptlprintf("Could not open streaming url with frame rate = %f and bit rate = %d\n", videoFrameRate, streamVideoBitRate);
        //appendLog("流媒体服务器连接失败\n");
        //syncErrorMessage = "流媒体服务器连接失败。";
        appendLog(getText(TI_SERVER_CONNECT_FAIL) + "\n");
        syncErrorMessage = getText(TI_SERVER_CONNECT_FAIL);
        return false;
    }

    //appendLog("流媒体服务器连接成功\n");
    appendLog(getText(TI_SERVER_CONNECT_SUCCESS) + "\n");

    procFrameBufferForSend.resume();
    streamEndFlag = 0;
    streamThreadJoined = 0;
    streamThread.reset(new std::thread(&PanoramaLiveStreamTask2::Impl::streamSend, this));

    //appendLog("推流任务启动成功\n");
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
        streamOpenSuccess = 0;
        streamThreadJoined = 1;
        sendFramePool.clear();

        //appendLog("推流任务结束\n");
        //appendLog("断开与流媒体服务器连接\n");
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
        syncErrorMessage = getText(TI_SOURCE_NOT_OPENED_CANNOT_LAUNCH_WRITE)/*"尚未打开音视频源，无法启动保存任务。"*/;
        return false;
    }

    if (!renderPrepareSuccess || renderThreadJoined)
    {
        ptlprintf("Error in %s, render not running, cannot launch saving to disk\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_NOT_RUNNING_CANNOT_LAUNCH_WRITE)/*"尚未启动拼接任务，无法启动保存任务。"*/;
        return false;
    }

    if (!fileThreadJoined)
    {
        ptlprintf("Error in %s, saving to disk running, stop before launching new saving to disk\n", __FUNCTION__);
        syncErrorMessage = getText(TI_WRITE_RUNNING_CLOSE_BEFORE_LAUNCH_NEW)/*"保存任务正在进行中，请先关闭正在执行的任务，再启动新的任务。"*/;
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
        syncErrorMessage = getText(TI_WRITE_PARAM_ERROR_CANNOT_LAUNCH_WRITE)/*"参数错误，无法启动保存任务。"*/;
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
    if (fileVideoEncodePreset != "ultrafast" || fileVideoEncodePreset != "superfast" ||
        fileVideoEncodePreset != "veryfast" || fileVideoEncodePreset != "faster" ||
        fileVideoEncodePreset != "fast" || fileVideoEncodePreset != "medium" || fileVideoEncodePreset != "slow" ||
        fileVideoEncodePreset != "slower" || fileVideoEncodePreset != "veryslow")
        fileVideoEncodePreset = "veryfast";
    fileConfigSet = 1;

    procFrameBufferForSave.resume();
    fileEndFlag = 0;
    fileThreadJoined = 0;
    fileThread.reset(new std::thread(&PanoramaLiveStreamTask2::Impl::fileSave, this));

    //appendLog("启动保存视频任务\n");
    appendLog(getText(TI_WRITE_LAUNCH) + "\n");

    fileIsLibX264 = (fileVideoEncoder == "h264" || fileVideoEncoder == "libx264") ? 1 : 0;
    saveFramePool.init(fileIsLibX264 ? avp::PixelTypeYUV420P : avp::PixelTypeNV12, width, height);

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

        //appendLog("结束保存视频任务\n");
        appendLog(getText(TI_WRITE_FINISH) + "\n");
    }
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

void PanoramaLiveStreamTask2::Impl::getLastAsyncErrorMessage(std::string& message)
{
    if (hasAsyncError)
    {
        std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
        message = asyncErrorMessage;
        hasAsyncError = 0;
    }
    else
        message.clear();
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
    MixedAudioVideoFrame renderFrame, sendFrame, saveFrame;
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
            //procTimer.start();
            ok = render.render(src, timeStamps, bgr32);
            //procTimer.end();
            if (!ok)
            {
                ptlprintf("Error in %s [%8x], render failed\n", __FUNCTION__, id);
                //setAsyncErrorMessage("视频拼接发生错误，任务终止。");
                setAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE));
                finish = 1;
                break;
            }

            if (addLogo)
            {
                ok = logoFilter.addLogo(bgr32);
                if (!ok)
                {
                    ptlprintf("Error in %s [%8x], render failed\n", __FUNCTION__, id);
                    //setAsyncErrorMessage("视频拼接发生错误，任务终止。");
                    setAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE));
                    finish = 1;
                    break;
                }
            }

            ok = procFramePool.get(renderFrame);
            if (!ok)
            {
                ptlprintf("Error in %s [%8x], could not get cpu memory to copy from gpu\n", __FUNCTION__, id);
                //setAsyncErrorMessage("视频拼接发生错误，任务终止。");
                setAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE));
                finish = 1;
                break;
            }
            cv::Mat bgr = renderFrame.planes[0].createMatHeader();
            bgr32.download(bgr);
            renderFrame.frame.timeStamp = timeStamps[0];
            procFrameBufferForShow.push(renderFrame);

            if (streamOpenSuccess)
            {
                if (streamPanoType == PanoTypeEquiRect)
                {
                    if (streamFrameSize == renderFrameSize)
                        bgr1 = bgr32;
                    else
                        cv::cuda::resize(bgr32, bgr1, streamFrameSize, 0, 0, cv::INTER_LINEAR);
                }
                else
                    cudaReproject(bgr32, bgr1, streamXMap, streamYMap);

                sendFramePool.get(sendFrame);
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

            if (fileConfigSet)
            {
                if (filePanoType == PanoTypeEquiRect)
                {
                    if (fileFrameSize == renderFrameSize)
                        bgr2 = bgr32;
                    else
                        cv::cuda::resize(bgr32, bgr2, fileFrameSize, 0, 0, cv::INTER_LINEAR);
                }
                else
                    cudaReproject(bgr32, bgr2, fileXMap, fileYMap);

                saveFramePool.get(saveFrame);
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

//void PanoramaLiveStreamTask2::Impl::postProc()
//{
//    size_t id = std::this_thread::get_id().hash();
//    ptlprintf("Thread %s [%8x] started\n", __FUNCTION__, id);
//
//    avp::AudioVideoFrame2 frame;
//    while (true)
//    {
//        if (finish || renderEndFlag)
//            break;
//
//        procFramePool.get(frame);
//        cv::Mat result(frame.height, frame.width, elemType, frame.data[0], frame.steps[0]);
//
//        if (!render.getResult(result, frame.timeStamp))
//            continue;
//
//        //ztool::Timer timer;
//        logoFilter.addLogo(result);
//        procFrameBufferForShow.push(frame);
//        if (streamOpenSuccess)
//            procFrameBufferForSend.push(frame);
//        if (fileConfigSet)
//            procFrameBufferForSave.push(frame);
//        //timer.end();
//        //ptlprintf("%f\n", timer.elapse());
//    }
//
//    //procFrameBufferForShow.stop();
//    procFrameBufferForSend.stop();
//    procFrameBufferForSave.stop();
//
//    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
//}

void PanoramaLiveStreamTask2::Impl::streamSend()
{
    size_t id = std::this_thread::get_id().hash();
    ptlprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    cv::Mat dstMat;
    MixedAudioVideoFrame frame;
    while (true)
    {
        if (finish || streamEndFlag)
            break;
        procFrameBufferForSend.pull(frame);
        if (frame.frame.data[0] && (frame.frame.mediaType == avp::AUDIO || frame.frame.mediaType == avp::VIDEO))
        {
            bool ok = streamWriter.write(frame.frame);
            if (!ok)
            {
                ptlprintf("Error in %s [%8x], cannot write frame\n", __FUNCTION__, id);
                //setAsyncErrorMessage("推流发生错误，任务终止。");
                setAsyncErrorMessage(getText(TI_LIVE_FAIL_TASK_TERMINATE));
                finish = 1;
                break;
            }
        }
        /*
        if (frame.data[0])
        {
            avp::AudioVideoFrame2 shallow;
            //ptlprintf("%s, %lld\n", frame.mediaType == avp::VIDEO ? "VIDEO" : "AUDIO", frame.timeStamp);
            if (frame.mediaType == avp::VIDEO && streamFrameSize != renderFrameSize)
            {
                cv::Mat srcMat(renderFrameSize, elemType, frame.data[0], frame.steps[0]);
                cv::resize(srcMat, dstMat, streamFrameSize, 0, 0, cv::INTER_NEAREST);
                unsigned char* data[4] = { dstMat.data };
                int steps[4] = { dstMat.step };
                shallow = avp::AudioVideoFrame2(data, steps, pixelType, dstMat.cols, dstMat.rows, frame.timeStamp);
            }
            else
                shallow = frame;
            bool ok = streamWriter.write(shallow);
            if (!ok)
            {
                ptlprintf("Error in %s [%8x], cannot write frame\n", __FUNCTION__, id);
                setAsyncErrorMessage("推流发生错误，任务终止。");
                finish = 1;
                break;
            }
        }
        */
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
    MixedAudioVideoFrame frame;
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
        //appendLog(std::string(buf) + " 无法打开，任务终止\n");
        appendLog(std::string(buf) + " " + getText(TI_FILE_OPEN_FAIL_TASK_TERMINATE) + "\n");
        return;
    }
    else
        //appendLog(std::string(buf) + " 开始写入\n");
        appendLog(std::string(buf) + " " + getText(TI_BEGIN_WRITE) + "\n");
    long long int fileFirstTimeStamp = -1;
    while (true)
    {
        if (finish || fileEndFlag)
            break;
        procFrameBufferForSave.pull(frame);
        //printf("pass pull frame\n");
        if (frame.frame.data[0] && (frame.frame.mediaType == avp::AUDIO || frame.frame.mediaType == avp::VIDEO))
        {
            if (fileFirstTimeStamp < 0)
                fileFirstTimeStamp = frame.frame.timeStamp;

            if (frame.frame.timeStamp - fileFirstTimeStamp > fileDuration * 1000000LL)
            {
                writer.close();
                //appendLog(std::string(buf) + " 写入结束\n");
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
                    //appendLog(std::string(buf) + " 无法打开，任务终止\n");
                    appendLog(std::string(buf) + " " + getText(TI_FILE_OPEN_FAIL_TASK_TERMINATE) + "\n");
                    break;
                }
                else
                    //appendLog(std::string(buf) + " 开始写入\n");
                    appendLog(std::string(buf) + " " + getText(TI_BEGIN_WRITE) + "\n");
                fileFirstTimeStamp = frame.frame.timeStamp;
            }
            /*
            avp::AudioVideoFrame2 shallow;
            if (frame.mediaType == avp::VIDEO && fileFrameSize != renderFrameSize)
            {
                cv::Mat srcMat(renderFrameSize, elemType, frame.data[0], frame.steps[0]);
                cv::resize(srcMat, dstMat, fileFrameSize, 0, 0, cv::INTER_NEAREST);
                unsigned char* data[4] = { dstMat.data };
                int steps[4] = { dstMat.step };
                shallow = avp::AudioVideoFrame2(data, steps, pixelType, dstMat.cols, dstMat.rows, frame.timeStamp);
            }
            else
                shallow = frame;
            */

            ok = writer.write(frame.frame);
            if (!ok)
            {
                ptlprintf("Error in %s [%8x], could not write current frame\n", __FUNCTION__, id);
                //setAsyncErrorMessage(std::string(buf) + " 写入发生错误，任务终止。");
                setAsyncErrorMessage(std::string(buf) + " " + getText(TI_WRITE_FAIL_TASK_TERMINATE));
                break;
            }
        }
        else
        {
            ptlprintf("Frame error\n");
        }
    }
    if (ok)
        //appendLog(std::string(buf) + " 写入结束\n");
        appendLog(std::string(buf) + " " + getText(TI_END_WRITE) + "\n");
    writer.close();

    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask2::Impl::setAsyncErrorMessage(const std::string& message)
{
    std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
    hasAsyncError = 1;
    asyncErrorMessage = message;
}

void PanoramaLiveStreamTask2::Impl::clearAsyncErrorMessage()
{
    std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
    hasAsyncError = 0;
    asyncErrorMessage.clear();
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

bool PanoramaLiveStreamTask2::beginVideoStitch(const std::string& configFileName, int width, int height, bool highQualityBlend)
{
    return ptrImpl->beginVideoStitch(configFileName, width, height, highQualityBlend);
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

void PanoramaLiveStreamTask2::getLastAsyncErrorMessage(std::string& message)
{
    return ptrImpl->getLastAsyncErrorMessage(message);
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