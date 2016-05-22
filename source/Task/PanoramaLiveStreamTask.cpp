#include "CompileControl.h"
#include "PanoramaTask.h"
#include "ConcurrentQueue.h"
#include "PinnedMemoryFrameQueue.h"
#include "SharedAudioVideoFramePool.h"
#include "RicohUtil.h"
#include "PanoramaTaskUtil.h"
#include "Timer.h"
#include "Image.h"
#include "oclobject.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

// for video source
//typedef ForceWaitRealTimeQueue<avp::SharedAudioVideoFrame> CompleteFrameQueue;
// for synced video source
//typedef ForceWaitRealTimeQueue<std::vector<avp::SharedAudioVideoFrame> > RealTimeFrameVectorQueue;
// for audio source and proc result
//typedef ForceWaitRealTimeQueue<avp::SharedAudioVideoFrame> RealTimeFrameQueue;

// for individual video and audio sources and proc result
typedef ForceWaitRealTimeQueue<avp::SharedAudioVideoFrame> ForceWaitFrameQueue;
// for synced video source frames
typedef ForceWaitRealTimeQueue<std::vector<avp::SharedAudioVideoFrame> > ForceWaitFrameVectorQueue;
// for video frame for show
typedef RealTimeQueue<avp::SharedAudioVideoFrame> ForShowFrameQueue;
// for video frames for show
typedef RealTimeQueue<std::vector<avp::SharedAudioVideoFrame> > ForShowFrameVectorQueue;

struct PanoramaLiveStreamTask::Impl
{
    Impl();
    ~Impl();

    bool openVideoDevices(const std::vector<avp::Device>& devices, int width, int height, int frameRate, std::vector<int>& success);
    void closeVideoDevices();

    bool openAudioDevice(const avp::Device& device, int sampleRate);
    void closeAudioDevice();

    bool openVideoStreams(const std::vector<std::string>& urls);
    bool openAudioStream(const std::string& url);

    bool beginVideoStitch(const std::string& configFileName, int width, int height, bool highQualityBlend);
    void stopVideoStitch();

    bool openLiveStream(const std::string& name, int width, int height, int videoBPS, 
        const std::string& videoEncoder, const std::string& videoPreset, int audioBPS);
    void closeLiveStream();

    bool beginSaveToDisk(const std::string& dir, int width, int height, int videoBPS, 
        const std::string& videoEncoder, const std::string& videoPreset, int audioBPS, int fileDuration);
    void stopSaveToDisk();

    bool getVideoSourceFrames(std::vector<avp::SharedAudioVideoFrame>& frames);
    bool getStitchedVideoFrame(avp::SharedAudioVideoFrame& frame);
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

    std::vector<avp::AudioVideoReader> videoReaders;
    std::vector<avp::Device> videoDevices;
    cv::Size videoFrameSize;
    double videoFrameRate;
    int roundedVideoFrameRate;
    int numVideos;
    int videoOpenSuccess;
    int videoCheckFrameRate;
    std::vector<std::unique_ptr<std::thread> > videoSourceThreads;
    std::unique_ptr<std::thread> videoSinkThread;
    int videoEndFlag;
    int videoThreadsJoined;
    void videoSource(int index);
    void videoSink();

    avp::AudioVideoReader audioReader;
    avp::Device audioDevice;
    int audioSampleRate;
    int audioOpenSuccess;
    std::unique_ptr<std::thread> audioThread;
    int audioEndFlag;
    int audioThreadJoined;
    void audioSource();

#if COMPILE_CUDA
    CudaPanoramaRender render;
#else
    IOclPanoramaRender render;
#endif
    std::string renderConfigName;
    cv::Size renderFrameSize;
    int renderPrepareSuccess;
    std::unique_ptr<std::thread> renderThread;
    int renderEndFlag;
    int renderThreadJoined;
    void procVideo();

    LogoFilter logoFilter;
    std::unique_ptr<std::thread> postProcThread;
    void postProc();

    avp::AudioVideoWriter2 streamWriter;
    std::string streamURL;
    cv::Size streamFrameSize;
    int streamVideoBitRate;
    std::string streamVideoEncodePreset;
    int streamAudioBitRate;
    int streamOpenSuccess;
    std::unique_ptr<std::thread> streamThread;
    int streamEndFlag;
    int streamThreadJoined;
    void streamSend();

    std::string fileWriterFormat;
    cv::Size fileFrameSize;
    int fileVideoBitRate;
    std::string fileVideoEncoder;
    std::string fileVideoEncodePreset;
    int fileAudioBitRate;
    int fileDuration;
    int fileConfigSet;
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
    std::unique_ptr<std::vector<ForceWaitFrameQueue> > ptrFrameBuffers;
    ForShowFrameVectorQueue syncedFramesBufferForShow;
#if COMPILE_CUDA
    BoundedPinnedMemoryFrameQueue syncedFramesBufferForProc;
#else
    ForShowFrameVectorQueue syncedFramesBufferForProc;
#endif
    SharedAudioVideoFramePool procFramePool;
    ForShowFrameQueue procFrameBufferForShow;
    ForceWaitFrameQueue procFrameBufferForSend, procFrameBufferForSave;

#if COMPILE_CUDA
#else
    OpenCLBasic ocl;
#endif
};

PanoramaLiveStreamTask::Impl::Impl()
#if !COMPILE_CUDA
    : ocl("Intel", "GPU")
#endif
{
    initAll();
    //initCallback();
}

PanoramaLiveStreamTask::Impl::~Impl()
{
    closeAll();
    printf("live stream task destructor called\n");
}

bool PanoramaLiveStreamTask::Impl::openVideoDevices(const std::vector<avp::Device>& devices, int width, int height, int frameRate, std::vector<int>& success)
{
    if (videoOpenSuccess || !videoThreadsJoined)
    {
        printf("Error in %s, video sources should be closed first before open again\n", __FUNCTION__);
        syncErrorMessage = "视频源任务正在运行中，先关闭当前运行的任务，再启动新的任务。";
        return false;
    }

    videoDevices = devices;
    videoFrameSize.width = width;
    videoFrameSize.height = height;
    videoFrameRate = frameRate;
    roundedVideoFrameRate = videoFrameRate + 0.5;
    numVideos = devices.size();

    std::vector<avp::Option> opts;
    std::string frameRateStr = std::to_string(videoFrameRate);
    std::string videoSizeStr = std::to_string(videoFrameSize.width) + "x" + std::to_string(videoFrameSize.height);
    opts.push_back(std::make_pair("framerate", frameRateStr));
    opts.push_back(std::make_pair("video_size", videoSizeStr));

    bool ok;
    bool failExists = false;
    success.resize(numVideos);
    videoReaders.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        // IMPORTANT NOTICE!!!!!!
        // FORCE PIXEL_TYPE_BGR_32
        // SUPPORT GPU ONLY
        opts.resize(2);
        opts.push_back(std::make_pair("video_device_number", videoDevices[i].numString));
        ok = videoReaders[i].open("video=" + videoDevices[i].shortName,
            false, true, pixelType/*avp::PixelTypeBGR32*/, "dshow", opts);
        if (!ok)
        {
            printf("Could not open DirectShow video device %s[%s] with framerate = %s and video_size = %s\n",
                videoDevices[i].shortName.c_str(), videoDevices[i].numString.c_str(),
                frameRateStr.c_str(), videoSizeStr.c_str());
            failExists = true;
        }
        success[i] = ok;
    }

    videoOpenSuccess = !failExists;
    if (failExists)
    {
        appendLog("视频源打开失败\n");
        syncErrorMessage = "视频源打开失败。";
        return false;
    }

    appendLog("视频源打开成功\n");

    ptrFrameBuffers.reset(new std::vector<ForceWaitFrameQueue>(numVideos));
    for (int i = 0; i < numVideos; i++)
        (*ptrFrameBuffers)[i].setMaxSize(36);
    syncedFramesBufferForShow.clear();
    syncedFramesBufferForProc.clear();

    videoEndFlag = 0;
    videoThreadsJoined = 0;
    videoSourceThreads.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        videoSourceThreads[i].reset(new std::thread(&PanoramaLiveStreamTask::Impl::videoSource, this, i));
    }
    videoSinkThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::videoSink, this));

    appendLog("视频源任务启动成功\n");

    return true;
}

void PanoramaLiveStreamTask::Impl::closeVideoDevices()
{
    if (videoOpenSuccess && !videoThreadsJoined)
    {
        videoEndFlag = 1;
        for (int i = 0; i < numVideos; i++)
            videoSourceThreads[i]->join();
        videoSinkThread->join();
        videoSourceThreads.clear();
        videoSinkThread.reset(0);
        videoOpenSuccess = 0;
        videoThreadsJoined = 1;

        appendLog("视频源任务结束\n");
        appendLog("视频源关闭\n");
    }
}

bool PanoramaLiveStreamTask::Impl::openAudioDevice(const avp::Device& device, int sampleRate)
{
    if (audioOpenSuccess || !audioThreadJoined)
    {
        printf("Error in %s, audio source should be closed first before open again\n", __FUNCTION__);
        syncErrorMessage = "音频源任务正在运行中，先关闭当前运行的任务，再启动新的任务。";
        return false;
    }

    audioDevice = device;
    audioSampleRate = sampleRate;

    std::vector<avp::Option> audioOpts;
    std::string sampleRateStr = std::to_string(audioSampleRate);
    audioOpts.push_back(std::make_pair("ar", sampleRateStr));
    audioOpts.push_back(std::make_pair("audio_device_number", audioDevice.numString));
    audioOpenSuccess = audioReader.open("audio=" + audioDevice.shortName, true,
        false, avp::PixelTypeUnknown, "dshow", audioOpts);
    if (!audioOpenSuccess)
    {
        printf("Could not open DirectShow audio device %s[%s], skip\n",
            audioDevice.shortName.c_str(), audioDevice.numString.c_str());
        appendLog("音频源打开失败\n");
        syncErrorMessage = "音频源打开失败。";
        return false;
    }

    appendLog("音频源打开成功\n");

    procFrameBufferForSave.clear();
    procFrameBufferForSend.clear();

    audioEndFlag = 0;
    audioThreadJoined = 0;
    audioThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::audioSource, this));

    appendLog("音频源任务启动成功\n");

    return true;
}

void PanoramaLiveStreamTask::Impl::closeAudioDevice()
{
    if (audioOpenSuccess && !audioThreadJoined)
    {
        audioEndFlag = 1;
        audioThread->join();
        audioThread.reset(0);
        audioOpenSuccess = 0;
        audioThreadJoined = 1;

        appendLog("音频源任务结束\n");
        appendLog("音频源关闭\n");
    }
}

bool PanoramaLiveStreamTask::Impl::openVideoStreams(const std::vector<std::string>& urls)
{
    if (videoOpenSuccess || !videoThreadsJoined)
    {
        printf("Error in %s, video sources should be closed first before open again\n", __FUNCTION__);
        syncErrorMessage = "推流任务正在进行中，先关闭当前运行的任务，再启动新的推流任务。";
        return false;
    }

    if (urls.empty())
    {
        printf("Error in %s, no urls assigned\n", __FUNCTION__);
        syncErrorMessage = "视频源地址为空，请重新设定。";
        return false;
    }        

    bool ok;
    bool failExists = false;
    numVideos = urls.size();
    videoReaders.resize(numVideos);
    std::vector<avp::Option> opts;
    opts.push_back(std::make_pair("max_delay", "1000000"));
    for (int i = 0; i < numVideos; i++)
    {
        ok = videoReaders[i].open(urls[i], false, true, avp::PixelTypeBGR32);
        if (!ok)
        {
            printf("Could not open video stream %s\n", urls[i].c_str());
            failExists = true;
            break;
        }
    }

    videoOpenSuccess = !failExists;
    if (failExists)
    {
        appendLog("视频源打开失败\n");
        syncErrorMessage = "视频源打开失败";
        return false;
    }

    videoFrameSize.width = videoReaders[0].getVideoWidth();
    videoFrameSize.height = videoReaders[0].getVideoHeight();
    videoFrameRate = videoReaders[0].getVideoFps();
    roundedVideoFrameRate = videoFrameRate + 0.5;
    for (int i = 1; i < numVideos; i++)
    {
        if (videoReaders[i].getVideoWidth() != videoFrameSize.width)
        {
            failExists = true;
            printf("Error, video streams width not match\n");
            syncErrorMessage = "所有视频源需要有同样的分辨率和帧率。";
            break;
        }
        if (videoReaders[i].getVideoHeight() != videoFrameSize.height)
        {
            failExists = true;
            printf("Error, video streams height not match\n");
            syncErrorMessage = "所有视频源需要有同样的分辨率和帧率。";
            break;
        }
        if (fabs(videoReaders[i].getVideoFps() - videoFrameRate) > 0.001)
        {
            failExists = true;
            printf("Error, video streams frame rate not match\n");
            syncErrorMessage = "所有视频源需要有同样的分辨率和帧率。";
            break;
        }
    }

    videoOpenSuccess = !failExists;
    if (failExists)
    {
        appendLog("视频源打开失败\n");
        return false;
    }

    appendLog("视频源打开成功\n");

    ptrFrameBuffers.reset(new std::vector<ForceWaitFrameQueue>(numVideos));
    for (int i = 0; i < numVideos; i++)
        (*ptrFrameBuffers)[i].setMaxSize(36);
    syncedFramesBufferForShow.clear();
    syncedFramesBufferForProc.clear();

    videoEndFlag = 0;
    videoThreadsJoined = 0;
    videoSourceThreads.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        videoSourceThreads[i].reset(new std::thread(&PanoramaLiveStreamTask::Impl::videoSource, this, i));
    }
    videoSinkThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::videoSink, this));

    appendLog("视频源任务启动成功\n");

    return true;
}

bool PanoramaLiveStreamTask::Impl::openAudioStream(const std::string& url)
{
    if (audioOpenSuccess || !audioThreadJoined)
    {
        printf("Error in %s, audio source should be closed first before open again\n", __FUNCTION__);
        return false;
    }

    audioOpenSuccess = audioReader.open(url, true, false, avp::PixelTypeUnknown);
    if (!audioOpenSuccess)
    {
        printf("Could not open DirectShow audio device %s[%s], skip\n",
            audioDevice.shortName.c_str(), audioDevice.numString.c_str());
        appendLog("音频源打开失败\n");
        syncErrorMessage = "音频源打开失败。";
        return false;
    }

    audioSampleRate = audioReader.getAudioSampleRate();

    appendLog("音频源打开成功\n");

    procFrameBufferForSave.clear();
    procFrameBufferForSend.clear();

    audioEndFlag = 0;
    audioThreadJoined = 0;
    audioThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::audioSource, this));

    appendLog("音频源任务启动成功\n");

    return true;
}

bool PanoramaLiveStreamTask::Impl::beginVideoStitch(const std::string& configFileName, int width, int height, bool highQualityBlend)
{
    if (!videoOpenSuccess)
    {
        printf("Error in %s, audio video sources not opened\n", __FUNCTION__);
        syncErrorMessage = "尚未打开音视频源，无法启动拼接任务。";
        return false;
    }

    if (!renderThreadJoined)
    {
        printf("Error in %s, stitching running, stop before launching new stitching\n", __FUNCTION__);
        syncErrorMessage = "视频拼接任务正在进行中，请先关闭正在执行的任务，再启动新的任务。";
        return false;
    }

    renderConfigName = configFileName;
    renderFrameSize.width = width;
    renderFrameSize.height = height;

#if COMPILE_CUDA
    renderPrepareSuccess = render.prepare(renderConfigName, highQualityBlend, false,
        videoFrameSize, renderFrameSize);
#else
    renderPrepareSuccess = render.prepare(renderConfigName, highQualityBlend, false,
        videoFrameSize, renderFrameSize, &ocl);
#endif
    if (!renderPrepareSuccess)
    {
        printf("Could not prepare for video stitch\n");
        appendLog("视频拼接初始化失败\n");
        syncErrorMessage = "视频拼接初始化失败。";
        return false;
    }

    if (render.getNumImages() != numVideos)
    {
        renderPrepareSuccess = 0;
        printf("Error in %s, num images in render not equal to num videos\n", __FUNCTION__);
        appendLog("视频拼接初始化失败\n");
        syncErrorMessage = "视频拼接初始化失败。";
        return false;
    }

    renderPrepareSuccess = procFramePool.initAsVideoFramePool(pixelType, width, height);
    if (!renderPrepareSuccess)
    {
        printf("Could not init proc frame pool\n");
        appendLog("视频拼接初始化失败\n");
        syncErrorMessage = "视频拼接初始化失败。";
        return false;
    }

    if (addLogo)
        renderPrepareSuccess = logoFilter.init(width, height, elemType);
    if (!renderPrepareSuccess)
    {
        printf("Could not init logo filter\n");
        appendLog("视频拼接初始化失败\n");
        syncErrorMessage = "视频拼接初始化失败。";
        return false;
    }

    appendLog("视频拼接初始化成功\n");
    
    syncedFramesBufferForProc.clear();
    procFrameBufferForShow.clear();
    procFrameBufferForSave.clear();
    procFrameBufferForSend.clear();

    renderEndFlag = 0;
    renderThreadJoined = 0;
    renderThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::procVideo, this));
    postProcThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::postProc, this));

    appendLog("视频拼接任务启动成功\n");

    return true;
}

void PanoramaLiveStreamTask::Impl::stopVideoStitch()
{
    if (renderPrepareSuccess && !renderThreadJoined)
    {
        renderEndFlag = 1;
#if COMPILE_CUDA
        syncedFramesBufferForProc.stop();
#endif
        renderThread->join();
        renderThread.reset(0);
        render.clear();
        postProcThread->join();
        postProcThread.reset(0);
        renderPrepareSuccess = 0;
        renderThreadJoined = 1;

        appendLog("视频拼接任务结束\n");
    }
}

bool PanoramaLiveStreamTask::Impl::getVideoSourceFrames(std::vector<avp::SharedAudioVideoFrame>& frames)
{
    return syncedFramesBufferForShow.pull(frames);
    /*std::vector<avp::SharedAudioVideoFrame> tempFrames;
    bool ok = syncedFramesBufferForShow.pull(tempFrames);
    if (ok)
    {
        int size = tempFrames.size();
        frames.resize(size);
        for (int i = 0; i < size; i++)
            avp::copy(tempFrames[i], frames[i]);
    }
    return ok;*/
}

bool PanoramaLiveStreamTask::Impl::getStitchedVideoFrame(avp::SharedAudioVideoFrame& frame)
{
    return procFrameBufferForShow.pull(frame);
    /*avp::SharedAudioVideoFrame tempFrame;
    bool ok = procFrameBufferForShow.pull(tempFrame);
    if (ok)
        avp::copy(tempFrame, frame);
    return ok;*/
}

void PanoramaLiveStreamTask::Impl::cancelGetVideoSourceFrames()
{
    //syncedFramesBufferForShow.stop();
}

void PanoramaLiveStreamTask::Impl::cancelGetStitchedVideoFrame()
{
    //procFrameBufferForShow.stop();
}

bool PanoramaLiveStreamTask::Impl::openLiveStream(const std::string& name,
    int width, int height, int videoBPS, const std::string& videoEncoder, const std::string& videoPreset, int audioBPS)
{
    if (!videoOpenSuccess)
    {
        printf("Error in %s, audio video sources not opened\n", __FUNCTION__);
        syncErrorMessage = "尚未打开音视频源，无法启动推流任务。";
        return false;
    }

    if (!renderPrepareSuccess || renderThreadJoined)
    {
        printf("Error in %s, render not running, cannot launch live streaming\n", __FUNCTION__);
        syncErrorMessage = "尚未启动拼接任务，无法启动推流任务。";
        return false;
    }

    if (!streamThreadJoined)
    {
        printf("Error in %s, live streaming running, stop before launching new live streaming\n", __FUNCTION__);
        syncErrorMessage = "推流任务正在进行中，请先关闭正在执行的任务，再启动新的任务。";
        return false;
    }

    streamURL = name;
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
    streamOpenSuccess = streamWriter.open(streamURL, streamURL.substr(0, 4) == "rtmp" ? "flv" : "rtsp", true,
        audioOpenSuccess, "aac", audioReader.getAudioSampleType(),
        audioReader.getAudioChannelLayout(), audioReader.getAudioSampleRate(), streamAudioBitRate,
        true, videoEncoder == "h264_qsv" ? "h264_qsv" : "h264", pixelType, streamFrameSize.width, streamFrameSize.height,
        videoFrameRate, streamVideoBitRate, writerOpts);
    if (!streamOpenSuccess)
    {
        printf("Could not open streaming url with frame rate = %f and bit rate = %d\n", videoFrameRate, streamVideoBitRate);
        appendLog("流媒体服务器连接失败\n");
        syncErrorMessage = "流媒体服务器连接失败。";
        return false;
    }

    appendLog("流媒体服务器连接成功\n");

    procFrameBufferForSend.resume();
    streamEndFlag = 0;
    streamThreadJoined = 0;
    streamThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::streamSend, this));

    appendLog("推流任务启动成功\n");

    return true;
}

void PanoramaLiveStreamTask::Impl::closeLiveStream()
{
    if (streamOpenSuccess && !streamThreadJoined)
    {
        streamEndFlag = 1;
        procFrameBufferForSend.stop();
        streamThread->join();
        streamThread.reset(0);
        streamOpenSuccess = 0;
        streamThreadJoined = 1;

        appendLog("推流任务结束\n");
        appendLog("断开与流媒体服务器连接\n");
    }
}

bool PanoramaLiveStreamTask::Impl::beginSaveToDisk(const std::string& dir, int width, int height, int videoBPS, 
    const std::string& videoEncoder, const std::string& videoPreset, int audioBPS, int fileDurationInSeconds)
{
    if (!videoOpenSuccess)
    {
        printf("Error in %s, audio video sources not opened\n", __FUNCTION__);
        syncErrorMessage = "尚未打开音视频源，无法启动推流任务。";
        return false;
    }

    if (!renderPrepareSuccess || renderThreadJoined)
    {
        printf("Error in %s, render not running, cannot launch saving to disk\n", __FUNCTION__);
        syncErrorMessage = "尚未启动拼接任务，无法启动保存任务。";
        return false;
    }

    if (!fileThreadJoined)
    {
        printf("Error in %s, saving to disk running, stop before launching new saving to disk\n", __FUNCTION__);
        syncErrorMessage = "保存任务正在进行中，请先关闭正在执行的任务，再启动新的任务。";
        return false;
    }

    fileWriterFormat = dir.empty() ? "temp%d.mp4" : dir + "/temp%d.mp4";
    fileFrameSize.width = width;
    fileFrameSize.height = height;
    fileVideoBitRate = videoBPS;
    fileVideoEncoder = videoEncoder;
    fileVideoEncodePreset = videoPreset;
    fileAudioBitRate = audioBPS;
    fileDuration = fileDurationInSeconds;
    if (fileVideoEncoder != "h264" && fileVideoEncoder != "h264_qsv")
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
    fileThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::fileSave, this));

    appendLog("启动保存视频任务\n");

    return true;
}

void PanoramaLiveStreamTask::Impl::stopSaveToDisk()
{
    if (fileConfigSet && !fileThreadJoined)
    {
        fileEndFlag = 1;
        procFrameBufferForSave.stop();
        fileThread->join();
        fileThread.reset(0);
        fileThreadJoined = 1;
        fileConfigSet = 0;

        appendLog("结束保存视频任务\n");
    }
}

double PanoramaLiveStreamTask::Impl::getVideoSourceFrameRate() const
{
    return videoSourceFrameRate;
}

double PanoramaLiveStreamTask::Impl::getStitchFrameRate() const
{
    return stitchVideoFrameRate;
}

void PanoramaLiveStreamTask::Impl::getLastSyncErrorMessage(std::string& message) const
{
    message = syncErrorMessage;
}

bool PanoramaLiveStreamTask::Impl::hasAsyncErrorMessage() const
{
    return hasAsyncError;
}

void PanoramaLiveStreamTask::Impl::getLastAsyncErrorMessage(std::string& message)
{
    std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
    message = asyncErrorMessage;
}

void PanoramaLiveStreamTask::Impl::getLog(std::string& logInfo)
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

void PanoramaLiveStreamTask::Impl::initAll()
{
    videoSourceFrameRate = 0;
    stitchVideoFrameRate = 0;
    syncErrorMessage.clear();
    clearAsyncErrorMessage();
    clearLog();

    videoFrameRate = 0;
    roundedVideoFrameRate = 0;
    numVideos = 0;
    videoOpenSuccess = 0;
    videoCheckFrameRate = 0;
    videoEndFlag = 0;
    videoThreadsJoined = 1;

    audioSampleRate = 0;
    audioOpenSuccess = 0;
    audioEndFlag = 0;
    audioThreadJoined = 1;

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

void PanoramaLiveStreamTask::Impl::closeAll()
{
    closeVideoDevices();
    closeAudioDevice();
    stopVideoStitch();
    closeLiveStream();
    stopSaveToDisk();

    printf("Live stream task's all threads closed\n");

    // IMPORTANT NOTICE!!!!!!
    // When this library is used in the gui exe, if only video devices are opened, and the following stitch and so on not started,
    // and user quits the program, the program will crash. Finally I find that it is the ptrFrameBuffers.reset(0) that causes the
    // crash. Or ptrFrameBuffers->clear() also causes the crash. If user performs stitch, and then quits, the program does not 
    // crash, so I commented the lines that causes crash.
    // APPEND!!!!!!
    // ptrFrameBuffers.reset(0) is not faulty, other reason causes crash.

    //ptrFrameBuffers.reset(0);
    if (!ptrFrameBuffers)
    {
        for (int i = 0; i < numVideos; i++)
            (*ptrFrameBuffers)[i].clear();
        //ptrFrameBuffers->clear();
    }    
    syncedFramesBufferForShow.clear();
    syncedFramesBufferForProc.clear();
    procFramePool.clear();
    procFrameBufferForShow.clear();
    procFrameBufferForSend.clear();
    procFrameBufferForSave.clear();

    printf("Live stream task's all buffer cleared\n");
}

bool PanoramaLiveStreamTask::Impl::hasFinished() const
{
    return finish;
}

// This variable controls that we check whether frame rate matches the set one
// after first time of synchronization has finished.
//int checkFrameRate = 0;

// This variable controls how frequently the synchronization procedure is called,
// measured in seconds.
const static int syncInterval = 60;

inline void stopCompleteFrameBuffers(std::vector<ForceWaitFrameQueue>* ptrFrameBuffers)
{
    std::vector<ForceWaitFrameQueue>& frameBuffers = *ptrFrameBuffers;
    int numBuffers = frameBuffers.size();
    for (int i = 0; i < numBuffers; i++)
        frameBuffers[i].stop();
}

void PanoramaLiveStreamTask::Impl::videoSource(int index)
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started, index = %d\n", __FUNCTION__, id, index);

    std::vector<ForceWaitFrameQueue>& frameBuffers = *ptrFrameBuffers;
    ForceWaitFrameQueue& buffer = frameBuffers[index];
    avp::AudioVideoReader& reader = videoReaders[index];

    long long int count = 0, beginCheckCount = roundedVideoFrameRate * 5;
    ztool::Timer timer;
    avp::AudioVideoFrame frame;
    bool ok;
    while (true)
    {
        ok = reader.read(frame);
        if (!ok)
        {
            printf("Error in %s [%8x], cannot read video frame\n", __FUNCTION__, id);
            setAsyncErrorMessage("获取视频源数据发生错误，任务终止。");
            stopCompleteFrameBuffers(ptrFrameBuffers.get());
            finish = 1;
            break;
        }

        count++;
        if (count == beginCheckCount)
            timer.start();
        if ((count > beginCheckCount) && (count % roundedVideoFrameRate == 0))
        {
            timer.end();
            double actualFps = (count - beginCheckCount) / timer.elapse();
            //printf("[%8x] fps = %f\n", id, actualFps);
            if (index == 0)
                videoSourceFrameRate = actualFps;
            if (abs(actualFps - videoFrameRate) > 2 && videoCheckFrameRate)
            {
                printf("Error in %s [%8x], actual fps = %f, far away from the set one\n", __FUNCTION__, id, actualFps);
                //buffer.stop();
                //stopCompleteFrameBuffers(ptrFrameBuffers.get());
                //finish = 1;
                //break;
            }
        }

        // NOTICE, for simplicity, I do not check whether the frame has the right property.
        // For the sake of program robustness, we should at least check whether the frame
        // is of type VIDEO, and is not empty, and has the correct pixel type and width and height.
        buffer.push(frame);

        if (finish || videoEndFlag)
        {
            //buffer.stop();
            stopCompleteFrameBuffers(ptrFrameBuffers.get());
            finish = 1;
            break;
        }
    }
    reader.close();

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask::Impl::videoSink()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    std::this_thread::sleep_for(std::chrono::microseconds(100));
    std::vector<ForceWaitFrameQueue>& frameBuffers = *ptrFrameBuffers;

    if (finish || videoEndFlag)
    {
        printf("Thread %s [%8x] end\n", __FUNCTION__, id);
        return;
    }

    for (int i = 0; i < numVideos; i++)
        printf("size = %d\n", frameBuffers[i].size());

    if (finish || videoEndFlag)
    {
        printf("Thread %s [%8x] end\n", __FUNCTION__, id);
        return;
    }

    while (true)
    {
        if (finish || videoEndFlag)
            break;

        long long int currMaxTS = -1;
        int currMaxIndex = -1;
        for (int i = 0; i < numVideos; i++)
        {
            avp::SharedAudioVideoFrame sharedFrame;
            bool ok = frameBuffers[i].pull(sharedFrame);
            if (!ok)
            {
                printf("Error in %s [%8x], pull frame failed\n", __FUNCTION__, id);
                finish = 1;
                break;
            }
            if (sharedFrame.timeStamp > currMaxTS)
            {
                currMaxIndex = i;
                currMaxTS = sharedFrame.timeStamp;
            }
        }

        if (finish || videoEndFlag)
            break;

        std::vector<avp::SharedAudioVideoFrame> syncedFrames(numVideos);
        avp::SharedAudioVideoFrame slowestFrame;
        frameBuffers[currMaxIndex].pull(slowestFrame);
        syncedFrames[currMaxIndex] = slowestFrame;
        printf("slowest ts = %lld\n", slowestFrame.timeStamp);
        for (int i = 0; i < numVideos; i++)
        {
            if (finish || videoEndFlag)
                break;

            if (i == currMaxIndex)
                continue;

            avp::SharedAudioVideoFrame sharedFrame;
            while (true)
            {
                if (finish || videoEndFlag)
                    break;

                bool ok = frameBuffers[i].pull(sharedFrame);
                printf("this ts = %lld\n", sharedFrame.timeStamp);
                if (!ok)
                {
                    printf("Error in %s [%8x], pull frame failed\n", __FUNCTION__, id);
                    finish = 1;
                    break;
                }
                if (sharedFrame.timeStamp > slowestFrame.timeStamp)
                {
                    syncedFrames[i] = sharedFrame;
                    printf("break\n");
                    break;
                }
            }
        }
        if (finish || videoEndFlag)
            break;

        syncedFramesBufferForShow.push(syncedFrames);
        syncedFramesBufferForProc.push(syncedFrames);

        if (!videoCheckFrameRate)
            videoCheckFrameRate = 1;

        int pullCount = 0;
        std::vector<avp::SharedAudioVideoFrame> frames(numVideos);
        while (true)
        {
            if (finish || videoEndFlag)
                break;

            bool ok = true;
            for (int i = 0; i < numVideos; i++)
            {
                if (!frameBuffers[i].pull(frames[i]))
                {
                    printf("Error in %s [%8x], pull frame failed, buffer index %d\n", __FUNCTION__, id, i);
                    ok = false;
                    break;
                }
            }
            if (!ok)
            {
                finish = 1;
                break;
            }

            syncedFramesBufferForShow.push(frames);
            syncedFramesBufferForProc.push(frames);

            pullCount++;
            int needSync = 0;
            if (pullCount == roundedVideoFrameRate * syncInterval)
            {
                printf("Checking frames synchronization status, ");
                long long int maxDiff = 1000000.0 / videoFrameRate * 1.1 + 0.5;
                long long int baseTimeStamp = frames[0].timeStamp;
                for (int i = 1; i < numVideos; i++)
                {
                    if (abs(baseTimeStamp - frames[i].timeStamp) > maxDiff)
                    {
                        needSync = 1;
                        break;
                    }
                }

                if (needSync)
                {
                    printf("frames badly synchronized, resync\n");
                    break;
                }
                else
                {
                    printf("frames well synchronized, continue\n");
                    pullCount = 0;
                }
            }
        }
    }

    //syncedFramesBufferForShow.stop();
#if COMPILE_CUDA
    syncedFramesBufferForProc.stop();
#endif

END:
    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask::Impl::procVideo()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

#if COMPILE_CUDA
    std::vector<cv::cuda::HostMem> mems;
    long long int timeStamp;
#else
    std::vector<avp::SharedAudioVideoFrame> frames;
#endif
    std::vector<cv::Mat> src;
    bool ok;
    int roundedFrameRate = videoFrameRate + 0.5;
    int count = -1;
    ztool::Timer timer;
    while (true)
    {
        ztool::Timer localTimer, procTimer;
        if (finish || renderEndFlag)
            break;
        //printf("show\n");
#if COMPILE_CUDA
        if (!syncedFramesBufferForProc.pull(mems, timeStamp))
        {
            //std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
#else
        if (!syncedFramesBufferForProc.pull(frames))
        {
            continue;
        }
#endif
        //printf("ts %lld\n", timeStamp);
        //printf("before check size\n");
        // NOTICE: it would be better to check frames's pixelType and other properties.
#if COMPILE_CUDA
        if (mems.size() == numVideos)
#else
        if (frames.size() == numVideos)
#endif
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

#if COMPILE_CUDA
            src.resize(numVideos);
            for (int i = 0; i < numVideos; i++)
                src[i] = mems[i].createMatHeader();
            //procTimer.start();
            ok = render.render(src, timeStamp);
            //procTimer.end();
#else
            src.resize(numVideos);
            for (int i = 0; i < numVideos; i++)
                src[i] = cv::Mat(frames[i].height, frames[i].width, elemType, frames[i].data, frames[i].step);
            procTimer.start();
            ok = render.render(src, frames[0].timeStamp);
            procTimer.end();
#endif
            if (!ok)
            {
                printf("Error in %s [%8x], render failed\n", __FUNCTION__, id);
                setAsyncErrorMessage("视频拼接发生错误，任务终止。");
                finish = 1;
                break;
            }

            localTimer.end();
            //printf("%f, %f\n", procTimer.elapse(), localTimer.elapse());
        }        
    }

    render.stop();

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask::Impl::postProc()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    avp::SharedAudioVideoFrame frame;
    while (true)
    {
        if (finish || renderEndFlag)
            break;

        procFramePool.get(frame);
        cv::Mat result(frame.height, frame.width, elemType, frame.data, frame.step);
        
        if (!render.getResult(result, frame.timeStamp))
            continue;

        //ztool::Timer timer;
        logoFilter.addLogo(result);
        procFrameBufferForShow.push(frame);
        if (streamOpenSuccess)
            procFrameBufferForSend.push(frame);
        if (fileConfigSet)
            procFrameBufferForSave.push(frame);
        //timer.end();
        //printf("%f\n", timer.elapse());
    }

    //procFrameBufferForShow.stop();
    procFrameBufferForSend.stop();
    procFrameBufferForSave.stop();

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask::Impl::audioSource()
{
    if (!audioOpenSuccess)
        return;

    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    ztool::Timer timer;
    avp::AudioVideoFrame frame;
    bool ok;
    while (true)
    {
        if (finish || audioEndFlag)
            break;

        ok = audioReader.read(frame);
        if (!ok)
        {
            printf("Error in %s [%8x], cannot read audio frame\n", __FUNCTION__, id);
            setAsyncErrorMessage("获取音频源数据发生错误，任务终止。");
            finish = 1;
            break;
        }

        if (streamOpenSuccess || fileConfigSet)
        {
            avp::SharedAudioVideoFrame deep(frame);
            if (streamOpenSuccess)
                procFrameBufferForSend.push(deep);
            if (fileConfigSet)
                procFrameBufferForSave.push(deep);
        }
    }

    procFrameBufferForSend.stop();
    procFrameBufferForSave.stop();

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask::Impl::streamSend()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    cv::Mat dstMat;
    avp::SharedAudioVideoFrame frame;
    while (true)
    {
        if (finish || streamEndFlag)
            break;
        procFrameBufferForSend.pull(frame);
        if (frame.data)
        {
            avp::AudioVideoFrame shallow;
            //printf("%s, %lld\n", frame.mediaType == avp::VIDEO ? "VIDEO" : "AUDIO", frame.timeStamp);
            if (frame.mediaType == avp::VIDEO && streamFrameSize != renderFrameSize)
            {
                cv::Mat srcMat(renderFrameSize, elemType, frame.data, frame.step);
                cv::resize(srcMat, dstMat, streamFrameSize, 0, 0, cv::INTER_NEAREST);
                shallow = avp::videoFrame(dstMat.data, dstMat.step, pixelType, dstMat.cols, dstMat.rows, frame.timeStamp);
            }
            else
                shallow = frame;
            bool ok = streamWriter.write(shallow);
            if (!ok)
            {
                printf("Error in %s [%8x], cannot write frame\n", __FUNCTION__, id);
                setAsyncErrorMessage("推流发生错误，任务终止。");
                finish = 1;
                break;
            }
        }
    }
    streamWriter.close();

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask::Impl::fileSave()
{
    if (!fileConfigSet)
        return;

    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    char buf[1024];
    int count = 0;
    cv::Mat dstMat;
    avp::SharedAudioVideoFrame frame;
    avp::AudioVideoWriter2 writer;
    std::vector<avp::Option> writerOpts;
    writerOpts.push_back(std::make_pair("preset", fileVideoEncodePreset));
    sprintf(buf, fileWriterFormat.c_str(), count++);
    bool ok = writer.open(buf, "mp4", true,
        audioOpenSuccess, "aac", audioReader.getAudioSampleType(),
        audioReader.getAudioChannelLayout(), audioReader.getAudioSampleRate(), fileAudioBitRate,
        true, fileVideoEncoder, pixelType, fileFrameSize.width, fileFrameSize.height,
        videoFrameRate, fileVideoBitRate, writerOpts);
    if (!ok)
    {
        printf("Error in %s [%d], could not save current audio video\n", __FUNCTION__, id);
        appendLog("保存视频 " + std::string(buf) + "失败\n");
        return;
    }
    else
    {
        appendLog("开始保存视频 " + std::string(buf) + "\n");
    }
    long long int fileFirstTimeStamp = -1;
    while (true)
    {
        if (finish || fileEndFlag)
            break;
        procFrameBufferForSave.pull(frame);
        if (frame.data)
        {
            if (fileFirstTimeStamp < 0)
                fileFirstTimeStamp = frame.timeStamp;

            if (frame.timeStamp - fileFirstTimeStamp > fileDuration * 1000000LL)
            {
                writer.close();
                appendLog("保存视频 " + std::string(buf) + "结束\n");
                sprintf(buf, fileWriterFormat.c_str(), count++);
                ok = writer.open(buf, "mp4", true,
                    audioOpenSuccess, "aac", audioReader.getAudioSampleType(),
                    audioReader.getAudioChannelLayout(), audioReader.getAudioSampleRate(), fileAudioBitRate,
                    true, fileVideoEncoder, pixelType, fileFrameSize.width, fileFrameSize.height,
                    videoFrameRate, fileVideoBitRate, writerOpts);
                if (!ok)
                {
                    printf("Error in %s [%d], could not save current audio video\n", __FUNCTION__, id);
                    appendLog("保存视频 " + std::string(buf) + "失败\n");
                    break;
                }
                else
                {
                    appendLog("开始保存视频 " + std::string(buf) + "\n");
                }
                fileFirstTimeStamp = frame.timeStamp;
            }
            avp::AudioVideoFrame shallow;
            if (frame.mediaType == avp::VIDEO && fileFrameSize != renderFrameSize)
            {
                cv::Mat srcMat(renderFrameSize, elemType, frame.data, frame.step);
                cv::resize(srcMat, dstMat, fileFrameSize, 0, 0, cv::INTER_NEAREST);
                shallow = avp::videoFrame(dstMat.data, dstMat.step, pixelType, dstMat.cols, dstMat.rows, frame.timeStamp);
            }
            else
                shallow = frame;
            ok = writer.write(shallow);
            if (!ok)
            {
                printf("Error in %s [%d], could not write current frame\n", __FUNCTION__, id);
                setAsyncErrorMessage("保存发生错误，任务终止。");
                break;
            }
        }
    }
    appendLog("保存视频 " + std::string(buf) + "结束\n");
    writer.close();

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask::Impl::setAsyncErrorMessage(const std::string& message)
{
    std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
    hasAsyncError = 1;
    asyncErrorMessage = message;
}

void PanoramaLiveStreamTask::Impl::clearAsyncErrorMessage()
{
    std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
    hasAsyncError = 0;
    asyncErrorMessage.clear();
}

void PanoramaLiveStreamTask::Impl::appendLog(const std::string& msg)
{
    std::lock_guard<std::mutex> lg(mtxLog);
    log.append(msg);
}

void PanoramaLiveStreamTask::Impl::clearLog()
{
    std::lock_guard<std::mutex> lg(mtxLog);
    log.clear();
}

PanoramaLiveStreamTask::PanoramaLiveStreamTask()
{
    ptrImpl.reset(new Impl);
}

PanoramaLiveStreamTask::~PanoramaLiveStreamTask()
{

}

bool PanoramaLiveStreamTask::openVideoDevices(const std::vector<avp::Device>& devices, int width, int height, int frameRate, std::vector<int>& success)
{
    return ptrImpl->openVideoDevices(devices, width, height, frameRate, success);
}

void PanoramaLiveStreamTask::closeVideoDevices()
{
    ptrImpl->closeVideoDevices();
}

bool PanoramaLiveStreamTask::openAudioDevice(const avp::Device& device, int sampleRate)
{
    return ptrImpl->openAudioDevice(device, sampleRate);
}

void PanoramaLiveStreamTask::closeAudioDevice()
{
    return ptrImpl->closeAudioDevice();
}

bool PanoramaLiveStreamTask::openVideoStreams(const std::vector<std::string>& urls)
{
    return ptrImpl->openVideoStreams(urls);
}

bool PanoramaLiveStreamTask::openAudioStream(const std::string& url)
{
    return ptrImpl->openAudioStream(url);
}

bool PanoramaLiveStreamTask::beginVideoStitch(const std::string& configFileName, int width, int height, bool highQualityBlend)
{
    return ptrImpl->beginVideoStitch(configFileName, width, height, highQualityBlend);
}

void PanoramaLiveStreamTask::stopVideoStitch()
{
    ptrImpl->stopVideoStitch();
}

bool PanoramaLiveStreamTask::openLiveStream(const std::string& name, int width, int height, int videoBPS, 
    const std::string& videoEncoder, const std::string& videoPreset, int audioBPS)
{
    return ptrImpl->openLiveStream(name, width, height, videoBPS, videoEncoder, videoPreset, audioBPS);
}

void PanoramaLiveStreamTask::closeLiveStream()
{
    ptrImpl->closeLiveStream();
}

bool PanoramaLiveStreamTask::beginSaveToDisk(const std::string& dir, int width, int height, int videoBPS, 
    const std::string& videoEncoder, const std::string& videoPreset, int audioBPS, int fileDuration)
{
    return ptrImpl->beginSaveToDisk(dir, width, height, videoBPS, videoEncoder, videoPreset, audioBPS, fileDuration);
}

void PanoramaLiveStreamTask::stopSaveToDisk()
{
    ptrImpl->stopSaveToDisk();
}

double PanoramaLiveStreamTask::getVideoSourceFrameRate() const
{
    return ptrImpl->getVideoSourceFrameRate();
}

double PanoramaLiveStreamTask::getStitchFrameRate() const
{
    return ptrImpl->getStitchFrameRate();
}

void PanoramaLiveStreamTask::getLastSyncErrorMessage(std::string& message) const
{
    ptrImpl->getLastSyncErrorMessage(message);
}

bool PanoramaLiveStreamTask::hasAsyncErrorMessage() const
{
    return ptrImpl->hasAsyncErrorMessage();
}

void PanoramaLiveStreamTask::getLastAsyncErrorMessage(std::string& message)
{
    ptrImpl->getLastAsyncErrorMessage(message);
}

void PanoramaLiveStreamTask::getLog(std::string& logInfo)
{
    ptrImpl->getLog(logInfo);
}

bool PanoramaLiveStreamTask::getVideoSourceFrames(std::vector<avp::SharedAudioVideoFrame>& frames)
{
    return ptrImpl->getVideoSourceFrames(frames);
}

bool PanoramaLiveStreamTask::getStitchedVideoFrame(avp::SharedAudioVideoFrame& frame)
{
    return ptrImpl->getStitchedVideoFrame(frame);
}

void PanoramaLiveStreamTask::cancelGetVideoSourceFrames()
{
    return ptrImpl->cancelGetVideoSourceFrames();
}

void PanoramaLiveStreamTask::cancelGetStitchedVideoFrame()
{
    return ptrImpl->cancelGetStitchedVideoFrame();
}

void PanoramaLiveStreamTask::initAll()
{
    ptrImpl->initAll();
}

void PanoramaLiveStreamTask::closeAll()
{
    ptrImpl->closeAll();
}

bool PanoramaLiveStreamTask::hasFinished() const
{
    return ptrImpl->hasFinished();
}