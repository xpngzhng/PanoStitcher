#include "PanoramaTask.h"
#include "Timer.h"
#include "Image.h"
#include "opencv2/imgproc/imgproc.hpp"

// for video source
typedef ForceWaitRealTimeQueue<avp::SharedAudioVideoFrame> CompleteFrameQueue;
// for synced video source
typedef ForceWaitRealTimeQueue<std::vector<avp::SharedAudioVideoFrame> > RealTimeFrameVectorQueue;
// for audio source and proc result
typedef ForceWaitRealTimeQueue<avp::SharedAudioVideoFrame> RealTimeFrameQueue;

struct PanoramaLiveStreamTask::Impl
{
    Impl();
    ~Impl();

    bool openVideoDevices(const std::vector<avp::Device>& devices, int width, int height, int frameRate, std::vector<int>& success);
    void closeVideoDevices();

    bool openAudioDevice(const avp::Device& device, int sampleRate);
    void closeAudioDevice();

    bool beginVideoStitch(const std::string& configFileName, int width, int height, bool highQualityBlend);
    void stopVideoStitch();

    bool openLiveStream(const std::string& name,
        int width, int height, int videoBPS, const std::string& videoPreset, int audioBPS);
    void closeLiveStream();

    void beginSaveToDisk(const std::string& dir,
        int width, int height, int videoBPS, const std::string& videoPreset, int audioBPS, int fileDuration);
    void stopSaveToDisk();

    void beginShowVideoSourceFrames(ShowVideoSourceFramesCallbackFunction func, void* data);
    void stopShowVideoSourceFrames();

    void beginShowStitchedFrame(ShowStichedFrameCallbackFunction func, void* data);
    void stopShowStitchedFrame();

    void setVideoSourceFrameRateCallback(FrameRateCallbackFunction func, void* data);
    void setStitchFrameRateCallback(FrameRateCallbackFunction func, void* data);
    void setLogCallback(LogCallbackFunction func, void* data);
    void initCallback();

    bool getLatestVideoSourceFrames(std::vector<avp::SharedAudioVideoFrame>& frames);
    bool getLatestStitchedFrame(avp::SharedAudioVideoFrame& frame);

    bool getVideoSourceFrames(std::vector<avp::SharedAudioVideoFrame>& frames);
    bool getStitchedVideoFrame(avp::SharedAudioVideoFrame& frame);
    void cancelGetVideoSourceFrames();
    void cancelGetStitchedVideoFrame();

    void initAll();
    void closeAll();
    bool hasFinished() const;

    std::vector<avp::AudioVideoReader> videoReaders;
    std::vector<avp::Device> videoDevices;
    cv::Size videoFrameSize;
    int videoFrameRate;
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

    CudaMultiCameraPanoramaRender2 render;
    std::string renderConfigName;
    cv::Size renderFrameSize;
    int renderPrepareSuccess;
    std::unique_ptr<std::thread> renderThread;
    int renderEndFlag;
    int renderThreadJoined;
    void procVideo();

    std::unique_ptr<std::thread> postProcThread;
    void postProc();

    avp::AudioVideoWriter streamWriter;
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

    avp::AudioVideoWriter fileWriter;
    std::string fileWriterFormat;
    cv::Size fileFrameSize;
    int fileVideoBitRate;
    std::string fileVideoEncodePreset;
    int fileAudioBitRate;
    int fileDuration;
    int fileConfigSet;
    std::unique_ptr<std::thread> fileThread;
    int fileEndFlag;
    int fileThreadJoined;
    void fileSave();

    std::unique_ptr<std::thread> showVideoSourceThread;
    int showVideoSourceEndFlag;
    int showVideoSourceThreadJoined;
    void showVideoSource(ShowVideoSourceFramesCallbackFunction func, void* data);

    std::unique_ptr<std::thread> showStitchedThread;
    int showStitchedEndFlag;
    int showStitchedThreadJoined;
    void showStitched(ShowStichedFrameCallbackFunction func, void* data);

    std::mutex videoSourceFramesMutex;
    std::vector<avp::SharedAudioVideoFrame> videoSourceFrames;

    std::mutex stitchedFrameMutex;
    avp::SharedAudioVideoFrame stitchedFrame;

    LogCallbackFunction logCallbackFunc;
    void* logCallbackData;

    FrameRateCallbackFunction videoFrameRateCallbackFunc;
    void* videoFrameRateCallbackData;

    FrameRateCallbackFunction stitchFrameRateCallbackFunc;
    void* stitchFrameRateCallbackData;

    int pixelType;
    int finish;
    std::unique_ptr<std::vector<CompleteFrameQueue> > ptrFrameBuffers;
    RealTimeFrameVectorQueue syncedFramesBufferForShow;
    StampedPinnedMemoryPool syncedFramesBufferForProc;
    SharedAudioVideoFramePool procFramePool;
    RealTimeFrameQueue procFrameBufferForPostProc, procFrameBufferForShow, procFrameBufferForSend, procFrameBufferForSave;
};


PanoramaLiveStreamTask::Impl::Impl()
{
    initAll();
    initCallback();
}

PanoramaLiveStreamTask::Impl::~Impl()
{
    closeAll();
    printf("live stream task destructore called\n");
}

bool PanoramaLiveStreamTask::Impl::openVideoDevices(const std::vector<avp::Device>& devices, int width, int height, int frameRate, std::vector<int>& success)
{
    videoDevices = devices;
    videoFrameSize.width = width;
    videoFrameSize.height = height;
    videoFrameRate = frameRate;
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
            false, true, avp::PixelTypeBGR32, "dshow", opts);
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
        if (logCallbackFunc)
            logCallbackFunc("Video sources open failed", logCallbackData);
        return false;
    }

    if (logCallbackFunc)
        logCallbackFunc("Video sources open success", logCallbackData);

    ptrFrameBuffers.reset(new std::vector<CompleteFrameQueue>(numVideos));
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

    if (logCallbackFunc)
        logCallbackFunc("Video sources related threads create success\n", logCallbackData);

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

        if (logCallbackFunc)
        {
            logCallbackFunc("Video sources related threads close success", logCallbackData);
            logCallbackFunc("Video sources close success", logCallbackData);
        }
    }
}

bool PanoramaLiveStreamTask::Impl::openAudioDevice(const avp::Device& device, int sampleRate)
{
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

        if (logCallbackFunc)
            logCallbackFunc("Audio source open failed", logCallbackData);

        return false;
    }

    if (logCallbackFunc)
        logCallbackFunc("Audio source open success", logCallbackData);

    procFrameBufferForSave.clear();
    procFrameBufferForSend.clear();

    audioEndFlag = 0;
    audioThreadJoined = 0;
    audioThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::audioSource, this));

    if (logCallbackFunc)
        logCallbackFunc("Audio source thread create success", logCallbackData);

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

        if (logCallbackFunc)
        {
            logCallbackFunc("Audio source thread close success", logCallbackData);
            logCallbackFunc("Audio source close success", logCallbackData);
        }
    }
}

bool PanoramaLiveStreamTask::Impl::beginVideoStitch(const std::string& configFileName, int width, int height, bool highQualityBlend)
{
    pixelType = avp::PixelTypeBGR32;
    renderConfigName = configFileName;
    renderFrameSize.width = width;
    renderFrameSize.height = height;

    renderPrepareSuccess = render.prepare(renderConfigName, 
        highQualityBlend ? PanoramaRender::BlendTypeMultiband : PanoramaRender::BlendTypeLinear, 
        videoFrameSize, renderFrameSize);

    if (!renderPrepareSuccess)
    {
        printf("Could not prepare for video stitch\n");

        if (logCallbackFunc)
            logCallbackFunc("Video stitch prepare failed", logCallbackData);

        return false;
    }

    renderPrepareSuccess = procFramePool.initAsVideoFramePool(pixelType, width, height);

    if (!renderPrepareSuccess)
    {
        printf("Could not init proc frame pool\n");

        if (logCallbackFunc)
            logCallbackFunc("Video stitch prepare failed", logCallbackData);

        return false;
    }

    if (logCallbackFunc)
        logCallbackFunc("Video stitch prepare success", logCallbackData);
    
    syncedFramesBufferForProc.resume();
    procFrameBufferForPostProc.clear();
    procFrameBufferForShow.clear();
    if (!audioOpenSuccess)
    {
        procFrameBufferForSave.clear();
        procFrameBufferForSend.clear();
    }

    renderEndFlag = 0;
    renderThreadJoined = 0;
    renderThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::procVideo, this));
    postProcThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::postProc, this));

    if (logCallbackFunc)
        logCallbackFunc("Video stitch thread create success", logCallbackData);

    return true;
}

void PanoramaLiveStreamTask::Impl::stopVideoStitch()
{
    if (renderPrepareSuccess && !renderThreadJoined)
    {
        renderEndFlag = 1;
        syncedFramesBufferForProc.stop();
        renderThread->join();
        renderThread.reset(0);
        procFrameBufferForPostProc.stop();
        postProcThread->join();
        postProcThread.reset(0);
        renderPrepareSuccess = 0;
        renderThreadJoined = 1;

        if (logCallbackFunc)
            logCallbackFunc("Video stitch thread close success", logCallbackData);
    }
}

bool PanoramaLiveStreamTask::Impl::getLatestVideoSourceFrames(std::vector<avp::SharedAudioVideoFrame>& frames)
{
    if (finish)
    {
        frames.clear();
        return false;
    }
    {
        std::lock_guard<std::mutex> lg(videoSourceFramesMutex);

        if (videoSourceFrames.size() != numVideos)
            return false;

        frames.resize(numVideos);
        for (int i = 0; i < numVideos; i++)
        {
            avp::AudioVideoFrame shallow = videoSourceFrames[i];
            frames[i] = shallow;
        }
    }
    return true;
}

bool PanoramaLiveStreamTask::Impl::getLatestStitchedFrame(avp::SharedAudioVideoFrame& frame)
{
    if (finish)
    {
        frame = avp::SharedAudioVideoFrame();
        return false;
    }
    {
        std::lock_guard<std::mutex> lg(stitchedFrameMutex);
        avp::AudioVideoFrame shallow = stitchedFrame;
        frame = shallow;
    }
    return true;
}

bool PanoramaLiveStreamTask::Impl::getVideoSourceFrames(std::vector<avp::SharedAudioVideoFrame>& frames)
{
    return syncedFramesBufferForShow.pull(frames);
}

bool PanoramaLiveStreamTask::Impl::getStitchedVideoFrame(avp::SharedAudioVideoFrame& frame)
{
    return procFrameBufferForShow.pull(frame);
}

void PanoramaLiveStreamTask::Impl::cancelGetVideoSourceFrames()
{
    syncedFramesBufferForShow.stop();
}

void PanoramaLiveStreamTask::Impl::cancelGetStitchedVideoFrame()
{
    procFrameBufferForShow.stop();
}

bool PanoramaLiveStreamTask::Impl::openLiveStream(const std::string& name,
    int width, int height, int videoBPS, const std::string& videoPreset, int audioBPS)
{
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
        true, "h264", pixelType, streamFrameSize.width, streamFrameSize.height,
        videoFrameRate, streamVideoBitRate, writerOpts);
    if (!streamOpenSuccess)
    {
        printf("Could not open streaming url with frame rate = %d and bit rate = %d\n", videoFrameRate, streamVideoBitRate);

        if (logCallbackFunc)
            logCallbackFunc("Live stream open failed", logCallbackData);

        return false;
    }

    if (logCallbackFunc)
        logCallbackFunc("Live stream open success", logCallbackData);

    streamEndFlag = 0;
    streamThreadJoined = 0;
    streamThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::streamSend, this));

    if (logCallbackFunc)
        logCallbackFunc("Live stream thread create success", logCallbackData);

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

        if (logCallbackFunc)
        {
            logCallbackFunc("Live stream close success", logCallbackData);
            logCallbackFunc("Live stream thread close success", logCallbackData);
        }
    }
}

void PanoramaLiveStreamTask::Impl::beginSaveToDisk(const std::string& dir,
    int width, int height, int videoBPS, const std::string& videoPreset, int audioBPS, int fileDurationInSeconds)
{
    fileWriterFormat = dir + "/temp%d.mp4";
    fileFrameSize.width = width;
    fileFrameSize.height = height;
    fileVideoBitRate = videoBPS;
    fileVideoEncodePreset = videoPreset;
    fileAudioBitRate = audioBPS;
    fileDuration = fileDurationInSeconds;
    if (fileVideoEncodePreset != "ultrafast" || fileVideoEncodePreset != "superfast" ||
        fileVideoEncodePreset != "veryfast" || fileVideoEncodePreset != "faster" ||
        fileVideoEncodePreset != "fast" || fileVideoEncodePreset != "medium" || fileVideoEncodePreset != "slow" ||
        fileVideoEncodePreset != "slower" || fileVideoEncodePreset != "veryslow")
        fileVideoEncodePreset = "veryfast";
    fileConfigSet = 1;

    fileEndFlag = 0;
    fileThreadJoined = 0;
    fileThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::fileSave, this));
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
    }
}

void PanoramaLiveStreamTask::Impl::beginShowVideoSourceFrames(ShowVideoSourceFramesCallbackFunction func, void* data)
{
    showVideoSourceEndFlag = 0;
    showVideoSourceThreadJoined = 0;
    showVideoSourceThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::showVideoSource, this, func, data));
    //if (logCallbackFunc)
    //    logCallbackFunc("Show video source thread create success", logCallbackData);
}

void PanoramaLiveStreamTask::Impl::stopShowVideoSourceFrames()
{
    if (showVideoSourceThread && !showVideoSourceThreadJoined)
    {
        showVideoSourceEndFlag = 1;
        syncedFramesBufferForShow.stop();
        showVideoSourceThread->join();
        showVideoSourceThread.reset(0);
        showVideoSourceThreadJoined = 1;
    }    
}

void PanoramaLiveStreamTask::Impl::beginShowStitchedFrame(ShowStichedFrameCallbackFunction func, void* data)
{
    showStitchedEndFlag = 0;
    showStitchedThreadJoined = 0;
    showStitchedThread.reset(new std::thread(&PanoramaLiveStreamTask::Impl::showStitched, this, func, data));
}

void PanoramaLiveStreamTask::Impl::stopShowStitchedFrame()
{
    if (showStitchedThread && !showStitchedThreadJoined)
    {
        showStitchedEndFlag = 1;
        procFrameBufferForShow.stop();
        showStitchedThread->join();
        showStitchedThread.reset(0);
        showStitchedThreadJoined = 1;
    }
}

void PanoramaLiveStreamTask::Impl::initAll()
{
    videoFrameRate = 0;
    numVideos = 0;
    videoOpenSuccess = 0;
    videoCheckFrameRate = 0;
    videoEndFlag = 0;
    videoThreadsJoined = 0;

    audioSampleRate = 0;
    audioOpenSuccess = 0;
    audioEndFlag = 0;
    audioThreadJoined = 0;

    renderPrepareSuccess = 0;
    renderEndFlag = 0;
    renderThreadJoined = 0;

    streamVideoBitRate = 0;
    streamAudioBitRate = 0;
    streamOpenSuccess = 0;
    streamEndFlag = 0;
    streamThreadJoined = 0;

    fileVideoBitRate = 0;
    fileAudioBitRate = 0;
    fileDuration = 0;
    fileConfigSet = 0;
    fileEndFlag = 0;
    fileThreadJoined = 0;

    showVideoSourceEndFlag = 0;
    showVideoSourceThreadJoined = 0;

    showStitchedEndFlag = 0;
    showStitchedThreadJoined = 0;

    finish = 0;
}

void PanoramaLiveStreamTask::Impl::closeAll()
{
    closeVideoDevices();
    closeAudioDevice();
    stopVideoStitch();
    closeLiveStream();
    stopSaveToDisk();
    stopShowVideoSourceFrames();
    stopShowStitchedFrame();

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
    procFrameBufferForPostProc.clear();
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

inline void stopCompleteFrameBuffers(std::vector<CompleteFrameQueue>* ptrFrameBuffers)
{
    std::vector<CompleteFrameQueue>& frameBuffers = *ptrFrameBuffers;
    int numBuffers = frameBuffers.size();
    for (int i = 0; i < numBuffers; i++)
        frameBuffers[i].stop();
}

void PanoramaLiveStreamTask::Impl::videoSource(int index)
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started, index = %d\n", __FUNCTION__, id, index);

    std::vector<CompleteFrameQueue>& frameBuffers = *ptrFrameBuffers;
    CompleteFrameQueue& buffer = frameBuffers[index];
    avp::AudioVideoReader& reader = videoReaders[index];

    long long int count = 0, beginCheckCount = videoFrameRate * 5;
    ztool::Timer timer;
    avp::AudioVideoFrame frame;
    bool ok;
    while (true)
    {
        ok = reader.read(frame);
        if (!ok)
        {
            printf("Error in %s [%8x], cannot read video frame\n", __FUNCTION__, id);
            stopCompleteFrameBuffers(ptrFrameBuffers.get());
            finish = 1;
            break;
        }

        count++;
        if (count == beginCheckCount)
            timer.start();
        if ((count > beginCheckCount) && (count % videoFrameRate == 0))
        {
            timer.end();
            double actualFps = (count - beginCheckCount) / timer.elapse();
            //printf("[%8x] fps = %f\n", id, actualFps);
            if (index == 0 && videoFrameRateCallbackFunc)
                videoFrameRateCallbackFunc(actualFps, videoFrameRateCallbackData);
            if (abs(actualFps - videoFrameRate) > 2 && videoCheckFrameRate)
            {
                printf("Error in %s [%8x], fps far away from the set one\n", __FUNCTION__, id);
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
    std::vector<CompleteFrameQueue>& frameBuffers = *ptrFrameBuffers;

    if (finish || videoEndFlag)
    {
        printf("Thread %s [%8x] end\n", __FUNCTION__, id);
        return;
    }

    for (int i = 0; i < numVideos; i++)
        printf("size = %d\n", frameBuffers[i].size());

    for (int j = 0; j < 25; j++)
    {
        for (int i = 0; i < numVideos; i++)
        {
            avp::SharedAudioVideoFrame sharedFrame;
            frameBuffers[i].pull(sharedFrame);
        }
    }

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
            frameBuffers[i].pull(sharedFrame);
            if (sharedFrame.timeStamp < 0)
            {
                printf("Error in %s [%8x], cannot read valid frame with non-negative time stamp\n", __FUNCTION__, id);
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

                frameBuffers[i].pull(sharedFrame);
                printf("this ts = %lld\n", sharedFrame.timeStamp);
                if (sharedFrame.timeStamp < 0)
                {
                    printf("Error in %s [%8x], cannot read valid frame with non-negative time stamp\n", __FUNCTION__, id);
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

        {
            std::lock_guard<std::mutex> lg(videoSourceFramesMutex);
            videoSourceFrames = syncedFrames;
        }

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

            for (int i = 0; i < numVideos; i++)
            {
                frameBuffers[i].pull(frames[i]);
                //printf("%d ", frameBuffers[i].size());
            }
            //printf("\n");

            {
                std::lock_guard<std::mutex> lg(videoSourceFramesMutex);
                videoSourceFrames = frames;
            }

            syncedFramesBufferForShow.push(frames);
            syncedFramesBufferForProc.push(frames);

            pullCount++;
            int needSync = 0;
            if (pullCount == videoFrameRate * syncInterval)
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

    syncedFramesBufferForShow.stop();
    syncedFramesBufferForProc.stop();

END:
    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask::Impl::procVideo()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    std::vector<cv::cuda::HostMem> mems;
    long long int timeStamp;
    std::vector<cv::Mat> src;
    avp::SharedAudioVideoFrame frame;
    cv::Mat result;
    bool ok;
    int roundedFrameRate = videoFrameRate + 0.5;
    int count = -1;
    ztool::Timer timer;
    while (true)
    {
        if (finish || renderEndFlag)
            break;
        //printf("show\n");
        if (!syncedFramesBufferForProc.pull(mems, timeStamp))
        {
            //std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        //printf("ts %lld\n", timeStamp);
        //printf("before check size\n");
        // NOTICE: it would be better to check frames's pixelType and other properties.
        if (mems.size() == numVideos)
        {
            ztool::Timer localTimer, procTimer;
            if (count < 0)
            {
                count = 0;
                timer.start();
            }
            else
            {
                count++;
                if (count == roundedFrameRate)
                {
                    printf("%d ", count);
                    timer.end();
                    double elapse = timer.elapse();
                    printf(" %f, %f\n", elapse, count / elapse);
                    timer.start();
                    count = 0;

                    if (stitchFrameRateCallbackFunc)
                        stitchFrameRateCallbackFunc(roundedFrameRate / elapse, stitchFrameRateCallbackData);
                }
            }

            src.resize(numVideos);
            for (int i = 0; i < numVideos; i++)
                src[i] = mems[i].createMatHeader();
            //frame = avp::sharedVideoFrame(pixelType, renderFrameSize.width, renderFrameSize.height, timeStamp);
            procFramePool.get(frame);
            result = cv::Mat(frame.height, frame.width, CV_8UC4, frame.data, frame.step);
            procTimer.start();
            ok = render.render(src, result);
            procTimer.end();
            if (!ok)
            {
                printf("Error in %s [%8x], render failed\n", __FUNCTION__, id);
                finish = 1;
                break;
            }
            frame.timeStamp = timeStamp;
            procFrameBufferForPostProc.push(frame);

            localTimer.end();
            //printf("%f, %f\n", procTimer.elapse(), localTimer.elapse());
        }
    }

    procFrameBufferForPostProc.stop();

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask::Impl::postProc()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    cv::Mat origImage(128, 256, CV_8UC4, imageData8UC4);
    cv::Mat origMask(128, 256, CV_8UC1, maskData);
    cv::Mat addImage;
    cv::Mat addMask;
    cv::Rect addROI;
    if (renderFrameSize.width <= 256 || renderFrameSize.height <= 128)
    {
        cv::resize(origImage, addImage, renderFrameSize);
        cv::resize(origMask, addMask, renderFrameSize);
        addROI = cv::Rect(0, 0, renderFrameSize.width, renderFrameSize.height);
    }
    else
    {
        addImage = origImage;
        addMask = origMask;
        addROI = cv::Rect(renderFrameSize.width / 2 - 128, renderFrameSize.height / 2 - 64, 256, 128);
    }

    avp::SharedAudioVideoFrame frame;
    while (true)
    {
        if (finish || renderEndFlag)
            break;

        if (!procFrameBufferForPostProc.pull(frame))
            continue;

        //ztool::Timer timer;
        cv::Mat result(frame.height, frame.width, CV_8UC4, frame.data, frame.step);
        cv::Mat resultROI = result(addROI);
        addImage.copyTo(resultROI, addMask);
        {
            std::lock_guard<std::mutex> lg(stitchedFrameMutex);
            stitchedFrame = frame;
        }
        procFrameBufferForShow.push(frame);
        if (streamOpenSuccess)
            procFrameBufferForSend.push(frame);
        if (fileConfigSet)
            procFrameBufferForSave.push(frame);
        //timer.end();
        //printf("%f\n", timer.elapse());
    }

    procFrameBufferForShow.stop();
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
            if (frame.mediaType == avp::VIDEO && streamFrameSize != renderFrameSize)
            {
                cv::Mat srcMat(renderFrameSize, pixelType == avp::PixelTypeBGR24 ? CV_8UC3 : CV_8UC4, frame.data, frame.step);
                cv::resize(srcMat, dstMat, streamFrameSize, 0, 0, cv::INTER_NEAREST);
                shallow = avp::videoFrame(dstMat.data, dstMat.step, avp::PixelTypeBGR24, dstMat.cols, dstMat.rows, frame.timeStamp);
            }
            else
                shallow = frame;
            bool ok = streamWriter.write(shallow);
            if (!ok)
            {
                printf("Error in %s [%8x], cannot write frame\n", __FUNCTION__, id);
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
    avp::AudioVideoWriter writer;
    std::vector<avp::Option> writerOpts;
    writerOpts.push_back(std::make_pair("preset", fileVideoEncodePreset));
    sprintf(buf, fileWriterFormat.c_str()/*"temp%d.mp4"*/, count++);
    bool ok = writer.open(buf, "mp4", true,
        audioOpenSuccess, "aac", audioReader.getAudioSampleType(),
        audioReader.getAudioChannelLayout(), audioReader.getAudioSampleRate(), fileAudioBitRate,
        true, "h264", pixelType, fileFrameSize.width, fileFrameSize.height,
        videoFrameRate, fileVideoBitRate, writerOpts);
    if (!ok)
    {
        printf("Error in %s [%d], could not save current audio video\n", __FUNCTION__, id);
        if (logCallbackFunc)
            logCallbackFunc(std::string("Could not write local file ") + buf, logCallbackData);
        return;
    }
    else
    {
        if (logCallbackFunc)
            logCallbackFunc(std::string("Begin write local file ") + buf, logCallbackData);
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
                if (logCallbackFunc)
                    logCallbackFunc(std::string("Finish write local file ") + buf, logCallbackData);
                sprintf(buf, fileWriterFormat.c_str()/*"temp%d.mp4"*/, count++);
                ok = writer.open(buf, "mp4", true,
                    audioOpenSuccess, "aac", audioReader.getAudioSampleType(),
                    audioReader.getAudioChannelLayout(), audioReader.getAudioSampleRate(), fileAudioBitRate,
                    true, "h264", pixelType, fileFrameSize.width, fileFrameSize.height,
                    videoFrameRate, fileVideoBitRate, writerOpts);
                if (!ok)
                {
                    printf("Error in %s [%d], could not save current audio video\n", __FUNCTION__, id);
                    if (logCallbackFunc)
                        logCallbackFunc(std::string("Could not write local file ") + buf, logCallbackData);
                    break;
                }
                else
                {
                    if (logCallbackFunc)
                        logCallbackFunc(std::string("Begin write local file ") + buf, logCallbackData);
                }
                fileFirstTimeStamp = frame.timeStamp;
            }
            avp::AudioVideoFrame shallow;
            if (frame.mediaType == avp::VIDEO && fileFrameSize != renderFrameSize)
            {
                cv::Mat srcMat(renderFrameSize, pixelType == avp::PixelTypeBGR24 ? CV_8UC3 : CV_8UC4, frame.data, frame.step);
                cv::resize(srcMat, dstMat, fileFrameSize, 0, 0, cv::INTER_NEAREST);
                shallow = avp::videoFrame(dstMat.data, dstMat.step, avp::PixelTypeBGR24, dstMat.cols, dstMat.rows, frame.timeStamp);
            }
            else
                shallow = frame;
            ok = writer.write(shallow);
            if (!ok)
            {
                printf("Error in %s [%d], could not write current frame\n", __FUNCTION__, id);
                break;
            }
        }
    }
    if (logCallbackFunc)
        logCallbackFunc(std::string("Finish write local file ") + buf, logCallbackData);
    writer.close();

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask::Impl::showVideoSource(ShowVideoSourceFramesCallbackFunction func, void* data)
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    int waitTime = (videoFrameRate ? std::max(int(1000.0 / videoFrameRate + 0.5) - 2, 5) : 10);
    std::vector<avp::SharedAudioVideoFrame> frames;
    while (true)
    {
        if (finish || showVideoSourceEndFlag)
            break;

        if (func && data)
        {
            if (syncedFramesBufferForShow.pull(frames))
                func(frames, data);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
    }

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask::Impl::showStitched(ShowStichedFrameCallbackFunction func, void* data)
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    int waitTime = (videoFrameRate ? std::max(int(1000.0 / videoFrameRate + 0.5) - 2, 5) : 10);
    avp::SharedAudioVideoFrame frame;
    while (true)
    {
        if (finish || showStitchedEndFlag)
            break;

        if (func && data)
        {
            if (procFrameBufferForShow.pull(frame))
                func(frame, data);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
    }

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask::Impl::setVideoSourceFrameRateCallback(FrameRateCallbackFunction func, void* data)
{
    videoFrameRateCallbackFunc = func;
    videoFrameRateCallbackData = data;
}

void PanoramaLiveStreamTask::Impl::setStitchFrameRateCallback(FrameRateCallbackFunction func, void* data)
{
    stitchFrameRateCallbackFunc = func;
    stitchFrameRateCallbackData = data;
}

void PanoramaLiveStreamTask::Impl::setLogCallback(LogCallbackFunction func, void* data)
{
    logCallbackFunc = func;
    logCallbackData = data;
}

void PanoramaLiveStreamTask::Impl::initCallback()
{
    logCallbackFunc = 0;
    videoFrameRateCallbackFunc = 0;
    stitchFrameRateCallbackFunc = 0;
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

bool PanoramaLiveStreamTask::beginVideoStitch(const std::string& configFileName, int width, int height, bool highQualityBlend)
{
    return ptrImpl->beginVideoStitch(configFileName, width, height, highQualityBlend);
}

void PanoramaLiveStreamTask::stopVideoStitch()
{
    ptrImpl->stopVideoStitch();
}

bool PanoramaLiveStreamTask::openLiveStream(const std::string& name,
    int width, int height, int videoBPS, const std::string& videoPreset, int audioBPS)
{
    return ptrImpl->openLiveStream(name, width, height, videoBPS, videoPreset, audioBPS);
}

void PanoramaLiveStreamTask::closeLiveStream()
{
    ptrImpl->closeLiveStream();
}

void PanoramaLiveStreamTask::beginSaveToDisk(const std::string& dir,
    int width, int height, int videoBPS, const std::string& videoPreset, int audioBPS, int fileDuration)
{
    ptrImpl->beginSaveToDisk(dir, width, height, videoBPS, videoPreset, audioBPS, fileDuration);
}

void PanoramaLiveStreamTask::stopSaveToDisk()
{
    ptrImpl->stopSaveToDisk();
}

void PanoramaLiveStreamTask::beginShowVideoSourceFrames(ShowVideoSourceFramesCallbackFunction func, void* data)
{
    ptrImpl->beginShowVideoSourceFrames(func, data);
}

void PanoramaLiveStreamTask::stopShowVideoSourceFrames()
{
    ptrImpl->stopShowVideoSourceFrames();
}

void PanoramaLiveStreamTask::beginShowStitchedFrame(ShowStichedFrameCallbackFunction func, void* data)
{
    ptrImpl->beginShowStitchedFrame(func, data);
}

void PanoramaLiveStreamTask::stopShowStitchedFrame()
{
    ptrImpl->stopShowStitchedFrame();
}

void PanoramaLiveStreamTask::setVideoSourceFrameRateCallback(FrameRateCallbackFunction func, void* data)
{
    ptrImpl->setVideoSourceFrameRateCallback(func, data);
}

void PanoramaLiveStreamTask::setStitchFrameRateCallback(FrameRateCallbackFunction func, void* data)
{
    ptrImpl->setStitchFrameRateCallback(func, data);
}

void PanoramaLiveStreamTask::setLogCallback(LogCallbackFunction func, void* data)
{
    ptrImpl->setLogCallback(func, data);
}

void PanoramaLiveStreamTask::initCallback()
{
    ptrImpl->initCallback();
}

bool PanoramaLiveStreamTask::getLatestVideoSourceFrames(std::vector<avp::SharedAudioVideoFrame>& frames)
{
    return ptrImpl->getLatestVideoSourceFrames(frames);
}

bool PanoramaLiveStreamTask::getLatestStitchedFrame(avp::SharedAudioVideoFrame& frame)
{
    return ptrImpl->getLatestStitchedFrame(frame);
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