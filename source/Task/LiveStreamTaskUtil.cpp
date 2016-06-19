#include "PanoramaTaskUtil.h"
#include "LiveStreamTaskUtil.h"
#include "Timer.h"

bool isIPAddress(const std::string& text)
{
    if (text.empty())
        return false;

    std::string::size_type size = text.size();
    // Assuming we use ipv4
    if (size > 15)
        return false;

    const char* ptr = text.data();
    int numDots = 0;
    std::string::size_type pos[3];
    for (std::string::size_type i = 0; i < size; i++)
    {
        if (isdigit(ptr[i]))
            continue;
        if (ptr[i] == '.')
        {
            if (numDots == 3)
                return false;
            pos[numDots] = i;
            numDots++;
        }
        else
            return false;
    }
    if (numDots != 3)
        return false;
    if (pos[0] > 3 || pos[1] - pos[0] > 4 || pos[2] - pos[1] > 4 || size - pos[2] > 4)
        return false;
    return true;
}

bool areAllIPAdresses(const std::vector<std::string>& texts)
{
    if (texts.empty())
        return false;
    int size = texts.size();
    for (int i = 0; i < size; i++)
    {
        if (!isIPAddress(texts[i]))
            return false;
    }
    return true;
}

bool areAllNotIPAdresses(const std::vector<std::string>& texts)
{
    if (texts.empty())
        return true;
    int size = texts.size();
    for (int i = 0; i < size; i++)
    {
        if (isIPAddress(texts[i]))
            return false;
    }
    return true;
}

bool isURL(const std::string& text)
{
    std::string::size_type size = text.size();
    if (size <= 4)
        return false;
    std::string beg = text.substr(0, 4);
    return (beg == "http") || (beg == "rtsp") || (beg == "rtmp");
}

bool areAllURLs(const std::vector<std::string>& texts)
{
    if (texts.empty())
        return false;
    int size = texts.size();
    for (int i = 0; i < size; i++)
    {
        if (!isURL(texts[i]))
            return false;
    }
    return true;
}

bool areAllNotURLs(const std::vector<std::string>& texts)
{
    if (texts.empty())
        return true;
    int size = texts.size();
    for (int i = 0; i < size; i++)
    {
        if (isURL(texts[i]))
            return false;
    }
    return true;
}

AudioVideoSource::AudioVideoSource()
{

}

AudioVideoSource::~AudioVideoSource()
{

}

void AudioVideoSource::setProp(ForShowFrameVectorQueue* ptrSyncedFramesBufferForShow_,
    void* ptrSyncedFramesBufferForProc_, int forCuda_,
    ForceWaitFrameQueue* ptrProcFrameBufferForSend_, ForceWaitFrameQueue* ptrProcFrameBufferForSave_, 
    int* ptrFinish_, LogCallbackFunction logCallbackFunc_, void* logCallbackData_,
    FrameRateCallbackFunction videoFrameRateCallbackFunc_, void* videoFrameRateCallbackData_)
{
    pixelType = avp::PixelTypeBGR32;
    ptrFinish = ptrFinish_;
    logCallbackFunc = logCallbackFunc_;
    logCallbackData = logCallbackData_;
    videoFrameRateCallbackFunc = videoFrameRateCallbackFunc_;
    videoFrameRateCallbackData = videoFrameRateCallbackData_;
    ptrSyncedFramesBufferForShow = ptrSyncedFramesBufferForShow_;
    ptrSyncedFramesBufferForProc = ptrSyncedFramesBufferForProc_;
    ptrProcFrameBufferForSend = ptrProcFrameBufferForSend_;
    ptrProcFrameBufferForSave = ptrProcFrameBufferForSave_;
    forCuda = forCuda_;
}

bool AudioVideoSource::isVideoOpened() const
{
    return videoOpenSuccess != 0;
}

bool AudioVideoSource::isAudioOpened() const
{
    return audioOpenSuccess != 0;
}

bool AudioVideoSource::isRunning() const
{
    return running != 0;
}

void AudioVideoSource::init()
{
    ptrFinish = 0;
    finish = 0;
    running = 0;
    forCuda = 0;

    videoOpenSuccess = 0;
    videoEndFlag = 0;
    videoThreadsJoined = 0;

    audioOpenSuccess = 0;
    audioEndFlag = 0;
    audioThreadJoined = 0;

    logCallbackFunc = 0;
    logCallbackData = 0;

    videoFrameRateCallbackFunc = 0;
    videoFrameRateCallbackData = 0;
}

static int syncInterval = 60;

void AudioVideoSource::videoSink()
{
    size_t id = std::this_thread::get_id().hash();
    ptlprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    std::this_thread::sleep_for(std::chrono::microseconds(100));
    std::vector<ForceWaitFrameQueue>& frameBuffers = *ptrFrameBuffers;
    ForShowFrameVectorQueue& syncedFramesBufferForShow = *ptrSyncedFramesBufferForShow;
    BoundedPinnedMemoryFrameQueue& syncedFramesBufferForProcCuda = *(BoundedPinnedMemoryFrameQueue*)ptrSyncedFramesBufferForProc;
    ForShowFrameVectorQueue& syncedFramesBufferForProcIOcl = *(ForShowFrameVectorQueue*)ptrSyncedFramesBufferForProc;

    if (finish || videoEndFlag)
    {
        ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
        return;
    }

    for (int i = 0; i < numVideos; i++)
        ptlprintf("size = %d\n", frameBuffers[i].size());

    if (finish || videoEndFlag)
    {
        ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
        return;
    }

    while (true)
    {
        if (finish || videoEndFlag)
            break;

        // Initialize currMaxTS to the smallest value,
        // if video sources are local video files, the first several frames may
        // have negative time stamps
        long long int currMaxTS = 0x8000000000000000;
        int currMaxIndex = -1;
        for (int i = 0; i < numVideos; i++)
        {
            avp::AudioVideoFrame2 sharedFrame;
            bool ok = frameBuffers[i].pull(sharedFrame);
            if (!ok)
            {
                ptlprintf("Error in %s [%8x], pull frame failed\n", __FUNCTION__, id);
                finish = 1;
                break;
            }
            if (sharedFrame.timeStamp > currMaxTS)
            {
                currMaxIndex = i;
                currMaxTS = sharedFrame.timeStamp;
            }
        }
        if (currMaxIndex < 0)
        {
            ptlprintf("Error in %s [%8x], failed to find the frame with smallest time stamp\n", __FUNCTION__, id);
            finish = 1;
        }

        if (finish || videoEndFlag)
            break;

        std::vector<avp::AudioVideoFrame2> syncedFrames(numVideos);
        avp::AudioVideoFrame2 slowestFrame;
        frameBuffers[currMaxIndex].pull(slowestFrame);
        syncedFrames[currMaxIndex] = slowestFrame;
        ptlprintf("slowest ts = %lld\n", slowestFrame.timeStamp);
        for (int i = 0; i < numVideos; i++)
        {
            if (finish || videoEndFlag)
                break;

            if (i == currMaxIndex)
                continue;

            avp::AudioVideoFrame2 sharedFrame;
            while (true)
            {
                if (finish || videoEndFlag)
                    break;

                bool ok = frameBuffers[i].pull(sharedFrame);
                ptlprintf("this ts = %lld\n", sharedFrame.timeStamp);
                if (!ok)
                {
                    ptlprintf("Error in %s [%8x], pull frame failed\n", __FUNCTION__, id);
                    finish = 1;
                    break;
                }
                if (sharedFrame.timeStamp >= slowestFrame.timeStamp)
                {
                    syncedFrames[i] = sharedFrame;
                    ptlprintf("break\n");
                    break;
                }
            }
        }
        if (finish || videoEndFlag)
            break;

        syncedFramesBufferForShow.push(syncedFrames);
        if (forCuda)
            syncedFramesBufferForProcCuda.push(syncedFrames);
        else
            syncedFramesBufferForProcIOcl.push(syncedFrames);

        if (!videoCheckFrameRate)
            videoCheckFrameRate = 1;

        int pullCount = 0;
        std::vector<avp::AudioVideoFrame2> frames(numVideos);
        while (true)
        {
            if (finish || videoEndFlag)
                break;

            bool ok = true;
            for (int i = 0; i < numVideos; i++)
            {
                if (!frameBuffers[i].pull(frames[i]))
                {
                    ptlprintf("Error in %s [%8x], pull frame failed, buffer index %d\n", __FUNCTION__, id, i);
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
            if (forCuda)
                syncedFramesBufferForProcCuda.push(frames);
            else
                syncedFramesBufferForProcIOcl.push(frames);

            pullCount++;
            int needSync = 0;
            if (pullCount == roundedVideoFrameRate * syncInterval)
            {
                ptlprintf("Checking frames synchronization status, ");
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
                    ptlprintf("frames badly synchronized, resync\n");
                    break;
                }
                else
                {
                    ptlprintf("frames well synchronized, continue\n");
                    pullCount = 0;
                }
            }
        }
    }

    //syncedFramesBufferForShow.stop();
    if (forCuda)
        syncedFramesBufferForProcCuda.stop();
    else
        syncedFramesBufferForProcIOcl.stop();

END:
    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

FFmpegAudioVideoSource::FFmpegAudioVideoSource(ForShowFrameVectorQueue* ptrSyncedFramesBufferForShow,
    void* ptrSyncedFramesBufferForProc, int forCuda,
    ForceWaitFrameQueue* ptrProcFrameBufferForSend, ForceWaitFrameQueue* ptrProcFrameBufferForSave,
    int* ptrFinish, LogCallbackFunction logCallbackFunc, void* logCallbackData, 
    FrameRateCallbackFunction videoFrameRateCallbackFunc, void* videoFrameRateCallbackData)
{
    init();
    setProp(ptrSyncedFramesBufferForShow, ptrSyncedFramesBufferForProc, forCuda,
        ptrProcFrameBufferForSend, ptrProcFrameBufferForSave,
        ptrFinish, logCallbackFunc, logCallbackData,
        videoFrameRateCallbackFunc, videoFrameRateCallbackData);
    areSourceFiles = 0;
}

FFmpegAudioVideoSource::~FFmpegAudioVideoSource()
{
    close();
}

bool FFmpegAudioVideoSource::open(const std::vector<avp::Device>& devices, int width, int height, int frameRate,
    bool openAudio, const avp::Device& device, int sampleRate)
{
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
    videoReaders.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        // IMPORTANT NOTICE!!!!!!
        // FORCE PIXEL_TYPE_BGR_32
        // SUPPORT GPU ONLY
        opts.resize(2);
        opts.push_back(std::make_pair("video_device_number", videoDevices[i].numString));
        ok = videoReaders[i].open("video=" + videoDevices[i].shortName,
            false, avp::SampleTypeUnknown, true, pixelType, "dshow", opts);
        if (!ok)
        {
            ptlprintf("Could not open DirectShow video device %s[%s] with framerate = %s and video_size = %s\n",
                videoDevices[i].shortName.c_str(), videoDevices[i].numString.c_str(),
                frameRateStr.c_str(), videoSizeStr.c_str());
            failExists = true;
        }
    }

    videoOpenSuccess = !failExists;
    if (failExists)
    {
        for (int i = 0; i < numVideos; i++)
            videoReaders[i].close();

        if (logCallbackFunc)
            logCallbackFunc("Video sources open failed", logCallbackData);
        return false;
    }

    if (logCallbackFunc)
        logCallbackFunc("Video sources open success", logCallbackData);

    ptrFrameBuffers.reset(new std::vector<ForceWaitFrameQueue>(numVideos));
    for (int i = 0; i < numVideos; i++)
        (*ptrFrameBuffers)[i].setMaxSize(36);
    ptrSyncedFramesBufferForShow->clear();
    if (forCuda)
        ((BoundedPinnedMemoryFrameQueue*)ptrSyncedFramesBufferForProc)->clear();
    else
        ((ForceWaitFrameVectorQueue*)ptrSyncedFramesBufferForProc)->clear();

    videoEndFlag = 0;
    videoThreadsJoined = 0;
    videoSourceThreads.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        videoSourceThreads[i].reset(new std::thread(&FFmpegAudioVideoSource::videoSource, this, i));
    }
    videoSinkThread.reset(new std::thread(&FFmpegAudioVideoSource::videoSink, this));

    if (logCallbackFunc)
        logCallbackFunc("Video sources related threads create success\n", logCallbackData);

    if (openAudio)
    {
        audioDevice = device;
        audioSampleRate = sampleRate;

        std::vector<avp::Option> audioOpts;
        std::string sampleRateStr = std::to_string(audioSampleRate);
        audioOpts.push_back(std::make_pair("ar", sampleRateStr));
        audioOpts.push_back(std::make_pair("audio_device_number", audioDevice.numString));
        audioOpenSuccess = audioReader.open("audio=" + audioDevice.shortName, true, avp::SampleTypeUnknown,
            false, avp::PixelTypeUnknown, "dshow", audioOpts);
        if (!audioOpenSuccess)
        {
            ptlprintf("Could not open DirectShow audio device %s[%s], skip\n",
                audioDevice.shortName.c_str(), audioDevice.numString.c_str());

            if (logCallbackFunc)
                logCallbackFunc("Audio source open failed", logCallbackData);
        }
        else
        {
            if (logCallbackFunc)
                logCallbackFunc("Audio source open success", logCallbackData);

            ptrProcFrameBufferForSave->clear();
            ptrProcFrameBufferForSend->clear();

            audioEndFlag = 0;
            audioThreadJoined = 0;
            audioThread.reset(new std::thread(&FFmpegAudioVideoSource::audioSource, this));

            if (logCallbackFunc)
                logCallbackFunc("Audio source thread create success", logCallbackData);
        }        
    }

    areSourceFiles = 0;
    finish = 0;
    running = 1;

    return videoOpenSuccess == 1;
}

bool FFmpegAudioVideoSource::open(const std::vector<std::string>& urls, bool openAudio, const std::string& url)
{
    if (urls.empty())
        return false;

    if (!areAllURLs(urls) && !areAllNotURLs(urls))
    {
        ptlprintf("All input string in urls should be all URLs, or disk files\n");
        return false;
    }

    areSourceFiles = areAllNotURLs(urls);

    bool ok;
    bool failExists = false;
    numVideos = urls.size();
    videoReaders.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        ok = videoReaders[i].open(urls[i], false, avp::SampleTypeUnknown, true, avp::PixelTypeBGR32);
        if (!ok)
        {
            ptlprintf("Could not open video stream %s\n", urls[i].c_str());
            failExists = true;
            break;
        }
    }

    videoOpenSuccess = !failExists;
    if (failExists)
    {
        for (int i = 0; i < numVideos; i++)
            videoReaders[i].close();

        if (logCallbackFunc)
            logCallbackFunc("Video sources open failed", logCallbackData);
        return false;
    }

    videoFrameSize.width = videoReaders[0].getVideoWidth();
    videoFrameSize.height = videoReaders[0].getVideoHeight();
    videoFrameRate = videoReaders[0].getVideoFrameRate();
    roundedVideoFrameRate = videoFrameRate + 0.5;
    for (int i = 1; i < numVideos; i++)
    {
        if (videoReaders[i].getVideoWidth() != videoFrameSize.width)
        {
            failExists = true;
            ptlprintf("Error, video streams width not match\n");
            break;
        }
        if (videoReaders[i].getVideoHeight() != videoFrameSize.height)
        {
            failExists = true;
            ptlprintf("Error, video streams height not match\n");
            break;
        }
        if (fabs(videoReaders[i].getVideoFrameRate() - videoFrameRate) > 0.001)
        {
            failExists = true;
            ptlprintf("Error, video streams frame rate not match\n");
            break;
        }
    }

    videoOpenSuccess = !failExists;
    if (failExists)
    {
        if (logCallbackFunc)
            logCallbackFunc("Video sources properties not match", logCallbackData);
        return false;
    }

    if (logCallbackFunc)
        logCallbackFunc("Video sources open success", logCallbackData);

    ptrFrameBuffers.reset(new std::vector<ForceWaitFrameQueue>(numVideos));
    for (int i = 0; i < numVideos; i++)
        (*ptrFrameBuffers)[i].setMaxSize(36);
    ptrSyncedFramesBufferForShow->clear();
    if (forCuda)
        ((BoundedPinnedMemoryFrameQueue*)ptrSyncedFramesBufferForProc)->clear();
    else
        ((ForceWaitFrameVectorQueue*)ptrSyncedFramesBufferForProc)->clear();

    videoEndFlag = 0;
    videoThreadsJoined = 0;
    videoSourceThreads.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        videoSourceThreads[i].reset(new std::thread(&FFmpegAudioVideoSource::videoSource, this, i));
    }
    videoSinkThread.reset(new std::thread(&FFmpegAudioVideoSource::videoSink, this));

    if (logCallbackFunc)
        logCallbackFunc("Video sources related threads create success\n", logCallbackData);

    if (openAudio)
    {
        if (areSourceFiles && isURL(url) || (!areSourceFiles && !isURL(url)))
        {
            ptlprintf("Audio input type does not match with video input type, should be all URLs or all files\n");
            return false;
        }
        audioOpenSuccess = audioReader.open(url, true, avp::SampleTypeUnknown, false, avp::PixelTypeUnknown);
        if (!audioOpenSuccess)
        {
            ptlprintf("Could not open DirectShow audio device %s[%s], skip\n",
                audioDevice.shortName.c_str(), audioDevice.numString.c_str());

            if (logCallbackFunc)
                logCallbackFunc("Audio source open failed", logCallbackData);

            return false;
        }

        audioSampleRate = audioReader.getAudioSampleRate();

        if (logCallbackFunc)
            logCallbackFunc("Audio source open success", logCallbackData);

        ptrProcFrameBufferForSave->clear();
        ptrProcFrameBufferForSend->clear();

        audioEndFlag = 0;
        audioThreadJoined = 0;
        audioThread.reset(new std::thread(&FFmpegAudioVideoSource::audioSource, this));

        if (logCallbackFunc)
            logCallbackFunc("Audio source thread create success", logCallbackData);
    }

    finish = 0;
    running = 1;

    return videoOpenSuccess == 1;
}

void FFmpegAudioVideoSource::close()
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

    finish = 1;
    running = 0;
}

int FFmpegAudioVideoSource::getNumVideos() const
{
    return videoOpenSuccess ? numVideos : 0;
}

int FFmpegAudioVideoSource::getVideoFrameWidth() const
{
    return videoOpenSuccess ? videoFrameSize.width : 0;
}

int FFmpegAudioVideoSource::getVideoFrameHeight() const
{
    return videoOpenSuccess ? videoFrameSize.height : 0;
}

double FFmpegAudioVideoSource::getVideoFrameRate() const
{
    return videoOpenSuccess ? videoFrameRate : 0;
}

int FFmpegAudioVideoSource::getAudioSampleRate() const
{
    return audioOpenSuccess ? audioSampleRate : 0;
}

int FFmpegAudioVideoSource::getAudioSampleType() const
{
    return audioOpenSuccess ? audioReader.getAudioSampleType() : 0;
}

int FFmpegAudioVideoSource::getAudioNumChannels() const
{
    return audioOpenSuccess ? audioReader.getAudioNumChannels() : 0;
}

int FFmpegAudioVideoSource::getAudioChannelLayout() const
{
    return audioOpenSuccess ? audioReader.getAudioChannelLayout() : 0;
}

inline void stopCompleteFrameBuffers(std::vector<ForceWaitFrameQueue>* ptrFrameBuffers)
{
    std::vector<ForceWaitFrameQueue>& frameBuffers = *ptrFrameBuffers;
    int numBuffers = frameBuffers.size();
    for (int i = 0; i < numBuffers; i++)
        frameBuffers[i].stop();
}

void FFmpegAudioVideoSource::videoSource(int index)
{
    size_t id = std::this_thread::get_id().hash();
    ptlprintf("Thread %s [%8x] started, index = %d\n", __FUNCTION__, id, index);

    std::vector<ForceWaitFrameQueue>& frameBuffers = *ptrFrameBuffers;
    ForceWaitFrameQueue& buffer = frameBuffers[index];
    avp::AudioVideoReader3& reader = videoReaders[index];

    int waitTime = 0;
    if (areSourceFiles)
    {
        waitTime = 1000 / videoFrameRate - 5;
        if (waitTime <= 0)
            waitTime = 1;
    }

    long long int count = 0, beginCheckCount = roundedVideoFrameRate * 5;
    ztool::Timer timer;
    avp::AudioVideoFrame2 frame;
    bool ok;
    while (true)
    {
        ok = reader.read(frame);
        if (areSourceFiles)
            std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
        if (!ok)
        {
            ptlprintf("Error in %s [%8x], cannot read video frame\n", __FUNCTION__, id);
            stopCompleteFrameBuffers(ptrFrameBuffers.get());
            finish = 1;
            *ptrFinish = 1;
            break;
        }

        count++;
        if (count == beginCheckCount)
            timer.start();
        if ((count > beginCheckCount) && (count % roundedVideoFrameRate == 0))
        {
            timer.end();
            double actualFps = (count - beginCheckCount) / timer.elapse();
            //ptlprintf("[%8x] fps = %f\n", id, actualFps);
            if (index == 0 && videoFrameRateCallbackFunc)
                videoFrameRateCallbackFunc(actualFps, videoFrameRateCallbackData);
            if (abs(actualFps - videoFrameRate) > 2 && videoCheckFrameRate)
            {
                ptlprintf("Error in %s [%8x], actual fps = %f, far away from the set one\n", __FUNCTION__, id, actualFps);
                //buffer.stop();
                //stopCompleteFrameBuffers(ptrFrameBuffers.get());
                //finish = 1;
                //break;
            }
        }

        // NOTICE, for simplicity, I do not check whether the frame has the right property.
        // For the sake of program robustness, we should at least check whether the frame
        // is of type VIDEO, and is not empty, and has the correct pixel type and width and height.
        buffer.push(frame.clone());

        if (finish || videoEndFlag)
        {
            //buffer.stop();
            stopCompleteFrameBuffers(ptrFrameBuffers.get());
            finish = 1;
            *ptrFinish = 1;
            break;
        }
    }
    reader.close();

    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void FFmpegAudioVideoSource::audioSource()
{
    if (!audioOpenSuccess)
        return;

    size_t id = std::this_thread::get_id().hash();
    ptlprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    ztool::Timer timer;
    avp::AudioVideoFrame2 frame;
    bool ok;
    while (true)
    {
        if (finish || audioEndFlag)
            break;

        ok = audioReader.read(frame);
        if (!ok)
        {
            ptlprintf("Error in %s [%8x], cannot read audio frame\n", __FUNCTION__, id);
            finish = 1;
            *ptrFinish = 1;
            break;
        }

        avp::AudioVideoFrame2 deep = frame.clone();
        ptrProcFrameBufferForSend->push(deep);
        ptrProcFrameBufferForSave->push(deep);
    }

    ptrProcFrameBufferForSend->stop();
    ptrProcFrameBufferForSave->stop();

    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

// Package Head struct
typedef struct _PACKAGE_HEAD_ 
{
    char MsgFlag[8];
    unsigned int nFrameId;
    int  nFrameLen;
    int  nFrameType;
    char szReserv[20];
} PACKAGEHEAD;

SOCKET ConnectTCP(const char *pszIP, int nPort)
{
    //----------------------
    // Initialize Winsock
    WSADATA wsaData;
    int iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != NO_ERROR) {
        ptlprintf("WSAStartup function failed with error: %d\n", iResult);
        return -1;
    }
    //----------------------
    // Create a SOCKET for connecting to server
    SOCKET ConnectSocket;
    ConnectSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (ConnectSocket == INVALID_SOCKET) {
        ptlprintf("socket function failed with error: %ld\n", WSAGetLastError());
        WSACleanup();
        return -1;
    }
    //----------------------
    // The sockaddr_in structure specifies the address family,
    // IP address, and port of the server to be connected to.
    sockaddr_in clientService;
    clientService.sin_family = AF_INET;
    clientService.sin_addr.s_addr = inet_addr(pszIP);
    clientService.sin_port = htons(nPort);

    //----------------------
    // Connect to server.
    iResult = connect(ConnectSocket, (SOCKADDR *)& clientService, sizeof (clientService));
    if (iResult == SOCKET_ERROR) {
        ptlprintf("connect function failed with error: %ld, ip is %s\n", WSAGetLastError(), pszIP);
        iResult = closesocket(ConnectSocket);
        if (iResult == SOCKET_ERROR)
            ptlprintf("closesocket function failed with error: %ld\n", WSAGetLastError());
        WSACleanup();
        return -1;
    }

    char szCmd[64] = { 0 };
    sprintf(szCmd, "GETFRAME RUNS 0");
    iResult = send(ConnectSocket, szCmd, strlen(szCmd), 0);

    //Sleep(1000);

    //sprintf(szCmd, "GETFRAME SYNC 0");
    //iResult = send(ConnectSocket, szCmd, strlen(szCmd), 0);

    return ConnectSocket;
}

int RecvbyLen(SOCKET recvSocket, char *pszBuf, int nRecvLen)
{
    int iResult = 0;
    int nAlreadyLen = 0;
    int nWillRecvLen = nRecvLen;

    while (nWillRecvLen > 0) {
        iResult = recv(recvSocket, pszBuf + nAlreadyLen, nWillRecvLen, 0);
        if (0 > iResult) {
            return SOCKET_ERROR;
        }

        nWillRecvLen -= iResult;
        nAlreadyLen += iResult;
    }

    return nAlreadyLen;
}

JuJingAudioVideoSource::JuJingAudioVideoSource(ForShowFrameVectorQueue* ptrSyncedFramesBufferForShow,
    void* ptrSyncedFramesBufferForProc, int forCuda,
    ForceWaitFrameQueue* ptrProcFrameBufferForSend, ForceWaitFrameQueue* ptrProcFrameBufferForSave,
    int* ptrFinish, LogCallbackFunction logCallbackFunc, void* logCallbackData,
    FrameRateCallbackFunction videoFrameRateCallbackFunc, void* videoFrameRateCallbackData)
{
    init();
    setProp(ptrSyncedFramesBufferForShow, ptrSyncedFramesBufferForProc, forCuda,
        ptrProcFrameBufferForSend, ptrProcFrameBufferForSave,
        ptrFinish, logCallbackFunc, logCallbackData,
        videoFrameRateCallbackFunc, videoFrameRateCallbackData);
}

JuJingAudioVideoSource::~JuJingAudioVideoSource()
{
    close();
}

bool JuJingAudioVideoSource::open(const std::vector<std::string>& urls)
{
    if (urls.empty())
        return false;

    bool ok;
    bool failExists = false;
    numVideos = urls.size();
    sockets.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        sockets[i] = ConnectTCP(urls[i].c_str(), 8889);
        if (sockets[i] == SOCKET_ERROR)
        {
            failExists = true;
            break;
        }
    }

    videoOpenSuccess = !failExists;
    if (failExists)
    {
        if (logCallbackFunc)
            logCallbackFunc("Video sources open failed", logCallbackData);
        return false;
    }

    char cmd[64] = { 0 };
    sprintf(cmd, "GETFRAME SYNC 0");
    int cmdLen = strlen(cmd);
    //sprintf(szCmd, "GETFRAME STOP 0");
    int ret;
    for (int i = 0; i < numVideos; i++)
    {
        ret = send(sockets[i], cmd, cmdLen, 0);
        if (ret == SOCKET_ERROR)
        {
            failExists = true;
            break;
        }
    }        

    videoOpenSuccess = !failExists;
    if (failExists)
    {
        if (logCallbackFunc)
            logCallbackFunc("Video sources sync failed", logCallbackData);
        return false;
    }

    videoFrameSize.width = 1920;
    videoFrameSize.height = 1080;
    videoFrameRate = 25;
    roundedVideoFrameRate = videoFrameRate + 0.5;

    if (logCallbackFunc)
        logCallbackFunc("Video sources open success", logCallbackData);

    ptrDataPacketQueues.reset(new std::vector<RealTimeDataPacketQueue>(numVideos));
    ptrFrameBuffers.reset(new std::vector<ForceWaitFrameQueue>(numVideos));
    for (int i = 0; i < numVideos; i++)
        (*ptrFrameBuffers)[i].setMaxSize(36);
    ptrSyncedFramesBufferForShow->clear();
    if (forCuda)
        ((BoundedPinnedMemoryFrameQueue*)ptrSyncedFramesBufferForProc)->clear();
    else
        ((ForceWaitFrameVectorQueue*)ptrSyncedFramesBufferForProc)->clear();

    videoEndFlag = 0;
    videoThreadsJoined = 0;
    videoReceiveThreads.resize(numVideos);
    videoDecodeThreads.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        videoReceiveThreads[i].reset(new std::thread(&JuJingAudioVideoSource::videoRecieve, this, i));
        videoDecodeThreads[i].reset(new std::thread(&JuJingAudioVideoSource::videoDecode, this, i));
    }
    videoSinkThread.reset(new std::thread(&JuJingAudioVideoSource::videoSink, this));

    if (logCallbackFunc)
        logCallbackFunc("Video sources related threads create success\n", logCallbackData);

    audioOpenSuccess = 0;

    finish = 0;
    running = 1;

    return true;
}

void JuJingAudioVideoSource::close()
{
    if (videoOpenSuccess && !videoThreadsJoined)
    {
        videoEndFlag = 1;
        for (int i = 0; i < numVideos; i++)
        {
            videoReceiveThreads[i]->join();
            videoDecodeThreads[i]->join();
        }            
        videoSinkThread->join();
        videoReceiveThreads.clear();
        videoDecodeThreads.clear();
        videoSinkThread.reset(0);
        videoOpenSuccess = 0;
        videoThreadsJoined = 1;

        if (logCallbackFunc)
        {
            logCallbackFunc("Video sources related threads close success", logCallbackData);
            logCallbackFunc("Video sources close success", logCallbackData);
        }
    }

    finish = 1;
    running = 0;
}

int JuJingAudioVideoSource::getNumVideos() const
{
    return videoOpenSuccess ? numVideos : 0;
}

int JuJingAudioVideoSource::getVideoFrameWidth() const
{
    return videoOpenSuccess ? videoFrameSize.width : 0;
}

int JuJingAudioVideoSource::getVideoFrameHeight() const
{
    return videoOpenSuccess ? videoFrameSize.height : 0;
}

double JuJingAudioVideoSource::getVideoFrameRate() const
{
    return videoOpenSuccess ? videoFrameRate : 0;
}

int JuJingAudioVideoSource::getAudioSampleRate() const
{
    return 0;
}

int JuJingAudioVideoSource::getAudioSampleType() const
{
    return 0;
}

int JuJingAudioVideoSource::getAudioNumChannels() const
{
    return 0;
}

int JuJingAudioVideoSource::getAudioChannelLayout() const
{
    return 0;
}

void JuJingAudioVideoSource::videoRecieve(int index)
{
    size_t id = std::this_thread::get_id().hash();
    ptlprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    PACKAGEHEAD sHead;
    SOCKET connectSocket = SOCKET_ERROR;
    char *pszRecvBuf = NULL;
    fd_set fdread;
    timeval timeout;
    int nBufLen = 1024 * 1024;
    int nRecvLen = 0;
    int nRet = 0;
    int receiveZeroPts = 0;
    int receiveIntraFrame = 0;
    int errorOccurred = 0;

    pszRecvBuf = (char *)malloc(nBufLen);
    connectSocket = sockets[index];
    RealTimeDataPacketQueue& dataPacketQueue = (*ptrDataPacketQueues)[index];

    while (!videoEndFlag && !finish) 
    {
        FD_ZERO(&fdread);
        FD_SET(connectSocket, &fdread);

        timeout.tv_sec = 2;
        timeout.tv_usec = 0;
        nRet = select(0, &fdread, NULL, NULL, &timeout);
        if (0 == nRet) 
        {
            //ptlprintf("Timeout.\n");
            continue;
        }
        else if (SOCKET_ERROR == nRet) 
        {
            ptlprintf("Error in %s [%8x], select function failed with error: %u", __FUNCTION__, id, WSAGetLastError());
            errorOccurred = 1;
            break;
        }

        if (FD_ISSET(connectSocket, &fdread))
        {
            // 接收数据头
            memset(&sHead, 0, sizeof(sHead));
            nRecvLen = recv(connectSocket, (char *)&sHead, sizeof(sHead), 0);
            if (nRecvLen > 0) 
            {
                //ptlprintf("Bytes received: %d\n", nRecvLen);
            }
            else if (nRecvLen == 0) 
            {
                ptlprintf("Error in %s [%8x], connection closed\n", __FUNCTION__, id);
                errorOccurred = 1;
                break;
            }
            else 
            {
                ptlprintf("Error in %s [%8x], recv failed: %d\n", __FUNCTION__, id, WSAGetLastError());
                errorOccurred = 1;
                break;
            }

            if (0 != strcmp(sHead.MsgFlag, "IPCAMVR")) 
            {
                ptlprintf("Warning in %s [%8x], invalid data, continue\n", __FUNCTION__, id);
                continue;
                //break;
            }

            // 接收数据体
            if (sHead.nFrameLen > nBufLen) 
            {
                pszRecvBuf = (char *)realloc(pszRecvBuf, sHead.nFrameLen);
                nBufLen = sHead.nFrameLen;
            }

            nRecvLen = RecvbyLen(connectSocket, pszRecvBuf, sHead.nFrameLen);
            if (SOCKET_ERROR == nRecvLen) 
            {
                ptlprintf("socket failed\n");
                //closesocket(connectSocket);
                //continue;
                break;
            }
            else if (nRecvLen != sHead.nFrameLen) 
            {
                ptlprintf("invalid body.\n");
                //continue;
                break;
            }

            if (!receiveZeroPts)
            {
                if (sHead.nFrameId == 0)
                    receiveZeroPts = 1;
            }

            //ptlprintf("idx = %d\n", sHead.nFrameId);
            if (receiveZeroPts && !receiveIntraFrame)
            {
                if (sHead.nFrameType == 1)
                    receiveIntraFrame = 1;
            }

            if (receiveIntraFrame)
                dataPacketQueue.push(DataPacket((unsigned char*)pszRecvBuf, sHead.nFrameLen, sHead.nFrameType, sHead.nFrameId * 1000LL));
        }
    }

    if (errorOccurred)
    {
        finish = 1;
        *ptrFinish = 1;
    }

    char cmd[64] = { 0 };
    sprintf(cmd, "GETFRAME STOP 0");
    int cmdLen = strlen(cmd);
    int ret = send(connectSocket, cmd, cmdLen, 0);

    ret = closesocket(connectSocket);

    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void JuJingAudioVideoSource::videoDecode(int index)
{
    size_t id = std::this_thread::get_id().hash();
    ptlprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    RealTimeDataPacketQueue& dataPacketQueue = (*ptrDataPacketQueues)[index];
    ForceWaitFrameQueue& frameQueue = (*ptrFrameBuffers)[index];
    avp::AudioVideoDecoder* decoder = avp::createVideoDecoder("h264", pixelType);
    DataPacket pkt;
    avp::AudioVideoFrame2 frame;
    bool ok;
    while (!videoEndFlag && !finish)
    {
        dataPacketQueue.pull(pkt);
        if (pkt.data.get())
        {
            ok = decoder->decode(pkt.data.get(), pkt.dataSize, pkt.pts, frame);
            if (ok && frame.data[0])
            {
                frame.timeStamp = pkt.pts;
                frameQueue.push(frame.clone());
            }
        }
    }
    delete decoder;

    *ptrFinish = 1;
    stopCompleteFrameBuffers(ptrFrameBuffers.get());

    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}