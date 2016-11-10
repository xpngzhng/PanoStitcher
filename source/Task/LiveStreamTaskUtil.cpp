#include "PanoramaTaskUtil.h"
#include "LiveStreamTaskUtil.h"
#include "Tool/Timer.h"
#include "Tool/Print.h"

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
    ForceWaitMixedFrameQueue* ptrProcFrameBufferForSend_, 
    ForceWaitMixedFrameQueue* ptrProcFrameBufferForSave_, int* ptrFinish_)
{
    pixelType = avp::PixelTypeBGR32;
    ptrFinish = ptrFinish_;
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

    syncInterval = 60;
}

//static int syncInterval = 60;

void AudioVideoSource::videoSink()
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    std::this_thread::sleep_for(std::chrono::microseconds(100));
    std::vector<ForceWaitFrameQueue>& frameBuffers = *ptrFrameBuffers;
    ForShowFrameVectorQueue& syncedFramesBufferForShow = *ptrSyncedFramesBufferForShow;
    BoundedPinnedMemoryFrameQueue& syncedFramesBufferForProcCuda = *(BoundedPinnedMemoryFrameQueue*)ptrSyncedFramesBufferForProc;
    ForShowFrameVectorQueue& syncedFramesBufferForProcIOcl = *(ForShowFrameVectorQueue*)ptrSyncedFramesBufferForProc;

    if (finish || videoEndFlag)
    {
        ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
        return;
    }

    //for (int i = 0; i < numVideos; i++)
    //    ztool::lprintf("size = %d\n", frameBuffers[i].size());

    if (finish || videoEndFlag)
    {
        ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
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
                ztool::lprintf("Error in %s [%8x], pull frame failed\n", __FUNCTION__, id);
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
            ztool::lprintf("Error in %s [%8x], failed to find the frame with smallest time stamp\n", __FUNCTION__, id);
            finish = 1;
        }

        if (finish || videoEndFlag)
            break;

        std::vector<avp::AudioVideoFrame2> syncedFrames(numVideos);
        avp::AudioVideoFrame2 slowestFrame;
        frameBuffers[currMaxIndex].pull(slowestFrame);
        syncedFrames[currMaxIndex] = slowestFrame;
        ztool::lprintf("Info in %s [%8x], slowest ts = %lld at source index %d\n", __FUNCTION__, id, slowestFrame.timeStamp, currMaxIndex);
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
                ztool::lprintf("Info in %s [%8x], this ts = %lld at source index %d\n", __FUNCTION__, id, sharedFrame.timeStamp, i);
                if (!ok)
                {
                    ztool::lprintf("Error in %s [%8x], pull frame failed\n", __FUNCTION__, id);
                    finish = 1;
                    break;
                }
                if (sharedFrame.timeStamp >= slowestFrame.timeStamp)
                {
                    syncedFrames[i] = sharedFrame;
                    ztool::lprintf("Info in %s [%8x], break\n", __FUNCTION__, id);
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
                    ztool::lprintf("Error in %s [%8x], pull frame failed, buffer index %d\n", __FUNCTION__, id, i);
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
                ztool::lprintf("Info in %s [%8x], checking frames synchronization status, ", __FUNCTION__, id);
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
                    ztool::lprintf("frames badly synchronized, resync\n");
                    break;
                }
                else
                {
                    ztool::lprintf("frames well synchronized, continue\n");
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
    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

FFmpegAudioVideoSource::FFmpegAudioVideoSource(ForShowFrameVectorQueue* ptrSyncedFramesBufferForShow,
    void* ptrSyncedFramesBufferForProc, int forCuda,
    ForceWaitMixedFrameQueue* ptrProcFrameBufferForSend, 
    ForceWaitMixedFrameQueue* ptrProcFrameBufferForSave, int* ptrFinish)
{
    init();
    setProp(ptrSyncedFramesBufferForShow, ptrSyncedFramesBufferForProc, forCuda,
        ptrProcFrameBufferForSend, ptrProcFrameBufferForSave, ptrFinish);
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
            ztool::lprintf("Error in %s, could not open DirectShow video device %s[%s] with framerate = %s and video_size = %s\n",
                __FUNCTION__, videoDevices[i].shortName.c_str(), videoDevices[i].numString.c_str(),
                frameRateStr.c_str(), videoSizeStr.c_str());
            failExists = true;
        }
    }

    videoOpenSuccess = !failExists;
    if (failExists)
    {
        for (int i = 0; i < numVideos; i++)
            videoReaders[i].close();
        return false;
    }

    ptrFrameBuffers.reset(new std::vector<ForceWaitFrameQueue>(numVideos));
    for (int i = 0; i < numVideos; i++)
        (*ptrFrameBuffers)[i].setMaxSize(36);

    ptrVideoFramePools.reset(new std::vector<AudioVideoFramePool>(numVideos));
    for (int i = 0; i < numVideos; i++)
        (*ptrVideoFramePools)[i].initAsVideoFramePool(pixelType, width, height);

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
            ztool::lprintf("Error in %s, could not open DirectShow audio device %s[%s], skip\n",
                __FUNCTION__, audioDevice.shortName.c_str(), audioDevice.numString.c_str());
        }
        else
        {
            ptrProcFrameBufferForSave->clear();
            ptrProcFrameBufferForSend->clear();

            audioEndFlag = 0;
            audioThreadJoined = 0;
            audioThread.reset(new std::thread(&FFmpegAudioVideoSource::audioSource, this));
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
        ztool::lprintf("Error in %s, all input string in urls should be all URLs, or disk files\n", __FUNCTION__);
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
            ztool::lprintf("Error in %s, could not open video stream %s\n", __FUNCTION__, urls[i].c_str());
            failExists = true;
            break;
        }
    }

    videoOpenSuccess = !failExists;
    if (failExists)
    {
        for (int i = 0; i < numVideos; i++)
            videoReaders[i].close();
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
            ztool::lprintf("Error in %s, video streams width not match\n", __FUNCTION__);
            break;
        }
        if (videoReaders[i].getVideoHeight() != videoFrameSize.height)
        {
            failExists = true;
            ztool::lprintf("Error in %s, video streams height not match\n", __FUNCTION__);
            break;
        }
        if (fabs(videoReaders[i].getVideoFrameRate() - videoFrameRate) > 0.001)
        {
            failExists = true;
            ztool::lprintf("Error in %s, video streams frame rate not match\n", __FUNCTION__);
            break;
        }
    }

    videoOpenSuccess = !failExists;
    if (failExists)
        return false;

    ptrFrameBuffers.reset(new std::vector<ForceWaitFrameQueue>(numVideos));
    for (int i = 0; i < numVideos; i++)
        (*ptrFrameBuffers)[i].setMaxSize(36);

    ptrVideoFramePools.reset(new std::vector<AudioVideoFramePool>(numVideos));
    for (int i = 0; i < numVideos; i++)
        (*ptrVideoFramePools)[i].initAsVideoFramePool(pixelType, videoFrameSize.width, videoFrameSize.height);

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

    if (openAudio)
    {
        if (areSourceFiles && isURL(url) || (!areSourceFiles && !isURL(url)))
        {
            ztool::lprintf("Error in %s, audio input type does not match with video input type, "
                "should be all URLs or all files\n", __FUNCTION__);
            return false;
        }
        audioOpenSuccess = audioReader.open(url, true, avp::SampleTypeUnknown, false, avp::PixelTypeUnknown);
        if (!audioOpenSuccess)
        {
            ztool::lprintf("Error in %s, could not open DirectShow audio device %s[%s], skip\n",
                __FUNCTION__, audioDevice.shortName.c_str(), audioDevice.numString.c_str());
            return false;
        }

        audioSampleRate = audioReader.getAudioSampleRate();

        ptrProcFrameBufferForSave->clear();
        ptrProcFrameBufferForSend->clear();

        audioEndFlag = 0;
        audioThreadJoined = 0;
        audioThread.reset(new std::thread(&FFmpegAudioVideoSource::audioSource, this));
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
    }

    if (audioOpenSuccess && !audioThreadJoined)
    {
        audioEndFlag = 1;
        audioThread->join();
        audioThread.reset(0);
        audioOpenSuccess = 0;
        audioThreadJoined = 1;
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
    ztool::lprintf("Thread %s [%8x] started, index = %d\n", __FUNCTION__, id, index);

    std::vector<ForceWaitFrameQueue>& frameBuffers = *ptrFrameBuffers;
    ForceWaitFrameQueue& buffer = frameBuffers[index];
    AudioVideoFramePool& pool = (*ptrVideoFramePools)[index];
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
    avp::AudioVideoFrame2 frame, dummyAudioFrame;
    int mediaType;
    bool ok;
    while (true)
    {
        pool.get(frame);
        ok = reader.readTo(dummyAudioFrame, frame, mediaType);
        if (areSourceFiles)
            std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
        if (!ok || mediaType != avp::VIDEO)
        {
            ztool::lprintf("Error in %s [%8x], cannot read video frame\n", __FUNCTION__, id);
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
            //ztool::lprintf("[%8x] fps = %f\n", id, actualFps);
            if (abs(actualFps - videoFrameRate) > 2 && videoCheckFrameRate)
            {
                ztool::lprintf("Error in %s [%8x], actual fps = %f, far away from the set one\n", __FUNCTION__, id, actualFps);
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
            *ptrFinish = 1;
            break;
        }
    }
    reader.close();

    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void FFmpegAudioVideoSource::audioSource()
{
    if (!audioOpenSuccess)
        return;

    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

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
            ztool::lprintf("Error in %s [%8x], cannot read audio frame\n", __FUNCTION__, id);
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

    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
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
        ztool::lprintf("Error in %s, WSAStartup function failed with error: %d\n", __FUNCTION__, iResult);
        return -1;
    }
    //----------------------
    // Create a SOCKET for connecting to server
    SOCKET ConnectSocket;
    ConnectSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (ConnectSocket == INVALID_SOCKET) {
        ztool::lprintf("Error in %s, socket function failed with error: %ld\n", __FUNCTION__, WSAGetLastError());
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
        ztool::lprintf("Error in %s, connect function failed with error: %ld, ip is %s\n", __FUNCTION__, WSAGetLastError(), pszIP);
        iResult = closesocket(ConnectSocket);
        if (iResult == SOCKET_ERROR)
            ztool::lprintf("Error in %s, closesocket function failed with error: %ld\n", __FUNCTION__, WSAGetLastError());
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
    ForceWaitMixedFrameQueue* ptrProcFrameBufferForSend, 
    ForceWaitMixedFrameQueue* ptrProcFrameBufferForSave, int* ptrFinish)
{
    init();
    setProp(ptrSyncedFramesBufferForShow, ptrSyncedFramesBufferForProc, forCuda,
        ptrProcFrameBufferForSend, ptrProcFrameBufferForSave, ptrFinish);
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
        return false;

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
        return false;

    //videoFrameSize.width = 1920;
    //videoFrameSize.height = 1080;
    //videoFrameRate = 25;
    videoFrameSize.width = 2048;
    videoFrameSize.height = 1536;
    videoFrameRate = 30;
    roundedVideoFrameRate = videoFrameRate + 0.5;

    ptrDataPacketQueues.reset(new std::vector<RealTimeDataPacketQueue>(numVideos));
    ptrFrameBuffers.reset(new std::vector<ForceWaitFrameQueue>(numVideos));
    for (int i = 0; i < numVideos; i++)
        (*ptrFrameBuffers)[i].setMaxSize(36);

    ptrVideoFramePools.reset(new std::vector<AudioVideoFramePool>(numVideos));
    for (int i = 0; i < numVideos; i++)
        (*ptrVideoFramePools)[i].initAsVideoFramePool(pixelType, videoFrameSize.width, videoFrameSize.height);

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
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

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
            //ztool::lprintf("Timeout.\n");
            continue;
        }
        else if (SOCKET_ERROR == nRet) 
        {
            ztool::lprintf("Error in %s [%8x], select function failed with error: %u", __FUNCTION__, id, WSAGetLastError());
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
                //ztool::lprintf("Bytes received: %d\n", nRecvLen);
            }
            else if (nRecvLen == 0) 
            {
                ztool::lprintf("Error in %s [%8x], connection closed\n", __FUNCTION__, id);
                errorOccurred = 1;
                break;
            }
            else 
            {
                ztool::lprintf("Error in %s [%8x], recv failed: %d\n", __FUNCTION__, id, WSAGetLastError());
                errorOccurred = 1;
                break;
            }

            if (0 != strcmp(sHead.MsgFlag, "IPCAMVR")) 
            {
                ztool::lprintf("Warning in %s [%8x], invalid data, continue\n", __FUNCTION__, id);
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
                ztool::lprintf("Error in %s, wsocket failed\n", __FUNCTION__);
                //closesocket(connectSocket);
                //continue;
                break;
            }
            else if (nRecvLen != sHead.nFrameLen) 
            {
                ztool::lprintf("Error in %s, invalid body\n", __FUNCTION__);
                //continue;
                break;
            }

            if (!receiveZeroPts)
            {
                if (sHead.nFrameId == 0)
                    receiveZeroPts = 1;
            }

            //ztool::lprintf("idx = %d\n", sHead.nFrameId);
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

    dataPacketQueue.stop();

    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void JuJingAudioVideoSource::videoDecode(int index)
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    RealTimeDataPacketQueue& dataPacketQueue = (*ptrDataPacketQueues)[index];
    ForceWaitFrameQueue& frameQueue = (*ptrFrameBuffers)[index];
    AudioVideoFramePool& pool = (*ptrVideoFramePools)[index];
    avp::AudioVideoDecoder* decoder = avp::createVideoDecoder("h264_qsv", pixelType);
    DataPacket pkt;
    avp::AudioVideoFrame2 frame, copyFrame;
    bool ok;
    ztool::Timer timer;
    int count = 0;
    while (!videoEndFlag && !finish)
    {
        //if (count == 20)
        //{
        //    count = 0;
        //    timer.end();
        //    printf("fps = %f\n", 20 / timer.elapse());
        //    timer.start();
        //}
        ok = dataPacketQueue.pull(pkt);
        if (!ok)
            break;
        if (pkt.data.get())
        {
            ok = decoder->decode(pkt.data.get(), pkt.dataSize, pkt.pts, frame);
            if (ok && frame.data[0])
            {
                frame.timeStamp = pkt.pts;
                pool.get(copyFrame);
                frame.copyTo(copyFrame);
                frameQueue.push(copyFrame);
            }
        }
        count++;
    }
    delete decoder;

    *ptrFinish = 1;
    stopCompleteFrameBuffers(ptrFrameBuffers.get());

    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

#include "NsdNetSDK.h"

#define ENABLE_NSD_SYNC_READ 0

static void printErrorMessage(int code)
{
    if (!code) return;
    char buf[4096];
    NSD_FormatMessage(code, buf, 4096);
    printf("%s\n", buf);
}

HuaTuAudioVideoSource::HuaTuAudioVideoSource(ForShowFrameVectorQueue* ptrSyncedFramesBufferForShow,
    void* ptrSyncedFramesBufferForProc, int forCuda,
    ForceWaitMixedFrameQueue* ptrProcFrameBufferForSend,
    ForceWaitMixedFrameQueue* ptrProcFrameBufferForSave, int* ptrFinish)
{
    init();
    setProp(ptrSyncedFramesBufferForShow, ptrSyncedFramesBufferForProc, forCuda,
        ptrProcFrameBufferForSend, ptrProcFrameBufferForSave, ptrFinish);
    syncInterval = 5;
    devHandle = 0;
    camHandle = 0;
    int ret = 0;
    ret = NSD_Init();
}

HuaTuAudioVideoSource::~HuaTuAudioVideoSource()
{
    close();
}

bool HuaTuAudioVideoSource::open(const std::string& url)
{
    if (url.empty())
        return false;

    NSD_INETADDR devAddr;
    memset(&devAddr, 0, sizeof(NSD_INETADDR));
    devAddr.byIPProtoVer = NSD_IPPROTO_V4;
    devAddr.wPORT = 60000;
    strcpy_s(devAddr.szHostIP, url.c_str());

    int ret;
    ret = NSD_Login(&devAddr, "admin", "admin", 1, &devHandle);
    printErrorMessage(ret);
    if (ret) return false;
    ret = NSD_RemoteCamera_Init(devHandle, &camHandle);
    printErrorMessage(ret);
    if (ret) return false;

#if ENABLE_NSD_SYNC_READ
    ret = NSD_RemoteCamera_SetTransferProtocol(camHandle, NSD_TRANSFER_PROTOCOL_TCP);
    printErrorMessage(ret);
    if (ret) return false;
    ret = NSD_RemoteCamera_SetTimeout(camHandle, 100000);
    printErrorMessage(ret);
    if (ret) return false;
    ret = NSD_RemoteCamera_SetChannelId(camHandle, 0);
    printErrorMessage(ret);
    if (ret) return false;
    ret = NSD_RemoteCamera_SetStreamId(camHandle, 1);
    printErrorMessage(ret);
    if (ret) return false;

    ret = NSD_RemoteCamera_Open(camHandle);
    printErrorMessage(ret);
    if (ret) return false;
#endif

    videoOpenSuccess = 1;

    videoFrameSize.width = 2592;
    videoFrameSize.height = 1944;
    videoFrameRate = 30;
    roundedVideoFrameRate = videoFrameRate + 0.5;

    numVideos = 4;

    ptrDataPacketQueues.reset(new std::vector<RealTimeDataPacketQueue>(numVideos));
    for (int i = 0; i < numVideos; i++)
        (*ptrDataPacketQueues)[i].setMaxSize(36);
    ptrFrameBuffers.reset(new std::vector<ForceWaitFrameQueue>(numVideos));
    for (int i = 0; i < numVideos; i++)
        (*ptrFrameBuffers)[i].setMaxSize(36);

    ptrVideoFramePools.reset(new std::vector<AudioVideoFramePool>(numVideos));
    for (int i = 0; i < numVideos; i++)
        (*ptrVideoFramePools)[i].initAsVideoFramePool(pixelType, videoFrameSize.width, videoFrameSize.height);

    ptrSyncedFramesBufferForShow->clear();
    if (forCuda)
        ((BoundedPinnedMemoryFrameQueue*)ptrSyncedFramesBufferForProc)->clear();
    else
        ((ForceWaitFrameVectorQueue*)ptrSyncedFramesBufferForProc)->clear();

    videoEndFlag = 0;
    videoThreadsJoined = 0;
    videoReceiveThread.reset(new std::thread(&HuaTuAudioVideoSource::videoRecieve, this));
    videoDecodeThreads.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
        videoDecodeThreads[i].reset(new std::thread(&HuaTuAudioVideoSource::videoDecode, this, i));
    videoSinkThread.reset(new std::thread(&HuaTuAudioVideoSource::videoSink, this));

    audioOpenSuccess = 0;

    finish = 0;
    running = 1;

    return true;
}

void HuaTuAudioVideoSource::close()
{
    if (videoOpenSuccess && !videoThreadsJoined)
    {
        videoEndFlag = 1;
        videoReceiveThread->join();
        for (int i = 0; i < numVideos; i++)
            videoDecodeThreads[i]->join();
        videoSinkThread->join();
        videoDecodeThreads.clear();
        videoSinkThread.reset(0);
        videoOpenSuccess = 0;
        videoThreadsJoined = 1;
    }

    if (camHandle || devHandle)
    {
        int ret = 0;
#if ENABLE_NSD_SYNC_READ
        ret = NSD_RemoteCamera_Close(camHandle);
        printErrorMessage(ret);
        ret = NSD_RemoteCamera_Uninit(camHandle);
        printErrorMessage(ret);
#endif
        ret = NSD_Logout(devHandle);
        printErrorMessage(ret);
        ret = NSD_Cleanup();
        printErrorMessage(ret);
        camHandle = 0;
        devHandle = 0;
    }

    finish = 1;
    running = 0;
}

int HuaTuAudioVideoSource::getNumVideos() const
{
    return videoOpenSuccess ? numVideos : 0;
}

int HuaTuAudioVideoSource::getVideoFrameWidth() const
{
    return videoOpenSuccess ? videoFrameSize.width : 0;
}

int HuaTuAudioVideoSource::getVideoFrameHeight() const
{
    return videoOpenSuccess ? videoFrameSize.height : 0;
}

double HuaTuAudioVideoSource::getVideoFrameRate() const
{
    return videoOpenSuccess ? videoFrameRate : 0;
}

int HuaTuAudioVideoSource::getAudioSampleRate() const
{
    return 0;
}

int HuaTuAudioVideoSource::getAudioSampleType() const
{
    return 0;
}

int HuaTuAudioVideoSource::getAudioNumChannels() const
{
    return 0;
}

int HuaTuAudioVideoSource::getAudioChannelLayout() const
{
    return 0;
}

#if ENABLE_NSD_SYNC_READ
void HuaTuAudioVideoSource::videoRecieve()
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    int seeIFrame[4] = { 0 };

    while (!videoEndFlag && !finish)
    {
        LPNSD_AVFRAME_DATA frameData = NULL;
        long frameCount = 0;
        int ret = NSD_RemoteCamera_ReadEx(camHandle, &frameData, &frameCount);
        if (ret != NSD_ERRNO_SOCKET_RECEIVE_TIMEOUT)
            printErrorMessage(ret);

        if (!ret)
        {
            for (int j = 0; j < frameCount; j++)
            {
                int index = frameData[j].byChannel - 1;
                // NOTICE!!!
                // We invoke Intel QSV to decode received video packets,
                // we have to make sure the very first packet fed to the decoder is I frame.
                // So we use seeIFrame array to ensure this.
                if (!seeIFrame[index])
                {
                    if (frameData[j].byFrameType != 1)
                        continue;

                    seeIFrame[index] = 1;
                }
                (*ptrDataPacketQueues)[index].push(DataPacket((unsigned char*)frameData[j].pszData,
                    frameData[j].lDataLength, -1, frameData[j].lTimeStamp));
            }
            //printf("read pts = %lld\n", frameData[0].lTimeStamp);

            for (int j = 0; j < frameCount; j++)
                printf("[%d] %lld %d ", frameData[j].byChannel, frameData[j].lTimeStamp, frameData[j].lDataLength);
            printf("\n");
        }

        //Sleep(5);
    }

    for (int i = 0; i < numVideos; i++)
        (*ptrDataPacketQueues)[i].stop();

    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

#else

#include <stdio.h>

struct CallbackData
{
    int index;
    int seeIFrame;
    RealTimeDataPacketQueue* dataPacketQueue;
    ztool::Timer timer;
    int count;
    int period;
    FILE* file;
};

void __stdcall receiveDataCallback(NSD_HANDLE lRealHandle, NSD_AVFRAME_DATA* frameData, void* pUserData)
{
    CallbackData* data = (CallbackData*)pUserData;
    if (data->count == data->period)
    {
        data->timer.end();
        printf("data receive [%d] fps = %f\n", data->index, data->period / data->timer.elapse());
        data->timer.start();
        data->count = 0;
    }
    data->count++;
    if (!data->seeIFrame)
    {
        if (frameData->byFrameType != 1)
            return;

        data->seeIFrame = 1;
    }
    fprintf(data->file, "%lld\n", frameData->lTimeStamp);
    data->dataPacketQueue->push(DataPacket((unsigned char*)frameData->pszData,
        frameData->lDataLength, -1, frameData->lTimeStamp));
}

void HuaTuAudioVideoSource::videoRecieve()
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    int ret = 0;
    std::vector<NSD_HANDLE> camHandles(numVideos, 0);
    std::vector<CallbackData> callbackData(numVideos);

    for (int i = 0; i < numVideos; i++)
    {
        NSD_CLIENTINFO client = { 0 };
        client.bEnableAutoReconnect = 1;
        client.byChannel = i + 1;
        client.byStreamID = 1;
        client.byTransferProtocol = NSD_TRANSFER_PROTOCOL_TCP;
        client.lReconnectCount = 10;
        client.bStretchMode = 1;

        CallbackData& data = callbackData[i];
        data.index = i;
        data.seeIFrame = 0;
        data.dataPacketQueue = (*ptrDataPacketQueues).data() + i;
        data.count = 0;
        data.period = roundedVideoFrameRate;
        char buf[256];
        sprintf(buf, "ts%d.txt", i);
        data.file = fopen(buf, "w");
        data.timer.start();

        ret = NSD_StartRealPlay(devHandle, &client, 0, &camHandles[i], 0);
        if (ret)
        {
            printErrorMessage(ret);
            break;
        }

        ret = NSD_SetStandardDataCallBack(camHandles[i], receiveDataCallback, &data);
        if (ret)
        {
            printErrorMessage(ret);
            break;
        }
    }

    while (!videoEndFlag && !finish)
        std::this_thread::sleep_for(std::chrono::seconds(1));

    for (int i = 0; i < numVideos; i++)
    {
        ret = NSD_StopRealPlay(camHandles[i]);
        printErrorMessage(ret);
    }

    for (int i = 0; i < numVideos; i++)
        fclose(callbackData[i].file);

    for (int i = 0; i < numVideos; i++)
        (*ptrDataPacketQueues)[i].stop();

    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}
#endif

void HuaTuAudioVideoSource::videoDecode(int index)
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    RealTimeDataPacketQueue& dataPacketQueue = (*ptrDataPacketQueues)[index];
    ForceWaitFrameQueue& frameQueue = (*ptrFrameBuffers)[index];
    AudioVideoFramePool& pool = (*ptrVideoFramePools)[index];
    avp::AudioVideoDecoder* decoder = avp::createVideoDecoder("h264_qsv", pixelType);
    DataPacket pkt;
    avp::AudioVideoFrame2 frame, copyFrame, lastFrame;
    bool ok;
    ztool::Timer timer;
    int count = 0;
    while (!videoEndFlag && !finish)
    {
        if (count == roundedVideoFrameRate)
        {
            count = 0;
            timer.end();
            printf("decoder [%d] fps = %f\n", index, roundedVideoFrameRate / timer.elapse());
            timer.start();
        }
        dataPacketQueue.pull(pkt);
        if (pkt.data.get())
        {
            pool.get(frame);
            bool gotFrame;
            //timer.start();
            ok = decoder->decodeTo(pkt.data.get(), pkt.dataSize, pkt.pts, frame, gotFrame);
            //timer.end();
            //if (index == 0)
            //    printf("decode time = %f\n", timer.elapse());
            if (ok && gotFrame)
            {
                frame.timeStamp = pkt.pts;
                frameQueue.push(frame);
                lastFrame = frame;
            }
            // Sometimes, decoder does not produce a frame, 
            // in order to avoid out of synchronization, we insert the last decoded frame into frameQueue.
            else if (lastFrame.data[0])
            {
                lastFrame.timeStamp = pkt.pts;
                frameQueue.push(lastFrame);
            }
            //timer.end();
            //if (index == 0)
            //    printf("decode total time = %f\n", timer.elapse());
        }
        count++;
    }
    delete decoder;

    *ptrFinish = 1;
    stopCompleteFrameBuffers(ptrFrameBuffers.get());

    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}