#include "PanoramaTask.h"
#include "ConcurrentQueue.h"
#include "PinnedMemoryFrameQueue.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdio.h>
#include <memory>
#include <thread>

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

class AudioVideoSource
{
public:
    AudioVideoSource();
    virtual ~AudioVideoSource();
    
    bool isVideoOpened() const;
    bool isAudioOpened() const;
    bool isRunning() const;

    virtual int getNumVideos() const = 0;
    virtual int getVideoFrameWidth() const = 0;
    virtual int  getVideoFrameHeight() const = 0;
    virtual double getVideoFrameRate() const = 0;
    virtual int getAudioSampleRate() const = 0;
    virtual int getAudioSampleType() const = 0;
    virtual int getAudioNumChannels() const = 0;
    virtual int getAudioChannelLayout() const = 0;
    virtual void close() = 0;

protected:
    void setProp(ForShowFrameVectorQueue* ptrSyncedFramesBufferForShow,
        void* ptrSyncedFramesBufferForProc, int forCuda,
        ForceWaitFrameQueue* ptrProcFrameBufferForSend, ForceWaitFrameQueue* ptrProcFrameBufferForSave, 
        int* ptrFinish, LogCallbackFunction logCallbackFunc, void* logCallbackData,
        FrameRateCallbackFunction videoFrameRateCallbackFunc, void* videoFrameRateCallbackData);
    void init();
    void videoSink();

    cv::Size videoFrameSize;
    double videoFrameRate;
    int roundedVideoFrameRate;
    int numVideos;
    int videoOpenSuccess;
    int videoCheckFrameRate;
    int videoEndFlag;
    int videoThreadsJoined;
    int pixelType;

    int audioSampleRate;
    int audioOpenSuccess;
    int audioEndFlag;
    int audioThreadJoined;

    LogCallbackFunction logCallbackFunc;
    void* logCallbackData;
    FrameRateCallbackFunction videoFrameRateCallbackFunc;
    void* videoFrameRateCallbackData;
    
    std::unique_ptr<std::vector<ForceWaitFrameQueue> > ptrFrameBuffers;
    ForShowFrameVectorQueue* ptrSyncedFramesBufferForShow;
    void* ptrSyncedFramesBufferForProc;
    ForceWaitFrameQueue* ptrProcFrameBufferForSend; 
    ForceWaitFrameQueue* ptrProcFrameBufferForSave;
    int* ptrFinish;
    int finish;
    int running;
    int forCuda;
};

struct FFmpegAudioVideoSource : public AudioVideoSource
{
public:
    FFmpegAudioVideoSource(ForShowFrameVectorQueue* ptrSyncedFramesBufferForShow,
        void* ptrSyncedFramesBufferForProc, int forCuda,
        ForceWaitFrameQueue* ptrProcFrameBufferForSend, ForceWaitFrameQueue* ptrProcFrameBufferForSave,
        int* ptrFinish, LogCallbackFunction logCallbackFunc = 0, void* logCallbackData = 0,
        FrameRateCallbackFunction videoFrameRateCallbackFunc = 0, void* videoFrameRateCallbackData = 0);
    ~FFmpegAudioVideoSource();
    bool open(const std::vector<avp::Device>& devices, int width, int height, int frameRate,
        bool openAudio = false, const avp::Device& device = avp::Device(), int sampleRate = 0);
    bool open(const std::vector<std::string>& urls, bool openAudio = false, const std::string& url = "");
    void close();

    int getNumVideos() const;
    int getVideoFrameWidth() const;
    int  getVideoFrameHeight() const;
    double getVideoFrameRate() const;
    int getAudioSampleRate() const;
    int getAudioSampleType() const;
    int getAudioNumChannels() const;
    int getAudioChannelLayout() const;

private:
    std::vector<avp::AudioVideoReader> videoReaders;
    std::vector<avp::Device> videoDevices;
    std::vector<std::unique_ptr<std::thread> > videoSourceThreads;
    std::unique_ptr<std::thread> videoSinkThread;    
    void videoSource(int index);

    avp::AudioVideoReader audioReader;
    avp::Device audioDevice;    
    std::unique_ptr<std::thread> audioThread;    
    void audioSource();
};

struct DataPacket
{
    DataPacket()
    {
        dataSize = 0;
        nalUnitType = 0;
        pts = -1;
    }

    DataPacket(const unsigned char* data_, size_t dataSize_, int nalUnitType_, long long int pts_)
    {
        data.reset(new unsigned char[dataSize_]);
        memcpy(data.get(), data_, dataSize_);
        dataSize = dataSize_;
        nalUnitType = nalUnitType_;
        pts = pts_;
    };

    std::shared_ptr<unsigned char> data;
    size_t dataSize;
    int nalUnitType;
    long long int pts;
};

typedef ForceWaitRealTimeQueue<DataPacket> RealTimeDataPacketQueue;

struct JuJingAudioVideoSource : public AudioVideoSource
{
public:
    JuJingAudioVideoSource(ForShowFrameVectorQueue* ptrSyncedFramesBufferForShow,
        void* ptrSyncedFramesBufferForProc, int forCuda,
        ForceWaitFrameQueue* ptrProcFrameBufferForSend, ForceWaitFrameQueue* ptrProcFrameBufferForSave,
        int* ptrFinish, LogCallbackFunction logCallbackFunc = 0, void* logCallbackData = 0,
        FrameRateCallbackFunction videoFrameRateCallbackFunc = 0, void* videoFrameRateCallbackData = 0);
    ~JuJingAudioVideoSource();
    bool open(const std::vector<std::string>& urls);
    void close();

    int getNumVideos() const;
    int getVideoFrameWidth() const;
    int  getVideoFrameHeight() const;
    double getVideoFrameRate() const;
    int getAudioSampleRate() const;
    int getAudioSampleType() const;
    int getAudioNumChannels() const;
    int getAudioChannelLayout() const;

private:
    std::vector<SOCKET> sockets;
    std::vector<std::unique_ptr<std::thread> > videoReceiveThreads;
    std::vector<std::unique_ptr<std::thread> > videoDecodeThreads;
    std::unique_ptr<std::thread> videoSinkThread;
    std::unique_ptr<std::vector<RealTimeDataPacketQueue> > ptrDataPacketQueues;
    void videoRecieve(int index);
    void videoDecode(int index);
};