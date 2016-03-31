#pragma once

#include "AudioVideoProcessor.h"
#include "StampedFrameQueue.h"
#include "PinnedMemoryPool.h"
#include "StampedPinnedMemoryPool.h"
#include "SharedAudioVideoFramePool.h"
#include "RicohUtil.h"
#include "ZBlend.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class PanoramaPreviewTask
{
public:
    virtual ~PanoramaPreviewTask() {}
    virtual bool init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile, 
        int dstWidth, int dstHeight) = 0;
    virtual bool reset(const std::string& cameraParamFile) = 0;
    virtual bool seek(const std::vector<long long int>& timeStamps) = 0;
    virtual bool stitch(cv::Mat& result, std::vector<long long int>& timeStamps, int frameIncrement) = 0;
};

class CPUPanoramaPreviewTask : public PanoramaPreviewTask
{
public:
    CPUPanoramaPreviewTask();
    ~CPUPanoramaPreviewTask();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile,
        int dstWidth, int dstHeight);
    bool reset(const std::string& cameraParamFile);
    bool seek(const std::vector<long long int>& timeStamps);
    bool stitch(cv::Mat& result, std::vector<long long int>& timeStamps, int frameIncrement = 1);
private:
    struct Impl;
    std::unique_ptr<Impl> ptrImpl;
};

class CudaPanoramaPreviewTask : public PanoramaPreviewTask
{
public:
    CudaPanoramaPreviewTask();
    ~CudaPanoramaPreviewTask();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile,
        int dstWidth, int dstHeight);
    bool reset(const std::string& cameraParamFile);
    bool seek(const std::vector<long long int>& timeStamps);
    bool stitch(cv::Mat& result, std::vector<long long int>& timeStamps, int frameIncrement = 1);
private:
    struct Impl;
    std::unique_ptr<Impl> ptrImpl;
};

typedef void(*ProgressCallbackFunction)(double p, void* data);

class PanoramaLocalDiskTask
{
public:
    virtual ~PanoramaLocalDiskTask() {};
    virtual bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
        const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight,
        int dstVideoBitRate, ProgressCallbackFunction func, void* data) = 0;
    virtual bool start() = 0;
    virtual void waitForCompletion() = 0;
    virtual int getProgress() const = 0;
    virtual void cancel() = 0;
};

class CPUPanoramaLocalDiskTask : public PanoramaLocalDiskTask
{
public:
    CPUPanoramaLocalDiskTask();
    ~CPUPanoramaLocalDiskTask();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
        const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight,
        int dstVideoBitRate, ProgressCallbackFunction func, void* data);
    bool start();
    void waitForCompletion();
    int getProgress() const;
    void cancel();

private:
    void run();
    void clear();

    int numVideos;
    int audioIndex;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader> readers;
    std::vector<cv::Mat> dstSrcMaps, dstMasks;
    TilingMultibandBlendFastParallel blender;
    std::vector<cv::Mat> reprojImages;
    cv::Mat blendImage;
    avp::AudioVideoWriter2 writer;
    bool endFlag;

    std::atomic<int> finishPercent;

    int validFrameCount;

    ProgressCallbackFunction progressCallbackFunc;
    void* progressCallbackData;

    std::unique_ptr<std::thread> thread;

    bool initSuccess;
    bool finish;
};

struct StampedPinnedMemoryVector
{
    std::vector<cv::cuda::HostMem> frames;
    long long int timeStamp;
};

typedef BoundedCompleteQueue<avp::SharedAudioVideoFrame> FrameBuffer;
typedef BoundedCompleteQueue<StampedPinnedMemoryVector> FrameVectorBuffer;

class CudaPanoramaLocalDiskTask : public PanoramaLocalDiskTask
{
public:
    CudaPanoramaLocalDiskTask();
    ~CudaPanoramaLocalDiskTask();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
        const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight,
        int dstVideoBitRate, ProgressCallbackFunction func, void* data);
    bool start();
    void waitForCompletion();
    int getProgress() const;
    void cancel();

private:
    void clear();

    int numVideos;
    int audioIndex;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader> readers;
    CudaMultiCameraPanoramaRender2 render;
    PinnedMemoryPool srcFramesMemoryPool;
    SharedAudioVideoFramePool audioFramesMemoryPool, dstFramesMemoryPool;
    FrameVectorBuffer decodeFramesBuffer;
    FrameBuffer procFrameBuffer;
    cv::Mat blendImageCpu;
    avp::AudioVideoWriter2 writer;

    int decodeCount;
    int procCount;
    int encodeCount;
    std::atomic<int> finishPercent;

    int validFrameCount;

    ProgressCallbackFunction progressCallbackFunc;
    void* progressCallbackData;

    void decode();
    void proc();
    void encode();

    std::unique_ptr<std::thread> decodeThread;
    std::unique_ptr<std::thread> procThread;
    std::unique_ptr<std::thread> encodeThread;

    bool initSuccess;
    bool finish;
    bool isCanceled;
};

// for video source
typedef ForceWaitRealTimeQueue<avp::SharedAudioVideoFrame> CompleteFrameQueue;
// for synced video source
typedef ForceWaitRealTimeQueue<std::vector<avp::SharedAudioVideoFrame> > RealTimeFrameVectorQueue;
// for audio source and proc result
typedef ForceWaitRealTimeQueue<avp::SharedAudioVideoFrame> RealTimeFrameQueue;

typedef void (*LogCallbackFunction)(const std::string& line, void* data);
typedef void (*FrameRateCallbackFunction)(double fps, void* data);
typedef void (*ShowVideoSourceFramesCallbackFunction)(const std::vector<avp::SharedAudioVideoFrame>& frames, void* data);
typedef void (*ShowStichedFrameCallbackFunction)(const avp::SharedAudioVideoFrame& frame, void* data);

class PanoramaLiveStreamTask
{
public:
    PanoramaLiveStreamTask();
    ~PanoramaLiveStreamTask();

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

private:
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
