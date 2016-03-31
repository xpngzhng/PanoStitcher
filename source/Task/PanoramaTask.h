#pragma once

#include "AudioVideoProcessor.h"
#include "StampedFrameQueue.h"
#include "StampedPinnedMemoryPool.h"
#include "RicohUtil.h"
#include "ZBlend.h"
#include "opencv2/core/core.hpp"
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
    void clear();

    int numVideos;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader> readers;
    std::vector<cv::Mat> dstSrcMaps, dstMasks;
    TilingMultibandBlendFastParallel blender;
    std::vector<cv::Mat> reprojImages;
    cv::Mat blendImage;
    bool initSuccess;
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
    void clear();

    int numVideos;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader> readers;
    std::unique_ptr<PanoramaRender> ptrRender;
    cv::Mat blendImage;
    bool initSuccess;
};

typedef void(*ProgressCallbackFunction)(double p, void* data);

class PanoramaLocalDiskTask
{
public:
    virtual ~PanoramaLocalDiskTask() {};
    virtual bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
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
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
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

class CudaPanoramaLocalDiskTask : public PanoramaLocalDiskTask
{
public:
    CudaPanoramaLocalDiskTask();
    ~CudaPanoramaLocalDiskTask();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
        const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight,
        int dstVideoBitRate, ProgressCallbackFunction func, void* data);
    bool start();
    void waitForCompletion();
    int getProgress() const;
    void cancel();

private:
    //void run();
    void clear();

    int numVideos;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader> readers;
    std::vector<cv::gpu::GpuMat> xmapsGpu, ymapsGpu;
    CudaTilingMultibandBlendFast blender;
    std::vector<cv::gpu::Stream> streams;
    std::vector<cv::gpu::CudaMem> pinnedMems;
    std::vector<cv::gpu::GpuMat> imagesGpu, reprojImagesGpu;
    cv::gpu::GpuMat blendImageGpu;
    cv::Mat blendImageCpu;
    avp::AudioVideoWriter2 writer;

    int decodeCount;
    int procCount;
    int encodeCount;
    std::atomic<int> finishPercent;

    int validFrameCount;

    std::mutex mtxDecodedImages;
    std::condition_variable cvDecodedImagesForWrite, cvDecodedImagesForRead;
    bool decodedImagesOwnedByDecodeThread;
    bool videoEnd;

    std::mutex mtxEncodedImage;
    std::condition_variable cvEncodedImageForWrite, cvEncodedImageForRead;
    bool encodedImageOwnedByProcThread;
    bool procEnd;

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

class QWidget;

class QtCPUPanoramaLocalDiskTask
{
public:
    QtCPUPanoramaLocalDiskTask();
    ~QtCPUPanoramaLocalDiskTask();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
        const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight,
        int dstVideoBitRate);
    void run(QWidget* obj);
    void cancel();

private:
    void clear();

    int numVideos;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader> readers;
    std::vector<cv::Mat> dstSrcMaps, dstMasks;
    TilingMultibandBlendFast blender;
    std::vector<cv::Mat> reprojImages;
    cv::Mat blendImage;
    avp::AudioVideoWriter2 writer;
    bool endFlag;

    int validFrameCount;

    bool initSuccess;
    bool finish;
};

class QtCudaPanoramaLocalDiskTask
{
public:
    QtCudaPanoramaLocalDiskTask();
    ~QtCudaPanoramaLocalDiskTask();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
        const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight,
        int dstVideoBitRate);
    void run(QWidget* obj);
    void cancel();

private:
    void clear();

    int numVideos;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader> readers;
    std::vector<cv::gpu::GpuMat> xmapsGpu, ymapsGpu;
    CudaTilingMultibandBlendFast blender;
    std::vector<cv::gpu::Stream> streams;
    std::vector<cv::gpu::CudaMem> pinnedMems;
    std::vector<cv::gpu::GpuMat> imagesGpu, reprojImagesGpu;
    cv::gpu::GpuMat blendImageGpu;
    cv::Mat blendImageCpu;
    avp::AudioVideoWriter2 writer;

    int decodeCount;
    int procCount;
    int encodeCount;
    std::atomic<int> finishPercent;

    int validFrameCount;

    std::mutex mtxDecodedImages;
    std::condition_variable cvDecodedImagesForWrite, cvDecodedImagesForRead;
    bool decodedImagesOwnedByDecodeThread;
    bool videoEnd;

    std::mutex mtxEncodedImage;
    std::condition_variable cvEncodedImageForWrite, cvEncodedImageForRead;
    bool encodedImageOwnedByProcThread;
    bool procEnd;

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
typedef CompleteQueue<avp::SharedAudioVideoFrame> CompleteFrameQueue;
// for synced video source
typedef RealTimeQueue<std::vector<avp::SharedAudioVideoFrame> > RealTimeFrameVectorQueue;
// for audio source and proc result
typedef RealTimeQueue<avp::SharedAudioVideoFrame> RealTimeFrameQueue;

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

    bool beginVideoStitch(const std::string& configFileName, int width, int height, bool useCuda);
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
    RealTimeFrameQueue procFrameBufferForShow, procFrameBufferForSend, procFrameBufferForSave;
};
