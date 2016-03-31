#pragma once

#include "AudioVideoProcessor.h"
#include "opencv2/core.hpp"
#include <memory>
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
    struct Impl;
    std::unique_ptr<Impl> ptrImpl;
};

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
    struct Impl;
    std::unique_ptr<Impl> ptrImpl;
};

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
    struct Impl;
    std::unique_ptr<Impl> ptrImpl;
};
