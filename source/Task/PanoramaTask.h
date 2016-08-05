#pragma once

#include "AudioVideoProcessor.h"
#include "opencv2/core.hpp"
#include <memory>
#include <vector>

void setAddWatermark(bool addWatermark);

void setLanguage(bool isChinese);

void setCPUMultibandBlendMultiThread(bool multiThread);

typedef void(*PanoTaskLogCallbackFunc)(const char*, va_list);

PanoTaskLogCallbackFunc setPanoTaskLogCallback(PanoTaskLogCallbackFunc func);

class PanoramaPreviewTask
{
public:
    virtual ~PanoramaPreviewTask() {}
    virtual bool init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile, 
        int dstWidth, int dstHeight) = 0;
    virtual bool reset(const std::string& cameraParamFile) = 0;
    virtual bool seek(const std::vector<long long int>& timeStamps) = 0;
    virtual bool stitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst, int frameIncrement) = 0;
};

class CPUPanoramaPreviewTask : public PanoramaPreviewTask
{
public:
    CPUPanoramaPreviewTask();
    ~CPUPanoramaPreviewTask();

    bool init(const std::vector<std::string>& srcVideoFiles, const std::string& cameraParamFile,
        int dstWidth, int dstHeight);
    bool reset(const std::string& cameraParamFile);

    bool isValid() const;
    int getNumSourceVideos() const;
    double getVideoFrameRate() const;
    bool getMasks(std::vector<cv::Mat>& masks) const;
    bool getUniqueMasks(std::vector<cv::Mat>& masks) const;

    bool seek(const std::vector<int>& indexes);
    bool stitch(std::vector<cv::Mat>& src, std::vector<int>& indexes, cv::Mat& dst, int frameIncrement = 1);
    bool restitch(std::vector<cv::Mat>& src, std::vector<int>& indexes, cv::Mat& dst);    

    bool getCurrReprojectForAll(std::vector<cv::Mat>& images, std::vector<int>& indexes) const;
    bool reReprojectForAll(std::vector<cv::Mat>& images, std::vector<int>& indexes);
    bool readNextAndReprojectForAll(std::vector<cv::Mat>& images, std::vector<int>& indexes);
    bool readNextAndReprojectForOne(int videoIndex, cv::Mat& image, int& frameIndex);
    bool readPrevAndReprojectForOne(int videoIndex, cv::Mat& image, int& frameIndex);

    bool setCustomMaskForOne(int videoIndex, int begFrameIndexInc, int endFrameIndexExc, const cv::Mat& mask);
    void eraseCustomMaskForOne(int videoIndex, int begFrameIndexInc, int endFrameIndexExc);
    void eraseAllMasksForOne(int index);

    bool getCustomMaskIfHasOrUniqueMaskForOne(int videoIndex, int frameIndex, cv::Mat& mask) const;
    bool getCustomMasksIfHaveOrUniqueMasksForAll(const std::vector<int>& indexes, std::vector<cv::Mat>& masks) const;
    bool getAllCustomMasksForOne(int videoIndex, std::vector<int>& begFrameIndexesInc, std::vector<int>& endFrameIndexesInc,
        std::vector<cv::Mat>& masks) const;

    // deprecated interface
    bool seek(const std::vector<long long int>& timeStamps);
    bool stitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst, int frameIncrement);
    bool restitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst);

    bool getCurrReprojectForAll(std::vector<cv::Mat>& images, std::vector<long long int>& timeStamps) const;
    bool reReprojectForAll(std::vector<cv::Mat>& images, std::vector<long long int>& timeStamps);
    bool readNextAndReprojectForAll(std::vector<cv::Mat>& images, std::vector<long long int>& timeStamps);
    bool readNextAndReprojectForOne(int index, cv::Mat& image, long long int& timeStamp);
    bool readPrevAndReprojectForOne(int index, cv::Mat& image, long long int& timeStamp);

    bool setCustomMaskForOne(int index, long long int begInc, long long int endExc, const cv::Mat& mask);
    void eraseCustomMaskForOne(int index, long long int begInc, long long int endExc, long long int precision = 1000);

    bool getCustomMaskIfHasOrUniqueMaskForOne(int index, long long int timeStamp, cv::Mat& mask) const;
    bool getCustomMasksIfHaveOrUniqueMasksForAll(const std::vector<long long int>& timeStamps, std::vector<cv::Mat>& masks) const;
    bool getAllCustomMasksForOne(int index, std::vector<long long int>& begIncs, std::vector<long long int>& endExcs,
        std::vector<cv::Mat>& masks) const;

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
    bool stitch(std::vector<cv::Mat>& src, std::vector<long long int>& timeStamps, cv::Mat& dst, int frameIncrement = 1);
private:
    struct Impl;
    std::unique_ptr<Impl> ptrImpl;
};

// video encoder:
// h264
// h264_qsv
// nvenc_h264

// video preset
// ultrafast
// superfast
// veryfast
// faster
// fast
// medium
// slow
// slower
// veryslow

enum PanoStitchType
{
    PanoStitchTypeMISO, // multiple input single output
    PanoStitchTypeRicoh
};

class PanoramaLocalDiskTask
{
public:
    virtual ~PanoramaLocalDiskTask() {};
    virtual bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
        int panoStitchType, const std::string& cameraParamFile, const std::string& customMaskFile, const std::string& logoFile, int logoHFov,
        int highQualityBlend, const std::string& dstVideoFile, int dstWidth, int dstHeight, int dstVideoBitRate, 
        const std::string& dstVideoEncoder, const std::string& dstVideoPreset, int dstVideoMaxFrameCount) = 0;
    virtual bool start() = 0;
    virtual void waitForCompletion() = 0;
    virtual int getProgress() const = 0;
    virtual void cancel() = 0;
    virtual void getLastSyncErrorMessage(std::string& message) const = 0;
    virtual bool hasAsyncErrorMessage() const = 0;
    virtual void getLastAsyncErrorMessage(std::string& message) = 0;
};

class CPUPanoramaLocalDiskTask : public PanoramaLocalDiskTask
{
public:
    CPUPanoramaLocalDiskTask();
    ~CPUPanoramaLocalDiskTask();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
        int panoStitchType, const std::string& cameraParamFile, const std::string& customMaskFile, const std::string& logoFile, int logoHFov,
        int highQualityBlend, const std::string& dstVideoFile, int dstWidth, int dstHeight, int dstVideoBitRate,
        const std::string& dstVideoEncoder, const std::string& dstVideoPreset, int dstVideoMaxFrameCount);
    bool start();
    void waitForCompletion();
    int getProgress() const;
    void cancel();
    void getLastSyncErrorMessage(std::string& message) const;
    bool hasAsyncErrorMessage() const;
    void getLastAsyncErrorMessage(std::string& message);
private:
    struct Impl;
    std::unique_ptr<Impl> ptrImpl;
};

class IOclPanoramaLocalDiskTask : public PanoramaLocalDiskTask
{
public:
    IOclPanoramaLocalDiskTask();
    ~IOclPanoramaLocalDiskTask();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
        int panoStitchType, const std::string& cameraParamFile, const std::string& customMaskFile, const std::string& logoFile, int logoHFov,
        int highQualityBlend, const std::string& dstVideoFile, int dstWidth, int dstHeight, int dstVideoBitRate,
        const std::string& dstVideoEncoder, const std::string& dstVideoPreset, int dstVideoMaxFrameCount);
    bool start();
    void waitForCompletion();
    int getProgress() const;
    void cancel();
    void getLastSyncErrorMessage(std::string& message) const;
    bool hasAsyncErrorMessage() const;
    void getLastAsyncErrorMessage(std::string& message);
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
        int panoStitchType, const std::string& cameraParamFile, const std::string& customMaskFile, const std::string& logoFile, int logoHFov,
        int highQualityBlend, const std::string& dstVideoFile, int dstWidth, int dstHeight, int dstVideoBitRate,
        const std::string& dstVideoEncoder, const std::string& dstVideoPreset, int dstVideoMaxFrameCount);
    bool start();
    void waitForCompletion();
    int getProgress() const;
    void cancel();
    void getLastSyncErrorMessage(std::string& message) const;
    bool hasAsyncErrorMessage() const;
    void getLastAsyncErrorMessage(std::string& message);
private:
    struct Impl;
    std::unique_ptr<Impl> ptrImpl;
};

class DOclPanoramaLocalDiskTask : public PanoramaLocalDiskTask
{
public:
    DOclPanoramaLocalDiskTask();
    ~DOclPanoramaLocalDiskTask();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
        int panoStitchType, const std::string& cameraParamFile, const std::string& customMaskFile, const std::string& logoFile, int logoHFov,
        int highQualityBlend, const std::string& dstVideoFile, int dstWidth, int dstHeight, int dstVideoBitRate,
        const std::string& dstVideoEncoder, const std::string& dstVideoPreset, int dstVideoMaxFrameCount);
    bool start();
    void waitForCompletion();
    int getProgress() const;
    void cancel();
    void getLastSyncErrorMessage(std::string& message) const;
    bool hasAsyncErrorMessage() const;
    void getLastAsyncErrorMessage(std::string& message);
private:
    struct Impl;
    std::unique_ptr<Impl> ptrImpl;
};

class PanoramaLiveStreamTask
{
public:
    PanoramaLiveStreamTask();
    ~PanoramaLiveStreamTask();

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

    double getVideoSourceFrameRate() const;
    double getStitchFrameRate() const;
    void getLastSyncErrorMessage(std::string& message) const;
    bool hasAsyncErrorMessage() const;
    void getLastAsyncErrorMessage(std::string& message);
    void getLog(std::string& logInfo);

    bool getVideoSourceFrames(std::vector<avp::AudioVideoFrame2>& frames);
    bool getStitchedVideoFrame(avp::AudioVideoFrame2& frame);
    void cancelGetVideoSourceFrames();
    void cancelGetStitchedVideoFrame();

    void initAll();
    void closeAll();
    bool hasFinished() const;

private:
    struct Impl;
    std::unique_ptr<Impl> ptrImpl;
};

enum PanoramaType
{
    PanoTypeEquiRect,
    PanoTypeCube6x1,
    PanoTypeCube3x2,
    PanoTypeCube180,
    PanoTypeCount
};

enum AsyncErrorSource
{
    ErrorNone = -1,
    ErrorFromSources,
    ErrorFromStitch,
    ErrorFromLiveStream,
    ErrorFromSaveToDisk,
    ErrorCount
};

class PanoramaLiveStreamTask2
{
public:
    PanoramaLiveStreamTask2();
    ~PanoramaLiveStreamTask2();

    bool openAudioVideoSources(const std::vector<avp::Device>& devices, int width, int height, int frameRate, 
        bool openAudio = false, const avp::Device& device = avp::Device(), int sampleRate = 0);
    bool openAudioVideoSources(const std::vector<std::string>& urls, 
        bool openAudio = false, const std::string& url = std::string());
    void closeAudioVideoSources();

    bool beginVideoStitch(int stitchType, const std::string& configFileName, 
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

    double getVideoSourceFrameRate() const;
    double getStitchFrameRate() const;
    void getLastSyncErrorMessage(std::string& message) const;
    bool hasAsyncErrorMessage() const;
    void getLastAsyncErrorMessage(std::string& message, int& fromWhere);
    void getLog(std::string& logInfo);

    bool getVideoSourceFrames(std::vector<avp::AudioVideoFrame2>& frames);
    bool getStitchedVideoFrame(avp::AudioVideoFrame2& frame);
    void cancelGetVideoSourceFrames();
    void cancelGetStitchedVideoFrame();

    void initAll();
    void closeAll();
    bool hasFinished() const;

    int getNumVideos() const;
    int getVideoWidth() const;
    int getVideoHeight() const;
    double getVideoFrameRate() const;
    int getAudioSampleRate() const;

private:
    struct Impl;
    std::unique_ptr<Impl> ptrImpl;
};
