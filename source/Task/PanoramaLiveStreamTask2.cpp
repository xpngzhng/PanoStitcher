#include "CompileControl.h"
#include "PanoramaTask.h"
#include "ConcurrentQueue.h"
#include "PinnedMemoryFrameQueue.h"
#include "SharedAudioVideoFramePool.h"
#include "RicohUtil.h"
#include "PanoramaTaskUtil.h"
#include "LiveStreamTaskUtil.h"
#include "Timer.h"
#include "Image.h"
#include "oclobject.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

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

    bool openLiveStream(const std::string& name, int width, int height, int videoBPS,
        const std::string& videoEncoder, const std::string& videoPreset, int audioBPS);
    void closeLiveStream();

    bool beginSaveToDisk(const std::string& dir, int width, int height, int videoBPS,
        const std::string& videoEncoder, const std::string& videoPreset, int audioBPS, int fileDuration);
    void stopSaveToDisk();

    void setVideoSourceFrameRateCallback(FrameRateCallbackFunction func, void* data);
    void setStitchFrameRateCallback(FrameRateCallbackFunction func, void* data);
    void setLogCallback(LogCallbackFunction func, void* data);
    void initCallback();

    bool getVideoSourceFrames(std::vector<avp::AudioVideoFrame2>& frames);
    bool getStitchedVideoFrame(avp::AudioVideoFrame2& frame);
    void cancelGetVideoSourceFrames();
    void cancelGetStitchedVideoFrame();

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

    avp::AudioVideoWriter3 streamWriter;
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

    LogCallbackFunction logCallbackFunc;
    void* logCallbackData;

    FrameRateCallbackFunction videoFrameRateCallbackFunc;
    void* videoFrameRateCallbackData;

    FrameRateCallbackFunction stitchFrameRateCallbackFunc;
    void* stitchFrameRateCallbackData;

    int pixelType;
    int elemType;
    int finish;
    ForShowFrameVectorQueue syncedFramesBufferForShow;
#if COMPILE_CUDA
    BoundedPinnedMemoryFrameQueue syncedFramesBufferForProc;
#else
    ForShowFrameVectorQueue syncedFramesBufferForProc;
#endif
    AudioVideoFramePool procFramePool;
    ForShowFrameQueue procFrameBufferForShow;
    ForceWaitFrameQueue procFrameBufferForSend, procFrameBufferForSave;

#if COMPILE_CUDA
#else
    OpenCLBasic ocl;
#endif
};

PanoramaLiveStreamTask2::Impl::Impl()
#if !COMPILE_CUDA
: ocl("Intel", "GPU")
#endif
{
    initAll();
    initCallback();
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
        return false;
    }

    audioVideoSource.reset(new FFmpegAudioVideoSource(&syncedFramesBufferForShow, &syncedFramesBufferForProc, COMPILE_CUDA,
        &procFrameBufferForSend, &procFrameBufferForSave, &finish, logCallbackFunc, logCallbackData,
        videoFrameRateCallbackFunc, videoFrameRateCallbackData));
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
        return false;
    }

    bool ok = false;
    if (areAllIPAdresses(urls))
    {
        audioVideoSource.reset(new JuJingAudioVideoSource(&syncedFramesBufferForShow, &syncedFramesBufferForProc, COMPILE_CUDA,
            &procFrameBufferForSend, &procFrameBufferForSave, &finish, logCallbackFunc, logCallbackData,
            videoFrameRateCallbackFunc, videoFrameRateCallbackData));
        ok = ((JuJingAudioVideoSource*)audioVideoSource.get())->open(urls);
    }
    else
    {
        audioVideoSource.reset(new FFmpegAudioVideoSource(&syncedFramesBufferForShow, &syncedFramesBufferForProc, COMPILE_CUDA,
            &procFrameBufferForSend, &procFrameBufferForSave, &finish, logCallbackFunc, logCallbackData,
            videoFrameRateCallbackFunc, videoFrameRateCallbackData));
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
        return false;
    }

    if (!renderThreadJoined)
    {
        ptlprintf("Error in %s, stitching running, stop before launching new stitching\n", __FUNCTION__);
        return false;
    }

    renderConfigName = configFileName;
    renderFrameSize.width = width;
    renderFrameSize.height = height;

#if COMPILE_CUDA
    renderPrepareSuccess = render.prepare(renderConfigName, "", highQualityBlend, false,
        videoFrameSize, renderFrameSize);
#else
    renderPrepareSuccess = render.prepare(renderConfigName, highQualityBlend, false,
        videoFrameSize, renderFrameSize, &ocl);
#endif
    if (!renderPrepareSuccess)
    {
        ptlprintf("Could not prepare for video stitch\n");

        if (logCallbackFunc)
            logCallbackFunc("Video stitch prepare failed", logCallbackData);

        return false;
    }

    if (render.getNumImages() != numVideos)
    {
        ptlprintf("Error in %s, num images in render not equal to num videos\n", __FUNCTION__);
        return false;
    }

    renderPrepareSuccess = procFramePool.initAsVideoFramePool(pixelType, width, height);
    if (!renderPrepareSuccess)
    {
        ptlprintf("Could not init proc frame pool\n");

        if (logCallbackFunc)
            logCallbackFunc("Video stitch prepare failed", logCallbackData);

        return false;
    }

    if (addLogo)
        renderPrepareSuccess = logoFilter.init(width, height, elemType);
    if (!renderPrepareSuccess)
    {
        ptlprintf("Could not init logo filter\n");

        if (logCallbackFunc)
            logCallbackFunc("Video stitch prepare failed", logCallbackData);

        return false;
    }

    if (logCallbackFunc)
        logCallbackFunc("Video stitch prepare success", logCallbackData);

    syncedFramesBufferForProc.clear();
    procFrameBufferForShow.clear();
    procFrameBufferForSave.clear();
    procFrameBufferForSend.clear();

    renderEndFlag = 0;
    renderThreadJoined = 0;
    renderThread.reset(new std::thread(&PanoramaLiveStreamTask2::Impl::procVideo, this));
    postProcThread.reset(new std::thread(&PanoramaLiveStreamTask2::Impl::postProc, this));

    if (logCallbackFunc)
        logCallbackFunc("Video stitch thread create success", logCallbackData);

    return true;
}

void PanoramaLiveStreamTask2::Impl::stopVideoStitch()
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

        if (logCallbackFunc)
            logCallbackFunc("Video stitch thread close success", logCallbackData);
    }
}

bool PanoramaLiveStreamTask2::Impl::getVideoSourceFrames(std::vector<avp::AudioVideoFrame2>& frames)
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

bool PanoramaLiveStreamTask2::Impl::getStitchedVideoFrame(avp::AudioVideoFrame2& frame)
{
    return procFrameBufferForShow.pull(frame);
    /*avp::SharedAudioVideoFrame tempFrame;
    bool ok = procFrameBufferForShow.pull(tempFrame);
    if (ok)
    avp::copy(tempFrame, frame);
    return ok;*/
}

void PanoramaLiveStreamTask2::Impl::cancelGetVideoSourceFrames()
{
    //syncedFramesBufferForShow.stop();
}

void PanoramaLiveStreamTask2::Impl::cancelGetStitchedVideoFrame()
{
    //procFrameBufferForShow.stop();
}

bool PanoramaLiveStreamTask2::Impl::openLiveStream(const std::string& name,
    int width, int height, int videoBPS, const std::string& videoEncoder, const std::string& videoPreset, int audioBPS)
{
    if (!videoOpenSuccess || !audioVideoSource)
    {
        ptlprintf("Error in %s, audio video sources not opened\n", __FUNCTION__);
        return false;
    }

    if (!renderPrepareSuccess || renderThreadJoined)
    {
        ptlprintf("Error in %s, render not running, cannot launch live streaming\n", __FUNCTION__);
        return false;
    }

    if (!streamThreadJoined)
    {
        ptlprintf("Error in %s, live streaming running, stop before launching new live streaming\n", __FUNCTION__);
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
        audioOpenSuccess, "aac", audioVideoSource->getAudioSampleType(),
        audioVideoSource->getAudioChannelLayout(), audioVideoSource->getAudioSampleRate(), streamAudioBitRate,
        true, videoEncoder == "h264_qsv" ? "h264_qsv" : "h264", pixelType, streamFrameSize.width, streamFrameSize.height,
        videoFrameRate, streamVideoBitRate, writerOpts);
    if (!streamOpenSuccess)
    {
        ptlprintf("Could not open streaming url with frame rate = %f and bit rate = %d\n", videoFrameRate, streamVideoBitRate);

        if (logCallbackFunc)
            logCallbackFunc("Live stream open failed", logCallbackData);

        return false;
    }

    if (logCallbackFunc)
        logCallbackFunc("Live stream open success", logCallbackData);

    procFrameBufferForSend.resume();
    streamEndFlag = 0;
    streamThreadJoined = 0;
    streamThread.reset(new std::thread(&PanoramaLiveStreamTask2::Impl::streamSend, this));

    if (logCallbackFunc)
        logCallbackFunc("Live stream thread create success", logCallbackData);

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

        if (logCallbackFunc)
        {
            logCallbackFunc("Live stream close success", logCallbackData);
            logCallbackFunc("Live stream thread close success", logCallbackData);
        }
    }
}

bool PanoramaLiveStreamTask2::Impl::beginSaveToDisk(const std::string& dir, int width, int height, int videoBPS,
    const std::string& videoEncoder, const std::string& videoPreset, int audioBPS, int fileDurationInSeconds)
{
    if (!videoOpenSuccess || !audioVideoSource)
    {
        ptlprintf("Error in %s, audio video sources not opened\n", __FUNCTION__);
        return false;
    }

    if (!renderPrepareSuccess || renderThreadJoined)
    {
        ptlprintf("Error in %s, render not running, cannot launch saving to disk\n", __FUNCTION__);
        return false;
    }

    if (!fileThreadJoined)
    {
        ptlprintf("Error in %s, saving to disk running, stop before launching new saving to disk\n", __FUNCTION__);
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
    fileThread.reset(new std::thread(&PanoramaLiveStreamTask2::Impl::fileSave, this));

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
    }
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

#if COMPILE_CUDA
    std::vector<cv::cuda::HostMem> mems;
    std::vector<long long int> timeStamps;
#else
    std::vector<avp::SharedAudioVideoFrame> frames;
    std::vector<long long int> timeStamps;
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
        //ptlprintf("show\n");
#if COMPILE_CUDA
        if (!syncedFramesBufferForProc.pull(mems, timeStamps))
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
        //ptlprintf("ts %lld\n", timeStamp);
        //ptlprintf("before check size\n");
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
                    ptlprintf("%d  %f, %f\n", count, elapse, r);
                    timer.start();
                    count = 0;

                    if (stitchFrameRateCallbackFunc)
                        stitchFrameRateCallbackFunc(r, stitchFrameRateCallbackData);
                }
            }

#if COMPILE_CUDA
            src.resize(numVideos);
            for (int i = 0; i < numVideos; i++)
                src[i] = mems[i].createMatHeader();
            //procTimer.start();
            ok = render.render(src, timeStamps);
            //procTimer.end();
#else
            src.resize(numVideos);
            timeStamps.resize(numVideos);
            for (int i = 0; i < numVideos; i++)
            {
                src[i] = cv::Mat(frames[i].height, frames[i].width, elemType, frames[i].data, frames[i].step);
                timeStamps[i] = frames[i].timeStamp;
            }
            procTimer.start();
            ok = render.render(src, timeStamps);
            procTimer.end();
#endif
            if (!ok)
            {
                ptlprintf("Error in %s [%8x], render failed\n", __FUNCTION__, id);
                finish = 1;
                break;
            }

            localTimer.end();
            //ptlprintf("%f, %f\n", procTimer.elapse(), localTimer.elapse());
        }
    }

    render.stop();

    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask2::Impl::postProc()
{
    size_t id = std::this_thread::get_id().hash();
    ptlprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    avp::AudioVideoFrame2 frame;
    while (true)
    {
        if (finish || renderEndFlag)
            break;

        procFramePool.get(frame);
        cv::Mat result(frame.height, frame.width, elemType, frame.data[0], frame.steps[0]);

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
        //ptlprintf("%f\n", timer.elapse());
    }

    //procFrameBufferForShow.stop();
    procFrameBufferForSend.stop();
    procFrameBufferForSave.stop();

    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask2::Impl::streamSend()
{
    size_t id = std::this_thread::get_id().hash();
    ptlprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    cv::Mat dstMat;
    avp::AudioVideoFrame2 frame;
    while (true)
    {
        if (finish || streamEndFlag)
            break;
        procFrameBufferForSend.pull(frame);
        if (frame.data)
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
                finish = 1;
                break;
            }
        }
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

    char buf[1024];
    int count = 0;
    cv::Mat dstMat;
    avp::AudioVideoFrame2 frame;
    avp::AudioVideoWriter3 writer;
    std::vector<avp::Option> writerOpts;
    writerOpts.push_back(std::make_pair("preset", fileVideoEncodePreset));
    sprintf(buf, fileWriterFormat.c_str(), count++);
    bool ok = writer.open(buf, "mp4", true,
        audioOpenSuccess, "aac", audioVideoSource->getAudioSampleType(),
        audioVideoSource->getAudioChannelLayout(), audioVideoSource->getAudioSampleRate(), fileAudioBitRate,
        true, fileVideoEncoder, pixelType, fileFrameSize.width, fileFrameSize.height,
        videoFrameRate, fileVideoBitRate, writerOpts);
    if (!ok)
    {
        ptlprintf("Error in %s [%d], could not save current audio video\n", __FUNCTION__, id);
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
                sprintf(buf, fileWriterFormat.c_str(), count++);
                ok = writer.open(buf, "mp4", true,
                    audioOpenSuccess, "aac", audioVideoSource->getAudioSampleType(),
                    audioVideoSource->getAudioChannelLayout(), audioVideoSource->getAudioSampleRate(), fileAudioBitRate,
                    true, fileVideoEncoder, pixelType, fileFrameSize.width, fileFrameSize.height,
                    videoFrameRate, fileVideoBitRate, writerOpts);
                if (!ok)
                {
                    ptlprintf("Error in %s [%d], could not save current audio video\n", __FUNCTION__, id);
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
            ok = writer.write(shallow);
            if (!ok)
            {
                ptlprintf("Error in %s [%d], could not write current frame\n", __FUNCTION__, id);
                break;
            }
        }
    }
    if (logCallbackFunc)
        logCallbackFunc(std::string("Finish write local file ") + buf, logCallbackData);
    writer.close();

    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void PanoramaLiveStreamTask2::Impl::setVideoSourceFrameRateCallback(FrameRateCallbackFunction func, void* data)
{
    videoFrameRateCallbackFunc = func;
    videoFrameRateCallbackData = data;
}

void PanoramaLiveStreamTask2::Impl::setStitchFrameRateCallback(FrameRateCallbackFunction func, void* data)
{
    stitchFrameRateCallbackFunc = func;
    stitchFrameRateCallbackData = data;
}

void PanoramaLiveStreamTask2::Impl::setLogCallback(LogCallbackFunction func, void* data)
{
    logCallbackFunc = func;
    logCallbackData = data;
}

void PanoramaLiveStreamTask2::Impl::initCallback()
{
    logCallbackFunc = 0;
    videoFrameRateCallbackFunc = 0;
    stitchFrameRateCallbackFunc = 0;
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

bool PanoramaLiveStreamTask2::openLiveStream(const std::string& name, int width, int height, int videoBPS,
    const std::string& videoEncoder, const std::string& videoPreset, int audioBPS)
{
    return ptrImpl->openLiveStream(name, width, height, videoBPS, videoEncoder, videoPreset, audioBPS);
}

void PanoramaLiveStreamTask2::closeLiveStream()
{
    ptrImpl->closeLiveStream();
}

bool PanoramaLiveStreamTask2::beginSaveToDisk(const std::string& dir, int width, int height, int videoBPS,
    const std::string& videoEncoder, const std::string& videoPreset, int audioBPS, int fileDuration)
{
    return ptrImpl->beginSaveToDisk(dir, width, height, videoBPS, videoEncoder, videoPreset, audioBPS, fileDuration);
}

void PanoramaLiveStreamTask2::stopSaveToDisk()
{
    ptrImpl->stopSaveToDisk();
}

void PanoramaLiveStreamTask2::setVideoSourceFrameRateCallback(FrameRateCallbackFunction func, void* data)
{
    ptrImpl->setVideoSourceFrameRateCallback(func, data);
}

void PanoramaLiveStreamTask2::setStitchFrameRateCallback(FrameRateCallbackFunction func, void* data)
{
    ptrImpl->setStitchFrameRateCallback(func, data);
}

void PanoramaLiveStreamTask2::setLogCallback(LogCallbackFunction func, void* data)
{
    ptrImpl->setLogCallback(func, data);
}

void PanoramaLiveStreamTask2::initCallback()
{
    ptrImpl->initCallback();
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