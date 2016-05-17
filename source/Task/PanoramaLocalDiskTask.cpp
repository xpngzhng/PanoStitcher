#include "PanoramaTask.h"
#include "PanoramaTaskUtil.h"
#include "ConcurrentQueue.h"
#include "ZBlend.h"
#include "ZReproject.h"
#include "RicohUtil.h"
#include "PinnedMemoryPool.h"
#include "SharedAudioVideoFramePool.h"
#include "Timer.h"
#include "Image.h"

struct CPUPanoramaLocalDiskTask::Impl
{
    Impl();
    ~Impl();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
        const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight,
        int dstVideoBitRate, const std::string& dstVideoEncoder, const std::string& dstVideoPreset, 
        int dstVideoMaxFrameCount, ProgressCallbackFunction func, void* data);
    bool start();
    void waitForCompletion();
    int getProgress() const;
    void cancel();

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
    LogoFilter logoFilter;
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

CPUPanoramaLocalDiskTask::Impl::Impl()
{
    clear();
}

CPUPanoramaLocalDiskTask::Impl::~Impl()
{
    clear();
}

bool CPUPanoramaLocalDiskTask::Impl::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
    int tryAudioIndex, const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight,
    int dstVideoBitRate, const std::string& dstVideoEncoder, const std::string& dstVideoPreset,
    int dstVideoMaxFrameCount, ProgressCallbackFunction func, void* data)
{
    clear();

    if (srcVideoFiles.empty() || (srcVideoFiles.size() != offsets.size()))
    {
        printf("Error in %s, size of srcVideoFiles and size of offsets empty or unmatch.\n", __FUNCTION__);
        return false;
    }

    numVideos = srcVideoFiles.size();

    std::vector<PhotoParam> params;
    if (!loadPhotoParams(cameraParamFile, params))
    {
        printf("Error in %s, failed to load params\n", __FUNCTION__);
    }
    if (params.size() < numVideos)
    {
        printf("Error in %s, params.size() < numVideos\n", __FUNCTION__);
        return false;
    }
    else if (params.size() > numVideos)
    {
        printf("Warning in %s, params.size() > numVideos\n", __FUNCTION__);
    }

    dstSize.width = dstWidth;
    dstSize.height = dstHeight;

    printf("Info in %s, open videos and set to the correct frames\n", __FUNCTION__);

    bool ok = false;
    ok = prepareSrcVideos(srcVideoFiles, true, offsets, tryAudioIndex, readers, audioIndex, srcSize, validFrameCount);
    if (!ok)
    {
        printf("Error in %s, could not open video file(s)\n", __FUNCTION__);
        return false;
    }

    if (dstVideoMaxFrameCount > 0 && validFrameCount > dstVideoMaxFrameCount)
        validFrameCount = dstVideoMaxFrameCount;

    printf("Info in %s, open videos done\n", __FUNCTION__);
    printf("Info in %s, prepare for reproject and blend\n", __FUNCTION__);

    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, dstMasks);

    ok = blender.prepare(dstMasks, 16, 2);
    //ok = blender.prepare(dstMasks, 50);
    if (!ok)
    {
        printf("Error in %s, blender prepare failed\n", __FUNCTION__);
        return false;
    }

    ok = logoFilter.init(dstSize.width, dstSize.height, CV_8UC3);
    if (!ok)
    {
        printf("Error in %s, init logo filter failed\n", __FUNCTION__);
        return false;
    }

    printf("Info in %s, prepare finish\n", __FUNCTION__);

    printf("Info in %s, open dst video\n", __FUNCTION__);
    std::vector<avp::Option> options;
    options.push_back(std::make_pair("preset", dstVideoPreset));
    std::string format = dstVideoEncoder == "h264_qsv" ? "h264_qsv" : "h264";
    if (audioIndex >= 0 && audioIndex < numVideos)
    {
        ok = writer.open(dstVideoFile, "", true, 
            true, "aac", readers[audioIndex].getAudioSampleType(), readers[audioIndex].getAudioChannelLayout(), 
            readers[audioIndex].getAudioSampleRate(), 128000,
            true, format, avp::PixelTypeBGR24, dstSize.width, dstSize.height, readers[0].getVideoFps(), dstVideoBitRate, options);
    }
    else
    {
        ok = writer.open(dstVideoFile, "", false, false, "", avp::SampleTypeUnknown, 0, 0, 0,
            true, format, avp::PixelTypeBGR24, dstSize.width, dstSize.height, readers[0].getVideoFps(), dstVideoBitRate, options);
    }
    if (!ok)
    {
        printf("Error in %s, video writer open failed\n", __FUNCTION__);
        return false;
    }
    else
        printf("Info in %s, video writer open success\n", __FUNCTION__);

    finishPercent.store(0);
    progressCallbackFunc = func;
    progressCallbackData = data;

    initSuccess = true;
    finish = false;
    return true;
}

void CPUPanoramaLocalDiskTask::Impl::run()
{
    if (!initSuccess)
        return;

    if (finish)
        return;

    printf("Info in %s, write video begin\n", __FUNCTION__);

    int count = 0;
    int step = 1;
    if (validFrameCount > 0)
        step = validFrameCount / 100.0 + 0.5;
    if (step < 1)
        step = 1;
    printf("validFrameCount = %d, step = %d\n", validFrameCount, step);

    std::vector<avp::AudioVideoFrame> frames(numVideos);
    std::vector<cv::Mat> images(numVideos);
    bool ok = true;
    blendImage.create(dstSize, CV_8UC3);
    while (true)
    {
        ok = true;
        //printf("begin ");
        if (audioIndex >= 0 && audioIndex < numVideos)
        {
            if (!readers[audioIndex].read(frames[audioIndex]))
                break;

            //printf("[%d] ", audioIndex);
            if (frames[audioIndex].mediaType == avp::AUDIO)
            {
                //printf("audio ");
                ok = writer.write(frames[audioIndex]);
                if (!ok)
                {
                    printf("write fail\n");
                    break;
                }
                continue;
            }
            else
            {
                //printf("video ");
                images[audioIndex] = cv::Mat(frames[audioIndex].height, frames[audioIndex].width, CV_8UC3, 
                    frames[audioIndex].data, frames[audioIndex].step);
            }
        }
        for (int i = 0; i < numVideos; i++)
        {
            if (i == audioIndex)
                continue;

            //printf("[%d] ", i);
            if (!readers[i].read(frames[i]))
            {
                ok = false;
                break;
            }

            images[i] = cv::Mat(frames[i].height, frames[i].width, CV_8UC3, frames[i].data, frames[i].step);
        }
        //printf("\n");
        if (!ok || endFlag)
            break;

        reprojectParallelTo16S(images, reprojImages, dstSrcMaps);
        blender.blend(reprojImages, blendImage);
        //printf("blend finish\n");
        if (addLogo)
            logoFilter.addLogo(blendImage);
        avp::AudioVideoFrame frame = avp::videoFrame(blendImage.data, blendImage.step, avp::PixelTypeBGR24, 
            blendImage.cols, blendImage.rows, frames[0].timeStamp);
        ok = writer.write(frame);

        if (!ok)
        {
            printf("write fail\n");
            break;
        }

        count++;
        //printf("write count = %d\n", count);
        //if (progressCallbackFunc && (count % step == 0))
        //    progressCallbackFunc(double(count) / (validFrameCount > 0 ? validFrameCount : 100), progressCallbackData);
        if (count % step == 0)
            finishPercent.store(double(count) / (validFrameCount > 0 ? validFrameCount : 100) * 100);

        if (count >= validFrameCount)
            break;

        //printf("finish\n");
    }

    for (int i = 0; i < numVideos; i++)
        readers[i].close();
    writer.close();

    //if (progressCallbackFunc)
    //    progressCallbackFunc(1.0, progressCallbackData);
    finishPercent.store(100);

    printf("Info in %s, write video finish\n", __FUNCTION__);

    finish = true;
}

bool CPUPanoramaLocalDiskTask::Impl::start()
{
    if (!initSuccess)
        return false;

    if (finish)
        return false;

    thread.reset(new std::thread(&CPUPanoramaLocalDiskTask::Impl::run, this));
    return true;
}

void CPUPanoramaLocalDiskTask::Impl::waitForCompletion()
{
    if (thread && thread->joinable())
        thread->join();
    thread.reset(0);
}

int CPUPanoramaLocalDiskTask::Impl::getProgress() const
{
    return finishPercent.load();
}

void CPUPanoramaLocalDiskTask::Impl::cancel()
{
    endFlag = true;
}

void CPUPanoramaLocalDiskTask::Impl::clear()
{
    numVideos = 0;
    srcSize = cv::Size();
    dstSize = cv::Size();
    readers.clear();
    dstMasks.clear();
    dstSrcMaps.clear();
    reprojImages.clear();
    writer.close();
    endFlag = false;

    finishPercent.store(0);

    validFrameCount = 0;

    progressCallbackFunc = 0;
    progressCallbackData = 0;

    if (thread && thread->joinable())
        thread->join();
    thread.reset(0);

    initSuccess = false;
    finish = true;
}

CPUPanoramaLocalDiskTask::CPUPanoramaLocalDiskTask()
{
    ptrImpl.reset(new Impl);
}

CPUPanoramaLocalDiskTask::~CPUPanoramaLocalDiskTask()
{

}

bool CPUPanoramaLocalDiskTask::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
    const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight,
    int dstVideoBitRate, const std::string& dstVideoEncoder, const std::string& dstVideoPreset, 
    int dstVideoMaxFrameCount, ProgressCallbackFunction func, void* data)
{
    return ptrImpl->init(srcVideoFiles, offsets, audioIndex, cameraParamFile, dstVideoFile, dstWidth, dstHeight,
        dstVideoBitRate, dstVideoEncoder, dstVideoPreset, dstVideoMaxFrameCount, func, data);
}

bool CPUPanoramaLocalDiskTask::start()
{
    return ptrImpl->start();
}

void CPUPanoramaLocalDiskTask::waitForCompletion()
{
    ptrImpl->waitForCompletion();
}

int CPUPanoramaLocalDiskTask::getProgress() const
{
    return ptrImpl->getProgress();
}

void CPUPanoramaLocalDiskTask::cancel()
{
    ptrImpl->cancel();
}

struct StampedPinnedMemoryVector
{
    std::vector<cv::cuda::HostMem> frames;
    long long int timeStamp;
};

typedef BoundedCompleteQueue<avp::SharedAudioVideoFrame> FrameBuffer;
typedef BoundedCompleteQueue<StampedPinnedMemoryVector> FrameVectorBuffer;

struct CudaPanoramaLocalDiskTask::Impl
{
    Impl();
    ~Impl();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
        const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight,
        int dstVideoBitRate, const std::string& dstVideoEncoder, const std::string& dstVideoPreset, 
        int dstVideoMaxFrameCount, ProgressCallbackFunction func, void* data);
    bool start();
    void waitForCompletion();
    int getProgress() const;
    void cancel();

    void clear();

    int numVideos;
    int audioIndex;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader> readers;
    CudaPanoramaRender render;
    PinnedMemoryPool srcFramesMemoryPool;
    SharedAudioVideoFramePool audioFramesMemoryPool, dstFramesMemoryPool;
    FrameVectorBuffer decodeFramesBuffer;
    FrameBuffer procFrameBuffer;
    cv::Mat blendImageCpu;
    LogoFilter logoFilter;
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
    void postProc();
    void encode();

    std::unique_ptr<std::thread> decodeThread;
    std::unique_ptr<std::thread> procThread;
    std::unique_ptr<std::thread> postProcThread;
    std::unique_ptr<std::thread> encodeThread;

    bool initSuccess;
    bool finish;
    bool isCanceled;
};

CudaPanoramaLocalDiskTask::Impl::Impl()
{
    clear();
}

CudaPanoramaLocalDiskTask::Impl::~Impl()
{
    clear();
}

bool CudaPanoramaLocalDiskTask::Impl::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
    int tryAudioIndex, const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight,
    int dstVideoBitRate, const std::string& dstVideoEncoder, const std::string& dstVideoPreset, 
    int dstVideoMaxFrameCount, ProgressCallbackFunction func, void* data)
{
    clear();

    if (srcVideoFiles.empty() || (srcVideoFiles.size() != offsets.size()))
    {
        printf("Error in %s, size of srcVideoFiles and size of offsets empty or unmatch.\n", __FUNCTION__);
        return false;
    }

    numVideos = srcVideoFiles.size();

    dstSize.width = dstWidth;
    dstSize.height = dstHeight;

    printf("Info in %s, open videos and set to the correct frames\n", __FUNCTION__);

    bool ok = false;
    ok = prepareSrcVideos(srcVideoFiles, false, offsets, tryAudioIndex, readers, audioIndex, srcSize, validFrameCount);
    if (!ok)
    {
        printf("Error in %s, could not open video file(s)\n", __FUNCTION__);
        return false;
    }

    if (dstVideoMaxFrameCount > 0 && validFrameCount > dstVideoMaxFrameCount)
        validFrameCount = dstVideoMaxFrameCount;

    ok = srcFramesMemoryPool.init(readers[0].getVideoHeight(), readers[0].getVideoWidth(), CV_8UC4);
    if (!ok)
    {
        printf("Error in %s, could not init memory pool\n", __FUNCTION__);
        return false;
    }

    if (audioIndex >= 0 && audioIndex < numVideos)
    {
        ok = audioFramesMemoryPool.initAsAudioFramePool(readers[audioIndex].getAudioSampleType(),
            readers[audioIndex].getAudioNumChannels(), readers[audioIndex].getAudioChannelLayout(),
            readers[audioIndex].getAudioNumSamples());
        if (!ok)
        {
            printf("Error in %s, could not init memory pool\n", __FUNCTION__);
            return false;
        }
    }

    printf("Info in %s, open videos done\n", __FUNCTION__);
    printf("Info in %s, prepare for reproject and blend\n", __FUNCTION__);

    ok = render.prepare(cameraParamFile, true, true, srcSize, dstSize);
    if (!ok)
    {
        printf("Error in %s, render prepare failed\n", __FUNCTION__);
        return false;
    }

    ok = dstFramesMemoryPool.initAsVideoFramePool(avp::PixelTypeBGR32, dstSize.width, dstSize.height);
    if (!ok)
    {
        printf("Error in %s, could not init memory pool\n", __FUNCTION__);
        return false;
    }

    ok = logoFilter.init(dstSize.width, dstSize.height, CV_8UC4);
    if (!ok)
    {
        printf("Error in %s, init logo filter failed\n", __FUNCTION__);
        return false;
    }

    printf("Info in %s, prepare finish\n", __FUNCTION__);

    printf("Info in %s, open dst video\n", __FUNCTION__);
    std::vector<avp::Option> options;
    options.push_back(std::make_pair("preset", dstVideoPreset));
    std::string format = dstVideoEncoder == "h264_qsv" ? "h264_qsv" : "h264";
    if (audioIndex >= 0 && audioIndex < numVideos)
    {
        ok = writer.open(dstVideoFile, "", true, true, "aac", readers[audioIndex].getAudioSampleType(),
            readers[audioIndex].getAudioChannelLayout(), readers[audioIndex].getAudioSampleRate(), 128000,
            true, format, avp::PixelTypeBGR32, dstSize.width, dstSize.height, readers[0].getVideoFps(), dstVideoBitRate, options);
    }
    else
    {
        ok = writer.open(dstVideoFile, "", false, false, "", avp::SampleTypeUnknown, 0, 0, 0,
            true, format, avp::PixelTypeBGR32, dstSize.width, dstSize.height, readers[0].getVideoFps(), dstVideoBitRate, options);
    }
    if (!ok)
    {
        printf("Error in %s, video writer open failed\n", __FUNCTION__);
        return false;
    }
    else
        printf("Info in %s, video writer open success\n", __FUNCTION__);

    decodeFramesBuffer.setMaxSize(4);
    procFrameBuffer.setMaxSize(8);

    finishPercent.store(0);
    progressCallbackFunc = func;
    progressCallbackData = data;

    initSuccess = true;
    finish = false;
    return true;
}

bool CudaPanoramaLocalDiskTask::Impl::start()
{
    if (!initSuccess)
        return false;

    if (finish)
        return false;

    printf("Info in %s, write video begin\n", __FUNCTION__);

    decodeThread.reset(new std::thread(&CudaPanoramaLocalDiskTask::Impl::decode, this));
    procThread.reset(new std::thread(&CudaPanoramaLocalDiskTask::Impl::proc, this));
    postProcThread.reset(new std::thread(&CudaPanoramaLocalDiskTask::Impl::postProc, this));
    encodeThread.reset(new std::thread(&CudaPanoramaLocalDiskTask::Impl::encode, this));

    return true;
}

void CudaPanoramaLocalDiskTask::Impl::waitForCompletion()
{
    if (decodeThread && decodeThread->joinable())
        decodeThread->join();
    decodeThread.reset();
    if (procThread && procThread->joinable())
        procThread->join();
    procThread.reset(0);
    if (postProcThread && postProcThread->joinable())
        postProcThread->join();
    postProcThread.reset(0);
    if (encodeThread && encodeThread->joinable())
        encodeThread->join();
    encodeThread.reset(0);

    if (!finish)
        printf("Info in %s, write video finish\n", __FUNCTION__);

    finish = true;
}

int CudaPanoramaLocalDiskTask::Impl::getProgress() const
{
    return finishPercent.load();
}

void CudaPanoramaLocalDiskTask::Impl::clear()
{
    numVideos = 0;
    srcSize = cv::Size();
    dstSize = cv::Size();
    readers.clear();
    writer.close();

    srcFramesMemoryPool.clear();
    audioFramesMemoryPool.clear();
    dstFramesMemoryPool.clear();

    decodeFramesBuffer.clear();
    procFrameBuffer.clear();

    decodeCount = 0;
    procCount = 0;
    encodeCount = 0;
    finishPercent.store(0);

    validFrameCount = 0;

    if (decodeThread && decodeThread->joinable())
        decodeThread->join();
    decodeThread.reset(0);
    if (procThread && procThread->joinable())
        procThread->join();
    procThread.reset(0);
    if (postProcThread && postProcThread->joinable())
        postProcThread->join();
    postProcThread.reset(0);
    if (encodeThread && encodeThread->joinable())
        encodeThread->join();
    encodeThread.reset(0);

    progressCallbackFunc = 0;
    progressCallbackData = 0;

    initSuccess = false;
    finish = true;
    isCanceled = false;
}

void CudaPanoramaLocalDiskTask::Impl::cancel()
{
    isCanceled = true;
}

void CudaPanoramaLocalDiskTask::Impl::decode()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    decodeCount = 0;
    std::vector<avp::AudioVideoFrame> shallowFrames(numVideos);
    avp::SharedAudioVideoFrame audioFrame;

    while (true)
    {
        if (audioIndex >= 0 && audioIndex < numVideos)
        {
            if (!readers[audioIndex].read(shallowFrames[audioIndex]))
                break;

            if (shallowFrames[audioIndex].mediaType == avp::AUDIO)
            {
                audioFramesMemoryPool.get(audioFrame);
                avp::copy(shallowFrames[audioIndex], audioFrame);
                procFrameBuffer.push(audioFrame);
                continue;
            }
        }

        bool successRead = true;
        for (int i = 0; i < numVideos; i++)
        {
            if (i == audioIndex)
                continue;

            if (!readers[i].read(shallowFrames[i]))
            {
                successRead = false;
                break;
            }
        }
        if (!successRead || isCanceled)
            break;

        StampedPinnedMemoryVector deepFrames;
        deepFrames.timeStamp = shallowFrames[0].timeStamp;
        deepFrames.frames.resize(numVideos);
        for (int i = 0; i < numVideos; i++)
        {
            srcFramesMemoryPool.get(deepFrames.frames[i]);
            cv::Mat src(shallowFrames[i].height, shallowFrames[i].width, CV_8UC4, shallowFrames[i].data, shallowFrames[i].step);
            cv::Mat dst = deepFrames.frames[i].createMatHeader();
            src.copyTo(dst);
        }

        decodeFramesBuffer.push(deepFrames);
        decodeCount++;
        //printf("decode count = %d\n", decodeCount);

        if (decodeCount >= validFrameCount)
            break;
    }

    while (decodeFramesBuffer.size())
        std::this_thread::sleep_for(std::chrono::microseconds(25));
    decodeFramesBuffer.stop();

    for (int i = 0; i < numVideos; i++)
        readers[i].close();

    printf("total decode %d\n", decodeCount);
    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void CudaPanoramaLocalDiskTask::Impl::proc()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    procCount = 0;
    StampedPinnedMemoryVector srcFrames;
    std::vector<cv::Mat> images(numVideos);
    while (true)
    {
        if (!decodeFramesBuffer.pull(srcFrames))
            break;
        
        for (int i = 0; i < numVideos; i++)
            images[i] = srcFrames.frames[i].createMatHeader();        
        render.render(images, srcFrames.timeStamp);
        procCount++;
        //printf("proc count = %d\n", procCount);
    }
    
    render.waitForCompletion();
    render.stop();

    printf("total proc %d\n", procCount);
    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void CudaPanoramaLocalDiskTask::Impl::postProc()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    avp::SharedAudioVideoFrame dstFrame;
    while (true)
    {
        dstFramesMemoryPool.get(dstFrame);
        cv::Mat result(dstSize, CV_8UC4, dstFrame.data, dstFrame.step);
        if (!render.getResult(result, dstFrame.timeStamp))
            break;

        cv::Mat image(dstFrame.height, dstFrame.width, CV_8UC4, dstFrame.data, dstFrame.step);
        if (addLogo)
            logoFilter.addLogo(image);

        procFrameBuffer.push(dstFrame);
    }

    while (procFrameBuffer.size())
        std::this_thread::sleep_for(std::chrono::microseconds(25));
    procFrameBuffer.stop();

    printf("total proc %d\n", procCount);
    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void CudaPanoramaLocalDiskTask::Impl::encode()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    encodeCount = 0;
    int step = 1;
    if (validFrameCount > 0)
        step = validFrameCount / 100.0 + 0.5;
    if (step < 1)
        step = 1;
    printf("validFrameCount = %d, step = %d\n", validFrameCount, step);
    ztool::Timer timerEncode;
    encodeCount = 0;
    avp::SharedAudioVideoFrame deepFrame;
    while (true)
    {
        if (!procFrameBuffer.pull(deepFrame))
            break;

        timerEncode.start();
        writer.write(avp::AudioVideoFrame(deepFrame));
        timerEncode.end();

        // Only when the frame is of type video can we increase encodeCount
        if (deepFrame.mediaType == avp::VIDEO)
            encodeCount++;
        printf("frame %d finish, encode time = %f\n", encodeCount, timerEncode.elapse());

        //if (progressCallbackFunc && (encodeCount % step == 0))
        //    progressCallbackFunc(double(encodeCount) / (validFrameCount > 0 ? validFrameCount : 100), progressCallbackData);
        if (encodeCount % step == 0)
            finishPercent.store(double(encodeCount) / (validFrameCount > 0 ? validFrameCount : 100) * 100);
    }

    writer.close();

    finishPercent.store(100);

    printf("total encode %d\n", encodeCount);
    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

CudaPanoramaLocalDiskTask::CudaPanoramaLocalDiskTask()
{
    ptrImpl.reset(new Impl);
}

CudaPanoramaLocalDiskTask::~CudaPanoramaLocalDiskTask()
{

}

bool CudaPanoramaLocalDiskTask::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
    const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight,
    int dstVideoBitRate, const std::string& dstVideoEncoder, const std::string& dstVideoPreset, 
    int dstVideoMaxFrameCount, ProgressCallbackFunction func, void* data)
{
    return ptrImpl->init(srcVideoFiles, offsets, audioIndex, cameraParamFile, dstVideoFile, dstWidth, dstHeight,
        dstVideoBitRate, dstVideoEncoder, dstVideoPreset, dstVideoMaxFrameCount, func, data);
}

bool CudaPanoramaLocalDiskTask::start()
{
    return ptrImpl->start();
}

void CudaPanoramaLocalDiskTask::waitForCompletion()
{
    ptrImpl->waitForCompletion();
}

int CudaPanoramaLocalDiskTask::getProgress() const
{
    return ptrImpl->getProgress();
}

void CudaPanoramaLocalDiskTask::cancel()
{
    ptrImpl->cancel();
}