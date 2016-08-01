#include "PanoramaTask.h"
#include "PanoramaTaskUtil.h"
#include "ConcurrentQueue.h"
#include "ZBlend.h"
#include "ZReproject.h"
#include "RicohUtil.h"
#include "PinnedMemoryPool.h"
#include "SharedAudioVideoFramePool.h"
#include "DOclPanoramaTaskUtil.h"
#include "Timer.h"
#include "Image.h"
#include "Text.h"
#include "opencv2/highgui.hpp"
#include <deque>

enum DOclEncodeState
{
    VideoFrameNotCome,
    FirstVideoFrameCome,
    ClearTempAudioBuffer
};

class PinnedMemoryPoolForDOcl
{
public:
    PinnedMemoryPoolForDOcl() :
        rows(0), cols(0), type(0), hasInit(0)
    {
    }

    bool init(int rows_, int cols_, int type_)
    {
        clear();

        std::lock_guard<std::mutex> lock(mtx);

        docl::HostMem test;
        try
        {
            test.create(rows_, cols_, type_);
        }
        catch (...)
        {
            return false;
        }
        if (!test.data)
            return false;

        rows = rows_;
        cols = cols_;
        type = type_;

        hasInit = 1;
        return true;
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mtx);
        pool.clear();
    }

    bool get(docl::HostMem& mem)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!hasInit)
        {
            mem = docl::HostMem();
            return false;
        }

        int size = pool.size();
        int index = -1;
        for (int i = 0; i < size; i++)
        {
            if (pool[i].mem && pool[i].smem.use_count() == 1)
            {
                index = i;
                break;
            }
        }
        if (index >= 0)
        {
            mem = pool[index];
            return true;
        }

        docl::HostMem newMem(rows, cols, type);
        if (!newMem.data)
        {
            mem = docl::HostMem();
            return false;
        }

        mem = newMem;
        pool.push_back(newMem);
        return true;
    }
private:
    int rows, cols, type;
    std::vector<docl::HostMem> pool;
    std::mutex mtx;
    int hasInit;
};

struct StampedPinnedMemoryVectorForDOcl
{
    std::vector<docl::HostMem> frames;
    std::vector<long long int> timeStamps;
    std::vector<int> frameIndexes;
};

typedef BoundedCompleteQueue<StampedPinnedMemoryVectorForDOcl> FrameVectorBufferForDOcl;
typedef BoundedCompleteQueue<DOclMixedAudioVideoFrame> MixedFrameBufferForDOcl;
typedef std::deque<DOclMixedAudioVideoFrame> TempAudioFrameBufferForDOcl;

struct DOclPanoramaLocalDiskTask::Impl
{
    Impl();
    ~Impl();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
        const std::string& cameraParamFile, const std::string& customMaskFile, const std::string& logoFile, int logoHFov,
        int highQualityBlend, const std::string& dstVideoFile, int dstWidth, int dstHeight,
        int dstVideoBitRate, const std::string& dstVideoEncoder, const std::string& dstVideoPreset,
        int dstVideoMaxFrameCount);
    bool start();
    void waitForCompletion();
    int getProgress() const;
    void cancel();

    void getLastSyncErrorMessage(std::string& message) const;
    bool hasAsyncErrorMessage() const;
    void getLastAsyncErrorMessage(std::string& message);

    void clear();

    int numVideos;
    int audioIndex;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader3> readers;
    DOclPanoramaRender render;
    PinnedMemoryPoolForDOcl srcFramesMemoryPool;
    AudioVideoFramePool audioFramesMemoryPool;
    FrameVectorBufferForDOcl decodeFramesBuffer;
    DOclHostMemVideoFrameMemoryPool dstFramesMemoryPool;
    MixedFrameBufferForDOcl procFrameBuffer;
    cv::Mat blendImageCpu;
    DOclWatermarkFilter watermarkFilter;
    std::unique_ptr<DOclLogoFilter> logoFilter;
    avp::AudioVideoWriter3 writer;
    int isLibX264;

    int decodeCount;
    int procCount;
    int encodeCount;
    std::atomic<int> finishPercent;
    int validFrameCount;

    void decode();
    void proc();
    void encode();
    std::unique_ptr<std::thread> decodeThread;
    std::unique_ptr<std::thread> procThread;
    std::unique_ptr<std::thread> encodeThread;

    std::string syncErrorMessage;
    std::mutex mtxAsyncErrorMessage;
    std::string asyncErrorMessage;
    int hasAsyncError;
    void setAsyncErrorMessage(const std::string& message);
    void clearAsyncErrorMessage();

    bool initSuccess;
    bool finish;
    bool isCanceled;
};

DOclPanoramaLocalDiskTask::Impl::Impl()
{
    clear();
}

DOclPanoramaLocalDiskTask::Impl::~Impl()
{
    clear();
}

bool DOclPanoramaLocalDiskTask::Impl::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
    int tryAudioIndex, const std::string& cameraParamFile, const std::string& customMaskFile,
    const std::string& logoFile, int logoHFov, int highQualityBlend, const std::string& dstVideoFile,
    int dstWidth, int dstHeight, int dstVideoBitRate, const std::string& dstVideoEncoder,
    const std::string& dstVideoPreset, int dstVideoMaxFrameCount)
{
    clear();

    if (srcVideoFiles.empty() || (srcVideoFiles.size() != offsets.size()))
    {
        ptlprintf("Error in %s, size of srcVideoFiles and size of offsets empty or unmatch.\n", __FUNCTION__);
        syncErrorMessage = getText(TI_PARAM_CHECK_FAIL);
        return false;
    }

    numVideos = srcVideoFiles.size();

    dstSize.width = dstWidth;
    dstSize.height = dstHeight;

    bool ok = false;

    ok = docl::init();
    if (!ok)
    {
        ptlprintf("Error in %s, opencl init failed\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    ok = prepareSrcVideos(srcVideoFiles, avp::PixelTypeBGR32, offsets, tryAudioIndex, readers, audioIndex, srcSize, validFrameCount);
    if (!ok)
    {
        ptlprintf("Error in %s, could not open video file(s)\n", __FUNCTION__);
        syncErrorMessage = getText(TI_OPEN_VIDEO_FAIL);
        return false;
    }

    if (dstVideoMaxFrameCount > 0 && validFrameCount > dstVideoMaxFrameCount)
        validFrameCount = dstVideoMaxFrameCount;

    ok = srcFramesMemoryPool.init(readers[0].getVideoHeight(), readers[0].getVideoWidth(), CV_8UC4);
    if (!ok)
    {
        ptlprintf("Error in %s, could not init memory pool for source video frames\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (audioIndex >= 0 && audioIndex < numVideos)
    {
        ok = audioFramesMemoryPool.initAsAudioFramePool(readers[audioIndex].getAudioSampleType(),
            readers[audioIndex].getAudioNumChannels(), readers[audioIndex].getAudioChannelLayout(),
            readers[audioIndex].getAudioNumSamples());
        if (!ok)
        {
            ptlprintf("Error in %s, could not init memory pool for audio frames\n", __FUNCTION__);
            syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
            return false;
        }
    }

    ok = render.prepare(cameraParamFile, highQualityBlend, srcSize, dstSize);
    if (!ok)
    {
        ptlprintf("Error in %s, render prepare failed\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (render.getNumImages() != numVideos)
    {
        ptlprintf("Error in %s, num images in render not equal to num videos\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    isLibX264 = dstVideoEncoder == "h264" ? 1 : 0;

    ok = dstFramesMemoryPool.init(isLibX264 ? avp::PixelTypeYUV420P : avp::PixelTypeNV12, dstSize.width, dstSize.height);
    if (!ok)
    {
        ptlprintf("Error in %s, could not init memory pool for dst video frames\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    ok = watermarkFilter.init(dstSize.width, dstSize.height);
    if (!ok)
    {
        ptlprintf("Error in %s, init watermark filter failed\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (!logoFile.empty() && logoHFov > 0)
    {
        logoFilter.reset(new DOclLogoFilter);
        ok = logoFilter->init(logoFile, logoHFov, dstSize.width, dstSize.height);
        if (!ok)
        {
            ptlprintf("Error in %s, init logo filter failed\n", __FUNCTION__);
            syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
            logoFilter.reset();
            return false;
        }
    }

    std::vector<avp::Option> options;
    options.push_back(std::make_pair("preset", dstVideoPreset));
    options.push_back(std::make_pair("bf", "0"));
    std::string format = (dstVideoEncoder == "h264_qsv" || dstVideoEncoder == "nvenc_h264") ? dstVideoEncoder : "h264";
    if (audioIndex >= 0 && audioIndex < numVideos)
    {
        ok = writer.open(dstVideoFile, "", true, true, "aac", readers[audioIndex].getAudioSampleType(),
            readers[audioIndex].getAudioChannelLayout(), readers[audioIndex].getAudioSampleRate(), 128000,
            true, format, isLibX264 ? avp::PixelTypeYUV420P : avp::PixelTypeNV12,
            dstSize.width, dstSize.height, readers[0].getVideoFrameRate(), dstVideoBitRate, options);
    }
    else
    {
        ok = writer.open(dstVideoFile, "", false, false, "", avp::SampleTypeUnknown, 0, 0, 0,
            true, format, isLibX264 ? avp::PixelTypeYUV420P : avp::PixelTypeNV12,
            dstSize.width, dstSize.height, readers[0].getVideoFrameRate(), dstVideoBitRate, options);
    }
    if (!ok)
    {
        ptlprintf("Error in %s, video writer open failed\n", __FUNCTION__);
        syncErrorMessage = getText(TI_CREATE_STITCH_VIDEO_FAIL);
        return false;
    }

    decodeFramesBuffer.setMaxSize(4);
    procFrameBuffer.setMaxSize(16);

    decodeFramesBuffer.resume();
    procFrameBuffer.resume();

    finishPercent.store(0);

    initSuccess = true;
    finish = false;
    return true;
}

bool DOclPanoramaLocalDiskTask::Impl::start()
{
    if (!initSuccess)
    {
        ptlprintf("Error in %s, init not success, could not start\n", __FUNCTION__);
        return false;
    }

    if (finish)
        return false;

    decodeThread.reset(new std::thread(&DOclPanoramaLocalDiskTask::Impl::decode, this));
    procThread.reset(new std::thread(&DOclPanoramaLocalDiskTask::Impl::proc, this));
    encodeThread.reset(new std::thread(&DOclPanoramaLocalDiskTask::Impl::encode, this));

    return true;
}

void DOclPanoramaLocalDiskTask::Impl::waitForCompletion()
{
    if (decodeThread && decodeThread->joinable())
        decodeThread->join();
    decodeThread.reset();
    if (procThread && procThread->joinable())
        procThread->join();
    procThread.reset(0);
    if (encodeThread && encodeThread->joinable())
        encodeThread->join();
    encodeThread.reset(0);

    srcFramesMemoryPool.clear();
    audioFramesMemoryPool.clear();
    dstFramesMemoryPool.clear();

    decodeFramesBuffer.clear();
    procFrameBuffer.clear();

    watermarkFilter.clear();
    logoFilter.reset();

    if (!finish)
        ptlprintf("Info in %s, write video finish\n", __FUNCTION__);

    finish = true;
}

int DOclPanoramaLocalDiskTask::Impl::getProgress() const
{
    return finishPercent.load();
}

void DOclPanoramaLocalDiskTask::Impl::getLastSyncErrorMessage(std::string& message) const
{
    message = syncErrorMessage;
}

bool DOclPanoramaLocalDiskTask::Impl::hasAsyncErrorMessage() const
{
    return hasAsyncError;
}

void DOclPanoramaLocalDiskTask::Impl::getLastAsyncErrorMessage(std::string& message)
{
    if (hasAsyncError)
    {
        std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
        message = asyncErrorMessage;
        hasAsyncError = 0;
    }
    else
        message.clear();
}

void DOclPanoramaLocalDiskTask::Impl::clear()
{
    if (decodeThread && decodeThread->joinable())
        decodeThread->join();
    decodeThread.reset(0);
    if (procThread && procThread->joinable())
        procThread->join();
    procThread.reset(0);
    if (encodeThread && encodeThread->joinable())
        encodeThread->join();
    encodeThread.reset(0);

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

    render.clear();

    watermarkFilter.clear();
    logoFilter.reset();

    decodeCount = 0;
    procCount = 0;
    encodeCount = 0;
    finishPercent.store(0);

    validFrameCount = 0;

    syncErrorMessage.clear();
    clearAsyncErrorMessage();

    initSuccess = false;
    finish = true;
    isCanceled = false;
}

void DOclPanoramaLocalDiskTask::Impl::cancel()
{
    isCanceled = true;
}

void DOclPanoramaLocalDiskTask::Impl::decode()
{
    size_t id = std::this_thread::get_id().hash();
    ptlprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    decodeCount = 0;
    int mediaType;
    while (true)
    {
        StampedPinnedMemoryVectorForDOcl videoFrames;
        avp::AudioVideoFrame2 audioFrame;
        unsigned char* data[4] = { 0 };
        int steps[4] = { 0 };

        videoFrames.frameIndexes.resize(numVideos);
        videoFrames.timeStamps.resize(numVideos);
        videoFrames.frames.resize(numVideos);

        if (audioIndex >= 0 && audioIndex < numVideos)
        {
            audioFramesMemoryPool.get(audioFrame);
            srcFramesMemoryPool.get(videoFrames.frames[audioIndex]);
            data[0] = videoFrames.frames[audioIndex].data;
            steps[0] = videoFrames.frames[audioIndex].step;
            avp::AudioVideoFrame2 videoFrame(data, steps, avp::PixelTypeBGR32, srcSize.width, srcSize.height, -1LL);
            if (!readers[audioIndex].readTo(audioFrame, videoFrame, mediaType))
                break;
            if (mediaType == avp::AUDIO)
            {
                procFrameBuffer.push(audioFrame);
                continue;
            }
            else if (mediaType == avp::VIDEO)
            {
                videoFrames.timeStamps[audioIndex] = videoFrame.timeStamp;
                videoFrames.frameIndexes[audioIndex] = videoFrame.frameIndex;
            }
            else
                break;
        }

        bool successRead = true;
        for (int i = 0; i < numVideos; i++)
        {
            if (i == audioIndex)
                continue;

            srcFramesMemoryPool.get(videoFrames.frames[i]);
            data[0] = videoFrames.frames[i].data;
            steps[0] = videoFrames.frames[i].step;
            avp::AudioVideoFrame2 videoFrame(data, steps, avp::PixelTypeBGR32, srcSize.width, srcSize.height, -1LL);
            if (!readers[i].readTo(audioFrame, videoFrame, mediaType))
            {
                successRead = false;
                break;
            }
            if (mediaType == avp::VIDEO)
            {
                videoFrames.timeStamps[i] = videoFrame.timeStamp;
                videoFrames.frameIndexes[i] = videoFrame.frameIndex;
            }
            else
            {
                successRead = false;
                break;
            }
        }
        if (!successRead || isCanceled)
            break;

        decodeFramesBuffer.push(videoFrames);
        decodeCount++;
        //ptlprintf("decode count = %d\n", decodeCount);

        if (decodeCount >= validFrameCount)
            break;
    }

    if (!isCanceled)
    {
        while (decodeFramesBuffer.size())
            std::this_thread::sleep_for(std::chrono::microseconds(25));
    }
    decodeFramesBuffer.stop();

    for (int i = 0; i < numVideos; i++)
        readers[i].close();

    ptlprintf("In %s, total decode %d\n", __FUNCTION__, decodeCount);
    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void DOclPanoramaLocalDiskTask::Impl::proc()
{
    size_t id = std::this_thread::get_id().hash();
    ptlprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    procCount = 0;
    StampedPinnedMemoryVectorForDOcl srcFrames;
    std::vector<docl::GpuMat> images(numVideos);
    docl::GpuMat bgr32;
    DOclMixedAudioVideoFrame videoFrame;
    docl::GpuMat y, u, v, uv;
    int index = audioIndex >= 0 ? audioIndex : 0;
    while (true)
    {
        if (!decodeFramesBuffer.pull(srcFrames))
            break;

        if (isCanceled)
            break;

        for (int i = 0; i < numVideos; i++)
        {
            srcFrames.frames[i].lock();
            images[i].upload(srcFrames.frames[i]);
            srcFrames.frames[i].unlock();
        }

        bool ok = render.render(images, bgr32);
        if (!ok)
        {
            ptlprintf("Error in %s, render failed\n", __FUNCTION__);
            setAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE));
            isCanceled = true;
            break;
        }

        if (logoFilter)
        {
            ok = logoFilter->addLogo(bgr32);
            if (!ok)
            {
                ptlprintf("Error in %s, add logo failed\n", __FUNCTION__);
                setAsyncErrorMessage(getText(TI_WRITE_TO_VIDEO_FAIL_TASK_TERMINATE));
                break;
            }
        }

        if (addWatermark)
        {
            ok = watermarkFilter.addWatermark(bgr32);
            if (!ok)
            {
                ptlprintf("Error in %s, add watermark failed\n", __FUNCTION__);
                setAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE));
                isCanceled = true;
                break;
            }
        }

        // IMPORTANT NOTICE!!!
        // I use cv::cuda::GpuMat::download to copy gpu memory to cpu memory.
        // If cpu memory is not page-locked, download will take quite a long time.
        // But in the following, cpu memory is page-locked, which costs just a little time.
        // NVIDIA's documentation does not mention that calling cudaMemcpy2D to copy
        // gpu memory to page-locked cpu memory costs less time than pageable memory.
        // Another implementation is to make the cpu memory as zero-copy,
        // then gpu color conversion writes result directly to cpu zero-copy memory.
        // If image size is too large, such writing costs a large amount of time.
        dstFramesMemoryPool.get(videoFrame);
        videoFrame.frame.timeStamp = srcFrames.timeStamps[index];
        if (isLibX264)
        {
            cvtBGR32ToYUV420P(bgr32, y, u, v);
            //cv::Mat yy = videoFrame.planes[0].createMatHeader();
            //cv::Mat uu = videoFrame.planes[1].createMatHeader();
            //cv::Mat vv = videoFrame.planes[2].createMatHeader();
            videoFrame.planes[0].lock();
            y.download(videoFrame.planes[0]);
            videoFrame.planes[0].unlock();
            videoFrame.planes[1].lock();
            u.download(videoFrame.planes[1]);
            videoFrame.planes[1].unlock();
            videoFrame.planes[2].lock();
            v.download(videoFrame.planes[2]);
            videoFrame.planes[2].unlock();
        }
        else
        {
            cvtBGR32ToNV12(bgr32, y, uv);
            //cv::Mat yy = videoFrame.planes[0].createMatHeader();
            //cv::Mat uvuv = videoFrame.planes[1].createMatHeader();
            videoFrame.planes[0].lock();
            y.download(videoFrame.planes[0]);
            videoFrame.planes[0].unlock();
            videoFrame.planes[1].lock();
            uv.download(videoFrame.planes[1]);
            videoFrame.planes[1].unlock();
        }

        procFrameBuffer.push(videoFrame);
        procCount++;
        //ptlprintf("proc count = %d\n", procCount);
    }

    if (!isCanceled)
    {
        while (procFrameBuffer.size())
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    procFrameBuffer.stop();

    ptlprintf("In %s, total proc %d\n", __FUNCTION__, procCount);
    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void DOclPanoramaLocalDiskTask::Impl::encode()
{
    size_t id = std::this_thread::get_id().hash();
    ptlprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    encodeCount = 0;
    int step = 1;
    if (validFrameCount > 0)
        step = validFrameCount / 100.0 + 0.5;
    if (step < 1)
        step = 1;
    ptlprintf("In %s, validFrameCount = %d, step = %d\n", __FUNCTION__, validFrameCount, step);
    ztool::Timer timerEncode;
    encodeCount = 0;
    DOclMixedAudioVideoFrame frame;
    int encodeState = VideoFrameNotCome;
    int hasAudio = audioIndex >= 0 && audioIndex < numVideos;
    TempAudioFrameBufferForDOcl tempAudioFrames;
    while (true)
    {
        if (!procFrameBuffer.pull(frame))
            break;

        if (isCanceled)
            break;

        if (hasAudio)
        {
            if (frame.frame.mediaType == avp::AUDIO)
            {
                if (encodeState == VideoFrameNotCome)
                {
                    tempAudioFrames.push_back(frame);
                    continue;
                }
                else if (encodeState == FirstVideoFrameCome)
                {
                    while (tempAudioFrames.size())
                    {
                        avp::AudioVideoFrame2 audioFrame = tempAudioFrames.front().frame;
                        writer.write(audioFrame);
                        tempAudioFrames.pop_front();
                    }
                    encodeState = ClearTempAudioBuffer;
                }
            }

            if (frame.frame.mediaType == avp::VIDEO && encodeState == VideoFrameNotCome)
                encodeState = FirstVideoFrameCome;
        }

        //timerEncode.start();
        bool ok = writer.write(frame.frame);
        //timerEncode.end();
        if (!ok)
        {
            ptlprintf("Error in %s, render failed\n", __FUNCTION__);
            setAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE));
            isCanceled = true;
            break;
        }

        // Only when the frame is of type video can we increase encodeCount
        if (frame.frame.mediaType == avp::VIDEO)
            encodeCount++;
        //ptlprintf("frame %d finish, encode time = %f\n", encodeCount, timerEncode.elapse());

        if (encodeCount % step == 0)
            finishPercent.store(double(encodeCount) / (validFrameCount > 0 ? validFrameCount : 100) * 100);
    }

    writer.close();

    finishPercent.store(100);

    ptlprintf("In %s, total encode %d\n", __FUNCTION__, encodeCount);
    ptlprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void DOclPanoramaLocalDiskTask::Impl::setAsyncErrorMessage(const std::string& message)
{
    std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
    hasAsyncError = 1;
    asyncErrorMessage = message;
}

void DOclPanoramaLocalDiskTask::Impl::clearAsyncErrorMessage()
{
    std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
    hasAsyncError = 0;
    asyncErrorMessage.clear();
}

DOclPanoramaLocalDiskTask::DOclPanoramaLocalDiskTask()
{
    ptrImpl.reset(new Impl);
}

DOclPanoramaLocalDiskTask::~DOclPanoramaLocalDiskTask()
{

}

bool DOclPanoramaLocalDiskTask::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
    const std::string& cameraParamFile, const std::string& customMaskFile, const std::string& logoFile, int logoHFov,
    int highQualityBlend, const std::string& dstVideoFile, int dstWidth, int dstHeight,
    int dstVideoBitRate, const std::string& dstVideoEncoder, const std::string& dstVideoPreset,
    int dstVideoMaxFrameCount)
{
    return ptrImpl->init(srcVideoFiles, offsets, audioIndex, cameraParamFile, customMaskFile,
        logoFile, logoHFov, highQualityBlend, dstVideoFile, dstWidth, dstHeight,
        dstVideoBitRate, dstVideoEncoder, dstVideoPreset, dstVideoMaxFrameCount);
}

bool DOclPanoramaLocalDiskTask::start()
{
    return ptrImpl->start();
}

void DOclPanoramaLocalDiskTask::waitForCompletion()
{
    ptrImpl->waitForCompletion();
}

int DOclPanoramaLocalDiskTask::getProgress() const
{
    return ptrImpl->getProgress();
}

void DOclPanoramaLocalDiskTask::cancel()
{
    ptrImpl->cancel();
}

void DOclPanoramaLocalDiskTask::getLastSyncErrorMessage(std::string& message) const
{
    ptrImpl->getLastSyncErrorMessage(message);
}

bool DOclPanoramaLocalDiskTask::hasAsyncErrorMessage() const
{
    return ptrImpl->hasAsyncErrorMessage();
}

void DOclPanoramaLocalDiskTask::getLastAsyncErrorMessage(std::string& message)
{
    return ptrImpl->getLastAsyncErrorMessage(message);
}