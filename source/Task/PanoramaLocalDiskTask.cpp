#include "PanoramaTask.h"
#include "PanoramaTaskUtil.h"
#include "ConcurrentQueue.h"
#include "RicohUtil.h"
#include "PinnedMemoryPool.h"
#include "SharedAudioVideoFramePool.h"
#include "CudaPanoramaTaskUtil.h"
#include "Image.h"
#include "Text.h"
#include "CompileControl.h"
#include "Blend/ZBlend.h"
#include "Warp/ZReproject.h"
#include "Tool/Timer.h"
#include "Tool/Print.h"
#include "opencv2/highgui.hpp"
#include <deque>

typedef BoundedCompleteQueue<avp::AudioVideoFrame2> FrameBufferForCpu;
typedef std::vector<avp::AudioVideoFrame2> FrameVectorForCpu;
typedef BoundedCompleteQueue<FrameVectorForCpu> FrameVectorBufferForCpu;
typedef std::deque<avp::AudioVideoFrame2> TempAudioFrameBufferForCpu;

enum EncodeState
{
    VideoFrameNotCome,
    FirstVideoFrameCome,
    ClearTempAudioBuffer
};

struct CPUPanoramaLocalDiskTask::Impl
{
    Impl();
    ~Impl();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
        int panoType, const std::string& cameraParamFile, const std::string& exposureWhiteBalanceFile, 
        const std::string& customMaskFile, const std::string& logoFile, int logoHFov,
        int highQualityBlend, const std::string& dstVideoFile, int dstWidth, int dstHeight, int dstVideoBitRate,
        const std::string& dstVideoEncoder, const std::string& dstVideoPreset, 
        int dstVideoMaxFrameCount);
    bool init(const std::string& configFile);
    bool start();
    void waitForCompletion();
    int getProgress() const;
    void cancel();

    void getLastSyncErrorMessage(std::string& message) const;
    bool hasAsyncErrorMessage() const;
    void getLastAsyncErrorMessage(std::string& message);

    void run();
    void clear();

    int numVideos;
    int audioIndex;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader3> readers;
    std::vector<std::vector<std::vector<unsigned char> > > luts;
    std::unique_ptr<CPUPanoramaRender> render;
    WatermarkFilter watermarkFilter;
    std::unique_ptr<LogoFilter> logoFilter;
    avp::AudioVideoWriter3 writer;

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

    AudioVideoFramePool audioFramesMemoryPool;
    AudioVideoFramePool srcVideoFramesMemoryPool;
    AudioVideoFramePool dstVideoFramesMemoryPool;

    FrameVectorBufferForCpu decodeFramesBuffer;
    FrameBufferForCpu procFrameBuffer;

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

CPUPanoramaLocalDiskTask::Impl::Impl()
{
    clear();
}

CPUPanoramaLocalDiskTask::Impl::~Impl()
{
    clear();
}

bool CPUPanoramaLocalDiskTask::Impl::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
    int tryAudioIndex, int panoType, const std::string& cameraParamFile, 
    const std::string& exposureWhiteBalanceFile, const std::string& customMaskFile,
    const std::string& logoFile, int logoHFov, int highQualityBlend, const std::string& dstVideoFile, 
    int dstWidth, int dstHeight,  int dstVideoBitRate, const std::string& dstVideoEncoder, 
    const std::string& dstVideoPreset, int dstVideoMaxFrameCount)
{
    ztool::lprintf("Info in %s, params: src video files num = %d, ", __FUNCTION__, srcVideoFiles.size());
    for (int i = 0; i < srcVideoFiles.size(); i++)
        ztool::lprintf("[%d] %s, ", i, srcVideoFiles[i].c_str());
    ztool::lprintf("offsets num = %d, ", offsets.size());
    for (int i = 0; i < offsets.size(); i++)
        ztool::lprintf("[%d] %d, ", i, offsets[i]);
    ztool::lprintf("try audio index = %d, ", tryAudioIndex);
    ztool::lprintf("pano type = %d(%s), ", panoType, getPanoStitchTypeString(panoType));
    ztool::lprintf("camera param file = %s, expo white balance file = %s, custom mask file = %s, "
        "logo file = %s, logo hfov = %d, high quality blend = %d, ",
        cameraParamFile.c_str(), exposureWhiteBalanceFile.c_str(), customMaskFile.c_str(), 
        logoFile.c_str(), logoHFov, highQualityBlend);
    ztool::lprintf("dst video file = %s, dst width = %d, dst height = %d, dst video bps = %d, ",
        dstVideoFile.c_str(), dstWidth, dstHeight, dstVideoBitRate);
    ztool::lprintf("dst video encoder = %s, dst video preset = %s, dst video max frame count = %d\n",
        dstVideoEncoder.c_str(), dstVideoPreset.c_str(), dstVideoMaxFrameCount);

    clear();

    if (srcVideoFiles.empty() || (srcVideoFiles.size() != offsets.size()))
    {
        ztool::lprintf("Error in %s, size of srcVideoFiles and size of offsets empty or unmatch.\n", __FUNCTION__);
        syncErrorMessage = getText(TI_PARAM_CHECK_FAIL);
        return false;
    }

    numVideos = srcVideoFiles.size();

    dstSize.width = dstWidth;
    dstSize.height = dstHeight;

    bool ok = false;
    ok = prepareSrcVideos(srcVideoFiles, avp::PixelTypeBGR24, offsets, tryAudioIndex, readers, audioIndex, srcSize, validFrameCount);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not open video file(s)\n", __FUNCTION__);
        syncErrorMessage = getText(TI_OPEN_VIDEO_FAIL);
        return false;
    }

    if (dstVideoMaxFrameCount > 0 && validFrameCount > dstVideoMaxFrameCount)
        validFrameCount = dstVideoMaxFrameCount;

    ok = srcVideoFramesMemoryPool.initAsVideoFramePool(avp::PixelTypeBGR24, readers[0].getVideoWidth(), readers[0].getVideoHeight());
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not init memory pool for source video frames\n", __FUNCTION__);
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
            ztool::lprintf("Error in %s, could not init memory pool for audio frames\n", __FUNCTION__);
            syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
            return false;
        }
    }

    ok = dstVideoFramesMemoryPool.initAsVideoFramePool(avp::PixelTypeBGR24, dstSize.width, dstSize.height);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not init memory pool for dst video frames\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (!exposureWhiteBalanceFile.empty())
    {
        std::vector<double> es, rs, bs;
        ok = loadExposureWhiteBalance(exposureWhiteBalanceFile, es, rs, bs);
        if (!ok)
        {
            ztool::lprintf("Error in %s, could not load exposure white balance file\n", __FUNCTION__);
            syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
            return false;
        }
        
        if (es.size() != numVideos || rs.size() != numVideos || bs.size() != numVideos)
        {
            ztool::lprintf("Error in %s, exposure and white balance param size unsatisfied, %d, %d, %d, should be %d\n", 
                __FUNCTION__, es.size(), rs.size(), bs.size(), numVideos);
            syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
            return false;
        }

        if (needCorrectExposureWhiteBalance(es, rs, bs))
            getExposureColorOptimizeLUTs(es, rs, bs, luts);
    }

    if (panoType == PanoStitchTypeMISO)
        render.reset(new CPUPanoramaRender);
    else if (panoType == PanoStitchTypeRicoh)
        render.reset(new CPURicohPanoramaRender);
    else
    {
        ztool::lprintf("Error in %s, unsupported pano stitch type %d, should be %d or %d\n",
            __FUNCTION__, panoType, PanoStitchTypeMISO, PanoStitchTypeRicoh);
    }

    ok = render->prepare(cameraParamFile, highQualityBlend, srcSize, dstSize);
    if (!ok)
    {
        ztool::lprintf("Error in %s, render prepare failed\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (render->getNumImages() != readers.size())
    {
        ztool::lprintf("Error in %s, num images in render not equal to num videos\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    //useCustomMasks = 0;
    //if (customMaskFile.size())
    //{
    //    std::vector<std::vector<IntervaledContour> > contours;
    //    ok = loadIntervaledContours(customMaskFile, contours);
    //    if (!ok)
    //    {
    //        ztool::lprintf("Error in %s, load custom masks failed\n", __FUNCTION__);
    //        syncErrorMessage = getText(TI_STITCH_INIT_FAIL)/*"初始化拼接失败。"*/;
    //        return false;
    //    }
    //    // Notice !! 
    //    // For new implementation of loadIntervaledContours, the two level vectors
    //    // are num intervals x num videos in each interval, instead of
    //    // num videos x num intervals of each video,
    //    // so the following if condition should be deleted
    //    //if (contours.size() != numVideos)
    //    //{
    //    //    ztool::lprintf("Error in %s, loaded contours.size() != numVideos\n", __FUNCTION__);
    //    //    syncErrorMessage = getText(TI_STITCH_INIT_FAIL)/*"初始化拼接失败。"*/;
    //    //    return false;
    //    //}
    //    if (!cvtContoursToMasks(contours, dstMasks, customMasks))
    //    {
    //        ztool::lprintf("Error in %s, convert contours to customMasks failed\n", __FUNCTION__);
    //        syncErrorMessage = getText(TI_STITCH_INIT_FAIL)/*"初始化拼接失败。"*/;
    //        return false;
    //    }
    //    blender.getUniqueMasks(dstUniqueMasks);
    //    useCustomMasks = 1;
    //}

    ok = watermarkFilter.init(dstSize.width, dstSize.height, CV_8UC3);
    if (!ok)
    {
        ztool::lprintf("Error in %s, init watermark filter failed\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (!logoFile.empty() && logoHFov > 0)
    {
        logoFilter.reset(new LogoFilter);
        ok = logoFilter->init(logoFile, logoHFov, dstSize.width, dstSize.height);
        if (!ok)
        {
            ztool::lprintf("Error in %s, init logo filter failed\n", __FUNCTION__);
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
        ok = writer.open(dstVideoFile, "", true, 
            true, "aac", readers[audioIndex].getAudioSampleType(), readers[audioIndex].getAudioChannelLayout(), 
            readers[audioIndex].getAudioSampleRate(), 128000,
            true, format, avp::PixelTypeBGR24, dstSize.width, dstSize.height, readers[0].getVideoFrameRate(), dstVideoBitRate, options);
    }
    else
    {
        ok = writer.open(dstVideoFile, "", false, false, "", avp::SampleTypeUnknown, 0, 0, 0,
            true, format, avp::PixelTypeBGR24, dstSize.width, dstSize.height, readers[0].getVideoFrameRate(), dstVideoBitRate, options);
    }
    if (!ok)
    {
        ztool::lprintf("Error in %s, video writer open failed\n", __FUNCTION__);
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

bool CPUPanoramaLocalDiskTask::Impl::init(const std::string& configFile)
{
    std::vector<std::string> srcVideoFiles;
    std::vector<int> offsets;
    int tryAudioIndex;
    int panoType;
    std::string logoFile;
    int logoHFov;
    int highQualityBlend;
    std::string dstVideoFile;
    int dstWidth;
    int dstHeight;
    int dstVideoBitRate;
    std::string dstVideoEncoder;
    std::string dstVideoPreset;
    int dstVideoMaxFrameCount;

    bool ok = false;
    ok = loadVideoFileNamesAndOffset(configFile, srcVideoFiles, offsets);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not load video file names and offsets\n", __FUNCTION__);
        syncErrorMessage = getText(TI_CONFIG_FILE_PARSE_FAIL);
        return false;
    }
    ok = loadOutputConfig(configFile, tryAudioIndex, panoType, logoFile, logoHFov,
        highQualityBlend, dstVideoFile, dstWidth, dstHeight, dstVideoBitRate,
        dstVideoEncoder, dstVideoPreset, dstVideoMaxFrameCount);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not load output video params\n", __FUNCTION__);
        syncErrorMessage = getText(TI_CONFIG_FILE_PARSE_FAIL);
        return false;
    }

    return init(srcVideoFiles, offsets, tryAudioIndex, panoType, configFile, configFile, configFile,
        logoFile, logoHFov, highQualityBlend, dstVideoFile, dstWidth, dstHeight, dstVideoBitRate,
        dstVideoEncoder, dstVideoPreset, dstVideoMaxFrameCount);
}

void CPUPanoramaLocalDiskTask::Impl::decode()
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    decodeCount = 0;
    int mediaType;
    while (true)
    {
        FrameVectorForCpu videoFrames(numVideos);
        avp::AudioVideoFrame2 audioFrame;
        unsigned char* data[4] = { 0 };
        int steps[4] = { 0 };

        if (audioIndex >= 0 && audioIndex < numVideos)
        {
            audioFramesMemoryPool.get(audioFrame);
            srcVideoFramesMemoryPool.get(videoFrames[audioIndex]);
            if (!readers[audioIndex].readTo(audioFrame, videoFrames[audioIndex], mediaType))
                break;
            if (mediaType == avp::AUDIO)
            {
                procFrameBuffer.push(audioFrame);
                continue;
            }
            else if (mediaType == avp::VIDEO)
            {
                
            }
            else
                break;
        }

        bool successRead = true;
        for (int i = 0; i < numVideos; i++)
        {
            if (i == audioIndex)
                continue;

            srcVideoFramesMemoryPool.get(videoFrames[i]);
            if (!readers[i].readTo(audioFrame, videoFrames[i], mediaType))
            {
                successRead = false;
                break;
            }
            if (mediaType == avp::VIDEO)
            {
                
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
        //ztool::lprintf("decode count = %d\n", decodeCount);

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

    ztool::lprintf("In %s, total decode %d\n", __FUNCTION__, decodeCount);
    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void CPUPanoramaLocalDiskTask::Impl::proc()
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    procCount = 0;
    FrameVectorForCpu frames;
    std::vector<cv::Mat> images(numVideos);
    int index = audioIndex >= 0 ? audioIndex : 0;
    bool ok = false;
    while (true)
    {
        if (!decodeFramesBuffer.pull(frames))
            break;

        if (isCanceled)
            break;

        for (int i = 0; i < numVideos; i++)
            images[i] = cv::Mat(frames[i].height, frames[i].width, CV_8UC3, frames[i].data[0], frames[i].steps[0]);
        //reprojectParallelTo16S(images, reprojImages, dstSrcMaps);

        avp::AudioVideoFrame2 videoFrame;
        dstVideoFramesMemoryPool.get(videoFrame);
        cv::Mat blendImage(videoFrame.height, videoFrame.width, CV_8UC3, videoFrame.data[0], videoFrame.steps[0]);

        if (luts.empty())
            render->render(images, blendImage);
        else
            render->render(images, blendImage, luts);

        //if (useCustomMasks)
        //{
        //    bool custom = false;
        //    currMasks.resize(numVideos);
        //    for (int i = 0; i < numVideos; i++)
        //    {
        //        if (customMasks[i].getMask2(frames[i].frameIndex, currMasks[i]))
        //            custom = true;
        //        else
        //            currMasks[i] = dstUniqueMasks[i];
        //    }

        //    if (custom)
        //    {
        //        //printf("custom masks\n");
        //        blender.blend(reprojImages, currMasks, blendImage);
        //    }
        //    else
        //        blender.blend(reprojImages, blendImage);
        //}
        //else
        //    blender.blend(reprojImages, blendImage);

        if (logoFilter)
        {
            ok = logoFilter->addLogo(blendImage);
            if (!ok)
            {
                ztool::lprintf("Error in %s, add logo fail\n", __FUNCTION__);
                setAsyncErrorMessage(getText(TI_WRITE_TO_VIDEO_FAIL_TASK_TERMINATE));
                break;
            }
        }

        if (addWatermark)
        {
            ok = watermarkFilter.addWatermark(blendImage);
            if (!ok)
            {
                ztool::lprintf("Error in %s, add watermark fail\n", __FUNCTION__);
                setAsyncErrorMessage(getText(TI_WRITE_TO_VIDEO_FAIL_TASK_TERMINATE));
                break;
            }
        }

        videoFrame.timeStamp = frames[index].timeStamp;
        procFrameBuffer.push(videoFrame);
        procCount++;
        //ztool::lprintf("proc count = %d\n", procCount);
    }

    if (!isCanceled)
    {
        while (procFrameBuffer.size())
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    procFrameBuffer.stop();

    ztool::lprintf("In %s, total proc %d\n", __FUNCTION__, procCount);
    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void CPUPanoramaLocalDiskTask::Impl::encode()
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    encodeCount = 0;
    int step = 1;
    if (validFrameCount > 0)
        step = validFrameCount / 100.0 + 0.5;
    if (step < 1)
        step = 1;
    ztool::lprintf("In %s, validFrameCount = %d, step = %d\n", __FUNCTION__, validFrameCount, step);
    ztool::Timer timerEncode;
    encodeCount = 0;
    avp::AudioVideoFrame2 frame;
    int encodeState = VideoFrameNotCome;
    int hasAudio = audioIndex >= 0 && audioIndex < numVideos;
    TempAudioFrameBufferForCpu tempAudioFrames;
    while (true)
    {
        if (!procFrameBuffer.pull(frame))
            break;

        if (isCanceled)
            break;

        if (hasAudio)
        {
            if (frame.mediaType == avp::AUDIO)
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
                        avp::AudioVideoFrame2 audioFrame = tempAudioFrames.front();
                        writer.write(audioFrame);
                        tempAudioFrames.pop_front();
                    }
                    encodeState = ClearTempAudioBuffer;
                }
            }

            if (frame.mediaType == avp::VIDEO && encodeState == VideoFrameNotCome)
                encodeState = FirstVideoFrameCome;
        }

        //timerEncode.start();
        bool ok = writer.write(frame);
        //timerEncode.end();
        if (!ok)
        {
            ztool::lprintf("Error in %s, render failed\n", __FUNCTION__);
            setAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE));
            isCanceled = true;
            break;
        }

        // Only when the frame is of type video can we increase encodeCount
        if (frame.mediaType == avp::VIDEO)
            encodeCount++;
        //ztool::lprintf("frame %d finish, encode time = %f\n", encodeCount, timerEncode.elapse());

        if (encodeCount % step == 0)
            finishPercent.store(double(encodeCount) / (validFrameCount > 0 ? validFrameCount : 100) * 100);
    }

    writer.close();

    finishPercent.store(100);

    ztool::lprintf("In %s, total encode %d\n", __FUNCTION__, encodeCount);
    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

bool CPUPanoramaLocalDiskTask::Impl::start()
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not start\n", __FUNCTION__);
        return false;
    }

    if (finish)
        return false;

    decodeThread.reset(new std::thread(&CPUPanoramaLocalDiskTask::Impl::decode, this));
    procThread.reset(new std::thread(&CPUPanoramaLocalDiskTask::Impl::proc, this));
    encodeThread.reset(new std::thread(&CPUPanoramaLocalDiskTask::Impl::encode, this));
    return true;
}

void CPUPanoramaLocalDiskTask::Impl::waitForCompletion()
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

    audioFramesMemoryPool.clear();
    srcVideoFramesMemoryPool.clear();
    dstVideoFramesMemoryPool.clear();

    decodeFramesBuffer.clear();
    procFrameBuffer.clear();

    watermarkFilter.clear();
    logoFilter.reset();

    if (!finish)
        ztool::lprintf("Info in %s, write video finish\n", __FUNCTION__);

    finish = true;
}

int CPUPanoramaLocalDiskTask::Impl::getProgress() const
{
    return finishPercent.load();
}

void CPUPanoramaLocalDiskTask::Impl::cancel()
{
    isCanceled = true;
}

void CPUPanoramaLocalDiskTask::Impl::getLastSyncErrorMessage(std::string& message) const
{
    message = syncErrorMessage;
}

bool CPUPanoramaLocalDiskTask::Impl::hasAsyncErrorMessage() const
{
    return hasAsyncError;
}

void CPUPanoramaLocalDiskTask::Impl::getLastAsyncErrorMessage(std::string& message)
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

void CPUPanoramaLocalDiskTask::Impl::clear()
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

    audioFramesMemoryPool.clear();
    srcVideoFramesMemoryPool.clear();
    dstVideoFramesMemoryPool.clear();

    decodeFramesBuffer.clear();
    procFrameBuffer.clear();

    watermarkFilter.clear();
    logoFilter.reset();

    numVideos = 0;
    srcSize = cv::Size();
    dstSize = cv::Size();
    readers.clear();
    luts.clear();
    render.reset();
    writer.close();
    isCanceled = false;

    finishPercent.store(0);

    validFrameCount = 0;

    syncErrorMessage.clear();
    clearAsyncErrorMessage();

    initSuccess = false;
    finish = true;
}

void CPUPanoramaLocalDiskTask::Impl::setAsyncErrorMessage(const std::string& message)
{
    std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
    hasAsyncError = 1;
    asyncErrorMessage = message;
}

void CPUPanoramaLocalDiskTask::Impl::clearAsyncErrorMessage()
{
    std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
    hasAsyncError = 0;
    asyncErrorMessage.clear();
}

CPUPanoramaLocalDiskTask::CPUPanoramaLocalDiskTask()
{
    ptrImpl.reset(new Impl);
}

CPUPanoramaLocalDiskTask::~CPUPanoramaLocalDiskTask()
{

}

bool CPUPanoramaLocalDiskTask::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
    int panoType, const std::string& cameraParamFile, const std::string& exposureWhiteBalanceFile,
    const std::string& customMaskFile, const std::string& logoFile, int logoHFov, 
    int highQualityBlend, const std::string& dstVideoFile, int dstWidth, int dstHeight,
    int dstVideoBitRate, const std::string& dstVideoEncoder, const std::string& dstVideoPreset, 
    int dstVideoMaxFrameCount)
{
    return ptrImpl->init(srcVideoFiles, offsets, audioIndex, panoType, 
        cameraParamFile, exposureWhiteBalanceFile, customMaskFile, 
        logoFile, logoHFov, highQualityBlend, dstVideoFile, dstWidth, dstHeight,
        dstVideoBitRate, dstVideoEncoder, dstVideoPreset, dstVideoMaxFrameCount);
}

bool CPUPanoramaLocalDiskTask::init(const std::string& configFile)
{
    return ptrImpl->init(configFile);
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

void CPUPanoramaLocalDiskTask::getLastSyncErrorMessage(std::string& message) const
{
    ptrImpl->getLastSyncErrorMessage(message);
}

bool CPUPanoramaLocalDiskTask::hasAsyncErrorMessage() const
{
    return ptrImpl->hasAsyncErrorMessage();
}

void CPUPanoramaLocalDiskTask::getLastAsyncErrorMessage(std::string& message)
{
    return ptrImpl->getLastAsyncErrorMessage(message);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if COMPILE_INTEGRATED_OPENCL
struct IOclPanoramaLocalDiskTask::Impl
{
    Impl();
    ~Impl();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
        const std::string& cameraParamFile, const std::string& customMaskFile, const std::string& logoFile, int logoHFov,
        int highQualityBlend, const std::string& dstVideoFile, int dstWidth, int dstHeight, int dstVideoBitRate,
        const std::string& dstVideoEncoder, const std::string& dstVideoPreset,
        int dstVideoMaxFrameCount);
    bool start();
    void waitForCompletion();
    int getProgress() const;
    void cancel();

    void getLastSyncErrorMessage(std::string& message) const;
    bool hasAsyncErrorMessage() const;
    void getLastAsyncErrorMessage(std::string& message);

    void run();
    void clear();

    int numVideos;
    int audioIndex;
    cv::Size srcSize, dstSize;
    std::vector<avp::AudioVideoReader3> readers;
    IOclPanoramaRender render;
    WatermarkFilter watermarkFilter;
    std::unique_ptr<LogoFilter> logoFilter;
    avp::AudioVideoWriter3 writer;

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

    AudioVideoFramePool audioFramesMemoryPool;
    AudioVideoFramePool srcVideoFramesMemoryPool;
    AudioVideoFramePool dstVideoFramesMemoryPool;

    FrameVectorBufferForCpu decodeFramesBuffer;
    FrameBufferForCpu procFrameBuffer;

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

IOclPanoramaLocalDiskTask::Impl::Impl()
{
    clear();
}

IOclPanoramaLocalDiskTask::Impl::~Impl()
{
    clear();
}

bool IOclPanoramaLocalDiskTask::Impl::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
    int tryAudioIndex, const std::string& cameraParamFile, const std::string& customMaskFile,
    const std::string& logoFile, int logoHFov, int highQualityBlend, const std::string& dstVideoFile,
    int dstWidth, int dstHeight, int dstVideoBitRate, const std::string& dstVideoEncoder,
    const std::string& dstVideoPreset, int dstVideoMaxFrameCount)
{
    ztool::lprintf("Info in %s, params: src video files num = %d, ", __FUNCTION__, srcVideoFiles.size());
    for (int i = 0; i < srcVideoFiles.size(); i++)
        ztool::lprintf("[%d] %s, ", i, srcVideoFiles[i].c_str());
    ztool::lprintf("offsets num = %d, ", offsets.size());
    for (int i = 0; i < offsets.size(); i++)
        ztool::lprintf("[%d] %d, ", i, offsets[i]);
    ztool::lprintf("try audio index = %d, ", tryAudioIndex);
    ztool::lprintf("camera param file = %s, custom mask file = %s, logo file = %s, logo hfov = %d, high quality blend = %d, ",
        cameraParamFile.c_str(), customMaskFile.c_str(), logoFile.c_str(), logoHFov, highQualityBlend);
    ztool::lprintf("dst video file = %s, dst width = %d, dst height = %d, dst video bps = %d, ",
        dstVideoFile.c_str(), dstWidth, dstHeight, dstVideoBitRate);
    ztool::lprintf("dst video encoder = %s, dst video preset = %s, dst video max frame count = %d\n",
        dstVideoEncoder.c_str(), dstVideoPreset.c_str(), dstVideoMaxFrameCount);

    clear();

    if (srcVideoFiles.empty() || (srcVideoFiles.size() != offsets.size()))
    {
        ztool::lprintf("Error in %s, size of srcVideoFiles and size of offsets empty or unmatch.\n", __FUNCTION__);
        syncErrorMessage = getText(TI_PARAM_CHECK_FAIL);
        return false;
    }

    numVideos = srcVideoFiles.size();

    dstSize.width = dstWidth;
    dstSize.height = dstHeight;

    bool ok = false;
    ok = prepareSrcVideos(srcVideoFiles, avp::PixelTypeBGR32, offsets, tryAudioIndex, readers, audioIndex, srcSize, validFrameCount);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not open video file(s)\n", __FUNCTION__);
        syncErrorMessage = getText(TI_OPEN_VIDEO_FAIL);
        return false;
    }

    if (dstVideoMaxFrameCount > 0 && validFrameCount > dstVideoMaxFrameCount)
        validFrameCount = dstVideoMaxFrameCount;

    ok = srcVideoFramesMemoryPool.initAsVideoFramePool(avp::PixelTypeBGR32, readers[0].getVideoWidth(), readers[0].getVideoHeight());
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not init memory pool for source video frames\n", __FUNCTION__);
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
            ztool::lprintf("Error in %s, could not init memory pool for audio frames\n", __FUNCTION__);
            syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
            return false;
        }
    }

    ok = dstVideoFramesMemoryPool.initAsVideoFramePool(avp::PixelTypeBGR32, dstSize.width, dstSize.height);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not init memory pool for dst video frames\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    ok = render.prepare(cameraParamFile, highQualityBlend, srcSize, dstSize);
    if (!ok)
    {
        ztool::lprintf("Error in %s, render prepare failed\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (render.getNumImages() != readers.size())
    {
        ztool::lprintf("Error in %s, num images in render not equal to num videos\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    ok = watermarkFilter.init(dstSize.width, dstSize.height, CV_8UC4);
    if (!ok)
    {
        ztool::lprintf("Error in %s, init watermark filter failed\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (!logoFile.empty() && logoHFov > 0)
    {
        logoFilter.reset(new LogoFilter);
        ok = logoFilter->init(logoFile, logoHFov, dstSize.width, dstSize.height);
        if (!ok)
        {
            ztool::lprintf("Error in %s, init logo filter failed\n", __FUNCTION__);
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
        ok = writer.open(dstVideoFile, "", true,
            true, "aac", readers[audioIndex].getAudioSampleType(), readers[audioIndex].getAudioChannelLayout(),
            readers[audioIndex].getAudioSampleRate(), 128000,
            true, format, avp::PixelTypeBGR32, dstSize.width, dstSize.height, readers[0].getVideoFrameRate(), dstVideoBitRate, options);
    }
    else
    {
        ok = writer.open(dstVideoFile, "", false, false, "", avp::SampleTypeUnknown, 0, 0, 0,
            true, format, avp::PixelTypeBGR32, dstSize.width, dstSize.height, readers[0].getVideoFrameRate(), dstVideoBitRate, options);
    }
    if (!ok)
    {
        ztool::lprintf("Error in %s, video writer open failed\n", __FUNCTION__);
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

void IOclPanoramaLocalDiskTask::Impl::decode()
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    decodeCount = 0;
    int mediaType;
    while (true)
    {
        FrameVectorForCpu videoFrames(numVideos);
        avp::AudioVideoFrame2 audioFrame;
        unsigned char* data[4] = { 0 };
        int steps[4] = { 0 };

        if (audioIndex >= 0 && audioIndex < numVideos)
        {
            audioFramesMemoryPool.get(audioFrame);
            srcVideoFramesMemoryPool.get(videoFrames[audioIndex]);
            if (!readers[audioIndex].readTo(audioFrame, videoFrames[audioIndex], mediaType))
                break;
            if (mediaType == avp::AUDIO)
            {
                procFrameBuffer.push(audioFrame);
                continue;
            }
            else if (mediaType == avp::VIDEO)
            {

            }
            else
                break;
        }

        bool successRead = true;
        for (int i = 0; i < numVideos; i++)
        {
            if (i == audioIndex)
                continue;

            srcVideoFramesMemoryPool.get(videoFrames[i]);
            if (!readers[i].readTo(audioFrame, videoFrames[i], mediaType))
            {
                successRead = false;
                break;
            }
            if (mediaType == avp::VIDEO)
            {

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
        //ztool::lprintf("decode count = %d\n", decodeCount);

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

    ztool::lprintf("In %s, total decode %d\n", __FUNCTION__, decodeCount);
    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

#include "RunTimeObjects.h"
void IOclPanoramaLocalDiskTask::Impl::proc()
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    procCount = 0;
    FrameVectorForCpu frames;
    std::vector<iocl::UMat> images(numVideos);
    int index = audioIndex >= 0 ? audioIndex : 0;
    bool ok = false;
    while (true)
    {
        if (!decodeFramesBuffer.pull(frames))
            break;

        if (isCanceled)
            break;

        for (int i = 0; i < numVideos; i++)
            images[i] = iocl::UMat(frames[i].height, frames[i].width, CV_8UC4, frames[i].data[0], frames[i].steps[0]);
        //reprojectParallelTo16S(images, reprojImages, dstSrcMaps);

        avp::AudioVideoFrame2 videoFrame;
        dstVideoFramesMemoryPool.get(videoFrame);
        iocl::UMat blendImageHeader(videoFrame.height, videoFrame.width, CV_8UC4, videoFrame.data[0], videoFrame.steps[0]);
        cv::Mat blendImage(videoFrame.height, videoFrame.width, CV_8UC4, videoFrame.data[0], videoFrame.steps[0]);
        render.render(images, blendImageHeader);

        if (logoFilter)
        {
            ok = logoFilter->addLogo(blendImage);
            if (!ok)
            {
                ztool::lprintf("Error in %s, add logo fail\n", __FUNCTION__);
                setAsyncErrorMessage(getText(TI_WRITE_TO_VIDEO_FAIL_TASK_TERMINATE));
                break;
            }
        }

        if (addWatermark)
        {
            ok = watermarkFilter.addWatermark(blendImage);
            if (!ok)
            {
                ztool::lprintf("Error in %s, add watermark fail\n", __FUNCTION__);
                setAsyncErrorMessage(getText(TI_WRITE_TO_VIDEO_FAIL_TASK_TERMINATE));
                break;
            }
        }

        videoFrame.timeStamp = frames[index].timeStamp;
        procFrameBuffer.push(videoFrame);
        procCount++;
        //ztool::lprintf("proc count = %d\n", procCount);
    }

    if (!isCanceled)
    {
        while (procFrameBuffer.size())
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    procFrameBuffer.stop();

    ztool::lprintf("In %s, total proc %d\n", __FUNCTION__, procCount);
    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void IOclPanoramaLocalDiskTask::Impl::encode()
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    encodeCount = 0;
    int step = 1;
    if (validFrameCount > 0)
        step = validFrameCount / 100.0 + 0.5;
    if (step < 1)
        step = 1;
    ztool::lprintf("In %s, validFrameCount = %d, step = %d\n", __FUNCTION__, validFrameCount, step);
    ztool::Timer timerEncode;
    encodeCount = 0;
    avp::AudioVideoFrame2 frame;
    int encodeState = VideoFrameNotCome;
    int hasAudio = audioIndex >= 0 && audioIndex < numVideos;
    TempAudioFrameBufferForCpu tempAudioFrames;
    while (true)
    {
        if (!procFrameBuffer.pull(frame))
            break;

        if (isCanceled)
            break;

        if (hasAudio)
        {
            if (frame.mediaType == avp::AUDIO)
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
                        avp::AudioVideoFrame2 audioFrame = tempAudioFrames.front();
                        writer.write(audioFrame);
                        tempAudioFrames.pop_front();
                    }
                    encodeState = ClearTempAudioBuffer;
                }
            }

            if (frame.mediaType == avp::VIDEO && encodeState == VideoFrameNotCome)
                encodeState = FirstVideoFrameCome;
        }

        //timerEncode.start();
        bool ok = writer.write(frame);
        //timerEncode.end();
        if (!ok)
        {
            ztool::lprintf("Error in %s, render failed\n", __FUNCTION__);
            setAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE));
            isCanceled = true;
            break;
        }

        // Only when the frame is of type video can we increase encodeCount
        if (frame.mediaType == avp::VIDEO)
            encodeCount++;
        //ztool::lprintf("frame %d finish, encode time = %f\n", encodeCount, timerEncode.elapse());

        if (encodeCount % step == 0)
            finishPercent.store(double(encodeCount) / (validFrameCount > 0 ? validFrameCount : 100) * 100);
    }

    writer.close();

    finishPercent.store(100);

    ztool::lprintf("In %s, total encode %d\n", __FUNCTION__, encodeCount);
    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

bool IOclPanoramaLocalDiskTask::Impl::start()
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not start\n", __FUNCTION__);
        return false;
    }

    if (finish)
        return false;

    decodeThread.reset(new std::thread(&IOclPanoramaLocalDiskTask::Impl::decode, this));
    procThread.reset(new std::thread(&IOclPanoramaLocalDiskTask::Impl::proc, this));
    encodeThread.reset(new std::thread(&IOclPanoramaLocalDiskTask::Impl::encode, this));
    return true;
}

void IOclPanoramaLocalDiskTask::Impl::waitForCompletion()
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

    audioFramesMemoryPool.clear();
    srcVideoFramesMemoryPool.clear();
    dstVideoFramesMemoryPool.clear();

    decodeFramesBuffer.clear();
    procFrameBuffer.clear();

    watermarkFilter.clear();
    logoFilter.reset();

    if (!finish)
        ztool::lprintf("Info in %s, write video finish\n", __FUNCTION__);

    finish = true;
}

int IOclPanoramaLocalDiskTask::Impl::getProgress() const
{
    return finishPercent.load();
}

void IOclPanoramaLocalDiskTask::Impl::cancel()
{
    isCanceled = true;
}

void IOclPanoramaLocalDiskTask::Impl::getLastSyncErrorMessage(std::string& message) const
{
    message = syncErrorMessage;
}

bool IOclPanoramaLocalDiskTask::Impl::hasAsyncErrorMessage() const
{
    return hasAsyncError;
}

void IOclPanoramaLocalDiskTask::Impl::getLastAsyncErrorMessage(std::string& message)
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

void IOclPanoramaLocalDiskTask::Impl::clear()
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

    audioFramesMemoryPool.clear();
    srcVideoFramesMemoryPool.clear();
    dstVideoFramesMemoryPool.clear();

    decodeFramesBuffer.clear();
    procFrameBuffer.clear();

    watermarkFilter.clear();
    logoFilter.reset();

    numVideos = 0;
    srcSize = cv::Size();
    dstSize = cv::Size();
    readers.clear();
    render.clear();
    writer.close();
    isCanceled = false;

    finishPercent.store(0);

    validFrameCount = 0;

    syncErrorMessage.clear();
    clearAsyncErrorMessage();

    initSuccess = false;
    finish = true;
}

void IOclPanoramaLocalDiskTask::Impl::setAsyncErrorMessage(const std::string& message)
{
    std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
    hasAsyncError = 1;
    asyncErrorMessage = message;
}

void IOclPanoramaLocalDiskTask::Impl::clearAsyncErrorMessage()
{
    std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
    hasAsyncError = 0;
    asyncErrorMessage.clear();
}

IOclPanoramaLocalDiskTask::IOclPanoramaLocalDiskTask()
{
    ptrImpl.reset(new Impl);
}

IOclPanoramaLocalDiskTask::~IOclPanoramaLocalDiskTask()
{

}

bool IOclPanoramaLocalDiskTask::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
    int panoType, const std::string& cameraParamFile, const std::string& customMaskFile, const std::string& logoFile, int logoHFov,
    int highQualityBlend, const std::string& dstVideoFile, int dstWidth, int dstHeight,
    int dstVideoBitRate, const std::string& dstVideoEncoder, const std::string& dstVideoPreset,
    int dstVideoMaxFrameCount)
{
    return ptrImpl->init(srcVideoFiles, offsets, audioIndex, cameraParamFile, customMaskFile,
        logoFile, logoHFov, highQualityBlend, dstVideoFile, dstWidth, dstHeight,
        dstVideoBitRate, dstVideoEncoder, dstVideoPreset, dstVideoMaxFrameCount);
}

bool IOclPanoramaLocalDiskTask::start()
{
    return ptrImpl->start();
}

void IOclPanoramaLocalDiskTask::waitForCompletion()
{
    ptrImpl->waitForCompletion();
}

int IOclPanoramaLocalDiskTask::getProgress() const
{
    return ptrImpl->getProgress();
}

void IOclPanoramaLocalDiskTask::cancel()
{
    ptrImpl->cancel();
}

void IOclPanoramaLocalDiskTask::getLastSyncErrorMessage(std::string& message) const
{
    ptrImpl->getLastSyncErrorMessage(message);
}

bool IOclPanoramaLocalDiskTask::hasAsyncErrorMessage() const
{
    return ptrImpl->hasAsyncErrorMessage();
}

void IOclPanoramaLocalDiskTask::getLastAsyncErrorMessage(std::string& message)
{
    return ptrImpl->getLastAsyncErrorMessage(message);
}
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct StampedPinnedMemoryVector
{
    std::vector<cv::cuda::HostMem> frames;
    std::vector<long long int> timeStamps;
    std::vector<int> frameIndexes;
};

typedef BoundedCompleteQueue<StampedPinnedMemoryVector> FrameVectorBufferForCuda;
typedef BoundedCompleteQueue<CudaMixedAudioVideoFrame> MixedFrameBufferForCuda;
typedef std::deque<CudaMixedAudioVideoFrame> TempAudioFrameBufferForCuda;

struct CudaPanoramaLocalDiskTask::Impl
{
    Impl();
    ~Impl();
    bool init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
        int panoType, const std::string& cameraParamFile, const std::string& exposureWhiteBalanceFile,
        const std::string& customMaskFile, const std::string& logoFile, int logoHFov,
        int highQualityBlend, const std::string& dstVideoFile, int dstWidth, int dstHeight,
        int dstVideoBitRate, const std::string& dstVideoEncoder, const std::string& dstVideoPreset, 
        int dstVideoMaxFrameCount);
    bool init(const std::string& configFile);
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
    std::vector<std::vector<std::vector<unsigned char> > > luts;
    std::unique_ptr<CudaPanoramaRender2> render;
    PinnedMemoryPool srcFramesMemoryPool;
    AudioVideoFramePool audioFramesMemoryPool;
    FrameVectorBufferForCuda decodeFramesBuffer;
    CudaHostMemVideoFrameMemoryPool dstFramesMemoryPool;
    MixedFrameBufferForCuda procFrameBuffer;
    cv::Mat blendImageCpu;
    CudaWatermarkFilter watermarkFilter;
    std::unique_ptr<CudaLogoFilter> logoFilter;
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

CudaPanoramaLocalDiskTask::Impl::Impl()
{
    clear();
}

CudaPanoramaLocalDiskTask::Impl::~Impl()
{
    clear();
}

bool CudaPanoramaLocalDiskTask::Impl::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
    int tryAudioIndex, int panoType, const std::string& cameraParamFile, 
    const std::string& exposureWhiteBalanceFile, const std::string& customMaskFile, 
    const std::string& logoFile, int logoHFov, int highQualityBlend, const std::string& dstVideoFile, 
    int dstWidth, int dstHeight, int dstVideoBitRate, const std::string& dstVideoEncoder, 
    const std::string& dstVideoPreset, int dstVideoMaxFrameCount)
{
    ztool::lprintf("Info in %s, params: src video files num = %d, ", __FUNCTION__, srcVideoFiles.size());
    for (int i = 0; i < srcVideoFiles.size(); i++)
        ztool::lprintf("[%d] %s, ", i, srcVideoFiles[i].c_str());
    ztool::lprintf("offsets num = %d, ", offsets.size());
    for (int i = 0; i < offsets.size(); i++)
        ztool::lprintf("[%d] %d, ", i, offsets[i]);
    ztool::lprintf("try audio index = %d, ", tryAudioIndex);
    ztool::lprintf("pano type = %d(%s), ", panoType, getPanoStitchTypeString(panoType));
    ztool::lprintf("camera param file = %s, expo white balance file = %s, custom mask file = %s, "
        "logo file = %s, logo hfov = %d, high quality blend = %d, ",
        cameraParamFile.c_str(), exposureWhiteBalanceFile.c_str(), customMaskFile.c_str(),
        logoFile.c_str(), logoHFov, highQualityBlend);
    ztool::lprintf("dst video file = %s, dst width = %d, dst height = %d, dst video bps = %d, ",
        dstVideoFile.c_str(), dstWidth, dstHeight, dstVideoBitRate);
    ztool::lprintf("dst video encoder = %s, dst video preset = %s, dst video max frame count = %d\n",
        dstVideoEncoder.c_str(), dstVideoPreset.c_str(), dstVideoMaxFrameCount);

    clear();

    if (srcVideoFiles.empty() || (srcVideoFiles.size() != offsets.size()))
    {
        ztool::lprintf("Error in %s, size of srcVideoFiles and size of offsets empty or unmatch.\n", __FUNCTION__);
        syncErrorMessage = getText(TI_PARAM_CHECK_FAIL);
        return false;
    }

    numVideos = srcVideoFiles.size();

    dstSize.width = dstWidth;
    dstSize.height = dstHeight;

    bool ok = false;
    ok = prepareSrcVideos(srcVideoFiles, avp::PixelTypeBGR32, offsets, tryAudioIndex, readers, audioIndex, srcSize, validFrameCount);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not open video file(s)\n", __FUNCTION__);
        syncErrorMessage = getText(TI_OPEN_VIDEO_FAIL);
        return false;
    }

    if (dstVideoMaxFrameCount > 0 && validFrameCount > dstVideoMaxFrameCount)
        validFrameCount = dstVideoMaxFrameCount;

    ok = srcFramesMemoryPool.init(readers[0].getVideoHeight(), readers[0].getVideoWidth(), CV_8UC4);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not init memory pool for source video frames\n", __FUNCTION__);
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
            ztool::lprintf("Error in %s, could not init memory pool for audio frames\n", __FUNCTION__);
            syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
            return false;
        }
    }

    if (!exposureWhiteBalanceFile.empty())
    {
        std::vector<double> es, rs, bs;
        ok = loadExposureWhiteBalance(exposureWhiteBalanceFile, es, rs, bs);
        if (!ok)
        {
            ztool::lprintf("Error in %s, could not load exposure white balance file\n", __FUNCTION__);
            syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
            return false;
        }

        if (es.size() != numVideos || rs.size() != numVideos || bs.size() != numVideos)
        {
            ztool::lprintf("Error in %s, exposure and white balance param size unsatisfied, %d, %d, %d, should be %d\n",
                __FUNCTION__, es.size(), rs.size(), bs.size(), numVideos);
            syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
            return false;
        }

        if (needCorrectExposureWhiteBalance(es, rs, bs))
            getExposureColorOptimizeLUTs(es, rs, bs, luts);
    }

    if (panoType == PanoStitchTypeMISO)
        render.reset(new CudaPanoramaRender2);
    else if (panoType == PanoStitchTypeRicoh)
        render.reset(new CudaRicohPanoramaRender);
    else
    {
        ztool::lprintf("Error in %s, unsupported pano stitch type %d, should be %d or %d\n",
            __FUNCTION__, panoType, PanoStitchTypeMISO, PanoStitchTypeRicoh);
    }

    ok = render->prepare(cameraParamFile, highQualityBlend, srcSize, dstSize);
    if (!ok)
    {
        ztool::lprintf("Error in %s, render prepare failed\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (render->getNumImages() != numVideos)
    {
        ztool::lprintf("Error in %s, num images in render not equal to num videos\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    isLibX264 = dstVideoEncoder == "h264" ? 1 : 0;

    ok = dstFramesMemoryPool.init(isLibX264 ? avp::PixelTypeYUV420P : avp::PixelTypeNV12, dstSize.width, dstSize.height);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not init memory pool for dst video frames\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    ok = watermarkFilter.init(dstSize.width, dstSize.height);
    if (!ok)
    {
        ztool::lprintf("Error in %s, init watermark filter failed\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (!logoFile.empty() && logoHFov > 0)
    {
        logoFilter.reset(new CudaLogoFilter);
        ok = logoFilter->init(logoFile, logoHFov, dstSize.width, dstSize.height);
        if (!ok)
        {
            ztool::lprintf("Error in %s, init logo filter failed\n", __FUNCTION__);
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
        ztool::lprintf("Error in %s, video writer open failed\n", __FUNCTION__);
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

bool CudaPanoramaLocalDiskTask::Impl::init(const std::string& configFile)
{
    std::vector<std::string> srcVideoFiles;
    std::vector<int> offsets;
    int tryAudioIndex;
    int panoType;
    std::string logoFile;
    int logoHFov;
    int highQualityBlend;
    std::string dstVideoFile;
    int dstWidth;
    int dstHeight;
    int dstVideoBitRate;
    std::string dstVideoEncoder;
    std::string dstVideoPreset;
    int dstVideoMaxFrameCount;

    bool ok = false;
    ok = loadVideoFileNamesAndOffset(configFile, srcVideoFiles, offsets);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not load video file names and offsets\n", __FUNCTION__);
        syncErrorMessage = getText(TI_CONFIG_FILE_PARSE_FAIL);
        return false;
    }
    ok = loadOutputConfig(configFile, tryAudioIndex, panoType, logoFile, logoHFov,
        highQualityBlend, dstVideoFile, dstWidth, dstHeight, dstVideoBitRate,
        dstVideoEncoder, dstVideoPreset, dstVideoMaxFrameCount);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not load output video params\n", __FUNCTION__);
        syncErrorMessage = getText(TI_CONFIG_FILE_PARSE_FAIL);
        return false;
    }

    return init(srcVideoFiles, offsets, tryAudioIndex, panoType, configFile, configFile, configFile,
        logoFile, logoHFov, highQualityBlend, dstVideoFile, dstWidth, dstHeight, dstVideoBitRate,
        dstVideoEncoder, dstVideoPreset, dstVideoMaxFrameCount);
}

bool CudaPanoramaLocalDiskTask::Impl::start()
{
    if (!initSuccess)
    {
        ztool::lprintf("Error in %s, init not success, could not start\n", __FUNCTION__);
        return false;
    }

    if (finish)
        return false;

    decodeThread.reset(new std::thread(&CudaPanoramaLocalDiskTask::Impl::decode, this));
    procThread.reset(new std::thread(&CudaPanoramaLocalDiskTask::Impl::proc, this));
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
        ztool::lprintf("Info in %s, write video finish\n", __FUNCTION__);

    finish = true;
}

int CudaPanoramaLocalDiskTask::Impl::getProgress() const
{
    return finishPercent.load();
}

void CudaPanoramaLocalDiskTask::Impl::getLastSyncErrorMessage(std::string& message) const
{
    message = syncErrorMessage;
}

bool CudaPanoramaLocalDiskTask::Impl::hasAsyncErrorMessage() const
{
    return hasAsyncError;
}

void CudaPanoramaLocalDiskTask::Impl::getLastAsyncErrorMessage(std::string& message)
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

void CudaPanoramaLocalDiskTask::Impl::clear()
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

    render.reset();

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

void CudaPanoramaLocalDiskTask::Impl::cancel()
{
    isCanceled = true;
}

void CudaPanoramaLocalDiskTask::Impl::decode()
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    decodeCount = 0;
    int mediaType;
    while (true)
    {
        StampedPinnedMemoryVector videoFrames;
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
        //ztool::lprintf("decode count = %d\n", decodeCount);

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

    ztool::lprintf("In %s, total decode %d\n", __FUNCTION__, decodeCount);
    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void CudaPanoramaLocalDiskTask::Impl::proc()
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    procCount = 0;
    StampedPinnedMemoryVector srcFrames;
    std::vector<cv::Mat> images(numVideos);
    cv::cuda::GpuMat bgr32;
    CudaMixedAudioVideoFrame videoFrame;
    cv::cuda::GpuMat y, u, v, uv;
    int index = audioIndex >= 0 ? audioIndex : 0;
    while (true)
    {
        if (!decodeFramesBuffer.pull(srcFrames))
            break;

        if (isCanceled)
            break;
        
        for (int i = 0; i < numVideos; i++)
            images[i] = srcFrames.frames[i].createMatHeader();        
        bool ok = false;
        if (luts.empty())
            ok = render->render(images, bgr32);
        else
            ok = render->render(images, bgr32, luts);
        if (!ok)
        {
            ztool::lprintf("Error in %s, render failed\n", __FUNCTION__);
            setAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE));
            isCanceled = true;
            break;
        }

        if (logoFilter)
        {
            ok = logoFilter->addLogo(bgr32);
            if (!ok)
            {
                ztool::lprintf("Error in %s, add logo failed\n", __FUNCTION__);
                setAsyncErrorMessage(getText(TI_WRITE_TO_VIDEO_FAIL_TASK_TERMINATE));
                break;
            }
        }

        if (addWatermark)
        {
            ok = watermarkFilter.addWatermark(bgr32);
            if (!ok)
            {
                ztool::lprintf("Error in %s, add watermark failed\n", __FUNCTION__);
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
            cv::Mat yy = videoFrame.planes[0].createMatHeader();
            cv::Mat uu = videoFrame.planes[1].createMatHeader();
            cv::Mat vv = videoFrame.planes[2].createMatHeader();
            y.download(yy);
            u.download(uu);
            v.download(vv);
        }
        else
        {
            cvtBGR32ToNV12(bgr32, y, uv);
            cv::Mat yy = videoFrame.planes[0].createMatHeader();
            cv::Mat uvuv = videoFrame.planes[1].createMatHeader();
            y.download(yy);
            uv.download(uvuv);
        }

        procFrameBuffer.push(videoFrame);
        procCount++;
        //ztool::lprintf("proc count = %d\n", procCount);
    }
    
    if (!isCanceled)
    {
        while (procFrameBuffer.size())
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    procFrameBuffer.stop();

    ztool::lprintf("In %s, total proc %d\n", __FUNCTION__, procCount);
    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void CudaPanoramaLocalDiskTask::Impl::encode()
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    encodeCount = 0;
    int step = 1;
    if (validFrameCount > 0)
        step = validFrameCount / 100.0 + 0.5;
    if (step < 1)
        step = 1;
    ztool::lprintf("In %s, validFrameCount = %d, step = %d\n", __FUNCTION__, validFrameCount, step);
    ztool::Timer timerEncode;
    encodeCount = 0;
    CudaMixedAudioVideoFrame frame;
    int encodeState = VideoFrameNotCome;
    int hasAudio = audioIndex >= 0 && audioIndex < numVideos;
    TempAudioFrameBufferForCuda tempAudioFrames;
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
            ztool::lprintf("Error in %s, render failed\n", __FUNCTION__);
            setAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE));
            isCanceled = true;
            break;
        }

        // Only when the frame is of type video can we increase encodeCount
        if (frame.frame.mediaType == avp::VIDEO)
            encodeCount++;
        //ztool::lprintf("frame %d finish, encode time = %f\n", encodeCount, timerEncode.elapse());

        if (encodeCount % step == 0)
            finishPercent.store(double(encodeCount) / (validFrameCount > 0 ? validFrameCount : 100) * 100);
    }

    writer.close();

    finishPercent.store(100);

    ztool::lprintf("In %s, total encode %d\n", __FUNCTION__, encodeCount);
    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void CudaPanoramaLocalDiskTask::Impl::setAsyncErrorMessage(const std::string& message)
{
    std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
    hasAsyncError = 1;
    asyncErrorMessage = message;
}

void CudaPanoramaLocalDiskTask::Impl::clearAsyncErrorMessage()
{
    std::lock_guard<std::mutex> lg(mtxAsyncErrorMessage);
    hasAsyncError = 0;
    asyncErrorMessage.clear();
}

CudaPanoramaLocalDiskTask::CudaPanoramaLocalDiskTask()
{
    ptrImpl.reset(new Impl);
}

CudaPanoramaLocalDiskTask::~CudaPanoramaLocalDiskTask()
{

}

bool CudaPanoramaLocalDiskTask::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets, int audioIndex,
    int panoType, const std::string& cameraParamFile, const std::string& exposureWhiteBalanceFile,
    const std::string& customMaskFile, const std::string& logoFile, int logoHFov,
    int highQualityBlend, const std::string& dstVideoFile, int dstWidth, int dstHeight,
    int dstVideoBitRate, const std::string& dstVideoEncoder, const std::string& dstVideoPreset, 
    int dstVideoMaxFrameCount)
{
    return ptrImpl->init(srcVideoFiles, offsets, audioIndex, panoType, 
        cameraParamFile, exposureWhiteBalanceFile, customMaskFile, 
        logoFile, logoHFov, highQualityBlend, dstVideoFile, dstWidth, dstHeight,
        dstVideoBitRate, dstVideoEncoder, dstVideoPreset, dstVideoMaxFrameCount);
}

bool CudaPanoramaLocalDiskTask::init(const std::string& configFile)
{
    return ptrImpl->init(configFile);
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

void CudaPanoramaLocalDiskTask::getLastSyncErrorMessage(std::string& message) const
{
    ptrImpl->getLastSyncErrorMessage(message);
}

bool CudaPanoramaLocalDiskTask::hasAsyncErrorMessage() const
{
    return ptrImpl->hasAsyncErrorMessage();
}

void CudaPanoramaLocalDiskTask::getLastAsyncErrorMessage(std::string& message)
{
    return ptrImpl->getLastAsyncErrorMessage(message);
}

#if COMPILE_DISCRETE_OPENCL

#include "DOclPanoramaTaskUtil.h"

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
    DOclPinnedMemoryPool srcFramesMemoryPool;
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
    ztool::lprintf("Info in %s, params: src video files num = %d, ", __FUNCTION__, srcVideoFiles.size());
    for (int i = 0; i < srcVideoFiles.size(); i++)
        ztool::lprintf("[%d] %s, ", i, srcVideoFiles[i].c_str());
    ztool::lprintf("offsets num = %d, ", offsets.size());
    for (int i = 0; i < offsets.size(); i++)
        ztool::lprintf("[%d] %d, ", i, offsets[i]);
    ztool::lprintf("try audio index = %d, ", tryAudioIndex);
    ztool::lprintf("camera param file = %s, custom mask file = %s, logo file = %s, logo hfov = %d, high quality blend = %d, ",
        cameraParamFile.c_str(), customMaskFile.c_str(), logoFile.c_str(), logoHFov, highQualityBlend);
    ztool::lprintf("dst video file = %s, dst width = %d, dst height = %d, dst video bps = %d, ",
        dstVideoFile.c_str(), dstWidth, dstHeight, dstVideoBitRate);
    ztool::lprintf("dst video encoder = %s, dst video preset = %s, dst video max frame count = %d\n",
        dstVideoEncoder.c_str(), dstVideoPreset.c_str(), dstVideoMaxFrameCount);

    clear();

    if (srcVideoFiles.empty() || (srcVideoFiles.size() != offsets.size()))
    {
        ztool::lprintf("Error in %s, size of srcVideoFiles and size of offsets empty or unmatch.\n", __FUNCTION__);
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
        ztool::lprintf("Error in %s, opencl init failed\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    ok = prepareSrcVideos(srcVideoFiles, avp::PixelTypeBGR32, offsets, tryAudioIndex, readers, audioIndex, srcSize, validFrameCount);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not open video file(s)\n", __FUNCTION__);
        syncErrorMessage = getText(TI_OPEN_VIDEO_FAIL);
        return false;
    }

    if (dstVideoMaxFrameCount > 0 && validFrameCount > dstVideoMaxFrameCount)
        validFrameCount = dstVideoMaxFrameCount;

    ok = srcFramesMemoryPool.init(readers[0].getVideoHeight(), readers[0].getVideoWidth(), CV_8UC4,
        docl::HostMem::BufferFlagAllocHostPtr | docl::HostMem::BufferFlagReadOnly, docl::HostMem::MapFlagWrite);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not init memory pool for source video frames\n", __FUNCTION__);
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
            ztool::lprintf("Error in %s, could not init memory pool for audio frames\n", __FUNCTION__);
            syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
            return false;
        }
    }

    ok = render.prepare(cameraParamFile, highQualityBlend, srcSize, dstSize);
    if (!ok)
    {
        ztool::lprintf("Error in %s, render prepare failed\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (render.getNumImages() != numVideos)
    {
        ztool::lprintf("Error in %s, num images in render not equal to num videos\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    isLibX264 = dstVideoEncoder == "h264" ? 1 : 0;

    ok = dstFramesMemoryPool.init(isLibX264 ? avp::PixelTypeYUV420P : avp::PixelTypeNV12, dstSize.width, dstSize.height,
        docl::HostMem::BufferFlagAllocHostPtr | docl::HostMem::BufferFlagWriteOnly, docl::HostMem::MapFlagRead);
    if (!ok)
    {
        ztool::lprintf("Error in %s, could not init memory pool for dst video frames\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    ok = watermarkFilter.init(dstSize.width, dstSize.height);
    if (!ok)
    {
        ztool::lprintf("Error in %s, init watermark filter failed\n", __FUNCTION__);
        syncErrorMessage = getText(TI_STITCH_INIT_FAIL);
        return false;
    }

    if (!logoFile.empty() && logoHFov > 0)
    {
        logoFilter.reset(new DOclLogoFilter);
        ok = logoFilter->init(logoFile, logoHFov, dstSize.width, dstSize.height);
        if (!ok)
        {
            ztool::lprintf("Error in %s, init logo filter failed\n", __FUNCTION__);
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
        ztool::lprintf("Error in %s, video writer open failed\n", __FUNCTION__);
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
        ztool::lprintf("Error in %s, init not success, could not start\n", __FUNCTION__);
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
        ztool::lprintf("Info in %s, write video finish\n", __FUNCTION__);

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
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

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
        //ztool::lprintf("decode count = %d\n", decodeCount);

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

    ztool::lprintf("In %s, total decode %d\n", __FUNCTION__, decodeCount);
    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void DOclPanoramaLocalDiskTask::Impl::proc()
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    procCount = 0;
    StampedPinnedMemoryVectorForDOcl srcFrames;
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

        bool ok = render.render(srcFrames.frames, bgr32);
        if (!ok)
        {
            ztool::lprintf("Error in %s, render failed\n", __FUNCTION__);
            setAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE));
            isCanceled = true;
            break;
        }

        if (logoFilter)
        {
            ok = logoFilter->addLogo(bgr32);
            if (!ok)
            {
                ztool::lprintf("Error in %s, add logo failed\n", __FUNCTION__);
                setAsyncErrorMessage(getText(TI_WRITE_TO_VIDEO_FAIL_TASK_TERMINATE));
                break;
            }
        }

        if (addWatermark)
        {
            ok = watermarkFilter.addWatermark(bgr32);
            if (!ok)
            {
                ztool::lprintf("Error in %s, add watermark failed\n", __FUNCTION__);
                setAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE));
                isCanceled = true;
                break;
            }
        }

        dstFramesMemoryPool.get(videoFrame);
        videoFrame.frame.timeStamp = srcFrames.timeStamps[index];
        if (isLibX264)
        {
            cvtBGR32ToYUV420P(bgr32, y, u, v);
            y.download(videoFrame.planes[0], 
                docl::HostMem::BufferFlagAllocHostPtr | docl::HostMem::BufferFlagWriteOnly, 
                docl::HostMem::MapFlagRead);
            u.download(videoFrame.planes[1], 
                docl::HostMem::BufferFlagAllocHostPtr | docl::HostMem::BufferFlagWriteOnly, 
                docl::HostMem::MapFlagRead);
            v.download(videoFrame.planes[2], 
                docl::HostMem::BufferFlagAllocHostPtr | docl::HostMem::BufferFlagWriteOnly, 
                docl::HostMem::MapFlagRead);
        }
        else
        {
            cvtBGR32ToNV12(bgr32, y, uv);
            y.download(videoFrame.planes[0], 
                docl::HostMem::BufferFlagAllocHostPtr | docl::HostMem::BufferFlagWriteOnly, 
                docl::HostMem::MapFlagRead);
            uv.download(videoFrame.planes[1], 
                docl::HostMem::BufferFlagAllocHostPtr | docl::HostMem::BufferFlagWriteOnly, 
                docl::HostMem::MapFlagRead);
        }

        procFrameBuffer.push(videoFrame);
        procCount++;
        //ztool::lprintf("proc count = %d\n", procCount);
    }

    if (!isCanceled)
    {
        while (procFrameBuffer.size())
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    procFrameBuffer.stop();

    ztool::lprintf("In %s, total proc %d\n", __FUNCTION__, procCount);
    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void DOclPanoramaLocalDiskTask::Impl::encode()
{
    size_t id = std::this_thread::get_id().hash();
    ztool::lprintf("Thread %s [%8x] started\n", __FUNCTION__, id);

    encodeCount = 0;
    int step = 1;
    if (validFrameCount > 0)
        step = validFrameCount / 100.0 + 0.5;
    if (step < 1)
        step = 1;
    ztool::lprintf("In %s, validFrameCount = %d, step = %d\n", __FUNCTION__, validFrameCount, step);
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
            ztool::lprintf("Error in %s, render failed\n", __FUNCTION__);
            setAsyncErrorMessage(getText(TI_STITCH_FAIL_TASK_TERMINATE));
            isCanceled = true;
            break;
        }

        // Only when the frame is of type video can we increase encodeCount
        if (frame.frame.mediaType == avp::VIDEO)
            encodeCount++;
        //ztool::lprintf("frame %d finish, encode time = %f\n", encodeCount, timerEncode.elapse());

        if (encodeCount % step == 0)
            finishPercent.store(double(encodeCount) / (validFrameCount > 0 ? validFrameCount : 100) * 100);
    }

    writer.close();

    finishPercent.store(100);

    ztool::lprintf("In %s, total encode %d\n", __FUNCTION__, encodeCount);
    ztool::lprintf("Thread %s [%8x] end\n", __FUNCTION__, id);
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
    int panoType, const std::string& cameraParamFile, const std::string& customMaskFile, const std::string& logoFile, int logoHFov,
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

#endif