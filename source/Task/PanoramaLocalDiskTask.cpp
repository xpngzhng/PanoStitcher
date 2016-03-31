#include "PanoramaTask.h"
#include "PanoramaTaskUtil.h"
#include "ZReproject.h"
#include "Timer.h"

CPUPanoramaLocalDiskTask::CPUPanoramaLocalDiskTask()
{
    clear();
}

CPUPanoramaLocalDiskTask::~CPUPanoramaLocalDiskTask()
{
    clear();
}

bool CPUPanoramaLocalDiskTask::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
    int tryAudioIndex, const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight,
    int dstVideoBitRate, ProgressCallbackFunction func, void* data)
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

    printf("Info in %s, prepare finish\n", __FUNCTION__);

    printf("Info in %s, open dst video\n", __FUNCTION__);
    std::vector<avp::Option> options;
    options.push_back(std::make_pair("preset", "medium"));
    if (audioIndex >= 0 && audioIndex < numVideos)
    {
        ok = writer.open(dstVideoFile, "", true, 
            true, "aac", readers[audioIndex].getAudioSampleType(), readers[audioIndex].getAudioChannelLayout(), 
            readers[audioIndex].getAudioSampleRate(), 128000,
            true, "h264", avp::PixelTypeBGR24, dstSize.width, dstSize.height, readers[0].getVideoFps(), dstVideoBitRate, options);
    }
    else
    {
        ok = writer.open(dstVideoFile, "", false, false, "", avp::SampleTypeUnknown, 0, 0, 0,
            true, "h264", avp::PixelTypeBGR24, dstSize.width, dstSize.height, readers[0].getVideoFps(), dstVideoBitRate, options);
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

void CPUPanoramaLocalDiskTask::run()
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

bool CPUPanoramaLocalDiskTask::start()
{
    if (!initSuccess)
        return false;

    if (finish)
        return false;

    thread.reset(new std::thread(&CPUPanoramaLocalDiskTask::run, this));
    return true;
}

void CPUPanoramaLocalDiskTask::waitForCompletion()
{
    if (thread && thread->joinable())
        thread->join();
    thread.reset(0);
}

int CPUPanoramaLocalDiskTask::getProgress() const
{
    return finishPercent.load();
}

void CPUPanoramaLocalDiskTask::cancel()
{
    endFlag = true;
}

void CPUPanoramaLocalDiskTask::clear()
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

CudaPanoramaLocalDiskTask::CudaPanoramaLocalDiskTask()
{
    clear();
}

CudaPanoramaLocalDiskTask::~CudaPanoramaLocalDiskTask()
{
    clear();
}

bool CudaPanoramaLocalDiskTask::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
    int tryAudioIndex, const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight,
    int dstVideoBitRate, ProgressCallbackFunction func, void* data)
{
    clear();

    if (srcVideoFiles.empty() || (srcVideoFiles.size() != offsets.size()))
    {
        printf("Error in %s, size of srcVideoFiles and size of offsets empty or unmatch.\n", __FUNCTION__);
        return false;
    }

    numVideos = srcVideoFiles.size();

    //std::vector<PhotoParam> params;
    //if (!loadPhotoParams(cameraParamFile, params))
    //{
    //    printf("Error in %s, failed to load params\n", __FUNCTION__);
    //}
    //if (params.size() < numVideos)
    //{
    //    printf("Error in %s, params.size() < numVideos\n", __FUNCTION__);
    //    return false;
    //}
    //else if (params.size() > numVideos)
    //{
    //    printf("Warning in %s, params.size() > numVideos\n", __FUNCTION__);
    //}

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

    //std::vector<cv::Mat> dstMasks;
    //std::vector<cv::Mat> dstSrcMaps, xmaps, ymaps;
    //getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, dstMasks);

    //xmapsGpu.resize(numVideos);
    //ymapsGpu.resize(numVideos);
    //cv::Mat map32F;
    //cv::Mat map64F[2];
    //for (int i = 0; i < numVideos; i++)
    //{
    //    cv::split(dstSrcMaps[i], map64F);
    //    map64F[0].convertTo(map32F, CV_32F);
    //    xmapsGpu[i].upload(map32F);
    //    map64F[1].convertTo(map32F, CV_32F);
    //    ymapsGpu[i].upload(map32F);
    //}
    //dstSrcMaps.clear();
    //map32F.release();
    //map64F[0].release();
    //map64F[1].release();

    //ok = blender.prepare(dstMasks, 16, 2);
    //ok = blender.prepare(dstMasks, 50);
    ok = render.prepare(cameraParamFile, PanoramaRender::BlendTypeMultiband, srcSize, dstSize);
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
    //dstMasks.clear();

    //streams.resize(numVideos);
    //pinnedMems.resize(numVideos);
    //for (int i = 0; i < numVideos; i++)
    //    pinnedMems[i].create(srcSize, CV_8UC4);

    //imagesGpu.resize(numVideos);
    //reprojImagesGpu.resize(numVideos);

    printf("Info in %s, prepare finish\n", __FUNCTION__);

    printf("Info in %s, open dst video\n", __FUNCTION__);
    std::vector<avp::Option> options;
    options.push_back(std::make_pair("preset", "medium"));
    //ok = writer.open(dstVideoFile, "", false, false, "", avp::SampleTypeUnknown, 0, 0, 0,
    //    true, "h264", avp::PixelTypeBGR32, dstSize.width, dstSize.height, readers[0].getVideoFps(), dstVideoBitRate, options);
    if (audioIndex >= 0 && audioIndex < numVideos)
    {
        ok = writer.open(dstVideoFile, "", true, true, "aac", readers[audioIndex].getAudioSampleType(),
            readers[audioIndex].getAudioChannelLayout(), readers[audioIndex].getAudioSampleRate(), 128000,
            true, "h264", avp::PixelTypeBGR32, dstSize.width, dstSize.height, readers[0].getVideoFps(), 48000000, options);
    }
    else
    {
        ok = writer.open(dstVideoFile, "", false, false, "", avp::SampleTypeUnknown, 0, 0, 0,
            true, "h264", avp::PixelTypeBGR32, dstSize.width, dstSize.height, readers[0].getVideoFps(), 48000000, options);
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

bool CudaPanoramaLocalDiskTask::start()
{
    if (!initSuccess)
        return false;

    if (finish)
        return false;

    printf("Info in %s, write video begin\n", __FUNCTION__);

    //decodedImagesOwnedByDecodeThread = true;
    //encodedImageOwnedByProcThread = true;

    decodeThread.reset(new std::thread(&CudaPanoramaLocalDiskTask::decode, this));
    procThread.reset(new std::thread(&CudaPanoramaLocalDiskTask::proc, this));
    encodeThread.reset(new std::thread(&CudaPanoramaLocalDiskTask::encode, this));

    return true;
}

void CudaPanoramaLocalDiskTask::waitForCompletion()
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

    if (!finish)
        printf("Info in %s, write video finish\n", __FUNCTION__);

    finish = true;
}

int CudaPanoramaLocalDiskTask::getProgress() const
{
    return finishPercent.load();
}

void CudaPanoramaLocalDiskTask::clear()
{
    numVideos = 0;
    srcSize = cv::Size();
    dstSize = cv::Size();
    readers.clear();
    //xmapsGpu.clear();
    //ymapsGpu.clear();
    //streams.clear();
    //pinnedMems.clear();
    //imagesGpu.clear();
    //reprojImagesGpu.clear();
    writer.close();

    //videoEnd = false;
    //procEnd = false;

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

    //decodedImagesOwnedByDecodeThread = true;
    //encodedImageOwnedByProcThread = true;

    if (decodeThread && decodeThread->joinable())
        decodeThread->join();
    decodeThread.reset(0);
    if (procThread && procThread->joinable())
        procThread->join();
    procThread.reset(0);
    if (encodeThread && encodeThread->joinable())
        encodeThread->join();
    encodeThread.reset(0);

    progressCallbackFunc = 0;
    progressCallbackData = 0;

    initSuccess = false;
    finish = true;
    isCanceled = false;
}

void CudaPanoramaLocalDiskTask::cancel()
{
    isCanceled = true;
}

void CudaPanoramaLocalDiskTask::decode()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    decodeCount = 0;
/*
#if ENABLE_CALC_TIME
    ztool::Timer timer;
#endif
    while (true)
    {
#if ENABLE_CALC_TIME
        timer.start();
#endif
        {
            std::unique_lock<std::mutex> ul(mtxDecodedImages);
            cvDecodedImagesForWrite.wait(ul, [this]{return decodedImagesOwnedByDecodeThread; });
            bool successRead = true;
            //timerDecode.start();
            for (int i = 0; i < numVideos; i++)
            {
                avp::AudioVideoFrame frame;
                if (!readers[i].read(frame))
                {
                    successRead = false;
                    break;
                }
                cv::Mat src(frame.height, frame.width, CV_8UC4, frame.data, frame.step);
                cv::Mat dst(pinnedMems[i]);
                src.copyTo(dst);
            }
            //timerDecode.end();

            if (!successRead || isCanceled)
                break;

            // NOTICE!!!!!!
            // The following line had better be after break.
            // If not, before the lock at the end of the function, if proc() arrives at the first lock and wait,
            // it may find decodedImagesOwnedByDecodeThread == false, but procEnd == false, 
            // then the last frames will be reprojected and blends twice.
            decodedImagesOwnedByDecodeThread = false;
        }
        cvDecodedImagesForRead.notify_one();
        decodeCount++;
        //printf("decode count = %d\n", decodeCount);

#if ENABLE_CALC_TIME
        timer.end();
        printf("d %f %f\n", timerDecode.elapse(), timer.elapse());
#endif
    }

    {
        std::unique_lock<std::mutex> ul(mtxDecodedImages);
        cvDecodedImagesForWrite.wait(ul, [this]{return decodedImagesOwnedByDecodeThread; });
        decodedImagesOwnedByDecodeThread = false;
        videoEnd = true;
    }
    cvDecodedImagesForRead.notify_one();
*/
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
            cv::Mat dst = deepFrames.frames[i];
            src.copyTo(dst);
        }

        decodeFramesBuffer.push(deepFrames);
        decodeCount++;
        //printf("decode count = %d\n", decodeCount);
    }

    while (decodeFramesBuffer.size())
        std::this_thread::sleep_for(std::chrono::microseconds(25));
    decodeFramesBuffer.stop();

    for (int i = 0; i < numVideos; i++)
        readers[i].close();

    printf("total decode %d\n", decodeCount);
    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void CudaPanoramaLocalDiskTask::proc()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    procCount = 0;
    /*
#if ENABLE_CALC_TIME
    ztool::Timer timer, timerWR, timerWW;
#endif
    while (true)
    {
#if ENABLE_CALC_TIME
        timer.start();
#endif
        {
#if ENABLE_CALC_TIME
            timerWR.start();
#endif
            std::unique_lock<std::mutex> ul(mtxDecodedImages);
            cvDecodedImagesForRead.wait(ul, [this]{return !decodedImagesOwnedByDecodeThread; });
#if ENABLE_CALC_TIME
            timerWR.end();
#endif
            if (videoEnd)
                break;

            //timerReproject.start();
            for (int i = 0; i < numVideos; i++)
                streams[i].enqueueUpload(pinnedMems[i], imagesGpu[i]);
            for (int i = 0; i < numVideos; i++)
                cudaReprojectTo16S(imagesGpu[i], reprojImagesGpu[i], xmapsGpu[i], ymapsGpu[i], streams[i]);
            for (int i = 0; i < numVideos; i++)
                streams[i].waitForCompletion();
            //timerReproject.end();

            decodedImagesOwnedByDecodeThread = true;
        }
        cvDecodedImagesForWrite.notify_one();

#if ENABLE_CALC_TIME
        timerBlend.start();
#endif
        blender.blend(reprojImagesGpu, blendImageGpu);
        cv::gpu::Stream::Null().waitForCompletion();

#if ENABLE_CALC_TIME
        timerBlend.end();
#endif

        {
#if ENABLE_CALC_TIME
            timerWW.start();
#endif
            std::unique_lock<std::mutex> ul(mtxEncodedImage);
            cvEncodedImageForWrite.wait(ul, [this]{return encodedImageOwnedByProcThread; });
#if ENABLE_CALC_TIME
            timerWW.end();
#endif

            blendImageGpu.download(blendImageCpu);
            encodedImageOwnedByProcThread = false;
        }
        cvEncodedImageForRead.notify_one();
        procCount++;
        //printf("proc count = %d\n", procCount);
#if ENABLE_CALC_TIME
        timer.end();
        printf("p wr %f, r %f, b %f, ww %f, %f\n",
            timerWR.elapse(), timerReproject.elapse(), timerBlend.elapse(), timerWW.elapse(), timer.elapse());
#endif
    }

    {
        std::unique_lock<std::mutex> ul(mtxDecodedImages);
        cvDecodedImagesForRead.wait(ul, [this]{return !decodedImagesOwnedByDecodeThread; });
        decodedImagesOwnedByDecodeThread = true;
    }
    cvDecodedImagesForWrite.notify_one();

    {
        std::unique_lock<std::mutex> ul(mtxEncodedImage);
        cvEncodedImageForWrite.wait(ul, [this]{return encodedImageOwnedByProcThread; });
        encodedImageOwnedByProcThread = false;
        procEnd = true;
    }
    cvEncodedImageForRead.notify_one();
    */

    StampedPinnedMemoryVector srcFrames;
    avp::SharedAudioVideoFrame dstFrame;
    std::vector<cv::Mat> images(numVideos);
    while (true)
    {
        if (!decodeFramesBuffer.pull(srcFrames))
            break;

        dstFramesMemoryPool.get(dstFrame);
        for (int i = 0; i < numVideos; i++)
            images[i] = srcFrames.frames[i];
        cv::Mat result(dstSize, CV_8UC4, dstFrame.data, dstFrame.step);
        render.render(images, result);
        dstFrame.timeStamp = srcFrames.timeStamp;
        procFrameBuffer.push(dstFrame);

        procCount++;
        //printf("proc count = %d\n", procCount);
    }

    while (procFrameBuffer.size())
        std::this_thread::sleep_for(std::chrono::microseconds(25));
    procFrameBuffer.stop();

    printf("total proc %d\n", procCount);
    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void CudaPanoramaLocalDiskTask::encode()
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
    //cv::Mat smallImage;
    /*
#if ENABLE_CALC_TIME
    ztool::Timer timer;
#endif
    while (true)
    {
#if ENABLE_CALC_TIME
        timer.start();
#endif
        {
            std::unique_lock<std::mutex> ul(mtxEncodedImage);
            cvEncodedImageForRead.wait(ul, [this]{return !encodedImageOwnedByProcThread; });

            if (procEnd)
                break;

            //cv::resize(blendImageCpu, smallImage, cv::Size(), 0.25, 0.25, cv::INTER_NEAREST);
            //cv::imshow("preview", smallImage);
            //cv::waitKey(1);

            //timerEncode.start();
            avp::AudioVideoFrame frame = avp::videoFrame(blendImageCpu.data, blendImageCpu.step, avp::PixelTypeBGR32, blendImageCpu.cols, blendImageCpu.rows, -1LL);
            writer.write(frame);
            //timerEncode.end();

            encodeCount++;
            //printf("frame %d finish, encode time = %f\n", count, timerEncode.elapse());
            //printf("encode frame count = %d\n", encodeCount);

            //if (progressCallbackFunc && (encodeCount % step == 0))
            //    progressCallbackFunc(double(encodeCount) / (validFrameCount > 0 ? validFrameCount : 100), progressCallbackData);
            if (encodeCount % step == 0)
                finishPercent.store(double(encodeCount) / (validFrameCount > 0 ? validFrameCount : 100) * 100);

            encodedImageOwnedByProcThread = true;
        }
        cvEncodedImageForWrite.notify_one();
#if ENABLE_CALC_TIME
        timer.end();
        printf("e %f, %f\n", timerEncode.elapse(), timer.elapse());
#endif
    }

    //if (progressCallbackFunc)
    //    progressCallbackFunc(1.0, progressCallbackData);
    */

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