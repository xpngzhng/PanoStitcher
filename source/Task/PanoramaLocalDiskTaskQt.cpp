#include "PanoramaTask.h"
#include "PanoramaTaskUtil.h"
#include "ZReproject.h"

#include <QtCore/QTime>
#include <QtCore/QObject>
#include <QtWidgets/QApplication>
#include <QtWidgets/QProgressDialog>

QtCPUPanoramaLocalDiskTask::QtCPUPanoramaLocalDiskTask()
{
    clear();
}

QtCPUPanoramaLocalDiskTask::~QtCPUPanoramaLocalDiskTask()
{
    clear();
}

bool QtCPUPanoramaLocalDiskTask::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
    const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight, int dstVideoBitRate)
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
    ok = prepareSrcVideos(srcVideoFiles, true, offsets, readers, srcSize, validFrameCount);
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
    ok = writer.open(dstVideoFile, "", false, false, "", avp::SampleTypeUnknown, 0, 0, 0,
        true, "h264_qsv", avp::PixelTypeBGR24, dstSize.width, dstSize.height, readers[0].getVideoFps(), dstVideoBitRate, options);
    if (!ok)
    {
        printf("Error in %s, video writer open failed\n", __FUNCTION__);
        return false;
    }
    else
        printf("Info in %s, video writer open success\n", __FUNCTION__);

    initSuccess = true;
    finish = false;
    return true;
}

void QtCPUPanoramaLocalDiskTask::run(QWidget* obj)
{
	QProgressDialog* progressDialog = new QProgressDialog(obj);
	progressDialog->setMinimumSize(400, 50);
	progressDialog->resize(QSize(400, 50));
	progressDialog->setModal(true);
	progressDialog->setWindowModality(Qt::WindowModal);
	progressDialog->setWindowTitle(QObject::tr("Generating Panorama Video ......"));
	progressDialog->setRange(0, 100);
	progressDialog->setStyleSheet("background:rgb(88,88,88)");
	progressDialog->setValue(0);
	progressDialog->show();
	QApplication::processEvents();

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

    std::vector<cv::Mat> images(numVideos);
    bool ok = true;
    while (true)
    {
		if (progressDialog->wasCanceled())
		{
			QApplication::processEvents();
			break;
		}

        for (int i = 0; i < numVideos; i++)
        {
            avp::AudioVideoFrame frame;
            if (!readers[i].read(frame))
            {
                ok = false;
                break;
            }

            images[i] = cv::Mat(frame.height, frame.width, CV_8UC3, frame.data, frame.step);
        }
        if (!ok || endFlag)
            break;

        reprojectParallelTo16S(images, reprojImages, dstSrcMaps);
        blender.blend(reprojImages, blendImage);
        avp::AudioVideoFrame frame = avp::videoFrame(blendImage.data, blendImage.step, avp::PixelTypeBGR24, blendImage.cols, blendImage.rows, -1LL);
        ok = writer.write(frame);

        if (!ok)
        {
            printf("write fail\n");
            break;
        }

        count++;

		if (&progressDialog && (count % step == 0))
		{
			//printf("................%f...............\n", double(count) / (validFrameCount > 0 ? validFrameCount : 100) * 100);
			progressDialog->setValue(double(count) / (validFrameCount > 0 ? validFrameCount : 100) * 100);
			QApplication::processEvents();
		}

      //  if (progressCallbackFunc && (count % step == 0))
       //     progressCallbackFunc(double(count) / (validFrameCount > 0 ? validFrameCount : 100), progressCallbackData);
    }

    for (int i = 0; i < numVideos; i++)
        readers[i].close();
    writer.close();

    //if (progressCallbackFunc)
    //    progressCallbackFunc(1.0, progressCallbackData);

    printf("Info in %s, write video finish\n", __FUNCTION__);
    finish = true;
}

void QtCPUPanoramaLocalDiskTask::cancel()
{
    endFlag = true;
}

void QtCPUPanoramaLocalDiskTask::clear()
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

    validFrameCount = 0;

    initSuccess = false;
    finish = true;
}

QtCudaPanoramaLocalDiskTask::QtCudaPanoramaLocalDiskTask()
{
    clear();
}

QtCudaPanoramaLocalDiskTask::~QtCudaPanoramaLocalDiskTask()
{
    clear();
}

bool QtCudaPanoramaLocalDiskTask::init(const std::vector<std::string>& srcVideoFiles, const std::vector<int> offsets,
    const std::string& cameraParamFile, const std::string& dstVideoFile, int dstWidth, int dstHeight, int dstVideoBitRate)
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
    ok = prepareSrcVideos(srcVideoFiles, false, offsets, readers, srcSize, validFrameCount);
    if (!ok)
    {
        printf("Error in %s, could not open video file(s)\n", __FUNCTION__);
        return false;
    }

    printf("Info in %s, open videos done\n", __FUNCTION__);
    printf("Info in %s, prepare for reproject and blend\n", __FUNCTION__);

    std::vector<cv::Mat> dstMasks;
    std::vector<cv::Mat> dstSrcMaps, xmaps, ymaps;
    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, dstMasks);

    xmapsGpu.resize(numVideos);
    ymapsGpu.resize(numVideos);
    cv::Mat map32F;
    cv::Mat map64F[2];
    for (int i = 0; i < numVideos; i++)
    {
        cv::split(dstSrcMaps[i], map64F);
        map64F[0].convertTo(map32F, CV_32F);
        xmapsGpu[i].upload(map32F);
        map64F[1].convertTo(map32F, CV_32F);
        ymapsGpu[i].upload(map32F);
    }
    dstSrcMaps.clear();
    map32F.release();
    map64F[0].release();
    map64F[1].release();

    ok = blender.prepare(dstMasks, 16, 2);
    //ok = blender.prepare(dstMasks, 50);
    if (!ok)
    {
        printf("Error in %s, blender prepare failed\n", __FUNCTION__);
        return false;
    }
    dstMasks.clear();

    streams.resize(numVideos);
    pinnedMems.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
        pinnedMems[i].create(srcSize, CV_8UC4);

    imagesGpu.resize(numVideos);
    reprojImagesGpu.resize(numVideos);

    printf("Info in %s, prepare finish\n", __FUNCTION__);

    printf("Info in %s, open dst video\n", __FUNCTION__);
    std::vector<avp::Option> options;
    options.push_back(std::make_pair("preset", "medium"));
    ok = writer.open(dstVideoFile, "", false, false, "", avp::SampleTypeUnknown, 0, 0, 0,
        true, "h264_qsv", avp::PixelTypeBGR32, dstSize.width, dstSize.height, readers[0].getVideoFps(), dstVideoBitRate, options);
    if (!ok)
    {
        printf("Error in %s, video writer open failed\n", __FUNCTION__);
        return false;
    }
    else
        printf("Info in %s, video writer open success\n", __FUNCTION__);

    initSuccess = true;
    finish = false;
    return true;
}

void QtCudaPanoramaLocalDiskTask::run(QWidget* obj)
{
    if (!initSuccess)
        return;

    if (finish)
        return;

    printf("Info in %s, write video begin\n", __FUNCTION__);

    QProgressDialog* progressDialog = new QProgressDialog(obj);
    progressDialog->setMinimumSize(400, 50);
    progressDialog->resize(QSize(400, 50));
    progressDialog->setModal(true);
    progressDialog->setWindowModality(Qt::WindowModal);
    progressDialog->setWindowTitle(QObject::tr("Generating Panorama Video ......"));
    progressDialog->setRange(0, 100);
    progressDialog->setStyleSheet("background:rgb(88,88,88)");
    progressDialog->setValue(0);
    progressDialog->show();
    QCoreApplication::processEvents();

    finishPercent.store(0);

    decodedImagesOwnedByDecodeThread = true;
    encodedImageOwnedByProcThread = true;
    isCanceled = false;

	decodeThread.reset(new std::thread(&QtCudaPanoramaLocalDiskTask::decode, this));
    procThread.reset(new std::thread(&QtCudaPanoramaLocalDiskTask::proc, this));
    encodeThread.reset(new std::thread(&QtCudaPanoramaLocalDiskTask::encode, this));

    bool runCancel = false;
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        int p = finishPercent.load();
        progressDialog->setValue(p);
        if (p >= 100)
            break;
        if (!runCancel && progressDialog->wasCanceled())
        {
            cancel();
            runCancel = true;
        }
    }

    decodeThread->join();
    procThread->join();
    encodeThread->join();

    for (int i = 0; i < numVideos; i++)
        readers[i].close();
    writer.close();

    //if (progressCallbackFunc)
    //    progressCallbackFunc(1.0, progressCallbackData);

    printf("Info in %s, write video finish\n", __FUNCTION__);

    finish = true;
}

void QtCudaPanoramaLocalDiskTask::clear()
{
    numVideos = 0;
    srcSize = cv::Size();
    dstSize = cv::Size();
    readers.clear();
    xmapsGpu.clear();
    ymapsGpu.clear();
    streams.clear();
    pinnedMems.clear();
    imagesGpu.clear();
    reprojImagesGpu.clear();
    writer.close();

    videoEnd = false;
    procEnd = false;

    decodeCount = 0;
    procCount = 0;
    encodeCount = 0;

    validFrameCount = 0;

    decodedImagesOwnedByDecodeThread = true;
    encodedImageOwnedByProcThread = true;

    if (decodeThread && decodeThread->joinable())
        decodeThread->join();
    decodeThread.reset(0);
    if (procThread && procThread->joinable())
        procThread->join();
    procThread.reset(0);
    if (encodeThread && encodeThread->joinable())
        encodeThread->join();
    encodeThread.reset(0);

    initSuccess = false;
    finish = true;
    isCanceled = false;
}

void QtCudaPanoramaLocalDiskTask::cancel()
{
    isCanceled = true;
}

void QtCudaPanoramaLocalDiskTask::decode()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    decodeCount = 0;
#if ENABLE_CALC_TIME
    ztool::Timer timer;
#endif
    while (true)
    {
#if ENABLE_CALC_TIME
        timer.start();
#endif
        {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
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

    printf("total decode %d\n", decodeCount);
    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void QtCudaPanoramaLocalDiskTask::proc()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    procCount = 0;
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
            std::this_thread::sleep_for(std::chrono::microseconds(50));
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

    printf("total proc %d\n", procCount);
    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void QtCudaPanoramaLocalDiskTask::encode()
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
    //cv::Mat smallImage;
#if ENABLE_CALC_TIME
    ztool::Timer timer;
#endif
    while (true)
    {
#if ENABLE_CALC_TIME
        timer.start();
#endif
        {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
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

			if (encodeCount % step == 0)
			{
				//printf("................%f...............\n", double(encodeCount) / (validFrameCount > 0 ? validFrameCount : 100) * 100);
				int value = double(encodeCount) / (validFrameCount > 0 ? validFrameCount : 100) * 100;
                finishPercent.store(value);
			}
            //printf("frame %d finish, encode time = %f\n", count, timerEncode.elapse());
            //printf("encode frame count = %d\n", encodeCount);

           // if (progressCallbackFunc && (encodeCount % step == 0))
            //    progressCallbackFunc(double(encodeCount) / (validFrameCount > 0 ? validFrameCount : 100), progressCallbackData);

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
    finishPercent.store(100);

    printf("total encode %d\n", encodeCount);
    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

