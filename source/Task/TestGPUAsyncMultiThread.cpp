#include "ZBlend.h"
#include "ZReproject.h"
#include "AudioVideoProcessor.h"
#include "Timer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/gpumat.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>

#define ENABLE_CALC_TIME 0

int numVideos;
std::vector<avp::AudioVideoReader> readers;
std::vector<cv::gpu::GpuMat> dstMasksGpu;
std::vector<cv::gpu::GpuMat> xmapsGpu, ymapsGpu;
CudaTilingMultibandBlendFast blender;
//CudaTilingLinearBlend blender;
std::vector<cv::gpu::Stream> streams;
std::vector<cv::gpu::CudaMem> pinnedMems;
std::vector<cv::gpu::GpuMat> imagesGpu, reprojImagesGpu;
cv::gpu::GpuMat blendImageGpu;
cv::Mat blendImageCpu;
ztool::Timer timerAll, timerTotal, timerDecode, timerUpload, timerReproject, timerBlend, timerDownload, timerEncode;
int frameCount;
int maxFrameCount = 600;
int actualWriteFrame = 0;
avp::AudioVideoWriter2 writer;
bool success;

int decodeCount = 0;
int procCount = 0;
int encodeCount = 0;

std::mutex mtxDecodedImages;
std::condition_variable cvDecodedImagesForWrite, cvDecodedImagesForRead;
bool decodedImagesOwnedByDecodeThread;
bool videoEnd = false;

std::mutex mtxEncodedImage;
std::condition_variable cvEncodedImageForWrite, cvEncodedImageForRead;
bool encodedImageOwnedByProcThread;
bool procEnd = false;

static void parseVideoPathsAndOffsets(const std::string& infoFileName, std::vector<std::string>& videoPath, std::vector<int>& offset)
{
    videoPath.clear();
    offset.clear();

    std::ifstream fstrm(infoFileName);
    std::string line;
    while (!fstrm.eof())
    {
        std::getline(fstrm, line);
        if (line.empty())
            continue;

        std::string::size_type pos = line.find(',');
        if (pos == std::string::npos)
            continue;

        videoPath.push_back(line.substr(0, pos));
        offset.push_back(atoi(line.substr(pos + 1).c_str()));
    }
}

void decodeThread()
{
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
            cvDecodedImagesForWrite.wait(ul, []{return decodedImagesOwnedByDecodeThread;});
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
            
            if (!successRead || decodeCount >= maxFrameCount)
            {
                videoEnd = true;
                break;
            }

            // NOTICE!!!!!!
            // The following line had better be after break.
            // If not, before the lock at the end of the function, if proc() arrives at the first lock and wait,
            // it may find decodedImagesOwnedByDecodeThread == false, but procEnd == false, 
            // then the last frames will be reprojected and blends twice.
            // This file has not produces such result yet, but CPUPanoramaLocalDiskTask has. 
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
        cvDecodedImagesForWrite.wait(ul, []{return decodedImagesOwnedByDecodeThread; });
        decodedImagesOwnedByDecodeThread = false;
        videoEnd = true;
    }
    cvDecodedImagesForRead.notify_one();

    printf("decode thread end, %d\n", decodeCount);
}

void gpuProcThread()
{
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
            cvDecodedImagesForRead.wait(ul, []{return !decodedImagesOwnedByDecodeThread;});
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
            cvEncodedImageForWrite.wait(ul, []{return encodedImageOwnedByProcThread;});
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
        cvDecodedImagesForRead.wait(ul, []{return !decodedImagesOwnedByDecodeThread; });
        decodedImagesOwnedByDecodeThread = true;
    }
    cvDecodedImagesForWrite.notify_one();

    // Without the following if statement, encodeThread will not terminate.
    {
        std::unique_lock<std::mutex> ul(mtxEncodedImage);
        cvEncodedImageForWrite.wait(ul, []{return encodedImageOwnedByProcThread; });
        encodedImageOwnedByProcThread = false;
        procEnd = true;
    }
    cvEncodedImageForRead.notify_one();

    printf("gpu proc thread end, %d\n", procCount);
}

void encodeThread()
{
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
            std::unique_lock<std::mutex> ul(mtxEncodedImage);
            cvEncodedImageForRead.wait(ul, []{return !encodedImageOwnedByProcThread;});
            if (procEnd)
                break;

            //cv::resize(blendImageCpu, smallImage, cv::Size(), 0.25, 0.25, cv::INTER_NEAREST);
            //cv::imshow("preview", smallImage);
            //cv::waitKey(1);

            timerEncode.start();
            avp::AudioVideoFrame frame = avp::videoFrame(blendImageCpu.data, blendImageCpu.step, avp::PixelTypeBGR32, blendImageCpu.cols, blendImageCpu.rows, -1LL);
            writer.write(frame);
            timerEncode.end();
            
            encodeCount++;
            //printf("%d\n", encodeCount);
            //printf("%d, %d, %d\n", decodeCount, procCount, encodeCount);
            printf("frame %d finish, encode time = %f\n", encodeCount, timerEncode.elapse());

            encodedImageOwnedByProcThread = true;
        }
        cvEncodedImageForWrite.notify_one();
#if ENABLE_CALC_TIME
        timer.end();
        printf("e %f, %f\n", timerEncode.elapse(), timer.elapse());
#endif
    }
    printf("encode thread end, %d\n", encodeCount);
}

int main(int argc, char* argv[])
{
    //cv::Mat src(2048, 4096, CV_8UC4);
    //ztool::Timer timer;
    //for (int i = 0; i < 100; i++)
    //{
    //    cv::Mat dst;
    //    src.copyTo(dst);
    //}
    //timer.end();
    //printf("time = %f\n", timer.elapse());
    //return 0;

    //cv::Size dstSize = cv::Size(2048, 1024);
    //cv::Size srcSize = cv::Size(1280, 960);

    //ReprojectParam pi;
    //pi.LoadConfig("F:\\QQRecord\\452103256\\FileRecv\\test1\\changtai_cam_param.xml");
    //pi.SetPanoSize(dstSize);

    //std::vector<std::string> srcVideoNames;
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0078.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0081.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0087.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0108.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0118.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0518.mp4");
    //numVideos = srcVideoNames.size();

    //int offset[] = { 563, 0, 268, 651, 91, 412 };
    //int numSkip = /*200*//*1*/2100;

    //ReprojectParam pi;
    //pi.LoadConfig("F:\\panovideo\\test\\test1\\test_test1_cam_param.xml");
    //pi.SetPanoSize(dstSize);

    //std::vector<std::string> srcVideoNames;
    //srcVideoNames.push_back("F:\\panovideo\\test\\test1\\YDXJ0094.mp4");
    //srcVideoNames.push_back("F:\\panovideo\\test\\test1\\YDXJ0096.mp4");
    //srcVideoNames.push_back("F:\\panovideo\\test\\test1\\YDXJ0103.mp4");
    //srcVideoNames.push_back("F:\\panovideo\\test\\test1\\YDXJ0124.mp4");
    //srcVideoNames.push_back("F:\\panovideo\\test\\test1\\YDXJ0136.mp4");
    //srcVideoNames.push_back("F:\\panovideo\\test\\test1\\YDXJ0535.mp4");
    //numVideos = srcVideoNames.size();

    //int offset[] = { 0, 198, 246, 283, 144, 373 };
    //int numSkip = 200/*1*//*2100*/;

    //ReprojectParam pi;
    //pi.LoadConfig("F:\\QQRecord\\452103256\\FileRecv\\test2\\changtai.xml");
    //pi.SetPanoSize(frameSize);

    //std::vector<std::string> srcVideoNames;
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0072.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0075.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0080.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0101.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0112.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0512.mp4");
    //numVideos = srcVideoNames.size();

    //int offset[] = {554, 0, 436, 1064, 164, 785};
    //int numSkip = 3000;

    const char* keys =
        "{a | camera_param_file |  | camera param file path}"
        "{b | video_path_offset_file |  | video path and offset file path}"
        "{c | num_frames_skip | 100 | number of frames to skip}"
        "{d | pano_width | 2048 | pano picture width}"
        "{e | pano_height | 1024 | pano picture height}"
        "{h | pano_video_name | panoh264qsv_4k.mp4 | xml param file path}"
        "{g | pano_video_num_frames | 1000 | number of frames to write}";

    cv::CommandLineParser parser(argc, argv, keys);

    cv::Size srcSize, dstSize;
    std::vector<std::string> srcVideoNames;
    std::vector<int> offset;
    //ReprojectParam pi;
    int numSkip = 1500;
    std::string cameraParamFile, videoPathAndOffsetFile;
    std::string panoVideoName;

    cameraParamFile = parser.get<std::string>("camera_param_file");
    if (cameraParamFile.empty())
    {
        printf("Could not find camera_param_file\n");
        return 0;
    }
    std::string::size_type pos = cameraParamFile.find_last_of(".");
    std::string ext = cameraParamFile.substr(pos + 1);
    //pi.LoadConfig(cameraParamFile);
    //pi.rotateCamera(0, -35.264 / 180 * 3.1415926536, -3.1415926536 / 4);
    //pi.rotateCamera(0, 3.1415926536 / 2 * 0.65, 0);

    std::vector<PhotoParam> params;
    if (ext == "pts")
        loadPhotoParamFromPTS(cameraParamFile, params);
    else
        loadPhotoParamFromXML(cameraParamFile, params);

    dstSize.width = parser.get<int>("pano_width");
    dstSize.height = parser.get<int>("pano_height");
    //pi.SetPanoSize(dstSize);

    videoPathAndOffsetFile = parser.get<std::string>("video_path_offset_file");
    if (videoPathAndOffsetFile.empty())
    {
        printf("Could not find video_path_offset_file\n");
        return 0;
    }
    parseVideoPathsAndOffsets(videoPathAndOffsetFile, srcVideoNames, offset);
    if (srcVideoNames.empty() || offset.empty())
    {
        printf("Could not parse video path and offset\n");
        return 0;
    }
    numVideos = srcVideoNames.size();

    numSkip = parser.get<int>("num_frames_skip");
    if (numSkip < 0)
        numSkip = 0;

    printf("Open videos and set to the correct frames\n");

    bool ok = false;
    readers.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        ok = readers[i].open(srcVideoNames[i], false, true, avp::PixelTypeBGR32/*avp::PixelTypeBGR24*/);
        if (!ok)
            break;
        int count = offset[i] + numSkip;
        avp::AudioVideoFrame frame;
        for (int j = 0; j < count; j++)
            readers[i].read(frame);
    }
    if (!ok)
    {
        printf("Could not open video file(s)\n");
        return 0;
    }

    printf("Open videos done\n");
    printf("Prepare for reproject and blend\n");

    srcSize.width = readers[0].getVideoWidth();
    srcSize.height = readers[0].getVideoHeight();

    std::vector<cv::Mat> dstMasks;
    std::vector<cv::Mat> dstSrcMaps, xmaps, ymaps;
    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, dstMasks);

    dstMasksGpu.resize(numVideos);
    xmapsGpu.resize(numVideos);
    ymapsGpu.resize(numVideos);
    cv::Mat map32F;
    cv::Mat map64F[2];
    for (int i = 0; i < numVideos; i++)
    {
        dstMasksGpu[i].upload(dstMasks[i]);
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

    //for (int i = 0; i < numVideos; i++)
    //{
    //    cv::imshow("mask", dstMasks[i]);
    //    cv::waitKey(0);
    //}    

    success = blender.prepare(dstMasks, 16, 2);
    //success = blender.prepare(dstMasks, 50);
    if (!success)
    {
        printf("Blender prepare failed, exit.\n");
        return 0;
    }
    dstMasks.clear();
    //return 0;

    streams.resize(numVideos);
    pinnedMems.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
        pinnedMems[i].create(srcSize, CV_8UC4);

    imagesGpu.resize(numVideos);
    reprojImagesGpu.resize(numVideos);

    printf("Prepare finish, begin stitching.\n");

    panoVideoName = parser.get<std::string>("pano_video_name");
    std::vector<avp::Option> options;
    options.push_back(std::make_pair("preset", "medium"));
    //success = writer.open(panoVideoName, avp::PixelTypeBGR32, dstSize.width, dstSize.height, 24, 12000000, avp::EncodeSpeedSlow);
    success = writer.open(panoVideoName, "", false, false, "", avp::SampleTypeUnknown, 0, 0, 0,
        true, "h264", avp::PixelTypeBGR32, dstSize.width, dstSize.height, readers[0].getVideoFps(), 48000000, options);
    if (!success)
    {
        printf("Video writer open failed, exit.\n");
        return 0;
    }

    maxFrameCount = parser.get<int>("pano_video_num_frames");

    timerAll.start();

    frameCount = 0;   
    decodedImagesOwnedByDecodeThread = true;
    encodedImageOwnedByProcThread = true;
    std::thread thrdDecode(decodeThread);
    std::thread thrdGpuProc(gpuProcThread);
    std::thread thrdEncode(encodeThread);

    // IMPORTANT!!! Forget to join may cause the program to abort. 
    // I do not know why now!!
    thrdDecode.join();
    thrdGpuProc.join();
    thrdEncode.join();

    for (int i = 0; i < numVideos; i++)
        readers[i].close();
    writer.close();
    
    timerAll.end();
    printf("finish, %d frames written\n", actualWriteFrame);
    printf("time elapsed total = %f seconds, average process time per frame = %f second\n", 
        timerAll.elapse(), timerAll.elapse() / (actualWriteFrame ? actualWriteFrame : 1));

    dstMasksGpu.clear();
    xmapsGpu.clear();
    ymapsGpu.clear();
    pinnedMems.clear();
    imagesGpu.clear();
    reprojImagesGpu.clear();
    blendImageGpu.release();
    blendImageCpu.release();
    return 0;
}