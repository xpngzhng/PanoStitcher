#include "ZBlend.h"
#include "ZReproject.h"
#include "AudioVideoProcessor.h"
#include "StampedFrameQueue.h"
#include "PinnedMemoryPool.h"
#include "SharedAudioVideoFramePool.h"
#include "Timer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/gpumat.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>

#define ENABLE_CALC_TIME 0

struct StampedPinnedMemoryVector
{
    std::vector<cv::gpu::CudaMem> mems;
    long long int timeStamp;
};

typedef BoundedCompleteQueue<avp::SharedAudioVideoFrame> FrameBuffer;
typedef BoundedCompleteQueue<StampedPinnedMemoryVector> FrameVectorBuffer;

int numVideos;
cv::Size srcSize, dstSize;
std::vector<avp::AudioVideoReader> readers;
std::vector<cv::gpu::GpuMat> dstMasksGpu;
std::vector<cv::gpu::GpuMat> xmapsGpu, ymapsGpu;
CudaTilingMultibandBlendFast blender;
std::vector<cv::gpu::Stream> streams;
std::vector<cv::gpu::CudaMem> pinnedMems;
std::vector<cv::gpu::GpuMat> imagesGpu, reprojImagesGpu;
PinnedMemoryPool srcFramesMemoryPool;
SharedAudioVideoFramePool dstFramesMemoryPool;
FrameVectorBuffer decodeFramesBuffer(2);
FrameBuffer procFrameBuffer(4);
ztool::Timer timerAll, timerTotal, timerDecode, timerUpload, timerReproject, timerBlend, timerDownload, timerEncode;
int frameCount;
int maxFrameCount = 600;
int actualWriteFrame = 0;
avp::AudioVideoWriter2 writer;
bool success;
bool videoEnd = false;

void parseVideoPathsAndOffsets(const std::string& infoFileName, std::vector<std::string>& videoPath, std::vector<int>& offset)
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
    int count = 0;
    std::vector<avp::AudioVideoFrame> shallowFrames(numVideos);
#if ENABLE_CALC_TIME
    ztool::Timer timer;
#endif
    while (true)
    {

        bool successRead = true;
        for (int i = 0; i < numVideos; i++)
        {
            if (!readers[i].read(shallowFrames[i]))
            {
                successRead = false;
                break;
            }
        }
        if (!successRead || count >= maxFrameCount)
        {
            videoEnd = true;
            break;
        }
#if ENABLE_CALC_TIME
        timer.start();
#endif
        //std::vector<avp::SharedAudioVideoFrame> deepFrames(numVideos);
        //for (int i = 0; i < numVideos; i++)
        //    deepFrames[i] = shallowFrames[i];
        StampedPinnedMemoryVector deepFrames;
        deepFrames.timeStamp = shallowFrames[0].timeStamp;
        deepFrames.mems.resize(numVideos);
        for (int i = 0; i < numVideos; i++)
        {
            srcFramesMemoryPool.get(deepFrames.mems[i]);
            cv::Mat src(shallowFrames[i].height, shallowFrames[i].width, CV_8UC4, shallowFrames[i].data, shallowFrames[i].step);
            cv::Mat dst = deepFrames.mems[i];
            src.copyTo(dst);
        }
#if ENABLE_CALC_TIME
        timer.end();
        printf("d cp %f\n", timer.elapse());
#endif
        decodeFramesBuffer.push(deepFrames);        
        count++;
    }

    while (decodeFramesBuffer.size())
        std::this_thread::sleep_for(std::chrono::microseconds(25));
    decodeFramesBuffer.stop();

    printf("decode thread end\n");
}

void gpuProcThread()
{
    int count = 0;
    //std::vector<avp::SharedAudioVideoFrame> deepFrames(numVideos);
    StampedPinnedMemoryVector deepFrames;
    cv::gpu::GpuMat blendImageGpu;
    //cv::Mat blendImageCpu;
    //cv::Mat transImage;
#if ENABLE_CALC_TIME
    ztool::Timer timer;
    double t;
#endif
    while (true)
    {
        if (!decodeFramesBuffer.pull(deepFrames))
            break;

#if ENABLE_CALC_TIME
        timer.start();
#endif
        //for (int i = 0; i < numVideos; i++)
        //{
        //    cv::Mat srcMat(deepFrames[i].height, deepFrames[i].width, CV_8UC4, deepFrames[i].data, deepFrames[i].step);
        //    cv::Mat dstMat(pinnedMems[i]);
        //    srcMat.copyTo(dstMat);
        //    //cv::imshow("dst", dstMat);
        //    //cv::waitKey(30);
        //    //cv::transpose(srcMat, dstMat);
        //    //cv::transpose(srcMat, transImage);
        //    //cv::imshow("trans", transImage);
        //    //cv::waitKey(10);
        //    //cv::flip(transImage, dstMat, 1);
        //    //cv::imshow("flip", transImage);
        //    //cv::waitKey(20);
        //}
#if ENABLE_CALC_TIME
        timer.end();
        t = timer.elapse();
#endif
        avp::SharedAudioVideoFrame deepFrame;
        dstFramesMemoryPool.get(deepFrame);
        cv::Mat blendImageCpu(dstSize, CV_8UC4, deepFrame.data, deepFrame.step);
        
        for (int i = 0; i < numVideos; i++)
            streams[i].enqueueUpload(deepFrames.mems[i]/*pinnedMems[i]*/, imagesGpu[i]);
        for (int i = 0; i < numVideos; i++)
            cudaReprojectTo16S(imagesGpu[i], reprojImagesGpu[i], xmapsGpu[i], ymapsGpu[i], streams[i]);
        for (int i = 0; i < numVideos; i++)
            streams[i].waitForCompletion();

        blender.blend(reprojImagesGpu, blendImageGpu);
        blendImageGpu.download(blendImageCpu);

        //cv::Mat tmp16S, tmp8U;
        //for (int i = 0; i < 1; i++)
        //{
        //    reprojImagesGpu[i].download(tmp16S);
        //    tmp16S.convertTo(tmp8U, CV_8U);
        //    cv::imshow("rep", tmp8U);
        //    cv::waitKey(30);
        //}
        //cv::imshow("blend", blendImageCpu);
        //cv::waitKey(30);

#if ENABLE_CALC_TIME
        timer.start();
#endif
        //avp::AudioVideoFrame shallowFrame = 
        //    avp::videoFrame(blendImageCpu.data, blendImageCpu.step, avp::PixelTypeBGR32, blendImageCpu.cols, blendImageCpu.rows, -1LL);
        //avp::SharedAudioVideoFrame deepFrame = shallowFrame;
#if ENABLE_CALC_TIME
        timer.end();
        printf("p cp %f %f\n", t, timer.elapse());
#endif
        procFrameBuffer.push(deepFrame);

        count++;
    }

    while (procFrameBuffer.size())
        std::this_thread::sleep_for(std::chrono::microseconds(25));
    procFrameBuffer.stop();
    
    printf("gpu proc thread end\n");
}

void encodeThread()
{
    int count = 0;
    actualWriteFrame = 0;
    avp::SharedAudioVideoFrame deepFrame;
    cv::Mat smallImage;
    while (true)
    {
        if (!procFrameBuffer.pull(deepFrame))
            break;

        //cv::Mat blendImage(deepFrame.height, deepFrame.width, CV_8UC4, deepFrame.data, deepFrame.step);
        //cv::resize(blendImage, smallImage, cv::Size(), 0.25, 0.25, cv::INTER_NEAREST);
        //cv::imshow("preview", smallImage);
        //cv::waitKey(1);

        timerEncode.start();
        writer.write(avp::AudioVideoFrame(deepFrame));
        timerEncode.end();
        count++;
        printf("frame %d finish, encode time = %f\n", count, timerEncode.elapse());
        actualWriteFrame++;
    }

    printf("encode thread end\n");
}

int main(int argc, char* argv[])
{
    //cv::Mat src(1440, 1920, CV_8UC4);
    //ztool::Timer timer;
    //int numDeeps = 6;
    //timer.start();
    //for (int i = 0; i < 100; i++)
    //{
    //    avp::AudioVideoFrame shallow = avp::videoFrame(src.data, src.step, avp::PixelTypeBGR32, src.cols, src.rows, -1LL);
    //    //avp::SharedAudioVideoFrame deep = shallow;
    //    std::vector<avp::SharedAudioVideoFrame> deeps(numDeeps);
    //    for (int j = 0; j < numDeeps; j++)
    //        deeps[j] = shallow;
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
        "{h | pano_video_name | panoptslibx264_4k.mp4 | xml param file path}"
        "{g | pano_video_num_frames | 1000 | number of frames to write}";

    cv::CommandLineParser parser(argc, argv, keys);

    
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
    //pi.rotateCamera(0, 0, 3.14159265359 / 2);
    //pi.rotateCamera(0, -35.264 / 180 * 3.1415926536, -3.1415926536 / 4);
    //pi.rotateCamera(0, 3.1415926536 / 2 * 0.65, 0);

    std::vector<PhotoParam> params;
    if (ext == "pts")
        loadPhotoParamFromPTS(cameraParamFile, params);
    else
        loadPhotoParamFromXML(cameraParamFile, params);
    //rotateCameras(params, 0, 0, -3.1415926535898 / 2);

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
        avp::AudioVideoFrame image;
        for (int j = 0; j < count; j++)
            readers[i].read(image);
    }
    if (!ok)
    {
        printf("Could not open video file(s)\n");
        return 0;
    }

    ok = srcFramesMemoryPool.init(readers[0].getVideoHeight(), readers[0].getVideoWidth(), CV_8UC4);
    if (!ok)
    {
        printf("Could not init memory pool\n");
        return 0;
    }

    printf("Open videos done\n");
    printf("Prepare for reproject and blend\n");

    srcSize.width = readers[0].getVideoWidth();
    srcSize.height = readers[0].getVideoHeight();
    //std::swap(srcSize.width, srcSize.height);

    std::vector<cv::Mat> dstMasks;
    std::vector<cv::Mat> dstSrcMaps, xmaps, ymaps;
    //getReprojectMapsAndMasks(pi, srcSize, dstSrcMaps, dstMasks);
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

    ok = dstFramesMemoryPool.initAsVideoFramePool(avp::PixelTypeBGR32, dstSize.width, dstSize.height);
    if (!ok)
    {
        printf("Could not init memory pool\n");
        return 0;
    }

    printf("Prepare finish, begin stitching.\n");

    panoVideoName = parser.get<std::string>("pano_video_name");
    std::vector<avp::Option> options;
    options.push_back(std::make_pair("preset", "medium"));
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
    return 0;
}