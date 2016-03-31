#include "ZBlend.h"
#include "ZReproject.h"
#include "RicohUtil.h"
#include "AudioVideoProcessor.h"
#include "StampedFrameQueue.h"
#include "PinnedMemoryPool.h"
#include "SharedAudioVideoFramePool.h"
#include "Timer.h"

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui.hpp"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>

struct StampedPinnedMemoryVector
{
    std::vector<cv::cuda::HostMem> frames;
    long long int timeStamp;
};

typedef BoundedCompleteQueue<avp::SharedAudioVideoFrame> FrameBuffer;
typedef BoundedCompleteQueue<StampedPinnedMemoryVector> FrameVectorBuffer;

int numVideos;
int audioIndex;
cv::Size srcSize, dstSize;
std::vector<avp::AudioVideoReader> readers;
CudaMultiCameraPanoramaRender2 render;
PinnedMemoryPool srcFramesMemoryPool;
SharedAudioVideoFramePool audioFramesMemoryPool, dstFramesMemoryPool;
FrameVectorBuffer decodeFramesBuffer(2);
FrameBuffer procFrameBuffer(4);
ztool::Timer timerAll, timerTotal, timerDecode, timerUpload, timerReproject, timerBlend, timerDownload, timerEncode;
int frameCount;
int maxFrameCount = 600;
int actualWriteFrame = 0;
avp::AudioVideoWriter2 writer;
bool success;

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
        if (!successRead || count >= maxFrameCount)
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
    StampedPinnedMemoryVector srcFrames;
    avp::SharedAudioVideoFrame dstFrame;
    std::vector<cv::Mat> images(numVideos);
    ztool::Timer t, gput;
    while (true)
    {
        if (!decodeFramesBuffer.pull(srcFrames))
            break;
        
        t.start();
        dstFramesMemoryPool.get(dstFrame);
        for (int i = 0; i < numVideos; i++)
            images[i] = srcFrames.frames[i].createMatHeader();
        cv::Mat result(dstSize, CV_8UC4, dstFrame.data, dstFrame.step);
        gput.start();
        render.render(images, result);
        gput.end();
        dstFrame.timeStamp = srcFrames.timeStamp;
        procFrameBuffer.push(dstFrame);
        t.end();
        printf("gpu time = %f, proc time = %f\n", gput.elapse(), t.elapse());

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
    while (true)
    {
        if (!procFrameBuffer.pull(deepFrame))
            break;

        timerEncode.start();
        writer.write(avp::AudioVideoFrame(deepFrame));
        timerEncode.end();
        if (deepFrame.mediaType == avp::VIDEO)
        {
            //count++;
            //printf("frame %d finish, encode time = %f\n", count, timerEncode.elapse());
        }
        actualWriteFrame++;
    }

    printf("encode thread end\n");
}

int main(int argc, char* argv[])
{
    const char* keys =
        "{camera_param_file      |      | camera param file path}"
        "{video_path_offset_file |      | video path and offset file path}"
        "{num_frames_skip        | 100  | number of frames to skip}"
        "{pano_width             | 2048 | pano picture width}"
        "{pano_height            | 1024 | pano picture height}"
        "{pano_video_name        | panoptslibx264withaudiolinear.mp4 | xml param file path}"
        "{pano_video_num_frames  | 1000 | number of frames to write}"
        "{audio_index            | -1    | select audio from which video}";

    cv::CommandLineParser parser(argc, argv, keys);
    
    std::vector<std::string> srcVideoNames;
    std::vector<int> offset;
    int numSkip = 1500;
    std::string cameraParamFile, videoPathAndOffsetFile;
    std::string panoVideoName;

    cameraParamFile = parser.get<std::string>("camera_param_file");
    if (cameraParamFile.empty())
    {
        printf("Could not find camera_param_file\n");
        return 0;
    }

    dstSize.width = parser.get<int>("pano_width");
    dstSize.height = parser.get<int>("pano_height");

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

    bool openAudio = false;
    audioIndex = parser.get<int>("audio_index");
    if (audioIndex >= numVideos)
    {
        printf("Warning, audio index out of bound, no audio in output file\n");
        audioIndex = -1;
    }
    if (audioIndex >= 0 && audioIndex < numVideos)
        openAudio = true;

    bool ok = false;
    readers.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        ok = readers[i].open(srcVideoNames[i], i == audioIndex, true, avp::PixelTypeBGR32);
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

    if (openAudio)
    {
        ok = audioFramesMemoryPool.initAsAudioFramePool(readers[audioIndex].getAudioSampleType(),
            readers[audioIndex].getAudioNumChannels(), readers[audioIndex].getAudioChannelLayout(),
            readers[audioIndex].getAudioNumSamples());
        if (!ok)
        {
            printf("Could not init memory pool\n");
            return 0;
        }
    }

    printf("Open videos done\n");
    printf("Prepare for reproject and blend\n");

    srcSize.width = readers[0].getVideoWidth();
    srcSize.height = readers[0].getVideoHeight();

    success = render.prepare(cameraParamFile, PanoramaRender::BlendTypeMultiband, srcSize, dstSize);
    if (!success)
    {
        printf("Blender prepare failed, exit.\n");
        return 0;
    }

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
    if (openAudio)
    {
        success = writer.open(panoVideoName, "", true, true, "aac", readers[audioIndex].getAudioSampleType(), 
            readers[audioIndex].getAudioChannelLayout(), readers[audioIndex].getAudioSampleRate(), 128000,
            true, "h264", avp::PixelTypeBGR32, dstSize.width, dstSize.height, readers[0].getVideoFps(), 48000000, options);
    }
    else
    {
        success = writer.open(panoVideoName, "", false, false, "", avp::SampleTypeUnknown, 0, 0, 0,
            true, "h264", avp::PixelTypeBGR32, dstSize.width, dstSize.height, readers[0].getVideoFps(), 48000000, options);
    }
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
    
    return 0;
}