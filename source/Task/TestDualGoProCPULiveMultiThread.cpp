#include "AudioVideoProcessor.h"
#include "StampedFrameQueue.h"
#include "RicohUtil.h"
#include "Timer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <thread>
#include <string>
#include <atomic>
#include <chrono>

typedef RealTimeQueue<std::vector<StampedFrame> > StampedFrameVectorQueue;

cv::Size dstSize(1440, 720);
cv::Size srcSize(1920, 1080);
int bitRate, frameRate;
std::string ptsPath, url;

const int numVideos = 2;
std::vector<CompleteStampedFrameQueue> frameBuffers(numVideos);
StampedFrameVectorQueue syncedFramesBufferForShow, syncedFramesBufferForProc;
StampedFrameQueue procFrameBufferForShow, procFrameBufferForSend;
std::vector<avp::VideoReader> readers(numVideos);
//avp::VideoWriter writer;
avp::AudioVideoSender writer;
DualGoProPanoramaRender render;

int finish = 0;

// This variable controls that we check whether frame rate matches the set one
// after first time of synchronization has finished.
int checkFrameRate = 0;

inline void stopCompleteFrameBuffers()
{
    for (int i = 0; i < numVideos; i++)
        frameBuffers[i].stop();        
}

void mediaSource(int index)
{
    size_t id = std::this_thread::get_id().hash();
    CompleteStampedFrameQueue& buffer = frameBuffers[index];
    avp::VideoReader& reader = readers[index];

    long long int count = 0, beginCheckCount = frameRate * 5;
    ztool::Timer timer;
    avp::Frame frame;
    bool ok;
    while (true)
    {
        ok = reader.read(frame);
        if (!ok)
        {
            printf("Error in %s [%8x], cannot read frame\n", __FUNCTION__, id);
            //buffer.stop();
            stopCompleteFrameBuffers();
            finish = 1;
            break;
        }

        count++;
        if (count == beginCheckCount)
            timer.start();
        if ((count > beginCheckCount) && (count % frameRate == 0))
        {
            timer.end();
            double actualFps = (count - beginCheckCount - 1) / timer.elapse();
            printf("[%8x] fps = %f\n", id, actualFps);
            if (abs(actualFps - frameRate) > 2 && checkFrameRate)
            {
                printf("Error in %s [%8x], fps far away from the set one\n", __FUNCTION__, id);
                //buffer.stop();
                stopCompleteFrameBuffers();
                finish = 1;
                break;
            }
        }

        cv::Mat shallow(frame.height, frame.width, CV_8UC3, frame.data, frame.step);
        cv::Mat deep;
        if (shallow.size() == srcSize)
            deep = shallow.clone();
        else
        {
            cv::Rect intersectRect(0, 0, std::min(srcSize.width, frame.width), std::min(srcSize.height, frame.height));
            deep.create(srcSize, CV_8UC3);
            cv::Mat deepPart = deep(intersectRect);
            cv::Mat shallowPart = shallow(intersectRect);
            shallowPart.copyTo(deepPart);
        }
        buffer.push(StampedFrame(deep, frame.timeStampInMilliSec));

        if (finish)
        {
            //buffer.stop();
            stopCompleteFrameBuffers();
            finish = 1;
            break;
        }
    }
}

void mediaSink()
{
    size_t id = std::this_thread::get_id().hash();
    std::this_thread::sleep_for(std::chrono::microseconds(100));

    if (finish)
        return;

    for (int i = 0; i < numVideos; i++)
        printf("size = %d\n", frameBuffers[i].size());
    
    for (int j = 0; j < 25; j++)
    {
        for (int i = 0; i < numVideos; i++)
        {
            StampedFrame stampedFrame;
            frameBuffers[i].pull(stampedFrame);
        }
    }

    if (finish)
        return;

    long long int currMaxTS = -1;
    int currMaxIndex = -1;
    for (int i = 0; i < numVideos; i++)
    {
        StampedFrame stampedFrame;
        frameBuffers[i].pull(stampedFrame);
        if (stampedFrame.timeStamp < 0)
        {
            printf("Error in %s [%8x], cannot read valid frame with non-negative time stamp\n", __FUNCTION__, id);
            finish = 1;
            break;
        }
        if (stampedFrame.timeStamp > currMaxTS)
        {
            currMaxIndex = i;
            currMaxTS = stampedFrame.timeStamp;
        }
    }

    if (finish)
        return;

    std::vector<StampedFrame> syncedFrames(numVideos);
    StampedFrame slowestFrame;
    frameBuffers[currMaxIndex].pull(slowestFrame);
    syncedFrames[currMaxIndex] = slowestFrame;
    printf("slowest ts = %lld\n", slowestFrame.timeStamp);
    for (int i = 0; i < numVideos; i++)
    {
        if (finish)
            break;

        if (i == currMaxIndex)
            continue;

        StampedFrame stampedFrame;
        while (true)
        {
            frameBuffers[i].pull(stampedFrame);
            printf("this ts = %lld\n", stampedFrame.timeStamp);
            /*if (stampedFrame.timeStamp < 0)
            {
                printf("Error in %s [%8x], cannot read valid frame with non-negative time stamp\n", __FUNCTION__, id);
                finish = 1;
                break;
            }*/
            if (stampedFrame.timeStamp > slowestFrame.timeStamp)
            {
                syncedFrames[i] = stampedFrame;
                printf("break\n");
                break;
            }
        }
    }
    if (finish)
        return;

    syncedFramesBufferForShow.push(syncedFrames);
    syncedFramesBufferForProc.push(syncedFrames);

    checkFrameRate = 1;

    std::vector<StampedFrame> frames(numVideos);
    while (true)
    {
        if (finish)
            break;

        for (int i = 0; i < numVideos; i++)
        {
            frameBuffers[i].pull(frames[i]);
            //printf("%d ", frameBuffers[i].size());
        }
        //printf("\n");
        syncedFramesBufferForShow.push(frames);
        syncedFramesBufferForProc.push(frames);
    }
}

void showSource()
{
    std::vector<StampedFrame> frames;
    while (true)
    {
        if (finish)
            break;
        //printf("show\n");
        syncedFramesBufferForShow.pull(frames);
        if (frames.size() == 2)
        {
            cv::imshow("frame 1", frames[0].frame);
            cv::imshow("frame 2", frames[1].frame);
            printf("%lld, %lld\n", frames[0].timeStamp, frames[1].timeStamp);
            cv::waitKey(25);
        }
    }
}

void proc()
{
    std::vector<StampedFrame> frames;
    cv::Mat result;
    while (true)
    {
        if (finish)
            break;
        //printf("show\n");
        syncedFramesBufferForShow.pull(frames);
        if (frames.size() == 2)
        {
            render.render(frames[0].frame, frames[1].frame, result);
            //cv::imshow("result", result);
            //cv::waitKey(1);
            cv::Mat deep = result.clone();
            procFrameBufferForShow.push(deep);
            procFrameBufferForSend.push(deep);
        }
    }
}

void showResult()
{
    StampedFrame frame;
    while (true)
    {
        if (finish)
            break;
        procFrameBufferForShow.pull(frame);
        if (frame.frame.data)
        {
            cv::imshow("result", frame.frame);
            cv::waitKey(25);
        }
    }
}

void send()
{
    size_t id = std::this_thread::get_id().hash();
    StampedFrame frame;
    while (true)
    {
        if (finish)
            break;
        procFrameBufferForSend.pull(frame);
        if (frame.frame.data)
        {
            avp::BGRImage image(frame.frame.data, frame.frame.cols, frame.frame.rows, frame.frame.step);
            bool ok = writer.write(image);
            if (!ok)
            {
                printf("Error in %s [%8x], cannot write frame\n", __FUNCTION__, id);
                finish = 1;
                break;
            }                
        }
    }
}

static void keepOnlyVideoDevices(std::vector<avp::Device>& devices)
{
    if (devices.empty())
        return;

    int numVideoDevices = 0;
    int numDevices = devices.size();
    for (int i = 0; i < numDevices; i++)
    {
        if (devices[i].deviceType == avp::VIDEO)
        {
            if (numDevices != i)
                devices[numVideoDevices] = devices[i];
            numVideoDevices++;
        }
    }
    devices.resize(numVideoDevices);
}

int main(int argc, char* argv[])
{
    const char* keys =
        "{a | camera_width | 1920 | camera picture width}"
        "{b | camera_height | 1080 | camera picture height}"
        "{c | frames_per_second | 30 | camera frame rate}"
        "{d | pano_width | 1440 | pano picture width}"
        "{e | pano_height | 720 | pano picture height}"
        "{f | pano_bits_per_second | 1000000 | pano live stream bits per second}"
        "{k | pano_encode_preset | veryfast | pano video x264 encode preset}"
        "{g | pano_url | rtmp://pili-publish.live.detu.com/detulive/detudemov550?key=detukey | pano live stream address}"
        "{h | pts_path | dualgopro.pts | ptgui pts file path}";

    cv::CommandLineParser parser(argc, argv, keys);

    srcSize.width = parser.get<int>("camera_width");
    srcSize.height = parser.get<int>("camera_height");
    dstSize.width = parser.get<int>("pano_width");
    dstSize.height = parser.get<int>("pano_height");
    if (dstSize.width <= 0 || dstSize.height <= 0 ||
        (dstSize.width & 1) || (dstSize.height & 1) ||
        (dstSize.width != dstSize.height * 2))
    {
        printf("pano_width and pano_height should be positive even numbers and pano_width should be two times of pano_height\n");
        return 0;
    }
    ptsPath = parser.get<std::string>("pts_path");
    bool ok;
    ok = render.prepare(ptsPath, srcSize, dstSize);
    if (!ok)
    {
        printf("Could not prepare for panorama render\n");
        return 0;
    }

    url = parser.get<std::string>("pano_url");
    frameRate = parser.get<int>("frames_per_second");
    bitRate = parser.get<int>("pano_bits_per_second");
    std::string preset = parser.get<std::string>("pano_encode_preset");
    int speed = avp::EncodeSpeedVeryFast;
    if (preset == "ultrafast")
        speed = avp::EncodeSpeedUltraFast;
    else if (preset == "superfast")
        speed = avp::EncodeSpeedSuperFast;
    else if (preset == "veryfast")
        speed = avp::EncodeSpeedVeryFast;
    else if (preset == "faster")
        speed = avp::EncodeSpeedFaster;
    else if (preset == "fast")
        speed = avp::EncodeSpeedFast;
    else if (preset == "medium")
        speed = avp::EncodeSpeedMedium;
    else if (preset == "slow")
        speed = avp::EncodeSpeedSlow;
    else if (preset == "slower")
        speed = avp::EncodeSpeedSlower;
    else if (preset == "veryslow")
        speed = avp::EncodeSpeedVerySlow;
    ok = writer.open(url, avp::PixelTypeBGR24, dstSize.width, dstSize.height, frameRate, bitRate, speed);
    if (!ok)
    {
        printf("Could not open rtmp streaming url with frame rate = %d and bit rate = %d\n", frameRate, bitRate);
        return 0;
    }

    std::vector<avp::Device> devices;
    avp::listDirectShowDevices(devices);
    keepOnlyVideoDevices(devices);
    if (devices.size() < 2)
    {
        printf("Not enough DirectShow video devices found\n");
        return 0;
    }

    std::vector<avp::Option> opts;
    std::string frameRateStr = parser.get<std::string>("frames_per_second");
    std::string videoSizeStr = parser.get<std::string>("camera_width") + "x" + parser.get<std::string>("camera_height");
    opts.push_back(std::make_pair("framerate", frameRateStr));
    opts.push_back(std::make_pair("video_size", videoSizeStr));
    
    if (devices[0].shortName == devices[1].shortName)
    {
        opts.push_back(std::make_pair("video_device_number", "0"));
        ok = readers[0].openDirectShowDevice("video=" + devices[0].shortName, avp::PixelTypeBGR24, opts);
        if (!ok)
        {
            printf("Could not open DirectShow video device %s[0] with framerate = %s and video_size = %s\n",
                devices[0].shortName.c_str(), frameRateStr.c_str(), videoSizeStr.c_str());
            return 0;
        }

        opts.resize(2);
        opts.push_back(std::make_pair("video_device_number", "1"));
        ok = readers[1].openDirectShowDevice("video=" + devices[1].shortName, avp::PixelTypeBGR24, opts);
        if (!ok)
        {
            printf("Could not open DirectShow video device %s[1] with framerate = %s and video_size = %s\n",
                devices[1].shortName.c_str(), frameRateStr.c_str(), videoSizeStr.c_str());
            return 0;
        }
    }
    else
    {
        ok = readers[0].openDirectShowDevice("video=" + devices[0].shortName, avp::PixelTypeBGR24, opts);
        if (!ok)
        {
            printf("Could not open DirectShow video device %s with framerate = %s and video_size = %s\n",
                devices[0].shortName.c_str(), frameRateStr.c_str(), videoSizeStr.c_str());
            return 0;
        }

        ok = readers[1].openDirectShowDevice("video=" + devices[1].shortName, avp::PixelTypeBGR24, opts);
        if (!ok)
        {
            printf("Could not open DirectShow video device %s with framerate = %s and video_size = %s\n",
                devices[1].shortName.c_str(), frameRateStr.c_str(), videoSizeStr.c_str());
            return 0;
        }
    }    

    /*std::string name1 = "video=XI100DUSB-HDMI Video";
    std::vector<avp::Option> options1;
    options1.push_back(std::make_pair("framerate", "30"));
    ok = readers[0].openDirectShowDevice(name1, avp::PixelTypeBGR24, options1);
    if (!ok)
    {
        printf("Could not open DirectShow divice\n");
        return 0;
    }

    std::string name2 = "video=J1455 ";
    std::vector<avp::Option> options2;
    options2.push_back(std::make_pair("framerate", "30"));
    options2.push_back(std::make_pair("video_size", "1024x768"));
    ok = readers[1].openDirectShowDevice(name2, avp::PixelTypeBGR24, options2);
    if (!ok)
    {
        printf("Could not open DirectShow divice\n");
        return 0;
    }*/

    std::thread t1(mediaSource, 0);
    std::thread t2(mediaSource, 1);
    std::thread t3(mediaSink);
    //std::thread t4(showSource);
    std::thread t5(proc);
    std::thread t6(showResult);
    std::thread t7(send);

    t1.join();
    t2.join();
    t3.join();
    //t4.join();
    t5.join();
    t6.join();
    t7.join();

    return 0;
}