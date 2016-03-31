#include "RicohUtil.h"
#include "AudioVideoProcessor.h"
#include "Timer.h"
#include "ConcurrentQueue.h"

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <thread>
#include <fstream>

cv::Size dstSize(2048, 1024);
cv::Size srcSize(1920, 1080);

std::string xmlPath;
RicohPanoramaRender render;

int frameCount = 0;
cv::Mat blendImage;

ztool::Timer timerAll, timerTotal, timerDecode, timerReproject, timerBlend, timerEncode;

avp::AudioVideoReader reader;
avp::AudioVideoWriter writer;

typedef RealTimeQueue<avp::SharedAudioVideoFrame> FrameQueue;
FrameQueue origQueue(36), displayQueue(36), writeQueue(36);

int waitTime = 30;
bool end = false;

void read()
{
    avp::AudioVideoFrame shallowFrame;
    while (true)
    {
        if (end)
            break;

        bool success = true;
        timerDecode.start();
        success = reader.read(shallowFrame);
        timerDecode.end();
        if (!success)
        {
            end = true;
            break;
        }
        /*if (shallowFrame.mediaType == avp::VIDEO)
        {
            cv::Mat showImage(srcSize, CV_8UC3, shallowFrame.data, shallowFrame.step);
            cv::imshow("orig", showImage);
            cv::waitKey(1);
        }*/
        origQueue.push(shallowFrame);
        //printf("orig queue size = %d\n", origQueue.size());
    }
}

void proc()
{
    avp::SharedAudioVideoFrame sharedFrame;
    while (!end)
    {
        origQueue.pull(sharedFrame);
        if (sharedFrame.data)
        {
            if (sharedFrame.mediaType == avp::VIDEO)
            {
                cv::Mat origImage(srcSize, CV_8UC3, sharedFrame.data, sharedFrame.step);
                render.render(origImage, blendImage);
                avp::AudioVideoFrame shallowFrame = avp::videoFrame(blendImage.data, blendImage.step, avp::PixelTypeBGR24, 
                    blendImage.cols, blendImage.rows, sharedFrame.timeStamp);
                avp::SharedAudioVideoFrame deepFrame(shallowFrame);
                displayQueue.push(deepFrame);
                writeQueue.push(deepFrame);
            }
            else if (sharedFrame.mediaType == avp::AUDIO)
            {
                writeQueue.push(sharedFrame);
            }
            //printf("write queue size = %d\n", writeQueue.size());
        }
    }
}

void display()
{
    avp::SharedAudioVideoFrame frame;
    while (!end)
    {
        displayQueue.pull(frame);
        if (frame.mediaType == avp::VIDEO && frame.data)
        {
            cv::Mat showImage(frame.height, frame.width, CV_8UC3, frame.data, frame.step);
            cv::imshow("frame", showImage);
            int key = cv::waitKey(waitTime);
            if (key == 'q')
            {
                end = true;
                break;
            }                
        }
    }
}

void write()
{
    avp::SharedAudioVideoFrame frame;
    while (!end)
    {
        writeQueue.pull(frame);
        if (frame.data)
        {
            avp::AudioVideoFrame shallowFrame = frame;
            bool ok = writer.write(shallowFrame);
            if (!ok)
                printf("Failed to write frame, check network connection.\n");
        }
    }
}

int main(int argc, char* argv[])
{
    const char* keys =
        "{camera_width | 1920 | camera picture width}"
        "{camera_height | 1080 | camera picture height}"
        "{frames_per_second | 30 | camera frame rate}"
        "{pano_width | 2048 | pano picture width}"
        "{pano_height | 1024 | pano picture height}"
        "{pano_bits_per_second | 1000000 | pano live stream bits per second}"
        "{pano_encode_preset | veryfast | pano video x264 encode preset}"
        "{pano_url | rtmp://pili-publish.live.detu.com/detulive/detudemov550?key=detukey | pano live stream address}"
        "{xml_path | paramricoh.xml | xml param file path}";

    cv::CommandLineParser parser(argc, argv, keys);

    std::vector<avp::Device> devices;
    avp::listDirectShowDevices(devices);
    if (devices.empty())
    {
        printf("Could not find DirectShow device.\n");
        return 0;
    }

    std::vector<avp::Device> audioDevices, videoDevices;
    avp::keepAudioDirectShowDevices(devices, audioDevices);
    avp::keepVideoDirectShowDevices(devices, videoDevices);
    if (videoDevices.empty())
    {
        printf("Could not find DirectShow video device.\n");
        return 0;
    }

    bool ok = false;
    std::vector<avp::Option> opts;
    std::string frameRateStr = parser.get<std::string>("frames_per_second");
    std::string videoSizeStr = parser.get<std::string>("camera_width") + "x" + parser.get<std::string>("camera_height");
    opts.push_back(std::make_pair("framerate", frameRateStr));
    opts.push_back(std::make_pair("video_size", videoSizeStr));
    std::string inputName = "video=" + videoDevices[0].shortName;
    if (audioDevices.size())
        inputName += (":audio=" + audioDevices[0].shortName);
    ok = reader.open(inputName, true, true, avp::PixelTypeBGR24, "dshow", opts);
    if (!ok)
    {
        printf("Could not open DirectShow video device '%s' with framerate = %s and video_size = %s\n",
            videoDevices[0].shortName.c_str(), frameRateStr.c_str(), videoSizeStr.c_str());
        return 0;
    }

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

    xmlPath = parser.get<std::string>("xml_path");
    ok = render.prepare(xmlPath, srcSize, dstSize);
    if (!ok)
    {
        printf("Could not prepare for render\n");
        return 0;
    }

    std::string url = parser.get<std::string>("pano_url");
    int frameRate = parser.get<int>("frames_per_second");
    int bitRate = parser.get<int>("pano_bits_per_second");
    std::string preset = parser.get<std::string>("pano_encode_preset");
    if (preset != "ultrafast" || preset != "superfast" ||
        preset != "veryfast" || preset != "faster" ||
        preset != "fast" || preset != "medium" || preset != "slow" ||
        preset != "slower" || preset != "veryslow")
        preset = "veryfast";
    opts.clear();
    opts.push_back(std::make_pair("preset", preset));
    opts.push_back(std::make_pair("ar", "44100"));
    ok = writer.open(url, url.substr(0, 4) == "rtsp" ? "rtsp" : "flv", true,
        true, "mp3", reader.getAudioSampleType(), reader.getAudioChannelLayout(), reader.getAudioSampleRate(), 128000,
        true, "h264", avp::PixelTypeBGR24, dstSize.width, dstSize.height, reader.getVideoFps(), 1500000, opts);
    if (!ok)
    {
        printf("could not open url %s\n", url.c_str());
        return 0;
    }

    waitTime = std::max(5.0, 1000.0 / frameRate - 5);

    timerAll.start();

    std::thread readThread(read);
    std::thread procThread(proc);
    std::thread displayThread(display);
    std::thread writeThread(write);

    readThread.join();
    procThread.join();
    displayThread.join();
    writeThread.join();

    timerAll.end();
    printf("all time %f\n", timerAll.elapse());

    reader.close();
    writer.close();
    return 0;
}