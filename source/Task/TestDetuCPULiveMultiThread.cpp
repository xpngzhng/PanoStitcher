#include "RicohUtil.h"
#include "AudioVideoProcessor.h"
#include "Timer.h"
#include "StampedFrameQueue.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <thread>
#include <fstream>

cv::Size dstSize(1440, 720);
cv::Size srcSize(1024, 768);

int frameCount = 0;
avp::BGRImage rawImage;
cv::Mat blendImage;
std::string xmlPath;
DetuPanoramaRender render;

ztool::Timer timerAll, timerTotal, timerDecode, timerReproject, timerBlend, timerEncode;

avp::VideoReader reader;
//avp::VideoWriter writer;
avp::AudioVideoSender writer;

StampedFrameQueue origQueue(36), displayQueue(36), writeQueue(36);

bool end = false;

void read()
{

    while (!end)
    {
        //printf("currCount = %d\n", frameCount++);
        //if (frameCount >= 4800)
        //{
        //    end = true;
        //    break;
        //}

        bool success = true;
        timerDecode.start();
        success = reader.read(rawImage);
        timerDecode.end();
        if (!success)
        {
            end = true;
            break;
        }
        cv::Mat shallow(rawImage.height, rawImage.width, CV_8UC3, rawImage.data, rawImage.step);
        //cv::imshow("orig", shallow);
        //cv::waitKey(1);
        if (shallow.size() == srcSize)
            origQueue.push(shallow.clone());
        else
        {
            cv::Mat temp;
            cv::resize(shallow, temp, srcSize);
            origQueue.push(temp);
        }

    }
}

void proc()
{
    StampedFrame frame;
    while (!end)
    {
        origQueue.pull(frame);
        if (frame.frame.data)
        {
            render.render(frame.frame, blendImage);
            frame.frame = blendImage.clone();
            displayQueue.push(frame);
            writeQueue.push(frame);
            //printf("proc finish\n");
        }
    }
}

void display()
{
    StampedFrame frame;
    while (!end)
    {
        displayQueue.pull(frame);
        if (frame.frame.data)
        {
            cv::imshow("frame", frame.frame);
            int key = cv::waitKey(30);
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
    StampedFrame frame;
    while (!end)
    {
        writeQueue.pull(frame);
        if (frame.frame.data)
        {
            avp::BGRImage image(frame.frame.data, frame.frame.cols, frame.frame.rows, frame.frame.step);
            bool ok = writer.write(image);
            if (!ok)
                printf("Failed to write video frame, check network connection.\n");
            //printf("finish write frame\n");
        }
    }
}

int main(int argc, char* argv[])
{
    const char* keys =
        "{a | camera_width | 1024 | camera picture width}"
        "{b | camera_height | 768 | camera picture height}"
        "{c | frames_per_second | 30 | camera frame rate}"
        "{d | pano_width | 1440 | pano picture width}"
        "{e | pano_height | 720 | pano picture height}"
        "{f | pano_bits_per_second | 1000000 | pano live stream bits per second}"
        "{k | pano_encode_preset | veryfast | pano video x264 encode preset}"
        "{g | pano_url | rtmp://pili-publish.live.detu.com/detulive/detudemov550?key=detukey | pano live stream address}"
        "{h | xml_path | paramdetusmall.xml | xml param file path}";

    cv::CommandLineParser parser(argc, argv, keys);

    std::vector<avp::Device> devices;
    avp::listDirectShowDevices(devices);
    if (devices.empty())
    {
        printf("Could not find DirectShow device.\n");
        return 0;
    }

    int numDevices = devices.size();
    int deviceIndex = -1;
    for (int i = 0; i < numDevices; i++)
    {
        if (devices[i].deviceType == avp::VIDEO)
        {
            deviceIndex = i;
            break;
        }
    }
    if (deviceIndex < 0)
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
    ok = reader.openDirectShowDevice("video=" + devices[deviceIndex].shortName, avp::PixelTypeBGR24, opts);
    if (!ok)
    {
        printf("Could not open DirectShow video device with framerate = %s and video_size = %s\n",
            frameRateStr.c_str(), videoSizeStr.c_str());
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
    //ok = writer.open("detulive.mp4", avp::PixelTypeBGR24, dstSize.width, dstSize.height, 30, 0, avp::EncodeSpeedUltraFast);

    //int count = 0;
    //char buf[1024];
    //for (int i = 0; i >= 0; i++)
    //{
    //    reader.read(rawImage);
    //    cv::Mat shallow(rawImage.height, rawImage.width, CV_8UC3, rawImage.data, rawImage.step);
    //    cv::imshow("cam", shallow);
    //    int key = cv::waitKey(10);
    //    if (key == 's')
    //    {
    //        count++;
    //        sprintf(buf, "detu%d.bmp", count);
    //        cv::imwrite(buf, shallow);
    //    }
    //    else if (key == 'q')
    //        break;
    //}
    //    
    //return 0;

    timerAll.start();

    //cv::namedWindow("frame");
    //cv::imshow("frame", intersect);
    //cv::waitKey(1);

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