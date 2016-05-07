#include "RicohUtil.h"
#include "AudioVideoProcessor.h"
#include "Timer.h"
#include "StampedFrameQueue.h"

#include "opencv2/core/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <thread>
#include <fstream>

cv::Size dstSize(2048, 1024);
cv::Size srcSize(1920, 1080);

RicohPanoramaRender render;

int frameCount = 0;
ztool::Timer timerAll, timerTotal, timerDecode, timerReproject, timerBlend, timerEncode;

avp::AudioVideoReader reader;
avp::AudioVideoWriter writer;

StampedFrameQueue displayQueue(36), writeQueue(36);

bool end = false;

void read()
{
    avp::AudioVideoFrame rawImage;
    while (!end)
    {
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
        if (shallow.size() == srcSize)
        {
            StampedFrame f;
            f.frame = shallow.clone();
            f.timeStamp = rawImage.timeStamp;
            displayQueue.push(f);
            writeQueue.push(f);
        }            
        else
        {
            cv::Mat temp;
            cv::resize(shallow, temp, srcSize);
            displayQueue.push(temp);
            writeQueue.push(temp);
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
            avp::AudioVideoFrame image = avp::videoFrame(frame.frame.data, frame.frame.step, 
                avp::PixelTypeBGR24, frame.frame.cols, frame.frame.rows, frame.timeStamp);
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
        "{camera_width         | 1920    | camera picture width}"
        "{camera_height        | 1080    | camera picture height}"
        "{frames_per_second    | 30      | camera frame rate}"
        "{pano_width           | 2048    | pano picture width}"
        "{pano_height          | 1024    | pano picture height}"
        "{pano_bits_per_second | 1000000 | pano live stream bits per second}"
        "{pano_url             | rtsp://127.0.0.1/ricoh.sdp | pano live stream address}";

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
    ok = reader.open("video=" + devices[deviceIndex].shortName, false, true, avp::PixelTypeBGR24, "dshow", opts);
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

    std::string url = parser.get<std::string>("pano_url");
    int frameRate = parser.get<int>("frames_per_second");
    int bitRate = parser.get<int>("pano_bits_per_second");
    ok = writer.open(url, url.substr(0, 4) == "rtmp" ? "flv" : "rtsp", true, false, "", 0, 0, 0, 0,
        true, "h264", avp::PixelTypeBGR24, srcSize.width, srcSize.height, frameRate, 2000000);
    if (!ok)
    {
        printf("could not open url %s\n", url.c_str());
        return 0;
    }

    timerAll.start();

    std::thread readThread(read);
    std::thread displayThread(display);
    std::thread writeThread(write);

    readThread.join();
    displayThread.join();
    writeThread.join();

    timerAll.end();
    printf("all time %f\n", timerAll.elapse());

    reader.close();
    writer.close();

    //cv::waitKey(0);
    return 0;
}