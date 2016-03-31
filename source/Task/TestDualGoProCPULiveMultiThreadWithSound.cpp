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

//typedef RealTimeQueue<std::vector<StampedFrame> > StampedFrameVectorQueue;

// for video source
typedef CompleteQueue<avp::SharedAudioVideoFrame> CompleteFrameQueue;
// for synced video source
typedef RealTimeQueue<std::vector<avp::SharedAudioVideoFrame> > RealTimeFrameVectorQueue;
// for audio source and proc result
typedef RealTimeQueue<avp::SharedAudioVideoFrame> RealTimeFrameQueue;

cv::Size dstSize(1440, 720);
cv::Size srcSize(1920, 1080);
int bitRate, frameRate;
std::string ptsPath, url;

const int numVideos = 2;
std::vector<CompleteFrameQueue> frameBuffers(numVideos);
RealTimeFrameVectorQueue syncedFramesBufferForShow, syncedFramesBufferForProc;
RealTimeFrameQueue procFrameBufferForShow, procFrameBufferForSend;
DualGoProPanoramaRender render;
std::vector<avp::AudioVideoReader> videoReaders(numVideos);
avp::AudioVideoReader audioReader;
avp::AudioVideoWriter writer;

int audioOpened = 0;
int finish = 0;

int waitTime = 30;

// This variable controls that we check whether frame rate matches the set one
// after first time of synchronization has finished.
int checkFrameRate = 0;

inline void stopCompleteFrameBuffers()
{
    for (int i = 0; i < numVideos; i++)
        frameBuffers[i].stop();
}

void videoSource(int index)
{
    size_t id = std::this_thread::get_id().hash();
    CompleteFrameQueue& buffer = frameBuffers[index];
    avp::AudioVideoReader& reader = videoReaders[index];

    long long int count = 0, beginCheckCount = frameRate * 5;
    ztool::Timer timer;
    avp::AudioVideoFrame frame;
    bool ok;
    while (true)
    {
        ok = reader.read(frame);
        if (!ok)
        {
            printf("Error in %s [%8x], cannot read video frame\n", __FUNCTION__, id);
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

        // NOTICE, for simplicity, I do not check whether the frame has the right property.
        // For the sake of program robustness, we should at least check whether the frame
        // is of type VIDEO, and is not empty, and has the correct pixel type and width and height.
        buffer.push(frame);

        if (finish)
        {
            //buffer.stop();
            stopCompleteFrameBuffers();
            finish = 1;
            break;
        }
    }
}

void videoSink()
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
            avp::SharedAudioVideoFrame sharedFrame;
            frameBuffers[i].pull(sharedFrame);
        }
    }

    if (finish)
        return;

    long long int currMaxTS = -1;
    int currMaxIndex = -1;
    for (int i = 0; i < numVideos; i++)
    {
        avp::SharedAudioVideoFrame sharedFrame;
        frameBuffers[i].pull(sharedFrame);
        if (sharedFrame.timeStamp < 0)
        {
            printf("Error in %s [%8x], cannot read valid frame with non-negative time stamp\n", __FUNCTION__, id);
            finish = 1;
            break;
        }
        if (sharedFrame.timeStamp > currMaxTS)
        {
            currMaxIndex = i;
            currMaxTS = sharedFrame.timeStamp;
        }
    }

    if (finish)
        return;

    std::vector<avp::SharedAudioVideoFrame> syncedFrames(numVideos);
    avp::SharedAudioVideoFrame slowestFrame;
    frameBuffers[currMaxIndex].pull(slowestFrame);
    syncedFrames[currMaxIndex] = slowestFrame;
    printf("slowest ts = %lld\n", slowestFrame.timeStamp);
    for (int i = 0; i < numVideos; i++)
    {
        if (finish)
            break;

        if (i == currMaxIndex)
            continue;

        avp::SharedAudioVideoFrame sharedFrame;
        while (true)
        {
            frameBuffers[i].pull(sharedFrame);
            printf("this ts = %lld\n", sharedFrame.timeStamp);
            /*if (stampedFrame.timeStamp < 0)
            {
            printf("Error in %s [%8x], cannot read valid frame with non-negative time stamp\n", __FUNCTION__, id);
            finish = 1;
            break;
            }*/
            if (sharedFrame.timeStamp > slowestFrame.timeStamp)
            {
                syncedFrames[i] = sharedFrame;
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

    std::vector<avp::SharedAudioVideoFrame> frames(numVideos);
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

void showVideoSource()
{
    std::vector<avp::SharedAudioVideoFrame> frames;
    while (true)
    {
        if (finish)
            break;
        //printf("show\n");
        syncedFramesBufferForShow.pull(frames);
        if (frames.size() == 2)
        {
            cv::Mat show1(frames[0].height, frames[0].width, CV_8UC3, frames[0].data, frames[0].step);
            cv::imshow("frame 1", show1);
            cv::Mat show2(frames[1].height, frames[1].width, CV_8UC3, frames[1].data, frames[1].step);
            cv::imshow("frame 2", show2);
            printf("%lld, %lld\n", frames[0].timeStamp, frames[1].timeStamp);
            cv::waitKey(25);
        }
    }
}

void procVideo()
{
    std::vector<avp::SharedAudioVideoFrame> frames;
    cv::Mat result;
    while (true)
    {
        if (finish)
            break;
        //printf("show\n");
        syncedFramesBufferForShow.pull(frames);
        if (frames.size() == 2)
        {
            cv::Mat src1(frames[0].height, frames[0].width, CV_8UC3, frames[0].data, frames[0].step);
            cv::Mat src2(frames[1].height, frames[1].width, CV_8UC3, frames[1].data, frames[1].step);
            render.render(src1, src2, result);
            //cv::imshow("result", result);
            //cv::waitKey(1);
            avp::AudioVideoFrame shallow = avp::videoFrame(result.data, result.step, avp::PixelTypeBGR24, result.cols, result.rows, frames[0].timeStamp);
            avp::SharedAudioVideoFrame deep(shallow);
            procFrameBufferForShow.push(deep);
            procFrameBufferForSend.push(deep);
        }
    }
}

void showVideoResult()
{
    avp::SharedAudioVideoFrame frame;
    while (true)
    {
        if (finish)
            break;
        procFrameBufferForShow.pull(frame);
        if (frame.data)
        {
            cv::Mat show(frame.height, frame.width, CV_8UC3, frame.data, frame.step);
            cv::imshow("result", show);
            int key = cv::waitKey(waitTime);
            if (key == 'q')
            {
                finish = 1;
                break;
            }
        }
    }
}

void audioSource()
{
    if (!audioOpened)
        return;

    size_t id = std::this_thread::get_id().hash();

    long long int count = 0, beginCheckCount = frameRate * 5;
    ztool::Timer timer;
    avp::AudioVideoFrame frame;
    bool ok;
    while (true)
    {
        if (finish)
            break;

        ok = audioReader.read(frame);
        if (!ok)
        {
            printf("Error in %s [%8x], cannot read audio frame\n", __FUNCTION__, id);
            finish = 1;
            break;
        }

        procFrameBufferForSend.push(frame);
    }
}

void send()
{
    size_t id = std::this_thread::get_id().hash();
    avp::SharedAudioVideoFrame frame;
    while (true)
    {
        if (finish)
            break;
        procFrameBufferForSend.pull(frame);
        if (frame.data)
        {
            avp::AudioVideoFrame shallow = frame;
            bool ok = writer.write(shallow);
            if (!ok)
            {
                printf("Error in %s [%8x], cannot write frame\n", __FUNCTION__, id);
                finish = 1;
                break;
            }
        }
    }
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

    std::vector<avp::Device> devices, videoDevices, audioDevices;
    avp::listDirectShowDevices(devices);
    avp::keepVideoDirectShowDevices(devices, videoDevices);
    if (videoDevices.size() != 2)
    {
        printf("DirectShow video devices should be two\n");
        return 0;
    }
    avp::keepAudioDirectShowDevices(devices, audioDevices);
    if (audioDevices.size() > 2)
    {
        printf("DirectShow audio devices shoul be no more than two\n");
        return 0;
    }

    if (audioDevices.size())
    {
        std::vector<avp::Option> audioOpts;
        audioOpts.push_back(std::make_pair("ar", "44100"));
        if (audioDevices.size() == 2 && audioDevices[0].shortName == audioDevices[1].shortName)
            audioOpts.push_back(std::make_pair("audio_device_number", "0"));
        ok = audioReader.open("audio=" + audioDevices[1].shortName, true, 
            false, avp::PixelTypeUnknown, "dshow", audioOpts);
        if (!ok)
            printf("Could not open DirectShow audio device %s, skip\n", audioDevices[1].shortName.c_str());
        else
            audioOpened = 1;
    }

    url = parser.get<std::string>("pano_url");
    frameRate = parser.get<int>("frames_per_second");
    bitRate = parser.get<int>("pano_bits_per_second");
    std::vector<avp::Option> writerOpts;
    writerOpts.push_back(std::make_pair("preset", "ultrafast"));
    ok = writer.open(url, url.substr(0, 4) == "rtmp" ? "flv" : "rtsp", true,
        audioOpened, "mp3", audioReader.getAudioSampleType(), 
        audioReader.getAudioChannelLayout(), audioReader.getAudioSampleRate(), 128000,
        true, "h264", avp::PixelTypeBGR24, dstSize.width, dstSize.height, 
        frameRate, bitRate, writerOpts);
    if (!ok)
    {
        printf("Could not open rtmp streaming url with frame rate = %d and bit rate = %d\n", frameRate, bitRate);
        return 0;
    }    

    std::vector<avp::Option> opts;
    std::string frameRateStr = parser.get<std::string>("frames_per_second");
    std::string videoSizeStr = parser.get<std::string>("camera_width") + "x" + parser.get<std::string>("camera_height");
    opts.push_back(std::make_pair("framerate", frameRateStr));
    opts.push_back(std::make_pair("video_size", videoSizeStr));
    if (videoDevices[0].shortName == videoDevices[1].shortName)
    {
        opts.push_back(std::make_pair("video_device_number", "0"));
        ok = videoReaders[0].open("video=" + videoDevices[0].shortName, false, true, avp::PixelTypeBGR24, "dshow", opts);
        if (!ok)
        {
            printf("Could not open DirectShow video device %s[0] with framerate = %s and video_size = %s\n",
                devices[0].shortName.c_str(), frameRateStr.c_str(), videoSizeStr.c_str());
            return 0;
        }

        opts.resize(2);
        opts.push_back(std::make_pair("video_device_number", "1"));
        ok = videoReaders[1].open("video=" + videoDevices[1].shortName, false, true, avp::PixelTypeBGR24, "dshow", opts);
        if (!ok)
        {
            printf("Could not open DirectShow video device %s[1] with framerate = %s and video_size = %s\n",
                devices[1].shortName.c_str(), frameRateStr.c_str(), videoSizeStr.c_str());
            return 0;
        }
    }
    else
    {
        ok = videoReaders[0].open("video=" + videoDevices[0].shortName, false, true, avp::PixelTypeBGR24, "dshow", opts);
        if (!ok)
        {
            printf("Could not open DirectShow video device %s with framerate = %s and video_size = %s\n",
                devices[0].shortName.c_str(), frameRateStr.c_str(), videoSizeStr.c_str());
            return 0;
        }

        ok = videoReaders[1].open("video=" + videoDevices[1].shortName, false, true, avp::PixelTypeBGR24, "dshow", opts);
        if (!ok)
        {
            printf("Could not open DirectShow video device %s with framerate = %s and video_size = %s\n",
                devices[1].shortName.c_str(), frameRateStr.c_str(), videoSizeStr.c_str());
            return 0;
        }
    }

    waitTime = std::max(5.0, 1000.0 / frameRate - 5);

    std::thread vsrc0(videoSource, 0);
    std::thread vsrc1(videoSource, 1);
    std::thread vsink(videoSink);
    std::thread asrc(audioSource);
    //std::thread svsrc(showVideoSource);
    std::thread pv(procVideo);
    std::thread svr(showVideoResult);
    std::thread s(send);

    vsrc0.join();
    vsrc1.join();
    vsink.join();
    asrc.join();
    //svsrc.join();
    pv.join();
    svr.join();
    s.join();

    return 0;
}