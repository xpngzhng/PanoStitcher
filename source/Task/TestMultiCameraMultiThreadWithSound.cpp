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
#include <iostream>

#define COMPILE_LIVE_STREAM_PROGRAM 1

// for video source
typedef CompleteQueue<avp::SharedAudioVideoFrame> CompleteFrameQueue;
// for synced video source
typedef RealTimeQueue<std::vector<avp::SharedAudioVideoFrame> > RealTimeFrameVectorQueue;
// for audio source and proc result
typedef RealTimeQueue<avp::SharedAudioVideoFrame> RealTimeFrameQueue;

cv::Size srcSize(1920, 1080);

int frameRate;
int audioBitRate = 96000;

cv::Size streamFrameSize(1440, 720);
int streamBitRate;
std::string streamEncodePreset;
std::string streamURL;

int saveFile;
cv::Size fileFrameSize(1440, 720);
int fileDuration;
int fileBitRate;
std::string fileEncodePreset;

std::string cameraParamPath;
std::string cameraModel;
std::unique_ptr<PanoramaRender> ptrRender;

int numCameras;
std::unique_ptr<std::vector<CompleteFrameQueue> > ptrFrameBuffers;
RealTimeFrameVectorQueue syncedFramesBufferForShow, syncedFramesBufferForProc;
RealTimeFrameQueue procFrameBufferForShow, procFrameBufferForSend, procFrameBufferForSave;
std::vector<avp::AudioVideoReader> videoReaders;
avp::AudioVideoReader audioReader;
avp::AudioVideoWriter writer;

int audioOpened = 0;
int finish = 0;

int waitTime = 30;

// This variable controls that we check whether frame rate matches the set one
// after first time of synchronization has finished.
int checkFrameRate = 0;

// This variable controls how frequently the synchronization procedure is called,
// measured in seconds.
int syncInterval = 60;

// This variable contros how frequently the gain adjust procedure is called,
// measured in seconds.
int gainAdjustInterval = 300;

inline void stopCompleteFrameBuffers()
{
    std::vector<CompleteFrameQueue>& frameBuffers = *ptrFrameBuffers;
    for (int i = 0; i < numCameras; i++)
        frameBuffers[i].stop();
}

void videoSource(int index)
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started, index = %d\n", __FUNCTION__, id, index);

    std::vector<CompleteFrameQueue>& frameBuffers = *ptrFrameBuffers;
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
            double actualFps = (count - beginCheckCount) / timer.elapse();
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
    reader.close();

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void videoSink()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    std::this_thread::sleep_for(std::chrono::microseconds(100));
    std::vector<CompleteFrameQueue>& frameBuffers = *ptrFrameBuffers;

    if (finish)
        return;

    for (int i = 0; i < numCameras; i++)
        printf("size = %d\n", frameBuffers[i].size());

    for (int j = 0; j < 25; j++)
    {
        for (int i = 0; i < numCameras; i++)
        {
            avp::SharedAudioVideoFrame sharedFrame;
            frameBuffers[i].pull(sharedFrame);
        }
    }

    if (finish)
        return;

    while (true)
    {
        if (finish)
            return;

        long long int currMaxTS = -1;
        int currMaxIndex = -1;
        for (int i = 0; i < numCameras; i++)
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

        std::vector<avp::SharedAudioVideoFrame> syncedFrames(numCameras);
        avp::SharedAudioVideoFrame slowestFrame;
        frameBuffers[currMaxIndex].pull(slowestFrame);
        syncedFrames[currMaxIndex] = slowestFrame;
        printf("slowest ts = %lld\n", slowestFrame.timeStamp);
        for (int i = 0; i < numCameras; i++)
        {
            if (finish)
                break;

            if (i == currMaxIndex)
                continue;

            avp::SharedAudioVideoFrame sharedFrame;
            while (true)
            {
                if (finish)
                    break;

                frameBuffers[i].pull(sharedFrame);
                printf("this ts = %lld\n", sharedFrame.timeStamp);
                if (sharedFrame.timeStamp < 0)
                {
                    printf("Error in %s [%8x], cannot read valid frame with non-negative time stamp\n", __FUNCTION__, id);
                    finish = 1;
                    break;
                }
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

        if (!checkFrameRate)
            checkFrameRate = 1;

        int pullCount = 0;        
        std::vector<avp::SharedAudioVideoFrame> frames(numCameras);
        while (true)
        {
            if (finish)
                return;

            for (int i = 0; i < numCameras; i++)
            {
                frameBuffers[i].pull(frames[i]);
                //printf("%d ", frameBuffers[i].size());
            }
            //printf("\n");
            syncedFramesBufferForShow.push(frames);
            syncedFramesBufferForProc.push(frames);

            pullCount++;
            int needSync = 0;
            if (pullCount == frameRate * syncInterval)
            {
                printf("Checking frames synchronization status, ");
                long long int maxDiff = 1000000.0 / frameRate * 1.1 + 0.5;
                long long int baseTimeStamp = frames[0].timeStamp;
                for (int i = 1; i < numCameras; i++)
                {
                    if (abs(baseTimeStamp - frames[i].timeStamp) > maxDiff)
                    {
                        needSync = 1;
                        break;
                    }
                }

                if (needSync)
                {
                    printf("frames badly synchronized, resync\n");
                    break;
                }                    
                else
                {
                    printf("frames well synchronized, continue\n");
                    pullCount = 0;
                }                    
            }
        }
    }

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void showVideoSource()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    std::vector<avp::SharedAudioVideoFrame> frames;
    int origWidth = 0, origHeight = 0;
    int showWidth = 0, showHeight = 0;
    int maxShowWidth = 512, maxShowHeight = 512;
    cv::Mat showImage;
    static int saveCount = 0;
    while (true)
    {
        if (finish)
            break;
        //printf("show\n");
        syncedFramesBufferForShow.pull(frames);
        if (frames.size() == numCameras)
        {
            bool ok = true;
            int width = frames[0].width, height = frames[0].height;
            for (int i = 1; i < numCameras; i++)
            {
                if (width != frames[i].width || height != frames[i].height)
                {
                    ok = false;
                    break;
                }
            }
            if (!ok)
                continue;

            if (origWidth == 0 || origHeight == 0)
            {
                origWidth = width;
                origHeight = height;
                showWidth = width;
                showHeight = height;
                while (showWidth > maxShowWidth && showHeight > maxShowHeight)
                {
                    showWidth /= 2;
                    showHeight /= 2;
                }
            }

            char buf[64];
            for (int i = 0; i < numCameras; i++)
            {
                sprintf(buf, "frame %d", i);
                //cv::Mat show(frames[i].height, frames[i].width, CV_8UC3, frames[i].data, frames[i].step);
                //cv::imshow(buf, show);
                cv::Mat origImage(frames[i].height, frames[i].width, CV_8UC3, frames[i].data, frames[i].step);
                cv::resize(origImage, showImage, cv::Size(showWidth, showHeight), 0, 0, cv::INTER_NEAREST);
                cv::imshow(buf, showImage);
            }

            int key = cv::waitKey(waitTime);
            if (key == 's')
            {
                for (int i = 0; i < numCameras; i++)
                {
                    sprintf(buf, "frame%d-%d.jpg", saveCount, i);
                    cv::Mat origImage(frames[i].height, frames[i].width, CV_8UC3, frames[i].data, frames[i].step);
                    cv::imwrite(buf, origImage);
                }
                saveCount++;
            }
            else if (key == 'q')
            {
                finish = 1;
                break;
            }
        }
    }

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void procVideo()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    std::vector<avp::SharedAudioVideoFrame> frames;
    std::vector<cv::Mat> src;
    cv::Mat result, scaledResult;
    bool ok;
    long long int begTimeStamp = -1LL;
    while (true)
    {
        if (finish)
            break;
        //printf("show\n");
        syncedFramesBufferForProc.pull(frames);
        //printf("before check size\n");
        if (frames.size() == numCameras)
        {
            src.resize(numCameras);
            for (int i = 0; i < numCameras; i++)
                src[i] = cv::Mat(frames[i].height, frames[i].width, CV_8UC3, frames[i].data, frames[i].step);

            ok = ptrRender->render(src, result);
            if (!ok)
            {
                printf("Error in %s [%8x], render failed\n", __FUNCTION__, id);
                finish = 1;
                break;
            }
            //cv::imshow("result", result);
            //cv::waitKey(1);
            avp::AudioVideoFrame shallow = avp::videoFrame(result.data, result.step, avp::PixelTypeBGR24, result.cols, result.rows, frames[0].timeStamp);
            avp::SharedAudioVideoFrame deep(shallow);
            if ((saveFile && (streamFrameSize == fileFrameSize)) || !saveFile)
            {
                procFrameBufferForShow.push(deep);
                procFrameBufferForSend.push(deep);
            }
            else
            {
                cv::resize(result, scaledResult, streamFrameSize);
                avp::AudioVideoFrame scaledShallow = 
                    avp::videoFrame(scaledResult.data, scaledResult.step, avp::PixelTypeBGR24, scaledResult.cols, scaledResult.rows, frames[0].timeStamp);
                avp::SharedAudioVideoFrame scaledDeep(scaledShallow);
                procFrameBufferForShow.push(scaledDeep);
                procFrameBufferForSend.push(scaledDeep);
            }
            if (saveFile)
                procFrameBufferForSave.push(deep);
        }
    }

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void showVideoResult()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

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

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void audioSource()
{
    if (!audioOpened)
        return;

    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

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

        avp::SharedAudioVideoFrame deep(frame);
        procFrameBufferForSend.push(deep);
        procFrameBufferForSave.push(deep);
    }

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void send()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

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
    writer.close();

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void save()
{
    if (!saveFile)
        return;

    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    char buf[128];
    int count = 0;
    avp::SharedAudioVideoFrame frame;
    avp::AudioVideoWriter writer;
    std::vector<avp::Option> writerOpts;
    writerOpts.push_back(std::make_pair("preset", fileEncodePreset));
    sprintf(buf, "temp%d.mp4", count++);
    bool ok = writer.open(buf, "mp4", true,
        audioOpened, "aac", audioReader.getAudioSampleType(),
        audioReader.getAudioChannelLayout(), audioReader.getAudioSampleRate(), audioBitRate,
        true, "h264", avp::PixelTypeBGR24, fileFrameSize.width, fileFrameSize.height,
        frameRate, fileBitRate, writerOpts);
    if (!ok)
    {
        printf("Error in %s [%d], could not save current audio video\n", __FUNCTION__, id);
        return;
    }
    long long int fileFirstTimeStamp = -1;
    while (true)
    {
        if (finish)
            break;
        procFrameBufferForSave.pull(frame);
        if (frame.data)
        {
            if (fileFirstTimeStamp < 0)
                fileFirstTimeStamp = frame.timeStamp;

            if (frame.timeStamp - fileFirstTimeStamp > fileDuration * 1000000LL)
            {
                writer.close();
                sprintf(buf, "temp%d.mp4", count++);
                ok = writer.open(buf, "mp4", true,
                    audioOpened, "aac", audioReader.getAudioSampleType(),
                    audioReader.getAudioChannelLayout(), audioReader.getAudioSampleRate(), audioBitRate,
                    true, "h264", avp::PixelTypeBGR24, fileFrameSize.width, fileFrameSize.height,
                    frameRate, fileBitRate, writerOpts);
                if (!ok)
                {
                    printf("Error in %s [%d], could not save current audio video\n", __FUNCTION__, id);
                    break;
                }
                fileFirstTimeStamp = frame.timeStamp;
            }
            avp::AudioVideoFrame shallow = frame;
            ok = writer.write(shallow);
            if (!ok)
            {
                printf("Error in %s [%d], could not write current frame\n", __FUNCTION__, id);
                break;
            }
        }
    }
    writer.close();

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

void seleveVideoDevices(bool interactive, int numCameras, std::vector<int>& videoIndexes)
{
    videoIndexes.resize(numCameras);
    if (interactive)
    {
        for (int i = 0; i < numCameras; i++)
        {
            while (true)
            {
                printf("Select video device #%d, input the number in parentheses\"()\": ", i);
                int val;
                std::cin >> val;
                if (val < 0 || val >= numCameras)
                {
                    printf("Error, input number should between 0 and %d, try again\n", numCameras - 1);
                    continue;
                }
                bool success = true;
                for (int j = 0; j < i; j++)
                {
                    if (videoIndexes[j] == val)
                    {
                        printf("Error, input number equals previous input one, try again\n");
                        success = false;
                        break;
                    }
                }
                if (!success)
                    continue;
                else
                {
                    videoIndexes[i] = val;
                    break;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < numCameras; i++)
            videoIndexes[i] = i;
    }
}

#if COMPILE_LIVE_STREAM_PROGRAM

int main(int argc, char* argv[])
{
    const char* keys =
        "{a | camera_model | dualgopro | camera model}"
        "{b | camera_param_path | dualgopro.pts | camera parameter file path, may be xml file path or ptgui pts file path}"
        "{c | num_cameras | 2 | number of cameras}"
        "{d | camera_width | 1920 | camera picture width}"
        "{e | camera_height | 1080 | camera picture height}"
        "{f | frames_per_second | 30 | camera frame rate}"
        "{g | pano_stream_frame_width | 1440 | pano video live stream picture width}"
        "{h | pano_stream_frame_height | 720 | pano video live stream picture height}"
        "{i | pano_stream_bits_per_second | 1000000 | pano video live stream bits per second}"
        "{j | pano_stream_encode_preset | veryfast | pano video live stream x264 encode preset}"
        "{k | pano_stream_url | rtmp://pili-publish.live.detu.com/detulive/detudemov550?key=detukey | pano live stream address}"        
        "{l | pano_save_file | false | whether to save audio video to local hard disk}"
        "{m | pano_file_duration | 60 | each local pano audio video file duration in seconds}"
        "{g | pano_file_frame_width | 1440 | pano video local file picture width}"
        "{h | pano_file_frame_height | 720 | pano video local file picture height}"
        "{n | pano_file_bits_per_second | 1000000 | pano video local file bits per second}"
        "{o | pano_file_encode_preset | veryfast | pano video local file x264 encode preset}"
        "{p | enable_audio | false | enable audio or not}"
        "{q | enable_interactive_select_devices | false | enable interactice select devices}";

    cv::CommandLineParser parser(argc, argv, keys);

    cameraModel = parser.get<std::string>("camera_model");
    if (cameraModel == "general")
        ptrRender.reset(new MultiCameraPanoramaRender);
    /*else if (cameraModel == "generalgpu")
        ptrRender.reset(new MultiCameraPanoramaRenderGPU);*/
    else if (cameraModel == "dualgopro")
        ptrRender.reset(new DualGoProPanoramaRender);
    else
    {
        printf("Unrecognized camera model %s\n", cameraModel.c_str());
        return 0;
    }
    //printf("pass alloc memory for render\n");

    if (cameraModel == "dualgopro")
        numCameras = 2;
    else
        numCameras = parser.get<int>("num_cameras");
    if (numCameras <= 0)
    {
        printf("num_cameras should be positive\n");
        return 0;
    }
    srcSize.width = parser.get<int>("camera_width");
    srcSize.height = parser.get<int>("camera_height");

    saveFile = parser.get<bool>("pano_save_file");
    fileDuration = parser.get<int>("pano_file_duration");
    fileBitRate = parser.get<int>("pano_file_bits_per_second");
    fileEncodePreset = parser.get<std::string>("pano_file_encode_preset");
    if (fileEncodePreset != "ultrafast" || fileEncodePreset != "superfast" ||
        fileEncodePreset != "veryfast" || fileEncodePreset != "faster" ||
        fileEncodePreset != "fast" || fileEncodePreset != "medium" || fileEncodePreset != "slow" ||
        fileEncodePreset != "slower" || fileEncodePreset != "veryslow")
        fileEncodePreset = "veryfast";

    fileFrameSize.width = parser.get<int>("pano_file_frame_width");
    fileFrameSize.height = parser.get<int>("pano_file_frame_height");
    if (fileFrameSize.width <= 0 || fileFrameSize.height <= 0 ||
        (fileFrameSize.width & 1) || (fileFrameSize.height & 1) ||
        (fileFrameSize.width != fileFrameSize.height * 2))
    {
        printf("pano_file_frame_width and pano_file_frame_height should be positive even numbers, "
            "and pano_file_frame_width should be two times of pano_file_frame_height\n");
        return 0;
    }
    
    streamFrameSize.width = parser.get<int>("pano_stream_frame_width");
    streamFrameSize.height = parser.get<int>("pano_stream_frame_height");
    if (streamFrameSize.width <= 0 || streamFrameSize.height <= 0 ||
        (streamFrameSize.width & 1) || (streamFrameSize.height & 1) ||
        (streamFrameSize.width != streamFrameSize.height * 2))
    {
        printf("pano_stream_frame_width and pano_stream_frame_height should be positive even numbers, "
            "and pano_stream_frame_width should be two times of pano_stream_frame_height\n");
        return 0;
    }

    cameraParamPath = parser.get<std::string>("camera_param_path");
    bool ok;
    ok = ptrRender->prepare(cameraParamPath, srcSize, saveFile ? fileFrameSize : streamFrameSize);
    if (!ok)
    {
        printf("Could not prepare for panorama render\n");
        return 0;
    }
    //printf("pass render prepare\n");

    std::vector<avp::Device> devices, videoDevices, audioDevices;
    avp::listDirectShowDevices(devices);

    avp::keepVideoDirectShowDevices(devices, videoDevices);
    int numVideoDevices = videoDevices.size();
    if (numVideoDevices)
    {
        printf("DirectShow video device(s) found:\n");
        for (int i = 0; i < numVideoDevices; i++)
        {
            printf("(%d) %s[%s] - %s\n", i, videoDevices[i].shortName.c_str(), 
                videoDevices[i].numString.c_str(), videoDevices[i].longName.c_str());
        }
    }    
    if (numVideoDevices < numCameras)
    {
        printf("DirectShow video devices not enough, %d devices are required\n", numCameras);
        return 0;
    }
    //printf("pass list video devices\n");

    bool interactive = parser.get<bool>("enable_interactive_select_devices");
    std::vector<int> videoIndexes(numCameras);
    seleveVideoDevices(interactive, numCameras, videoIndexes);
    
    bool tryOpenAudio = parser.get<bool>("enable_audio");
    if (tryOpenAudio)
    {
        avp::keepAudioDirectShowDevices(devices, audioDevices);
        int numAudioDevices = audioDevices.size();
        if (numAudioDevices)
        {
            printf("DirectShow audio device(s) found:\n");
            for (int i = 0; i < numAudioDevices; i++)
            {
                printf("(%d) %s[%s] - %s\n", i, audioDevices[i].shortName.c_str(), 
                    audioDevices[i].numString.c_str(), audioDevices[i].longName.c_str());
            }

            int audioIndex;
            if (interactive)
            {
                while (true)
                {
                    printf("Select audio device, input the number in parentheses\"()\": ");
                    int val;
                    std::cin >> val;
                    if (val < 0 || val >= numCameras)
                    {
                        printf("Error, input number should between 0 and %d, try again\n", numAudioDevices - 1);
                        continue;
                    }
                    audioIndex = val;
                    break;
                }
                printf("Try to open the DirectShow audio device indexed %d\n", audioIndex);
            }
            else
            {
                printf("Try to open the DirectShow audio device indexed 0\n");
                audioIndex = 0;
            }
            std::vector<avp::Option> audioOpts;
            audioOpts.push_back(std::make_pair("ar", "44100"));
            audioOpts.push_back(std::make_pair("audio_device_number", audioDevices[audioIndex].numString));
            ok = audioReader.open("audio=" + audioDevices[audioIndex].shortName, true,
                false, avp::PixelTypeUnknown, "dshow", audioOpts);
            if (!ok)
            {
                printf("Could not open DirectShow audio device %s[%s], skip\n",
                    audioDevices[audioIndex].shortName.c_str(), audioDevices[audioIndex].numString.c_str());
                audioOpened = 0;
            }
            else
                audioOpened = 1;
        }
        else
            audioOpened = 0;
    }
    else
        audioOpened = 0;

    frameRate = parser.get<int>("frames_per_second");
    streamURL = parser.get<std::string>("pano_stream_url");    
    streamBitRate = parser.get<int>("pano_stream_bits_per_second");
    streamEncodePreset = parser.get<std::string>("pano_stream_encode_preset");
    if (streamEncodePreset != "ultrafast" || streamEncodePreset != "superfast" ||
        streamEncodePreset != "veryfast" || streamEncodePreset != "faster" ||
        streamEncodePreset != "fast" || streamEncodePreset != "medium" || streamEncodePreset != "slow" ||
        streamEncodePreset != "slower" || streamEncodePreset != "veryslow")
        streamEncodePreset = "veryfast";
    std::vector<avp::Option> writerOpts;
    writerOpts.push_back(std::make_pair("preset", streamEncodePreset));
    ok = writer.open(streamURL, streamURL.substr(0, 4) == "rtmp" ? "flv" : "rtsp", true,
        audioOpened, "aac", audioReader.getAudioSampleType(),
        audioReader.getAudioChannelLayout(), audioReader.getAudioSampleRate(), audioBitRate,
        true, "h264", avp::PixelTypeBGR24, streamFrameSize.width, streamFrameSize.height,
        frameRate, streamBitRate, writerOpts);
    if (!ok)
    {
        printf("Could not open rtmp streaming url with frame rate = %d and bit rate = %d\n", frameRate, streamBitRate);
        return 0;
    }

    std::vector<avp::Option> opts;
    std::string frameRateStr = parser.get<std::string>("frames_per_second");
    std::string videoSizeStr = parser.get<std::string>("camera_width") + "x" + parser.get<std::string>("camera_height");
    opts.push_back(std::make_pair("framerate", frameRateStr));
    opts.push_back(std::make_pair("video_size", videoSizeStr));

    videoReaders.resize(numCameras);
    for (int i = 0; i < numCameras; i++)
    {
        opts.resize(2);
        opts.push_back(std::make_pair("video_device_number", videoDevices[videoIndexes[i]].numString));
        ok = videoReaders[i].open("video=" + videoDevices[videoIndexes[i]].shortName, 
            false, true, avp::PixelTypeBGR24, "dshow", opts);
        if (!ok)
        {
            printf("Could not open DirectShow video device %s[%s] with framerate = %s and video_size = %s\n",
                videoDevices[videoIndexes[i]].shortName.c_str(), videoDevices[videoIndexes[i]].numString.c_str(),
                frameRateStr.c_str(), videoSizeStr.c_str());
            return 0;
        }
    }

    ptrFrameBuffers.reset(new std::vector<CompleteFrameQueue>(numCameras));

    waitTime = std::max(5.0, 1000.0 / frameRate - 5);

    std::vector<std::unique_ptr<std::thread> > vsrcs(numCameras);
    for (int i = 0; i < numCameras; i++)
        vsrcs[i].reset(new std::thread(videoSource, i));
    std::thread vsink(videoSink);
    std::thread asrc(audioSource);
    //std::thread svsrc(showVideoSource);
    std::thread pv(procVideo);
    std::thread svr(showVideoResult);
    std::thread snd(send);
    std::thread sv(save);

    for (int i = 0; i < numCameras; i++)
        vsrcs[i]->join();
    vsink.join();
    asrc.join();
    //svsrc.join();
    pv.join();
    svr.join();
    snd.join();
    sv.join();

    return 0;
}

#else

int main(int argc, char* argv[])
{
    const char* keys =
        "{a | num_cameras | 2 | number of cameras}"
        "{b | camera_width | 1920 | camera picture width}"
        "{c | camera_height | 1080 | camera picture height}"
        "{d | frames_per_second | 30 | camera frame rate}"
        "{e | enable_interactive_select_devices | false | enable interactice select devices}";

    cv::CommandLineParser parser(argc, argv, keys);

    numCameras = parser.get<int>("num_cameras");
    if (numCameras <= 0)
    {
        printf("num_cameras should be positive\n");
        return 0;
    }

    srcSize.width = parser.get<int>("camera_width");
    srcSize.height = parser.get<int>("camera_height");

    std::vector<avp::Device> devices, videoDevices, audioDevices;
    avp::listDirectShowDevices(devices);

    avp::keepVideoDirectShowDevices(devices, videoDevices);
    int numVideoDevices = videoDevices.size();
    if (numVideoDevices)
    {
        printf("DirectShow video device(s) found:\n");
        for (int i = 0; i < numVideoDevices; i++)
        {
            printf("(%d) %s[%s] - %s\n", i, videoDevices[i].shortName.c_str(),
                videoDevices[i].numString.c_str(), videoDevices[i].longName.c_str());
        }
    }
    if (numVideoDevices < numCameras)
    {
        printf("DirectShow video devices not enough, %d devices are required\n", numCameras);
        return 0;
    }

    bool interactive = parser.get<bool>("enable_interactive_select_devices");
    std::vector<int> videoIndexes(numCameras);
    seleveVideoDevices(interactive, numCameras, videoIndexes);

    frameRate = parser.get<int>("frames_per_second");

    std::vector<avp::Option> opts;
    std::string frameRateStr = parser.get<std::string>("frames_per_second");
    std::string videoSizeStr = parser.get<std::string>("camera_width") + "x" + parser.get<std::string>("camera_height");
    opts.push_back(std::make_pair("framerate", frameRateStr));
    opts.push_back(std::make_pair("video_size", videoSizeStr));

    bool ok;
    videoReaders.resize(numCameras);
    for (int i = 0; i < numCameras; i++)
    {
        opts.resize(2);
        opts.push_back(std::make_pair("video_device_number", videoDevices[videoIndexes[i]].numString));
        ok = videoReaders[videoIndexes[i]].open("video=" + videoDevices[videoIndexes[i]].shortName,-
            false, true, avp::PixelTypeBGR24, "dshow", opts);
        if (!ok)
        {
            printf("Could not open DirectShow video device %s[%s] with framerate = %s and video_size = %s\n",
                devices[videoIndexes[i]].shortName.c_str(), devices[videoIndexes[i]].numString.c_str(),
                frameRateStr.c_str(), videoSizeStr.c_str());
            return 0;
        }
    }

    ptrFrameBuffers.reset(new std::vector<CompleteFrameQueue>(numCameras));

    waitTime = std::max(5.0, 1000.0 / frameRate - 5);

    std::vector<std::unique_ptr<std::thread> > vsrcs(numCameras);
    for (int i = 0; i < numCameras; i++)
        vsrcs[i].reset(new std::thread(videoSource, i));
    std::thread vsink(videoSink);
    std::thread svsrc(showVideoSource);

    for (int i = 0; i < numCameras; i++)
        vsrcs[i]->join();
    vsink.join();
    svsrc.join();

    return 0;
}

#endif