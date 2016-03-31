#include "PanoramaTask.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <thread>

cv::Size srcSize(1920, 1080);

int frameRate;
int audioBitRate = 96000;

cv::Size streamFrameSize(1440, 720);
int streamBitRate;
std::string streamEncoder;
std::string streamEncodePreset;
std::string streamURL;

int saveFile;
cv::Size fileFrameSize(1440, 720);
int fileDuration;
int fileBitRate;
std::string fileEncoder;
std::string fileEncodePreset;

std::string cameraParamPath;
std::string cameraModel;

int numCameras;
int audioOpened;
int waitTime = 30;

int enableCuda = 0;

PanoramaLiveStreamTask task;

void selectVideoDevices(bool interactive, int numCameras, std::vector<int>& videoIndexes)
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

void showVideoSources()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    std::vector<avp::SharedAudioVideoFrame> frames;
    while (true)
    {
        if (task.hasFinished())
            break;
        task.getVideoSourceFrames(frames);
        if (frames.size() == numCameras)
        {
            char winName[64];
            for (int i = 0; i < numCameras; i++)
            {
                cv::Mat show(frames[i].height, frames[i].width, 
                    frames[i].pixelType == avp::PixelTypeBGR24 ? CV_8UC3 : CV_8UC4, frames[i].data, frames[i].step);
                sprintf(winName, "source %d", i);
                cv::imshow(winName, show);
            }            
            int key = cv::waitKey(waitTime);
            if (key == 'q')
            {
                task.closeAll();
                break;
            }
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
        if (task.hasFinished())
            break;
        task.getStitchedVideoFrame(frame);
        if (frame.data)
        {
            cv::Mat show(frame.height, frame.width, frame.pixelType == avp::PixelTypeBGR24 ? CV_8UC3 : CV_8UC4, frame.data, frame.step);
            cv::imshow("result", show);
            int key = cv::waitKey(waitTime);
            if (key == 'q')
            {
                task.closeAll();
                break;
            }
        }
    }

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

int main(int argc, char* argv[])
{
    const char* keys =
        "{camera_model                | dualgopro     | camera model}"
        "{camera_param_path           | dualgopro.pts | camera parameter file path, may be xml file path or ptgui pts file path}"
        "{num_cameras                 | 2             | number of cameras}"
        "{camera_width                | 1920          | camera picture width}"
        "{camera_height               | 1080          | camera picture height}"
        "{frames_per_second           | 30            | camera frame rate}"
        "{pano_stream_frame_width     | 1440          | pano video live stream picture width}"
        "{pano_stream_frame_height    | 720           | pano video live stream picture height}"
        "{pano_stream_bits_per_second | 1000000       | pano video live stream bits per second}"
        "{pano_stream_encoder         | h264          | pano video live stream encoder}"
        "{pano_stream_encode_preset   | veryfast      | pano video live stream encode preset}"
        "{pano_stream_url             | rtmp://pili-publish.live.detu.com/detulive/detudemov550?key=detukey | pano live stream address}"
        "{pano_save_file              | false         | whether to save audio video to local hard disk}"
        "{pano_file_duration          | 60            | each local pano audio video file duration in seconds}"
        "{pano_file_frame_width       | 1440          | pano video local file picture width}"
        "{pano_file_frame_height      | 720           | pano video local file picture height}"
        "{pano_file_bits_per_second   | 1000000       | pano video local file bits per second}"
        "{pano_file_encoder           | h264          | pano video local file encoder}"
        "{pano_file_encode_preset     | veryfast      | pano video local file encode preset}"
        "{enable_audio                | false         | enable audio or not}"
        "{enable_interactive_select_devices | false   | enable interactice select devices}"
        "{enable_cuda                 | false         | enable cuda reproject and blend to render panorama image}";

    cv::CommandLineParser parser(argc, argv, keys);

    cameraModel = parser.get<std::string>("camera_model");

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
    fileEncoder = parser.get<std::string>("pano_file_encoder");
    if (fileEncoder != "h264_qsv")
        fileEncoder = "h264";
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
    
    bool ok;

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
    selectVideoDevices(interactive, numCameras, videoIndexes);

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
            ok = task.openAudioDevice(audioDevices[audioIndex], 44100);
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
    std::vector<avp::Option> opts;
    std::string frameRateStr = parser.get<std::string>("frames_per_second");
    std::string videoSizeStr = parser.get<std::string>("camera_width") + "x" + parser.get<std::string>("camera_height");
    opts.push_back(std::make_pair("framerate", frameRateStr));
    opts.push_back(std::make_pair("video_size", videoSizeStr));

    enableCuda = parser.get<bool>("enable_cuda");
    std::vector<avp::Device> newVideoDevices(numCameras);
    for (int i = 0; i < numCameras; i++)
        newVideoDevices[i] = videoDevices[videoIndexes[i]];
    std::vector<int> oks;
    ok = task.openVideoDevices(newVideoDevices, srcSize.width, srcSize.height, frameRate, oks);
    if (!ok)
    {
        for (int i = 0; i < numCameras; i++)
        {
            if (!oks[i])
            {
                printf("Could not open DirectShow video device %s[%s] with framerate = %s and video_size = %s\n",
                    videoDevices[videoIndexes[i]].shortName.c_str(), videoDevices[videoIndexes[i]].numString.c_str(),
                    frameRateStr.c_str(), videoSizeStr.c_str());
                return 0;
            }
        }
    }

    cameraParamPath = parser.get<std::string>("camera_param_path");
    bool equalStreamAndFile = streamFrameSize == fileFrameSize;
    ok = task.beginVideoStitch(cameraParamPath, saveFile ? fileFrameSize.width : streamFrameSize.width,
        saveFile ? fileFrameSize.height : streamFrameSize.height, enableCuda);
    if (!ok)
    {
        printf("Could not prepare for panorama render\n");
        return 0;
    }
    //printf("pass render prepare\n");

    streamURL = parser.get<std::string>("pano_stream_url");
    streamBitRate = parser.get<int>("pano_stream_bits_per_second");
    streamEncoder = parser.get<std::string>("pano_stream_encoder");
    if (streamEncoder != "h264_qsv")
        streamEncoder = "h264";
    streamEncodePreset = parser.get<std::string>("pano_stream_encode_preset");
    if (streamEncodePreset != "ultrafast" || streamEncodePreset != "superfast" ||
        streamEncodePreset != "veryfast" || streamEncodePreset != "faster" ||
        streamEncodePreset != "fast" || streamEncodePreset != "medium" || streamEncodePreset != "slow" ||
        streamEncodePreset != "slower" || streamEncodePreset != "veryslow")
        streamEncodePreset = "veryfast";
    if (streamURL.size() && streamURL != "null")
    {
        ok = task.openLiveStream(streamURL, streamFrameSize.width, streamFrameSize.height, 
            streamBitRate, streamEncoder, streamEncodePreset, 96000);
        if (!ok)
        {
            printf("Could not open rtmp streaming url with frame rate = %d and bit rate = %d\n", frameRate, streamBitRate);
            return 0;
        }
    }
    else
    {
        printf("pano_stream_url empty, no live stream\n");
    }

    if (saveFile)
    {
        task.beginSaveToDisk(".", fileFrameSize.width, fileFrameSize.height, 
            fileBitRate, fileEncoder, fileEncodePreset, 96000, fileDuration);
    }

    waitTime = std::max(5.0, 1000.0 / frameRate - 5);

    std::thread svr(showVideoResult);
    svr.join();

    task.closeVideoDevices();
    task.closeAudioDevice();
    task.stopVideoStitch();
    task.closeLiveStream();
    task.stopSaveToDisk();

    return 0;
}