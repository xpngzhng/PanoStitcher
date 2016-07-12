#include "PanoramaTask.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <thread>

struct ShowTiledImages
{
    ShowTiledImages() : hasInit(false) {};
    bool init(int width_, int height_, int numImages_)
    {
        origWidth = width_;
        origHeight = height_;
        numImages = numImages_;

        showWidth = 480;
        showHeight = origHeight * double(showWidth) / double(origWidth) + 0.5;

        int totalWidth = numImages * showWidth;
        if (totalWidth <= screenWidth)
            tileWidth = numImages * showWidth;
        else
            tileWidth = screenWidth;
        tileHeight = ((totalWidth + screenWidth - 1) / screenWidth) * showHeight;

        int horiNumImages = screenWidth / showWidth;
        locations.resize(numImages);
        for (int i = 0; i < numImages; i++)
        {
            int gridx = i % horiNumImages;
            int gridy = i / horiNumImages;
            locations[i] = cv::Rect(gridx * showWidth, gridy * showHeight, showWidth, showHeight);
        }

        hasInit = true;
        return true;
    }
    bool show(const std::string& winName, const std::vector<cv::Mat>& images)
    {
        if (!hasInit)
            return false;

        if (images.size() != numImages)
            return false;

        for (int i = 0; i < numImages; i++)
        {
            if (images[i].rows != origHeight || images[i].cols != origWidth ||
                (images[i].type() != CV_8UC4 && images[i].type() != CV_8UC3))
                return false;
        }

        tileImage.create(tileHeight, tileWidth, images[0].type());
        for (int i = 0; i < numImages; i++)
        {
            cv::Mat curr = tileImage(locations[i]);
            cv::resize(images[i], curr, cv::Size(showWidth, showHeight), 0, 0, CV_INTER_NN);
        }
        cv::imshow(winName, tileImage);

        return true;
    }
    
    const int screenWidth = 1920;
    int origWidth, origHeight;
    int showWidth, showHeight;
    int numImages;
    int tileWidth, tileHeight;
    cv::Mat tileImage;
    std::vector<cv::Rect> locations;    
    bool hasInit;
};

cv::Size srcSize(1920, 1080);

int frameRate;
int audioBitRate = 96000;
cv::Size stitchFrameSize(1440, 720);

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
int waitTime = 30;

bool highQualityBlend = true;

ShowTiledImages showTiledImages;
PanoramaLiveStreamTask2 task;

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

    std::vector<avp::AudioVideoFrame2> frames;
    std::vector<cv::Mat> images(numCameras);
    while (true)
    {
        if (task.hasFinished())
        {
            printf("%s break\n", __FUNCTION__);
            break;
        }
        task.getVideoSourceFrames(frames);
        if (frames.size() == numCameras)
        {
            for (int i = 0; i < numCameras; i++)
            {
                images[i] = cv::Mat(frames[i].height, frames[i].width,
                    frames[i].pixelType == avp::PixelTypeBGR24 ? CV_8UC3 : CV_8UC4, frames[i].data[0], frames[i].steps[0]);
            }
            showTiledImages.show("src images", images);
            int key = cv::waitKey(waitTime / 2);
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

    avp::AudioVideoFrame2 frame;
    while (true)
    {
        if (task.hasFinished())
        {
            printf("%s break\n", __FUNCTION__);
            break;
        }            
        task.getStitchedVideoFrame(frame);
        if (frame.data[0])
        {
            cv::Mat show(frame.height, frame.width, frame.pixelType == avp::PixelTypeBGR24 ? CV_8UC3 : CV_8UC4, frame.data[0], frame.steps[0]);
            cv::imshow("result", show);
            int key = cv::waitKey(waitTime / 2);
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
        "{pano_stitch_frame_width     | 1440          | pano video picture width}"
        "{pano_stitch_frame_height    | 720           | pano video picture height}"
        "{pano_stream_frame_width     | 1440          | pano video live stream picture width}"
        "{pano_stream_frame_height    | 720           | pano video live stream picture height}"
        "{pano_stream_bits_per_second | 1000000       | pano video live stream bits per second}"
        "{pano_stream_encoder         | h264          | pano video live stream encoder}"
        "{pano_stream_encode_preset   | veryfast      | pano video live stream encode preset}"
        "{pano_stream_url             | rtsp://127.0.0.1/test.sdp | pano live stream address}"
        "{pano_save_file              | false         | whether to save audio video to local hard disk}"
        "{pano_file_duration          | 60            | each local pano audio video file duration in seconds}"
        "{pano_file_frame_width       | 1440          | pano video local file picture width}"
        "{pano_file_frame_height      | 720           | pano video local file picture height}"
        "{pano_file_bits_per_second   | 1000000       | pano video local file bits per second}"
        "{pano_file_encoder           | h264          | pano video local file encoder}"
        "{pano_file_encode_preset     | veryfast      | pano video local file encode preset}"
        "{enable_audio                | false         | enable audio or not}"
        "{enable_interactive_select_devices | false   | enable interactice select devices}"
        "{high_quality_blend          | false         | use multiband blend}";

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

    cv::Size sz(2048, 1024);

    stitchFrameSize.width = parser.get<int>("pano_stitch_frame_width");
    stitchFrameSize.height = parser.get<int>("pano_stitch_frame_height");
    stitchFrameSize = sz;
    if (stitchFrameSize.width <= 0 || stitchFrameSize.height <= 0 ||
        (stitchFrameSize.width & 1) || (stitchFrameSize.height & 1) ||
        (stitchFrameSize.width != stitchFrameSize.height * 2))
    {
        printf("pano_stitch_frame_width and pano_stitch_frame_height should be positive even numbers, "
            "and pano_stitch_frame_width should be two times of pano_stitch_frame_height\n");
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
    int audioIndex = -1;
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

            if (interactive)
            {
                while (true)
                {
                    printf("Select audio device, input the number in parentheses\"()\": ");
                    int val;
                    std::cin >> val;
                    if (val < 0 || val >= numAudioDevices)
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
        }
    }

    frameRate = parser.get<int>("frames_per_second");
    std::vector<avp::Option> opts;
    std::string frameRateStr = parser.get<std::string>("frames_per_second");
    std::string videoSizeStr = parser.get<std::string>("camera_width") + "x" + parser.get<std::string>("camera_height");
    opts.push_back(std::make_pair("framerate", frameRateStr));
    opts.push_back(std::make_pair("video_size", videoSizeStr));

    highQualityBlend = parser.get<bool>("high_quality_blend");
    std::vector<avp::Device> newVideoDevices(numCameras);
    for (int i = 0; i < numCameras; i++)
        newVideoDevices[i] = videoDevices[videoIndexes[i]];
    ok = task.openAudioVideoSources(newVideoDevices, srcSize.width, srcSize.height, frameRate, 
        audioIndex >= 0, audioIndex >= 0 ? audioDevices[audioIndex] : avp::Device(), 44100);
    if (!ok)
    {
        printf("DirectShow devices open failed\n");
        return 0;
    }

    cameraParamPath = parser.get<std::string>("camera_param_path");
    ok = task.beginVideoStitch(cameraParamPath, stitchFrameSize.width, stitchFrameSize.height, highQualityBlend);
    if (!ok)
    {
        printf("Could not prepare for panorama render\n");
        return 0;
    }
    //printf("pass render prepare\n");

    streamURL = parser.get<std::string>("pano_stream_url");
    //streamURL = "rtsp://127.0.0.1/test.sdp";
    streamURL = "null";
    if (streamURL.size() && streamURL != "null")
    {
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

        streamBitRate = parser.get<int>("pano_stream_bits_per_second");
        streamEncoder = parser.get<std::string>("pano_stream_encoder");
        if (streamEncoder != "h264_qsv" && streamEncoder != "nvenc_h264")
            streamEncoder = "h264";
        streamEncoder = "nvenc_h264";
        streamEncodePreset = parser.get<std::string>("pano_stream_encode_preset");
        if (streamEncodePreset != "ultrafast" || streamEncodePreset != "superfast" ||
            streamEncodePreset != "veryfast" || streamEncodePreset != "faster" ||
            streamEncodePreset != "fast" || streamEncodePreset != "medium" || streamEncodePreset != "slow" ||
            streamEncodePreset != "slower" || streamEncodePreset != "veryslow")
            streamEncodePreset = "veryfast";

        //ok = task.openLiveStream(streamURL, PanoTypeEquiRect, streamFrameSize.width, streamFrameSize.height,
        //    streamBitRate, streamEncoder, streamEncodePreset, 96000);
        ok = task.openLiveStream(streamURL, PanoTypeCube3x2, 1800, 1200,
            streamBitRate, streamEncoder, streamEncodePreset, 96000);
        //ok = task.openLiveStream(streamURL, PanoTypeCube180, 2500, 1500,
        //    streamBitRate, streamEncoder, streamEncodePreset, 96000);
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

    saveFile = parser.get<bool>("pano_save_file");
    saveFile = true;
    fileFrameSize = stitchFrameSize;
    if (saveFile)
    {
        fileFrameSize.width = parser.get<int>("pano_file_frame_width");
        fileFrameSize.height = parser.get<int>("pano_file_frame_height");
        fileFrameSize = sz;
        if (fileFrameSize.width <= 0 || fileFrameSize.height <= 0 ||
            (fileFrameSize.width & 1) || (fileFrameSize.height & 1) ||
            (fileFrameSize.width != fileFrameSize.height * 2))
        {
            printf("pano_file_frame_width and pano_file_frame_height should be positive even numbers, "
                "and pano_file_frame_width should be two times of pano_file_frame_height\n");
            return 0;
        }

        fileDuration = parser.get<int>("pano_file_duration");
        fileBitRate = parser.get<int>("pano_file_bits_per_second");
        fileBitRate = 4000000;
        fileEncoder = parser.get<std::string>("pano_file_encoder");
        if (streamEncoder != "h264_qsv" && streamEncoder != "nvenc_h264")
            streamEncoder = "h264";
        fileEncoder = "h264";
        fileEncodePreset = parser.get<std::string>("pano_file_encode_preset");
        if (fileEncodePreset != "ultrafast" || fileEncodePreset != "superfast" ||
            fileEncodePreset != "veryfast" || fileEncodePreset != "faster" ||
            fileEncodePreset != "fast" || fileEncodePreset != "medium" || fileEncodePreset != "slow" ||
            fileEncodePreset != "slower" || fileEncodePreset != "veryslow")
            fileEncodePreset = "veryfast";

        //task.beginSaveToDisk(".", PanoTypeEquiRect, fileFrameSize.width, fileFrameSize.height, 
        //    fileBitRate, fileEncoder, fileEncodePreset, 96000, fileDuration);
        task.beginSaveToDisk(".", PanoTypeCube3x2, 1536, 1024,
            fileBitRate, fileEncoder, fileEncodePreset, 96000, fileDuration);
    }

    waitTime = std::max(5.0, 1000.0 / frameRate - 5);

    showTiledImages.init(srcSize.width, srcSize.height, numCameras);
    std::thread svr(showVideoResult);    
    std::thread svs(showVideoSources);
    svr.join();
    svs.join();

    task.closeAudioVideoSources();
    task.stopVideoStitch();
    task.closeLiveStream();
    task.stopSaveToDisk();

    return 0;
}