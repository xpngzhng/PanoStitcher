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

        showWidth = 960;
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
            if (images[i].rows != origHeight || images[i].cols != origWidth || images[i].type() != CV_8UC4)
                return false;
        }

        tileImage.create(tileHeight, tileWidth, CV_8UC4);
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
int audioOpened;
int waitTime = 30;

bool highQualityBlend = true;

ShowTiledImages showTiledImages;
PanoramaLiveStreamTask2 task;

int prevCount = 0;
void showVideoSources()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    std::vector<avp::AudioVideoFrame2> frames;
    std::vector<cv::Mat> images(numCameras);
    while (true)
    {
        if (task.hasFinished())
            break;
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
            if (key >= 0)
                printf("pressed in %s\n", __FUNCTION__);
            if (key == 'q')
            {
                task.closeAll();
                break;
            }
            else if (key == 's')
            {
                char buf[64];
                for (int i = 0; i < numCameras; i++)
                {
                    sprintf(buf, "snapshot%d.bmp", i);
                    cv::imwrite(buf, images[i]);
                }
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
            break;
        task.getStitchedVideoFrame(frame);
        if (frame.data[0])
        {
            cv::Mat show(frame.height, frame.width, frame.pixelType == avp::PixelTypeBGR24 ? CV_8UC3 : CV_8UC4, frame.data[0], frame.steps[0]);
            cv::imshow("result", show);
            int key = cv::waitKey(waitTime / 2);
            if (key >= 0)
                printf("pressed in %s\n", __FUNCTION__);
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
        "{@url0                       |               |}"
        "{@url1                       |               |}"
        "{@url2                       |               |}"
        "{@url3                       |               |}"
        "{camera_model                | dualgopro     | camera model}"
        "{camera_param_path           | null          | camera parameter file path, may be xml file path or ptgui pts file path}"
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

    setAddWatermark(false);

    stitchFrameSize.width = parser.get<int>("pano_stitch_frame_width");
    stitchFrameSize.height = parser.get<int>("pano_stitch_frame_height");
    if (stitchFrameSize.width <= 0 || stitchFrameSize.height <= 0 ||
        (stitchFrameSize.width & 1) || (stitchFrameSize.height & 1) ||
        (stitchFrameSize.width != stitchFrameSize.height * 2))
    {
        printf("pano_stitch_frame_width and pano_stitch_frame_height should be positive even numbers, "
            "and pano_stitch_frame_width should be two times of pano_stitch_frame_height\n");
        return 0;
    }

    bool ok;
    std::vector<std::string> urls;
    //urls.push_back("192.168.1.204");
    //urls.push_back("192.168.1.205");
    //urls.push_back("192.168.1.206");
    //urls.push_back("192.168.1.207");
    std::string url;
    url = parser.get<std::string>("@url0");
    if (url.empty())
    {
        printf("url empty\n");
        return 0;
    }
    urls.push_back(url);
    url = parser.get<std::string>("@url1");
    if (url.empty())
    {
        printf("url empty\n");
        return 0;
    }
    urls.push_back(url);
    url = parser.get<std::string>("@url2");
    if (url.empty())
    {
        printf("url empty\n");
        return 0;
    }
    urls.push_back(url);
    url = parser.get<std::string>("@url3");
    if (url.empty())
    {
        printf("url empty\n");
        return 0;
    }
    urls.push_back(url);

    ok = task.openAudioVideoSources(urls);
    if (!ok)
    {
        printf("Could not open urls\n");
        return 0;
    }

    //cv::Size sz(2048, 1024);
    //cv::Size sz(3072, 1536);
    cv::Size sz(2048, 1024);

    highQualityBlend = parser.get<bool>("high_quality_blend");
    highQualityBlend = false;
    cameraParamPath = parser.get<std::string>("camera_param_path");
    stitchFrameSize = sz;
    if (cameraParamPath.size() && cameraParamPath != "null")
    {
        ok = task.beginVideoStitch(PanoStitchTypeMISO, cameraParamPath, stitchFrameSize.width, stitchFrameSize.height, highQualityBlend);
        if (!ok)
        {
            printf("Could not prepare for panorama render\n");
            task.closeAll();
            return 0;
        }
    }
    else
    {
        printf("camera_param_path empty, no stitch\n");
    }

    std::vector<double> exposures;
    ok = task.calcExposures(exposures);
    if (!ok)
    {
        std::string msg;
        task.getLastSyncErrorMessage(msg);
        printf("%s\n", msg.c_str());
        task.closeAll();
        return 0;
    }
    printf("exposures: ");
    for (int i = 0; i < exposures.size(); i++)
        printf("%f ", exposures[i]);
    printf("\n");
    task.setExposures(exposures);

    streamURL = parser.get<std::string>("pano_stream_url");
    //streamURL = "rtsp://192.168.1.234/test.sdp";/*"rtsp://127.0.0.1/test.sdp"*/
    //streamURL = "rtmp://110.172.214.59:80/live/myStream";
    streamURL = "null";
    //streamURL = "rtsp://127.0.0.1/test.sdp";
    //streamURL = "rtsp://192.168.1.163/test.sdp";
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
            task.closeAll();
            return 0;
        }

        streamBitRate = parser.get<int>("pano_stream_bits_per_second");
        streamEncoder = parser.get<std::string>("pano_stream_encoder");
        if (streamEncoder != "h264_qsv" && streamEncoder != "nvenc_h264")
            streamEncoder = "h264";
        streamEncoder = "h264_qsv";
        streamEncodePreset = parser.get<std::string>("pano_stream_encode_preset");
        if (streamEncodePreset != "ultrafast" || streamEncodePreset != "superfast" ||
            streamEncodePreset != "veryfast" || streamEncodePreset != "faster" ||
            streamEncodePreset != "fast" || streamEncodePreset != "medium" || streamEncodePreset != "slow" ||
            streamEncodePreset != "slower" || streamEncodePreset != "veryslow")
            streamEncodePreset = "veryfast";

        //streamFrameSize = sz;
        streamFrameSize = cv::Size(1536, 1024);
        streamBitRate = 1000000;
        ok = task.openLiveStream(streamURL, PanoTypeCube3x2, streamFrameSize.width, streamFrameSize.height,
            streamBitRate, streamEncoder, streamEncodePreset, 96000);
        if (!ok)
        {
            printf("Could not open streaming url with frame rate = %d and bit rate = %d\n", frameRate, streamBitRate);
            task.closeAll();
            return 0;
        }
    }
    else
    {
        printf("pano_stream_url empty, no live stream\n");
    }

    saveFile = parser.get<bool>("pano_save_file");
    saveFile = false;
    //saveFile = true;
    if (saveFile)
    {
        fileFrameSize.width = parser.get<int>("pano_file_frame_width");
        fileFrameSize.height = parser.get<int>("pano_file_frame_height");
        if (fileFrameSize.width <= 0 || fileFrameSize.height <= 0 ||
            (fileFrameSize.width & 1) || (fileFrameSize.height & 1) ||
            (fileFrameSize.width != fileFrameSize.height * 2))
        {
            printf("pano_file_frame_width and pano_file_frame_height should be positive even numbers, "
                "and pano_file_frame_width should be two times of pano_file_frame_height\n");
            task.closeAll();
            return 0;
        }

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

        fileEncoder = "h264_qsv";
        fileFrameSize.width = 4096;
        fileFrameSize.height = 2048;
        fileBitRate = 8000000;
        task.beginSaveToDisk(".", PanoTypeEquiRect, fileFrameSize.width, fileFrameSize.height,
            fileBitRate, fileEncoder, fileEncodePreset, 96000, fileDuration);
        //fileFrameSize.width = 3072;
        //fileFrameSize.height = 2048;
        //fileBitRate = 20000000;
        //fileDuration = 6000;
        //fileEncoder = "h264_qsv";
        //task.beginSaveToDisk(".", PanoTypeCube3x2, fileFrameSize.width, fileFrameSize.height,
        //    fileBitRate, fileEncoder, fileEncodePreset, 96000, fileDuration);
    }

    frameRate = task.getVideoFrameRate();
    waitTime = std::max(5.0, 1000.0 / frameRate - 5);

    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::vector<double> expos;
    ok = task.calcExposures(expos);
    {
        if (!ok)
        {
            printf("Failed to calc exposures\n");
            task.closeAll();
            return 0;
        }
    }
    printf("exposure: ");
    for (int i = 0; i < expos.size(); i++)
        printf("%f, ", expos[i]);
    printf("\n");
    task.setExposures(expos);

    numCameras = task.getNumVideos();
    showTiledImages.init(task.getVideoWidth(), task.getVideoHeight(), task.getNumVideos());
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