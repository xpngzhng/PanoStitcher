#include "LiveStreamTaskUtil.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

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

ShowTiledImages showTiledImages;
int numVideos;
int waitTime;
int globalFinish = 0;

ForShowFrameVectorQueue syncedFramesBufferForShow;
BoundedPinnedMemoryFrameQueue syncedFramesBufferForProc;
ForceWaitFrameQueue procFrameBufferForSend, procFrameBufferForSave;
FFmpegAudioVideoSource* ptrSource;

void ShowThread()
{
    size_t id = std::this_thread::get_id().hash();
    printf("Thread %s [%8x] started\n", __FUNCTION__, id);

    std::vector<avp::SharedAudioVideoFrame> frames;
    std::vector<cv::Mat> images(numVideos);
    while (true)
    {
        if (!syncedFramesBufferForShow.pull(frames))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        if (frames.size() == numVideos)
        {
            //printf("first frame pts = %lld\n", frames[0].timeStamp);
            printf("pts: ");
            for (int i = 0; i < numVideos; i++)
            {
                printf("%lld ", frames[i].timeStamp);
            }
            printf("\n");
            for (int i = 0; i < numVideos; i++)
            {
                images[i] = cv::Mat(frames[i].height, frames[i].width,
                    frames[i].pixelType == avp::PixelTypeBGR24 ? CV_8UC3 : CV_8UC4, frames[i].data, frames[i].step);
            }
            showTiledImages.show("src images", images);
            int key = cv::waitKey(waitTime / 2);
            if (key == 'q')
            {
                ptrSource->close();
                break;
            }
        }
    }

    printf("Thread %s [%8x] end\n", __FUNCTION__, id);
}

int main()
{
    std::vector<avp::Device> ds, vds;
    avp::listDirectShowDevices(ds);
    avp::keepVideoDirectShowDevices(ds, vds);

    ptrSource = new FFmpegAudioVideoSource(&syncedFramesBufferForShow, &syncedFramesBufferForProc, 
        &procFrameBufferForSend, &procFrameBufferForSave, &globalFinish);

    ptrSource->open(vds, 1920, 1080, 30);

    numVideos = vds.size();
    waitTime = 30;
    showTiledImages.init(1920, 1080, vds.size());

    std::thread showThread(ShowThread);
    showThread.join();
    delete ptrSource;
    return 0;
}