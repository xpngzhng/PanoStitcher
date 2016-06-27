#include "PanoramaTaskUtil.h"
#include "ZReproject.h"
#include "ZBlend.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

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
                ((images[i].type() != CV_8UC4) && (images[i].type() != CV_8UC3)))
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

int main()
{
    std::string configFileName = "F:\\panovideo\\test\\test6\\zhanxiang.xml";

    std::vector<std::string> fileNames;
    std::vector<int> offsets;
    loadVideoFileNamesAndOffset(configFileName, fileNames, offsets);

    int numVideos = fileNames.size();
    int globalOffset = 1095;
    for (int i = 0; i < numVideos; i++)
        offsets[i] += globalOffset;

    std::vector<avp::AudioVideoReader3> readers;
    cv::Size srcSize;
    int audioIndex, validFrameCount;
    prepareSrcVideos(fileNames, avp::PixelTypeBGR24, offsets, -1, readers, audioIndex, srcSize, validFrameCount);

    ShowTiledImages shower;
    shower.init(srcSize.width, srcSize.height, numVideos);

    cv::Size dstSize(1280, 640);
    std::vector<PhotoParam> photoParams;
    loadPhotoParams(configFileName, photoParams);
    std::vector<cv::Mat> masks, maps;
    getReprojectMapsAndMasks(photoParams, srcSize, dstSize, maps, masks);

    std::vector<cv::Mat> images(numVideos), reprojImages(numVideos), adjustImages(numVideos);
    TilingLinearBlend linearBlender;
    linearBlender.prepare(masks, 75);
    TilingMultibandBlendFast multiBlender;
    multiBlender.prepare(masks, 10, 2);
    cv::Mat bareBlend, adjustLinearBlend, adjustMultiBlend;

    std::vector<std::vector<unsigned char> > luts;
    std::vector<int> corrected;

    std::vector<avp::AudioVideoFrame2> frames(numVideos);
    while (true)
    {
        bool ok = true;
        for (int i = 0; i < numVideos; i++)
        {
            ok = readers[i].read(frames[i]);
            if (!ok)
            {
                break;
            }
        }
        if (!ok)
            break;

        for (int i = 0; i < numVideos; i++)
            images[i] = cv::Mat(srcSize, CV_8UC3, frames[i].data[0], frames[i].steps[0]);
        reprojectParallel(images, reprojImages, maps);
        
        linearBlender.blend(reprojImages, bareBlend);

        exposureCorrect(reprojImages, masks, luts, corrected);
        for (int i = 0; i < numVideos; i++)
            transform(reprojImages[i], adjustImages[i], luts[i], masks[i]);
        linearBlender.blend(adjustImages, adjustLinearBlend);
        multiBlender.blend(adjustImages, adjustMultiBlend);

        shower.show("src", images);
        cv::imshow("bare", bareBlend);
        cv::imshow("ajudst linear", adjustLinearBlend);
        cv::imshow("adjust multiband", adjustMultiBlend);
        int key = cv::waitKey(0);
        if (key == 'q')
            break;

        for (int i = 0; i < numVideos; i++)
        {
            for (int j = 0; j < 20; j++)
            {
                ok = readers[i].read(frames[i]);
                if (!ok)
                {
                    break;
                }
            }
        }
        if (!ok)
            break;
    }

    return 0;
}