#include "RicohUtil.h"
#include "ZBlend.h"
#include "ReprojectionParam.h"
#include "Reprojection.h"
#include "ZReproject.h"
#include "AudioVideoProcessor.h"
#include "Timer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/gpumat.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
    cv::Size dstSize = cv::Size(2048, 1024);
    cv::Size srcSize = cv::Size(1920, 1080);
    cv::Size srcSizeLeft = cv::Size(960, 1080);
    cv::Size srcSizeRight = cv::Size(960, 1080);

    RicohPanoramaRender render;
    render.prepare("paramricoh.xml", srcSize, dstSize);

    int frameCount = 0;
    avp::BGRImage rawImage;
    cv::Mat image, image1, image2;
    cv::Mat blendImage;

    ztool::Timer timerAll, timerTotal, timerDecode, timerReproject, timerBlend, timerEncode;

    avp::VideoReader reader;
    reader.open("F:\\panovideo\\ricoh\\R0010008.MP4", avp::PixelTypeBGR24);
    avp::VideoWriter writer;
    writer.open("R0010008_our.mp4", avp::PixelTypeBGR24, dstSize.width, dstSize.height, 48, 8000000, avp::EncodeSpeedSlow);

    for (int i = 0; i < 200; i++)
        reader.read(rawImage);

    timerAll.start();
    while (true)
    {
        printf("currCount = %d\n", frameCount++);
        if (frameCount >= 4800)
            break;

        timerTotal.start();

        bool success = true;
        timerDecode.start();
        success = reader.read(rawImage);
        timerDecode.end();
        if (!success)
            break;

        cv::Mat raw(rawImage.height, rawImage.width, CV_8UC3, rawImage.data, rawImage.step);

        timerReproject.start();
        render.render(raw, blendImage);
        timerReproject.end();

        cv::imshow("blend image", blendImage);
        cv::waitKey(1);

        timerBlend.start();
        timerBlend.end();

        timerEncode.start();
        avp::BGRImage image(blendImage.data, blendImage.cols, blendImage.rows, blendImage.step);
        writer.write(image);
        timerEncode.end();

        timerTotal.end();
        printf("time elapsed = %f, dec = %f, proj = %f, blend = %f, enc = %f\n",
            timerTotal.elapse(), timerDecode.elapse(), timerReproject.elapse(),
            timerBlend.elapse(), timerEncode.elapse());
        printf("time = %f\n", timerTotal.elapse());
    }
    timerAll.end();
    printf("all time %f\n", timerAll.elapse());

    reader.close();
    writer.close();
    return 0;
}