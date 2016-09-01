#include "RicohUtil.h"
#include "Blend/ZBlend.h"
#include "Warp/ZReproject.h"
#include "Tool/Timer.h"
#include "AudioVideoProcessor.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

int main()
{
    int frameCount = 0;
    avp::AudioVideoFrame frame;
    cv::Mat image, image1, image2;
    cv::Mat blendImage;

    ztool::Timer timerAll, timerTotal, timerDecode, timerReproject, timerBlend, timerEncode;

    avp::AudioVideoReader reader;
    //reader.open("F:\\QQRecord\\452103256\\FileRecv\\vlc-record-2016-06-16-13h42m11s-rtsp___192.168.1.254-.mp4", false, true, avp::PixelTypeBGR24);
    //reader.open("F:\\panovideo\\ricoh\\R0010113.MP4", false, true, avp::PixelTypeBGR24);
    reader.open("F:\\panovideo\\ricoh m15\\R0010128.MOV", false, true, avp::PixelTypeBGR24);

    cv::Size dstSize = cv::Size(2048, 1024);
    cv::Size srcSize = cv::Size(reader.getVideoWidth(), reader.getVideoHeight());

    RicohPanoramaRender render;
    //render.prepare("F:\\QQRecord\\452103256\\FileRecv\\45678-mod.xml", srcSize, dstSize);
    //render.prepare("F:\\panovideo\\ricoh\\paramricoh.xml", srcSize, dstSize);
    render.prepare("F:\\panovideo\\ricoh m15\\param.xml", srcSize, dstSize);

    avp::AudioVideoWriter writer;
    writer.open("ricohm15.mp4", "", false, 
        false, "", 0, 0, 0, 0, 
        true, "", avp::PixelTypeBGR24, dstSize.width, dstSize.height, reader.getVideoFrameRate(), 8000000);

    int failCount = 0;
    timerAll.start();
    while (true)
    {
        printf("currCount = %d\n", frameCount++);
        if (frameCount >= 4800)
            break;

        timerTotal.start();

        bool success = true;
        timerDecode.start();
        success = reader.read(frame);
        timerDecode.end();
        if (!success)
        {
            if (++failCount < 100)
                continue;
            break;
        }

        cv::Mat raw(frame.height, frame.width, CV_8UC3, frame.data, frame.step);

        timerReproject.start();
        render.render(raw, blendImage);
        timerReproject.end();

        cv::imshow("blend image", blendImage);
        cv::waitKey(1);

        timerBlend.start();
        timerBlend.end();

        timerEncode.start();
        avp::AudioVideoFrame image = avp::videoFrame(blendImage.data, blendImage.step, 
            avp::PixelTypeBGR24, blendImage.cols, blendImage.rows);
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