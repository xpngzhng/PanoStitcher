#include "AudioVideoProcessor.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

int main1()
{
    std::vector<std::string> names;
    names.push_back("F:\\panovideo\\test\\SP7\\1-7.MP4");
    names.push_back("F:\\panovideo\\test\\SP7\\2-7.MP4");
    names.push_back("F:\\panovideo\\test\\SP7\\3-7.MP4");
    names.push_back("F:\\panovideo\\test\\SP7\\4-7.MP4");
    names.push_back("F:\\panovideo\\test\\SP7\\5-7.MP4");
    names.push_back("F:\\panovideo\\test\\SP7\\6-7.MP4");
    char buf[128];
    int numVideos = names.size();
    for (int i = 0; i < numVideos; i++)
    {
        avp::AudioVideoReader reader;
        avp::AudioVideoWriter writer;
        avp::AudioVideoFrame srcFrame;
        cv::Size dstSize(1920, 1440);
        cv::Mat dstMat(dstSize, CV_8UC3);

        reader.open(names[i], false, true, avp::PixelTypeBGR24);
        sprintf(buf, "F:\\panovideo\\test\\SP7\\%d-7-l.mp4", i + 1);
        writer.open(buf, "", false, false, "", 0, 0, 0, 0,
            true, "", avp::PixelTypeBGR24, dstSize.width, dstSize.height, 
            reader.getVideoFps() + 0.5, 36000000);
        
        while (reader.read(srcFrame))
        {
            if (srcFrame.mediaType == avp::VIDEO)
            {
                /*cv::Mat srcMat(srcFrame.height, srcFrame.width, CV_8UC3, srcFrame.data, srcFrame.step);
                cv::resize(srcMat, dstMat, dstSize);
                avp::AudioVideoFrame dstFrame = avp::videoFrame(dstMat.data, dstMat.step, avp::PixelTypeBGR24,
                    dstMat.cols, dstMat.rows, -1LL);
                writer.write(dstFrame);*/
                writer.write(srcFrame);
            }
        }
    }
    return 0;
}