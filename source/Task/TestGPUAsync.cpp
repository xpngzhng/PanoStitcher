#include "ZBlend.h"
#include "ReprojectionParam.h"
#include "Reprojection.h"
#include "AudioVideoProcessor.h"
#include "Timer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/gpumat.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
    cv::Size dstSize = cv::Size(2048, 1024);
    cv::Size srcSize = cv::Size(1280, 960);

    //ReprojectParam pi;
    //pi.LoadConfig("F:\\QQRecord\\452103256\\FileRecv\\test1\\changtai_cam_param.xml");
    //pi.SetPanoSize(dstSize);

    //std::vector<std::string> srcVideoNames;
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0078.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0081.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0087.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0108.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0118.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0518.mp4");
    //int numVideos = srcVideoNames.size();

    //int offset[] = { 563, 0, 268, 651, 91, 412 };
    //int numSkip = 200/*1*//*2100*/;

    ReprojectParam pi;
    pi.LoadConfig("F:\\panovideo\\test\\test1\\test_test1_cam_param.xml");
    pi.SetPanoSize(dstSize);

    std::vector<std::string> srcVideoNames;
    srcVideoNames.push_back("F:\\panovideo\\test\\test1\\YDXJ0094.mp4");
    srcVideoNames.push_back("F:\\panovideo\\test\\test1\\YDXJ0096.mp4");
    srcVideoNames.push_back("F:\\panovideo\\test\\test1\\YDXJ0103.mp4");
    srcVideoNames.push_back("F:\\panovideo\\test\\test1\\YDXJ0124.mp4");
    srcVideoNames.push_back("F:\\panovideo\\test\\test1\\YDXJ0136.mp4");
    srcVideoNames.push_back("F:\\panovideo\\test\\test1\\YDXJ0535.mp4");
    int numVideos = srcVideoNames.size();

    int offset[] = { 0, 198, 246, 283, 144, 373 };
    int numSkip = 200/*1*//*2100*/;

    //ReprojectParam pi;
    //pi.LoadConfig("F:\\QQRecord\\452103256\\FileRecv\\test2\\changtai.xml");
    //pi.SetPanoSize(frameSize);

    //std::vector<std::string> srcVideoNames;
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0072.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0075.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0080.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0101.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0112.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0512.mp4");
    //int numVideos = srcVideoNames.size();

    //int offset[] = {554, 0, 436, 1064, 164, 785};
    //int numSkip = 3000;

    std::vector<avp::VideoReader> readers(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        readers[i].open(srcVideoNames[i], avp::PixelTypeBGR32/*avp::PixelTypeBGR24*/);
        int count = offset[i] + numSkip;
        avp::BGRImage image;
        for (int j = 0; j < count; j++)
            readers[i].read(image);
    }

    std::vector<cv::Mat> dstMasks;
    std::vector<cv::Mat> dstSrcMaps, xmaps, ymaps;
    getReprojectMapsAndMasks(pi, srcSize, dstSrcMaps, dstMasks);

    std::vector<cv::gpu::GpuMat> dstMasksGpu(numVideos);
    std::vector<cv::gpu::GpuMat> xmapsGpu(numVideos), ymapsGpu(numVideos);
    cv::Mat map32F;
    cv::Mat map64F[2];
    for (int i = 0; i < numVideos; i++)
    {
        dstMasksGpu[i].upload(dstMasks[i]);
        cv::split(dstSrcMaps[i], map64F);
        map64F[0].convertTo(map32F, CV_32F);
        xmapsGpu[i].upload(map32F);
        map64F[1].convertTo(map32F, CV_32F);
        ymapsGpu[i].upload(map32F);
    }
    dstSrcMaps.clear();
    map32F.release();
    map64F[0].release();
    map64F[1].release();

    CudaTilingMultibandBlendFast blender;
    bool success = blender.prepare(dstMasks, 16, 2);
    dstMasks.clear();

    std::vector<cv::gpu::Stream> streams(numVideos);
    std::vector<cv::gpu::CudaMem> pinnedMems(numVideos);
    std::vector<avp::BGRImage> rawImages(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        pinnedMems[i].create(srcSize, CV_8UC4);
        rawImages[i].data = pinnedMems[i].data;
        rawImages[i].width = pinnedMems[i].cols;
        rawImages[i].height = pinnedMems[i].rows;
        rawImages[i].step = pinnedMems[i].step;
    }

    std::vector<cv::gpu::GpuMat> imagesGpu(numVideos), reprojImagesGpu(numVideos);
    cv::gpu::GpuMat blendImageGpu;
    cv::Mat blendImageCpu;

    ztool::Timer timerAll, timerTotal, timerDecode, timerUpload, timerReproject, timerBlend, timerDownload, timerEncode;

    int frameCount = 0;
    avp::VideoWriter writer;
    writer.open("mbrblendfastasyncmp4.mp4", avp::PixelTypeBGR32, dstSize.width, dstSize.height, 48);
    timerAll.start();
    while (true)
    {
        printf("currCount = %d\n", frameCount++);
        if (frameCount >= 2400)
            break;

        timerTotal.start();

        bool success = true;
        timerDecode.start();
        for (int i = 0; i < numVideos; i++)
        {
            if (!readers[i].read(rawImages[i], true))
            {
                success = false;
                break;
            }
        }
        timerDecode.end();
        if (!success)
            break;

        //timerUpload.start();
        for (int i = 0; i < numVideos; i++)
        {
            cv::Mat imageCpu(rawImages[i].height, rawImages[i].width, CV_8UC4, rawImages[i].data, rawImages[i].step);
            streams[i].enqueueUpload(pinnedMems[i], imagesGpu[i]);
        }
        //timerUpload.end();

        //timerReproject.start();
        for (int i = 0; i < numVideos; i++)
        {
            cudaReprojectTo16S(imagesGpu[i], reprojImagesGpu[i], xmapsGpu[i], ymapsGpu[i], streams[i]);
        }
        //timerReproject.end();

        for (int i = 0; i < numVideos; i++)
            streams[i].waitForCompletion();

        //timerBlend.start();
        blender.blend(reprojImagesGpu, blendImageGpu);
        //timerBlend.end();

        //timerDownload.start();
        blendImageGpu.download(blendImageCpu);
        //timerDownload.end();

        timerEncode.start();
        avp::BGRImage image(blendImageCpu.data, blendImageCpu.cols, blendImageCpu.rows, blendImageCpu.step);
        writer.write(image);
        timerEncode.end();

        timerTotal.end();
        //printf("time elapsed = %f, dec = %f, upload = %f, proj = %f, blend = %f, download = %f, enc = %f\n",
        //    timerTotal.elapse(), timerDecode.elapse(), timerUpload.elapse(), timerReproject.elapse(),
        //    timerBlend.elapse(), timerDownload.elapse(), timerEncode.elapse());
        printf("time = %f, dec = %f, enc = %f, other = %f\n", timerTotal.elapse(), 
            timerDecode.elapse(), timerEncode.elapse(), timerTotal.elapse() - timerDecode.elapse() - timerEncode.elapse());
    }
    timerAll.end();
    printf("all time %f\n", timerAll.elapse());
    return 0;
}