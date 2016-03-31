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

	ReprojectParam pi;
	pi.LoadConfig("F:\\QQRecord\\452103256\\FileRecv\\test1\\changtai_cam_param.xml");
	pi.SetPanoSize(dstSize);

    std::vector<std::string> srcVideoNames;
    srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0078.mp4");
    srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0081.mp4");
    srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0087.mp4");
    srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0108.mp4");
    srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0118.mp4");
    srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0518.mp4");
    int numVideos = srcVideoNames.size();

    int offset[] = {563, 0, 268, 651, 91, 412};
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

    //for (int i = 0; i < numVideos; i++)
    //{
    //    avp::BGRImage bgrImage;
    //    readers[i].read(bgrImage);
    //    cv::Mat image(bgrImage.height, bgrImage.width, CV_8UC3, bgrImage.data, bgrImage.step);
    //    char buf[256];
    //    sprintf_s(buf, 256, "image%d.bmp", i);
    //    cv::imwrite(buf, image);
    //}
    //return 0;

    std::vector<cv::Mat> dstMasks;
    std::vector<cv::Mat> dstSrcMaps, xmaps, ymaps;
    getReprojectMapsAndMasks(pi, srcSize, dstSrcMaps, dstMasks);
    //getReprojectMapsAndMasks(pi, srcSize, xmaps, ymaps, dstMasks);

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
        //xmaps[i].convertTo(map32F, CV_32F);
        //xmapsGpu[i].upload(map32F);
        //ymaps[i].convertTo(map32F, CV_32F);
        //ymapsGpu[i].upload(map32F);
    }
    //dstMasks.clear();
    dstSrcMaps.clear();
    map32F.release();
    map64F[0].release();
    map64F[1].release();

    //CudaTilingMultibandBlend blender;
    //bool success = blender.prepare(dstMasks, 16, 2);
    CudaTilingMultibandBlendFast blender;
    bool success = blender.prepare(dstMasks, 16, 2);
    //CudaTilingFeatherBlend blender;
    //bool success = blender.prepare(dstMasks);
    dstMasks.clear();

    int frameCount = 0;
    std::vector<avp::BGRImage> rawImages(numVideos);
    std::vector<cv::gpu::GpuMat> imagesGpu(numVideos), reprojImagesGpu(numVideos);
    cv::gpu::GpuMat blendImageGpu;
    cv::Mat blendImageCpu;

    //for (int i = 0; i < numVideos; i++)
    //{
    //    readers[i].read(rawImages[i]);
    //    cv::Mat imageCpu(rawImages[i].height, rawImages[i].width, CV_8UC4, rawImages[i].data, rawImages[i].step);
    //    imageGpu.upload(imageCpu);
    //    cudaReproject(imageGpu, reprojImagesGpu[i], xmapsGpu[i], ymapsGpu[i]);
    //    char buf[128];
    //    sprintf(buf, "reprojimagegpu%d.bmp", i);
    //    cv::Mat reprojImageC4;
    //    reprojImagesGpu[i].download(reprojImageC4);
    //    cv::Mat reprojImageC3(dstSize, CV_8UC3);
    //    int fromTo[] = {0, 0, 1, 1, 2, 2};
    //    cv::mixChannels(&reprojImageC4, 1, &reprojImageC3, 1, fromTo, 3);
    //    cv::imwrite(buf, reprojImageC3);
    //}
    //return 0;

    ztool::Timer timerAll, timerTotal, timerDecode, timerUpload, timerReproject, timerBlend, timerDownload, timerEncode;

    avp::VideoWriter writer;
    writer.open("mbrblendfast2.mp4", avp::PixelTypeBGR32, dstSize.width, dstSize.height, 48);
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
            if (!readers[i].read(rawImages[i]))
            {
                success = false;
                break;
            }
        }
        timerDecode.end();
        if (!success)
            break;

        //if (frameCount == 1)
        //{
        //    ztool::Timer timer;
        //    std::vector<cv::Mat> imagesCpu(numVideos);
        //    std::vector<cv::gpu::GpuMat> imagesGpu(numVideos);
        //    timer.start();
        //    for (int i = 0; i < numVideos; i++)
        //    {
        //        imagesCpu[i] = cv::Mat(rawImages[i].height, rawImages[i].width, CV_8UC4, rawImages[i].data, rawImages[i].step);
        //        imagesGpu[i].upload(imagesCpu[i]);
        //    }
        //    timer.end();
        //    printf("upload 8UC4 time = %f\n", timer.elapse());

        //    for (int run = 0; run < 200; run++)
        //    {
        //        timer.start();
        //        std::vector<cv::Mat> imagesCpu(numVideos);
        //        std::vector<cv::gpu::GpuMat> imagesGpu(numVideos);
        //        std::vector<cv::gpu::GpuMat> rprjImagesGpu(numVideos);
        //        for (int i = 0; i < numVideos; i++)
        //        {
        //            imagesCpu[i] = cv::Mat(rawImages[i].height, rawImages[i].width, CV_8UC4, rawImages[i].data, rawImages[i].step);
        //            imagesGpu[i].upload(imagesCpu[i]);
        //            cudaReproject(imagesGpu[i], rprjImagesGpu[i], xmapsGpu[i], ymapsGpu[i]);
        //        }
        //        timer.end();
        //        printf("reproject time = %f\n", timer.elapse());
        //    }

        //    break;
        //}

        timerUpload.start();
        for (int i = 0; i < numVideos; i++)
        {
            cv::Mat imageCpu(rawImages[i].height, rawImages[i].width, CV_8UC4, rawImages[i].data, rawImages[i].step);
            imagesGpu[i].upload(imageCpu);
        }
        timerUpload.end();

        /*for (int run = 0; run < 2000; run++)
        {
            printf("%d ", run);
            for (int i = 0; i < numVideos; i++)
            {
                cudaReproject(imagesGpu[i], reprojImagesGpu[i], xmapsGpu[i], ymapsGpu[i]);
            }
            blender.blend(reprojImagesGpu, dstMasksGpu, blendImageGpu);
        }
        break;*/

        timerReproject.start();
        for (int i = 0; i < numVideos; i++)
        {
            cudaReproject(imagesGpu[i], reprojImagesGpu[i], xmapsGpu[i], ymapsGpu[i]);
        }
        timerReproject.end();

        //for (int run = 0; run < 200; run++)
        //{
        //    ztool::Timer timer;
        //    timer.start();
        //    blender.blend(reprojImagesGpu, dstMasksGpu, blendImageGpu);
        //    timer.end();
        //    printf("blend time elasped = %f\n", timer.elapse());
        //}
        //break;

        timerBlend.start();
        //blender.blend(reprojImagesGpu, dstMasksGpu, blendImageGpu);
        //blender.blend(reprojImagesGpu, blendImageGpu);
        blender.blend(reprojImagesGpu, blendImageGpu);
        timerBlend.end();

        timerDownload.start();
        blendImageGpu.download(blendImageCpu);
        timerDownload.end();

        timerEncode.start();
        avp::BGRImage image(blendImageCpu.data, blendImageCpu.cols, blendImageCpu.rows, blendImageCpu.step);
        writer.write(image);
        timerEncode.end();

        timerTotal.end();
        printf("time elapsed = %f, dec = %f, upload = %f, proj = %f, blend = %f, download = %f, enc = %f\n", 
            timerTotal.elapse(), timerDecode.elapse(), timerUpload.elapse(), timerReproject.elapse(), 
            timerBlend.elapse(), timerDownload.elapse(), timerEncode.elapse());
        //printf("time = %f\n", timerTotal.elapse());
    }
    timerAll.end();
    printf("all time %f\n", timerAll.elapse());
    return 0;
}