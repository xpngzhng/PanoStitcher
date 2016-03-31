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
    //cv::Size srcSizeLeft = cv::Size(922, 922);
    //cv::Size srcSizeRight = cv::Size(912, 912);
    //cv::Rect srcRectLeft(10, 17, 922, 922);
    //cv::Rect srcRectRight(25, 19, 912, 912);
    cv::Size srcSizeLeft = cv::Size(890, 890);
    cv::Size srcSizeRight = cv::Size(890, 890);
    cv::Rect srcRectLeft(25, 37, 890, 890);
    cv::Rect srcRectRight(45, 33, 890, 890);

	ReprojectParam paramLeft, paramRight;
	//paramLeft.LoadConfig("F:\\panovideo\\ricoh\\0left.xml");
    paramLeft.LoadConfig("F:\\panovideo\\ricoh\\1builtinleft.xml");
	paramLeft.SetPanoSize(dstSize);
    //paramRight.LoadConfig("F:\\panovideo\\ricoh\\0right.xml");
    paramRight.LoadConfig("F:\\panovideo\\ricoh\\1builtinright.xml");
    paramRight.SetPanoSize(dstSize);

    int numVideos = 2;
    avp::VideoReader reader;
    reader.open("F:\\panovideo\\ricoh\\R0010005.MP4", avp::PixelTypeBGR32);

    std::vector<cv::Mat> dstMaskLeft, dstMaskRight;
    std::vector<cv::Mat> dstSrcMapLeft, dstSrcMapRight;
    getReprojectMapsAndMasks(paramLeft, srcSizeLeft, dstSrcMapLeft, dstMaskLeft);
    getReprojectMapsAndMasks(paramRight, srcSizeRight, dstSrcMapRight, dstMaskRight);

    //dstMaskLeft[0](cv::Range::all(), cv::Range(0, 442)).setTo(0);
    //dstMaskLeft[0](cv::Range::all(), cv::Range(1609, 2048)).setTo(0);
    //dstMaskRight[0](cv::Range::all(), cv::Range(616, 1495)).setTo(0);
    //cv::imshow("leftmask", dstMaskLeft[0]);
    //cv::imshow("rightmask", dstMaskRight[0]);
    //cv::waitKey(0);
    //cv::imwrite("ricohmask0.bmp", dstMaskLeft[0]);
    //cv::imwrite("ricohmask1.bmp", dstMaskRight[0]);

    std::vector<cv::gpu::GpuMat> dstMasksGpu(numVideos);
    std::vector<cv::gpu::GpuMat> xmapsGpu(numVideos), ymapsGpu(numVideos);
    cv::Mat map32F;
    cv::Mat map64F[2];
    dstMasksGpu[0].upload(dstMaskLeft[0]);
    cv::split(dstSrcMapLeft[0], map64F);
    map64F[0].convertTo(map32F, CV_32F);
    xmapsGpu[0].upload(map32F);
    map64F[1].convertTo(map32F, CV_32F);
    ymapsGpu[0].upload(map32F);
    dstMasksGpu[1].upload(dstMaskRight[0]);
    cv::split(dstSrcMapRight[0], map64F);
    map64F[0].convertTo(map32F, CV_32F);
    xmapsGpu[1].upload(map32F);
    map64F[1].convertTo(map32F, CV_32F);
    ymapsGpu[1].upload(map32F);
    map32F.release();
    map64F[0].release();
    map64F[1].release();

    std::vector<cv::Mat> dstMasks(2);
    dstMasks[0] = dstMaskLeft[0];
    dstMasks[1] = dstMaskRight[0];
    //CudaTilingMultibandBlend blender;
    //bool success = blender.prepare(dstMasks, 16, 2);
    //CudaTilingMultibandBlendFast blender;
    //bool success = blender.prepare(dstMasks, 16, 2);
    CudaTilingFeatherBlend blender;
    bool success = blender.prepare(dstMasks);
    dstMasks.clear();

    int frameCount = 0;
    avp::BGRImage rawImage;
    std::vector<avp::BGRImage> rawImages(numVideos);
    std::vector<cv::gpu::GpuMat> imagesGpu(numVideos), reprojImagesGpu(numVideos);
    cv::gpu::GpuMat blendImageGpu;
    cv::Mat blendImageCpu;

    ztool::Timer timerAll, timerTotal, timerDecode, timerUpload, timerReproject, timerBlend, timerDownload, timerEncode;

    avp::VideoWriter writer;
    writer.open("ricohmbblend.mp4", avp::PixelTypeBGR32, dstSize.width, dstSize.height, 48);
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

        //rawImages[0].data = rawImage.data;
        //rawImages[0].width = rawImage.width / 2;
        //rawImages[0].height = rawImage.height;
        //rawImages[0].step = rawImage.step;

        //rawImages[1].data = rawImage.data + rawImage.width / 2 * 4;
        //rawImages[1].width = rawImage.width / 2;
        //rawImages[1].height = rawImage.height;
        //rawImages[1].step = rawImage.step;

        cv::Mat raw(rawImage.height, rawImage.width, CV_8UC4, rawImage.data, rawImage.step);
        cv::Mat left0 = raw(cv::Rect(0, 0, rawImage.width / 2, rawImage.height));
        cv::Mat left = left0(srcRectLeft);
        cv::Mat right0 = raw(cv::Rect(rawImage.width / 2, 0, rawImage.width / 2, rawImage.height));
        cv::Mat right = right0(srcRectRight);
        rawImages[0].data = left.data;
        rawImages[0].width = left.cols;
        rawImages[0].height = left.rows;
        rawImages[0].step = left.step;
        rawImages[1].data = right.data;
        rawImages[1].width = right.cols;
        rawImages[1].height = right.rows;
        rawImages[1].step = right.step;

        timerUpload.start();
        for (int i = 0; i < numVideos; i++)
        {
            cv::Mat imageCpu(rawImages[i].height, rawImages[i].width, CV_8UC4, rawImages[i].data, rawImages[i].step);
            imagesGpu[i].upload(imageCpu);
        }
        timerUpload.end();

        timerReproject.start();
        for (int i = 0; i < numVideos; i++)
        {
            cudaReproject(imagesGpu[i], reprojImagesGpu[i], xmapsGpu[i], ymapsGpu[i]);
        }
        timerReproject.end();

        //cv::Mat image1, image2;
        //reprojImagesGpu[0].download(image1);
        //reprojImagesGpu[1].download(image2);
        //cv::imshow("image1", image1);
        //cv::imshow("image2", image2);
        //cv::waitKey(0);
        //if (frameCount == 10)
        //{
        //    cv::imwrite("ricohimage0.bmp", image1);
        //    cv::imwrite("ricohimage1.bmp", image2);
        //    break;
        //}

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
        printf("time = %f\n", timerTotal.elapse());
    }
    timerAll.end();
    printf("all time %f\n", timerAll.elapse());
    return 0;
}