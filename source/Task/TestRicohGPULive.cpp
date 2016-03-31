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
    cv::Size dstSize = cv::Size(4096, 2048);
    cv::Size srcSizeLeft = cv::Size(922, 922);
    cv::Size srcSizeRight = cv::Size(912, 912);
    cv::Rect srcRectLeft(10, 17, 922, 922);
    cv::Rect srcRectRight(25, 19, 912, 912);

	ReprojectParam paramLeft, paramRight;
	paramLeft.LoadConfig("F:\\panovideo\\ricoh\\0left.xml");
	paramLeft.SetPanoSize(dstSize);
    paramRight.LoadConfig("F:\\panovideo\\ricoh\\0right.xml");
    paramRight.SetPanoSize(dstSize);

    int numVideos = 2;

    std::vector<cv::Mat> dstMaskLeft, dstMaskRight;
    std::vector<cv::Mat> dstSrcMapLeft, dstSrcMapRight;
    getReprojectMapsAndMasks(paramLeft, srcSizeLeft, dstSrcMapLeft, dstMaskLeft);
    getReprojectMapsAndMasks(paramRight, srcSizeRight, dstSrcMapRight, dstMaskRight);

    double rleft1 = double(485) / 1920, rleft2 = double(1466) / 1920;
    double rright1 = double(505) / 1920, rright2 = double(1446) / 1920;
    //dstMaskLeft[0](cv::Range::all(), cv::Range(0, 442)).setTo(0);
    //dstMaskLeft[0](cv::Range::all(), cv::Range(1609, 2048)).setTo(0);
    //dstMaskRight[0](cv::Range::all(), cv::Range(616, 1495)).setTo(0);
    //dstMaskLeft[0](cv::Range::all(), cv::Range(0, 485)).setTo(0);
    //dstMaskLeft[0](cv::Range::all(), cv::Range(1466, 1920)).setTo(0);
    //dstMaskRight[0](cv::Range::all(), cv::Range(505, 1446)).setTo(0);
    dstMaskLeft[0](cv::Range::all(), cv::Range(0, rleft1 * dstSize.width)).setTo(0);
    dstMaskLeft[0](cv::Range::all(), cv::Range(rleft2 * dstSize.width, dstSize.width)).setTo(0);
    dstMaskRight[0](cv::Range::all(), cv::Range(rright1 * dstSize.width, rright2 * dstSize.width)).setTo(0);
    //cv::imshow("leftmask", dstMaskLeft[0]);
    //cv::imshow("rightmask", dstMaskRight[0]);
    //cv::waitKey(0);
    //cv::imwrite("ricohmask0-1920x960.bmp", dstMaskLeft[0]);
    //cv::imwrite("ricohmask1-1920x960.bmp", dstMaskRight[0]);

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

    cv::VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
    int fromTo[] = {0, 0, 1, 1, 2, 2};
    cv::Mat frame8UC3, raw(1080, 1920, CV_8UC4);

    avp::AudioVideoSender sender;
    //sender.open("rtmp://pili-publish.live.detu.com/detulive/detudemov5?key=detukey", avp::PixelTypeBGR32, dstSize.width, dstSize.height, 25, 500000);
    //sender.open("rtmp://pili-publish.live.detu.com/detulive/p_detudemov28", avp::PixelTypeBGR32, dstSize.width, dstSize.height, 30, 100000);
    avp::AudioVideoMuxer muxer;
    muxer.open("livehighbps.mp4", avp::PixelTypeBGR32, dstSize.width, dstSize.height, 25, dstSize.width * dstSize.height * 1.5 * 25 * 0.1/*, 2500000*/);
    timerAll.start();
    while (true)
    {
        timerTotal.start();

        bool success = true;
        timerDecode.start();
        success = cap.read(frame8UC3);
        timerDecode.end();
        if (!success)
            continue;

        printf("currCount = %d\n", frameCount++);
        if (frameCount >= 800)
            break;

        cv::mixChannels(&frame8UC3, 1, &raw, 1, fromTo, 3);
        rawImage.data = raw.data;
        rawImage.width = raw.cols;
        rawImage.height = raw.rows;
        rawImage.step = raw.step;
        //cv::Mat raw(rawImage.height, rawImage.width, CV_8UC4, rawImage.data, rawImage.step);
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

        //cv::imshow("curr", blendImageCpu);
        //cv::waitKey(1);

        timerEncode.start();
        avp::BGRImage image(blendImageCpu.data, blendImageCpu.cols, blendImageCpu.rows, blendImageCpu.step);
        //sender.write(image);
        muxer.write(image);
        timerEncode.end();

        timerTotal.end();
        printf("time elapsed = %f, dec = %f, upload = %f, proj = %f, blend = %f, download = %f, enc = %f\n", 
            timerTotal.elapse(), timerDecode.elapse(), timerUpload.elapse(), timerReproject.elapse(), 
            timerBlend.elapse(), timerDownload.elapse(), timerEncode.elapse());
        printf("time = %f\n", timerTotal.elapse());
    }
    timerAll.end();
    printf("all time %f\n", timerAll.elapse());
    sender.close();
    muxer.close();
    return 0;
}