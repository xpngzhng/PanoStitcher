#include "ZBlend.h"
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

int main()
{
    int numImages = 6;
    std::vector<cv::Mat> masks(numImages), images(numImages), imagesg(numImages);
    for (int i = 0; i < numImages; i++)
    {        
        char buf[128];
        sprintf(buf, "mask%d.bmp", i);
        masks[i] = cv::imread(buf, -1);
        sprintf(buf, "reprojimage%d.bmp", i);
        images[i] = cv::imread(buf);
        sprintf(buf, "reprojimagegpu%d.bmp", i);
        imagesg[i] = cv::imread(buf);
        //cv::Mat result;
        //cv::bitwise_xor(images[i], imagesGpu[i], result);
        //cv::imshow("result", result);
        //cv::waitKey(0);
    }
    imagesg.clear();

    TilingMultibandBlend blender;
    blender.prepare(masks, 20, 2);
    cv::Mat blendResult;
    blender.blend(images, masks, blendResult);
    cv::imwrite("cpuresult.bmp", blendResult);

    CudaTilingMultibandBlend cudaBlender;
    cudaBlender.prepare(masks, 20, 2);
    std::vector<cv::gpu::GpuMat> masksGpu(numImages), imagesGpu(numImages);
    int fromTo[] = {0, 0, 1, 1, 2, 2};
    cv::Mat imageC4(images[0].size(), CV_8UC4);
    cv::gpu::GpuMat blendResultGpu;
    for (int i = 0; i < numImages; i++)
    {
        masksGpu[i].upload(masks[i]);
        cv::mixChannels(&images[i], 1, &imageC4, 1, fromTo, 3);
        imagesGpu[i].upload(imageC4);
    }
    cudaBlender.blend(imagesGpu, masksGpu, blendResultGpu);
    blendResultGpu.download(imageC4);
    cv::Mat blendResultBack(images[0].size(), CV_8UC3);
    cv::mixChannels(&imageC4, 1, &blendResultBack, 1, fromTo, 3);
    cv::imwrite("gpuresult.bmp", blendResultBack);

    cv::Mat diff;
    cv::bitwise_xor(blendResult, blendResultBack, diff);
    cv::imshow("diff", diff);
    cv::waitKey(0);

    return 0;
}