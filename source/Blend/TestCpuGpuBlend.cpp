#include "ZBlendAlgo.h"
#include "ZBlend.h"
#include "Timer.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>

int main()
{
    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\panoimage\\changtai\\1.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\2.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\3.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\4.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\5.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\6.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_1.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_2.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_3.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_4.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_5.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_6.bmp");

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\pano0000.tif");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\pano0001.tif");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\pano0002.tif");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\panomask0000.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\panomask0001.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\panomask0002.bmp");

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\0.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\1.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\2.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\0mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\1mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\2mask.bmp");

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\0.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\1.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\2.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\3.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\4.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\5.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\6.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\7.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\8mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\9mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\10mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\11mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\12mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\13mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\14mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images\\15mask.bmp");

    std::vector<std::string> contentPaths;
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\0.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\1.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\2.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\3.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\4.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\5.bmp");
    std::vector<std::string> maskPaths;
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\0mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\1mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\2mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\3mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\4mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\5mask.bmp");

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0000.tif");
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0001.tif");
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0002.tif");
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0003.tif");
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0004.tif");
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0005.tif");
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0006.tif");
    //contentPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\pano0007.tif");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0000.bmp");
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0001.bmp");
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0002.bmp");
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0003.bmp");
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0004.bmp");
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0005.bmp");
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0006.bmp");
    //maskPaths.push_back("E:\\Projects\\Stitching\\build\\Stitching\\PanoAll\\panomask0007.bmp");

    ztool::Timer timer;
    timer.start();

    int numImages = contentPaths.size();
    std::vector<cv::Mat> images, masks; 
    cv::Size imageSize;
    getImagesAndMasks(contentPaths, maskPaths, imageSize, images, masks);
    //images.clear();
    //masks.clear();

    //timer.start();
    cv::Mat blendImage = cv::Mat::zeros(imageSize, CV_8UC3);
    cv::Mat blendMask = cv::Mat::zeros(imageSize, CV_8UC1);
    //BlendConfig cfg;
    //cfg.setSeamSkip();
    //cfg.setBlendPaste();
    //cfg.setSeamDistanceTransform();
    //cfg.setSeamGraphCut(8, 8, 0, 2);
    //cfg.setBlendMultiBand();
    //for (int i = 0; i < numImages; i++)
    //{
    //    printf("image index = %d\n", i);
    //    //cv::Mat image = cv::imread(contentPaths[i], -1);
    //    //cv::Mat mask = cv::imread(maskPaths[i], -1);
    //    //blendSameSize(image, mask, blendImage, blendMask);
    //    //blendSameSize(images[i], masks[i], blendImage, blendMask);
    //    blendSameSize(cfg, images[i], masks[i], blendImage, blendMask);
    //}
    TilingMultibandBlend blender;
    blender.prepare(masks, 20, 2);
    for (int i = 0; i < 20; i++)
    {
        timer.start();
        cv::Mat result;
        blender.blend(images, masks, result);
        timer.end();
        printf("cpu tiling blend %f\n", timer.elapse());
    }

    std::vector<cv::cuda::GpuMat> imagesGpu, masksGpu;
    imagesGpu.resize(numImages);
    masksGpu.resize(numImages);
    cv::Mat imageC4(images[0].size(), CV_8UC4);
    int fromTo[] = {0, 0, 1, 1, 2, 2};
    for (int i = 0; i < numImages; i++)
    {
        cv::mixChannels(&images[i], 1, &imageC4, 1, fromTo, 3);
        imagesGpu[i].upload(imageC4);
        masksGpu[i].upload(masks[i]);
    }

    CudaTilingMultibandBlend cudaBlender;
    cudaBlender.prepare(masks, 20, 2);
    bool show = false;
    for (int i = 0; i < 20; i++)
    {
        timer.start();
        cv::cuda::GpuMat result;
        cudaBlender.blend(imagesGpu, masksGpu, result);
        timer.end();
        printf("gpu tiling blend %f\n", timer.elapse());
        if (!show)
        {
            show = true;
            cv::Mat resultC4;
            result.download(resultC4);
            cv::Mat resultC3(result.size(), CV_8UC3);
            cv::mixChannels(&resultC4, 1, &resultC3, 1, fromTo, 3);
            cv::imshow("result", resultC3);
            cv::waitKey(0);
            cv::imwrite("gpuresult.bmp", resultC3);
        }
    }
    return 0;
}