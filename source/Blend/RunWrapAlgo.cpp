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

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\0.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\1.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\2.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\3.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\4.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\5.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\0mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\1mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\2mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\3mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\4mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\5mask.bmp");

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\panoimage\\changtai\\reprojimage0.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\reprojimage1.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\reprojimage2.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\reprojimage3.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\reprojimage4.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\reprojimage5.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask0.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask1.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask2.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask3.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask4.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask5.bmp");

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\panoimage\\detuoffice2\\image0.bmp");
    //contentPaths.push_back("F:\\panoimage\\detuoffice2\\image1.bmp");
    //contentPaths.push_back("F:\\panoimage\\detuoffice2\\image2.bmp");
    //contentPaths.push_back("F:\\panoimage\\detuoffice2\\image3.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\detuoffice2\\mask0.bmp");
    //maskPaths.push_back("F:\\panoimage\\detuoffice2\\mask1.bmp");
    //maskPaths.push_back("F:\\panoimage\\detuoffice2\\mask2.bmp");
    //maskPaths.push_back("F:\\panoimage\\detuoffice2\\mask3.bmp");

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("E:\\Projects\\Reprojecting\\Reproject\\image0.bmp");
    //contentPaths.push_back("E:\\Projects\\Reprojecting\\Reproject\\image1.bmp");
    //contentPaths.push_back("E:\\Projects\\Reprojecting\\Reproject\\image2.bmp");
    //contentPaths.push_back("E:\\Projects\\Reprojecting\\Reproject\\image3.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("E:\\Projects\\Reprojecting\\Reproject\\mask0.bmp");
    //maskPaths.push_back("E:\\Projects\\Reprojecting\\Reproject\\mask1.bmp");
    //maskPaths.push_back("E:\\Projects\\Reprojecting\\Reproject\\mask2.bmp");
    //maskPaths.push_back("E:\\Projects\\Reprojecting\\Reproject\\mask3.bmp");

    std::vector<std::string> contentPaths;
    contentPaths.push_back("F:\\panoimage\\detuoffice2\\image0.bmp");
    contentPaths.push_back("F:\\panoimage\\detuoffice2\\image1.bmp");
    contentPaths.push_back("F:\\panoimage\\detuoffice2\\image2.bmp");
    contentPaths.push_back("F:\\panoimage\\detuoffice2\\image3.bmp");
    std::vector<std::string> maskPaths;
    maskPaths.push_back("F:\\panoimage\\detuoffice2\\mask0.bmp");
    maskPaths.push_back("F:\\panoimage\\detuoffice2\\mask1.bmp");
    maskPaths.push_back("F:\\panoimage\\detuoffice2\\mask2.bmp");
    maskPaths.push_back("F:\\panoimage\\detuoffice2\\mask3.bmp");

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

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\panoimage\\sky\\1full.jpg");
    //contentPaths.push_back("F:\\panoimage\\sky\\2.jpg");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\sky\\mask1.bmp");
    //maskPaths.push_back("F:\\panoimage\\sky\\mask2.bmp");

    ztool::Timer timer;
    timer.start();

    int numImages = contentPaths.size();
    std::vector<cv::Mat> images, masks; 
    cv::Size imageSize;
    getImagesAndMasks(contentPaths, maskPaths, imageSize, images, masks);

    //std::vector<cv::Mat> compensateImages;
    //GainCompensate comp;
    //comp.prepare(images, masks);
    //comp.compensate(images, compensateImages);
    //compensate(images, masks, compensateImages);
    //compensate3(images, masks, compensateImages);
    //compensateGray(images, masks, -1, compensateImages);
    //compensateLightAndSaturation(images, masks, -1, compensateImages);
    //for (int i = 0; i < numImages; i++)
    //{
    //    cv::imshow("image", images[i]);
    //    cv::imshow("image comp", compensateImages[i]);
    //    cv::waitKey(0);
    //}

    timer.start();
    cv::Mat blendImage = cv::Mat::zeros(imageSize, CV_8UC3);
    cv::Mat blendMask = cv::Mat::zeros(imageSize, CV_8UC1);
    BlendConfig cfg;
    //cfg.setSeamSkip();
    //cfg.setBlendPaste();
    cfg.setSeamDistanceTransform();
    //cfg.setSeamGraphCut(8, 4, 1, 2);
    cfg.setBlendMultiBand();
    //cfg.setBlendLinear(50);
    masks[2].setTo(0);
    for (int i = 0; i < numImages; i++)
    {
        printf("image index = %d\n", i);
        //cv::Mat image = cv::imread(contentPaths[i], -1);
        //cv::Mat mask = cv::imread(maskPaths[i], -1);
        cv::imshow("image", images[i]);
        cv::imshow("mask", masks[i]);
        serialBlend(cfg, images[i]/*compensateImages[i]*/, masks[i], blendImage, blendMask);
        cv::imshow("blend image", blendImage);
        cv::imshow("blend mask", blendMask);
        cv::waitKey(0);
    }

    //parallelBlend(cfg, /*compensateImages*/images, masks, blendImage);

    //TilingMultibandBlend blender;
    //blender.prepare(masks, 20, 2);
    //for (int i = 0; i < numImages; i++)
    //    blender.tile(images[i], masks[i], i);
    //blender.composite(blendImage);

    timer.end();

    printf("time used = %f\n", timer.elapse());
    cv::imshow("blend image", blendImage);
    //cv::imshow("blend mask", blendMask);
    //cv::imshow("belonging", indexImage);
    //cv::imwrite("blendmultibanddetu2new.bmp", blendImage);
    //return 0;
    cv::waitKey(0);
    //cv::imwrite("resultsky.bmp", blendImage);
    return 0;
}