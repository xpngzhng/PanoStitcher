#include "Blend/ZBlend.h"
#include "Warp/ZReproject.h"
#include "CudaAccel/CudaInterface.h"
#include "Tool/Timer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

const double PI = 3.1415926535898;
const double scale = 3.1415926535898 / 180;

int main(int argc, char* argv[])
{
    std::vector<std::string> contentPaths;
    contentPaths.push_back("F:\\panoimage\\beijing\\image0.bmp");
    contentPaths.push_back("F:\\panoimage\\beijing\\image1.bmp");
    contentPaths.push_back("F:\\panoimage\\beijing\\image2.bmp");
    contentPaths.push_back("F:\\panoimage\\beijing\\image3.bmp");
    contentPaths.push_back("F:\\panoimage\\beijing\\image4.bmp");
    contentPaths.push_back("F:\\panoimage\\beijing\\image5.bmp");
    int numImages = contentPaths.size();

    std::vector<cv::Mat> images(numImages);
    for (int i = 0; i < numImages; i++)
        images[i] = cv::imread(contentPaths[i]);

    cv::Size dstSize(2048, 1024);

    std::vector<PhotoParam> params;
    loadPhotoParamFromXML("F:\\panoimage\\beijing\\temp_camera_param.xml", params);
    //pi.rotateCamera(0, -35.264 / 180 * PI, -PI / 4);
    //pi.rotateCamera(0, -0.621986, -0.702595);
    //pi.rotateCamera(-0.716718, -0.628268, -0.695993);
    //pi.rotateCamera(0, -35.264 / 180 * PI, -PI / 4); // panotools
    //pi.rotateCamera(0.710973 + 1, -0.510681, -1.910918); // distort
    //pi.rotateCamera(1.410963, -0.540605, -0.908481); // distort
    //pi.rotateCamera(0.736832, -0.489683, -2.0011); // distort
    //pi.rotateCamera(1.599374, -0.516422, -0.703413); // distort
    rotateCameras(params, 0, 3.1415926536 / 2 * 0.65, 0);

    ztool::Timer timer;

    std::vector<cv::cuda::GpuMat> src(numImages), xmaps(numImages), ymaps(numImages), masks(numImages);
    cudaGenerateReprojectMapsAndMasks(params, images[0].size(), dstSize, xmaps, ymaps, masks);
    std::vector<cv::Mat> masksCpu(numImages);
    cv::Mat temp;
    for (int i = 0; i < numImages; i++)
    {
        cv::cvtColor(images[i], temp, CV_BGR2BGRA);
        src[i].upload(temp);
        masks[i].download(masksCpu[i]);
    }

    int numIter = 100;
    /*{
        CudaTilingMultibandBlend blender;
        blender.prepare(masksCpu, 20, 4);
        cv::cuda::GpuMat reprojImage, blendImage;
        timer.start();
        for (int k = 0; k < numIter; k++)
        {
            for (int i = 0; i < numImages; i++)
            {
                cudaReprojectTo16S(src[i], reprojImage, dstSize, params[i]);
                blender.tile(reprojImage, i);
            }
            blender.composite(blendImage);
        }
        timer.end();
        printf("serial reproj serial blend %f\n", timer.elapse());
    }

    {
        CudaTilingMultibandBlend blender;
        blender.prepare(masksCpu, 20, 4);
        std::vector<cv::cuda::GpuMat> reprojImages(numImages);
        cv::cuda::GpuMat blendImage;
        timer.start();
        for (int k = 0; k < numIter; k++)
        {
            for (int i = 0; i < numImages; i++)
                cudaReprojectTo16S(src[i], reprojImages[i], dstSize, params[i]);
            blender.blend(reprojImages, blendImage);
        }
        timer.end();
        printf("serial reproj joint blend %f\n", timer.elapse());
    }

    {
        CudaTilingMultibandBlendFast blender;
        blender.prepare(masksCpu, 20, 4);
        std::vector<cv::cuda::GpuMat> reprojImages(numImages);
        cv::cuda::GpuMat blendImage;
        timer.start();
        for (int k = 0; k < numIter; k++)
        {
            for (int i = 0; i < numImages; i++)
                cudaReprojectTo16S(src[i], reprojImages[i], dstSize, params[i]);
            blender.blend(reprojImages, blendImage);
        }
        timer.end();
        printf("serial reproj fast blend %f\n", timer.elapse());
    }*/

    /*{
        CudaTilingMultibandBlend blender;
        blender.prepare(masksCpu, 20, 4);
        std::vector<cv::cuda::GpuMat> reprojImages(numImages);
        cv::cuda::GpuMat blendImage;
        timer.start();
        for (int k = 0; k < numIter; k++)
        {
            for (int i = 0; i < numImages; i++)
                cudaReprojectTo16S(src[i], reprojImages[i], xmaps[i], ymaps[i]);
            blender.blend(reprojImages, blendImage);
        }
        timer.end();
        printf("map reproj joint blend %f\n", timer.elapse());
    }*/

    {
        CudaTilingMultibandBlendFast blender;
        blender.prepare(masksCpu, 20, 4);
        std::vector<cv::cuda::GpuMat> reprojImages(numImages);
        cv::cuda::GpuMat blendImage;
        timer.start();
        for (int k = 0; k < numIter; k++)
        {
            for (int i = 0; i < numImages; i++)
                cudaReprojectTo16S(src[i], reprojImages[i], xmaps[i], ymaps[i]);
            blender.blend(reprojImages, blendImage);
        }
        timer.end();
        printf("map reproj fast blend %f\n", timer.elapse());
    }

    return 0;
}