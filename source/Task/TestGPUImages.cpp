#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/ocl/ocl.hpp>
#include "Reprojection.h"
#include "Timer.h"
#include "ZBlend.h"

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

    ReprojectParam pi;
    pi.LoadConfig("F:\\panoimage\\beijing\\temp_camera_param.xml");
    pi.SetPanoSize(cv::Size(2048, 1024));
    //pi.rotateCamera(0, -35.264 / 180 * PI, -PI / 4);
    //pi.rotateCamera(0, -0.621986, -0.702595);
    //pi.rotateCamera(-0.716718, -0.628268, -0.695993);
    //pi.rotateCamera(0, -35.264 / 180 * PI, -PI / 4); // panotools
    //pi.rotateCamera(0.710973 + 1, -0.510681, -1.910918); // distort
    //pi.rotateCamera(1.410963, -0.540605, -0.908481); // distort
    //pi.rotateCamera(0.736832, -0.489683, -2.0011); // distort
    //pi.rotateCamera(1.599374, -0.516422, -0.703413); // distort
    pi.rotateCamera(0, 3.1415926536 / 2 * 0.65, 0);

    ztool::Timer timer;
    
    std::vector<cv::Mat> imagesMask;
    std::vector<cv::Mat> imagesMap;
    getReprojectMapsAndMasks(pi, images[0].size(), imagesMap, imagesMask);

    std::vector<cv::Mat> imagesXMap, imagesYMap;
    imagesXMap.resize(numImages);
    imagesYMap.resize(numImages);
    std::vector<cv::gpu::GpuMat> imagesXMapGpu, imagesYMapGpu;
    imagesXMapGpu.resize(numImages);
    imagesYMapGpu.resize(numImages);
    cv::Mat temp;
    timer.start();
    for (int i = 0; i < numImages; i++)
    {
        cv::Mat mapTemp, map[2];
        imagesMap[i].convertTo(mapTemp, CV_32F);
        cv::split(mapTemp, map);
        imagesXMap[i] = map[0];
        imagesYMap[i] = map[1];
        imagesXMapGpu[i].upload(map[0]);
        imagesYMapGpu[i].upload(map[1]);
    }
    timer.end();
    printf("upload map time = %f\n", timer.elapse());

    std::vector<cv::gpu::CudaMem> pinnedMems(numImages);
    std::vector<cv::Mat> pinnedImages(numImages);
    int fromTo[] = { 0, 0, 1, 1, 2, 2 };
    for (int i = 0; i < numImages; i++)
    {
        pinnedMems[i].create(images[i].size(), CV_8UC4);
        pinnedImages[i] = pinnedMems[i];
        cv::mixChannels(&images[i], 1, &pinnedImages[i], 1, fromTo, 3);
    }

    CudaTilingMultibandBlendFast blender;
    blender.prepare(imagesMask, 16, 2);

    std::vector<cv::gpu::GpuMat> imagesGpu(numImages), imagesReprojGpu(numImages);
    cv::gpu::GpuMat blendImageGpu;
    cv::Mat blendImageC4;
    std::vector<cv::gpu::Stream> streams(numImages);
    for (int i = 0; i < 100; i++)
    {
        timer.start();
        for (int j = 0; j < numImages; j++)
            streams[j].enqueueUpload(pinnedImages[j], imagesGpu[j]);
        for (int j = 0; j < numImages; j++)
            cudaReprojectTo16S(imagesGpu[j], imagesReprojGpu[j], imagesXMapGpu[j], imagesYMapGpu[j], streams[j]);
        for (int j = 0; j < numImages; j++)
            streams[j].waitForCompletion();
        blender.blend(imagesReprojGpu, blendImageGpu);
        cv::gpu::Stream::Null().waitForCompletion();
        blendImageGpu.download(blendImageC4);
        timer.end();
        printf("time elapsed = %f\n", timer.elapse());
    }

    timer.start();
    for (int i = 0; i < numImages; i++)
    {
        cv::Mat download, download16S;
        imagesReprojGpu[i].download(download16S);
        download16S.convertTo(download, CV_8U);
        // download16S has negative values, why???
        std::cout << download16S(cv::Rect(0, 0, 8, 8)) << std::endl << std::endl;
        cv::imshow("gpu image", download);
        cv::waitKey(0);
    }
    timer.end();
    printf("download image dst time = %f\n", timer.elapse());
    cv::imshow("result", blendImageC4);
    cv::waitKey(0);

    return 0;
}