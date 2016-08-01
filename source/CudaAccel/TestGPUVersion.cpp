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
#include "ZBlend.h"
#include "ZBlendAlgo.h"
#include "Timer.h"

const double PI = 3.1415926535898;
const double scale = 3.1415926535898 / 180;

int main(int argc, char* argv[])
{
	std::string filelist;
	if (argc < 2)
		std::cout << "[Info]: Not enough parameters...... "<< std::endl;
	else
		filelist = argv[1];

	std::ifstream fin(filelist.c_str());
	std::string imagePath;

	ReprojectParam pi;
	pi.LoadConfig("stitchparam/distort.xml");
	pi.SetPanoSize(cv::Size(2000,1000));
    //pi.rotateCamera(0, -35.264 / 180 * PI, -PI / 4);
    //pi.rotateCamera(0, -0.621986, -0.702595);
    //pi.rotateCamera(-0.716718, -0.628268, -0.695993);
    //pi.rotateCamera(0, -35.264 / 180 * PI, -PI / 4); // panotools
    //pi.rotateCamera(0.710973 + 1, -0.510681, -1.910918); // distort
    //pi.rotateCamera(1.410963, -0.540605, -0.908481); // distort
    //pi.rotateCamera(0.736832, -0.489683, -2.0011); // distort
    pi.rotateCamera(1.599374, -0.516422, -0.703413); // distort

	std::vector<cv::Mat> imagesSrc;
	std::vector<cv::Mat> imagesDst;
	std::vector<cv::Mat> imagesMask;
    std::vector<cv::Mat> imagesMap;
	std::vector<std::string > imagesName;
	while (fin >> imagePath)
	{
		cv::Mat image = cv::imread(imagePath);
		imagesSrc.push_back(image);
		std::string imageName = imagePath.substr(imagePath.find_last_of("/\\") + 1);
		imagesName.push_back(imageName);
	}
    int numImages = imagesSrc.size();
    //img_src.clear();
    //cv::Scalar colors[] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
    //                       cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
    //                       cv::Scalar(128, 0, 0), cv::Scalar(0, 128, 0), cv::Scalar(0, 0, 128),
    //                       cv::Scalar(128, 128, 0), cv::Scalar(128, 0, 128), cv::Scalar(0, 128, 128)};
    //for (int i = 0; i < 6; i++)
    //{
    //    cv::Mat curr(960, 1280, CV_8UC3);
    //    curr.setTo(colors[i]);
    //    img_src.push_back(curr);
    //}

    ztool::Timer timer;
    
    printf("begin reproject\n");
    timer.start();
    getReprojectMapsAndMasks(pi, cv::Size(4608, 3456), imagesMap, imagesMask);
    timer.end();
    printf("get map time = %f\n", timer.elapse());
    timer.start();
    //reproject(imagesSrc, imagesDst, imagesMap);
    reprojectParallel(imagesSrc, imagesDst, imagesMap);
    timer.end();
    printf("reproject time = %f\n", timer.elapse());

    cv::Mat dst;
    TilingMultibandBlend blender;
    timer.start();
    blender.prepare(imagesMask, 16, 2);
    timer.end();
    printf("blender prepare time = %f\n", timer.elapse());
    timer.start();
    blender.blend(imagesDst, imagesMask, dst);
    timer.end();
    printf("blender blend time = %f\n", timer.elapse());
    //cv::imshow("dst", dst);
    //cv::waitKey(0);

    std::vector<cv::Mat> imagesXMap, imagesYMap;
    imagesXMap.resize(numImages);
    imagesYMap.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        cv::Mat mapTemp, map[2];
        imagesMap[i].convertTo(mapTemp, CV_32F);
        cv::split(mapTemp, map);
        imagesXMap[i] = map[0];
        imagesYMap[i] = map[1];
    }

    std::vector<cv::Mat> imagesRemap;
    imagesRemap.resize(numImages);
    timer.start();
    for (int i = 0; i < numImages; i++)
        cv::remap(imagesSrc[i], imagesRemap[i], imagesXMap[i], imagesYMap[i], cv::INTER_CUBIC);
    timer.end();
    printf("opencv remap time = %f\n", timer.elapse());

    //for (int i = 0; i < numImages; i++)
    //{
    //    cv::imshow("remap", imagesRemap[i]);
    //    cv::waitKey(0);
    //}
    
    std::vector<cv::gpu::GpuMat> imagesXMapGpu, imagesYMapGpu;
    std::vector<cv::gpu::GpuMat> imagesSrcGpu, imagesDstGpu;    
    imagesSrcGpu.resize(numImages);
    imagesDstGpu.resize(numImages);
    imagesXMapGpu.resize(numImages);
    imagesYMapGpu.resize(numImages);
    timer.start();
    for (int i = 0; i < numImages; i++)
    {
        imagesSrcGpu[i].upload(imagesSrc[i]);
    }
    timer.end();
    printf("upload images src time = %f\n", timer.elapse());

    timer.start();
    for (int i = 0; i < numImages; i++)
    {        
        imagesXMapGpu[i].upload(imagesXMap[i]);
        imagesYMapGpu[i].upload(imagesYMap[i]);
    }
    timer.end();
    printf("upload map time = %f\n", timer.elapse());

    for (int j = 0; j < 10; j++)
    {
        timer.start();
        for (int i = 0; i < numImages; i++)
        {
            cv::gpu::remap(imagesSrcGpu[i], imagesDstGpu[i], imagesXMapGpu[i], imagesYMapGpu[i], cv::INTER_CUBIC);
        }
        timer.end();
        printf("gpu time = %f\n", timer.elapse());
    }

    timer.start();
    for (int i = 0; i < numImages; i++)
    {
        cv::Mat download;
        imagesDstGpu[i].download(download);
        //cv::imshow("gpu image", download);
        //cv::waitKey(0);
    }
    timer.end();
    printf("download image dst time = %f\n", timer.elapse());
    
    //const cv::ocl::DeviceInfo& device = cv::ocl::Context::getContext()->getDeviceInfo();
    //const void * p = cv::ocl::getClContextPtr();
    //std::vector<cv::ocl::oclMat> imagesXMapOcl, imagesYMapOcl;
    //std::vector<cv::ocl::oclMat> imagesSrcOcl, imagesDstOcl;    
    //imagesSrcOcl.resize(numImages);
    //imagesDstOcl.resize(numImages);
    //imagesXMapOcl.resize(numImages);
    //imagesYMapOcl.resize(numImages);
    //timer.start();
    //for (int i = 0; i < numImages; i++)
    //{
    //    imagesSrcOcl[i].upload(imagesSrc[i]);
    //}
    //timer.end();
    //printf("upload images src time = %f\n", timer.elapse());

    //timer.start();
    //for (int i = 0; i < numImages; i++)
    //{        
    //    imagesXMapOcl[i].upload(imagesXMap[i]);
    //    imagesYMapOcl[i].upload(imagesYMap[i]);
    //}
    //timer.end();
    //printf("upload map time = %f\n", timer.elapse());

    //for (int j = 0; j < 10; j++)
    //{
    //    timer.start();
    //    for (int i = 0; i < numImages; i++)
    //    {
    //        cv::ocl::remap(imagesSrcOcl[i], imagesDstOcl[i], imagesXMapOcl[i], imagesYMapOcl[i], cv::INTER_LINEAR, cv::BORDER_REFLECT101);
    //    }
    //    timer.end();
    //    printf("ocl time = %f\n", timer.elapse());
    //}

    //timer.start();
    //for (int i = 0; i < numImages; i++)
    //{
    //    cv::Mat download;
    //    imagesDstOcl[i].download(download);
    //    //cv::imshow("ocl image", download);
    //    //cv::waitKey(0);
    //}
    //timer.end();
    //printf("download image dst time = %f\n", timer.elapse());
    //return 0;
    
    //long long int beg = cv::getTickCount();
    //PanoReprojection(imagesSrc, imagesDst, imagesMap, pi);
    //long long int end = cv::getTickCount();
    //double freq = cv::getTickFrequency();
    //printf("%f\n", (end - beg) / freq);    

    //cv::Mat result = cv::Mat::zeros(cv::Size(1000, 500), CV_8UC3);
    //for (int i = 0; i < imagesSrc.size(); i++)
    //{
    //    cv::imshow("this", imagesDst[i]);
    //    cv::waitKey(0);
    //    imagesDst[i].copyTo(result, imagesMask[i]);
    //}
    //cv::imshow("result", result);
    ////cv::imwrite("result1.bmp", result);
    //cv::waitKey(0);

    return 0;
}