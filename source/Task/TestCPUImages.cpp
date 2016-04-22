#include "ZReproject.h"
#include "ZBlend.h"
#include "Timer.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//int main()
//{
//    cv::Size dstSize = cv::Size(2048, 1024);
//
//    std::vector<std::string> paths;
//    paths.push_back("F:\\panoimage\\outdoor\\1.MOV.tif");
//    paths.push_back("F:\\panoimage\\outdoor\\2.MOV.tif");
//    paths.push_back("F:\\panoimage\\outdoor\\3.MOV.tif");
//    paths.push_back("F:\\panoimage\\outdoor\\4.MOV.tif");
//
//    int numImages = paths.size();
//    std::vector<cv::Mat> src(numImages);
//    for (int i = 0; i < numImages; i++)
//        src[i] = cv::imread(paths[i]);
//
//    std::vector<PhotoParam> params;
//    loadPhotoParamFromPTS("F:\\panoimage\\outdoor\\Panorama.pts", params);
//    //rotateCameras(params, 0, 3.1415926536 / 2 * 0.65, 0);
//
//    std::vector<cv::Mat> maps, masks;
//    getReprojectMapsAndMasks(params, src[0].size(), dstSize, maps, masks);
//
//    std::vector<cv::Mat> dst(numImages);
//    for (int i = 0; i < numImages; i++)
//    {
//        char buf[64];
//        sprintf(buf, "mask%d.bmp", i);
//        //cv::imwrite(buf, masks[i]);
//        reproject(src[i], dst[i], maps[i]);
//        sprintf(buf, "reprojimage%d.bmp", i);
//        cv::imwrite(buf, dst[i]);
//        cv::imshow("dst", dst[i]);
//        cv::waitKey(0);
//    }
//
//    TilingMultibandBlend blender;
//    blender.prepare(masks, 20, 2);
//    cv::Mat blendImage;
//    blender.blend(dst, masks, blendImage);
//    cv::imshow("blend", blendImage);
//    cv::waitKey(0);
//}

int main()
{
    cv::Size dstSize = cv::Size(2048, 1024);

    //std::vector<std::string> paths;
    //paths.push_back("F:\\panoimage\\beijing\\image0.bmp");
    //paths.push_back("F:\\panoimage\\beijing\\image1.bmp");
    //paths.push_back("F:\\panoimage\\beijing\\image2.bmp");
    //paths.push_back("F:\\panoimage\\beijing\\image3.bmp");
    //paths.push_back("F:\\panoimage\\beijing\\image4.bmp");
    //paths.push_back("F:\\panoimage\\beijing\\image5.bmp");
    //std::string configFilePath = "F:\\panoimage\\beijing\\temp_camera_param_new.xml";

    std::vector<std::string> paths;
    paths.push_back("F:\\panoimage\\road\\image0.bmp");
    paths.push_back("F:\\panoimage\\road\\image1.bmp");
    paths.push_back("F:\\panoimage\\road\\image2.bmp");
    paths.push_back("F:\\panoimage\\road\\image3.bmp");
    paths.push_back("F:\\panoimage\\road\\image4.bmp");
    paths.push_back("F:\\panoimage\\road\\image5.bmp");
    std::string configFilePath = "F:\\panoimage\\road\\param.xml";

    //std::vector<std::string> paths;
    //paths.push_back("F:\\panoimage\\zhanxiang\\0.jpg");
    //paths.push_back("F:\\panoimage\\zhanxiang\\1.jpg");
    //paths.push_back("F:\\panoimage\\zhanxiang\\2.jpg");
    //paths.push_back("F:\\panoimage\\zhanxiang\\3.jpg");
    //paths.push_back("F:\\panoimage\\zhanxiang\\4.jpg");
    //paths.push_back("F:\\panoimage\\zhanxiang\\5.jpg");
    //std::string configFilePath = "F:\\panoimage\\zhanxiang\\zhanxiang.xml";

    int numImages = paths.size();
    std::vector<cv::Mat> src(numImages);
    for (int i = 0; i < numImages; i++)
        src[i] = cv::imread(paths[i]);

    std::vector<PhotoParam> params;
    loadPhotoParamFromXML(configFilePath, params);

    std::vector<cv::Mat> maps, masks;
    getReprojectMapsAndMasks(params, src[0].size(), dstSize, maps, masks);

    std::vector<cv::Mat> dst(numImages);
    //for (int i = 0; i < numImages; i++)
    //{
    //    char buf[64];
    //    sprintf(buf, "mask%d.bmp", i);
    //    //cv::imwrite(buf, masks[i]);
    //    reproject(src[i], dst[i], maps[i]);
    //    sprintf(buf, "reprojimage%d.bmp", i);
    //    cv::imwrite(buf, dst[i]);
    //    cv::imshow("dst", dst[i]);
    //    cv::waitKey(0);
    //}

    TilingMultibandBlend blender;
    blender.prepare(masks, 20, 2);
    cv::Mat blendImage;
    //blender.blend(dst, blendImage);
    //cv::imshow("blend", blendImage);
    //cv::waitKey(0);

    ztool::Timer timer;
    ztool::RepeatTimer timerReproject, timerBlend;
    for (int i = 0; i < 1; i++)
    {
        timerReproject.start();
        reprojectParallel(src, dst, maps);
        for (int j = 0; j < dst.size(); j++)
        {
            cv::imshow("dst", dst[j]);
            cv::waitKey(0);
        }
        timerReproject.end();
        timerBlend.start();
        blender.blend(dst, masks, blendImage);
        //BlendConfig config;
        //config.setSeamDistanceTransform();
        //parallelBlend(config, dst, masks, blendImage);
        timerBlend.end();
    }
    timer.end();
    cv::imshow("blend", blendImage);
    cv::waitKey(0);
    printf("%f, %f, %f\n", timer.elapse(), timerReproject.getAccTime(), timerBlend.getAccTime());

    CudaTilingMultibandBlendFast cudaBlender;
    cudaBlender.prepare(masks, 20, 2);
    std::vector<cv::cuda::GpuMat> cudaImages(numImages), cudaMasks(numImages);
    cv::Mat imageC4(dst[0].size(), CV_8UC4);
    for (int i = 0; i < numImages; i++)
    {
        cv::cvtColor(dst[i], imageC4, CV_BGR2BGRA);
        cudaImages[i].upload(imageC4);
        cudaMasks[i].upload(masks[i]);
    }
    cv::cuda::GpuMat cudaBlendImage;
    for (int i = 0; i < 10; i++)
        cudaBlender.blend(cudaImages, cudaBlendImage);
    cudaBlendImage.download(blendImage);
    cv::imshow("cuda blend", blendImage);
    cv::waitKey(0);

    return 0;
}