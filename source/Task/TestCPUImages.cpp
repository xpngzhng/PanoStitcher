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

    std::vector<std::string> paths;
    paths.push_back("F:\\panoimage\\beijing\\image0.bmp");
    paths.push_back("F:\\panoimage\\beijing\\image1.bmp");
    paths.push_back("F:\\panoimage\\beijing\\image2.bmp");
    paths.push_back("F:\\panoimage\\beijing\\image3.bmp");
    paths.push_back("F:\\panoimage\\beijing\\image4.bmp");
    paths.push_back("F:\\panoimage\\beijing\\image5.bmp");

    int numImages = paths.size();
    std::vector<cv::Mat> src(numImages);
    for (int i = 0; i < numImages; i++)
        src[i] = cv::imread(paths[i]);

    std::vector<PhotoParam> params;
    loadPhotoParamFromXML("F:\\panoimage\\beijing\\temp_camera_param_new.xml", params);
    //loadPhotoParamFromPTS("F:\\panoimage\\outdoor\\Panorama.pts", params);
    //rotateCameras(params, 0, 3.1415926536 / 2 * 0.65, 0);

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

    TilingMultibandBlendFastParallel blender;
    blender.prepare(masks, 20, 2);
    cv::Mat blendImage;
    //blender.blend(dst, blendImage);
    //cv::imshow("blend", blendImage);
    //cv::waitKey(0);

    ztool::Timer timer;
    ztool::RepeatTimer timerReproject, timerBlend;
    for (int i = 0; i < 500; i++)
    {
        timerReproject.start();
        reprojectParallelTo16S(src, dst, maps);
        timerReproject.end();
        timerBlend.start();
        blender.blend(dst, blendImage);
        timerBlend.end();
    }
    timer.end();
    printf("%f, %f, %f\n", timer.elapse(), timerReproject.getAccTime(), timerBlend.getAccTime());
    return 0;
}