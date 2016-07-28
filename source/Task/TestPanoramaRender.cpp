#include "RicohUtil.h"
#include "ZReproject.h"
#include "RunTimeObjects.h"
#include "Timer.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <fstream>

int main()
{
    std::vector<std::string> imagePaths;
    std::vector<PhotoParam> params;

    imagePaths.push_back("F:\\panoimage\\zhanxiang\\0.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\1.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\2.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\3.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\4.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\5.jpg");
    //loadPhotoParamFromXML("F:\\panoimage\\zhanxiang\\zhanxiang.xml", params);
    //double PI = 3.1415926;
    //rotateCameras(params, 0, 35.264 / 180 * PI, PI / 4);

    int numImages = imagePaths.size();
    std::vector<cv::Mat> src(numImages);
    for (int i = 0; i < numImages; i++)
        src[i] = cv::imread(imagePaths[i]);

    cv::Size dstSize(1920, 960);
    ztool::Timer t;

    std::string camPath = "F:\\panoimage\\zhanxiang\\zhanxiang.xml";
    CPUPanoramaRender cpuRender;
    cpuRender.prepare(camPath, false, src[0].size(), dstSize);
    cv::Mat cpuBlendImage;

    t.start();
    for (int i = 0; i < 100; i++)
    cpuRender.render(src, cpuBlendImage);
    t.end();
    printf("%f\n", t.elapse());

    cv::imshow("cpu", cpuBlendImage);
    cv::waitKey(0);

    bool ok = ioclInit();
    if (!ok)
    {
        printf("could not init intel opencl\n");
        return 0;
    }

    std::vector<IOclMat> ioclSrc(numImages);
    cv::Mat temp8U;
    for (int i = 0; i < numImages; i++)
    {
        cv::cvtColor(src[i], temp8U, CV_BGR2BGRA);
        ioclSrc[i].upload(temp8U, iocl::ocl->context);
    }
    IOclPanoramaRender gpuRender;
    gpuRender.prepare(camPath, false, src[0].size(), dstSize);
    IOclMat gpuBlendImage;

    t.start();
    for (int i = 0; i < 100; i++)
    gpuRender.render(ioclSrc, gpuBlendImage);
    t.end();
    printf("%f\n", t.elapse());

    cv::Mat header = gpuBlendImage.toOpenCVMat();
    cv::imshow("gpu", header);

    cv::waitKey(0);

    return 0;
}