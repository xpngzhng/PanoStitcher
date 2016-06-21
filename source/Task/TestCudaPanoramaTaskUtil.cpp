#include "CudaPanoramaTaskUtil.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "Timer.h"

int main()
{
    cv::Mat origC3 = cv::imread("E:\\Projects\\PanoSticher\\build\\Test\\snapshot3.bmp");
    cv::Mat origC4, origYUV;
    cv::cvtColor(origC3, origC4, CV_BGR2BGRA);
    cv::cvtColor(origC3, origYUV, CV_BGR2YUV_I420);
    cv::imshow("yuv", origYUV);
    cv::imwrite("yuv.bmp", origYUV);
    cv::waitKey(0);

    cv::cuda::GpuMat image(origC4);
    CudaLogoFilter filter;
    filter.init(image.cols, image.rows);
    filter.addLogo(image);

    cv::Mat proc;
    image.download(proc);
    cv::imshow("proc", proc);
    cv::waitKey(0);

    cv::cuda::GpuMat bgr32(origC4);
    int rows = origC4.rows, cols = origC4.cols;
    cv::cuda::GpuMat y1(rows, cols, CV_8UC1), u(rows / 2, cols / 2, CV_8UC1), v(rows / 2, cols / 2, CV_8UC1), 
        y2(rows, cols, CV_8UC1), uv(rows / 2, cols, CV_8UC1);
    cv::cuda::GpuMat bgr1(rows, cols, CV_8UC4), bgr2(rows, cols, CV_8UC4);
    ztool::Timer t;
    for (int i = 0; i < 1000; i++)
    {
        cvtBGR32ToYUV420P(bgr32, y1, u, v);
        cvtBGR32ToNV12(bgr32, y2, uv);
        cvtYUV420PToBGR32(y1, u, v, bgr1);
        cvtNV12ToBGR32(y2, uv, bgr2);
    }
    t.end();
    printf("%f\n", t.elapse());

    cv::Mat bgr1cpu, bgr2cpu;
    bgr1.download(bgr1cpu);
    bgr2.download(bgr2cpu);
    cv::imshow("bgr1", bgr1cpu);
    cv::imshow("bgr2", bgr2cpu);
    
    cv::Mat y1cpu, ucpu, vcpu, y2cpu, uvcpu;
    y1.download(y1cpu);
    u.download(ucpu);
    v.download(vcpu);
    y2.download(y2cpu);
    uv.download(uvcpu);
    cv::imshow("y1", y1cpu);
    cv::imshow("u", ucpu);
    cv::imshow("v", vcpu);
    cv::imshow("y2", y2cpu);
    cv::imshow("uv", uvcpu);
    cv::waitKey(0);
    
    return 0;
}