#include "ZBlend.h"
#include "ZBlendAlgo.h"
#include "Pyramid.h"
#include "Timer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thread>
#include <vector>
#include <string>

void compare(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst)
{
    CV_Assert(src1.data && src1.type() == CV_8UC3 &&
        src2.data && src2.type() == CV_8UC3 && src1.size() == src2.size());

    int rows = src1.rows, cols = src1.cols;
    dst.create(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrSrc1 = src1.ptr<unsigned char>(i);
        const unsigned char* ptrSrc2 = src2.ptr<unsigned char>(i);
        unsigned char* ptrDst = dst.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            /*if (ptrSrc1[0] == ptrSrc2[0] &&
                ptrSrc1[1] == ptrSrc2[1] &&
                ptrSrc1[2] == ptrSrc2[2])*/
            if (abs(int(ptrSrc1[0] - int(ptrSrc2[0]))) < 2 &&
                abs(int(ptrSrc1[1] - int(ptrSrc2[1]))) < 2 &&
                abs(int(ptrSrc1[2] - int(ptrSrc2[2]))) < 2)
                *ptrDst = 0;
            else
                *ptrDst = 255;
            ptrSrc1 += 3;
            ptrSrc2 += 3;
            ptrDst++;
        }
    }
}

void func(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, int numLoops)
{
    int numImages = images.size();
    std::vector<cv::Mat> localImages(numImages), localMasks(numImages);
    for (int i = 0; i < numImages; i++)
    {
        localImages[i] = images[i].clone();
        localMasks[i] = masks[i].clone();
    }
    TilingMultibandBlend blender;
    blender.prepare(localMasks, 20, 2);
    cv::Mat result;
    //ztool::Timer timer;
    for (int i = 0; i < numLoops; i++)
        blender.blend(localImages, localMasks, result);
    //timer.end();
    //printf("%f\n", timer.elapse());
}

void funcFast(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, int numLoops)
{
    //char* c = (char*)malloc(1);
    int numImages = images.size();
    std::vector<cv::Mat> localImages(numImages), localMasks(numImages);
    for (int i = 0; i < numImages; i++)
    {
        localImages[i] = images[i].clone();
        localMasks[i] = masks[i].clone();
    }
    TilingMultibandBlendFast blender;
    blender.prepare(localMasks, 20, 2);
    localMasks.clear();
    cv::Mat result;
    //ztool::Timer timer;
    for (int i = 0; i < numLoops; i++)
        blender.blend(localImages, result);
    //timer.end();
    //printf("%f\n", timer.elapse());
    system("pause");
}

void funcParallel(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, int numLoops)
{
    int numImages = images.size();
    std::vector<cv::Mat> localImages(numImages), localMasks(numImages);
    for (int i = 0; i < numImages; i++)
    {
        localImages[i] = images[i].clone();
        localMasks[i] = masks[i].clone();
    }
    TilingMultibandBlendFastParallel blender;
    blender.prepare(localMasks, 20, 2);
    localMasks.clear();
    cv::Mat result;
    //ztool::Timer timer;
    for (int i = 0; i < numLoops; i++)
        blender.blend(localImages, result);
    //timer.end();
    //printf("%f\n", timer.elapse());
}

//void funcPyramid(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, int numLoops)
//{
//    cv::Mat imageDown, imageUp, maskDown, maskUp;
//    cv::Mat imageDown32S, maskDown32S;
//    cv::Mat aux1(1024 * 1024, 1, CV_32SC1);
//    cv::Mat aux2(1024 * 1024, 1, CV_32SC1);
//    unsigned char * d1 = aux1.data, *d2 = aux2.data;
//    for (int i = 0; i < numLoops; i++)
//    {
//        //cv::pyrDown(images[0], imageDown);
//        pyramidDown(images[0], imageDown, d1, d2, cv::Size());
//        pyramidUp(images[0], imageUp, d1, d2, cv::Size());
//        pyramidDown(masks[0], maskDown, d1, d2, cv::Size());
//        pyramidUp(masks[0], maskUp, d1, d2, cv::Size());
//        pyramidDownTo32S(images[0], imageDown32S, d1, d2, cv::Size());
//        pyramidDownTo32S(masks[0], maskDown32S, d1, d2, cv::Size());
//    }
//    system("pause");
//}

void fakePyramid(const cv::Mat& src, cv::Mat& dst)
{
    int rows = src.rows, cols = src.cols;
    dst.create(rows, cols, CV_64FC1);
    std::vector<int> vec(cols * 16);
    std::vector<float> vecf(cols);
    for (int i = 0; i < rows; i++)
    {
        double* ptr = dst.ptr<double>(i);
        for (int j = 0; j < cols; j++)
        {
            *(ptr++) = sin(i);
        }
    }
}

void fakePyramid2(const cv::Mat& src, cv::Mat& dst)
{
    int rows = src.rows, cols = src.cols;
    dst.create(rows, cols, CV_64FC1);
    std::vector<char> vec(cols * 8);
    std::vector<float> vecf(cols *5);
    for (int i = 0; i < rows; i++)
    {
        double* ptr = dst.ptr<double>(i);
        for (int j = 0; j < cols; j++)
        {
            *(ptr++) = sin(i) * cos(j);
        }
    }
}

void funcFakePyramid(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, int numLoops)
{
    cv::Mat result;
    for (int i = 0; i < numLoops; i++)
    {
        fakePyramid(images[0], result);
        fakePyramid2(images[0], result);
    }
    system("pause");
}

int main()
{
    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("C:\\Users\\zhengxuping\\Desktop\\PanoReprojection\\PanoTest\\image0.bmp");
    //contentPaths.push_back("C:\\Users\\zhengxuping\\Desktop\\PanoReprojection\\PanoTest\\image1.bmp");
    //contentPaths.push_back("C:\\Users\\zhengxuping\\Desktop\\PanoReprojection\\PanoTest\\image2.bmp");
    //contentPaths.push_back("C:\\Users\\zhengxuping\\Desktop\\PanoReprojection\\PanoTest\\image3.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("C:\\Users\\zhengxuping\\Desktop\\PanoReprojection\\PanoTest\\mask0.bmp");
    //maskPaths.push_back("C:\\Users\\zhengxuping\\Desktop\\PanoReprojection\\PanoTest\\mask1.bmp");
    //maskPaths.push_back("C:\\Users\\zhengxuping\\Desktop\\PanoReprojection\\PanoTest\\mask2.bmp");
    //maskPaths.push_back("C:\\Users\\zhengxuping\\Desktop\\PanoReprojection\\PanoTest\\mask3.bmp");

    std::vector<std::string> contentPaths;
    contentPaths.push_back("F:\\panoimage\\changtai\\reprojimage0.bmp");
    contentPaths.push_back("F:\\panoimage\\changtai\\reprojimage1.bmp");
    contentPaths.push_back("F:\\panoimage\\changtai\\reprojimage2.bmp");
    contentPaths.push_back("F:\\panoimage\\changtai\\reprojimage3.bmp");
    contentPaths.push_back("F:\\panoimage\\changtai\\reprojimage4.bmp");
    contentPaths.push_back("F:\\panoimage\\changtai\\reprojimage5.bmp");
    std::vector<std::string> maskPaths;
    maskPaths.push_back("F:\\panoimage\\changtai\\mask0.bmp");
    maskPaths.push_back("F:\\panoimage\\changtai\\mask1.bmp");
    maskPaths.push_back("F:\\panoimage\\changtai\\mask2.bmp");
    maskPaths.push_back("F:\\panoimage\\changtai\\mask3.bmp");
    maskPaths.push_back("F:\\panoimage\\changtai\\mask4.bmp");
    maskPaths.push_back("F:\\panoimage\\changtai\\mask5.bmp");

    int numImages = contentPaths.size();
    std::vector<cv::Mat> images, masks;
    cv::Size imageSize;
    getImagesAndMasks(contentPaths, maskPaths, imageSize, images, masks);

    /*cv::Mat result1, result2, result3;

    TilingMultibandBlend blend;
    blend.prepare(masks, 20, 2);
    blend.blend(images, masks, result1);

    TilingMultibandBlendFast blendF;
    blendF.prepare(masks, 20, 2);
    blendF.blend(images, result2);

    TilingMultibandBlendFastParallel blendP;
    blendP.prepare(masks, 20, 2);
    blendP.blend(images, result3);

    cv::Mat comp1, comp2, comp3;
    compare(result1, result2, comp1);
    compare(result1, result3, comp2);
    compare(result2, result3, comp3);

    cv::imshow("result1", result1);
    cv::imshow("result2", result2);
    cv::imshow("result3", result3);
    cv::imshow("comp1", comp1);
    cv::imshow("comp2", comp2);
    cv::imshow("comp3", comp3);
    cv::waitKey(0);

    return 0;*/

    ztool::Timer timer;

    //std::thread t1(func, std::ref(images), std::ref(masks), 100);
    //std::thread t2(func, std::ref(images), std::ref(masks), 100);
    //std::thread t3(func, std::ref(images), std::ref(masks));
    /*std::thread t4(func, std::ref(images), std::ref(masks));
    std::thread t5(func, std::ref(images), std::ref(masks));
    std::thread t6(func, std::ref(images), std::ref(masks));*/
    //t1.join();
    //t2.join();
    //t3.join();
    /*t4.join();
    t5.join();
    t6.join();*/

    /*timer.start();
    func(images, masks, 100);
    timer.end();
    printf("%f\n", timer.elapse());*/


    /*timer.start();
    funcFast(images, masks, 1000);
    timer.end();
    printf("%f\n", timer.elapse());*/

    timer.start();
    funcParallel(images, masks, 500);
    timer.end();
    printf("%f\n", timer.elapse());

    /*timer.start();
    funcFakePyramid(images, masks, 5000);
    timer.end();
    printf("%f\n", timer.elapse());*/

    /*int maxNumThreas = 6;
    int loopStep = 50;
    int maxNumLoopStep = 6;
    for (int i = 1; i <= maxNumThreas; i++)
    {
        for (int j = 1; j <= maxNumLoopStep; j++)
        {
            timer.start();
            std::vector<std::unique_ptr<std::thread> > ts(i);
            for (int t = 0; t < i; t++)
                ts[t].reset(new std::thread(func, std::ref(images), std::ref(masks), j * loopStep));
            for (int t = 0; t < i; t++)
                ts[t]->join();
            timer.end();
            printf("t = %d, l = %3d, %9.6f\n", i, j * loopStep, timer.elapse());
        }
    }*/

    //char* c = (char*)malloc(1);

    //timer.end();
    //printf("%f\n", timer.elapse());
    return 0;
}

// use global images
//t = 1, l = 50, 6.775732
//t = 1, l = 100, 13.486186
//t = 1, l = 150, 19.956018
//t = 1, l = 200, 26.945191
//t = 1, l = 250, 33.528282
//t = 1, l = 300, 40.151085
//t = 2, l = 50, 8.250155
//t = 2, l = 100, 16.193831
//t = 2, l = 150, 24.551241
//t = 2, l = 200, 32.537970
//t = 2, l = 250, 40.517956
//t = 2, l = 300, 48.013669
//t = 3, l = 50, 10.918567
//t = 3, l = 100, 21.959334
//t = 3, l = 150, 32.478960
//t = 3, l = 200, 42.543967
//t = 3, l = 250, 52.996488
//t = 3, l = 300, 63.288188
//t = 4, l = 50, 13.859673
//t = 4, l = 100, 27.275501
//t = 4, l = 150, 41.101615
//t = 4, l = 200, 54.523178
//t = 4, l = 250, 67.897593
//t = 4, l = 300, 81.659181
//t = 5, l = 50, 17.145828
//t = 5, l = 100, 33.688833
//t = 5, l = 150, 51.118489
//t = 5, l = 200, 70.777801
//t = 5, l = 250, 86.186028
//t = 5, l = 300, 101.059971
//t = 6, l = 50, 20.713179
//t = 6, l = 100, 40.546216
//t = 6, l = 150, 61.062087
//t = 6, l = 200, 82.375783
//t = 6, l = 250, 102.841952
//t = 6, l = 300, 125.691677

// use local images
//t = 1, l = 50, 6.501849
//t = 1, l = 100, 12.912278
//t = 1, l = 150, 19.041279
//t = 1, l = 200, 25.922806
//t = 1, l = 250, 31.659684
//t = 1, l = 300, 38.015318
//t = 2, l = 50, 8.459057
//t = 2, l = 100, 16.654841
//t = 2, l = 150, 24.015945
//t = 2, l = 200, 31.927514
//t = 2, l = 250, 40.542233
//t = 2, l = 300, 49.332222
//t = 3, l = 50, 10.816548
//t = 3, l = 100, 21.286417
//t = 3, l = 150, 31.405726
//t = 3, l = 200, 42.310718
//t = 3, l = 250, 52.310543
//t = 3, l = 300, 63.277818
//t = 4, l = 50, 13.763444
//t = 4, l = 100, 27.263571
//t = 4, l = 150, 40.827993
//t = 4, l = 200, 53.971797
//t = 4, l = 250, 67.818349
//t = 4, l = 300, 81.187141
//t = 5, l = 50, 17.665572
//t = 5, l = 100, 34.652918
//t = 5, l = 150, 53.492598
//t = 5, l = 200, 70.002306
//t = 5, l = 250, 86.515721
//t = 5, l = 300, 102.478435
//t = 6, l = 50, 20.913238
//t = 6, l = 100, 41.185561
//t = 6, l = 150, 61.334610
//t = 6, l = 200, 81.146754
//t = 6, l = 250, 101.738680
//t = 6, l = 300, 125.377796
