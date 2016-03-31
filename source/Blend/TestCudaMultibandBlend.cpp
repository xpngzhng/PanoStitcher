#include "CudaPyramid.h"
#include "Timer.h"
#include "ZBlendAlgo.h"
#include "ZBlend.h"
#include "Pyramid.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>

#include <cuda_runtime.h>

int main1()
{
    cv::Mat tempCpu(8, 8, CV_8UC1);
    cv::gpu::GpuMat tempGpu(tempCpu);
    printf("start\n");
    //{
    //    int rows = 1024, cols = 2048, pad = 2;
    //    cv::Mat src(rows, cols, CV_32FC1), srcPad, dst, dstPad;
    //    cv::RNG rng;
    //    rng.fill(src, cv::RNG::NORMAL, 0, 1, true);
    //    ztool::Timer timer;
    //    cv::gpu::GpuMat srcGpu(src), dstGpu(srcGpu.size(), srcGpu.type());
    //    timer.start();
    //    for (int i = 0; i < 80000; i++)
    //    {
    //        //cv::gpu::GpuMat dstGpu;
    //        //timer.start();
    //        func(srcGpu, dstGpu);
    //        //timer.end();
    //        //printf("time = %f, \n", timer.elapse());
    //        //dstGpu.download(dst);
    //    }
    //    timer.end();
    //    printf("time = %f\n", timer.elapse());
    //    printf("end \n");
    //    return 0;
    //}
    {
        int rows = 16, cols = 16, pad = 2;
        cv::Mat src(rows, cols, CV_16SC1), srcPad, dst, dstPad;
        cv::RNG rng;
        rng.fill(src, cv::RNG::UNIFORM, 0, 255, true);
        /*for (int i = 0; i < rows; i++)
        {
        src.col(i).setTo(i + 1);
        }*/
        //src.setTo(1);
        cv::copyMakeBorder(src, srcPad, 2, 2, 2, 2, cv::BORDER_REFLECT_101);

        ztool::Timer timer;
        cv::gpu::GpuMat srcGpu(src), srcPadGpu(srcPad);
        cv::gpu::GpuMat dstGpu, dstPadGpu;
        //timer.start();
        for (int i = 0; i < 1; i++)
        {
            //cv::gpu::GpuMat dstGpu;
            //timer.start();
            pyramidDown16SC1To32SC1(srcGpu, dstGpu, cv::Size(), true);
            //timer.end();
            //printf("time = %f, \n", timer.elapse());
            dstGpu.download(dst);
        }
        //timer.end();
        //printf("elpase = %f\n", timer.elapse());
        //printf("end \n");
        //return 0;
        for (int i = 0; i < 1; i++)
        {
            //cv::gpu::GpuMat dstPadGpu;
            //timer.start();
            pyramidDownPad16SC1To32SC1(srcPadGpu, dstPadGpu, cv::Size(), true);
            //timer.end();
            //printf("time = %f, \n", timer.elapse());
            dstPadGpu.download(dstPad);
        }
        cv::Mat dstPadROI(dstPad, cv::Rect(pad, pad, cols / 2, rows / 2));
        printf("%d\n", cv::countNonZero(dst - dstPadROI));
        cv::imshow("diff", cv::Mat(dst - dstPadROI) != 0);

        std::cout << src << std::endl << std::endl;
        srcPadGpu.download(srcPad);
        std::cout << srcPad << std::endl << std::endl;

        std::cout << dst << std::endl << std::endl;
        std::cout << dstPadROI << std::endl << std::endl;

        cv::Mat dstCpu;
        pyramidDownTo32S(src, dstCpu, cv::Size(), cv::BORDER_WRAP, cv::BORDER_REFLECT_101);
        std::cout << dstCpu << std::endl;
        cv::waitKey(0);
    }
    return 0;
    {
        int rows = 1024, cols = 2048, pad = 2;
        cv::Mat src(rows, cols, CV_16SC4), srcPad, dst, dstPad;
        cv::RNG rng;
        rng.fill(src, cv::RNG::UNIFORM, 0, 255, true);
        cv::copyMakeBorder(src, srcPad, 2, 2, 2, 2, cv::BORDER_REFLECT_101);

        ztool::Timer timer;
        cv::gpu::GpuMat srcGpu(src), srcPadGpu(srcPad);
        for (int i = 0; i < 100; i++)
        {
            cv::gpu::GpuMat dstGpu;
            timer.start();
            pyramidDown16SC4To32SC4(srcGpu, dstGpu, cv::Size(), false);
            timer.end();
            printf("time = %f, \n", timer.elapse());
            dstGpu.download(dst);
        }
        printf("\n");
        for (int i = 0; i < 100; i++)
        {
            cv::gpu::GpuMat dstPadGpu;
            timer.start();
            pyramidDownPad16SC4To32SC4(srcPadGpu, dstPadGpu, cv::Size(), false);
            timer.end();
            printf("time = %f, \n", timer.elapse());
            dstPadGpu.download(dstPad);
        }
        cv::Mat dstPadROI(dstPad, cv::Rect(pad, pad, cols / 2, rows / 2));
        cv::Mat diffC4 = dst - dstPadROI;
        cv::Mat diffC1(diffC4.rows, diffC4.cols * 4, CV_32SC1, diffC4.data, diffC4.step);
        printf("%d\n", cv::countNonZero(diffC1));
        cv::imshow("diff", diffC1 != 0);
        cv::Mat tmp(diffC1, cv::Rect(0, 0, 16, 1));
        std::cout << tmp << std::endl;
        cv::waitKey(0);
    }

    //{
    //    int rows = 8, cols = 8, pad = 2;
    //    cv::Mat src(rows, cols, CV_16SC4), srcPad, dst, dstPad;
    //    cv::RNG rng;
    //    rng.fill(src, cv::RNG::UNIFORM, 0, 255, true);
    //    cv::copyMakeBorder(src, srcPad, 2, 2, 2, 2, cv::BORDER_REFLECT_101);
    //    std::cout << src << std::endl << std::endl;

    //    ztool::Timer timer;
    //    //cv::gpu::GpuMat srcGpu(src), srcPadGpu(srcPad);
    //    cv::gpu::GpuMat srcGpu, srcPadGpu;
    //    srcGpu.upload(src);
    //    srcPadGpu.upload(srcPad);
    //    for (int i = 0; i < 100; i++)
    //    {
    //        cv::gpu::GpuMat dstGpu;
    //        timer.start();
    //        pyramidUp16SC4To16SC4(srcGpu, dstGpu, cv::Size(), false);
    //        timer.end();
    //        printf("time = %f, \n", timer.elapse());
    //        dstGpu.download(dst);
    //    }
    //    std::cout << dst << std::endl << std::endl;
    //    printf("\n");
    //    for (int i = 0; i < 100; i++)
    //    {
    //        cv::gpu::GpuMat dstPadGpu;
    //        timer.start();
    //        pyramidUpPad16SC4To16SC4(srcPadGpu, dstPadGpu, cv::Size(), false);
    //        timer.end();
    //        printf("time = %f, \n", timer.elapse());
    //        dstPadGpu.download(dstPad);
    //    }
    //    std::cout << dstPad << std::endl << std::endl;
    //    cv::Mat dstPadROI(dstPad, cv::Rect(pad, pad, cols * 2, rows * 2));
    //    cv::Mat diffC4 = dst - dstPadROI;
    //    cv::Mat diffC1(diffC4.rows, diffC4.cols * 4, CV_16SC1, diffC4.data, diffC4.step);
    //    printf("%d\n", cv::countNonZero(diffC1));
    //    cv::imshow("diff", diffC1 != 0);
    //    std::cout << diffC1.row(rows * 2 - 2) << std::endl;
    //    cv::waitKey(0);
    //}

    return 0;
}

void cudaMultibandBlend(const cv::gpu::GpuMat& image1, const cv::gpu::GpuMat& image2,
    const cv::gpu::GpuMat& alpha1, const cv::gpu::GpuMat& alpha2,
    cv::gpu::GpuMat& mask1, const cv::gpu::GpuMat& mask2,
    bool horiWrap, int maxLevels, int minLength, cv::gpu::GpuMat& result);

int main2()
{
    cv::Mat blendImage = cv::imread("E:\\Projects\\Stitching\\build\\ZBlend\\blendimage.bmp", -1);
    cv::Mat blendMask = cv::imread("E:\\Projects\\Stitching\\build\\ZBlend\\blendmask.bmp", -1);
    cv::Mat blendRegion = cv::imread("E:\\Projects\\Stitching\\build\\ZBlend\\blendregion.bmp", -1);
    cv::Mat currImage = cv::imread("E:\\Projects\\Stitching\\build\\ZBlend\\currimage.bmp", -1);
    cv::Mat currMask = cv::imread("E:\\Projects\\Stitching\\build\\ZBlend\\currmask.bmp", -1);
    cv::Mat currRegion = cv::imread("E:\\Projects\\Stitching\\build\\ZBlend\\currregion.bmp", -1);

    ztool::Timer timer;
    cv::Mat blendImage32S, blendMask32S, blendRegion32S;
    cv::Mat currImage32S, currMask32S, currRegion32S;
    timer.start();
    blendImage.convertTo(blendImage32S, CV_32S);
    blendMask.convertTo(blendMask32S, CV_32S);
    blendRegion.convertTo(blendRegion32S, CV_32S);
    currImage.convertTo(currImage32S, CV_32S);
    currMask.convertTo(currMask32S, CV_32S);
    currRegion.convertTo(currRegion32S, CV_32S);
    timer.end();
    printf("time elpased = %f\n", timer.elapse());

    cv::Mat blendImageC4(blendImage.size(), CV_8UC4), currImageC4(currImage.size(), CV_8UC4);
    int fromTo[] = { 0, 0, 1, 1, 2, 2 };
    cv::mixChannels(&blendImage, 1, &blendImageC4, 1, fromTo, 3);
    cv::mixChannels(&currImage, 1, &currImageC4, 1, fromTo, 3);
    for (int i = 0; i < 50; i++)
    {
        //ztool::Timer timer;
        //cv::Mat resultC4;
        //timer.start();
        //cudaBlend(blendImageC4, currImageC4, blendMask, currMask, blendRegion, currRegion, false, resultC4);
        //timer.end();
        //printf("%f\n", timer.elapse());

        //cv::Mat blendImage16SC4, gb;
        //blendImageC4.convertTo(blendImage16SC4, CV_16S);
        //cv::gpu::GpuMat src(blendImage16SC4), dstPyr, dstBlur;
        //timer.start();
        //pyramidDown16SC4To32SC4(src, dstPyr, cv::Size());
        //timer.end();
        //printf("down time = %f\n", timer.elapse());
        //timer.start();
        //cv::gpu::GaussianBlur(src, dstBlur, cv::Size(5, 5), 1, 1);
        //timer.end();
        //printf("gpu blur time = %f\n", timer.elapse());
        //timer.start();
        //cv::GaussianBlur(blendImage16SC4, gb, cv::Size(5, 5), 1, 1);
        //timer.end();
        //printf("cpu blur time = %f\n", timer.elapse());

        //cv::imshow("result", result);
        //cv::waitKey(0);
        //cv::imwrite("result.bmp", result);

        cv::gpu::GpuMat blendImageGpu(blendImageC4), currImageGpu(currImageC4);
        cv::gpu::GpuMat blendMaskGpu(blendMask), currMaskGpu(currMask);
        cv::gpu::GpuMat blendRegionGpu(blendRegion), currRegionGpu(currRegion);
        cv::gpu::GpuMat resultGpu;
        timer.start();
        cudaMultibandBlend(blendImageGpu, currImageGpu, blendMaskGpu, currMaskGpu, blendRegionGpu, currRegionGpu, false, 5, 2, resultGpu);
        timer.end();
        printf("pure gpu %f\n", timer.elapse());

        //resultGpu.download(resultC4);
        //cv::Mat result(resultC4.size(), CV_8UC3);
        //cv::mixChannels(&resultC4, 1, &result, 1, fromTo, 3);
        //cv::imwrite("pureimpl.bmp", result);
    }
    for (int i = 0; i < 50; i++)
    {
        ztool::Timer timer;
        cv::Mat result;
        timer.start();
        multibandBlend(blendImage, currImage, blendMask, currMask, blendRegion, currRegion, false, 5, 2, result);
        timer.end();
        printf("%f\n", timer.elapse());
    }
    //cv::imshow("result", result);
    //cv::waitKey(0);
    return 0;
}

int main()
{
    cudaSetDevice(0);
    cudaFree(0);
    printf("setup finish\n");

    //cv::Mat tempCpu(8, 8, CV_8UC1);
    //cv::gpu::GpuMat tempGpu(tempCpu);
    //printf("start\n");
    //system("pause");
    //return 0;

    //printf("%d\n", CV_ELEM_SIZE(CV_8UC1));
    //printf("%d\n", CV_ELEM_SIZE(CV_8UC2));
    //printf("%d\n", CV_ELEM_SIZE(CV_8UC3));
    //printf("%d\n", CV_ELEM_SIZE(CV_8UC4));
    //printf("%d\n", CV_ELEM_SIZE(CV_16UC1));
    //printf("%d\n", CV_ELEM_SIZE(CV_16SC2));
    //printf("%d\n", CV_ELEM_SIZE(CV_32SC1));
    //printf("%d\n", CV_ELEM_SIZE(CV_32SC3));
    //printf("%d\n", CV_ELEM_SIZE(CV_32FC1));
    //printf("%d\n", CV_ELEM_SIZE(CV_32FC4));
    //printf("%d\n", CV_ELEM_SIZE(CV_64FC1));
    //printf("%d\n", CV_ELEM_SIZE(CV_64FC2));
    //for (int i = 1; i < 1024; i += 32)
    //{
    //    cv::gpu::GpuMat mat(100, i, CV_32SC1);
    //    printf("width = %d, depth = %d\n", i, mat.step);
    //}
    //return 0;
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

    std::vector<std::string> contentPaths;
    contentPaths.push_back("F:\\panoimage\\beijing\\reprojimage0.bmp");
    contentPaths.push_back("F:\\panoimage\\beijing\\reprojimage1.bmp");
    contentPaths.push_back("F:\\panoimage\\beijing\\reprojimage2.bmp");
    contentPaths.push_back("F:\\panoimage\\beijing\\reprojimage3.bmp");
    contentPaths.push_back("F:\\panoimage\\beijing\\reprojimage4.bmp");
    contentPaths.push_back("F:\\panoimage\\beijing\\reprojimage5.bmp");
    std::vector<std::string> maskPaths;
    maskPaths.push_back("F:\\panoimage\\beijing\\mask0.bmp");
    maskPaths.push_back("F:\\panoimage\\beijing\\mask1.bmp");
    maskPaths.push_back("F:\\panoimage\\beijing\\mask2.bmp");
    maskPaths.push_back("F:\\panoimage\\beijing\\mask3.bmp");
    maskPaths.push_back("F:\\panoimage\\beijing\\mask4.bmp");
    maskPaths.push_back("F:\\panoimage\\beijing\\mask5.bmp");

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\panoimage\\zhanxiang\\0.bmp");
    //contentPaths.push_back("F:\\panoimage\\zhanxiang\\1.bmp");
    //contentPaths.push_back("F:\\panoimage\\zhanxiang\\2.bmp");
    //contentPaths.push_back("F:\\panoimage\\zhanxiang\\3.bmp");
    //contentPaths.push_back("F:\\panoimage\\zhanxiang\\4.bmp");
    //contentPaths.push_back("F:\\panoimage\\zhanxiang\\5.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\0mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\1mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\2mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\3mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\4mask.bmp");
    //maskPaths.push_back("F:\\panoimage\\zhanxiang\\5mask.bmp");

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
    //contentPaths.push_back("E:\\Projects\\PanoVideo\\build\\GeneratePanoVideo\\ricohimage0.bmp");
    //contentPaths.push_back("E:\\Projects\\PanoVideo\\build\\GeneratePanoVideo\\ricohimage1.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("E:\\Projects\\PanoVideo\\build\\GeneratePanoVideo\\ricohmask0.bmp");
    //maskPaths.push_back("E:\\Projects\\PanoVideo\\build\\GeneratePanoVideo\\ricohmask1.bmp");

    cv::Mat tempCpu(8, 8, CV_8UC1);
    printf("a\n");
    cv::gpu::GpuMat tempGpu(tempCpu);
    printf("start\n");

    ztool::Timer timer;

    int numImages = contentPaths.size();
    std::vector<cv::Mat> images, masks;
    cv::Size imageSize;
    getImagesAndMasks(contentPaths, maskPaths, imageSize, images, masks);

    std::vector<cv::Mat> imagesC4(numImages);
    std::vector<cv::gpu::GpuMat> imagesGpu(numImages), masksGpu(numImages);
    int fromTo[] = { 0, 0, 1, 1, 2, 2 };
    cv::Mat tempC4(imageSize, CV_8UC4), blendImageC4;
    for (int i = 0; i < numImages; i++)
    {
        imagesC4[i].create(imageSize, CV_8UC4);
        cv::mixChannels(&images[i], 1, &imagesC4[i], 1, fromTo, 3);
        timer.start();
        imagesGpu[i].upload(imagesC4[i]);
        timer.end();
        printf("upload C4 time = %f, ", timer.elapse());
        timer.start();
        masksGpu[i].upload(masks[i]);
        timer.end();
        printf("upload C1 time = %f\n", timer.elapse());
    }

    /*{
    CudaTilingMultibandBlend blender;
    cv::gpu::GpuMat blendImageGpu;
    blender.prepare(masks, 20, 2);
    for (int i = 0; i < 100; i++)
    {
    timer.start();
    blender.blend(imagesGpu, masksGpu, blendImageGpu);
    timer.end();
    printf("time elapsed = %f\n", timer.elapse());
    }
    blendImageGpu.download(blendImageC4);
    cv::imshow("blend image", blendImageC4);
    cv::waitKey(0);
    }*/

    {
        CudaTilingMultibandBlendFast blender;
        cv::gpu::GpuMat blendImageGpu;
        blender.prepare(masks, 20, 2);
        for (int i = 0; i < 500; i++)
        {
            //timer.start();
            for (int i = 0; i < numImages; i++)
                imagesGpu[i].upload(imagesC4[i]);
            timer.start();
            blender.blend(imagesGpu, blendImageGpu);
            blendImageGpu.download(blendImageC4);
            timer.end();
            printf("time elapsed = %f\n", timer.elapse());
        }
        blendImageGpu.download(blendImageC4);
        cv::imshow("blend image", blendImageC4);
        cv::waitKey(0);
    }

    printf("\n\n");
    _sleep(1000);

    //{
    //    std::vector<cv::gpu::CudaMem> pinnedMems(numImages);
    //    std::vector<cv::Mat> pinnedImages(numImages);
    //    for (int i = 0; i < numImages; i++)
    //    {
    //        pinnedMems[i] = cv::gpu::CudaMem(imagesC4[i]);
    //        pinnedImages[i] = pinnedMems[i];
    //    }
    //    CudaTilingMultibandBlendFast blender;
    //    cv::gpu::GpuMat blendImageGpu;
    //    blender.prepare(masks, 20, 2);
    //    std::vector<cv::gpu::Stream> streams(numImages);
    //    for (int i = 0; i < 0; i++)
    //    {
    //        timer.start();
    //        for (int j = 0; j < numImages; j++)
    //            streams[j].enqueueUpload(pinnedImages[j], imagesGpu[j]);
    //        blender.blend(imagesGpu, blendImageGpu, streams);
    //        blendImageGpu.download(blendImageC4);
    //        timer.end();
    //        printf("time elapsed = %f\n", timer.elapse());
    //    }
    //    //cv::imshow("blend image stream", blendImageC4);
    //    //cv::waitKey(0);
    //}

    printf("\n\n");
    _sleep(1000);

    //{
    //    std::vector<cv::gpu::CudaMem> pinnedMems(numImages);
    //    std::vector<cv::Mat> pinnedImages(numImages);
    //    for (int i = 0; i < numImages; i++)
    //    {
    //        pinnedMems[i] = cv::gpu::CudaMem(imagesC4[i]);
    //        pinnedImages[i] = pinnedMems[i];
    //    }
    //    std::vector<std::vector<cv::gpu::GpuMat> > imagePyrs(numImages), image32SPyrs(numImages), imageUpPyrs(numImages);
    //    std::vector<std::vector<cv::gpu::GpuMat> > alphaPyrs(numImages), weightPyrs(numImages);
    //    std::vector<cv::gpu::GpuMat> resultPyr, resultUpPyr;
    //    cv::gpu::GpuMat blendImageGpu;
    //    prepare(masks, 20, 2, alphaPyrs, weightPyrs, resultPyr, image32SPyrs, imageUpPyrs, resultUpPyr);
    //    std::vector<cv::gpu::Stream> streams(numImages);
    //    ztool::Timer partTimer;
    //    for (int i = 0; i < 500; i++)
    //    {
    //        timer.start();
    //        /*for (int j = 0; j < numImages; j++)
    //        {
    //            streams[j].enqueueUpload(pinnedImages[j], imagesGpu[j]);
    //            calcImagePyramid(imagesGpu[j], alphaPyrs[j], imagePyrs[j], streams[j], image32SPyrs[j], imageUpPyrs[j]);
    //            streams[j].waitForCompletion();
    //        }*/
    //        for (int j = 0; j < numImages; j++)
    //            streams[j].enqueueUpload(pinnedImages[j], imagesGpu[j]);
    //        for (int j = 0; j < numImages; j++)
    //            calcImagePyramid(imagesGpu[j], alphaPyrs[j], imagePyrs[j], streams[j], image32SPyrs[j], imageUpPyrs[j]);
    //        for (int j = 0; j < numImages; j++)
    //            streams[j].waitForCompletion();
    //        partTimer.start();
    //        calcResult(imagePyrs, weightPyrs, blendImageGpu, resultPyr, resultUpPyr);
    //        blendImageGpu.download(blendImageC4);
    //        partTimer.end();
    //        timer.end();
    //        printf("time elapsed = %f, part = %f\n", timer.elapse(), partTimer.elapse());
    //    }
    //    cv::imshow("blend image stream", blendImageC4);
    //    cv::waitKey(0);
    //}
    return 0;
}

int main4()
{
    {
        //int rows = 49, cols = 98, blockSize = 16;
        //int rows = 49, cols = 49, blockSize = 16;
        int rows = 2048, cols = 4096, blockSize = 16;
        //int rows = 32, cols = 49, blockSize = 16;
        cv::Mat src(rows, cols, CV_16SC1), dst1, dst2, aux;
        cv::RNG rng;
        rng.fill(src, cv::RNG::UNIFORM, 0, 255, true);
        ztool::Timer timer;
        cv::gpu::GpuMat srcGpu(src);
        cv::gpu::GpuMat dstGpu1, dstGpu2, auxGpu;
        printf("begin\n");
        timer.start();
        for (int i = 0; i < 10000; i++)
        {
            //cv::gpu::GpuMat dstGpu;
            //timer.start();
            pyramidDown16SC1To32SC1(srcGpu, dstGpu1, cv::Size(), true);
            //timer.end();
            //printf("time = %f, \n", timer.elapse());
            dstGpu1.download(dst1);
            //timer.end();
            //printf("time = %f, \n", timer.elapse());
        }
        timer.end();
        printf("elpase = %f\n", timer.elapse());
        //printf("end \n");
        //return 0;
        printf("\n\n");
        _sleep(100);
        cv::Mat horiTab, vertTab;
        getIndexTab(cols, blockSize, cv::BORDER_WRAP, horiTab);
        getIndexTab(rows, blockSize, cv::BORDER_REFLECT_101, vertTab);
        cv::gpu::GpuMat horiTabGpu(horiTab), vertTabGpu(vertTab);
        dstGpu2.create((rows + 1) >> 1, (cols + 1) >> 1, CV_32SC1);
        auxGpu.create(rows, (cols + 1) >> 1, CV_32SC1);
        timer.start();
        for (int i = 0; i < 10000; i++)
        {
            //cv::gpu::GpuMat dstPadGpu;
            //timer.start();
            pyramidDown16SC1To32SC1(srcGpu, dstGpu2, auxGpu, horiTabGpu, vertTabGpu);
            //timer.end();
            //printf("time = %f, \n", timer.elapse());
            dstGpu2.download(dst2);
            //auxGpu.download(aux);
            //timer.end();
            //printf("time = %f, \n", timer.elapse());
        }
        timer.end();
        printf("elpase = %f\n", timer.elapse());
        //std::cout << src << std::endl << std::endl;
        //std::cout << dst1 << std::endl << std::endl;
        //std::cout << dst2 << std::endl << std::endl;
        //std::cout << aux << std::endl << std::endl;
        //std::cout << horiTab << std::endl << std::endl;
        //std::cout << vertTab << std::endl << std::endl;

        cv::Mat dst;
        pyramidDownTo32S(src, dst, cv::Size(), cv::BORDER_WRAP, cv::BORDER_REFLECT_101);
        //std::cout << dst << std::endl;

        printf("%d\n", cv::countNonZero(dst - dst1));
        cv::imshow("diff", cv::Mat(dst - dst1) != 0);
        cv::waitKey(0);
    }
    return 0;
}