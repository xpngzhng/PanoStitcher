#include "ZReproject.h"
#include "IntelOpenCLInterface.h"
#include "RunTimeObjects.h"
#include "Pyramid.h"
#include "MatOp.h"
#include "../../source/Blend/Timer.h"
#include "../../source/Blend/ZBlend.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>

void getWeightsLinearBlend32F(const std::vector<cv::Mat>& masks, int radius, std::vector<cv::Mat>& weights);

static void retrievePaths(const std::string& fileName, std::vector<std::string>& paths)
{
    paths.clear();
    std::ifstream f(fileName);
    std::string temp;
    while (!f.eof())
    {
        std::getline(f, temp);
        if (!temp.empty())
            paths.push_back(temp);
    }
}

int main1(int argc, char** argv)
{
    //cv::Size dstSize = cv::Size(3072, 1536);
    //cv::Size dstSize = cv::Size(2560, 1280);
    cv::Size dstSize = cv::Size(2048, 1024);

    std::vector<std::string> paths;
    //retrievePaths("F:\\panoimage\\beijing\\filelist.txt", paths);
    //std::string configFilePath = "F:\\panoimage\\beijing\\temp_camera_param.xml";
    retrievePaths("F:\\panoimage\\detuoffice2\\filelist.txt", paths);
    std::string configFilePath = "F:\\panoimage\\detuoffice2\\detu.xml";

    int numImages = paths.size();
    std::vector<cv::Mat> src(numImages);
    for (int i = 0; i < numImages; i++)
        src[i] = cv::imread(paths[i]);

    std::vector<PhotoParam> params;
    loadPhotoParamFromXML(configFilePath, params);

    std::vector<cv::Mat> maps, masks;
    getReprojectMapsAndMasks(params, src[0].size(), dstSize, maps, masks);

    //std::vector<cv::Mat> dst;
    //reprojectParallel(src, dst, maps);

    //for (int i = 0; i < numImages; i++)
    //{
    //    cv::imshow("image", dst[i]);
    //    cv::waitKey(0);
    //}
    //return 0;

    //cv::imshow("image", dst[0]);
    //cv::waitKey(0);

    //int count = 100;
    //ztool::Timer t;
    //t.start();
    //dst.resize(numImages);
    //for (int i = 0; i < count; i++)
    //    reprojectParallel(src, dst, maps);
    //t.end();
    //printf("%f\n", t.elapse());

    std::vector<cv::Mat> xmaps32F(numImages), ymaps32F(numImages);
    cv::Mat map32F;
    for (int i = 0; i < numImages; i++)
    {
        xmaps32F[i].create(dstSize, CV_32FC1);
        ymaps32F[i].create(dstSize, CV_32FC1);
        cv::Mat arr[] = { xmaps32F[i], ymaps32F[i] };
        maps[i].convertTo(map32F, CV_32F);
        cv::split(map32F, arr);
    }

    std::vector<cv::Mat> weights;
    getWeightsLinearBlend32F(masks, 50, weights);

    // Create the necessary OpenCL objects up to device queue.
    //OpenCLBasic oclobjects("Intel", "GPU");
    bool ok = iocl::init();
    if (!ok)
    {
        printf("OpenCL init failed\n");
        return 0;
    }

    OpenCLBasic& oclobjects = *iocl::ocl;

    std::vector<iocl::UMat> srcMats(numImages);
    std::vector<iocl::UMat> xmapMats(numImages), ymapMats(numImages), weightMats(numImages);
    for (int i = 0; i < numImages; i++)
    {
        srcMats[i].create(src[i].rows, src[i].cols, CV_8UC4);
        xmapMats[i].create(dstSize.height, dstSize.width, CV_32FC1);
        ymapMats[i].create(dstSize.height, dstSize.width, CV_32FC1);
        weightMats[i].create(dstSize.height, dstSize.width, CV_32FC1);

        cv::Mat srcMatWrapper(srcMats[i].rows, srcMats[i].cols, srcMats[i].type, srcMats[i].data, srcMats[i].step);
        cv::Mat xmapMatWrapper(xmapMats[i].rows, xmapMats[i].cols, xmapMats[i].type, xmapMats[i].data, xmapMats[i].step);
        cv::Mat ymapMatWrapper(ymapMats[i].rows, ymapMats[i].cols, ymapMats[i].type, ymapMats[i].data, ymapMats[i].step);
        cv::cvtColor(src[i], srcMatWrapper, CV_BGR2BGRA);
        xmaps32F[i].copyTo(xmapMatWrapper);
        ymaps32F[i].copyTo(ymapMatWrapper);

        cv::Mat weightWrapper(weightMats[i].rows, weightMats[i].cols, weightMats[i].type, weightMats[i].data, weightMats[i].step);
        weights[i].copyTo(weightWrapper);
    }

    iocl::UMat dstMat32F;
    dstMat32F.create(dstSize.height, dstSize.width, CV_32FC4);

    iocl::UMat dstMat;
    dstMat.create(dstSize.height, dstSize.width, CV_8UC4);

    iocl::UMat dstMat16S;
    dstMat16S.create(dstSize.height, dstSize.width, CV_16SC4);
    cv::Mat dst;
    try
    {
        for (int i = 0; i < numImages; i++)
        {
            ioclReproject(srcMats[i], dstMat, xmapMats[i], ymapMats[i]);
            cv::Mat head = dstMat.toOpenCVMat();
            cv::imshow("rprj", head);

            ioclReprojectTo16S(srcMats[i], dstMat16S, xmapMats[i], ymapMats[i]);
            head = dstMat16S.toOpenCVMat();
            head.convertTo(dst, CV_8U);
            cv::imshow("rprj16S", dst);

            cv::waitKey(0);
        }
    }
    catch (std::exception& e)
    {
        printf("%s\n", e.what());
    }
    //return 0;

    int width = dstSize.width;
    int height = dstSize.height;
    int ret;

    try
    {
        int numIters = 1000;
        ztool::Timer t;

        for (int k = 0; k < numIters; k++)
        {
            setZero(dstMat32F);
            for (int i = 0; i < numImages; i++)
                ioclReprojectWeightedAccumulateTo32F(srcMats[i], dstMat32F, xmapMats[i], ymapMats[i], weightMats[i]);
        }
        t.end();
        printf("time = %f\n", t.elapse() * 1000 / numIters);

        cv::Mat d;
        cv::Mat dstMat32FWrapper(dstMat32F.rows, dstMat32F.cols, dstMat32F.type, dstMat32F.data, dstMat32F.step);
        dstMat32FWrapper.convertTo(d, CV_8U);

        cv::imshow("dst", d);
        cv::waitKey(0);

    }
    catch (const Error& error)
    {
        cerr << "[ ERROR ] Sample application specific error: " << error.what() << "\n";
        ret = EXIT_FAILURE;
    }
    catch (const std::exception& error)
    {
        cerr << "[ ERROR ] " << error.what() << "\n";
        ret = EXIT_FAILURE;
    }
    catch (...)
    {
        cerr << "[ ERROR ] Unknown/internal error happened.\n";
        ret = EXIT_FAILURE;
    }

    return ret;
}

template<typename ElemType, int Depth, int NumChannels>
void compare(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst)
{
    CV_Assert(src1.data && src2.data && src1.depth() == Depth && src2.depth() == Depth &&
        src1.channels() == NumChannels && src2.channels() == NumChannels && src1.size() == src2.size());

    int rows = src1.rows, cols = src1.cols;
    dst.create(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++)
    {
        const ElemType* ptrSrc1 = src1.ptr<ElemType>(i);
        const ElemType* ptrSrc2 = src2.ptr<ElemType>(i);
        unsigned char* ptrDst = dst.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            int same = 1;
            for (int k = 0; k < NumChannels; k++)
            {
                if (ptrSrc1[k] != ptrSrc2[k])
                {
                    same = 0;
                    break;
                }
            }
            *(ptrDst++) = same ? 0 : 255;
            if (!same)
            {
                printf("diff at (%3d, %3d) ", j, i);
                for (int k = 0; k < NumChannels; k++)
                    printf("%d ", ptrSrc1[k]);
                printf("vs ");
                for (int k = 0; k < NumChannels; k++)
                    printf("%d ", ptrSrc2[k]);
                printf("\n");
            }
            ptrSrc1 += NumChannels;
            ptrSrc2 += NumChannels;
        }
    }
}

void calcDistImage(cv::Mat& dist, cv::Size size)
{
    dist.create(size, CV_32FC1);
    float centx = size.width / 2.F, centy = size.height / 2.F;
    float maxDist = centx * centx + centy * centy;
    float scale = 1.F / maxDist;
    for (int i = 0; i < size.height; i++)
    {
        float* ptr = dist.ptr<float>(i);
        for (int j = 0; j < size.width; j++)
        {
            float diffx = j - centx;
            float diffy = i - centy;
            //*(ptr++) = (maxDist - diffx * diffx - diffy * diffy) * scale;
            *(ptr++) = (diffx * diffx + diffy * diffy) * scale;
        }
    }
}

#include "../Blend/Pyramid.h"
int main2()
{
    bool ok = iocl::init();
    if (!ok)
    {
        printf("OpenCL init failed\n");
        return 0;
    }

    std::vector<std::string> paths;
    //retrievePaths("F:\\panoimage\\beijing\\filelist.txt", paths);
    //std::string configFilePath = "F:\\panoimage\\beijing\\temp_camera_param.xml";
    retrievePaths("F:\\panoimage\\detuoffice2\\filelist.txt", paths);

    cv::Mat color = cv::imread(paths[0]);
    cv::Mat colorSrc;
    cv::cvtColor(color, colorSrc, CV_BGR2BGRA);
    cv::Mat graySrc;
    cv::cvtColor(color, graySrc, CV_BGR2GRAY);
    //graySrc.setTo(255);
    //colorSrc.setTo(cv::Scalar::all(255));

    ztool::Timer t;

    int numRuns = 10;

    cv::Mat colorDst, colorDst32S, grayDst, grayDst32S;
    t.start();
    for (int i = 0; i < numRuns; i++)
    pyramidDown(colorSrc, colorDst, cv::Size(), cv::BORDER_WRAP, cv::BORDER_REFLECT_101);
    t.end();
    printf("t = %f\n", t.elapse());
    pyramidDownTo32S(colorSrc, colorDst32S, cv::Size(), cv::BORDER_WRAP, cv::BORDER_REFLECT_101);
    pyramidDown(graySrc, grayDst, cv::Size(), cv::BORDER_WRAP, cv::BORDER_REFLECT_101);
    pyramidDownTo32S(graySrc, grayDst32S, cv::Size(), cv::BORDER_WRAP, cv::BORDER_REFLECT_101);

    cv::Mat header, diff, cvtBack;

    // test floating point version
    /*
    {
        iocl::UMat iColorSrc32F(colorSrc.size(), CV_32FC4);
        iocl::UMat iGraySrc32F(graySrc.size(), CV_32FC1);
        iocl::UMat iColorDst32F, iGrayDst32F;

        header = iColorSrc32F.toOpenCVMat();
        colorSrc.convertTo(header, CV_32F);
        t.start();
        for (int i = 0; i < numRuns; i++)
            pyramidDown32FC4(iColorSrc32F, iColorDst32F, cv::Size());
        t.end();
        printf("t = %f\n", t.elapse());
        header = iColorDst32F.toOpenCVMat();
        header.convertTo(cvtBack, CV_8U);
        compare<unsigned char, CV_8U, 4>(colorDst, cvtBack, diff);
        cv::imshow("cpu color", colorDst);
        cv::imshow("intel gpu color", cvtBack);
        cv::imshow("diff color", diff);
        cv::waitKey(0);

        header = iGraySrc32F.toOpenCVMat();
        graySrc.convertTo(header, CV_32F);
        pyramidDown32FC1(iGraySrc32F, iGrayDst32F, cv::Size());
        header = iGrayDst32F.toOpenCVMat();
        header.convertTo(cvtBack, CV_8U);
        compare<unsigned char, CV_8U, 1>(grayDst, cvtBack, diff);
        cv::imshow("cpu gray", grayDst);
        cv::imshow("intel gpu gray", cvtBack);
        cv::imshow("diff gray", diff);
        cv::waitKey(0);
    }
    */
    // test short version
    /*
    {
        iocl::UMat iGraySrc16S(graySrc.size(), CV_16SC1);
        header = iGraySrc16S.toOpenCVMat();
        graySrc.convertTo(header, CV_16S);
        iocl::UMat iGrayDst16S;
        pyramidDown16SC1To16SC1(iGraySrc16S, iGrayDst16S);
        header = iGrayDst16S.toOpenCVMat();
        header.convertTo(cvtBack, CV_8U);
        compare<unsigned char, CV_8U, 1>(grayDst, cvtBack, diff);
        cv::imshow("diff gray 16S", diff);
        cv::waitKey(0);

        iocl::UMat iGrayDst32S;
        pyramidDown16SC1To32SC1(iGraySrc16S, iGrayDst32S);
        header = iGrayDst32S.toOpenCVMat();
        cv::Mat diffFor32S;
        compare<int, CV_32S, 1>(grayDst32S, header, diffFor32S);
        cv::imshow("diff gray 32S", diffFor32S);
        cv::waitKey(0);
    }
    */
    // test uchar version
    /*
    {
        iocl::UMat iColorSrc(colorSrc.size(), CV_8UC4);
        iocl::UMat iGraySrc(graySrc.size(), CV_8UC1);
        iocl::UMat iColorDst, iColorDst32S, iGrayDst;

        header = iColorSrc.toOpenCVMat();
        colorSrc.copyTo(header);

        //cv::imshow("color", header);
        t.start();
        for (int i = 0; i < numRuns; i++)
            pyramidDown8UC4To8UC4(iColorSrc, iColorDst, cv::Size());
        t.end();
        printf("t = %f\n", t.elapse());
        header = iColorDst.toOpenCVMat();

        compare<unsigned char, CV_8U, 4>(colorDst, header, diff);

        cv::imshow("cpu color", colorDst);
        cv::imshow("intel gpu color", header);
        cv::imshow("diff color", diff);
        cv::waitKey(0);

        pyramidDown8UC4To32SC4(iColorSrc, iColorDst32S, cv::Size());
        header = iColorDst32S.toOpenCVMat();
        cv::Mat diffColor32S;
        compare<int, CV_32S, 4>(colorDst32S, header, diffColor32S);
        cv::imshow("diff 32s", diffColor32S);
        cv::waitKey(0);

        header = iGraySrc.toOpenCVMat();
        graySrc.copyTo(header);
        pyramidDown8UC1To8UC1(iGraySrc, iGrayDst, cv::Size());
        header = iGrayDst.toOpenCVMat();

        cv::Mat diffGray;
        compare<unsigned char, CV_8U, 1>(grayDst, header, diffGray);

        cv::imshow("cpu gray", grayDst);
        cv::imshow("intel gpu gray", header);
        cv::imshow("diff gray", diffGray);
        cv::waitKey(0);
    }
    */
    // test pyramid down scale version
    /*
    {
        cv::Mat dist;
        calcDistImage(dist, cv::Size((colorSrc.cols + 1) / 2, (colorSrc.rows + 1) / 2));
        cv::imshow("dist", dist);
        cv::waitKey(0);

        iocl::UMat scale32S(dist.size(), CV_32SC1);
        header = scale32S.toOpenCVMat();
        dist.convertTo(header, CV_32S, 256 * 256);

        iocl::UMat iColorSrc16S(colorSrc.size(), CV_16SC4), iColorDst16S;
        header = iColorSrc16S.toOpenCVMat();
        colorSrc.convertTo(header, CV_16S);

        t.start();
        for (int i = 0; i < numRuns; i++)
            pyramidDown16SC4To16SC4(iColorSrc16S, scale32S, iColorDst16S);
        t.end();
        printf("t = %f\n", t.elapse());

        cv::Mat back;
        header = iColorDst16S.toOpenCVMat();
        header.convertTo(back, CV_8U);
        cv::imshow("scale color", back);
        cv::waitKey(0);
    }
    */
    {
        cv::Mat oldColorDst = colorDst;
        cv::resize(oldColorDst, colorDst, cv::Size(128, 64));
        cv::Size sz = colorDst.size();
        iocl::UMat colorSrc8U(sz, CV_8UC4);
        iocl::UMat colorSrc16S(sz, CV_16SC4);
        iocl::UMat colorSrc32S(sz, CV_32SC4);
        iocl::UMat colorDst8U, colorDst16S, colorDst32S;

        //colorDst.setTo(cv::Scalar::all(255));

        header = colorSrc8U.toOpenCVMat();
        colorDst.copyTo(header);
        cv::imshow("src", header);
        pyramidUp8UC4To8UC4(colorSrc8U, colorDst8U);
        header = colorDst8U.toOpenCVMat();
        cv::imshow("up 8U", header);
        cv::waitKey(0);

        header = colorSrc16S.toOpenCVMat();
        colorDst.convertTo(header, CV_16S);
        pyramidUp16SC4To16SC4(colorSrc16S, colorDst16S);
        header = colorDst16S.toOpenCVMat();
        header.convertTo(cvtBack, CV_8U);
        cv::imshow("up 16S", cvtBack);
        cv::waitKey(0);

        header = colorSrc16S.toOpenCVMat();
        colorDst.convertTo(header, CV_16S);
        pyramidUp16SC4To16SC4(colorSrc16S, colorDst16S);
        header = colorDst16S.toOpenCVMat();
        header.convertTo(cvtBack, CV_8U);
        cv::imshow("up 32S", cvtBack);
        cv::waitKey(0);
    }

    return 0;
}

#include "ZBlendAlgo.h"
int main()
{
    bool ok = ioclInit();
    if (!ok)
    {
        printf("OpenCL init failed\n");
        return 0;
    }

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

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\panoimage\\color\\1.bmp");
    //contentPaths.push_back("F:\\panoimage\\color\\2.bmp");
    //contentPaths.push_back("F:\\panoimage\\color\\3.bmp");
    //contentPaths.push_back("F:\\panoimage\\color\\4.bmp");
    //contentPaths.push_back("F:\\panoimage\\color\\5.bmp");
    //contentPaths.push_back("F:\\panoimage\\color\\6.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\color\\mask_1.bmp");
    //maskPaths.push_back("F:\\panoimage\\color\\mask_2.bmp");
    //maskPaths.push_back("F:\\panoimage\\color\\mask_3.bmp");
    //maskPaths.push_back("F:\\panoimage\\color\\mask_4.bmp");
    //maskPaths.push_back("F:\\panoimage\\color\\mask_5.bmp");
    //maskPaths.push_back("F:\\panoimage\\color\\mask_6.bmp");

    ztool::Timer timer;
    timer.start();

    int numImages = contentPaths.size();
    std::vector<cv::Mat> images, masks;
    cv::Size imageSize;
    getImagesAndMasks(contentPaths, maskPaths, imageSize, images, masks);

    cv::Mat temp8U, temp16S;
    std::vector<iocl::UMat> srcImages(numImages);
    for (int i = 0; i < numImages; i++)
    {
        cv::cvtColor(images[i], temp8U, CV_BGR2BGRA);
        temp8U.convertTo(temp16S, CV_16S);
        srcImages[i].upload(temp16S);
    }

    ztool::Timer t;

    IOclTilingMultibandBlendFast blender;
    blender.prepare(masks, 10, 4);
    iocl::UMat blendImage;

    t.start();
    for (int i = 0; i < 1; i++)
    blender.blend(srcImages, blendImage);
    t.end();
    printf("t = %f\n", t.elapse());

    cv::Mat header = blendImage.toOpenCVMat();
    cv::imshow("blend image", header);
    cv::waitKey(0);

    TilingMultibandBlendFast cpuBlender;
    cpuBlender.prepare(masks, 10, 4);
    cv::Mat cpuBlendImage;
    t.start();
    for (int i = 0; i < 10; i++)
    cpuBlender.blend(images, cpuBlendImage);
    t.end();
    printf("t = %f\n", t.elapse());

    return 0;
}