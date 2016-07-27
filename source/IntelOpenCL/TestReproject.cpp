#include "ZReproject.h"
#include "IntelOpenCLInterface.h"
#include "RunTimeObjects.h"
#include "../../source/Blend/Timer.h"
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

    std::vector<IOclMat> srcMats(numImages);
    std::vector<IOclMat> xmapMats(numImages), ymapMats(numImages), weightMats(numImages);
    for (int i = 0; i < numImages; i++)
    {
        srcMats[i].create(src[i].rows, src[i].cols, CV_8UC4, oclobjects.context);
        xmapMats[i].create(dstSize.height, dstSize.width, CV_32FC1, oclobjects.context);
        ymapMats[i].create(dstSize.height, dstSize.width, CV_32FC1, oclobjects.context);
        weightMats[i].create(dstSize.height, dstSize.width, CV_32FC1, oclobjects.context);

        cv::Mat srcMatWrapper(srcMats[i].rows, srcMats[i].cols, srcMats[i].type, srcMats[i].data, srcMats[i].step);
        cv::Mat xmapMatWrapper(xmapMats[i].rows, xmapMats[i].cols, xmapMats[i].type, xmapMats[i].data, xmapMats[i].step);
        cv::Mat ymapMatWrapper(ymapMats[i].rows, ymapMats[i].cols, ymapMats[i].type, ymapMats[i].data, ymapMats[i].step);
        cv::cvtColor(src[i], srcMatWrapper, CV_BGR2BGRA);
        xmaps32F[i].copyTo(xmapMatWrapper);
        ymaps32F[i].copyTo(ymapMatWrapper);

        cv::Mat weightWrapper(weightMats[i].rows, weightMats[i].cols, weightMats[i].type, weightMats[i].data, weightMats[i].step);
        weights[i].copyTo(weightWrapper);
    }

    IOclMat dstMat32F;
    dstMat32F.create(dstSize.height, dstSize.width, CV_32FC4, oclobjects.context);

    IOclMat dstMat;
    dstMat.create(dstSize.height, dstSize.width, CV_8UC4, oclobjects.context);

    IOclMat dstMat16S;
    dstMat16S.create(dstSize.height, dstSize.width, CV_16SC4, oclobjects.context);
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
            ioclSetZero(dstMat32F);
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

#include "../Blend/Pyramid.h"
int main()
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

    cv::Mat colorDst, colorDst32S, grayDst, grayDst32S;
    t.start();
    for (int i = 0; i < 100; i++)
    pyramidDown(colorSrc, colorDst, cv::Size(), cv::BORDER_WRAP, cv::BORDER_REFLECT_101);
    t.end();
    printf("t = %f\n", t.elapse());
    pyramidDownTo32S(colorSrc, colorDst32S, cv::Size(), cv::BORDER_WRAP, cv::BORDER_REFLECT_101);
    pyramidDown(graySrc, grayDst, cv::Size(), cv::BORDER_WRAP, cv::BORDER_REFLECT_101);
    pyramidDownTo32S(graySrc, grayDst32S, cv::Size(), cv::BORDER_WRAP, cv::BORDER_REFLECT_101);

    IOclMat iColorSrc32F(colorSrc.size(), CV_32FC4, iocl::ocl->context);
    IOclMat iGraySrc32F(graySrc.size(), CV_32FC1, iocl::ocl->context);
    IOclMat iColorDst32F, iGrayDst32F;

    cv::Mat header;

    cv::Mat diffFor32F, cvtFor32F;

    header = iColorSrc32F.toOpenCVMat();
    colorSrc.convertTo(header, CV_32F);
    t.start();
    for (int i = 0; i < 100; i++)
    ioclPyramidDown32FC4(iColorSrc32F, iColorDst32F, cv::Size());
    t.end();
    printf("t = %f\n", t.elapse());
    header = iColorDst32F.toOpenCVMat();
    header.convertTo(cvtFor32F, CV_8U);
    compare<unsigned char, CV_8U, 4>(colorDst, cvtFor32F, diffFor32F);
    cv::imshow("cpu color", colorDst);
    cv::imshow("intel gpu color", cvtFor32F);
    cv::imshow("diff color", diffFor32F);
    cv::waitKey(0);

    header = iGraySrc32F.toOpenCVMat();
    graySrc.convertTo(header, CV_32F);
    ioclPyramidDown32FC1(iGraySrc32F, iGrayDst32F, cv::Size());
    header = iGrayDst32F.toOpenCVMat();
    header.convertTo(cvtFor32F, CV_8U);
    compare<unsigned char, CV_8U, 1>(grayDst, cvtFor32F, diffFor32F);
    cv::imshow("cpu gray", grayDst);
    cv::imshow("intel gpu gray", cvtFor32F);
    cv::imshow("diff gray", diffFor32F);
    cv::waitKey(0);

    cv::Mat diffFor16S, cvtFor16S;
    IOclMat iGraySrc16S(graySrc.size(), CV_16SC1, iocl::ocl->context);
    header = iGraySrc16S.toOpenCVMat();
    graySrc.convertTo(header, CV_16S);
    IOclMat iGrayDst16S;
    pyramidDown16SC1To16SC1(iGraySrc16S, iGrayDst16S);
    header = iGrayDst16S.toOpenCVMat();
    header.convertTo(cvtFor16S, CV_8U);
    compare<unsigned char, CV_8U, 1>(grayDst, cvtFor16S, diffFor16S);
    cv::imshow("diff gray 16S", diffFor16S);
    cv::waitKey(0);

    IOclMat iGrayDst32S;
    pyramidDown16SC1To32SC1(iGraySrc16S, iGrayDst32S);
    header = iGrayDst32S.toOpenCVMat();
    cv::Mat diffFor32S;
    compare<int, CV_32S, 1>(grayDst32S, header, diffFor32S);
    cv::imshow("diff gray 32S", diffFor32S);
    cv::waitKey(0);

    IOclMat iColorSrc(colorSrc.size(), CV_8UC4, iocl::ocl->context);
    IOclMat iGraySrc(graySrc.size(), CV_8UC1, iocl::ocl->context);
    IOclMat iColorDst, iColorDst32S, iGrayDst;

    header = iColorSrc.toOpenCVMat();
    colorSrc.copyTo(header);

    //cv::imshow("color", header);
    t.start();
    for (int i = 0; i < 100; i++)
    ioclPyramidDown8UC4To8UC4(iColorSrc, iColorDst, cv::Size());
    t.end();
    printf("t = %f\n", t.elapse());
    header = iColorDst.toOpenCVMat();

    cv::Mat diffColor;
    compare<unsigned char, CV_8U, 4>(colorDst, header, diffColor);

    cv::imshow("cpu color", colorDst);
    cv::imshow("intel gpu color", header);
    cv::imshow("diff color", diffColor);
    cv::waitKey(0);

    ioclPyramidDown8UC4To32SC4(iColorSrc, iColorDst32S, cv::Size());
    header = iColorDst32S.toOpenCVMat();
    cv::Mat diffColor32S;
    compare<int, CV_32S, 4>(colorDst32S, header, diffColor32S);
    cv::imshow("diff 32s", diffColor32S);
    cv::waitKey(0);

    header = iGraySrc.toOpenCVMat();
    graySrc.copyTo(header);
    ioclPyramidDown8UC1To8UC1(iGraySrc, iGrayDst, cv::Size());
    header = iGrayDst.toOpenCVMat();

    cv::Mat diffGray;
    compare<unsigned char, CV_8U, 1>(grayDst, header, diffGray);

    cv::imshow("cpu gray", grayDst);
    cv::imshow("intel gpu gray", header);
    cv::imshow("diff gray", diffGray);
    cv::waitKey(0);

    return 0;
}