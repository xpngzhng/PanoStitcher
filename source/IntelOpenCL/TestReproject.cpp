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

int main(int argc, char** argv)
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