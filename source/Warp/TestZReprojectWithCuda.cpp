#include "ZReproject.h"
#include "CudaAccel/CudaInterface.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

void compare(const cv::Mat& mat32F, const cv::Mat& mat64F)
{
    CV_Assert(mat32F.size() == mat64F.size() &&
        mat32F.type() == CV_32FC1 && mat64F.type() == CV_64FC1);

    int rows = mat32F.rows, cols = mat64F.cols;
    cv::Mat diff = cv::Mat::zeros(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++)
    {
        const float* ptr32F = mat32F.ptr<float>(i);
        const double* ptr64F = mat64F.ptr<double>(i);
        unsigned char* ptrDiff = diff.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (abs(ptr32F[j] - ptr64F[j]) > 0.001)
                ptrDiff[j] = 255;
        }
    }
    cv::imshow("diff", diff);
    cv::waitKey(0);
}

int main()
{
    int height = 800;
    cv::Size dstSize = cv::Size(height * 2, height);

    //{
    //    cv::Mat ss(2000, 4000, CV_64FC4);
    //    cv::cuda::GpuMat dd(ss);
    //    //dd.upload(ss);
    //    printf("OK\n");

    //    std::vector<std::string> paths;
    //    paths.push_back("F:\\panoimage\\2\\1\\1.jpg");
    //    paths.push_back("F:\\panoimage\\2\\1\\2.jpg");
    //    paths.push_back("F:\\panoimage\\2\\1\\3.jpg");
    //    paths.push_back("F:\\panoimage\\2\\1\\4.jpg");
    //    paths.push_back("F:\\panoimage\\2\\1\\5.jpg");
    //    paths.push_back("F:\\panoimage\\2\\1\\6.jpg");

    //    int numImages = paths.size();
    //    std::vector<cv::Mat> src(numImages);
    //    for (int i = 0; i < numImages; i++)
    //        src[i] = cv::imread(paths[i]);

    //    std::vector<PhotoParam> params;
    //    loadPhotoParamFromXML("F:\\panoimage\\2\\1\\distort.xml", params);

    //    std::vector<cv::Mat> maps, masks;
    //    getReprojectMapsAndMasks(params, src[0].size(), dstSize, maps, masks);

    //    std::vector<cv::cuda::GpuMat> xmaps, ymaps;
    //    cudaGenerateReprojectMaps(params, src[0].size(), dstSize, xmaps, ymaps);

    //    cv::cuda::GpuMat gmat;
    //    gmat.create(cv::Size(2048, 1024), CV_8UC4);

    //    cv::Mat splitMats[2];
    //    cv::Mat fromGpuMats[2];
    //    for (int i = 0; i < numImages; i++)
    //    {
    //        cv::split(maps[i], splitMats);
    //        xmaps[i].download(fromGpuMats[0]);
    //        ymaps[i].download(fromGpuMats[1]);
    //        compare(fromGpuMats[0], splitMats[0]);
    //        compare(fromGpuMats[1], splitMats[1]);
    //    }
    //}
    //printf("finish\n");

    {
        std::vector<std::string> paths;
        std::vector<PhotoParam> params;

        //paths.push_back("F:\\panoimage\\detuoffice2\\input-01.jpg");
        //paths.push_back("F:\\panoimage\\detuoffice2\\input-00.jpg");
        //paths.push_back("F:\\panoimage\\detuoffice2\\input-03.jpg");
        //paths.push_back("F:\\panoimage\\detuoffice2\\input-02.jpg");
        //loadPhotoParamFromPTS("F:\\panoimage\\detuoffice2\\4port.pts", params);
        //loadPhotoParamFromXML("F:\\panoimage\\detuoffice2\\detu.xml", params);
        //rotateCameras(params, 0, 0, 3.1415926 / 2);

        paths.push_back("F:\\panoimage\\919-4\\snapshot0.bmp");
        paths.push_back("F:\\panoimage\\919-4\\snapshot1.bmp");
        paths.push_back("F:\\panoimage\\919-4\\snapshot2.bmp");
        paths.push_back("F:\\panoimage\\919-4\\snapshot3.bmp");        
        loadPhotoParamFromXML("F:\\panoimage\\919-4\\vrdl4.xml", params);

        //paths.push_back("F:\\panovideo\\ricoh m15\\image0.bmp");
        //loadPhotoParamFromXML("F:\\panovideo\\ricoh m15\\parambestcircle.xml", params);

        int numImages = paths.size();
        std::vector<cv::Mat> src(numImages);
        cv::Mat temp;
        for (int i = 0; i < numImages; i++)
        {
            temp = cv::imread(paths[i]);
            cv::cvtColor(temp, src[i], CV_BGR2BGRA);
        }

        //numImages = 2;
        //src.push_back(src[0]);

        std::vector<cv::Mat> maps, masks;
        getReprojectMapsAndMasks(params, src[0].size(), dstSize, maps, masks);
        for (int i = 0; i < numImages; i++)
        {
            cv::imshow("mask", masks[i]);
            cv::waitKey(0);
        }

        std::vector<cv::cuda::GpuMat> xmaps, ymaps;
        cudaGenerateReprojectMaps(params, src[0].size(), dstSize, xmaps, ymaps);

        cv::cuda::GpuMat orig;
        cv::cuda::GpuMat rprj;
        cv::Mat rprj16S, rprj8U;
        //rprj.create(dstSize, CV_16SC4);
        //rprj.download(rprj16S);
        for (int i = 0; i < numImages; i++)
        {
            orig.upload(src[i]);
            //rprj.create(dstSize, CV_16SC4);
            cudaReprojectTo16S(orig, rprj, xmaps[i], ymaps[i]);
            rprj.download(rprj16S);
            rprj16S.convertTo(rprj8U, CV_8U);
            cv::imshow("dst", rprj8U);
            cv::waitKey(0);
        }

        cv::Mat splitMats[2];
        cv::Mat fromGpuMats[2];
        for (int i = 0; i < numImages; i++)
        {
            cv::split(maps[i], splitMats);
            xmaps[i].download(fromGpuMats[0]);
            ymaps[i].download(fromGpuMats[1]);
            compare(fromGpuMats[0], splitMats[0]);
            compare(fromGpuMats[1], splitMats[1]);
        }
    }

    return 0;
}