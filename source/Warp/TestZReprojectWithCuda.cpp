#include "ZReproject.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui.hpp"
#include "cuda_runtime.h"

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

    //std::vector<PhotoParam> params;
    //loadPhotoParamFromXML("F:\\QQRecord\\452103256\\FileRecv\\bbb.vrdl", params);

    //return 0;

    cudaSetDevice(0);
    cudaFree(0);
    printf("setup finish\n");

    cv::Mat tempCpu(8, 8, CV_8UC1);
    cv::cuda::GpuMat tempGpu(tempCpu);
    printf("start\n");

    cv::Size dstSize = cv::Size(1024, 512);

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
        paths.push_back("F:\\panoimage\\detuoffice2\\input-01.jpg");
        paths.push_back("F:\\panoimage\\detuoffice2\\input-00.jpg");
        paths.push_back("F:\\panoimage\\detuoffice2\\input-03.jpg");
        paths.push_back("F:\\panoimage\\detuoffice2\\input-02.jpg");

        int numImages = paths.size();
        std::vector<cv::Mat> src(numImages);
        for (int i = 0; i < numImages; i++)
            src[i] = cv::imread(paths[i]);

        std::vector<PhotoParam> params;
        loadPhotoParamFromPTS("F:\\panoimage\\detuoffice2\\4port.pts", params);
        //loadPhotoParamFromXML("F:\\panoimage\\detuoffice2\\detu.xml", params);
        //rotateCameras(params, 0, 0, 3.1415926 / 2);

        std::vector<cv::Mat> maps, masks;
        getReprojectMapsAndMasks(params, src[0].size(), dstSize, maps, masks);

        std::vector<cv::cuda::GpuMat> xmaps, ymaps;
        cudaGenerateReprojectMaps(params, src[0].size(), dstSize, xmaps, ymaps);

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