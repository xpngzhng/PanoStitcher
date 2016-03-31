#include "ZBlend.h"
#include "ZBlendAlgo.h"
#include "Timer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#define NEED_MAIN 0

void accumulate8UC4To32SC4(const cv::gpu::GpuMat& src, const cv::gpu::GpuMat& weight, cv::gpu::GpuMat& dst);
void normalize32SC4Feather(cv::gpu::GpuMat& image);
void calcWeightsFeatherBlend(const std::vector<cv::Mat>& dists, std::vector<cv::Mat>& weights);
void distanceTransformFeatherBlend(const cv::Mat& mask, cv::Mat& dist);

//static void distanceTransform(const cv::Mat& mask, cv::Mat& dist)
//{
//    CV_Assert(mask.data && mask.type() == CV_8UC1);
//
//    int width = mask.cols, height = mask.rows;
//    cv::Mat largeMask = cv::Mat::zeros(height, width * 2, CV_8UC1), largeDist;
//    cv::Mat largeMaskROI(largeMask, cv::Rect(width / 2, 0, width, height));
//    mask.copyTo(largeMaskROI);
//    horiCircularRepeat(largeMask, -width / 2, width);
//    cv::distanceTransform(largeMask, largeDist, CV_DIST_L1, 3);
//    largeDist(cv::Rect(width / 2, 0, width, height)).copyTo(dist);
//}
//
//static const int UNIT_SHIFT = 16;
//static const int UNIT = 1 << UNIT_SHIFT;
//static const float eps = 1.0F / UNIT;

class CudaTilingFeatherBlend
{
public:
    CudaTilingFeatherBlend() : numImages(0), rows(0), cols(0), success(false) {}
    bool prepare(const std::vector<cv::Mat>& masks);
    void blend(const std::vector<cv::gpu::GpuMat>& images, cv::gpu::GpuMat& blendImage);
private:
    std::vector<cv::gpu::GpuMat> weights;
    cv::gpu::GpuMat accumImage;
    int numImages;
    int rows, cols;
    bool success;
};

bool CudaTilingFeatherBlend::prepare(const std::vector<cv::Mat>& masks)
{
    success = false;
    if (masks.empty())
        return false;

    int currNumMasks = masks.size();
    int currRows = masks[0].rows, currCols = masks[0].cols;
    for (int i = 0; i < currNumMasks; i++)
    {
        if (!masks[i].data || masks[i].type() != CV_8UC1 ||
            masks[i].rows != currRows || masks[i].cols != currCols)
            return false;
    }
    numImages = currNumMasks;
    rows = currRows;
    cols = currCols;

    ztool::Timer timer;
    std::vector<cv::Mat> dists(numImages);
    for (int i = 0; i < numImages; i++)
        distanceTransformFeatherBlend(masks[i], dists[i]);
    timer.end();
    //printf("dist trans time = %f\n", timer.elapse());

    std::vector<cv::Mat> weightsCpu(numImages);
    calcWeightsFeatherBlend(dists, weightsCpu);
    //for (int i = 0; i < numImages; i++)
    //    weightsCpu[i].create(rows, cols, CV_32SC1);

    //std::vector<float*> ptrDistVector(numImages);
    //float** ptrDist = &ptrDistVector[0];
    //std::vector<int*> ptrWeightVector(numImages);
    //int** ptrWeight = &ptrWeightVector[0];
    //for (int i = 0; i < rows; i++)
    //{
    //    for (int k = 0; k < numImages; k++)
    //    {
    //        ptrDist[k] = dists[k].ptr<float>(i);
    //        ptrWeight[k] = weightsCpu[k].ptr<int>(i);
    //    }
    //    for (int j = 0; j < cols; j++)
    //    {
    //        float sum = 0;
    //        for (int k = 0; k < numImages; k++)
    //            sum += ptrDist[k][j];
    //        sum = fabs(sum) <= FLT_MIN ? 0 : 1.0F / sum;
    //        int intSum = 0;
    //        for (int k = 0; k < numImages; k++)
    //        {
    //            // WE MUST ENSURE THAT intSum >= UNIT, 
    //            // SO WE ADD 1 AFTER SCALING.
    //            int weight = ptrDist[k][j] * sum * UNIT + 1;
    //            intSum += weight;
    //            ptrWeight[k][j] = weight;
    //        }
    //    }
    //}

    weights.resize(numImages);
    for (int i = 0; i < numImages; i++)
        weights[i].upload(weightsCpu[i]);

    success = true;
    return true;
}

void CudaTilingFeatherBlend::blend(const std::vector<cv::gpu::GpuMat>& images, cv::gpu::GpuMat& blendImage)
{
    if (!success)
        return;

    CV_Assert(images.size() == numImages);

    accumImage.create(rows, cols, CV_32SC4);
    accumImage.setTo(0);
    for (int i = 0; i < numImages; i++)
        accumulate8UC4To32SC4(images[i], weights[i], accumImage);
    normalize32SC4Feather(accumImage);
    accumImage.convertTo(blendImage, CV_8U);
}

#if NEED_MAIN

int main()
{
    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\panoimage\\changtai\\1.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\2.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\3.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\4.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\5.bmp");
    //contentPaths.push_back("F:\\panoimage\\changtai\\6.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_1.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_2.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_3.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_4.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_5.bmp");
    //maskPaths.push_back("F:\\panoimage\\changtai\\mask_6.bmp");

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\pano0000.tif");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\pano0001.tif");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\pano0002.tif");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\panomask0000.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\panomask0001.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\panomask0002.bmp");

    //std::vector<std::string> contentPaths;
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\0.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\1.bmp");
    //contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\2.bmp");
    //std::vector<std::string> maskPaths;
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\0mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\1mask.bmp");
    //maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\img\\2mask.bmp");

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

    std::vector<std::string> contentPaths;
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\0.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\1.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\2.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\3.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\4.bmp");
    contentPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\5.bmp");
    std::vector<std::string> maskPaths;
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\0mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\1mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\2mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\3mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\4mask.bmp");
    maskPaths.push_back("F:\\QQRecord\\452103256\\FileRecv\\images1\\5mask.bmp");

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

    ztool::Timer timer;

    int numImages = contentPaths.size();
    std::vector<cv::Mat> images, masks; 
    cv::Size imageSize;
    getImagesAndMasks(contentPaths, maskPaths, imageSize, images, masks);

    std::vector<cv::Mat> results;
    compensate(images, masks, results);

    cv::Mat result;
    TilingFeatherBlend blender;
    timer.start();
    blender.prepare(masks);
    timer.end();
    printf("prepare time = %f\n", timer.elapse());
    for (int i = 0; i < 50; i++)
    {
    timer.start();
    blender.blend(results, result);
    timer.end();
    printf("blend time = %f\n", timer.elapse());
    }
    //cv::imshow("compensate image result", result);
    //cv::waitKey(0);

    std::vector<cv::gpu::GpuMat> imagesGpu(numImages);
    cv::gpu::GpuMat resultGpu;
    int fromTo[] = {0, 0, 1, 1, 2, 2};
    cv::Mat imageC4(images[0].size(), CV_8UC4);
    for (int i = 0; i < numImages; i++)
    {
        cv::mixChannels(&results[i], 1, &imageC4, 1, fromTo, 3);
        imagesGpu[i].upload(imageC4);
    }
    imageC4.release();

    CudaTilingFeatherBlend cudaBlender;
    cudaBlender.prepare(masks);
    for (int i = 0 ; i < 50; i++)
    {
    timer.start();
    cudaBlender.blend(imagesGpu, resultGpu);
    timer.end();
    printf("gpu blend time = %f\n", timer.elapse());
    }
    resultGpu.download(result);
    cv::imshow("gpu result", result);
    cv::waitKey(0);
    
    return 0;
}

#endif

#undef NEED_MAIN