#include "CudaInterface.h"
#include "Tool/Timer.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#define NEED_MAIN 0

void accumulate8UC4To32SC4(const cv::cuda::GpuMat& src, const cv::cuda::GpuMat& weight, cv::cuda::GpuMat& dst);
void normalize32SC4Feather(cv::cuda::GpuMat& image);
void getWeightsLinearBlend(const std::vector<cv::Mat>& masks, int radius, std::vector<cv::Mat>& weights);

bool CudaTilingLinearBlend::prepare(const std::vector<cv::Mat>& masks, int radius)
{
    success = false;
    if (masks.empty())
        return false;
    if (radius < 1)
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

    //ztool::Timer timer;
    //std::vector<cv::Mat> dists(numImages);
    //for (int i = 0; i < numImages; i++)
    //    distanceTransformFeatherBlend(masks[i], dists[i]);
    //timer.end();
    ////printf("dist trans time = %f\n", timer.elapse());

    //std::vector<cv::Mat> weightsCpu(numImages);
    //calcWeightsFeatherBlend(dists, weightsCpu);
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

    std::vector<cv::Mat> weightsCpu;
    getWeightsLinearBlend(masks, radius, weightsCpu);

    weights.resize(numImages);
    for (int i = 0; i < numImages; i++)
        weights[i].upload(weightsCpu[i]);

    success = true;
    return true;
}

void CudaTilingLinearBlend::blend(const std::vector<cv::cuda::GpuMat>& images, cv::cuda::GpuMat& blendImage)
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

#undef NEED_MAIN
