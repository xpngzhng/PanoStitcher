#include "opencv2/core.hpp"

void calcHist(const cv::Mat& image, std::vector<int>& hist)
{
    CV_Assert(image.data && image.type() == CV_8UC1);

    hist.resize(256, 0);
    int rows = image.rows, cols = image.cols;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptr = image.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
            hist[ptr[j]] += 1;
    }
}

void calcHist(const cv::Mat& image, const cv::Mat& mask, std::vector<int>& hist)
{
    CV_Assert(image.data && image.type() == CV_8UC1 &&
        mask.data && mask.type() == CV_8UC1 && image.size() == mask.size());

    hist.resize(256, 0);
    int rows = image.rows, cols = image.cols;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptr = image.ptr<unsigned char>(i);
        const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrMask[j])
                hist[ptr[j]] += 1;
        }
    }
}

int countNonZeroHistBins(const std::vector<int>& hist)
{
    CV_Assert(hist.size() == 256);
    int ret = 0;
    for (int i = 0; i < 256; i++)
    {
        if (hist[i] > 0)
            ret++;
    }
    return ret;
}