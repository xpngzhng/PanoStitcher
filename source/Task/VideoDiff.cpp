#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

void imageDiff(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst)
{
    CV_Assert(src1.size() == src2.size() && src1.type() == CV_8UC1 && src2.type() == CV_8UC1);
    dst.create(src1.size(), CV_8UC1);
    int rows = src1.rows, cols = src2.cols;
    int minVal = 256, maxVal = 0;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrSrc1 = src1.ptr<unsigned char>(i);
        const unsigned char* ptrSrc2 = src2.ptr<unsigned char>(i);
        unsigned char* ptrDst = dst.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            int val = abs((int)ptrSrc1[j] - (int)ptrSrc2[j]);
            ptrDst[j] = val;
            minVal = std::min(val, minVal);
            maxVal = std::max(val, maxVal);
        }
    }
    double k = 255.0 / (maxVal - minVal);
    for (int i = 0; i < rows; i++)
    {
        unsigned char* ptrDst = dst.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            ptrDst[j] = (ptrDst[j] - minVal) * k;
        }
    }
}

int main()
{
    cv::VideoCapture cap1, cap2;
    cap1.open("F:\\panovideo\\test\\outdoor\\from.image.kava.mp4");
    cap2.open("F:\\panovideo\\test\\outdoor\\panopts.mp4");
    cv::Mat frame1, frame2;
    cv::Mat image1, image2;
    cv::Mat diffImage;
    while (cap1.read(frame1) && cap2.read(frame2))
    {
        cv::cvtColor(frame1, image1, CV_BGR2GRAY);
        cv::cvtColor(frame2, image2, CV_BGR2GRAY);
        imageDiff(image1, image2, diffImage);
        cv::imshow("image1", image1);
        cv::imshow("image2", image2);
        cv::imshow("diff", diffImage);
        cv::waitKey(0);
    }
    return 0;
}