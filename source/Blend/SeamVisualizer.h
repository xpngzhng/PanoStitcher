#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

struct SeamVisualizer
{
    SeamVisualizer(const cv::Mat& mask1, const cv::Mat& mask2, const cv::Mat& diff_)
    {
        CV_Assert(mask1.data && mask1.type() == CV_8UC1 &&
            mask2.data && mask2.type() == CV_8UC1&&
            diff_.data && diff_.type() == CV_32FC1);
        size = diff_.size();
        CV_Assert(mask1.size() == size && mask2.size() == size);
        
        cv::Mat diff32FC1 = diff_.clone();
        cv::Mat intersect = mask1 & mask2;
        cv::bitwise_not(intersect, intersect);
        diff32FC1.setTo(0, intersect);

        cv::Mat diff8UC1;
        double minVal, maxVal;
        cv::minMaxLoc(diff32FC1, &minVal, &maxVal);
        diff32FC1 -= minVal;
        diff32FC1 *= 1.0 / (maxVal - minVal);
        diff32FC1.convertTo(diff8UC1, CV_8UC1, 255);

        diff.create(size, CV_8UC3);
        int fromTo[] = {0, 0, 0, 1, 0, 2};
        cv::mixChannels(&diff8UC1, 1, &diff, 1, fromTo, 3);
    }

    void drawSeam(const cv::Mat& mask1, const cv::Mat& mask2)
    {
        CV_Assert(mask1.data && mask1.type() == CV_8UC1 &&
            mask2.data && mask2.type() == CV_8UC1 &&
            mask1.size() == size && mask2.size() == size);

        cv::Vec3b color(0, 255, 255);
        for (int i = 1; i < size.height - 1; i++)
        {
            //for (int j = 1; j < size.width - 1; j++)
            //{
            //    if (!mask1.at<unsigned char>(i, j) || mask2.at<unsigned char>(i, j))
            //        continue;
            //    if (!mask1.at<unsigned char>(i, j + 1) && mask2.at<unsigned char>(i, j + 1))
            //    {
            //        diff.at<cv::Vec3b>(i, j) = color;
            //        //diff.at<cv::Vec3b>(i, j + 1) = color;
            //    }
            //    if (!mask1.at<unsigned char>(i, j - 1) && mask2.at<unsigned char>(i, j - 1))
            //    {
            //        diff.at<cv::Vec3b>(i, j) = color;
            //        //diff.at<cv::Vec3b>(i, j - 1) = color;
            //    }
            //    if (!mask1.at<unsigned char>(i - 1, j - 1) && mask2.at<unsigned char>(i - 1, j - 1))
            //    {
            //        diff.at<cv::Vec3b>(i, j) = color;
            //        //diff.at<cv::Vec3b>(i - 1, j - 1) = color;
            //    }
            //    if (!mask1.at<unsigned char>(i - 1, j) && mask2.at<unsigned char>(i - 1, j))
            //    {
            //        diff.at<cv::Vec3b>(i, j) = color;
            //        //diff.at<cv::Vec3b>(i - 1, j) = color;
            //    }
            //    if (!mask1.at<unsigned char>(i - 1, j + 1) && mask2.at<unsigned char>(i - 1, j + 1))
            //    {
            //        diff.at<cv::Vec3b>(i, j) = color;
            //        //diff.at<cv::Vec3b>(i - 1, j + 1) = color;
            //    }
            //    if (!mask1.at<unsigned char>(i + 1, j - 1) && mask2.at<unsigned char>(i + 1, j - 1))
            //    {
            //        diff.at<cv::Vec3b>(i, j) = color;
            //        //diff.at<cv::Vec3b>(i + 1, j - 1) = color;
            //    }
            //    if (!mask1.at<unsigned char>(i + 1, j) && mask2.at<unsigned char>(i + 1, j))
            //    {
            //        diff.at<cv::Vec3b>(i, j) = color;
            //        //diff.at<cv::Vec3b>(i + 1, j) = color;
            //    }
            //    if (!mask1.at<unsigned char>(i + 1, j + 1) && mask2.at<unsigned char>(i + 1, j + 1))
            //    {
            //        diff.at<cv::Vec3b>(i, j) = color;
            //        //diff.at<cv::Vec3b>(i + 1, j + 1) = color;
            //    }
            //}
            const unsigned char* ptr1Rows[] = {mask1.ptr<unsigned char>(i - 1) + 1, 
                                               mask1.ptr<unsigned char>(i) + 1, 
                                               mask1.ptr<unsigned char>(i + 1) + 1};
            const unsigned char* ptr2Rows[] = {mask2.ptr<unsigned char>(i - 1) + 1, 
                                               mask2.ptr<unsigned char>(i) + 1,
                                               mask2.ptr<unsigned char>(i + 1) + 1};
            const unsigned char** ptr1 = ptr1Rows + 1;
            const unsigned char** ptr2 = ptr2Rows + 1;
            cv::Vec3b* ptrDiffRows[] = {diff.ptr<cv::Vec3b>(i - 1) + 1,
                                        diff.ptr<cv::Vec3b>(i) + 1,
                                        diff.ptr<cv::Vec3b>(i + 1) + 1};
            cv::Vec3b** ptrDiff = ptrDiffRows + 1;
            for (int j = 1; j < size.width - 1; j++)
            {
                if (ptr1[0][0] && !ptr2[0][0])
                {
                    if (!ptr1[0][1] && ptr2[0][1])
                    {
                        ptrDiff[0][0] = color;
                        //ptrDiff[0][1] = color;
                    }
                    if (!ptr1[0][-1] && ptr2[0][-1])
                    {
                        ptrDiff[0][0] = color;
                        //ptrDiff[0][-1] = color;
                    }
                    if (!ptr1[-1][-1] && ptr2[-1][-1])
                    {
                        ptrDiff[0][0] = color;
                        //ptrDiff[-1][-1] = color;
                    }
                    if (!ptr1[-1][0] && ptr2[-1][0])
                    {
                        ptrDiff[0][0] = color;
                        //ptrDiff[-1][0] = color;
                    }
                    if (!ptr1[-1][1] && ptr2[-1][1])
                    {
                        ptrDiff[0][0] = color;
                        //ptrDiff[-1][1] = color;
                    }
                    if (!ptr1[1][-1] && ptr2[1][-1])
                    {
                        ptrDiff[0][0] = color;
                        //ptrDiff[1][-1] = color;
                    }
                    if (!ptr1[1][0] && ptr2[1][0])
                    {
                        ptrDiff[0][0] = color;
                        //ptrDiff[1][0] = color;
                    }
                    if (!ptr1[1][1] && ptr2[1][1])
                    {
                        ptrDiff[0][0] = color;
                        //ptrDiff[1][1] = color;
                    }
                }

                ptr1Rows[0]++;
                ptr1Rows[1]++;
                ptr1Rows[2]++;
                ptr2Rows[0]++;
                ptr2Rows[1]++;
                ptr2Rows[2]++;
                ptrDiffRows[0]++;
                ptrDiffRows[1]++;
                ptrDiffRows[2]++;
            }
        }
    }

    void show(const std::string& name)
    {
        cv::imshow(name, diff);
    }

    cv::Mat diff;
    cv::Size size;
};