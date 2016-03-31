#include "ConnectedComponents.h"
#include "Pyramid.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

void main()
{
    cv::Mat mask = cv::imread("mask.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat labels, stats;
    int numComps = connectedComponentsWithStats(mask, labels, stats);
    cv::Mat newMask = cv::Mat::zeros(mask.size(), mask.type());
    for (int i = 0; i < numComps; i++)
    {
        const int* ptrStat = stats.ptr<int>(i);
        printf("comp[%d]: rect = (%d, %d, %d, %d)\n",
            i, ptrStat[CC_STAT_LEFT], ptrStat[CC_STAT_TOP], 
            ptrStat[CC_STAT_WIDTH], ptrStat[CC_STAT_HEIGHT]);
        if (i != 0)
        {
            newMask.setTo(255, labels == i);
            cv::rectangle(newMask, cv::Rect(ptrStat[CC_STAT_LEFT], ptrStat[CC_STAT_TOP], 
                ptrStat[CC_STAT_WIDTH], ptrStat[CC_STAT_HEIGHT]), cv::Scalar(255));
        }
        cv::imshow("new mask", newMask);
        cv::waitKey(0);
    }
}