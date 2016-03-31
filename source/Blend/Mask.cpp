#include "ZBlendAlgo.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

static cv::Range getNonZeroBoundingRange(const unsigned char* arr, int length)
{
    int start = length, end = -1;
    for (int i = 0; i < length; i++)
    {
        if (arr[i])
        {
            start = i;
            break;
        }
    }
    for (int i = length - 1; i >= 0; i--)
    {
        if (arr[i])
        {
            end = i + 1;
            break;
        }
    }
    return cv::Range(start, end);
}

cv::Rect getNonZeroBoundingRect(const cv::Mat& mask)
{
    CV_Assert(mask.data && mask.type() == CV_8UC1);
    int rows = mask.rows, cols = mask.cols;
    int left = cols, right = -1, top = rows, bottom = -1;
    int nextRow = rows + 1;
    cv::Range badRowRange(cols, -1);
    for (int i = 0; i < rows; i++)
    {
        cv::Range range = getNonZeroBoundingRange(mask.ptr<unsigned char>(i), cols);
        if (range != badRowRange)
        {
            left = range.start;
            right = range.end;
            top = i;
            bottom = i + 1;
            nextRow = i + 1;
            break;
        }
    }
    for (int i = nextRow; i < rows; i++)
    {
        cv::Range range = getNonZeroBoundingRange(mask.ptr<unsigned char>(i), cols);
        if (range != badRowRange)
        {
            left = std::min(left, range.start);
            right = std::max(right, range.end);
            bottom = i + 1;
        }
    }

    return cv::Rect(left, top, right - left, bottom - top);
}

void getIntersect(const cv::Mat& mask1, const cv::Mat& mask2, 
    cv::Mat& intersectMask, cv::Rect& intersectRect)
{
    CV_Assert(mask1.data && mask2.data && 
        mask1.type() == CV_8UC1 && mask2.type() == CV_8UC1 &&
        mask1.size() == mask2.size());

    intersectMask.release();
    intersectRect = cv::Rect();

    cv::Mat intersect = mask1 & mask2;
    cv::Rect roi = getNonZeroBoundingRect(intersect);
    if (roi.width < 0 || roi.height < 0)
        return;

    intersect(roi).copyTo(intersectMask);
    intersectRect = roi;
}

void separateMask(cv::Mat& mask1, cv::Mat& mask2, const cv::Mat& intersect)
{
    CV_Assert(mask1.data && mask2.data && intersect.data &&
        mask1.type() == CV_8UC1 && mask2.type() == CV_8UC1 && intersect.type() == CV_8UC1 &&
        mask1.size() == mask2.size() && mask1.size() == intersect.size());

    cv::Mat dist1, dist2;
    cv::distanceTransform(mask1, dist1, CV_DIST_L1, 3);
    cv::distanceTransform(mask2, dist2, CV_DIST_L1, 3);
    cv::Mat canvas = dist1 < dist2;
    canvas &= intersect;
    mask1.setTo(0, canvas);
    mask2.setTo(0, mask1);
}

void splitRegion(const cv::Mat& mask1, const cv::Mat& mask2, cv::Mat& region1, cv::Mat& region2)
{
    CV_Assert(mask1.data && mask2.data && 
        mask1.type() == CV_8UC1 && mask2.type() == CV_8UC1 &&
        mask1.size() == mask2.size());

    cv::Mat dist1, dist2;
    cv::Mat canvas;

    canvas = ~mask1;
    cv::distanceTransform(canvas, dist1, CV_DIST_L1, 3);
    canvas = ~mask2;
    cv::distanceTransform(canvas, dist2, CV_DIST_L1, 3);

    cv::Mat intersect = mask1 & mask2;
    region1 = dist1 < dist2;
    region2 = ~region1;
    region2 -= intersect;

    cv::distanceTransform(mask1, dist1, CV_DIST_L1, 3);
    cv::distanceTransform(mask2, dist2, CV_DIST_L1, 3);
    canvas = dist1 > dist2;
    canvas &= intersect;
    canvas.copyTo(region1, canvas);
    intersect -= canvas;
    intersect.copyTo(region2, intersect);

    //cv::Mat intersect = mask1 & mask2;
    //region1 = dist1 < dist2;
    //cv::distanceTransform(mask1, dist1, CV_DIST_L1, 3);
    //cv::distanceTransform(mask2, dist2, CV_DIST_L1, 3);
    //canvas = dist1 > dist2;
    //canvas &= intersect;
    //canvas.copyTo(region1, canvas);
    //region2 = ~region1;
}

void splitRegion(const cv::Mat& mask1, const cv::Mat& mask2, const cv::Mat& intersect, cv::Mat& region1, cv::Mat& region2)
{
    CV_Assert(mask1.data && mask2.data && intersect.data &&
        mask1.type() == CV_8UC1 && mask2.type() == CV_8UC1 && intersect.type() == CV_8UC1 &&
        mask1.size() == mask2.size() && mask1.size() == intersect.size());

    cv::Mat dist1, dist2;
    cv::Mat canvas;

    canvas = ~mask1;
    cv::distanceTransform(canvas, dist1, CV_DIST_L1, 3);
    canvas = ~mask2;
    cv::distanceTransform(canvas, dist2, CV_DIST_L1, 3);

    region1 = dist1 < dist2;
    region2 = ~region1;
    region2 -= intersect;

    cv::distanceTransform(mask1, dist1, CV_DIST_L1, 3);
    cv::distanceTransform(mask2, dist2, CV_DIST_L1, 3);
    canvas = dist1 > dist2;
    canvas &= intersect;
    canvas.copyTo(region1, canvas);
    intersect -= canvas;
    intersect.copyTo(region2, intersect);
}

void fillExclusiveRegion(const cv::Mat& mask1, const cv::Mat& mask2, 
    const cv::Mat& intersect, cv::Mat& fill1, cv::Mat& fill2)
{
    CV_Assert(mask1.data && mask2.data && intersect.data &&
        mask1.type() == CV_8UC1 && mask2.type() == CV_8UC1 && intersect.type() == CV_8UC1);
    CV_Assert(fill1.data && fill2.data &&
        fill1.type() == CV_8UC1 && fill2.type() == CV_8UC1);
    cv::Size size = intersect.size();
    CV_Assert(mask1.size() == size && mask2.size() == size &&
        fill1.size() == size && fill2.size() == size);

    cv::Mat dist1, dist2;
    cv::Mat canvas;

    canvas = ~mask1;
    cv::distanceTransform(canvas, dist1, CV_DIST_L1, 3);
    canvas = ~mask2;
    cv::distanceTransform(canvas, dist2, CV_DIST_L1, 3);

    canvas = dist1 < dist2;
    fill1 |= canvas;
    canvas = ~canvas;
    canvas -= intersect;
    fill2 |= canvas;
    //fill2 = ~fill1;
}

void getZeroSegments(const char* arr, int length, std::vector<std::pair<int, int> >& segments)
{
    CV_Assert(arr && length > 0);
    segments.clear();
    int begInc = length, endExc;
    if (!arr[0])
        begInc = 0;
    for (int i = 0; i < length; i++)
    {
        if (!arr[i] && arr[i - 1])
            begInc = i;
        if (arr[i] && !arr[i - 1])
        {
            endExc = i;
            segments.push_back(std::make_pair(begInc, endExc));
        }
    }
    if (!arr[length - 1])
        segments.push_back(std::make_pair(begInc, length));
}

bool horiSplit(const cv::Mat& mask, int* zeroBegInc, int* zeroEndExc)
{
    CV_Assert(mask.data && mask.type() == CV_8UC1);
    int rows = mask.rows, cols = mask.cols;
    std::vector<char> nonZero(cols, 0);
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        char* ptrNonZero = &nonZero[0];
        for (int j = 0; j < cols; j++)
        {
            *ptrNonZero = *ptrMask ? 1 : *ptrNonZero;
            ptrNonZero++;
            ptrMask++;
        }
    }
    std::vector<std::pair<int, int> > zeroSegments;
    getZeroSegments(&nonZero[0], cols, zeroSegments);
    if (zeroSegments.empty())
    {
        *zeroBegInc = cols;
        *zeroEndExc = 0;
        return false;
    }
    int numSegments = zeroSegments.size();
    int index = 0, maxLength = zeroSegments[0].second - zeroSegments[0].first;
    for (int i = 1; i < numSegments; i++)
    {
        int length = zeroSegments[i].second - zeroSegments[i].first;
        if (length > maxLength)
        {
            maxLength = length;
            index = i;
        }
    }
    *zeroBegInc = zeroSegments[index].first;
    *zeroEndExc = zeroSegments[index].second;
    return true;
}

static void simplePaste(const cv::Mat& image, const cv::Mat& mask, cv::Mat& blendImage, cv::Mat& blendMask)
{
    CV_Assert(image.data &&
        mask.data && mask.type() == CV_8UC1 &&
        blendImage.data && blendImage.type() == image.type() &&
        blendMask.data && blendMask.type() == CV_8UC1);

    cv::Size imageSize = image.size();
    CV_Assert(mask.size() == imageSize &&
        blendImage.size() == imageSize &&
        blendMask.size() == imageSize);

    cv::Rect imageRect(0, 0, imageSize.width, imageSize.height);
    // If blendMask is all zero, blendRect is invalid
    cv::Rect blendRect = getNonZeroBoundingRect(blendMask);

    cv::Mat intersectMask = mask & blendMask;
    cv::Rect intersectRect = getNonZeroBoundingRect(intersectMask);
    int intersectMaskNonZero = cv::countNonZero(intersectMask);

    if (intersectMaskNonZero == 0)
    {
#if WRITE_CONSOLE
        printf("curr mask does not intersect blend mask, copy curr image and mask\n");
#endif
        cv::Rect currNonZeroRect = getNonZeroBoundingRect(mask);
        cv::Mat currNonZeroMask(mask, currNonZeroRect);
        cv::Mat blendImageROI(blendImage, currNonZeroRect);
        image(currNonZeroRect).copyTo(blendImageROI, currNonZeroMask);
        cv::Mat blendMaskROI(blendMask, currNonZeroRect);
        currNonZeroMask.copyTo(blendMaskROI, currNonZeroMask);
        return;
    }

    int blendMaskNonZero = cv::countNonZero(blendMask);
    int currMaskNonZero = cv::countNonZero(mask);
    if (intersectMaskNonZero == currMaskNonZero)
    {
#if WRITE_CONSOLE
        printf("curr mask totally inside blend mask, return\n");
#endif
        return;
    }
    if (intersectMaskNonZero == blendMaskNonZero)
    {
        if (currMaskNonZero == blendMaskNonZero)
        {
#if WRITE_CONSOLE
            printf("blend mask equals curr mask, return");
#endif
        }
        else
        {
#if WRITE_CONSOLE
            printf("blend mask totally inside curr mask, copy curr image and mask, then return");
#endif
            cv::Rect currNonZeroRect = getNonZeroBoundingRect(mask);
            cv::Mat currNonZeroMask(mask, currNonZeroRect);
            cv::Mat blendImageROI(blendImage, currNonZeroRect);
            image(currNonZeroRect).copyTo(blendImageROI, currNonZeroMask);
            cv::Mat blendMaskROI(blendMask, currNonZeroRect);
            currNonZeroMask.copyTo(blendMaskROI, currNonZeroMask);
        }
        return;
    }

    cv::Mat blendRegionWork;
    cv::Mat currRegionWork;
#if WRITE_CONSOLE
    timer.start();
#endif
    splitRegion(blendMask, mask, intersectMask, blendRegionWork, currRegionWork);
    //cv::imshow("blend region", blendRegionWork);
    //cv::imshow("curr  region", currRegionWork);
#if WRITE_CONSOLE
    timer.end();
    printf("findSeam time elapse: %f\n", timer.elapse());
#endif
    image.copyTo(blendImage, currRegionWork);
    blendMask |= mask;
    //cv::waitKey(0);
}

void getUniqueMasks(const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& uniqueMasks)
{
    int currNumMasks = masks.size();
    CV_Assert(currNumMasks <= 255);

    int currRows = masks[0].rows, currCols = masks[0].cols;
    for (int i = 0; i < currNumMasks; i++)
    {
        if (!masks[i].data || masks[i].type() != CV_8UC1 ||
            masks[i].rows != currRows || masks[i].cols != currCols)
            CV_Assert(0);
    }
    int rows = currRows;
    int cols = currCols;
    int numImages = currNumMasks;

    cv::Mat indexImage = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat indexMask = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat currIndex(rows, cols, CV_8UC1);
    for (int i = 0; i < numImages; i++)
    {
        currIndex.setTo(0);
        currIndex.setTo(i + 1, masks[i]);
        simplePaste(currIndex, masks[i], indexImage, indexMask);
    }

    uniqueMasks.resize(numImages);
    cv::Mat belong;
    for (int i = 0; i < numImages; i++)
    {
        uniqueMasks[i].create(rows, cols, CV_8UC1);
        uniqueMasks[i].setTo(0);
        belong = indexImage == (i + 1);
        uniqueMasks[i].setTo(255, belong);
    }
    indexImage.release();
    indexMask.release();
    currIndex.release();
    belong.release();
}

void getNonIntersectingMasks(const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& notIntMasks)
{
    CV_Assert(checkType(masks, CV_8UC1) && checkSize(masks));

    int numImages = masks.size();
    notIntMasks.resize(numImages);
    for (int i = 0; i < numImages; i++)
        masks[i].copyTo(notIntMasks[i]);
    cv::Mat dist, currDist;
    cv::distanceTransform(notIntMasks[0], dist, CV_DIST_L1, 3);
    cv::Mat mask = notIntMasks[0].clone();
    for (int i = 1; i < numImages; i++)
    {
        cv::distanceTransform(mask, dist, CV_DIST_L1, 3);
        cv::distanceTransform(notIntMasks[i], currDist, CV_DIST_L1, 3);
        notIntMasks[i] = currDist > dist;
        //cv::imshow("curr mask", notIntMasks[i]);
        //cv::waitKey(0);
        for (int j = 0; j < i; j++)
            notIntMasks[j].setTo(0, notIntMasks[i]);
        mask |= notIntMasks[i];
    }
}