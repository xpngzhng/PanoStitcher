#include "ZBlendAlgo.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

void getImagesAndMasks(const std::vector<std::string>& imagePaths, 
    const std::vector<std::string>& maskPaths, cv::Size& imageSize, 
    std::vector<cv::Mat>& images, std::vector<cv::Mat>& masks)
{
    int numImages = imagePaths.size();
    if (numImages == 0 || numImages != maskPaths.size())
        return;
    
    int contentFromTo[] = {0, 0, 1, 1, 2, 2};
    int maskFromTo[] = {0, 0};
    cv::Mat origImage, origMask;
    images.resize(numImages);
    masks.resize(numImages);

    origImage = cv::imread(imagePaths[0], -1);
    origMask = cv::imread(maskPaths[0], -1);
    if (!origImage.data || !origMask.data)
        return;
    imageSize = origImage.size();
    if (origMask.size() != imageSize)
        return;
    images[0].create(origImage.size(), CV_8UC3);
    masks[0].create(origMask.size(), CV_8UC1);
    cv::mixChannels(&origImage, 1, &images[0], 1, contentFromTo, 3);
    cv::mixChannels(&origMask, 1, &masks[0], 1, maskFromTo, 1);
    for (int i = 1; i < numImages; i++)
    {
        origImage = cv::imread(imagePaths[i], -1);
        origMask = cv::imread(maskPaths[i], -1);
        if (!origImage.data || !origMask.data || 
            origImage.size() != imageSize ||
            origMask.size() != imageSize)
            return;

        images[i].create(origImage.size(), CV_8UC3);
        cv::mixChannels(&origImage, 1, &images[i], 1, contentFromTo, 3);
        masks[i].create(origMask.size(), CV_8UC1);
        cv::mixChannels(&origMask, 1, &masks[i], 1, maskFromTo, 1);
    }
}

void getParts(const std::vector<cv::Mat>& images, 
    const std::vector<cv::Mat>& masks, cv::Size& imageSize, 
    std::vector<cv::Mat>& imageParts, std::vector<cv::Mat>& maskParts,
    std::vector<cv::Rect>& rects)
{
    imageParts.clear();
    maskParts.clear();
    rects.clear();

    if (images.empty() || masks.empty())
        return;
    CV_Assert(images.size() == masks.size());

    int numImages = images.size();
    imageSize = images[0].size();
    for (int i = 0; i < numImages; i++)
    {
        if (images[i].size() != imageSize ||
            masks[i].size() != imageSize)
            return;
    }

    int pad = 2;
    cv::Point ofs(pad, pad);

    for (int i = 0; i < numImages; i++)
    {
        std::vector<std::vector<cv::Point> > contours;
        cv::Mat mask = cv::Mat::zeros(masks[i].rows + 2 * pad, masks[i].cols + 2 * pad, CV_8UC1);
        cv::Mat maskROI(mask, cv::Rect(pad, pad, masks[i].cols, masks[i].rows));
        masks[i].copyTo(maskROI);
        cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        int numContours = contours.size();
        if (numContours == 1)
        {
            cv::Rect currRect(cv::boundingRect(contours[0]) - ofs);
            rects.push_back(currRect);
            imageParts.push_back(images[i](currRect).clone());
            maskParts.push_back(masks[i](currRect).clone());
        }
        else
        {
            cv::Rect srcLeftRect, srcRightRect;
            bool srcLeftRectSet = false, srcRightRectSet = false;
            for (int j = 0; j < numContours; j++)
            {
                cv::Rect currRect = cv::boundingRect(contours[j]) - ofs;
                if (currRect.x == 0)
                {
                    if (srcLeftRectSet)
                        srcLeftRect |= currRect;
                    else
                    {
                        srcLeftRectSet = true;
                        srcLeftRect = currRect;
                    }
                }
                else
                {
                    if (srcRightRectSet)
                        srcRightRect |= currRect;
                    else
                    {
                        srcRightRectSet = true;
                        srcRightRect = currRect;
                    }
                }
            }
            cv::Rect srcLeftRectShift = srcLeftRect + cv::Point(imageSize.width, 0);
            cv::Rect srcMergeRect = srcRightRect | srcLeftRectShift;

            cv::Rect dstLeftRect = srcRightRect - srcMergeRect.tl();
            cv::Rect dstRightRect = srcLeftRectShift - srcMergeRect.tl();
            
            cv::Mat image = cv::Mat::zeros(srcMergeRect.size(), CV_8UC3); 
            cv::Mat imageROI;
            imageROI = image(dstRightRect);
            images[i](srcLeftRect).copyTo(imageROI);
            imageROI = image(dstLeftRect);
            images[i](srcRightRect).copyTo(imageROI);

            cv::Mat mask = cv::Mat::zeros(srcMergeRect.size(), CV_8UC1);
            cv::Mat maskROI;
            maskROI = mask(dstRightRect);
            masks[i](srcLeftRect).copyTo(maskROI);
            maskROI = mask(dstLeftRect);
            masks[i](srcRightRect).copyTo(maskROI);

            rects.push_back(srcMergeRect);
            imageParts.push_back(image);
            maskParts.push_back(mask);
        }
    }
}

void getParts(const std::vector<std::string>& contentPaths, 
    const std::vector<std::string>& maskPaths, cv::Size& imageSize, 
    std::vector<cv::Mat>& imageParts, std::vector<cv::Mat>& maskParts,
    std::vector<cv::Rect>& rects)
{
    imageSize = cv::Size(0, 0);
    imageParts.clear();
    maskParts.clear();
    rects.clear();

    int numImages = contentPaths.size();
    if (numImages == 0 || numImages != maskPaths.size())
        return;
    
    int contentFromTo[] = {0, 0, 1, 1, 2, 2};
    int maskFromTo[] = {0, 0};
    cv::Mat origImage, origMask;
    std::vector<cv::Mat> images(numImages), masks(numImages);

    origImage = cv::imread(contentPaths[0], -1);
    origMask = cv::imread(maskPaths[0], -1);
    if (!origImage.data || !origMask.data)
        return;
    imageSize = origImage.size();
    if (origMask.size() != imageSize)
        return;
    images[0].create(origImage.size(), CV_8UC3);
    masks[0].create(origMask.size(), CV_8UC1);
    cv::mixChannels(&origImage, 1, &images[0], 1, contentFromTo, 3);
    cv::mixChannels(&origMask, 1, &masks[0], 1, maskFromTo, 1);
    for (int i = 1; i < numImages; i++)
    {
        origImage = cv::imread(contentPaths[i], -1);
        origMask = cv::imread(maskPaths[i], -1);
        if (!origImage.data || !origMask.data || 
            origImage.size() != imageSize ||
            origMask.size() != imageSize)
            return;

        images[i].create(origImage.size(), CV_8UC3);
        cv::mixChannels(&origImage, 1, &images[i], 1, contentFromTo, 3);
        masks[i].create(origMask.size(), CV_8UC1);
        cv::mixChannels(&origMask, 1, &masks[i], 1, maskFromTo, 1);
    }

    getParts(images, masks, imageSize, imageParts, maskParts, rects);
}

bool checkType(const std::vector<cv::Mat>& images, int type)
{
    if (images.empty())
        return false;

    int num = images.size();
    for (int i = 0; i < num; i++)
    {
        if (images[i].type() != type)
            return false;
    }
    return true;
}

bool checkSize(const std::vector<cv::Mat>& images)
{
    if (images.empty())
        return false;

    int num = images.size();
    cv::Size size = images[0].size();
    for (int i = 1; i < num; i++)
    {
        if (images[i].size() != size)
            return false;
    }
    return true;
}

bool checkSize(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks)
{
    if (images.empty() || masks.empty())
        return false;
    if (images.size() != masks.size())
        return false;

    int num = images.size();
    cv::Size size = images[0].size();
    for (int i = 0; i < num; i++)
    {
        if (images[i].size() != size ||
            masks[i].size() != size)
            return false;
    }

    return true;
}

bool checkSize(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    const std::vector<cv::Rect>& rects, const cv::Size& blendSize)
{
    if (images.empty() || masks.empty() || rects.empty() || 
        blendSize.width <= 0 || blendSize.height <= 0)
        return false;
    if (images.size() != masks.size() && images.size() != rects.size())
        return false;

    int num = images.size();
    for (int i = 0; i < num; i++)
    {
        if (images[i].size() != rects[i].size() ||
            masks[i].size() != rects[i].size() ||
            rects[i].y < 0 || rects[i].y + rects[i].height > blendSize.height ||
            rects[i].width > blendSize.width)
            return false;
    }

    return true;
}

bool probeMasks(const std::vector<cv::Mat>& masks, int& numMasks, int& rows, int& cols)
{
    if (masks.empty()) return false;

    int currNumMasks = masks.size();
    int currRows = masks[0].rows, currCols = masks[0].cols;
    for (int i = 0; i < currNumMasks; i++)
    {
        if (!masks[i].data || masks[i].type() != CV_8UC1 ||
            masks[i].rows != currRows || masks[i].cols != currCols)
            return false;
    }

    rows = currRows;
    cols = currCols;
    numMasks = currNumMasks;

    return true;
}