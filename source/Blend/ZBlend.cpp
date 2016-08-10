#include "ZBlendAlgo.h"
#include "ZBlend.h"
#include "Tool/Timer.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

static const int pad = 8;
static const bool fastBlend = false;
static const int maxLevels = 16;
static const int minLength = 2;

#define WRITE_CONSOLE 0

void blendParts(const std::vector<cv::Mat>& imageParts, const std::vector<cv::Mat>& maskParts, 
    const std::vector<cv::Rect>& rects, const cv::Size& imageSize,
    cv::Mat& blendImage, cv::Mat& blendMask)
{
    if (!checkSize(imageParts, maskParts, rects, imageSize))
        return;

    ztool::Timer timer;

    int numImages = imageParts.size();
    blendImage.create(imageSize, CV_8UC3);
    blendImage.setTo(0);
    blendMask.create(imageSize, CV_8UC1);
    blendMask.setTo(0);
    cv::Rect imageRect(0, 0, imageSize.width, imageSize.height);
    cv::Rect blendRect;
    if (contains(imageRect, rects[0]))
    {
        cv::Mat blendImageROI(blendImage, rects[0]);
        imageParts[0].copyTo(blendImageROI);
        cv::Mat blendMaskROI(blendMask, rects[0]);
        maskParts[0].copyTo(blendMaskROI);
        blendRect = rects[0];
    }
    else
    {
        horiCircularFold(imageParts[0], rects[0], blendImage);
        horiCircularFold(maskParts[0], rects[0], blendMask);
        std::vector<cv::Rect> foldRects;
        horiCircularFold(rects[0], imageSize.width, foldRects);
        blendRect = foldRects[0];
        blendRect |= foldRects[1];
    }
    for (int imageIndex = 1; imageIndex < numImages; imageIndex++)
    {
        printf("i = %d\n", imageIndex);
        bool imageRectContainsCurrRect = contains(imageRect, rects[imageIndex]);
        if (imageRectContainsCurrRect)
            printf("curr part inside canvas\n");
        else
            printf("curr part not totally inside canvas\n");

        cv::Mat blendMaskPart;
        if (imageRectContainsCurrRect)
            blendMaskPart = blendMask(rects[imageIndex]).clone();
        else
            horiCircularExpand(blendMask, rects[imageIndex], blendMaskPart);

        cv::Mat intersectMask;
        cv::Rect intersectRectRelative;
        getIntersect(blendMaskPart, maskParts[imageIndex], intersectMask, intersectRectRelative);

        if (intersectMask.empty())
        {
            printf("curr mask does not intersect blend mask, copy curr image and mask\n");
            if (imageRectContainsCurrRect)
            {
                cv::Mat blendImageROI(blendImage, rects[imageIndex]);
                imageParts[imageIndex].copyTo(blendImageROI, maskParts[imageIndex]);
                cv::Mat blendMaskROI(blendMask, rects[imageIndex]);
                maskParts[imageIndex].copyTo(blendMaskROI, maskParts[imageIndex]);
            }
            else
            {
                horiCircularFold(imageParts[imageIndex], maskParts[imageIndex], rects[imageIndex], blendImage);
                horiCircularFold(maskParts[imageIndex], maskParts[imageIndex], rects[imageIndex], blendMask);
            }
            continue;
        }

        int blendMaskNonZero = cv::countNonZero(blendMask);
        int currMaskNonZero = cv::countNonZero(maskParts[imageIndex]);
        int intersectMaskNonZero = cv::countNonZero(intersectMask);
        if (intersectMaskNonZero == currMaskNonZero)
        {
            printf("curr mask totally inside blend mask, continue\n");
            continue;
        }
        if (intersectMaskNonZero == blendMaskNonZero)
        {
            if (currMaskNonZero == blendMaskNonZero)
                printf("blend mask equals curr mask, continue");
            else
            {
                printf("blend mask totally inside curr mask, copy curr image and mask");
                if (imageRectContainsCurrRect)
                {
                    cv::Mat blendImageROI(blendImage, rects[imageIndex]);
                    imageParts[imageIndex].copyTo(blendImageROI, maskParts[imageIndex]);
                    cv::Mat blendMaskROI(blendMask, rects[imageIndex]);
                    maskParts[imageIndex].copyTo(blendMaskROI, maskParts[imageIndex]);
                }
                else
                {
                    horiCircularFold(imageParts[imageIndex], maskParts[imageIndex], rects[imageIndex], blendImage);
                    horiCircularFold(maskParts[imageIndex], maskParts[imageIndex], rects[imageIndex], blendMask);
                }
            }
            continue;
        }
        
        cv::Mat blendImagePart;
        if (imageRectContainsCurrRect)
            blendImagePart = blendImage(rects[imageIndex]);
        else
            horiCircularExpand(blendImage, rects[imageIndex], blendImagePart);

        //cv::imshow("intersect mask", intersectMask);
        //cv::waitKey(0);

        cv::Rect intersectRect = intersectRectRelative + rects[imageIndex].tl();
        int top = std::max(0, intersectRect.y - pad);
        int bottom = std::min(imageSize.height, intersectRect.y + intersectRect.height + pad);
        cv::Rect extendIntersectRect;
        if (imageRect.width - intersectRect.width > pad * 2)
            extendIntersectRect = cv::Rect(intersectRect.x - pad, top, intersectRect.width + 2 * pad, bottom - top);
        else
            extendIntersectRect = cv::Rect(intersectRect.x, top, imageRect.width, bottom - top);
        //printf("intersect rect = (%d, %d, %d, %d), extend intersect rect = (%d, %d, %d, %d)\n",
        //    intersectRect.x, intersectRect.y, intersectRect.width, intersectRect.height,
        //    extendIntersectRect.x, extendIntersectRect.y, extendIntersectRect.width, extendIntersectRect.height);

        cv::Mat blendImageROIExtend;
        cv::Mat blendMaskROIExtend;
        horiCircularExpand(blendImage, extendIntersectRect, blendImageROIExtend);
        horiCircularExpand(blendMask, extendIntersectRect, blendMaskROIExtend);

        cv::Mat currImageROIExtend = cv::Mat::zeros(extendIntersectRect.size(), CV_8UC3);
        cv::Mat currMaskROIExtend = cv::Mat::zeros(extendIntersectRect.size(), CV_8UC1);
        copyIfIntersect(imageParts[imageIndex], currImageROIExtend, rects[imageIndex], extendIntersectRect);
        copyIfIntersect(maskParts[imageIndex], currMaskROIExtend, rects[imageIndex], extendIntersectRect);
        horiCircularRepeat(currImageROIExtend, extendIntersectRect.x, imageSize.width);
        horiCircularRepeat(currMaskROIExtend, extendIntersectRect.x, imageSize.width);
        //cv::imshow("blend extend mask", blendMaskROIExtend);
        //cv::imshow("curr extend mask", currMaskROIExtend);
        //cv::waitKey(0);

        timer.start();
        //cv::Mat blendRegionROIExtend, currRegionROIExtend;
        //split(blendMaskROIExtend, currMaskROIExtend, blendRegionROIExtend, currRegionROIExtend);
        //findSeamByRegionSplitLine(blendImageROIExtend, currImageROIExtend, 
        //    blendRegionROIExtend, currRegionROIExtend, blendMaskROIExtend, currMaskROIExtend);
        findSeam(blendImageROIExtend, currImageROIExtend, blendMaskROIExtend, currMaskROIExtend, 
            extendIntersectRect.width == imageRect.width);
        timer.end();
        printf("findSeam time elapse: %f\n", timer.elapse());
        //cv::imshow("blend extend mask after", blendMaskROIExtend);
        //cv::imshow("curr extend mask after", currMaskROIExtend);
        //cv::waitKey(0);

        cv::Rect insideExtendRect = intersectRect - extendIntersectRect.tl();
        cv::Mat blendMaskROI(blendMaskROIExtend, insideExtendRect);
        cv::Mat currMaskROI(currMaskROIExtend, insideExtendRect);

        if (fastBlend)
        {
            cv::Mat blendRegionPart, currRegionPart;
            splitRegion(blendMaskPart, maskParts[imageIndex], blendRegionPart, currRegionPart);
            
            cv::Mat blendRegionPartROI(blendRegionPart, intersectRectRelative);
            blendMaskROI.copyTo(blendRegionPartROI, intersectMask);
            cv::Mat currRegionPartROI(currRegionPart, intersectRectRelative);
            currMaskROI.copyTo(currRegionPartROI, intersectMask);  

            //cv::imshow("this blend region after", blendRegionPart);            
            //cv::imshow("this curr region after", currRegionPart);

            cv::Mat result;
            multibandBlend(blendImagePart, imageParts[imageIndex], blendMaskPart, maskParts[imageIndex],
                blendRegionPart, currRegionPart, rects[imageIndex].width == imageRect.width, 
                maxLevels, minLength, result);

            if (imageRectContainsCurrRect)
            {
                //cv::Mat blendImageROI(blendImage, rects[imageIndex]);
                //result.copyTo(blendImageROI);
                //cv::Mat blendMaskROI(blendMask, rects[imageIndex]);
                //maskParts[imageIndex].copyTo(blendMaskROI, maskParts[imageIndex]);
                blendMaskPart |= maskParts[imageIndex];
                cv::Mat blendImageROI(blendImage, rects[imageIndex]);
                result.copyTo(blendImageROI, blendMaskPart);
                cv::Mat blendMaskROI(blendMask, rects[imageIndex]);
                blendMaskPart.copyTo(blendMaskROI);
            }
            else
            {
                blendMaskPart |= maskParts[imageIndex];
                horiCircularFold(result, blendMaskPart, rects[imageIndex], blendImage);
                horiCircularFold(blendMaskPart, rects[imageIndex], blendMask);
            }
        }
        else
        {    
            printf("begin prepare blend\n");
            cv::Rect unionRect = blendRect;
            if (imageRectContainsCurrRect)
                 unionRect |= rects[imageIndex];
            else
            {
                std::vector<cv::Rect> foldRects;
                horiCircularFold(rects[imageIndex], imageSize.width, foldRects);
                unionRect |= foldRects[0];
                unionRect |= foldRects[1];
            }

            cv::Mat blendImageWork(blendImage(unionRect));
            cv::Mat blendMaskWork(blendMask(unionRect));

            cv::Size unionSize = unionRect.size();
            cv::Rect partRectRelative = rects[imageIndex] - unionRect.tl();

            cv::Mat currImageWork = cv::Mat::zeros(unionSize, CV_8UC3);
            cv::Mat currMaskWork = cv::Mat::zeros(unionSize, CV_8UC1);
            if (imageRectContainsCurrRect)
            {
                cv::Mat currImageWorkROI(currImageWork, partRectRelative);
                imageParts[imageIndex].copyTo(currImageWorkROI);
                cv::Mat currMaskWorkROI(currMaskWork, partRectRelative);
                maskParts[imageIndex].copyTo(currMaskWorkROI);
            }
            else
            {
                horiCircularFold(imageParts[imageIndex], partRectRelative, currImageWork);
                horiCircularFold(maskParts[imageIndex], partRectRelative, currMaskWork);
            }            

            cv::Mat blendRegionWork, currRegionWork;
            splitRegion(blendMaskWork, currMaskWork, blendRegionWork, currRegionWork);
                
            cv::Rect intersectRectShift = intersectRect - unionRect.tl();
            if (imageRectContainsCurrRect)
            {
                cv::Mat blendRegionWorkROI(blendRegionWork, intersectRectShift);
                blendMaskROI.copyTo(blendRegionWorkROI, intersectMask);
                cv::Mat currRegionWorkROI(currRegionWork, intersectRectShift);
                currMaskROI.copyTo(currRegionWorkROI, intersectMask);
            }
            else
            {
                horiCircularFold(blendMaskROI, intersectMask, intersectRectShift, blendRegionWork);
                horiCircularFold(currMaskROI, intersectMask, intersectRectShift, currRegionWork);
            }

            //cv::imshow("blend image work", blendImageWork);
            //cv::imshow("blend mask work", blendMaskWork);
            //cv::imshow("blend region work", blendRegionWork);
            //cv::imshow("curr image work", currImageWork);
            //cv::imshow("curr mask work", currMaskWork);
            //cv::imshow("curr region work", currRegionWork);

            timer.start();
            cv::Mat result;
            multibandBlend(blendImageWork, currImageWork, blendMaskWork, currMaskWork,
                blendRegionWork, currRegionWork, unionSize.width == imageSize.width, 
                maxLevels, minLength, result);
            blendMaskWork |= currMaskWork;
            result.copyTo(blendImageWork, blendMaskWork);            
            blendRect = unionRect;
            timer.end();
            printf("blend time elapse: %f\n", timer.elapse());
        }

        //cv::imshow("blend image", blendImage);
        //cv::imshow("blend mask", blendMask);
        //cv::waitKey(0);
    }
}

void blendSameSize(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, 
    cv::Mat& blendImage, cv::Mat& blendMask)
{
    if (!checkSize(images, masks))
        return;   

    cv::Size imageSize = images[0].size();
    blendImage.create(imageSize, CV_8UC3);
    blendImage.setTo(0);
    blendMask.create(imageSize, CV_8UC1);
    blendMask.setTo(0);

    int numImages = images.size();
    for (int i = 0; i < numImages; i++)
    {
        printf("i = %d\n", i);
        blendSameSize(images[i], masks[i], blendImage, blendMask);
    }
}

// This is the first version of blendSameSize, memory consumption is similar to the preceding
// blendParts, but larger than the third blendSameSize version.
// This blendSameSize will not be maintained and may be deleted in future commit.
void blendSameSize1(const cv::Mat& image, const cv::Mat& mask, cv::Mat& blendImage, cv::Mat& blendMask)
{
    CV_Assert(image.data && image.type() == CV_8UC3 &&
        mask.data && mask.type() == CV_8UC1 &&
        blendImage.data && blendImage.type() == CV_8UC3 &&
        blendMask.data && blendMask.type() == CV_8UC1);

    cv::Size imageSize = image.size();
    CV_Assert(mask.size() == imageSize && 
        blendImage.size() == imageSize && 
        blendMask.size() == imageSize);

    ztool::Timer timer;

    cv::Rect imageRect(0, 0, imageSize.width, imageSize.height);
    // If blendMask is all zero, blendRect is invalid
    cv::Rect blendRect = getNonZeroBoundingRect(blendMask);
    
    cv::Mat intersectMask;
    cv::Rect intersectRect;
    getIntersect(blendMask, mask, intersectMask, intersectRect);

    if (intersectMask.empty())
    {
        printf("curr mask does not intersect blend mask, copy curr image and mask\n");
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
    int intersectMaskNonZero = cv::countNonZero(intersectMask);
    if (intersectMaskNonZero == currMaskNonZero)
    {
        printf("curr mask totally inside blend mask, continue\n");
        return;
    }
    if (intersectMaskNonZero == blendMaskNonZero)
    {
        if (currMaskNonZero == blendMaskNonZero)
            printf("blend mask equals curr mask, continue");
        else
        {
            printf("blend mask totally inside curr mask, copy curr image and mask");
            cv::Rect currNonZeroRect = getNonZeroBoundingRect(mask);
            cv::Mat currNonZeroMask(mask, currNonZeroRect);
            cv::Mat blendImageROI(blendImage, currNonZeroRect);
            image(currNonZeroRect).copyTo(blendImageROI, currNonZeroMask);
            cv::Mat blendMaskROI(blendMask, currNonZeroRect);
            currNonZeroMask.copyTo(blendMaskROI, currNonZeroMask);
        }
        return;
    }

    cv::Rect extendIntersectRect(intersectRect.x - pad, intersectRect.y - pad, 
            intersectRect.width + 2 * pad, intersectRect.height + 2 * pad);
    extendIntersectRect &= imageRect;
        
    cv::Mat blendImageROIExtend(blendImage, extendIntersectRect);
    cv::Mat blendMaskROIExtend = blendMask(extendIntersectRect).clone();
    cv::Mat currImageROIExtend(image, extendIntersectRect);
    cv::Mat currMaskROIExtend = mask(extendIntersectRect).clone();

    timer.start();
    //if (intersectMaskNonZero < extendIntersectRect.area() * 0.75)
    //{
    //    printf("intersect mask too smaller than intersect rect, use fast findSeam\n");
    //    cv::Mat extendIntersect;
    //    cv::bitwise_and(blendMask(extendIntersectRect), mask(extendIntersectRect), extendIntersect);
    //    findSeamInROI(blendImageROIExtend, currImageROIExtend, extendIntersect, 
    //        blendMaskROIExtend, currMaskROIExtend, extendIntersectRect.width == imageRect.width);
    //}
    //else
    //{
    //    findSeam(blendImageROIExtend, currImageROIExtend, blendMaskROIExtend, currMaskROIExtend, 
    //        extendIntersectRect.width == imageRect.width);
    //}
    findSeamScaleDown(blendImageROIExtend, currImageROIExtend, blendMaskROIExtend, currMaskROIExtend, 
        extendIntersectRect.width == imageRect.width, pad, true);
    timer.end();
    printf("findSeam time elapse: %f\n", timer.elapse());
    //cv::imshow("blend extend mask after", blendMaskROIExtend);
    //cv::imshow("curr extend mask after", currMaskROIExtend);
    //cv::imwrite("blendmask.bmp", blendMaskROIExtend);
    //cv::imwrite("currmask.bmp", currMaskROIExtend);
    //cv::waitKey(0);

    cv::Rect currNonZeroRect = getNonZeroBoundingRect(mask);
    cv::Rect unionRect = blendRect | currNonZeroRect;

    cv::Mat blendImageWork(blendImage, unionRect);
    cv::Mat blendMaskWork(blendMask, unionRect);
    cv::Mat currImageWork(image, unionRect);
    cv::Mat currMaskWork(mask, unionRect);

    cv::Mat blendRegionWork, currRegionWork;
    splitRegion(blendMaskWork, currMaskWork, blendRegionWork, currRegionWork);

    cv::Rect insideExtendRect = intersectRect - extendIntersectRect.tl();
    cv::Rect intersectRectShift = intersectRect - unionRect.tl();
    cv::Mat blendRegionWorkROI(blendRegionWork, intersectRectShift);
    blendMaskROIExtend(insideExtendRect).copyTo(blendRegionWorkROI, intersectMask);
    cv::Mat currRegionWorkROI(currRegionWork, intersectRectShift);
    currMaskROIExtend(insideExtendRect).copyTo(currRegionWorkROI, intersectMask);

    //cv::imshow("blend image work", blendImageWork);
    //cv::imshow("blend mask work", blendMaskWork);
    //cv::imshow("blend region work", blendRegionWork);
    //cv::imshow("curr image work", currImageWork);
    //cv::imshow("curr mask work", currMaskWork);
    //cv::imshow("curr region work", currRegionWork);

    blendImageROIExtend.release();
    blendMaskROIExtend.release();
    currImageROIExtend.release();
    currMaskROIExtend.release();
    
    printf("begin blend\n");
    timer.start();
    cv::Mat result;
    multibandBlend(blendImageWork, currImageWork, blendMaskWork, currMaskWork,
        blendRegionWork, currRegionWork, unionRect.width == imageRect.width, 
        maxLevels, minLength, result);    
    blendMaskWork |= currMaskWork;
    result.copyTo(blendImageWork, blendMaskWork); 
    timer.end();
    printf("blend time elapse: %f\n", timer.elapse());

    //cv::imshow("blend image", blendImage);
    //cv::imshow("blend mask", blendMask);
    //cv::waitKey(0);
}

// This version of blendSameSize tries to reduce memory consumption when
// the intersect rect spans a large portion of the panorama width 
// but the truely intersect non zero mask area only locates on the very left 
// and very right part of the intersect rect.
// According to the commited log, this implementation do not save too much computation,
// and this version of code may be deleted in future commit.
void blendSameSize2(const cv::Mat& image, const cv::Mat& mask, cv::Mat& blendImage, cv::Mat& blendMask)
{
    CV_Assert(image.data && image.type() == CV_8UC3 &&
        mask.data && mask.type() == CV_8UC1 &&
        blendImage.data && blendImage.type() == CV_8UC3 &&
        blendMask.data && blendMask.type() == CV_8UC1);

    cv::Size imageSize = image.size();
    CV_Assert(mask.size() == imageSize && 
        blendImage.size() == imageSize && 
        blendMask.size() == imageSize);

    ztool::Timer timer;

    cv::Rect imageRect(0, 0, imageSize.width, imageSize.height);
    // If blendMask is all zero, blendRect is invalid
    cv::Rect blendRect = getNonZeroBoundingRect(blendMask);
    
    cv::Mat intersectMask = mask & blendMask;
    cv::Rect intersectRect = getNonZeroBoundingRect(intersectMask);
    int intersectMaskNonZero = cv::countNonZero(intersectMask);

    if (intersectMaskNonZero == 0)
    {
        printf("curr mask does not intersect blend mask, copy curr image and mask\n");
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
        printf("curr mask totally inside blend mask, continue\n");
        return;
    }
    if (intersectMaskNonZero == blendMaskNonZero)
    {
        if (currMaskNonZero == blendMaskNonZero)
            printf("blend mask equals curr mask, continue");
        else
        {
            printf("blend mask totally inside curr mask, copy curr image and mask");
            cv::Rect currNonZeroRect = getNonZeroBoundingRect(mask);
            cv::Mat currNonZeroMask(mask, currNonZeroRect);
            cv::Mat blendImageROI(blendImage, currNonZeroRect);
            image(currNonZeroRect).copyTo(blendImageROI, currNonZeroMask);
            cv::Mat blendMaskROI(blendMask, currNonZeroRect);
            currNonZeroMask.copyTo(blendMaskROI, currNonZeroMask);
        }
        return;
    }

    cv::Rect extendIntersectRect(intersectRect.x - pad, intersectRect.y - pad, 
            intersectRect.width + 2 * pad, intersectRect.height + 2 * pad);
    extendIntersectRect &= imageRect;
        
    cv::Mat blendImageROIExtend(blendImage, extendIntersectRect);
    cv::Mat blendMaskROIExtend = blendMask(extendIntersectRect).clone();
    cv::Mat currImageROIExtend(image, extendIntersectRect);
    cv::Mat currMaskROIExtend = mask(extendIntersectRect).clone();
    cv::Mat intersectMaskROIExtend = intersectMask(extendIntersectRect);

    timer.start();
    if (extendIntersectRect.width == imageRect.width)
    {
        printf("extend intersect mask width equals to image width, ");
        int zeroBegInc, zeroEndExc;
        if (horiSplit(intersectMaskROIExtend, &zeroBegInc, &zeroEndExc))
        {
            if (zeroEndExc - zeroBegInc > 0.25 * imageRect.width &&
                zeroEndExc - zeroBegInc > 2 * pad)
            {
                printf("and sufficient number of columns of extend intersect mask are zero, "
                    "only find seam in the nonzero columns\n");
                findSeamLeftRightWrap(blendImageROIExtend, currImageROIExtend, 
                    blendMaskROIExtend, currMaskROIExtend, zeroBegInc + pad, zeroEndExc - pad);
            }
            else
            {
                printf("but only a few number of columns of extend intersect mask are zero, "
                    "run regular find seam\n");
                findSeam(blendImageROIExtend, currImageROIExtend, blendMaskROIExtend, currMaskROIExtend, true);
            }
        }
        else
        {
            printf("all colums of extend intersect mask are non zero, "
                "run regular find seam\n");
            findSeam(blendImageROIExtend, currImageROIExtend, blendMaskROIExtend, currMaskROIExtend, true);
        }
    }
    else
    {
        printf("extend intersect mask width smaller than image width, "
            "run regular find seam\n");
        findSeam(blendImageROIExtend, currImageROIExtend, blendMaskROIExtend, currMaskROIExtend, false);
    }
    //findSeam(blendImageROIExtend, currImageROIExtend, blendMaskROIExtend, currMaskROIExtend, extendIntersectRect.width == imageRect.width);
    timer.end();
    printf("findSeam time elapse: %f\n", timer.elapse());
    //cv::imshow("blend extend mask after", blendMaskROIExtend);
    //cv::imshow("curr extend mask after", currMaskROIExtend);
    //cv::imwrite("blendmask.bmp", blendMaskROIExtend);
    //cv::imwrite("currmask.bmp", currMaskROIExtend);
    //cv::waitKey(0);

    cv::Rect currNonZeroRect = getNonZeroBoundingRect(mask);
    cv::Rect unionRect = blendRect | currNonZeroRect;

    cv::Mat blendImageWork(blendImage, unionRect);
    cv::Mat blendMaskWork(blendMask, unionRect);
    cv::Mat currImageWork(image, unionRect);
    cv::Mat currMaskWork(mask, unionRect);

    cv::Mat blendRegionWork, currRegionWork;
    splitRegion(blendMaskWork, currMaskWork, blendRegionWork, currRegionWork);

    cv::Rect insideExtendRect = intersectRect - extendIntersectRect.tl();
    cv::Rect intersectRectShift = intersectRect - unionRect.tl();
    cv::Mat blendRegionWorkROI(blendRegionWork, intersectRectShift);
    blendMaskROIExtend(insideExtendRect).copyTo(blendRegionWorkROI, intersectMask(intersectRect));
    cv::Mat currRegionWorkROI(currRegionWork, intersectRectShift);
    currMaskROIExtend(insideExtendRect).copyTo(currRegionWorkROI, intersectMask(intersectRect));

    //cv::imshow("blend image work", blendImageWork);
    //cv::imshow("blend mask work", blendMaskWork);
    //cv::imshow("blend region work", blendRegionWork);
    //cv::imshow("curr image work", currImageWork);
    //cv::imshow("curr mask work", currMaskWork);
    //cv::imshow("curr region work", currRegionWork);

    blendImageROIExtend.release();
    blendMaskROIExtend.release();
    currImageROIExtend.release();
    currMaskROIExtend.release();
    
    printf("begin blend\n");
    timer.start();
    cv::Mat result;
    multibandBlend(blendImageWork, currImageWork, blendMaskWork, currMaskWork,
        blendRegionWork, currRegionWork, unionRect.width == imageRect.width, 
        maxLevels, minLength, result);    
    blendMaskWork |= currMaskWork;
    result.copyTo(blendImageWork, blendMaskWork); 
    timer.end();
    printf("blend time elapse: %f\n", timer.elapse());

    //cv::imshow("blend image", blendImage);
    //cv::imshow("blend mask", blendMask);
    //cv::waitKey(0);
}

// This is the third blendSameSize implementation, it reduces the memory consumption
// compared to the first implementation.
void blendSameSize(const cv::Mat& image, const cv::Mat& mask, cv::Mat& blendImage, cv::Mat& blendMask)
{
    CV_Assert(image.data && image.type() == CV_8UC3 &&
        mask.data && mask.type() == CV_8UC1 &&
        blendImage.data && blendImage.type() == CV_8UC3 &&
        blendMask.data && blendMask.type() == CV_8UC1);

    cv::Size imageSize = image.size();
    CV_Assert(mask.size() == imageSize && 
        blendImage.size() == imageSize && 
        blendMask.size() == imageSize);

    ztool::Timer timer;

    cv::Rect imageRect(0, 0, imageSize.width, imageSize.height);
    // If blendMask is all zero, blendRect is invalid
    cv::Rect blendRect = getNonZeroBoundingRect(blendMask);
    
    cv::Mat intersectMask = mask & blendMask;
    cv::Rect intersectRect = getNonZeroBoundingRect(intersectMask);
    int intersectMaskNonZero = cv::countNonZero(intersectMask);

    if (intersectMaskNonZero == 0)
    {
        printf("curr mask does not intersect blend mask, copy curr image and mask\n");
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
        printf("curr mask totally inside blend mask, continue\n");
        return;
    }
    if (intersectMaskNonZero == blendMaskNonZero)
    {
        if (currMaskNonZero == blendMaskNonZero)
            printf("blend mask equals curr mask, continue");
        else
        {
            printf("blend mask totally inside curr mask, copy curr image and mask");
            cv::Rect currNonZeroRect = getNonZeroBoundingRect(mask);
            cv::Mat currNonZeroMask(mask, currNonZeroRect);
            cv::Mat blendImageROI(blendImage, currNonZeroRect);
            image(currNonZeroRect).copyTo(blendImageROI, currNonZeroMask);
            cv::Mat blendMaskROI(blendMask, currNonZeroRect);
            currNonZeroMask.copyTo(blendMaskROI, currNonZeroMask);
        }
        return;
    }

    cv::Rect currRect = padRect(getNonZeroBoundingRect(mask), pad) & imageRect;
    blendRect = padRect(blendRect, pad) & imageRect;
    cv::Rect unionRect = blendRect | currRect;
    intersectRect = padRect(intersectRect, pad) & imageRect;
    cv::Rect intersectRectShift = intersectRect - unionRect.tl();
    printf("intersect rect = (%d, %d, %d, %d)\n", 
        intersectRect.x, intersectRect.y, intersectRect.width, intersectRect.height);

    cv::Mat blendRegionWork = blendMask(unionRect).clone();
    cv::Mat currRegionWork = mask(unionRect).clone();
    cv::Mat blendMaskROI(blendRegionWork, intersectRectShift);
    cv::Mat currMaskROI(currRegionWork, intersectRectShift);
    cv::Mat blendImageROI(blendImage, intersectRect);
    cv::Mat currImageROI(image, intersectRect);

    timer.start();
    //if (intersectMaskNonZero < intersectRect.area() * 0.75)
    //{
    //    printf("intersect mask too smaller than intersect rect, use fast findSeam\n");
    //    findSeamInROI(blendImageROI, currImageROI, intersectMask(intersectRect), 
    //        blendMaskROI, currMaskROI, intersectRect.width == imageRect.width);
    //}
    //else
    //{
    //    findSeam(blendImageROI, currImageROI, blendMaskROI, currMaskROI, 
    //        intersectRect.width == imageRect.width);
    //}
    findSeamScaleDown(blendImageROI, currImageROI, blendMaskROI, currMaskROI, 
        intersectRect.width == imageRect.width, pad, true);
    //findSeam(blendImageROI, currImageROI, blendMaskROI, currMaskROI, intersectRect.width == imageRect.width);
    timer.end();
    printf("findSeam time elapse: %f\n", timer.elapse());
    //cv::imshow("blend mask after", blendMaskROI);
    //cv::imshow("curr mask after", currMaskROI);
    //cv::imshow("");
    //cv::imwrite("blendmask.bmp", blendMaskROI);
    //cv::imwrite("currmask.bmp", currMaskROI);
    //cv::waitKey(0);   

    cv::Mat blendImageWork(blendImage, unionRect);
    cv::Mat blendMaskWork(blendMask, unionRect);
    cv::Mat currImageWork(image, unionRect);
    cv::Mat currMaskWork(mask, unionRect);

    fillExclusiveRegion(blendMask(unionRect), mask(unionRect), intersectMask(unionRect),
        blendRegionWork, currRegionWork);

    //cv::imshow("blend image work", blendImageWork);
    //cv::imshow("blend mask work", blendMaskWork);
    //cv::imshow("blend region work", blendRegionWork);
    //cv::imshow("curr image work", currImageWork);
    //cv::imshow("curr mask work", currMaskWork);
    //cv::imshow("curr region work", currRegionWork);
   
    printf("begin blend\n");
    timer.start();
    cv::Mat result;
    multibandBlend(blendImageWork, currImageWork, blendMaskWork, currMaskWork,
        blendRegionWork, currRegionWork, unionRect.width == imageRect.width, 
        maxLevels, minLength, result);
    blendMaskWork |= currMaskWork;
    result.copyTo(blendImageWork, blendMaskWork); 
    timer.end();
    printf("blend time elapse: %f\n", timer.elapse());

    //cv::imshow("blend image", blendImage);
    //cv::imshow("blend mask", blendMask);
    //cv::waitKey(0);
}

static BlendConfig validate(const BlendConfig& config)
{
    BlendConfig cfg = config;
#if WRITE_CONSOLE
    printf("check config\n");
    printf("seam mode: ");
#endif
    if (cfg.seamMode == BlendConfig::SEAM_SKIP)
    {
#if WRITE_CONSOLE
        printf("skip, seam finding procedure not called\n");
#endif
    }
    else if (cfg.seamMode == BlendConfig::SEAM_DISTANCE_TRANSFORM)
    {
#if WRITE_CONSOLE
        printf("distance transform\n");
#endif
    }
    else if (cfg.seamMode == BlendConfig::SEAM_GRAPH_CUT)
    {
#if WRITE_CONSOLE
        printf("graph cut\n");
#endif
    }
    else
    {
        cfg.seamMode = BlendConfig::SEAM_GRAPH_CUT;
#if WRITE_CONSOLE
        printf("invalid, use graph cut\n");
#endif
    }
#if WRITE_CONSOLE
    printf("blend mode: ");
#endif
    if (cfg.blendMode == BlendConfig::BLEND_PASTE)
    {
#if WRITE_CONSOLE
        printf("paste\n");
#endif
    }
    else if (cfg.blendMode == BlendConfig::BLEND_LINEAR)
    {
#if WRITE_CONSOLE
        printf("linear\n");
#endif
    }
    else if (cfg.blendMode == BlendConfig::BLEND_MULTIBAND)
    {
#if WRITE_CONSOLE
        printf("multiband\n");
#endif
    }
    else
    {
        cfg.blendMode = BlendConfig::BLEND_MULTIBAND;
#if WRITE_CONSOLE
        printf("invalid, use multiband");
#endif
    }
    if (cfg.blendMode == BlendConfig::BLEND_LINEAR)
    {
#if WRITE_CONSOLE
        printf("radius for linear: ");
#endif
        if (cfg.radiusForLinear < 0)
        {
            cfg.radiusForLinear = 100;
#if WRITE_CONSOLE
            printf("invalid, set to 100\n");
#endif
        }
        else
        {
#if WRITE_CONSOLE
            printf("%d\n", cfg.radiusForLinear);
#endif
        }
    }
    if (cfg.blendMode == BlendConfig::BLEND_MULTIBAND)
    {
#if WRITE_CONSOLE
        printf("max levels for multiband: ");
#endif
        if (cfg.maxLevelsForMultiBand <= 0)
        {
            cfg.maxLevelsForMultiBand = 16;
#if WRITE_CONSOLE
            printf("invalid, set to 16\n");
#endif
        }
        else
        {
#if WRITE_CONSOLE
            printf("%d\n", cfg.maxLevelsForMultiBand);
#endif
        }
#if WRITE_CONSOLE
        printf("min length for multiband: ");
#endif
        if (cfg.minLengthForMultiBand < 2 || cfg.minLengthForMultiBand > 64)
        {
            cfg.minLengthForMultiBand = 2;
#if WRITE_CONSOLE
            printf("invalid, set to 2");
#endif
        }
        else
        {
#if WRITE_CONSOLE
            printf("%d\n", cfg.minLengthForMultiBand);
#endif
        }
    }
    if (cfg.seamMode == BlendConfig::SEAM_GRAPH_CUT)
    {
#if WRITE_CONSOLE
        printf("pad for graphcut: ");
#endif
        if (cfg.padForGraphCut < 0 || cfg.padForGraphCut > 32)
        {
            cfg.padForGraphCut = 8;
#if WRITE_CONSOLE
            printf("invalid, set to 8\n");
#endif
        }
        else
        {
#if WRITE_CONSOLE
            printf("%d\n", cfg.padForGraphCut);
#endif
        }
#if WRITE_CONSOLE        
        printf("scale for graphcut: ");
#endif
        if (cfg.scaleForGraphCut < 0 || cfg.scaleForGraphCut > 32)
        {
            cfg.scaleForGraphCut = 8;
#if WRITE_CONSOLE
            printf("invalid, set to 8\n");
#endif
        }
        else
        {
#if WRITE_CONSOLE            
            printf("%d\n", cfg.scaleForGraphCut);
#endif
        }
        if (cfg.scaleForGraphCut > 1)
        {
#if WRITE_CONSOLE            
            printf("refine for graphcut: %d\n", cfg.refineForGraphCut != 0);
#endif
        }
        if (cfg.scaleForGraphCut == 1)
        {
#if WRITE_CONSOLE
            printf("ratio for graphcut: %f", cfg.ratioForGraphCut);
#endif
            if (cfg.ratioForGraphCut < 0)
            {
#if WRITE_CONSOLE
                printf(", always run findSeam without mask\n");
#endif
            }
            else if (cfg.ratioForGraphCut < 1)
            {
#if WRITE_CONSOLE
                printf(", selectively run findSeam with mask\n");
#endif
            }
            else
            {
#if WRITE_CONSOLE
                printf(", always run findSeam with mask\n");
#endif
            }
        }
    }
#if WRITE_CONSOLE    
    printf("end check config\n");
#endif

    return cfg;
}

// This implementation uses struct BlendConfig to customize blend details,
// and is derived from the third blendSameSize implementation.
void serialBlend(const BlendConfig& config, const cv::Mat& image, const cv::Mat& mask, 
    cv::Mat& blendImage, cv::Mat& blendMask)
{
    CV_Assert(image.data && image.type() == CV_8UC3 &&
        mask.data && mask.type() == CV_8UC1 &&
        blendImage.data && blendImage.type() == CV_8UC3 &&
        blendMask.data && blendMask.type() == CV_8UC1);

    cv::Size imageSize = image.size();
    CV_Assert(mask.size() == imageSize && 
        blendImage.size() == imageSize && 
        blendMask.size() == imageSize);

    BlendConfig cfg = validate(config);

    ztool::Timer timer;

    if (cfg.seamMode == BlendConfig::SEAM_SKIP)
    {
        if (cfg.blendMode == BlendConfig::BLEND_PASTE)
        {
            image.copyTo(blendImage, mask);
            blendMask |= mask;
        }
        else if (cfg.blendMode == BlendConfig::BLEND_LINEAR)
        {
            cv::Mat result;
            linearBlend(blendImage, image, blendMask, mask, blendMask, mask, cfg.radiusForLinear, result);
            blendMask |= mask;
            result.copyTo(blendImage, blendMask);
        }
        else if (cfg.blendMode == BlendConfig::BLEND_MULTIBAND)
        {
            cv::Mat result;
            multibandBlendAnyMask(blendImage, image, blendMask, mask, blendMask, mask, true,
                cfg.maxLevelsForMultiBand, cfg.minLengthForMultiBand, result);
            blendMask |= mask;
            result.copyTo(blendImage, blendMask);
        }
        return;
    }

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

    cv::Rect currRect = padRect(getNonZeroBoundingRect(mask), pad) & imageRect;
    blendRect = padRect(blendRect, pad) & imageRect;
    cv::Rect unionRect = blendRect | currRect;
    intersectRect = padRect(intersectRect, pad) & imageRect;
    cv::Rect intersectRectShift = intersectRect - unionRect.tl();
    //printf("intersect rect = (%d, %d, %d, %d)\n", 
    //    intersectRect.x, intersectRect.y, intersectRect.width, intersectRect.height);

    cv::Mat blendRegionWork = blendMask(unionRect).clone();
    cv::Mat currRegionWork = mask(unionRect).clone();

    if (cfg.seamMode == BlendConfig::SEAM_DISTANCE_TRANSFORM)
    {
#if WRITE_CONSOLE
        timer.start();
#endif
        //separateMask(blendRegionWork, currRegionWork, intersectMask(unionRect));
        splitRegion(blendMask(unionRect), mask(unionRect), intersectMask(unionRect),
            blendRegionWork, currRegionWork);
#if WRITE_CONSOLE
        timer.end();
        printf("findSeam time elapse: %f\n", timer.elapse());
#endif
    }
    else
    {
        cv::Mat blendMaskROI(blendRegionWork, intersectRectShift);
        cv::Mat currMaskROI(currRegionWork, intersectRectShift);
        cv::Mat blendImageROI(blendImage, intersectRect);
        cv::Mat currImageROI(image, intersectRect);

#if WRITE_CONSOLE
        timer.start();
#endif
        if (cfg.scaleForGraphCut == 1)
        {
            if (intersectMaskNonZero < intersectRect.area() * cfg.ratioForGraphCut)
            {
                findSeamInROI(blendImageROI, currImageROI, intersectMask(intersectRect), 
                    blendMaskROI, currMaskROI, intersectRect.width == imageRect.width);
            }
            else
            {
                findSeam(blendImageROI, currImageROI, blendMaskROI, currMaskROI, 
                    intersectRect.width == imageRect.width);
            }
        }
        else
        {
            findSeamScaleDown(blendImageROI, currImageROI, blendMaskROI, currMaskROI, 
                intersectRect.width == imageRect.width, cfg.padForGraphCut, cfg.refineForGraphCut);
        }
#if WRITE_CONSOLE
        timer.end();
        printf("findSeam time elapse: %f\n", timer.elapse());
#endif
        //cv::imshow("blend mask after", blendMaskROI);
        //cv::imshow("curr mask after", currMaskROI);
        //cv::imshow("");
        //cv::imwrite("blendmask.bmp", blendMaskROI);
        //cv::imwrite("currmask.bmp", currMaskROI);
        //cv::waitKey(0);        

        fillExclusiveRegion(blendMask(unionRect), mask(unionRect), intersectMask(unionRect),
            blendRegionWork, currRegionWork);
    }

    cv::Mat blendImageWork(blendImage, unionRect);
    cv::Mat blendMaskWork(blendMask, unionRect);
    cv::Mat currImageWork(image, unionRect);
    cv::Mat currMaskWork(mask, unionRect);

    //cv::imshow("blend image work", blendImageWork);
    //cv::imshow("blend mask work", blendMaskWork);
    //cv::imshow("blend region work", blendRegionWork);
    //cv::imshow("curr image work", currImageWork);
    //cv::imshow("curr mask work", currMaskWork);
    //cv::imshow("curr region work", currRegionWork);
   
#if WRITE_CONSOLE
    printf("begin blend\n");
    timer.start();
#endif
    if (cfg.blendMode == BlendConfig::BLEND_PASTE)
    {
        currImageWork.copyTo(blendImageWork, currRegionWork);
        blendMaskWork |= currMaskWork;
    }
    else
    {
        cv::Mat result;
        if (cfg.blendMode == BlendConfig::BLEND_LINEAR)
        {
            linearBlend(blendImageWork, currImageWork, blendMaskWork, currMaskWork,
                blendRegionWork, currRegionWork, cfg.radiusForLinear, result);
        }
        else
        {
            multibandBlendAnyMask(blendImageWork, currImageWork, blendMaskWork, currMaskWork,
                blendRegionWork, currRegionWork, unionRect.width == imageRect.width,
                cfg.maxLevelsForMultiBand, cfg.minLengthForMultiBand, result);
        }
        
        blendMaskWork |= currMaskWork;
        result.copyTo(blendImageWork, blendMaskWork); 
    }
#if WRITE_CONSOLE
    timer.end();
    printf("blend time elapse: %f\n", timer.elapse());
#endif

    //cv::imshow("blend image", blendImage);
    //cv::imshow("blend mask", blendMask);
    //cv::waitKey(0);
}

void parallelBlend(const BlendConfig& config, const std::vector<cv::Mat>& images,
    const std::vector<cv::Mat>& masks, cv::Mat& blendImage)
{
    CV_Assert(checkSize(images, masks) && checkType(images, CV_8UC3) && checkType(masks, CV_8UC1));

    BlendConfig cfg = validate(config);

    std::vector<cv::Mat> newMasks;
    if (cfg.seamMode == BlendConfig::SEAM_SKIP)
        newMasks = masks;
    else if (cfg.seamMode == BlendConfig::SEAM_DISTANCE_TRANSFORM)
        getNonIntersectingMasks(masks, newMasks);
    else
    {
        findSeams(images, masks, newMasks, cfg.padForGraphCut, 
            cfg.scaleForGraphCut, cfg.ratioForGraphCut, cfg.refineForGraphCut);
    }

    if (cfg.blendMode == BlendConfig::BLEND_PASTE)
    {
        int numImages = images.size();
        cv::Size size = images[0].size();
        blendImage.create(size, CV_8UC3);
        for (int i = 0; i < numImages; i++)
        {
            images[i].copyTo(blendImage, newMasks[i]);
        }
    }
    else if (cfg.blendMode == BlendConfig::BLEND_LINEAR)
    {
        linearBlend(images, masks, newMasks, cfg.radiusForLinear, blendImage);
    }
    else
    {
        multibandBlendAnyMask(images, masks, newMasks,
            cfg.maxLevelsForMultiBand, cfg.minLengthForMultiBand, blendImage);
    }
}

void checkBelonging(const cv::Mat& image, const cv::Mat& mask, cv::Mat& blendImage, cv::Mat& blendMask)
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