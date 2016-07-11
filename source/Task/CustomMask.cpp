#include "CustomMask.h"
#include "opencv2/imgproc.hpp"

void CustomIntervaledMasks::reset()
{
    clearAllMasks();
    width = 0;
    height = 0;
    initSuccess = 0;
}

bool CustomIntervaledMasks::init(int width_, int height_)
{
    clearAllMasks();

    initSuccess = 0;
    if (width_ < 0 || height_ < 0)
        return false;

    width = width_;
    height = height_;
    initSuccess = 1;
    return true;
}

bool CustomIntervaledMasks::getMask2(int index, cv::Mat& mask) const
{
    if (!initSuccess)
    {
        mask = cv::Mat();
        return false;
    }

    int size = masks.size();
    for (int i = 0; i < size; i++)
    {
        const IntervaledMask& currMask = masks[i];
        if (index >= currMask.begIndexInc && index < currMask.endIndexInc)
        {
            mask = currMask.mask;
            return true;
        }
    }
    mask = cv::Mat();
    return false;
}

bool CustomIntervaledMasks::addMask2(int begIndexInc, int endIndexInc, const cv::Mat& mask)
{
    if (!initSuccess)
        return false;

    if (!mask.data || mask.type() != CV_8UC1 || mask.cols != width || mask.rows != height)
        return false;

    masks.push_back(IntervaledMask(-1, begIndexInc, endIndexInc, mask.clone()));
    return true;
}

void CustomIntervaledMasks::clearMask2(int begIndexInc, int endIndexExc)
{
    for (std::vector<IntervaledMask>::iterator itr = masks.begin(); itr != masks.end();)
    {
        if (itr->begIndexInc == begIndexInc &&
            itr->endIndexInc == endIndexExc)
            itr = masks.erase(itr);
        else
            ++itr;
    }
}

void CustomIntervaledMasks::clearAllMasks()
{
    masks.clear();
}

void GeneralMasks::reset()
{
    customMasks.clear();
    defaultMasks.clear();
    width = 0;
    height = 0;
    initSuccess = 0;
    numVideos = 0;
}

bool GeneralMasks::init(const std::vector<cv::Mat>& masks)
{
    reset();
    
    numVideos = masks.size();
    if (numVideos == 0)
        return false;

    int ok = 1;
    int rows = masks[0].rows, cols = masks[0].cols;
    for (int i = 0; i < numVideos; i++)
    {
        if (masks[i].rows != rows || masks[i].cols != cols || masks[i].type() != CV_8UC1)
        {
            ok = 0;
            break;
        }
    }
    if (!ok)
    {
        return false;
    }

    defaultMasks.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
        defaultMasks[i] = masks[i].clone();
    return true;
}

bool GeneralMasks::getMasks(const std::vector<int>& frameIndexes, std::vector<cv::Mat>& masks)
{
    if (!initSuccess)
        return false;

    if (frameIndexes.size() != numVideos)
        return false;

    for (int i = 0; i < numVideos; i++)
    {
        bool hasCustomMask = false;
        int numCustomMasks = customMasks[i].size();
        int frameIndex = frameIndexes[i];
        for (int k = 0; k < numCustomMasks; k++)
        {
            if (customMasks[i][k].begIndexInc <= frameIndex &&
                customMasks[i][k].endIndexInc >= frameIndex)
            {
                hasCustomMask = true;
                masks[i] = customMasks[i][k].mask;
                break;
            }
        }
        if (!hasCustomMask)
            masks[i] = defaultMasks[i];
    }
    return true;
}

bool GeneralMasks::addMasks(const std::vector<IntervaledMask>& masks)
{
    if (!initSuccess)
        return false;

    std::vector<IntervaledMask> newMasks = masks;
    int numMasks = newMasks.size();
    for (int i = 0; i < numMasks; i++)
        newMasks[i].mask = masks[i].mask.clone();
    customMasks.push_back(newMasks);
    return true;
}

bool cvtMaskToContour(const IntervaledMask& mask, IntervaledContour& contour)
{
    if (!mask.mask.data || mask.mask.type() != CV_8UC1)
    {
        contour = IntervaledContour();
        return false;
    }

    contour.videoIndex = mask.videoIndex;
    contour.begIndexInc = mask.begIndexInc;
    contour.endIndexInc = mask.endIndexInc;

    int rows = mask.mask.rows, cols = mask.mask.cols;
    int pad = 4;
    std::vector<std::vector<cv::Point> > contours;
    if (cv::countNonZero(mask.mask.row(0)) || cv::countNonZero(mask.mask.row(rows - 1)) ||
        cv::countNonZero(mask.mask.col(0)) || cv::countNonZero(mask.mask.col(cols - 1)))
    {
        cv::Mat extendMask(rows + 2 * pad, cols + 2 * pad, CV_8UC1);
        extendMask.setTo(0);
        cv::Mat roi = extendMask(cv::Rect(pad, pad, cols, rows));
        mask.mask.copyTo(roi);
        cv::findContours(extendMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        int numCountors = contours.size();
        cv::Point offset(pad, pad);
        for (int i = 0; i < numCountors; i++)
        {
            int len = contours[i].size();
            for (int j = 0; j < len; j++)
                contours[i][j] -= offset;
        }
    }
    else
    {
        cv::Mat nonExtendMask;
        mask.mask.copyTo(nonExtendMask);
        cv::findContours(nonExtendMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    }
    contour.contours = contours;
    contour.width = cols;
    contour.height = rows;

    return true;
}

bool cvtContourToMask(const IntervaledContour& contour, const cv::Mat& boundedMask, IntervaledMask& customMask)
{
    if (contour.width <= 0 || contour.height <= 0 ||
        !boundedMask.data || boundedMask.type() != CV_8UC1)
    {
        customMask = IntervaledMask();
        return false;
    }

    customMask.videoIndex = contour.videoIndex;
    customMask.begIndexInc = contour.begIndexInc;
    customMask.endIndexInc = contour.endIndexInc;
    customMask.mask.create(boundedMask.size(), CV_8UC1);
    customMask.mask.setTo(0);
    cv::Mat temp;
    if (contour.contours.size())
    {
        if (contour.width == boundedMask.cols &&
            contour.height == boundedMask.rows)
            cv::drawContours(customMask.mask, contour.contours, -1, 255, CV_FILLED);
        else
        {
            temp.create(contour.height, contour.width, CV_8UC1);
            temp.setTo(0);
            cv::drawContours(temp, contour.contours, -1, 255, CV_FILLED);
            cv::resize(temp, customMask.mask, customMask.mask.size());
            customMask.mask.setTo(255, customMask.mask);
        }
        customMask.mask &= boundedMask;
    }
    return true;
}

bool cvtContoursToMasks(const std::vector<std::vector<IntervaledContour> >& contours,
    const std::vector<cv::Mat>& boundedMasks, std::vector<CustomIntervaledMasks>& customMasks)
{
    customMasks.clear();
    if (contours.size() != boundedMasks.size())
        return false;

    if (boundedMasks.empty())
        return true;

    int size = boundedMasks.size();
    int width = boundedMasks[0].cols, height = boundedMasks[0].rows;
    for (int i = 1; i < size; i++)
    {
        if (boundedMasks[i].cols != width || boundedMasks[i].rows != height)
            return false;
    }

    bool success = true;
    customMasks.resize(size);
    IntervaledMask currItvMask;
    for (int i = 0; i < size; i++)
    {
        customMasks[i].init(width, height);
        int num = contours[i].size();
        for (int j = 0; j < num; j++)
        {
            if (!cvtContourToMask(contours[i][j], boundedMasks[i], currItvMask))
            {
                success = false;
                break;
            }
            customMasks[i].addMask2(currItvMask.begIndexInc, currItvMask.endIndexInc, currItvMask.mask);
        }
        if (!success)
            break;
    }
    if (!success)
        customMasks.clear();

    return success;
}

bool cvtContoursToCudaMasks(const std::vector<std::vector<IntervaledContour> >& contours,
    const std::vector<cv::Mat>& boundedMasks, std::vector<CudaCustomIntervaledMasks>& customMasks)
{
    customMasks.clear();
    if (contours.size() != boundedMasks.size())
        return false;

    if (boundedMasks.empty())
        return true;

    int size = boundedMasks.size();
    int width = boundedMasks[0].cols, height = boundedMasks[0].rows;
    for (int i = 1; i < size; i++)
    {
        if (boundedMasks[i].cols != width || boundedMasks[i].rows != height)
            return false;
    }

    bool success = true;
    customMasks.resize(size);
    IntervaledMask currItvMask;
    cv::cuda::GpuMat cudaMask;
    for (int i = 0; i < size; i++)
    {
        customMasks[i].init(width, height);
        int num = contours[i].size();
        for (int j = 0; j < num; j++)
        {
            if (!cvtContourToMask(contours[i][j], boundedMasks[i], currItvMask))
            {
                success = false;
                break;
            }
            cudaMask.upload(currItvMask.mask);
            customMasks[i].addMask2(currItvMask.begIndexInc, currItvMask.endIndexInc, cudaMask);
        }
        if (!success)
            break;
    }
    if (!success)
        customMasks.clear();

    return success;
}
