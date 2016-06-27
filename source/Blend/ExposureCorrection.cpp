#include "ZBlend.h"
#include "ZBlendAlgo.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

void fitParabola(const std::vector<cv::Point>& pts, double& a, double& b, double& c)
{
    int numPoints = pts.size();
    CV_Assert(numPoints >= 3);

    //cv::Mat_<double> A(numPoints, 3);
    //cv::Mat_<double> B(numPoints, 1);
    //cv::Mat_<double> X;
    //for (int i = 0; i < numPoints; i++)
    //{
    //    double val = pts[i].x;
    //    A(i, 0) = 1;
    //    A(i, 1) = val;
    //    val *= val;
    //    A(i, 2) = val;
    //    B(i) = pts[i].y;
    //}
    //bool success = cv::solve(A, B, X, cv::DECOMP_NORMAL);
    //printf("result of fitting parabola:\n");
    //std::cout << X << "\n";
    //if (success)
    //{
    //    a = X(2);
    //    b = X(1);
    //    c = X(0);
    //}
    //else
    //{
    //    a = 0;
    //    b = 1;
    //    c = 0;
    //}

    double A = 0, B = 0;
    for (int i = 0; i < numPoints; i++)
    {
        double x = pts[i].x, y = pts[i].y;
        double temp1 = x * x - 255 * x;
        double temp2 = x - y;
        A += temp1 * temp1;
        B += temp1 * temp2;
    }
    if (abs(A) < 0.001)
    {
        a = 0;
        b = 1;
        c = 0;
    }
    else
    {
        a = -B / A;
        b = 1 - 255 * a;
        c = 0;
    }
}

void getLUT(std::vector<unsigned char>& lut, double a, double b, double c)
{
    lut.resize(256);
    for (int i = 0; i < 256; i++)
        lut[i] = cv::saturate_cast<unsigned char>(a * i * i + b * i + c);
}

void calcTransform(const cv::Mat& image, const cv::Mat& imageMask, const cv::Mat& base, const cv::Mat& baseMask,
    std::vector<unsigned char>& lut)
{
    CV_Assert(image.data && image.type() == CV_8UC3 &&
        base.data && base.type() == CV_8UC3 && imageMask.data && imageMask.type() == CV_8UC1 &&
        baseMask.data && baseMask.type() == CV_8UC1);
    int rows = image.rows, cols = image.cols;
    CV_Assert(imageMask.rows == rows && imageMask.cols == cols &&
        base.rows == rows && base.cols == cols && baseMask.rows == rows && baseMask.cols == cols);

    cv::Mat intersectMask = imageMask & baseMask;

    int count = cv::countNonZero(intersectMask);
    std::vector<cv::Point> valPairs(count);
    cv::Mat baseGray, imageGray;
    cv::cvtColor(base, baseGray, CV_BGR2GRAY);
    cv::cvtColor(image, imageGray, CV_BGR2GRAY);

    int index = 0;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrBase = baseGray.ptr<unsigned char>(i);
        const unsigned char* ptrImage = imageGray.ptr<unsigned char>(i);
        const unsigned char* ptrMask = intersectMask.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            int baseVal = ptrBase[j];
            int imageVal = ptrImage[j];
            if (ptrMask[j] && baseVal > 15 && baseVal < 240)
            {
                valPairs[index++] = cv::Point(imageVal, baseVal);
            }
        }
    }
    valPairs.resize(index);

    double a, b, c;
    fitParabola(valPairs, a, b, c);
    getLUT(lut, a, b, c);
    //showLUT("parabola lut", lut);
    //cv::waitKey(0);
}

void valuesInRange8UC1(const cv::Mat& image, int begInc, int endExc, cv::Mat& mask)
{
    CV_Assert(image.data && image.type() == CV_8UC1);
    int rows = image.rows, cols = image.cols;
    mask.create(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrImage = image.ptr<unsigned char>(i);
        unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            int val = *(ptrImage++);
            *(ptrMask++) = (val >= begInc && val < endExc) ? 255 : 0;
        }
    }
}

struct ImageInfo
{
    ImageInfo() : i(0), fullMean(0), mainMean(0), seamMean(0) {};
    int i;
    cv::Mat fullMask;
    cv::Mat mainMask;
    cv::Mat seamMask;
    cv::Mat gray;
    double fullMean;
    double mainMean;
    double seamMean;
};

struct IntersectionInfo
{
    IntersectionInfo() :
    i(0), j(0),
    numFullNonZero(0), numMainNonZero(0), numSeamNonZero(0),
    iFullMean(0), jFullMean(0), iMainMean(0), jMainMean(0), iSeamMean(0), jSeamMean(0)
    {}
    int i, j;
    cv::Mat fullMask;
    cv::Mat mainMask;
    cv::Mat seamMask;
    int numFullNonZero;
    int numMainNonZero;
    int numSeamNonZero;
    double iFullMean, jFullMean;
    double iMainMean, jMainMean;
    double iSeamMean, jSeamMean;
};

struct GroupInfo
{
    GroupInfo(int numTotal_)
    {
        numTotal = numTotal_;
        includes.resize(numTotal, 0);
    }
    std::vector<int> indexes;
    std::vector<int> includes;
    int numTotal;
};

void calcInfo(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    std::vector<ImageInfo>& imageInfos, std::vector<IntersectionInfo>& intersectInfos)
{
    imageInfos.clear();
    intersectInfos.clear();
    if (masks.empty())
        return;

    int size = masks.size();
    int rows = masks[0].rows, cols = masks[0].cols;
    for (int i = 0; i < size; i++)
    {
        CV_Assert(masks[i].data && masks[i].type() == CV_8UC1 &&
            masks[i].rows == rows && masks[i].cols == cols);
        CV_Assert(images[i].data && images[i].type() == CV_8UC3 &&
            images[i].rows == rows && images[i].cols == cols);
    }

    std::vector<cv::Mat> seamMasks;
    getExtendedMasks(masks, 100, seamMasks);

    imageInfos.resize(size);
    for (int i = 0; i < size; i++)
    {
        ImageInfo& imageInfo = imageInfos[i];
        imageInfo.i = i;
        imageInfo.fullMask = masks[i].clone();
        cv::cvtColor(images[i], imageInfo.gray, CV_BGR2GRAY);
        valuesInRange8UC1(imageInfo.gray, 16, 240, imageInfo.mainMask);
        imageInfo.seamMask = seamMasks[i];
        imageInfo.fullMean = cv::mean(imageInfo.gray, imageInfo.fullMask)[0];
        imageInfo.mainMean = cv::mean(imageInfo.gray, imageInfo.mainMask)[0];
        imageInfo.seamMean = cv::mean(imageInfo.gray, imageInfo.seamMask)[0];
    }

    for (int i = 0; i < size - 1; i++)
    {
        for (int j = i + 1; j < size; j++)
        {
            cv::Mat mask = masks[i] & masks[j];
            int numNonZero = cv::countNonZero(mask);
            if (numNonZero)
            {
                intersectInfos.push_back(IntersectionInfo());
                IntersectionInfo& intersect = intersectInfos.back();
                intersect.i = i;
                intersect.j = j;

                intersect.fullMask = mask.clone();
                intersect.numFullNonZero = numNonZero;
                intersect.iFullMean = cv::mean(imageInfos[i].gray, mask)[0];
                intersect.jFullMean = cv::mean(imageInfos[j].gray, mask)[0];

                mask = imageInfos[i].mainMask & imageInfos[j].mainMask;
                intersect.mainMask = mask.clone();
                intersect.numMainNonZero = cv::countNonZero(mask);
                intersect.iMainMean = cv::mean(imageInfos[i].gray, mask)[0];
                intersect.jMainMean = cv::mean(imageInfos[j].gray, mask)[0];

                mask = imageInfos[i].seamMask & imageInfos[j].seamMask;
                intersect.seamMask = mask.clone();
                intersect.numSeamNonZero = cv::countNonZero(mask);
                intersect.iSeamMean = cv::mean(imageInfos[i].gray, mask)[0];
                intersect.jSeamMean = cv::mean(imageInfos[j].gray, mask)[0];
            }
        }
    }
}

void pickAlwaysLargeOrSmall(const std::vector<IntersectionInfo>& intersectInfos, double thresh,
    std::vector<int>& alwaysSmallIndexes, std::vector<int>& alwaysLargeIndexes)
{
    alwaysSmallIndexes.clear();
    alwaysLargeIndexes.clear();
    int intersectSize = intersectInfos.size();
    if (!intersectSize)
        return;

    int numImages = 0;
    for (int i = 0; i < intersectSize; i++)
    {
        const IntersectionInfo& info = intersectInfos[i];
        numImages = std::max(numImages, std::max(info.i, info.j));
    }
    numImages++;

    std::vector<int> smallCount(numImages, 0);
    std::vector<int> largeCount(numImages, 0);
    std::vector<int> totalCount(numImages, 0);
    for (int i = 0; i < intersectSize; i++)
    {
        const IntersectionInfo& info = intersectInfos[i];
        if (!info.numSeamNonZero)
            continue;

        totalCount[info.i]++;
        totalCount[info.j]++;
        if (info.iSeamMean > info.jSeamMean + thresh)
        {
            smallCount[info.j]++;
            largeCount[info.i]++;
        }
        if (info.jSeamMean > info.iSeamMean + thresh)
        {
            smallCount[info.i]++;
            largeCount[info.j]++;
        }
    }

    for (int i = 0; i < numImages; i++)
    {
        if (smallCount[i] == totalCount[i])
            alwaysSmallIndexes.push_back(i);
        if (largeCount[i] == totalCount[i])
            alwaysLargeIndexes.push_back(i);
    }
}

void group(const std::vector<IntersectionInfo>& intersectInfos, double thresh,
    std::vector<GroupInfo>& groupInfos, std::vector<int>& groupIndexes)
{
    groupInfos.clear();
    groupIndexes.clear();
    int intersectSize = intersectInfos.size();
    if (!intersectSize)
        return;

    int numImages = 0;
    for (int i = 0; i < intersectSize; i++)
    {
        const IntersectionInfo& info = intersectInfos[i];
        numImages = std::max(numImages, std::max(info.i, info.j));
    }
    numImages++;

    std::vector<IntersectionInfo> groupIntersectInfos = intersectInfos;
    typedef std::vector<IntersectionInfo>::iterator InserectIterator;
    // Remove the intersections which numSeamNonZero is zero
    // or abs(iSeamMean - jSeamMean) is too large
    for (InserectIterator itr = groupIntersectInfos.begin(); itr != groupIntersectInfos.end();)
    {
        if (!itr->numSeamNonZero)
        {
            itr = groupIntersectInfos.erase(itr);
            continue;
        }

        if (abs(itr->iSeamMean - itr->jSeamMean) > thresh)
        {
            itr = groupIntersectInfos.erase(itr);
            continue;
        }

        ++itr;
    }

    for (int i = 0; i < intersectSize; i++)
    {
        printf("i = %d, j = %d, numNonZero = %d, iSeamMean = %8.4f, jSeamMean = %8.4f\n",
            intersectInfos[i].i, intersectInfos[i].j, intersectInfos[i].numSeamNonZero,
            intersectInfos[i].iSeamMean, intersectInfos[i].jSeamMean);
    }

    //std::vector<double> seamMeanAbsDiff(intersectSize);
    //for (int i = 0; i < intersectSize; i++)
    //    seamMeanAbsDiff[i] = abs(intersectInfos[i].iSeamMean - intersectInfos[i].jSeamMean);
    //double maxDiff = seamMeanAbsDiff[0], meanDiff = 0;
    //int effectNum = 0;
    //for (int i = 0; i < intersectSize; i++)
    //{
    //    maxDiff = std::max(maxDiff, seamMeanAbsDiff[i]);
    //    if (seamMeanAbsDiff[i] > 1)
    //    {
    //        meanDiff += seamMeanAbsDiff[i];
    //        effectNum++;
    //    }        
    //}
    //meanDiff /= effectNum;
    //printf("max diff = %f, mean diff = %f\n", maxDiff, meanDiff);

    // Group the images together, in each group, for each image i, 
    // there is at least one image j that has non zero numSeamNonZero with image i
    // and abs(iSeamMean - jSeamMean) < threshold
    GroupInfo gInfo(numImages);
    gInfo.indexes.push_back(0);
    gInfo.includes[0] = 1;
    groupInfos.push_back(gInfo);
    for (int i = 1; i < numImages; i++)
    {
        int groupIndex = -1;
        int imageIndex = -1;
        int numGroups = groupInfos.size();
        for (int j = 0; j < numGroups; j++)
        {
            int numGroupElems = groupInfos[j].indexes.size();
            for (int k = 0; k < numGroupElems; k++)
            {
                bool foundSmall = false;
                int otherIndex = groupInfos[j].indexes[k];
                for (int u = 0; u < intersectSize; u++)
                {
                    // We must check intersectInfos[u].numSeamNonZero > 0
                    // If no intersection, iSeamMean == jSeamMean == 0, 
                    // then abs(iSeamMean - jSeamMean) <= thresh is true
                    if (intersectInfos[u].numSeamNonZero > 0 &&
                        ((intersectInfos[u].i == i && intersectInfos[u].j == otherIndex) ||
                         (intersectInfos[u].i == otherIndex && intersectInfos[u].j == i)))
                    {
                        if (abs(intersectInfos[u].iSeamMean - intersectInfos[u].jSeamMean) <= thresh)
                        {
                            foundSmall = true;
                            groupIndex = j;
                            imageIndex = i;
                            break;
                        }
                    }
                }
                if (foundSmall)
                    break;
            }
            if (groupIndex >= 0 && imageIndex >= 0)
                break;
        }
        if (groupIndex >= 0 && imageIndex >= 0)
        {
            groupInfos[groupIndex].indexes.push_back(imageIndex);
            groupInfos[groupIndex].includes[imageIndex] = 1;
        }
        else
        {
            GroupInfo gInfo(numImages);
            gInfo.indexes.push_back(i);
            gInfo.includes[i] = 1;
            groupInfos.push_back(gInfo);
        }
    }

    int numGroups = groupInfos.size();
    for (int i = 0; i < numGroups; i++)
    {
        printf("old group [%d]: ", i);
        for (int j = 0; j < groupInfos[i].indexes.size(); j++)
            printf("%d ", groupInfos[i].indexes[j]);
        printf("\n");
    }

    // Further group the existing groups
    // For group i and j, if there exist an image u from group i and and image v from group j
    // satisfying numSeamNonZero is not zero and abs(uSeamMean - vSeamMean) < thresh,
    // then group i and j should be merged
    int lastNumGroups = groupInfos.size();
    typedef std::vector<GroupInfo>::iterator GroupIterator;
    while (true)
    {
        for (GroupIterator itrLeft = groupInfos.begin(); itrLeft - groupInfos.begin() < groupInfos.size() - 1; ++itrLeft)
        {
            for (GroupIterator itrRight = itrLeft + 1; itrRight != groupInfos.end();)
            {
                int leftSize = itrLeft->indexes.size();
                int rightSize = itrRight->indexes.size();
                bool deleteItem = false;
                for (int i = 0; i < leftSize; i++)
                {
                    int leftIndex = itrLeft->indexes[i];
                    for (int j = 0; j < rightSize; j++)
                    {
                        int rightIndex = itrRight->indexes[j];
                        for (int u = 0; u < intersectSize; u++)
                        {
                            if ((intersectInfos[u].numSeamNonZero > 0) &&
                                ((intersectInfos[u].i == leftIndex && intersectInfos[u].j == rightIndex) ||
                                 (intersectInfos[u].i == rightIndex && intersectInfos[u].j == leftIndex)))
                            {
                                if (abs(intersectInfos[u].iSeamMean - intersectInfos[u].jSeamMean) < thresh)
                                {
                                    int numGroupElems = itrRight->indexes.size();
                                    for (int v = 0; v < numGroupElems; v++)
                                    {
                                        itrLeft->indexes.push_back(itrRight->indexes[v]);
                                        itrLeft->includes[itrRight->indexes[v]] = 1;
                                    }
                                    deleteItem = true;
                                    break;
                                }
                            }
                        }
                        if (deleteItem)
                            break;
                    }
                    if (deleteItem)
                        break;
                }
                if (deleteItem)
                    itrRight = groupInfos.erase(itrRight);
                else
                    ++itrRight;
            }
        }
        if (groupInfos.size() == lastNumGroups)
            break;
        lastNumGroups = groupInfos.size();
    }

    numGroups = groupInfos.size();
    for (int i = 0; i < numGroups; i++)
    {
        printf("new group [%d]: ", i);
        for (int j = 0; j < groupInfos[i].indexes.size(); j++)
            printf("%d ", groupInfos[i].indexes[j]);
        printf("\n");
    }
}

void exposureCorrect(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    std::vector<std::vector<unsigned char> >& luts, std::vector<int>& corrected)
{
    CV_Assert(checkSize(images) && checkSize(masks) &&
        checkType(images, CV_8UC3) && checkType(masks, CV_8UC1));

    int numImages = images.size();

    std::vector<ImageInfo> imageInfos;
    std::vector<IntersectionInfo> intersectInfos;
    calcInfo(images, masks, imageInfos, intersectInfos);

    std::vector<GroupInfo> groupInfos;
    std::vector<int> groupIndexes;
    group(intersectInfos, 10, groupInfos, groupIndexes);

    std::vector<int> alwaysSmallIndexes, alwaysLargeIndexes;
    pickAlwaysLargeOrSmall(intersectInfos, 10, alwaysSmallIndexes, alwaysLargeIndexes);
    int numSmall = alwaysSmallIndexes.size();
    int numLarge = alwaysLargeIndexes.size();

    std::vector<int> mainIndexes;
    for (int i = 0; i < numImages; i++)
    {
        bool isMain = true;
        for (int j = 0; j < numSmall; j++)
        {
            if (alwaysSmallIndexes[j] == i)
            {
                isMain = false;
                break;
            }
        }
        if (!isMain)
            continue;
        for (int j = 0; j < numLarge; j++)
        {
            if (alwaysLargeIndexes[j] == i)
            {
                isMain = false;
                break;
            }
        }
        if (!isMain)
            continue;
        mainIndexes.push_back(i);
    }

    int numMain = mainIndexes.size();
    std::vector<cv::Mat> mainImages, mainMasks;
    cv::Mat mainMask = cv::Mat::zeros(masks[0].size(), CV_8UC1);
    for (int i = 0; i < numMain; i++)
    {
        mainImages.push_back(images[mainIndexes[i]]);
        mainMasks.push_back(masks[mainIndexes[i]]);
        mainMask |= masks[mainIndexes[i]];
    }

    BlendConfig blendConfig;
    blendConfig.setSeamDistanceTransform();
    blendConfig.setBlendMultiBand();
    cv::Mat mainBlend;
    parallelBlend(blendConfig, mainImages, mainMasks, mainBlend);
    cv::imshow("blend", mainBlend);
    cv::waitKey(0);

    //std::vector<cv::Mat> adjustLargeImages(numLarge), adjustSmallImages(numSmall);
    //std::vector<unsigned char> lut;
    //for (int i = 0; i < numLarge; i++)
    //{
    //    calcTransform(images[alwaysLargeIndexes[i]], masks[alwaysLargeIndexes[i]], mainBlend, mainMask, lut);
    //    transform(images[alwaysLargeIndexes[i]], adjustLargeImages[i], lut, masks[alwaysLargeIndexes[i]]);
    //}
    //for (int i = 0; i < numSmall; i++)
    //{
    //    calcTransform(images[alwaysSmallIndexes[i]], masks[alwaysSmallIndexes[i]], mainBlend, mainMask, lut);
    //    transform(images[alwaysSmallIndexes[i]], adjustSmallImages[i], lut, masks[alwaysSmallIndexes[i]]);
    //}

    printf("num large = %d: ", numLarge);
    for (int i = 0; i < numLarge; i++)
        printf("%d ", alwaysLargeIndexes[i]);
    printf("\n");
    printf("num small = %d: ", numSmall);
    for (int i = 0; i < numSmall; i++)
        printf("%d ", alwaysSmallIndexes[i]);
    printf("\n");

    luts.resize(numImages);
    corrected.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        int largeIndex = -1, smallIndex = -1;
        for (int k = 0; k < numLarge; k++)
        {
            if (alwaysLargeIndexes[k] == i)
            {
                largeIndex = k;
                break;
            }
        }
        for (int k = 0; k < numSmall; k++)
        {
            if (alwaysSmallIndexes[k] == i)
            {
                smallIndex = k;
                break;
            }
        }
        if (largeIndex >= 0 || smallIndex >= 0)
        {
            calcTransform(images[i], masks[i], mainBlend, mainMask, luts[i]);
            corrected[i] = 1;
        }
        else
        {
            luts[i].resize(256);
            for (int j = 0; j < 256; j++)
                luts[i][j] = j;
            corrected[i] = 0;
        }
    }
}