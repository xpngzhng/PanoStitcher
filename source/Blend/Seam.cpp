#include "gcgraph.hpp"
#include "ZBlendAlgo.h"
#include "SeamVisualizer.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define SHOW_SEAM 0

static void calcRGBColorDiff(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& diff)
{
    CV_Assert(img1.data && img1.type() == CV_8UC3 &&
        img2.data && img2.type() == CV_8UC3 &&
        img1.size() == img2.size());

    int width = img1.cols, height = img1.rows;
    diff.create(height, width, CV_32FC1);
    for (int i = 0; i < height; i++)
    {
        const unsigned char* ptr1 = img1.ptr<unsigned char>(i);
        const unsigned char* ptr2 = img2.ptr<unsigned char>(i);
        float* ptr = diff.ptr<float>(i);
        for (int j = 0; j < width; j++)
        {
            int diff0 = int(ptr1[0]) - int(ptr2[0]);
            int diff1 = int(ptr1[1]) - int(ptr2[1]);
            int diff2 = int(ptr1[2]) - int(ptr2[2]);
            *ptr = sqrtf(diff0 * diff0 + diff1 * diff1 + diff2 * diff2);
            ptr++;
            ptr1 += 3;
            ptr2 += 3;
        }
    }

    //cv::Mat hori, vert;
    //cv::Sobel(diff, hori, CV_32F, 1, 0);
    //cv::Sobel(diff, vert, CV_32F, 0, 1);
    //diff = cv::abs(hori) + cv::abs(vert);
}

static void calcLABColorDiff(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& diff)
{
    CV_Assert(img1.data && img1.type() == CV_8UC3 &&
        img2.data && img2.type() == CV_8UC3 &&
        img1.size() == img2.size());

    cv::Mat imgNorm1;
    img1.convertTo(imgNorm1, CV_32F);
    imgNorm1 *= (1.0F / 255);
    cv::cvtColor(imgNorm1, imgNorm1, CV_BGR2Lab);
    
    cv::Mat imgNorm2;
    img2.convertTo(imgNorm2, CV_32F);
    imgNorm2 *= (1.0F / 255);
    cv::cvtColor(imgNorm2, imgNorm2, CV_BGR2Lab);

    int width = img1.cols, height = img1.rows;
    diff.create(height, width, CV_32FC1);
    const float luminanceWeight = 1.F, chrominanceWeight = 4.F;
    for (int i = 0; i < height; i++)
    {
        const cv::Point3f* ptr1 = imgNorm1.ptr<cv::Point3f>(i);
        const cv::Point3f* ptr2 = imgNorm2.ptr<cv::Point3f>(i);
        float* ptr = diff.ptr<float>(i);
        for (int j = 0; j < width; j++)
        {            
            float diffL = ptr1->x - ptr2->x;
            float diffA = ptr1->y - ptr2->y;
            float diffB = ptr1->z - ptr2->z;
            *ptr = sqrt(diffL * diffL * luminanceWeight + 
                (diffA * diffA + diffB * diffB) * chrominanceWeight);
            ptr1++;
            ptr2++;
            ptr++;
        }
    }
}

static void calcHLColorDiff(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& diff)
{
    CV_Assert(img1.data && img1.type() == CV_8UC3 &&
        img2.data && img2.type() == CV_8UC3 &&
        img1.size() == img2.size());
    
    cv::Mat imgNorm1;
    img1.convertTo(imgNorm1, CV_32F);
    imgNorm1 *= (1.0F / 255);
    cv::cvtColor(imgNorm1, imgNorm1, CV_BGR2HLS);
    
    cv::Mat imgNorm2;
    img2.convertTo(imgNorm2, CV_32F);
    imgNorm2 *= (1.0F / 255);
    cv::cvtColor(imgNorm2, imgNorm2, CV_BGR2HLS);

    int width = img1.cols, height = img1.rows;
    diff.create(height, width, CV_32FC1);
    const float luminanceWeight = 1.F, chrominanceWeight = 1.F;
    const float scale = 1.0F / 360;
    for (int i = 0; i < height; i++)
    {
        const cv::Point3f* ptr1 = imgNorm1.ptr<cv::Point3f>(i);
        const cv::Point3f* ptr2 = imgNorm2.ptr<cv::Point3f>(i);
        float* ptr = diff.ptr<float>(i);
        for (int j = 0; j < width; j++)
        {            
            float h1 = ptr1->x * scale;
            float h2 = ptr2->x * scale;
            float diffH = std::abs(h1 - h2), compDiffH = 1.0F - diffH;
            diffH = std::min(diffH, compDiffH);
            if (diffH < 0 || diffH > 1)
            {
                printf("diffH = %f\n", diffH);
                system("pause");
            }
            float diffL = std::abs(ptr1->y - ptr2->y);
            *ptr = std::max(chrominanceWeight * diffH, luminanceWeight * diffL);
            ptr1++;
            ptr2++;
            ptr++;
        }
    }
}

typedef void (*ColorDiffFunc)(const cv::Mat&, const cv::Mat&, cv::Mat&);
ColorDiffFunc calcColorDiff = calcRGBColorDiff;

static void getSobelGrad(const cv::Mat& src, cv::Mat& dx, cv::Mat& dy)
{
    cv::Mat dxC3, dyC3;
    Sobel(src, dxC3, CV_32F, 1, 0);
    Sobel(src, dyC3, CV_32F, 0, 1);
    dx.create(src.size(), CV_32F);
    dy.create(src.size(), CV_32F);
    for (int y = 0; y < src.rows; ++y)
    {
        const cv::Point3f* dxRowC3 = dxC3.ptr<cv::Point3f>(y);
        const cv::Point3f* dyRowC3 = dyC3.ptr<cv::Point3f>(y);
        float* dxRow = dx.ptr<float>(y);
        float* dyRow = dy.ptr<float>(y);
        for (int x = 0; x < src.cols; ++x)
        {
            *dxRow = sqrt(dxRowC3->x * dxRowC3->x + 
                          dxRowC3->y * dxRowC3->y + 
                          dxRowC3->z * dxRowC3->z);
            *dyRow = sqrt(dyRowC3->x * dyRowC3->x + 
                          dyRowC3->y * dyRowC3->y + 
                          dyRowC3->z * dyRowC3->z);
            dxRow++;
            dyRow++;
            dxRowC3++;
            dyRowC3++;
        }
    }
}

static void showFloatMat(const std::string& winName, const cv::Mat& image)
{
    CV_Assert(image.data && image.type() == CV_32FC1);
    
    double minVal, maxVal;
    cv::minMaxLoc(image, &minVal, &maxVal);
    printf("%s, minVal = %f, maxVal = %f\n", winName.c_str(), minVal, maxVal);
    cv::imshow(winName, image / maxVal);
}

static void findSeam(const cv::Mat& diff, cv::Mat& mask1, cv::Mat& mask2, 
    float terminalCost, float badRegionPenalty, bool horiWrap)
{
    CV_Assert(diff.data && diff.type() == CV_32FC1 &&
        mask1.data && mask1.type() == CV_8UC1 &&
        mask2.data && mask2.type() == CV_8UC1 &&
        diff.size() == mask1.size() && diff.size() == mask2.size());

    int width = diff.cols, height = diff.rows;

    const int vertexCount = height * width;
    const int edgeCount = (height - 1) * width + (horiWrap ? width : width - 1) * height;
    GCGraph<float> graph(vertexCount, edgeCount); 

    // Set terminal weights
    for (int y = 0; y < height; ++y)
    {
        const unsigned char* ptr1 = mask1.ptr<unsigned char>(y);
        const unsigned char* ptr2 = mask2.ptr<unsigned char>(y);
        for (int x = 0; x < width; ++x)
        {
            int v = graph.addVtx();
            graph.addTermWeights(v, ptr1[x] ? terminalCost : 0.f,
                                    ptr2[x] ? terminalCost : 0.f);
        }
    }

    // Set regular edge weights
    const float weightEps = 1.f;
    for (int y = 0; y < height; ++y)
    {
        const float* ptrDiff = diff.ptr<float>(y);
        const unsigned char* ptrMask1 = mask1.ptr<unsigned char>(y);
        const unsigned char* ptrMask2 = mask2.ptr<unsigned char>(y);
        const float* ptrDiffNext = 0;
        const unsigned char* ptrMask1Next = 0;
        const unsigned char* ptrMask2Next = 0;
        if (y < height - 1)
        {
            ptrDiffNext = diff.ptr<float>(y + 1);
            ptrMask1Next = mask1.ptr<unsigned char>(y + 1);
            ptrMask2Next = mask2.ptr<unsigned char>(y + 1);
        }
        for (int x = 0; x < width; ++x)
        {
            int v = y * width + x;
            if (x < width - 1)
            {
                float weight = ptrDiff[x] + ptrDiff[x + 1] + weightEps;
                if (!ptrMask1[x] || !ptrMask1[x + 1] ||
                    !ptrMask2[x] || !ptrMask2[x + 1])
                    weight += badRegionPenalty;
                graph.addEdges(v, v + 1, weight, weight);
            }
            else if (horiWrap)
            {
                float weight = ptrDiff[x] + ptrDiff[x - width + 1] + weightEps;
                if (!ptrMask1[x] || !ptrMask1[x - width + 1] ||
                    !ptrMask2[x] || !ptrMask2[x - width + 1])
                    weight += badRegionPenalty;
                graph.addEdges(v, v - width + 1, weight, weight);
            }
            if (y < height - 1)
            {
                float weight = ptrDiff[x] + ptrDiffNext[x] + weightEps;
                if (!ptrMask1[x] || !ptrMask1Next[x] ||
                    !ptrMask2[x] || !ptrMask2Next[x])
                    weight += badRegionPenalty;
                graph.addEdges(v, v + width, weight, weight);
            }
        }
    }

    graph.maxFlow();

    for (int y = 0; y < height; ++y)
    {
        unsigned char* ptrMask1 = mask1.ptr<unsigned char>(y);
        unsigned char* ptrMask2 = mask2.ptr<unsigned char>(y);
        for (int x = 0; x < width; ++x)
        {
            if (graph.inSourceSegment(y * width + x))
            {
                if (ptrMask1[x])
                    ptrMask2[x] = 0;
            }
            else
            {
                if (ptrMask2[x])
                    ptrMask1[x] = 0;
            }
        }
    }
}

const static float eps32F = std::numeric_limits<float>::epsilon();

static void divDiff(const cv::Mat& num, const cv::Mat& den, cv::Mat& quo)
{
    CV_Assert(num.data && num.type() == CV_32FC1 &&
        den.data && den.type() == CV_32FC1 &&
        num.size() == den.size());
    int rows = num.rows, cols = num.cols;
    quo.create(rows, cols, CV_32FC1);
    for (int i = 0; i < rows; i++)
    {
        const float* ptrNum = num.ptr<float>(i);
        const float* ptrDen = den.ptr<float>(i);
        float* ptrQuo = quo.ptr<float>(i);
        for (int j = 0; j < cols; j++)
        {
            *ptrQuo = std::min(*ptrNum / (*ptrDen > 0 ? *ptrDen : eps32F), 5.0F);
            ptrNum++;
            ptrDen++;
            ptrQuo++;
        }
    }
}

static void divDiff(const cv::Mat& num, const cv::Mat& den, const cv::Mat& mask, cv::Mat& quo)
{
    CV_Assert(num.data && num.type() == CV_32FC1 &&
        den.data && den.type() == CV_32FC1 &&
        mask.data && mask.type() == CV_8UC1 &&
        num.size() == den.size() && num.size() == mask.size());
    int rows = num.rows, cols = num.cols;
    quo.create(rows, cols, CV_32FC1);
    for (int i = 0; i < rows; i++)
    {
        const float* ptrNum = num.ptr<float>(i);
        const float* ptrDen = den.ptr<float>(i);
        const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        float* ptrQuo = quo.ptr<float>(i);
        for (int j = 0; j < cols; j++)
        {
            float den = *ptrDen;
            if (den < 0 || !(*ptrMask))
                den = eps32F;
            else
                den = den * 255 / *ptrMask;
            *ptrQuo = std::min(*ptrNum / den, 5.0F);
            ptrNum++;
            ptrDen++;
            ptrMask++;
            ptrQuo++;
        }
    }
}

void findSeam(const cv::Mat& image1, const cv::Mat& image2,
    cv::Mat& mask1, cv::Mat& mask2, bool horiWrap)
{
    CV_Assert(image1.data && image1.type() == CV_8UC3 &&
        image2.data && image2.type() == CV_8UC3 &&
        mask1.data && mask1.type() == CV_8UC1 &&
        mask2.data && mask2.type() == CV_8UC1);
    cv::Size size = image1.size();
    CV_Assert(image2.size() == size && mask1.size() == size && mask2.size() == size);
    
    cv::Mat intersect = mask1 & mask2, intersectBlur;
    cv::blur(intersect, intersectBlur, cv::Size(9, 9));

    cv::Mat diff;
    calcColorDiff(image1, image2, diff);
    cv::blur(diff, diff, cv::Size(3, 3));
    
    //diff.setTo(0, ~intersect);
    //cv::Mat diffBlur, diffQuo;
    //cv::blur(diff, diffBlur, cv::Size(9, 9));    
    //divDiff(diff, diffBlur, intersectBlur, diffQuo);
    //showFloatMat("diff", diff);
    //showFloatMat("diff blur", diffBlur);
    //showFloatMat("diff quo", diffQuo);
    //diff = diffQuo;

#if SHOW_SEAM
    SeamVisualizer vis(mask1, mask2, diff);
    vis.show("diff without seam");
#endif
    findSeam(diff, mask1, mask2, /*FLT_MAX, FLT_MAX*/10000, 1000, horiWrap);
#if SHOW_SEAM
    vis.drawSeam(mask1, mask2);
    vis.show("diff with seam");
#endif
}

void findSeamByRegionSplitLine(const cv::Mat& image1, const cv::Mat& image2, 
    const cv::Mat& region1, const cv::Mat& region2, cv::Mat& mask1, cv::Mat& mask2, bool horiWrap)
{
    CV_Assert(image1.data && image1.type() == CV_8UC3 &&
        image2.data && image2.type() == CV_8UC3 &&
        region1.data && region1.type() == CV_8UC1 &&
        region2.data && region2.type() == CV_8UC1 &&
        mask1.data && mask1.type() == CV_8UC1 &&
        mask2.data && mask2.type() == CV_8UC1);
    cv::Size size = image1.size();
    CV_Assert(image2.size() == size && 
        region1.size() == size && region2.size() == size &&
        mask1.size() == size && mask2.size() == size);
    
    cv::Mat diff;
    calcColorDiff(image1, image2, diff);
    //showFloatMat("diff", diff);
    
    cv::Mat dist;
    cv::distanceTransform(region1, dist, CV_DIST_L1, 3);
    diff += dist * 0.5;
    cv::distanceTransform(region2, dist, CV_DIST_L1, 3);
    diff += dist * 0.5;
    findSeam(diff, mask1, mask2, /*FLT_MAX, FLT_MAX*/10000, 1000, horiWrap);
}

static void downMaskAugment(const cv::Mat& src, cv::Mat& dst, int scale)
{
    CV_Assert(src.data && src.type() == CV_8UC1 && 
        scale > 1 && src.rows > scale && src.cols > scale);

    int srcRows = src.rows, srcCols = src.cols;
    int dstRows = (srcRows + scale - 1) / scale, dstCols = (srcCols + scale - 1) / scale;
    dst.create(dstRows, dstCols, CV_8UC1);
    dst.setTo(0);
    const int scaleMinusOne = scale - 1;
    for (int i = 0; i < srcRows; i++)
    {
        const unsigned char* ptrSrc = src.ptr<unsigned char>(i);
        unsigned char* ptrDst = dst.ptr<unsigned char>(i / scale);
        for (int j = 0; j <srcCols; j++)
        {
            if (*(ptrSrc++))
                *ptrDst = 255;
            if (j % scale == scaleMinusOne)
                ptrDst++;
        }
    }
}

static void downMaskShrink(const cv::Mat& src, cv::Mat& dst, int scale)
{
    CV_Assert(src.data && src.type() == CV_8UC1 && 
        scale > 1 && src.rows > scale && src.cols > scale);

    int srcRows = src.rows, srcCols = src.cols;
    int dstRows = (srcRows + scale - 1) / scale, dstCols = (srcCols + scale - 1) / scale;
    dst.create(dstRows, dstCols, CV_8UC1);
    dst.setTo(0);
    for (int i = 0; i < dstRows; i++)
    {
        int srcRowBeg = i * scale;
        int srcRowEnd = std::min((i + 1) * scale, srcRows);
        unsigned char* ptrDst = dst.ptr<unsigned char>(i);
        for (int j = 0; j < dstCols; j++)
        {
            int srcColBeg = j * scale;
            int srcColEnd = std::min((j + 1) * scale, srcCols);
            cv::Rect srcRect(cv::Point(srcColBeg, srcRowBeg), cv::Point(srcColEnd, srcRowEnd));
            if (cv::countNonZero(src(srcRect)) == srcRect.area())
                *ptrDst = 255;
            ptrDst++;
            //*(ptrDst++) = (cv::countNonZero(src(srcRect)) == srcRect.area()) ? 255 : 0;
        }
    }
}

static void upMask(const cv::Mat& src, cv::Mat& dst, int scale, const cv::Size& dsize)
{
    CV_Assert(src.data && src.type() == CV_8UC1 && scale > 1);

    int srcRows = src.rows, srcCols = src.cols;
    int dstRows = dsize.height, dstCols = dsize.width;
    dst.create(dsize, CV_8UC1);
    const int scaleMinusOne = scale - 1;
    for (int i = 0; i < dstRows; i++)
    {
        const unsigned char* ptrSrc = src.ptr<unsigned char>(i / scale);
        unsigned char* ptrDst = dst.ptr<unsigned char>(i);
        for (int j = 0; j < dstCols; j++)
        {
            *(ptrDst++) = *ptrSrc;
            if (j % scale == scaleMinusOne)
                ptrSrc++;
        }
    }
}

static void downDiff(const cv::Mat& src, cv::Mat& dst, int scale)
{
    CV_Assert(src.data && src.type() == CV_32FC1);

    int length = (scale & 1) ? scale : scale + 1;
    cv::Mat srcBlur;
    cv::blur(src, srcBlur, cv::Size(length, length));
    //cv::imshow("diff blur", srcBlur * (1.0 / 255));
    //cv::GaussianBlur(src, srcBlur, cv::Size(length, length), length / 6.0, length / 6.0);
    int srcRows = src.rows, srcCols = src.cols;
    int dstRows = (srcRows + scale - 1) / scale, dstCols = (srcCols + scale - 1) / scale;
    dst.create(dstRows, dstCols, CV_32FC1);
    dst.setTo(0);
    for (int i = 0; i < dstRows; i++)
    {
        const float* ptrSrc = srcBlur.ptr<float>(i * scale);
        float* ptrDst = dst.ptr<float>(i);
        for (int j = 0; j < dstCols; j++)
        {
            *(ptrDst++) = *ptrSrc;
            ptrSrc += scale;
        }
    }
}

static int labelIndex(const cv::Mat& mask, cv::Mat& index)
{
    CV_Assert(mask.data && mask.type() == CV_8UC1);
    int rows = mask.rows, cols = mask.cols;
    index.create(rows, cols, CV_32SC1);
    int count = 0;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        int* ptrIndex = index.ptr<int>(i);
        for (int j = 0; j < cols; j++)
        //{
        //    if (*ptrMask)
        //    {
        //        *ptrIndex = count;
        //        count++;
        //    }
        //    else
        //        *ptrIndex = -1;
        //    ptrIndex++;
        //    ptrMask++;
        //}
            *(ptrIndex++) = (*(ptrMask++)) ? (count++) : -1;
    }
    return count;
}

static void findSeamInIndexedROI(const cv::Mat& diff, cv::Mat& mask1, cv::Mat& mask2, cv::Mat& index,
    int num, float terminalCost, float badRegionPenalty, bool horiWrap)
{
    CV_Assert(diff.data && diff.type() == CV_32FC1 &&
        mask1.data && mask1.type() == CV_8UC1 &&
        mask2.data && mask2.type() == CV_8UC1 &&
        index.data && index.type() == CV_32SC1 &&
        diff.size() == mask1.size() && 
        diff.size() == mask2.size() &&
        diff.size() == index.size());
    
    int width = diff.cols, height = diff.rows;

    const int vertexCount = num;
    const int edgeCount = 2 * num;
    GCGraph<float> graph(vertexCount, edgeCount); 

    for (int i = 0; i < num; i++)
        graph.addVtx();

    // Set terminal weights
    for (int y = 0; y < height; ++y)
    {
        const unsigned char* ptr1 = mask1.ptr<unsigned char>(y);
        const unsigned char* ptr2 = mask2.ptr<unsigned char>(y);
        const int* ptrIndex = index.ptr<int>(y);
        for (int x = 0; x < width; ++x)
        {
            int v = ptrIndex[x];
            if (v >= 0)
            {
                graph.addTermWeights(v, ptr1[x] ? terminalCost : 0.f,
                                        ptr2[x] ? terminalCost : 0.f);
            }
        }
    }

    // Set regular edge weights
    const float weightEps = 1.f;
    for (int y = 0; y < height; ++y)
    {
        const float* ptrDiff = diff.ptr<float>(y);
        const unsigned char* ptrMask1 = mask1.ptr<unsigned char>(y);
        const unsigned char* ptrMask2 = mask2.ptr<unsigned char>(y);
        const int* ptrIndex = index.ptr<int>(y);
        const float* ptrDiffNext = 0;
        const unsigned char* ptrMask1Next = 0;
        const unsigned char* ptrMask2Next = 0;
        const int* ptrIndexNext = 0;
        if (y < height - 1)
        {
            ptrDiffNext = diff.ptr<float>(y + 1);
            ptrMask1Next = mask1.ptr<unsigned char>(y + 1);
            ptrMask2Next = mask2.ptr<unsigned char>(y + 1);
            ptrIndexNext = index.ptr<int>(y + 1);
        }
        for (int x = 0; x < width; ++x)
        {
            if (x < width - 1)
            {
                int index = ptrIndex[x], index1 = ptrIndex[x + 1];
                if (index >= 0 && index1 >= 0)
                {
                    float weight = ptrDiff[x] + ptrDiff[x + 1] + weightEps;
                    if (!ptrMask1[x] || !ptrMask1[x + 1] ||
                        !ptrMask2[x] || !ptrMask2[x + 1])
                        weight += badRegionPenalty;
                    graph.addEdges(index, index1, weight, weight);
                }
            }
            else if (horiWrap)
            {
                int index = ptrIndex[x], index1 = ptrIndex[x - width + 1];
                if (index >= 0 && index1 >= 0)
                {
                    float weight = ptrDiff[x] + ptrDiff[x - width + 1] + weightEps;
                    if (!ptrMask1[x] || !ptrMask1[x - width + 1] ||
                        !ptrMask2[x] || !ptrMask2[x - width + 1])
                        weight += badRegionPenalty;
                    graph.addEdges(index, index1, weight, weight);
                }
            }
            if (y < height - 1)
            {
                int index = ptrIndex[x], index1 = ptrIndexNext[x];
                if (index >= 0 && index1 >= 0)
                {
                    float weight = ptrDiff[x] + ptrDiffNext[x] + weightEps;
                    if (!ptrMask1[x] || !ptrMask1Next[x] ||
                        !ptrMask2[x] || !ptrMask2Next[x])
                        weight += badRegionPenalty;
                    graph.addEdges(index, index1, weight, weight);
                }
            }
        }
    }

    graph.maxFlow();

    for (int y = 0; y < height; ++y)
    {
        unsigned char* ptrMask1 = mask1.ptr<unsigned char>(y);
        unsigned char* ptrMask2 = mask2.ptr<unsigned char>(y);
        const int* ptrIndex = index.ptr<int>(y);
        for (int x = 0; x < width; ++x)
        {
            int index = ptrIndex[x];
            if (index >= 0)
            {
                if (graph.inSourceSegment(index))
                {
                    if (ptrMask1[x])
                        ptrMask2[x] = 0;
                }
                else
                {
                    if (ptrMask2[x])
                        ptrMask1[x] = 0;
                }
            }
        }
    }
}

void findSeamInROI(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& roi,
    cv::Mat& mask1, cv::Mat& mask2, bool horiWrap)
{
    CV_Assert(image1.data && image1.type() == CV_8UC3 &&
        image2.data && image2.type() == CV_8UC3 &&
        roi.data && roi.type() == CV_8UC1 &&
        mask1.data && mask1.type() == CV_8UC1 &&
        mask2.data && mask2.type() == CV_8UC1);
    cv::Size size = image1.size();
    CV_Assert(image2.size() == size && roi.size() == size && 
        mask1.size() == size && mask2.size() == size);

    cv::Mat kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 8));
    cv::Mat dilateMask, index;
    cv::dilate(roi, dilateMask, kern);
    int num = labelIndex(dilateMask, index);
    dilateMask.release();
    cv::Mat diff;
    calcColorDiff(image1, image2, diff);
    cv::blur(diff, diff, cv::Size(3, 3));
#if SHOW_SEAM
    SeamVisualizer vis(mask1, mask2, diff);
    vis.show("diff without seam");
#endif
    findSeamInIndexedROI(diff, mask1, mask2, index, num, 10000, 1000, horiWrap);
#if SHOW_SEAM
    vis.drawSeam(mask1, mask2);
    vis.show("diff with seam");
#endif
}

void findSeamScaleDown(const cv::Mat& image1, const cv::Mat& image2, 
    cv::Mat& mask1, cv::Mat& mask2, bool horiWrap, int scale, bool refine)
{
    CV_Assert(image1.data && image1.type() == CV_8UC3 &&
        image2.data && image2.type() == CV_8UC3 &&
        mask1.data && mask1.type() == CV_8UC1 &&
        mask2.data && mask2.type() == CV_8UC1 &&
        scale > 1);
    cv::Size size = image1.size();
    CV_Assert(image2.size() == size && mask1.size() == size && mask2.size() == size);

    float terminalCost = 10000, badRegionPenalty = 1000; 

    cv::Mat diff;
    calcColorDiff(image1, image2, diff);
    //showFloatMat("diff", diff);

    cv::Mat smallDiff;
    downDiff(diff, smallDiff, scale);
    //showFloatMat("small diff", smallDiff);

    cv::Mat smallMask1, smallMask2, smallMask1Margin, smallMask2Margin;
    downMaskAugment(mask1, smallMask1, scale);
    downMaskAugment(mask2, smallMask2, scale);
    downMaskShrink(mask1, smallMask1Margin, scale);
    downMaskShrink(mask2, smallMask2Margin, scale);
    cv::subtract(smallMask1, smallMask1Margin, smallMask1Margin);
    cv::subtract(smallMask2, smallMask2Margin, smallMask2Margin);
    //cv::imshow("small mask1 before", smallMask1);
    //cv::imshow("small mask2 before", smallMask2);
    //cv::waitKey(0);
    //printf("findSeam small begin, union = %d\n", cv::countNonZero(smallMask1 | smallMask2));
#if SHOW_SEAM
    SeamVisualizer vis1(smallMask1, smallMask2, smallDiff);
    vis1.show("small scale diff without seam");
#endif
    findSeam(smallDiff, smallMask1, smallMask2, terminalCost, badRegionPenalty, horiWrap);
#if SHOW_SEAM
    vis1.drawSeam(smallMask1, smallMask2);
    vis1.show("small scale diff with seam");
#endif
    //printf("findSeam small scale finished, "
    //    "smallMask1 = %d, smallMask2 = %d, sum = %d\n",
    //    cv::countNonZero(smallMask1), cv::countNonZero(smallMask2), cv::countNonZero(smallMask1 | smallMask2));
    //cv::imshow("small mask1 after", smallMask1);
    //cv::imshow("small mask2 after", smallMask2);
    //cv::waitKey(0);

    //cv::Mat origMaskUnion = mask1 | mask2;
    //printf("orig scale mask, union = %d\n", cv::countNonZero(origMaskUnion));    
    cv::Mat bigMask1, bigMask2;
    smallMask1 |= smallMask2Margin;
    smallMask2 |= smallMask1Margin;
    upMask(smallMask1, bigMask1, scale, mask1.size());
    upMask(smallMask2, bigMask2, scale, mask2.size());
    if (!refine)
    {
        //printf("union mask non zero, before = %d, ", cv::countNonZero(mask1 | mask2));
        mask1 &= bigMask1;
        mask2 &= bigMask2;
        separateMask(mask1, mask2, mask1 & mask2);
        //printf("after = %d\n", cv::countNonZero(mask1 | mask2));
        //cv::imshow("mask1 & mask2", mask1 & mask2);
        return;
    }
    bigMask1 &= mask1;
    bigMask2 &= mask2;
    // *************
    // * IMPORTANT *
    // *************
    // To this point, origMask1 and origMask2 may have overlap pixels!!!

    //cv::imshow("big mask1 & big mask2", bigMask1 & bigMask2);
    //long long int beg = cv::getTickCount();
    //separateMask(bigMask1, bigMask2, bigMask1 & bigMask2);
    //long long int end = cv::getTickCount();
    //printf("time elapse = %f\n", (end - beg) / cv::getTickFrequency());
    //cv::imshow("big mask1 & big mask2", bigMask1 & bigMask2);
    //printf("union num pixels, orig = %d, proc = %d\n", 
    //    cv::countNonZero(mask1 | mask2), cv::countNonZero(bigMask1 | bigMask2));

    //cv::Mat procMaskUnion = mask1 | mask2;
    //printf("after merge big mask, mask1 = %d, mask2 = %d, union = %d\n",
    //    cv::countNonZero(mask1), cv::countNonZero(mask2), cv::countNonZero(procMaskUnion));
    //cv::imshow("mask1", mask1);
    //cv::imshow("mask2", mask2);
    //cv::imshow("residual", origMaskUnion != procMaskUnion);
    //cv::waitKey(0);
    
    smallDiff.release();
    smallMask1.release();
    smallMask2.release();
    
    cv::Mat kern = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(scale * 4 + 1, scale * 4 + 1));
    cv::dilate(bigMask1, bigMask1, kern);
    cv::dilate(bigMask2, bigMask2, kern);
    mask1 &= bigMask1;
    mask2 &= bigMask2;
    cv::Mat intersect = bigMask1 & bigMask2;
    //cv::imshow("intersect", intersect);
    //cv::waitKey(0);
    //return;

    cv::dilate(intersect, intersect, kern);
    cv::Rect roi = getNonZeroBoundingRect(intersect);
    cv::Mat intersectROI(intersect, roi);
    cv::Mat diffROI(diff, roi);
    cv::Mat mask1ROI(mask1, roi);
    cv::Mat mask2ROI(mask2, roi);
    cv::Mat indexROI;
    int numIndex = labelIndex(intersectROI, indexROI);
#if SHOW_SEAM
    SeamVisualizer vis2(mask1ROI, mask2ROI, diffROI);
    vis2.show("large scale diff without seam");
#endif
    findSeamInIndexedROI(diffROI, mask1ROI, mask2ROI, indexROI, numIndex, terminalCost, badRegionPenalty, horiWrap);
#if SHOW_SEAM
    vis2.drawSeam(mask1ROI, mask2ROI);
    vis2.show("large scale diff with seam");
#endif
    //cv::imshow("mask1", mask1);
    //cv::imshow("mask2", mask2);
    //cv::waitKey(0);
}

static void findSeamLeftRightWrap(const cv::Mat& diff, cv::Mat& mask1, cv::Mat& mask2, 
    int zeroBegInc, int zeroEndExc, float terminalCost, float badRegionPenalty)
{
    CV_Assert(diff.data && diff.type() == CV_32FC1 &&
        mask1.data && mask1.type() == CV_8UC1 &&
        mask2.data && mask2.type() == CV_8UC1 &&
        diff.size() == mask1.size() && diff.size() == mask2.size() &&
        zeroBegInc >=0 && zeroEndExc <= diff.cols && zeroBegInc < zeroEndExc);

    int width = diff.cols, height = diff.rows;
    int rightWidth = width - zeroEndExc;
    int effectWidth = width - (zeroEndExc - zeroBegInc);

    const int vertexCount = height * effectWidth;
    const int edgeCount = (height - 1) * effectWidth + (effectWidth - 1) * height;
    GCGraph<float> graph(vertexCount, edgeCount); 

    // Set terminal weights
    for (int y = 0; y < height; ++y)
    {
        const unsigned char* ptr1 = mask1.ptr<unsigned char>(y);
        const unsigned char* ptr2 = mask2.ptr<unsigned char>(y);
        for (int x = zeroEndExc; x < width; ++x)
        {
            int v = graph.addVtx();
            graph.addTermWeights(v, ptr1[x] ? terminalCost : 0.f,
                                    ptr2[x] ? terminalCost : 0.f);
        }
        for (int x = 0; x < zeroBegInc; ++x)
        {
            int v = graph.addVtx();
            graph.addTermWeights(v, ptr1[x] ? terminalCost : 0.f,
                                    ptr2[x] ? terminalCost : 0.f);
        }
    }

    // Set regular edge weights
    const float weightEps = 1.f;
    for (int y = 0; y < height; ++y)
    {
        const float* ptrDiff = diff.ptr<float>(y);
        const unsigned char* ptrMask1 = mask1.ptr<unsigned char>(y);
        const unsigned char* ptrMask2 = mask2.ptr<unsigned char>(y);
        const float* ptrDiffNext = 0;
        const unsigned char* ptrMask1Next = 0;
        const unsigned char* ptrMask2Next = 0;
        if (y < height - 1)
        {
            ptrDiffNext = diff.ptr<float>(y + 1);
            ptrMask1Next = mask1.ptr<unsigned char>(y + 1);
            ptrMask2Next = mask2.ptr<unsigned char>(y + 1);
        }
        for (int x = zeroEndExc; x < width; ++x)
        {
            int v = y * effectWidth + x - zeroEndExc;
            if (x < width - 1)
            {
                float weight = ptrDiff[x] + ptrDiff[x + 1] + weightEps;
                if (!ptrMask1[x] || !ptrMask1[x + 1] ||
                    !ptrMask2[x] || !ptrMask2[x + 1])
                    weight += badRegionPenalty;
                graph.addEdges(v, v + 1, weight, weight);
            }
            else
            {
                float weight = ptrDiff[x] + ptrDiff[x - width + 1] + weightEps;
                if (!ptrMask1[x] || !ptrMask1[x - width + 1] ||
                    !ptrMask2[x] || !ptrMask2[x - width + 1])
                    weight += badRegionPenalty;
                graph.addEdges(v, v + 1, weight, weight);
            }
            if (y < height - 1)
            {
                float weight = ptrDiff[x] + ptrDiffNext[x] + weightEps;
                if (!ptrMask1[x] || !ptrMask1Next[x] ||
                    !ptrMask2[x] || !ptrMask2Next[x])
                    weight += badRegionPenalty;
                graph.addEdges(v, v + effectWidth, weight, weight);
            }
        }
        for (int x = 0; x < zeroBegInc; ++x)
        {
            int v = y * effectWidth + x + rightWidth;
            if (x < zeroBegInc - 1)
            {
                float weight = ptrDiff[x] + ptrDiff[x + 1] + weightEps;
                if (!ptrMask1[x] || !ptrMask1[x + 1] ||
                    !ptrMask2[x] || !ptrMask2[x + 1])
                    weight += badRegionPenalty;
                graph.addEdges(v, v + 1, weight, weight);
            }
            if (y < height - 1)
            {
                float weight = ptrDiff[x] + ptrDiffNext[x] + weightEps;
                if (!ptrMask1[x] || !ptrMask1Next[x] ||
                    !ptrMask2[x] || !ptrMask2Next[x])
                    weight += badRegionPenalty;
                graph.addEdges(v, v + effectWidth, weight, weight);
            }
        }
    }

    graph.maxFlow();

    for (int y = 0; y < height; ++y)
    {
        unsigned char* ptrMask1 = mask1.ptr<unsigned char>(y);
        unsigned char* ptrMask2 = mask2.ptr<unsigned char>(y);
        for (int x = zeroEndExc; x < width; ++x)
        {
            if (graph.inSourceSegment(y * effectWidth + x - zeroEndExc))
            {
                if (ptrMask1[x])
                    ptrMask2[x] = 0;
            }
            else
            {
                if (ptrMask2[x])
                    ptrMask1[x] = 0;
            }
        }
        for (int x = 0; x < zeroBegInc; ++x)
        {
            if (graph.inSourceSegment(y * effectWidth + x + rightWidth))
            {
                if (ptrMask1[x])
                    ptrMask2[x] = 0;
            }
            else
            {
                if (ptrMask2[x])
                    ptrMask1[x] = 0;
            }
        }
    }
}

void findSeamLeftRightWrap(const cv::Mat& image1, const cv::Mat& image2,
    cv::Mat& mask1, cv::Mat& mask2, int zeroBegInc, int zeroEndExc)
{
    CV_Assert(image1.data && image1.type() == CV_8UC3 &&
        image2.data && image2.type() == CV_8UC3 &&
        mask1.data && mask1.type() == CV_8UC1 &&
        mask2.data && mask2.type() == CV_8UC1);
    cv::Size size = image1.size();
    CV_Assert(image2.size() == size && mask1.size() == size && mask2.size() == size);

    cv::Mat diff;
    calcColorDiff(image1, image2, diff);
    //SeamVisualizer vis(mask1, mask2, diff);
    //vis.show("diff without seam");
    findSeamLeftRightWrap(diff, mask1, mask2, zeroBegInc, zeroEndExc, 10000, 1000);
    //vis.drawSeam(mask1, mask2);
    //vis.show("diff with seam");
}

void findSeams(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, 
    std::vector<cv::Mat>& resultMasks, int pad, int scale, double ratio, bool refine)
{
    CV_Assert(checkSize(images, masks) && checkType(images, CV_8UC3) && checkType(masks, CV_8UC1));

    if (pad < 0 || pad > 32) pad = 8;
    if (scale < 0 || scale > 32) scale = 8;

    int numImages = images.size();
    resultMasks.resize(numImages);
    for (int i = 0; i < numImages; i++)
        masks[i].copyTo(resultMasks[i]);
    cv::Mat image = images[0].clone();
    cv::Mat mask = resultMasks[0].clone();
    cv::Mat intersect;
    cv::Rect imageRect(0, 0, images[0].cols, images[0].rows);
    for (int i = 1; i < numImages; i++)
    {
        intersect = mask & resultMasks[i];
        int numNonZero = cv::countNonZero(intersect);
        if (numNonZero)
        {
            cv::Rect boundingRect = getNonZeroBoundingRect(intersect);
            boundingRect = padRect(boundingRect, 8) & imageRect;

            cv::Mat baseMaskROI = mask(boundingRect);
            cv::Mat currMaskROI = resultMasks[i](boundingRect);
            cv::Mat baseImageROI = image(boundingRect);
            cv::Mat currImageROI = images[i](boundingRect);
            //cv::imshow("base mask roi before", baseMaskROI);
            //cv::imshow("curr mask roi before", currMaskROI);
            bool horiWrap = boundingRect.width == imageRect.width;
            if (scale == 1)
            {
                if (numNonZero < ratio * boundingRect.area())
                    findSeamInROI(baseImageROI, currImageROI, intersect(boundingRect), baseMaskROI, currMaskROI, horiWrap);
                else
                    findSeam(baseImageROI, currImageROI, baseMaskROI, currMaskROI, horiWrap);
            }
            else
                findSeamScaleDown(baseImageROI, currImageROI, baseMaskROI, currMaskROI, horiWrap, scale, refine);
            
            //cv::imshow("base mask roi after", baseMaskROI);
            //cv::imshow("curr mask roi after", currMaskROI);
        }

        mask |= resultMasks[i];
        images[i].copyTo(image, resultMasks[i]);
        for (int j = 0; j < i; j++)
            resultMasks[j].setTo(0, resultMasks[i]);
        //cv::imshow("image", image);
        //cv::waitKey(0);
    }
}