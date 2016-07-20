#include "ZReproject.h"

inline int weightedSum(const unsigned char rgb[4], const double w[4])
{
    int res = (int)(rgb[0] * w[0] + rgb[1] * w[1] + rgb[2] * w[2] + rgb[3] * w[3] + 0.5);
    res = res > 255 ? 255 : res;
    res = res < 0 ? 0 : res;
    return res;
}

inline double weightedSumTo64F(const unsigned char src[4], const double w[4])
{
    return src[0] * w[0] + src[1] * w[1] + src[2] * w[2] + src[3] * w[3];
}

inline double weightedSumTo32S(const double src[4], const double w[4])
{
    int res = src[0] * w[0] + src[1] * w[1] + src[2] * w[2] + src[3] * w[3];
    res = res > 255 ? 255 : res;
    res = res < 0 ? 0 : res;
    return res;
}

template<typename DstElemType>
inline DstElemType weightedSum(const double src[4], const double w[4])
{
    return cv::saturate_cast<DstElemType>(src[0] * w[0] + src[1] * w[1] + src[2] * w[2] + src[3] * w[3]);
}

inline void calcWeights(double deta, double weight[4])
{
    //double deta2 = deta * deta;
    //double deta2x2 = deta2 * 2;
    //double deta3 = deta2 * deta;
    //weight[3] = -deta2 + deta3;
    //weight[2] = deta + deta2 - deta3;
    //weight[1] = 1.0 - deta2x2 + deta3;
    //weight[0] = -deta + deta2x2 - deta3;
    weight[3] = (deta * deta * (-1 + deta));
    weight[2] = (deta * (1.0 + deta * (1.0 - deta)));
    weight[1] = (1.0 + deta * deta * (-2.0 + deta));
    weight[0] = (-deta * (1.0 + deta * (-2.0 + deta)));
}

template<typename DstElemType, int NumChannels>
inline void bicubicResampling(int width, int height, int step, const unsigned char* src,
    double x, double y, DstElemType dst[NumChannels])
{
    int x2 = (int)x;
    int y2 = (int)y;
    int nx[4];
    int ny[4];
    unsigned char rgb1[4];
    double rgb2[4];

    for (int i = 0; i < 4; ++i)
    {
        nx[i] = (x2 - 1 + i);
        ny[i] = (y2 - 1 + i);
        if (nx[i] < 0)
        {
            nx[i] = 0;
        }
        if (nx[i] > width - 1)
        {
            nx[i] = width - 1;
        }
        if (ny[i] < 0)
        {
            ny[i] = 0;
        }
        if (ny[i] > height - 1)
        {
            ny[i] = height - 1;
        }
    }

    double u = (x - nx[1]);
    double v = (y - ny[1]);
    //u,v vertical while /100 horizontal
    double tweight1[4], tweight2[4];
    calcWeights(u, tweight1);//weight
    calcWeights(v, tweight2);//weight

    for (int k = 0; k < NumChannels; ++k)
    {
        for (int j = 0; j < 4; j++)
        {
            // 按行去每个通道
            for (int i = 0; i < 4; i++)
            {
                rgb1[i] = src[ny[j] * step + nx[i] * NumChannels + k];
            }
            //4*4区域的三个通道
            rgb2[j] = weightedSumTo64F(rgb1, tweight1);
        }
        dst[k] = weightedSum<DstElemType>(rgb2, tweight2);
    }
}

template<typename DstElemType, int NumChannels>
inline void bilinearResampling(int width, int height, int step, const unsigned char* src,
    double x, double y, DstElemType dst[NumChannels])
{
    int x0 = x, y0 = y, x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0) x0 = 0;
    if (x1 > width - 1) x1 = width - 1;
    if (y0 < 0) y0 = 0;
    if (y1 > height - 1) y1 = height - 1;
    double wx0 = x - x0, wx1 = 1 - wx0;
    double wy0 = y - y0, wy1 = 1 - wy0;
    double w00 = wx1 * wy1, w01 = wx0 * wy1;
    double w10 = wx1 * wy0, w11 = wx0 * wy0;

    double res[NumChannels] = { 0 };
    const unsigned char* ptr;
    ptr = src + step * y0 + x0 * NumChannels;
    for (int i = 0; i < NumChannels; i++)
        res[i] += *(ptr++) * w00;
    for (int i = 0; i < NumChannels; i++)
        res[i] += *(ptr++) * w01;
    ptr = src + step * y1 + x0 * NumChannels;
    for (int i = 0; i < NumChannels; i++)
        res[i] += *(ptr++) * w10;
    for (int i = 0; i < NumChannels; i++)
        res[i] += *(ptr++) * w11;

    for (int i = 0; i < NumChannels; i++)
        dst[i] = cv::saturate_cast<DstElemType>(res[i]);
}

void getReprojectMapAndMask(const PhotoParam& param,
    const cv::Size& srcSize, const cv::Size& dstSize, cv::Mat& map, cv::Mat& mask)
{
    int dstWidth = dstSize.width, dstHeight = dstSize.height;
    int srcWidth = srcSize.width, srcHeight = srcSize.height;
    if (param.imageType != 3 && param.imageType != 2)
    {
        Remap remap;
        remap.init(param, dstWidth, dstHeight, srcWidth, srcHeight);
        map.create(dstHeight, dstWidth, CV_64FC2);
        mask.create(dstHeight, dstWidth, CV_8UC1);
        mask.setTo(0);
        for (int h = 0; h < dstHeight; h++)
        {
            cv::Point2d* ptrMap = map.ptr<cv::Point2d>(h);
            unsigned char* ptrMask = mask.ptr<unsigned char>(h);
            for (int w = 0; w < dstWidth; w++)
            {
                double sx, sy;
                remap.remapImage(sx, sy, w, h);
                ptrMap[w].x = sx;
                ptrMap[w].y = sy;
                if (sx >= 0 && sy >= 0 && sx < srcWidth && sy < srcHeight)
                    ptrMask[w] = 255;
            }
        }
    }
    else
    {
        Remap remap;
        remap.init(param, dstWidth, dstHeight, srcWidth, srcHeight);
        map.create(dstHeight, dstWidth, CV_64FC2);
        mask.create(dstHeight, dstWidth, CV_8UC1);
        mask.setTo(0);
        double centx, centy, sqrDist;
        if (param.circleR > 0)
        {
            centx = param.circleX;
            centy = param.circleY;
            sqrDist = param.circleR * param.circleR;
        }
        else
        {
            centx = param.cropX + param.cropWidth / 2;
            centy = param.cropY + param.cropHeight / 2;
            sqrDist = param.cropWidth > param.cropHeight ?
                param.cropWidth * param.cropWidth * 0.25 :
                param.cropHeight * param.cropHeight * 0.25;
        }
        for (int h = 0; h < dstHeight; h++)
        {
            cv::Point2d* ptrMap = map.ptr<cv::Point2d>(h);
            unsigned char* ptrMask = mask.ptr<unsigned char>(h);
            for (int w = 0; w < dstWidth; w++)
            {
                double sx, sy;
                remap.remapImage(sx, sy, w, h);
                if (sx >= 0 && sy >= 0 && sx < srcWidth && sy < srcHeight &&
                    (sx - centx) * (sx - centx) + (sy - centy) * (sy - centy) < sqrDist)
                {
                    ptrMap[w].x = sx;
                    ptrMap[w].y = sy;
                    ptrMask[w] = 255;
                }
                else
                {
                    // IMPORTANT NOTE!!!
                    // If the reproject params are later used to get reprojected images for multiband blending,
                    // the following two lines of code are necessary. In the reproject code, if the row and column 
                    // indexes are outside the bound of the src image, no reproject sampling is done.
                    // Multiband blending requires that src image pixle be zero if the corresponding mask image 
                    // position have zero value. Failing to meet the requirement will make the pyramid down operation
                    // produce wrong result.
                    ptrMap[w].x = -1;
                    ptrMap[w].y = -1;
                }
            }
        }
    }
}

void getReprojectMapsAndMasks(const std::vector<PhotoParam>& params,
    const cv::Size& srcSize, const cv::Size& dstSize, std::vector<cv::Mat>& maps, std::vector<cv::Mat>& masks)
{
    int num = params.size();
    maps.resize(num);
    masks.resize(num);
    for (int i = 0; i < num; i++)
        getReprojectMapAndMask(params[i], srcSize, dstSize, maps[i], masks[i]);
}

void getReprojectMap32FAndMask(const PhotoParam& param, const cv::Size& srcSize, const cv::Size& dstSize,
    cv::Mat& xmap, cv::Mat& ymap, cv::Mat& mask)
{
    cv::Mat map;
    getReprojectMapAndMask(param, srcSize, dstSize, map, mask);
    cv::Mat mapArr[2];
    cv::split(map, mapArr);
    mapArr[0].convertTo(xmap, CV_32F);
    mapArr[1].convertTo(ymap, CV_32F);
}

void getReprojectMaps32FAndMasks(const std::vector<PhotoParam>& params, const cv::Size& srcSize, const cv::Size& dstSize,
    std::vector<cv::Mat>& xmaps, std::vector<cv::Mat>& ymaps, std::vector<cv::Mat>& masks)
{
    int num = params.size();
    xmaps.resize(num);
    ymaps.resize(num);
    masks.resize(num);
    for (int i = 0; i < num; i++)
        getReprojectMap32FAndMask(params[i], srcSize, dstSize, xmaps[i], ymaps[i], masks[i]);
}

static const int BILINEAR_INTER_SHIFT = 10;
static const int BILINEAR_INTER_BACK_SHIFT = BILINEAR_INTER_SHIFT * 2;
static const int BILINEAR_UNIT = 1 << BILINEAR_INTER_SHIFT;

void getReprojectMap16SAndWeight32SAndMask(const PhotoParam& param, const cv::Size& srcSize, const cv::Size& dstSize,
    cv::Mat& map, cv::Mat& weight, cv::Mat& mask)
{
    cv::Mat map64F;
    getReprojectMapAndMask(param, srcSize, dstSize, map64F, mask);
    map.create(dstSize, CV_16SC2);
    weight.create(dstSize, CV_32SC4);
    for (int i = 0; i < dstSize.height; i++)
    {
        const double* ptrMap64F = map64F.ptr<double>(i);
        short* ptrMap = map.ptr<short>(i);
        int* ptrWeight = weight.ptr<int>(i);
        for (int j = 0; j < dstSize.width; j++)
        {
            if (ptrMap64F[0] < 0 || ptrMap64F[1] < 0)
            {
                ptrMap[0] = -1;
                ptrMap[1] = -1;
                ptrWeight[0] = 0;
                ptrWeight[1] = 0;
                ptrWeight[2] = 0;
                ptrWeight[3] = 0;
            }
            else
            {
                double x = ptrMap64F[0], y = ptrMap64F[1];
                short x0 = x, y0 = y;
                int deltax0 = (x - x0) * BILINEAR_UNIT, deltax1 = BILINEAR_UNIT - deltax0;
                int deltay0 = (y - y0) * BILINEAR_UNIT, deltay1 = BILINEAR_UNIT - deltay0;
                ptrMap[0] = x0;
                ptrMap[1] = y0;
                ptrWeight[0] = deltax1 * deltay1;
                ptrWeight[1] = deltax0 * deltay1;
                ptrWeight[2] = deltax1 * deltay0;
                ptrWeight[3] = deltax0 * deltay0;
            }
            ptrMap64F += 2;
            ptrMap += 2;
            ptrWeight += 4;
        }
    }
}

void getReprojectMaps16SAndWeights32SAndMasks(const std::vector<PhotoParam>& params, const cv::Size& srcSize, const cv::Size& dstSize,
    std::vector<cv::Mat>& maps, std::vector<cv::Mat>& weights, std::vector<cv::Mat>& masks)
{
    int num = params.size();
    maps.resize(num);
    weights.resize(num);
    masks.resize(num);
    for (int i = 0; i < num; i++)
        getReprojectMap16SAndWeight32SAndMask(params[i], srcSize, dstSize, maps[i], weights[i], masks[i]);
}

void reproject(const cv::Mat& src, cv::Mat& dst, cv::Mat& mask, 
    const PhotoParam& param, const cv::Size& dstSize)
{    
    cv::Point max = cv::Point(dstSize.width, dstSize.height);
    cv::Point min = cv::Point(0, 0);

    if (param.imageType != 3 && param.imageType != 2)
    {
        Remap remap;
        remap.init(param, dstSize.width, dstSize.height, src.cols, src.rows);
        dst.create(dstSize, CV_8UC3);
        dst.setTo(0);
        mask.create(dstSize, CV_8UC1);
        mask.setTo(0);

        int srcWidth = src.cols;
        int srcHeight = src.rows;
        int srcStep = src.step;
        const unsigned char* ptrSrc = src.data;
        for (int h = min.y; h < max.y; h++)
        {
            cv::Vec3b* ptrDst = dst.ptr<cv::Vec3b>(h);
            unsigned char* ptrMask = mask.ptr<uchar>(h);
            for (int w = min.x; w < max.x; w++)
            {
                double sx, sy;
                remap.remapImage(sx, sy, w, h);
                if (sx >= 0 && sy >= 0 && sx < srcWidth && sy < srcHeight)
                {
                    uchar dest[3];
                    bicubicResampling<unsigned char, 3>(srcWidth, srcHeight, srcStep, ptrSrc, sx, sy, dest);
                    ptrDst[w][0] = dest[0];
                    ptrDst[w][1] = dest[1];
                    ptrDst[w][2] = dest[2];
                    ptrMask[w] = 255;
                }
            }
        }
    }
    else
    {
        Remap remap;
        remap.init(param, dstSize.width, dstSize.height, src.cols, src.rows);
        dst.create(dstSize, CV_8UC3);
        dst.setTo(0);
        mask.create(dstSize, CV_8UC1);
        mask.setTo(0);

        double centx, centy, sqrDist;
        if (param.circleR > 0)
        {
            centx = param.circleX;
            centy = param.circleY;
            sqrDist = param.circleR * param.circleR;
        }
        else
        {
            centx = param.cropX + param.cropWidth / 2;
            centy = param.cropY + param.cropHeight / 2;
            sqrDist = param.cropWidth > param.cropHeight ?
                param.cropWidth * param.cropWidth * 0.25 :
                param.cropHeight * param.cropHeight * 0.25;
        }
        int srcWidth = src.cols;
        int srcHeight = src.rows;
        int srcStep = src.step;
        const unsigned char* ptrSrc = src.data;
        for (int h = min.y; h < max.y; h++)
        {
            cv::Vec3b* ptrDst = dst.ptr<cv::Vec3b>(h);
            unsigned char* ptrMask = mask.ptr<uchar>(h);
            for (int w = min.x; w < max.x; w++)
            {
                double sx, sy;
                remap.remapImage(sx, sy, w, h);
                if (sx >= 0 && sy >= 0 && sx < srcWidth && sy < srcHeight &&
                    (sx - centx) * (sx - centx) + (sy - centy) * (sy - centy) < sqrDist)
                {
                    uchar dest[3];
                    bicubicResampling<unsigned char, 3>(srcWidth, srcHeight, srcStep, ptrSrc, sx, sy, dest);
                    ptrDst[w][0] = dest[0];
                    ptrDst[w][1] = dest[1];
                    ptrDst[w][2] = dest[2];
                    ptrMask[w] = 255;
                }
            }
        }
    }
}

void reproject(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, std::vector<cv::Mat>& masks,
    const std::vector<PhotoParam>& params, const cv::Size& dstSize)
{
    CV_Assert(src.size() == params.size());
    int num = src.size();
    dst.resize(num);
    masks.resize(num);
    for (int i = 0; i < num; i++)
        reproject(src[i], dst[i], masks[i], params[i], dstSize);
}

void reproject(const cv::Mat& src, cv::Mat& dst, const cv::Mat& map)
{
    int srcWidth = src.cols, srcHeight = src.rows, srcStep = src.step;
    int dstWidth = map.cols, dstHeight = map.rows;
    dst.create(dstHeight, dstWidth, CV_8UC3);
    dst.setTo(0);
    const unsigned char* srcData = src.data;
    for (int h = 0; h < dstHeight; h++)
    {
        const cv::Point2d* ptrSrcPos = map.ptr<cv::Point2d>(h);
        cv::Vec3b* ptrDstRow = dst.ptr<cv::Vec3b>(h);
        for (int w = 0; w < dstWidth; w++)
        {
            cv::Point2d pt = ptrSrcPos[w];
            if (pt.x >= 0 && pt.y >= 0 && pt.x < srcWidth && pt.y < srcHeight)
            {
                uchar dest[3];
                bicubicResampling<unsigned char, 3>(srcWidth, srcHeight, srcStep, srcData, pt.x, pt.y, dest);
                ptrDstRow[w][0] = dest[0];
                ptrDstRow[w][1] = dest[1];
                ptrDstRow[w][2] = dest[2];
            }
        }
    }
}

void reproject(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, const std::vector<cv::Mat>& maps)
{
    CV_Assert(src.size() == maps.size());
    int num = src.size();
    dst.resize(num);
    for (int i = 0; i < num; i++)
        reproject(src[i], dst[i], maps[i]);
}

template<typename DstElemType, int NumChannels>
class ZReprojectLoop : public cv::ParallelLoopBody
{
public:
    ZReprojectLoop(const cv::Mat& src_, cv::Mat& dst_, const cv::Mat& map_)
        : src(src_), dst(dst_), map(map_)
    {
        srcWidth = src.cols, srcHeight = src.rows, srcStep = src.step;
        dstWidth = map.cols, dstHeight = map.rows;
    }

    virtual ~ZReprojectLoop() {}

    virtual void operator()(const cv::Range& r) const
    {
        const unsigned char* srcData = src.data;
        int start = r.start, end = std::min(r.end, dstHeight);
        for (int h = start; h < end; h++)
        {
            const cv::Point2d* ptrSrcPos = map.ptr<cv::Point2d>(h);
            DstElemType* ptrDstRow = dst.ptr<DstElemType>(h);
            for (int w = 0; w < dstWidth; w++)
            {
                cv::Point2d pt = ptrSrcPos[w];
                if (pt.x >= 0 && pt.y >= 0 && pt.x < srcWidth && pt.y < srcHeight)
                {
                    //DstElemType dest[3];
                    //resampling(srcWidth, srcHeight, srcStep, srcData, pt.x, pt.y, dest);
                    bilinearResampling<DstElemType, NumChannels>(srcWidth, srcHeight, srcStep, srcData, pt.x, pt.y, ptrDstRow);
                    //ptrDstRow[w * 3] = dest[0];
                    //ptrDstRow[w * 3 + 1] = dest[1];
                    //ptrDstRow[w * 3 + 2] = dest[2];
                }
                ptrDstRow += NumChannels;
            }
        }
    }

    const cv::Mat& src;
    cv::Mat& dst;
    const cv::Mat& map;
    int srcWidth, srcHeight, srcStep;
    int dstWidth, dstHeight;
};

void reprojectParallel(const cv::Mat& src, cv::Mat& dst, const cv::Mat& map)
{
    //dst.create(map.size(), CV_8UC3);
    //dst.setTo(0);
    //ZReprojectLoop<unsigned char, 3> loop(src, dst, map);
    //cv::parallel_for_(cv::Range(0, dst.rows), loop);

    CV_Assert(src.data && src.depth() == CV_8U);
    int numChannels = src.channels();
    CV_Assert(numChannels > 0 && numChannels <= 4);
    dst.create(map.size(), CV_MAKETYPE(CV_8U, numChannels));
    dst.setTo(0);
    if (numChannels == 1)
    {
        ZReprojectLoop<unsigned char, 1> loop(src, dst, map);
        cv::parallel_for_(cv::Range(0, dst.rows), loop);
    }
    else if (numChannels == 2)
    {
        ZReprojectLoop<unsigned char, 2> loop(src, dst, map);
        cv::parallel_for_(cv::Range(0, dst.rows), loop);
    }
    else if (numChannels == 3)
    {
        ZReprojectLoop<unsigned char, 3> loop(src, dst, map);
        cv::parallel_for_(cv::Range(0, dst.rows), loop);
    }
    else if (numChannels == 4)
    {
        ZReprojectLoop<unsigned char, 4> loop(src, dst, map);
        cv::parallel_for_(cv::Range(0, dst.rows), loop);
    }
}

void reprojectParallel(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, const std::vector<cv::Mat>& maps)
{
    int numImages = src.size();
    dst.resize(numImages);
    for (int i = 0; i < numImages; i++)
        reprojectParallel(src[i], dst[i], maps[i]);
}

void reprojectParallelTo16S(const cv::Mat& src, cv::Mat& dst, const cv::Mat& map)
{
    //dst.create(map.size(), CV_16SC3);
    //dst.setTo(0);
    //ZReprojectLoop<short, 3> loop(src, dst, map);
    //cv::parallel_for_(cv::Range(0, dst.rows), loop);

    CV_Assert(src.data && src.depth() == CV_8U);
    int numChannels = src.channels();
    CV_Assert(numChannels > 0 && numChannels <= 4);
    dst.create(map.size(), CV_MAKETYPE(CV_16S, numChannels));
    dst.setTo(0);
    if (numChannels == 1)
    {
        ZReprojectLoop<short, 1> loop(src, dst, map);
        cv::parallel_for_(cv::Range(0, dst.rows), loop);
    }
    else if (numChannels == 2)
    {
        ZReprojectLoop<short, 2> loop(src, dst, map);
        cv::parallel_for_(cv::Range(0, dst.rows), loop);
    }
    else if (numChannels == 3)
    {
        ZReprojectLoop<short, 3> loop(src, dst, map);
        cv::parallel_for_(cv::Range(0, dst.rows), loop);
    }
    else if (numChannels == 4)
    {
        ZReprojectLoop<short, 4> loop(src, dst, map);
        cv::parallel_for_(cv::Range(0, dst.rows), loop);
    }
}

void reprojectParallelTo16S(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, const std::vector<cv::Mat>& maps)
{
    int numImages = src.size();
    dst.resize(numImages);
    for (int i = 0; i < numImages; i++)
        reprojectParallelTo16S(src[i], dst[i], maps[i]);
}

void reprojectParallelTo32F(const cv::Mat& src, cv::Mat& dst, const cv::Mat& map)
{
    CV_Assert(src.data && src.depth() == CV_8U);
    int numChannels = src.channels();
    CV_Assert(numChannels > 0 && numChannels <= 4);
    dst.create(map.size(), CV_MAKETYPE(CV_32F, numChannels));
    dst.setTo(0);
    if (numChannels == 1)
    {
        ZReprojectLoop<float, 1> loop(src, dst, map);
        cv::parallel_for_(cv::Range(0, dst.rows), loop);
    }
    else if (numChannels == 2)
    {
        ZReprojectLoop<float, 2> loop(src, dst, map);
        cv::parallel_for_(cv::Range(0, dst.rows), loop);
    }
    else if (numChannels == 3)
    {
        ZReprojectLoop<float, 3> loop(src, dst, map);
        cv::parallel_for_(cv::Range(0, dst.rows), loop);
    }
    else if (numChannels == 4)
    {
        ZReprojectLoop<float, 4> loop(src, dst, map);
        cv::parallel_for_(cv::Range(0, dst.rows), loop);
    }
}

void reprojectWeightedAccumulateTo32F(const cv::Mat& src, cv::Mat& dst,
    const cv::Mat& map, const cv::Mat& weight)
{
    CV_Assert(src.data && src.type() == CV_8UC3 && dst.data && dst.type() == CV_32FC3 &&
        map.data && map.type() == CV_64FC2 && weight.data && weight.type() == CV_32FC1 &&
        dst.size() == map.size() && dst.size() == weight.size());

    int srcWidth = src.cols, srcHeight = src.rows, srcStep = src.step;
    int dstWidth = map.cols, dstHeight = map.rows;
    const unsigned char* srcData = src.data;
    for (int h = 0; h < dstHeight; h++)
    {
        const cv::Point2d* ptrSrcPos = map.ptr<cv::Point2d>(h);
        const float* ptrWeight = weight.ptr<float>(h);
        cv::Vec3f* ptrDstRow = dst.ptr<cv::Vec3f>(h);
        for (int w = 0; w < dstWidth; w++)
        {
            cv::Point2d pt = ptrSrcPos[w];
            if (pt.x >= 0 && pt.y >= 0 && pt.x < srcWidth && pt.y < srcHeight)
            {
                uchar dest[3];
                bilinearResampling<unsigned char, 3>(srcWidth, srcHeight, srcStep, srcData, pt.x, pt.y, dest);
                float alpha = ptrWeight[w];
                ptrDstRow[w][0] += dest[0] * alpha;
                ptrDstRow[w][1] += dest[1] * alpha;
                ptrDstRow[w][2] += dest[2] * alpha;
            }
        }
    }
}

class ReprojAccumLoop : public cv::ParallelLoopBody
{
public:
    ReprojAccumLoop(const cv::Mat& src_, cv::Mat& dst_, const cv::Mat& map_, const cv::Mat& weight_)
        : src(src_), dst(dst_), map(map_), weight(weight_)
    {
        srcWidth = src.cols, srcHeight = src.rows, srcStep = src.step;
        dstWidth = map.cols, dstHeight = map.rows;
    }

    virtual ~ReprojAccumLoop() {}

    virtual void operator()(const cv::Range& r) const
    {
        const unsigned char* srcData = src.data;
        int start = r.start, end = std::min(r.end, dstHeight);
        for (int h = start; h < end; h++)
        {
            const cv::Point2d* ptrSrcPos = map.ptr<cv::Point2d>(h);
            const float* ptrWeight = weight.ptr<float>(h);
            cv::Vec3f* ptrDstRow = dst.ptr<cv::Vec3f>(h);
            for (int w = 0; w < dstWidth; w++)
            {
                cv::Point2d pt = ptrSrcPos[w];
                if (pt.x >= 0 && pt.y >= 0 && pt.x < srcWidth && pt.y < srcHeight)
                {
                    uchar dest[3];
                    bilinearResampling<unsigned char, 3>(srcWidth, srcHeight, srcStep, srcData, pt.x, pt.y, dest);
                    float alpha = ptrWeight[w];
                    ptrDstRow[w][0] += dest[0] * alpha;
                    ptrDstRow[w][1] += dest[1] * alpha;
                    ptrDstRow[w][2] += dest[2] * alpha;
                }
            }
        }
    }

    const cv::Mat& src;
    cv::Mat& dst;
    const cv::Mat& map;
    const cv::Mat& weight;
    int srcWidth, srcHeight, srcStep;
    int dstWidth, dstHeight;
};

void reprojectWeightedAccumulateParallelTo32F(const cv::Mat& src, cv::Mat& dst,
    const cv::Mat& map, const cv::Mat& weight)
{
    CV_Assert(src.data && src.type() == CV_8UC3 && dst.data && dst.type() == CV_32FC3 &&
        map.data && map.type() == CV_64FC2 && weight.data && weight.type() == CV_32FC1 &&
        dst.size() == map.size() && dst.size() == weight.size());

    ReprojAccumLoop loop(src, dst, map, weight);
    cv::parallel_for_(cv::Range(0, dst.rows), loop);
}