#include "ConvertCoordinate.h"
#include "Rotation.h"

inline int clamp(int val, int low, int high)
{
    return val < low ? low : (val > high ? high : val);
}

void mapNearestNeighbor(const cv::Mat& src, cv::Mat& dst, const cv::Matx33d& rot)
{
    cv::Matx33d invRot = rot.t();
    int rows = src.rows, cols = src.cols;
    int rowsMinus1 = rows - 1, colsMinus1 = cols - 1;
    double halfWidth = cols * 0.5, halfHeight = rows * 0.5;
    dst.create(rows, cols, CV_8UC3);
    dst.setTo(0);
    //double minx = FLT_MAX, maxx = -FLT_MAX, miny = FLT_MAX, maxy = -FLT_MAX;
    for (int i = 0; i < rows; i++)
    {
        cv::Vec3b* ptrDstRow = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < cols; j++)
        {
            cv::Point2d srcPoint = findRotateEquiRectangularSrc(cv::Point(j, i), halfWidth, halfHeight, invRot);
            //minx = std::min(minx, srcPoint.x);
            //maxx = std::max(maxx, srcPoint.x);
            //miny = std::min(miny, srcPoint.y);
            //maxy = std::max(maxy, srcPoint.y);
            int x = srcPoint.x, y = srcPoint.y;
            ptrDstRow[j] = src.at<cv::Vec3b>(y, x);
        }
    }
    //printf("map, minx = %f, maxx = %f, miny = %f, maxy = %f\n", minx, maxx, miny, maxy);
}

class MapNNLoop : public cv::ParallelLoopBody
{
public:
    MapNNLoop(const cv::Mat& src_, cv::Mat& dst_, const cv::Matx33d& rot_)
        : src(src_), dst(dst_)
    {
        invRot = rot_.t();
        rows = src.rows, cols = src.cols;
        rowsMinus1 = rows - 1, colsMinus1 = cols - 1;
        halfWidth = cols * 0.5, halfHeight = rows * 0.5;
        dst.setTo(0);
    }

    virtual ~MapNNLoop() {}

    virtual void operator()(const cv::Range& r) const
    {
        for (int i = r.start; i < r.end; i++)
        {
            cv::Vec3b* ptrDstRow = dst.ptr<cv::Vec3b>(i);
            for (int j = 0; j < cols; j++)
            {
                cv::Point2d srcPoint = findRotateEquiRectangularSrc(cv::Point(j, i), halfWidth, halfHeight, invRot);
                int x = srcPoint.x, y = srcPoint.y;
                ptrDstRow[j] = src.at<cv::Vec3b>(y, x);
            }
        }
    }

    const cv::Mat& src;
    cv::Mat& dst;
    cv::Matx33d invRot;
    int rows, cols, rowsMinus1, colsMinus1;
    double halfWidth, halfHeight;
};

void mapNearestNeighborParallel(const cv::Mat& src, cv::Mat& dst, const cv::Matx33d& rot)
{
    dst.create(src.size(), src.type());
    MapNNLoop loop(src, dst, rot);
    cv::parallel_for_(cv::Range(0, src.rows), loop, src.total() / (double)(1 << 16));
}

static const int BILINEAR_INTER_SHIFT = 10;
static const int BILINEAR_INTER_BACK_SHIFT = BILINEAR_INTER_SHIFT * 2;
static const int BILINEAR_UNIT = 1 << BILINEAR_INTER_SHIFT;
void mapBilinear(const cv::Mat& src, cv::Mat& dst, const cv::Matx33d& rot)
{
    cv::Matx33d invRot = rot.t();
    int rows = src.rows, cols = src.cols;
    int rowsMinus1 = rows - 1, colsMinus1 = cols - 1;
    double halfWidth = cols * 0.5, halfHeight = rows * 0.5;
    dst.create(rows, cols, CV_8UC3);
    dst.setTo(0);
    //double minx = FLT_MAX, maxx = -FLT_MAX, miny = FLT_MAX, maxy = -FLT_MAX;
    for (int i = 0; i < rows; i++)
    {
        unsigned char* ptrDstRow = dst.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            cv::Point2d srcPoint = findRotateEquiRectangularSrc(cv::Point(j, i), halfWidth, halfHeight, invRot);
            //minx = std::min(minx, srcPoint.x);
            //maxx = std::max(maxx, srcPoint.x);
            //miny = std::min(miny, srcPoint.y);
            //maxy = std::max(maxy, srcPoint.y);
            int x0 = cvFloor(srcPoint.x), y0 = cvFloor(srcPoint.y);
            int x1 = x0 + 1, y1 = y0 + 1;
            int deltax0 = (srcPoint.x - x0) * BILINEAR_UNIT, deltax1 = BILINEAR_UNIT - deltax0;
            int deltay0 = (srcPoint.y - y0) * BILINEAR_UNIT, deltay1 = BILINEAR_UNIT - deltay0;
            if (x0 < 0) x0 = colsMinus1;
            if (x1 > colsMinus1) x1 = 0;
            if (y0 < 0) y0 = 0;
            if (y1 > rowsMinus1) y1 = rowsMinus1;
            const unsigned char* ptrSrcRow;
            ptrSrcRow = src.ptr<unsigned char>(y0);
            int b = 0, g = 0, r = 0, w = 0;
            w = deltax1 * deltay1;
            b += ptrSrcRow[x0 * 3] * w;
            g += ptrSrcRow[x0 * 3 + 1] * w;
            r += ptrSrcRow[x0 * 3 + 2] * w;
            w = deltax0 * deltay1;
            b += ptrSrcRow[x1 * 3] * w;
            g += ptrSrcRow[x1 * 3 + 1] * w;
            r += ptrSrcRow[x1 * 3 + 2] * w;
            ptrSrcRow = src.ptr<unsigned char>(y1);
            w = deltax1 * deltay0;
            b += ptrSrcRow[x0 * 3] * w;
            g += ptrSrcRow[x0 * 3 + 1] * w;
            r += ptrSrcRow[x0 * 3 + 2] * w;
            w = deltax0 * deltay0;
            b += ptrSrcRow[x1 * 3] * w;
            g += ptrSrcRow[x1 * 3 + 1] * w;
            r += ptrSrcRow[x1 * 3 + 2] * w;
            ptrDstRow[0] = b >> BILINEAR_INTER_BACK_SHIFT;
            ptrDstRow[1] = g >> BILINEAR_INTER_BACK_SHIFT;
            ptrDstRow[2] = r >> BILINEAR_INTER_BACK_SHIFT;
            ptrDstRow += 3;
        }
    }
    //printf("map, minx = %f, maxx = %f, miny = %f, maxy = %f\n", minx, maxx, miny, maxy);
}

void mapBilinear(const cv::Mat& src, cv::Mat& dst, const cv::Point3d& trans)
{
    cv::Point3d negTrans = -trans;
    int rows = src.rows, cols = src.cols;
    int rowsMinus1 = rows - 1, colsMinus1 = cols - 1;
    double halfWidth = cols * 0.5, halfHeight = rows * 0.5;
    dst.create(rows, cols, CV_8UC3);
    dst.setTo(0);
    double minx = FLT_MAX, maxx = -FLT_MAX, miny = FLT_MAX, maxy = -FLT_MAX;
    for (int i = 0; i < rows; i++)
    {
        unsigned char* ptrDstRow = dst.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            cv::Point2d srcPoint = findTransEquiRectangularSrc(cv::Point(j, i), halfWidth, halfHeight, negTrans);
            minx = std::min(minx, srcPoint.x);
            maxx = std::max(maxx, srcPoint.x);
            miny = std::min(miny, srcPoint.y);
            maxy = std::max(maxy, srcPoint.y);
            int x0 = cvFloor(srcPoint.x), y0 = cvFloor(srcPoint.y);
            int x1 = x0 + 1, y1 = y0 + 1;
            int deltax0 = (srcPoint.x - x0) * BILINEAR_UNIT, deltax1 = BILINEAR_UNIT - deltax0;
            int deltay0 = (srcPoint.y - y0) * BILINEAR_UNIT, deltay1 = BILINEAR_UNIT - deltay0;
            if (x0 < 0) x0 = colsMinus1;
            if (x1 > colsMinus1) x1 = 0;
            if (y0 < 0) y0 = 0;
            if (y1 > rowsMinus1) y1 = rowsMinus1;
            const unsigned char* ptrSrcRow;
            ptrSrcRow = src.ptr<unsigned char>(y0);
            int b = 0, g = 0, r = 0, w = 0;
            w = deltax1 * deltay1;
            b += ptrSrcRow[x0 * 3] * w;
            g += ptrSrcRow[x0 * 3 + 1] * w;
            r += ptrSrcRow[x0 * 3 + 2] * w;
            w = deltax0 * deltay1;
            b += ptrSrcRow[x1 * 3] * w;
            g += ptrSrcRow[x1 * 3 + 1] * w;
            r += ptrSrcRow[x1 * 3 + 2] * w;
            ptrSrcRow = src.ptr<unsigned char>(y1);
            w = deltax1 * deltay0;
            b += ptrSrcRow[x0 * 3] * w;
            g += ptrSrcRow[x0 * 3 + 1] * w;
            r += ptrSrcRow[x0 * 3 + 2] * w;
            w = deltax0 * deltay0;
            b += ptrSrcRow[x1 * 3] * w;
            g += ptrSrcRow[x1 * 3 + 1] * w;
            r += ptrSrcRow[x1 * 3 + 2] * w;
            ptrDstRow[0] = b >> BILINEAR_INTER_BACK_SHIFT;
            ptrDstRow[1] = g >> BILINEAR_INTER_BACK_SHIFT;
            ptrDstRow[2] = r >> BILINEAR_INTER_BACK_SHIFT;
            ptrDstRow += 3;
        }
    }
    printf("map, minx = %f, maxx = %f, miny = %f, maxy = %f\n", minx, maxx, miny, maxy);
}

class MapBilinearLoop : public cv::ParallelLoopBody
{
public:
    MapBilinearLoop(const cv::Mat& src_, cv::Mat& dst_, const cv::Matx33d& rot_)
        : src(src_), dst(dst_)
    {
        invRot = rot_.t();
        rows = src.rows, cols = src.cols;
        rowsMinus1 = rows - 1, colsMinus1 = cols - 1;
        halfWidth = cols * 0.5, halfHeight = rows * 0.5;
        dst.setTo(0);
    }

    virtual ~MapBilinearLoop() {}

    virtual void operator()(const cv::Range& r) const
    {
        for (int i = r.start; i < r.end; i++)
        {
            unsigned char* ptrDstRow = dst.ptr<unsigned char>(i);
            for (int j = 0; j < cols; j++)
            {
                cv::Point2d srcPoint = findRotateEquiRectangularSrc(cv::Point(j, i), halfWidth, halfHeight, invRot);
                int x0 = cvFloor(srcPoint.x), y0 = cvFloor(srcPoint.y);
                int x1 = x0 + 1, y1 = y0 + 1;
                int deltax0 = (srcPoint.x - x0) * BILINEAR_UNIT, deltax1 = BILINEAR_UNIT - deltax0;
                int deltay0 = (srcPoint.y - y0) * BILINEAR_UNIT, deltay1 = BILINEAR_UNIT - deltay0;
                if (x0 < 0) x0 = colsMinus1;
                if (x1 > colsMinus1) x1 = 0;
                if (y0 < 0) y0 = 0;
                if (y1 > rowsMinus1) y1 = rowsMinus1;
                const unsigned char* ptrSrcRow;
                ptrSrcRow = src.ptr<unsigned char>(y0);
                int b = 0, g = 0, r = 0, w = 0;
                w = deltax1 * deltay1;
                b += ptrSrcRow[x0 * 3] * w;
                g += ptrSrcRow[x0 * 3 + 1] * w;
                r += ptrSrcRow[x0 * 3 + 2] * w;
                w = deltax0 * deltay1;
                b += ptrSrcRow[x1 * 3] * w;
                g += ptrSrcRow[x1 * 3 + 1] * w;
                r += ptrSrcRow[x1 * 3 + 2] * w;
                ptrSrcRow = src.ptr<unsigned char>(y1);
                w = deltax1 * deltay0;
                b += ptrSrcRow[x0 * 3] * w;
                g += ptrSrcRow[x0 * 3 + 1] * w;
                r += ptrSrcRow[x0 * 3 + 2] * w;
                w = deltax0 * deltay0;
                b += ptrSrcRow[x1 * 3] * w;
                g += ptrSrcRow[x1 * 3 + 1] * w;
                r += ptrSrcRow[x1 * 3 + 2] * w;
                ptrDstRow[0] = b >> BILINEAR_INTER_BACK_SHIFT;
                ptrDstRow[1] = g >> BILINEAR_INTER_BACK_SHIFT;
                ptrDstRow[2] = r >> BILINEAR_INTER_BACK_SHIFT;
                ptrDstRow += 3;
            }
        }
    }

    const cv::Mat& src;
    cv::Mat& dst;
    cv::Matx33d invRot;
    int rows, cols, rowsMinus1, colsMinus1;
    double halfWidth, halfHeight;
};

void mapBilinearParallel(const cv::Mat& src, cv::Mat& dst, const cv::Matx33d& rot)
{
    dst.create(src.size(), src.type());
    MapBilinearLoop loop(src, dst, rot);
    cv::parallel_for_(cv::Range(0, src.rows), loop, src.total() / (double)(1 << 16));
}

void mapNearestNeighbor(const cv::Mat& src, cv::Mat& dst, const cv::Size& dstSize,
    double dstHFov, double srcHoriAngleOffset, double srcVertAngleOffset, bool isRectLinear)
{
    int rows = src.rows, cols = src.cols;
    dst.create(dstSize, CV_8UC3);
    dst.setTo(0);
    if (isRectLinear)
    {
        RectLinearBackToEquiRect2 transform(cols, rows, dstSize.width, dstSize.height,
            dstHFov, srcHoriAngleOffset, srcVertAngleOffset);
        for (int i = 0; i < dstSize.height; i++)
        {
            cv::Vec3b* ptrDstRow = dst.ptr<cv::Vec3b>(i);
            for (int j = 0; j < dstSize.width; j++)
            {
                cv::Point2d srcPoint = transform(j, i);
                int x = cvFloor(srcPoint.x), y = cvFloor(srcPoint.y);
                if (x >= 0 && x < cols && y >= 0 && y < rows)
                    ptrDstRow[j] = src.at<cv::Vec3b>(y, x);
                //else if (x != -1 && y != -1)
                //    printf("(%d, %d)\n", x, y);
            }
        }
    }
    else
    {
        FishEyeBackToEquiRect transform(cols, rows, dstSize.width, dstSize.height,
            dstHFov, srcHoriAngleOffset, srcVertAngleOffset);
        for (int i = 0; i < dstSize.height; i++)
        {
            cv::Vec3b* ptrDstRow = dst.ptr<cv::Vec3b>(i);
            for (int j = 0; j < dstSize.width; j++)
            {
                cv::Point2d srcPoint = transform(j, i);
                int x = cvFloor(srcPoint.x), y = cvFloor(srcPoint.y);
                if (x >= 0 && x < cols && y >= 0 && y < rows)
                    ptrDstRow[j] = src.at<cv::Vec3b>(y, x);
                //else 
                //    printf("(%d, %d)\n", x, y);
            }
        }
    }
}