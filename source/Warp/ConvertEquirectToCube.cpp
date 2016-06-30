#include "ZReproject.h"
#include "MathConstant.h"

static const double D90 = 0.5 * PI;
static const double D270 = 1.5 * PI;
static const double DualPI = 2 * PI;

static double mod(double v, double c)
{
    if (c != 0) {
        if (v >= 0.0) {
            int n = (int)(v / c);
            return v - n * c;
        }
        else {
            int n = (int)(v / c) - 1;
            return v - n * c;
        }
    }
    return v;
}

static void warp(double r, double x, double y, cv::Point2d& res)
{
    double theta = atan2(x, r);
    double phi = atan(cos(theta) * y / r);
    res.x = theta;
    res.y = phi;
}

static void warp1(double r, double x, double y, cv::Point2d& res)
{
    double theta = atan2(x, y);
    double phi = D90 - atan(sqrt(x * x + y * y) / r);
    res.x = theta;
    res.y = phi;
}

static void cubeToSphereAngle(int x, int y, int cubeHeight, cv::Point2d& pd)
{
    x = x * 2 + 1;
    y = y * 2 + 1;
    int h = cubeHeight * 2;

    int position = x / h;
    x = x % h;
    y = y % h;
    double r = cubeHeight;

    switch (position)
    {
    case 0://front
        warp(r, x - r, y - r, pd);
        pd.y = pd.y + D90;
        break;
    case 1://right
        warp(r, x - r, y - r, pd);
        pd.x += D90;
        pd.y += D90;
        break;
    case 2://back
        warp(r, x - r, y - r, pd);
        pd.x += PI;
        pd.y += D90;
        break;
    case 3://left
        warp(r, x - r, y - r, pd);
        pd.x += D270;
        pd.y += D90;
        break;
    case 4://top
        warp1(r, x - r, y - r, pd);
        pd.y = D90 - pd.y;
        break;
    case 5://bottom
        warp1(r, x - r, r - y, pd);
        pd.y += D90;
        break;
    default:
        break;

    }
    pd.x = mod(-pd.x, DualPI);
    pd.y = mod(pd.y, PI);

}

static void getMap(cv::Mat& map, int srcHeight, int dstHeight)
{
    int cubeHeight = dstHeight, cubeWidth = cubeHeight * 6;
    int equiHeight = srcHeight, equiWidth = equiHeight * 2;
    double scale = equiHeight / PI;
    map.create(cubeHeight, cubeWidth, CV_64FC2);
    for (int i = 0; i < cubeHeight; i++)
    {
        double* ptr = map.ptr<double>(i);
        for (int j = 0; j < cubeWidth; j++)
        {
            cv::Point2d pt;
            cubeToSphereAngle(j, i, cubeHeight, pt);
            *(ptr++) = (DualPI - pt.x) * scale; // x
            *(ptr++) = pt.y * scale;            // y
        }
    }
}

void getEquiRectToCubeMap(cv::Mat& dstSrcMap, int equiRectHeight, int cubeHeight, bool ratio6Over1)
{
    CV_Assert(equiRectHeight > 0 && cubeHeight > 0);
    if (ratio6Over1)
        getMap(dstSrcMap, equiRectHeight, cubeHeight);
    else
    {
        cv::Mat map;
        getMap(map, equiRectHeight, cubeHeight);
        int dstHeight = cubeHeight * 2, dstWidth = cubeHeight * 3;
        dstSrcMap.create(dstHeight, dstWidth, CV_64FC2);
        cv::Mat dstTop = dstSrcMap(cv::Rect(0, 0, dstWidth, cubeHeight));
        cv::Mat dstBot = dstSrcMap(cv::Rect(0, cubeHeight, dstWidth, cubeHeight));
        cv::Mat srcLeft = map(cv::Rect(0, 0, cubeHeight * 3, cubeHeight));
        cv::Mat srcRight = map(cv::Rect(cubeHeight * 3, 0, cubeHeight * 3, cubeHeight));
        srcLeft.copyTo(dstTop);
        srcRight.copyTo(dstBot);
    }
}

void getEquiRectToCubeMap(cv::Mat& dstSrcXMap, cv::Mat& dstSrcYMap, int equiRectHeight, int cubeHeight, bool ratio6Over1)
{
    cv::Mat map;
    getEquiRectToCubeMap(map, equiRectHeight, cubeHeight, ratio6Over1);
    cv::Mat maps[2];
    cv::split(map, maps);
    maps[0].convertTo(dstSrcXMap, CV_32F);
    maps[1].convertTo(dstSrcYMap, CV_32F);
}

#define RIGHT   0
#define LEFT    1
#define TOP     2
#define BOTTOM  3
#define FRONT   4
#define BACK    5

static const float P0[] = { -0.5f, -0.5f, -0.5f };
static const float P1[] = { 0.5f, -0.5f, -0.5f };
static const float P4[] = { -0.5f, -0.5f, 0.5f };
static const float P5[] = { 0.5f, -0.5f, 0.5f };
static const float P6[] = { -0.5f, 0.5f, 0.5f };

static const float PX[] = { 1.0f, 0.0f, 0.0f };
static const float PY[] = { 0.0f, 1.0f, 0.0f };
static const float PZ[] = { 0.0f, 0.0f, 1.0f };
static const float NX[] = { -1.0f, 0.0f, 0.0f };
static const float NZ[] = { 0.0f, 0.0f, -1.0f };

// We need to end up with X and Y coordinates in the range [0..1).
// Horizontally wrapping is easy: 1.25 becomes 0.25, -0.25 becomes 0.75.
// Vertically, if we pass through the north pole, we start coming back 'down'
// in the Y direction (ie, a reflection from the boundary) but we also are
// on the opposite side of the sphere so the X value changes by 0.5.
static inline void normalize_equirectangular(float x, float y, float *xout, float *yout) {
    if (y >= 1.0f) {
        // Example: y = 1.25 ; 2.0 - 1.25 = 0.75.
        y = 2.0f - y;
        x += 0.5f;
    }
    else if (y < 0.0f) {
        y = -y;
        x += 0.5f;
    }

    if (x >= 1.0f) {
        int ipart = (int)x;
        x -= ipart;
    }
    else if (x < 0.0f) {
        // Example: x = -1.25.  ipart = 1. x += 2 so x = 0.25.
        int ipart = (int)(-x);
        x += (ipart + 1);
    }

    *xout = x;
    *yout = y;
}

static inline void transform_pos(int cubeType, float x, float y, float *outX, float *outY) {

    float qx, qy, qz;
    float cos_y, cos_p, sin_y, sin_p;
    float d;
    y = 1.0f - y;

    const float *vx, *vy, *p;
    int face = 0;
    if (cubeType == CubeType6x1) {
        face = (int)(x * 6);
        x = x * 6.0f - face;
    }
    else if (cubeType == CubeType3x2) {
        int vface = (int)(y * 2);
        int hface = (int)(x * 3);
        x = x * 3.0f - hface;
        y = y * 2.0f - vface;
        face = hface + (1 - vface) * 3;
    }
    else if (cubeType == CubeType180) {
        // LAYOUT_CUBEMAP_180: layout for spatial resolution downsampling with 180 degree viewport size
        //
        // - Given a view (yaw,pitch) we can create a customized cube mapping to make the view center at the front cube face.
        // - A 180 degree viewport cut the cube into 2 equal-sized halves: front half and back half.
        // - The front half contains these faces of the cube: front, half of right, half of left, half of top, half of bottom.
        //   The back half contains these faces of the cube: back, half of right, half of left, half of top, half of bottom.
        //   Illutrasion on LAYOUT_CUBEMAP_32 (mono):
        //
        //   +---+---+---+---+---+---+
        //   |   |   |   |   |   5   |
        //   + 1 | 2 + 3 | 4 +-------+     Area 1, 4, 6, 7, 9 are in the front half
        //   |   |   |   |   |   6   |
        //   +---+---+---+---+---+---+     Area 2, 3, 5, 8, 0 are in the back half
        //   |   7   |       |       |
        //   +-------+   9   +   0   +
        //   |   8   |       |       |
        //   +---+---+---+---+---+---+
        //
        // - LAYOUT_CUBEMAP_180 reduces the spatial resolution of the back half to 25% (1/2 height, 1/2 width makes 1/4 size)
        //   and then re-pack the cube map like this:
        //
        //   +---+---+---+---+---+      Front half   Back half (1/4 size)
        //   |       |   |   c   |      ----------   --------------------
        //   +   a   + b +---+----      Area a = 9   Area f = 0
        //   |       |   | f |   |      Area b = 4   Area g = 3
        //   +---+---+---+---+ d +      Area c = 6   Area h = 2
        //   |g|h|-i-|   e   |   |      Area d = 1   Area i1(top) = 5
        //   +---+---+---+---+---+      Area e = 7   Area i2(bottom) = 8
        //
        if (0.0f <= y && y < 1.0f / 3 && 0.0f <= x && x < 0.8f) { // Area g, h, i1, i2, e
            if (0.0f <= x && x < 0.1f) { // g
                face = LEFT;
                x = x / 0.2f;
                y = y / (1.0f / 3);
            }
            else if (0.1f <= x && x < 0.2f) { // h
                face = RIGHT;
                x = (x - 0.1f) / 0.2f + 0.5f;
                y = y / (1.0f / 3);
            }
            else if (0.2f <= x && x < 0.4f) {
                if (y >= 1.0f / 6){ //i1
                    face = TOP;
                    x = (x - 0.2f) / 0.2f;
                    y = (y - 1.0f / 6) / (1.0f / 3) + 0.5f;
                }
                else { // i2
                    face = BOTTOM;
                    x = (x - 0.2f) / 0.2f;
                    y = y / (1.0f / 3);
                }
            }
            else if (0.4f <= x && x < 0.8f){ // e
                face = BOTTOM;
                x = (x - 0.4f) / 0.4f;
                y = y / (2.0f / 3) + 0.5f;
            }
        }
        else if (2.0f / 3 <= y && y < 1.0f && 0.6f <= x && x < 1.0f) { // Area c
            face = TOP;
            x = (x - 0.6f) / 0.4f;
            y = (y - 2.0f / 3) / (2.0f / 3);
        }
        else { // Area a, b, f, d
            if (0.0f <= x && x < 0.4f) { // a
                face = FRONT;
                x = x / 0.4f;
                y = (y - 1.0 / 3) / (2.0f / 3);
            }
            else if (0.4f <= x && x < 0.6f) { // b
                face = LEFT;
                x = (x - 0.4f) / 0.4f + 0.5f;
                y = (y - 1.0f / 3) / (2.0f / 3);
            }
            else if (0.6f <= x && x < 0.8f) { // f
                face = BACK;
                x = (x - 0.6f) / 0.2f;
                y = (y - 1.0f / 3) / (1.0f / 3);
            }
            else if (0.8f <= x && x < 1.0f) { // d
                face = RIGHT;
                x = (x - 0.8f) / 0.4f;
                y = y / (2.0f / 3);
            }
        }
    }

    switch (face) {
    case RIGHT:   p = P5; vx = NZ; vy = PY; break;
    case LEFT:    p = P0; vx = PZ; vy = PY; break;
    case TOP:     p = P6; vx = PX; vy = NZ; break;
    case BOTTOM:  p = P0; vx = PX; vy = PZ; break;
    case FRONT:   p = P4; vx = PX; vy = PY; break;
    case BACK:    p = P1; vx = NX; vy = PY; break;
    }
    qx = p[0] + vx[0] * x + vy[0] * y;
    qy = p[1] + vx[1] * x + vy[1] * y;
    qz = p[2] + vx[2] * x + vy[2] * y;

    d = sqrtf(qx * qx + qy * qy + qz * qz);
    *outX = -atan2f(-qx / d, qz / d) / (PI * 2.0f) + 0.5f;
    *outY = asinf(-qy / d) / PI + 0.5f;

}

void getEquiRectToCubeMap(cv::Mat& dstSrcMap, int equiRectHeight, int cubeHeight, int cubeType)
{
    CV_Assert(equiRectHeight > 0 && cubeHeight > 0 &&
        (cubeType == CubeType6x1 || cubeType == CubeType3x2 || cubeType == CubeType180));
    int srcWidth = equiRectHeight * 2, srcHeight = equiRectHeight;
    int dstWidth, dstHeight;
    if (cubeType == CubeType6x1)
    {
        dstWidth = 6 * cubeHeight;
        dstHeight = cubeHeight;
    }
    else if (cubeType == CubeType3x2)
    {
        dstWidth = 3 * cubeHeight;
        dstHeight = 2 * cubeHeight;
    }
    else
    {
        dstWidth = cubeHeight * 2.5;
        dstHeight = cubeHeight * 1.5;
    }
    dstSrcMap.create(dstHeight, dstWidth, CV_64FC2);
    for (int i = 0; i < dstHeight; i++)
    {
        double* ptr = dstSrcMap.ptr<double>(i);
        for (int j = 0; j < dstWidth; j++)
        {
            float inx = (j + 0.5f) / dstWidth, iny = (i + 0.5f) / dstHeight;
            float outx, outy;
            transform_pos(cubeType, inx, iny, &outx, &outy);
            *(ptr++) = outx * srcWidth;
            *(ptr++) = outy * srcHeight;
        }
    }
}

void getEquiRectToCubeMap(cv::Mat& dstSrcXMap, cv::Mat& dstSrcYMap, int equiRectHeight, int cubeHeight, int cubeType)
{
    cv::Mat map;
    getEquiRectToCubeMap(map, equiRectHeight, cubeHeight, cubeType);
    cv::Mat maps[2];
    cv::split(map, maps);
    maps[0].convertTo(dstSrcXMap, CV_32F);
    maps[1].convertTo(dstSrcYMap, CV_32F);
}