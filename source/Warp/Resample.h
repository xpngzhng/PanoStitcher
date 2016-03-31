#pragma once

inline unsigned char bicubicWeighted(const unsigned char rgb[4], const double w[4])
{
    int res = (int)(rgb[0] * w[0] + rgb[1] * w[1] + rgb[2] * w[2] + rgb[3] * w[3] + 0.5);
    res = res > 255 ? 255 : res;
    res = res < 0 ? 0 : res;
    return (unsigned char)res;
}

inline void calcBicubicWeight(double deta, double weight[4])
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

inline void bicubicResample(int width, int height, int step, const unsigned char* data,
    double x, double y, unsigned char rgb[3])
{
    register int i, j;
    int x2 = (int)x;
    int y2 = (int)y;
    int nx[4];
    int ny[4];
    unsigned char rgb1[4];
    unsigned char rgb2[4];

    for (int i = 0; i < 4; ++i)
    {
        nx[i] = (x2 - 1 + i);
        ny[i] = (y2 - 1 + i);
        if (nx[i]<0)
        {
            nx[i] = 0;
        }
        if (nx[i]>width - 1)
        {
            nx[i] = width - 1;
        }
        if (ny[i]<0)
        {
            ny[i] = 0;
        }
        if (ny[i]>height - 1)
        {
            ny[i] = height - 1;
        }
    }

    double u = (x - nx[1]);
    double v = (y - ny[1]);
    //u,v vertical while /100 horizontal
    double tweight1[4], tweight2[4];
    calcBicubicWeight(u, tweight1);//weight
    calcBicubicWeight(v, tweight2);//weight

    for (int k = 0; k<3; ++k)
    {
        for (j = 0; j<4; j++)
        {
            // 按行去每个通道
            for (i = 0; i<4; i++)
            {
                rgb1[i] = data[ny[j] * step + nx[i] * 3 + k];
            }
            //4*4区域的三个通道
            rgb2[j] = bicubicWeighted(rgb1, tweight1);
        }
        rgb[k] = bicubicWeighted(rgb2, tweight2);
    }
}

inline void bilinearResample(int width, int height, int step, const unsigned char* data,
    double x, double y, unsigned char rgb[3])
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

    double b = 0, g = 0, r = 0;
    const unsigned char* ptr;
    ptr = data + step * y0 + x0 * 3;
    b += *(ptr++) * w00;
    g += *(ptr++) * w00;
    r += *(ptr++) * w00;
    b += *(ptr++) * w01;
    g += *(ptr++) * w01;
    r += *(ptr++) * w01;
    ptr = data + step * y1 + x0 * 3;
    b += *(ptr++) * w10;
    g += *(ptr++) * w10;
    r += *(ptr++) * w10;
    b += *(ptr++) * w11;
    g += *(ptr++) * w11;
    r += *(ptr++) * w11;

    rgb[0] = b;
    rgb[1] = g;
    rgb[2] = r;
}