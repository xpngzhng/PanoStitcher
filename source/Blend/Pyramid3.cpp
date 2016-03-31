#include "Pyramid.h"
#include <xmmintrin.h>
#include <emmintrin.h>
#include <vector>

struct PyrDownVec_32s
{
    int operator()(int** src, int* dst, int, int width) const
    {
        if (!cv::checkHardwareSupport(CV_CPU_SSE2))
            return 0;

        int x = 0;
        const int *row0 = src[0], *row1 = src[1], *row2 = src[2], *row3 = src[3], *row4 = src[4];
        __m128i delta = _mm_set1_epi16(128);

        for (; x <= width - 4; x += 4)
        {
            __m128i r0, r1, r2, r3, r4;
            r0 = _mm_load_si128((const __m128i*)(row0 + x));
            r1 = _mm_load_si128((const __m128i*)(row1 + x));
            r2 = _mm_load_si128((const __m128i*)(row2 + x));
            r3 = _mm_load_si128((const __m128i*)(row3 + x));
            r4 = _mm_load_si128((const __m128i*)(row4 + x));
            r0 = _mm_add_epi32(r0, r4);
            r1 = _mm_add_epi32(_mm_add_epi32(r1, r3), r2);
            r0 = _mm_add_epi32(r0, _mm_add_epi32(r2, r2));
            r0 = _mm_add_epi32(r0, _mm_slli_epi32(r1, 2));
            _mm_storeu_si128((__m128i*)(dst + x), r0);
        }

        return x;
    }
};

template<typename SrcElemType> void
pyrDown_(const cv::Mat& _src, cv::Mat& _dst, std::vector<unsigned char>& aux1, std::vector<unsigned char>& aux2, int horiBorderType, int vertBorderType)
{
    const int PD_SZ = 5;

    cv::Size ssize = _src.size(), dsize = _dst.size();
    int cn = _src.channels();
    int bufstep = (int)cv::alignSize(dsize.width*cn, 16);
    //cv::AutoBuffer<int> _buf(bufstep*PD_SZ + 16);
    //int* buf = cv::alignPtr((int*)_buf, 16);
    aux1.resize(sizeof(int) * (bufstep*PD_SZ + 16));
    int* buf = cv::alignPtr((int*)aux1.data(), 16);
    int tabL[CV_CN_MAX*(PD_SZ + 2)], tabR[CV_CN_MAX*(PD_SZ + 2)];
    //cv::AutoBuffer<int> _tabM(dsize.width*cn);
    //int* tabM = _tabM;
    aux2.resize(sizeof(int)* dsize.width*cn);
    int* tabM = (int*)aux2.data();
    int* rows[PD_SZ];

    CV_Assert(std::abs(dsize.width * 2 - ssize.width) <= 2 &&
        std::abs(dsize.height * 2 - ssize.height) <= 2);
    int k, x, sy0 = -PD_SZ / 2, sy = sy0, width0 = std::min((ssize.width - PD_SZ / 2 - 1) / 2 + 1, dsize.width);

    for (x = 0; x <= PD_SZ + 1; x++)
    {
        int sx0 = cv::borderInterpolate(x - PD_SZ / 2, ssize.width, horiBorderType)*cn;
        int sx1 = cv::borderInterpolate(x + width0 * 2 - PD_SZ / 2, ssize.width, horiBorderType)*cn;
        for (k = 0; k < cn; k++)
        {
            tabL[x*cn + k] = sx0 + k;
            tabR[x*cn + k] = sx1 + k;
        }
    }

    ssize.width *= cn;
    dsize.width *= cn;
    width0 *= cn;

    for (x = 0; x < dsize.width; x++)
        tabM[x] = (x / cn) * 2 * cn + x % cn;

    PyrDownVec_32s vecOp;

    for (int y = 0; y < dsize.height; y++)
    {
        int* dst = (int*)(_dst.data + _dst.step*y);
        int *row0, *row1, *row2, *row3, *row4;

        // fill the ring buffer (horizontal convolution and decimation)
        for (; sy <= y * 2 + 2; sy++)
        {
            int* row = buf + ((sy - sy0) % PD_SZ)*bufstep;
            int _sy = cv::borderInterpolate(sy, ssize.height, vertBorderType);
            const SrcElemType* src = (const SrcElemType*)(_src.data + _src.step*_sy);
            int limit = cn;
            const int* tab = tabL;

            for (x = 0;;)
            {
                for (; x < limit; x++)
                {
                    row[x] = src[tab[x + cn * 2]] * 6 + (src[tab[x + cn]] + src[tab[x + cn * 3]]) * 4 +
                        src[tab[x]] + src[tab[x + cn * 4]];
                }

                if (x == dsize.width)
                    break;

                if (cn == 1)
                {
                    for (; x < width0; x++)
                        row[x] = src[x * 2] * 6 + (src[x * 2 - 1] + src[x * 2 + 1]) * 4 +
                        src[x * 2 - 2] + src[x * 2 + 2];
                }
                else if (cn == 3)
                {
                    for (; x < width0; x += 3)
                    {
                        const SrcElemType* s = src + x * 2;
                        int t0 = s[0] * 6 + (s[-3] + s[3]) * 4 + s[-6] + s[6];
                        int t1 = s[1] * 6 + (s[-2] + s[4]) * 4 + s[-5] + s[7];
                        int t2 = s[2] * 6 + (s[-1] + s[5]) * 4 + s[-4] + s[8];
                        row[x] = t0; row[x + 1] = t1; row[x + 2] = t2;
                    }
                }
                else if (cn == 4)
                {
                    for (; x < width0; x += 4)
                    {
                        const SrcElemType* s = src + x * 2;
                        int t0 = s[0] * 6 + (s[-4] + s[4]) * 4 + s[-8] + s[8];
                        int t1 = s[1] * 6 + (s[-3] + s[5]) * 4 + s[-7] + s[9];
                        row[x] = t0; row[x + 1] = t1;
                        t0 = s[2] * 6 + (s[-2] + s[6]) * 4 + s[-6] + s[10];
                        t1 = s[3] * 6 + (s[-1] + s[7]) * 4 + s[-5] + s[11];
                        row[x + 2] = t0; row[x + 3] = t1;
                    }
                }
                else
                {
                    for (; x < width0; x++)
                    {
                        int sx = tabM[x];
                        row[x] = src[sx] * 6 + (src[sx - cn] + src[sx + cn]) * 4 +
                            src[sx - cn * 2] + src[sx + cn * 2];
                    }
                }

                limit = dsize.width;
                tab = tabR - x;
            }
        }

        // do vertical convolution and decimation and write the result to the destination image
        for (k = 0; k < PD_SZ; k++)
            rows[k] = buf + ((y * 2 - PD_SZ / 2 + k - sy0) % PD_SZ)*bufstep;
        row0 = rows[0]; row1 = rows[1]; row2 = rows[2]; row3 = rows[3]; row4 = rows[4];

        x = vecOp(rows, dst, _dst.step, dsize.width);
        for ( /*x = 0*/; x < dsize.width; x++)
            dst[x] = row2[x] * 6 + (row1[x] + row3[x]) * 4 + row0[x] + row4[x];
    }
}

typedef void(*PyrFuncTo32S)(const cv::Mat&, cv::Mat&, std::vector<unsigned char>&, std::vector<unsigned char>&, int, int);

void pyramidDownTo32S(const cv::Mat& src, cv::Mat& dst, std::vector<unsigned char>& aux1, std::vector<unsigned char>& aux2,
    const cv::Size& _dsz, int horiBorderType, int vertBorderType)
{
    CV_Assert(horiBorderType == cv::BORDER_DEFAULT || horiBorderType == cv::BORDER_WRAP);
    CV_Assert(vertBorderType == cv::BORDER_DEFAULT || vertBorderType == cv::BORDER_WRAP);
    cv::Size dsz = _dsz == cv::Size() ? cv::Size((src.cols + 1) / 2, (src.rows + 1) / 2) : _dsz;
    dst.create(dsz, CV_MAKETYPE(CV_32S, src.channels()));

    int depth = src.depth();
    PyrFuncTo32S func = 0;
    if (depth == CV_8U)
        func = pyrDown_<uchar>;
    else if (depth == CV_16S)
        func = pyrDown_<short>;
    else if (depth == CV_16U)
        func = pyrDown_<unsigned short>;
    else
        CV_Error(CV_StsUnsupportedFormat, "");

    func(src, dst, aux1, aux2, horiBorderType, vertBorderType);
}

template<typename T, int shift> struct FixPtCast
{
    typedef int type1;
    typedef T rtype;
    rtype operator ()(type1 arg) const { return (T)((arg + (1 << (shift - 1))) >> shift); }
};

template<typename T, int shift> struct FltCast
{
    typedef T type1;
    typedef T rtype;
    rtype operator ()(type1 arg) const { return arg*(T)(1. / (1 << shift)); }
};

template<typename T1, typename T2> struct NoVec
{
    int operator()(T1**, T2*, int, int) const { return 0; }
};

#define CV_SSE2 1

#if CV_SSE2

struct PyrDownVec_32s8u
{
    int operator()(int** src, uchar* dst, int, int width) const
    {
        if (!cv::checkHardwareSupport(CV_CPU_SSE2))
            return 0;

        int x = 0;
        const int *row0 = src[0], *row1 = src[1], *row2 = src[2], *row3 = src[3], *row4 = src[4];
        __m128i delta = _mm_set1_epi16(128);

        for (; x <= width - 16; x += 16)
        {
            __m128i r0, r1, r2, r3, r4, t0, t1;
            r0 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row0 + x)),
                _mm_load_si128((const __m128i*)(row0 + x + 4)));
            r1 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row1 + x)),
                _mm_load_si128((const __m128i*)(row1 + x + 4)));
            r2 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row2 + x)),
                _mm_load_si128((const __m128i*)(row2 + x + 4)));
            r3 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row3 + x)),
                _mm_load_si128((const __m128i*)(row3 + x + 4)));
            r4 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row4 + x)),
                _mm_load_si128((const __m128i*)(row4 + x + 4)));
            r0 = _mm_add_epi16(r0, r4);
            r1 = _mm_add_epi16(_mm_add_epi16(r1, r3), r2);
            r0 = _mm_add_epi16(r0, _mm_add_epi16(r2, r2));
            t0 = _mm_add_epi16(r0, _mm_slli_epi16(r1, 2));
            r0 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row0 + x + 8)),
                _mm_load_si128((const __m128i*)(row0 + x + 12)));
            r1 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row1 + x + 8)),
                _mm_load_si128((const __m128i*)(row1 + x + 12)));
            r2 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row2 + x + 8)),
                _mm_load_si128((const __m128i*)(row2 + x + 12)));
            r3 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row3 + x + 8)),
                _mm_load_si128((const __m128i*)(row3 + x + 12)));
            r4 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row4 + x + 8)),
                _mm_load_si128((const __m128i*)(row4 + x + 12)));
            r0 = _mm_add_epi16(r0, r4);
            r1 = _mm_add_epi16(_mm_add_epi16(r1, r3), r2);
            r0 = _mm_add_epi16(r0, _mm_add_epi16(r2, r2));
            t1 = _mm_add_epi16(r0, _mm_slli_epi16(r1, 2));
            t0 = _mm_srli_epi16(_mm_add_epi16(t0, delta), 8);
            t1 = _mm_srli_epi16(_mm_add_epi16(t1, delta), 8);
            _mm_storeu_si128((__m128i*)(dst + x), _mm_packus_epi16(t0, t1));
        }

        for (; x <= width - 4; x += 4)
        {
            __m128i r0, r1, r2, r3, r4, z = _mm_setzero_si128();
            r0 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row0 + x)), z);
            r1 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row1 + x)), z);
            r2 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row2 + x)), z);
            r3 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row3 + x)), z);
            r4 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row4 + x)), z);
            r0 = _mm_add_epi16(r0, r4);
            r1 = _mm_add_epi16(_mm_add_epi16(r1, r3), r2);
            r0 = _mm_add_epi16(r0, _mm_add_epi16(r2, r2));
            r0 = _mm_add_epi16(r0, _mm_slli_epi16(r1, 2));
            r0 = _mm_srli_epi16(_mm_add_epi16(r0, delta), 8);
            *(int*)(dst + x) = _mm_cvtsi128_si32(_mm_packus_epi16(r0, r0));
        }

        return x;
    }
};

struct PyrDownVec_32f
{
    int operator()(float** src, float* dst, int, int width) const
    {
        if (!cv::checkHardwareSupport(CV_CPU_SSE))
            return 0;

        int x = 0;
        const float *row0 = src[0], *row1 = src[1], *row2 = src[2], *row3 = src[3], *row4 = src[4];
        __m128 _4 = _mm_set1_ps(4.f), _scale = _mm_set1_ps(1.f / 256);
        for (; x <= width - 8; x += 8)
        {
            __m128 r0, r1, r2, r3, r4, t0, t1;
            r0 = _mm_load_ps(row0 + x);
            r1 = _mm_load_ps(row1 + x);
            r2 = _mm_load_ps(row2 + x);
            r3 = _mm_load_ps(row3 + x);
            r4 = _mm_load_ps(row4 + x);
            r0 = _mm_add_ps(r0, r4);
            r1 = _mm_add_ps(_mm_add_ps(r1, r3), r2);
            r0 = _mm_add_ps(r0, _mm_add_ps(r2, r2));
            t0 = _mm_add_ps(r0, _mm_mul_ps(r1, _4));

            r0 = _mm_load_ps(row0 + x + 4);
            r1 = _mm_load_ps(row1 + x + 4);
            r2 = _mm_load_ps(row2 + x + 4);
            r3 = _mm_load_ps(row3 + x + 4);
            r4 = _mm_load_ps(row4 + x + 4);
            r0 = _mm_add_ps(r0, r4);
            r1 = _mm_add_ps(_mm_add_ps(r1, r3), r2);
            r0 = _mm_add_ps(r0, _mm_add_ps(r2, r2));
            t1 = _mm_add_ps(r0, _mm_mul_ps(r1, _4));

            t0 = _mm_mul_ps(t0, _scale);
            t1 = _mm_mul_ps(t1, _scale);

            _mm_storeu_ps(dst + x, t0);
            _mm_storeu_ps(dst + x + 4, t1);
        }

        return x;
    }
};

#else

typedef NoVec<int, uchar> PyrDownVec_32s8u;
typedef NoVec<float, float> PyrDownVec_32f;

#endif

template<class CastOp, class VecOp> void
pyrDown_(const cv::Mat& _src, cv::Mat& _dst, std::vector<unsigned char>& aux1, std::vector<unsigned char>& aux2, int horiBorderType, int vertBorderType)
{
    const int PD_SZ = 5;
    typedef typename CastOp::type1 WT;
    typedef typename CastOp::rtype T;

    cv::Size ssize = _src.size(), dsize = _dst.size();
    int cn = _src.channels();
    int bufstep = (int)cv::alignSize(dsize.width*cn, 16);
    //cv::AutoBuffer<WT> _buf(bufstep*PD_SZ + 16);
    //WT* buf = cv::alignPtr((WT*)_buf, 16);
    aux1.resize(sizeof(WT) * (bufstep*PD_SZ + 16));
    WT* buf = cv::alignPtr((WT*)aux1.data(), 16);
    int tabL[CV_CN_MAX*(PD_SZ + 2)], tabR[CV_CN_MAX*(PD_SZ + 2)];
    //cv::AutoBuffer<int> _tabM(dsize.width*cn);
    //int* tabM = _tabM;
    aux2.resize(sizeof(int)* dsize.width*cn);
    int* tabM = (int*)aux2.data();
    WT* rows[PD_SZ];
    CastOp castOp;
    VecOp vecOp;

    CV_Assert(std::abs(dsize.width * 2 - ssize.width) <= 2 &&
        std::abs(dsize.height * 2 - ssize.height) <= 2);
    int k, x, sy0 = -PD_SZ / 2, sy = sy0, width0 = std::min((ssize.width - PD_SZ / 2 - 1) / 2 + 1, dsize.width);

    for (x = 0; x <= PD_SZ + 1; x++)
    {
        int sx0 = cv::borderInterpolate(x - PD_SZ / 2, ssize.width, horiBorderType)*cn;
        int sx1 = cv::borderInterpolate(x + width0 * 2 - PD_SZ / 2, ssize.width, horiBorderType)*cn;
        for (k = 0; k < cn; k++)
        {
            tabL[x*cn + k] = sx0 + k;
            tabR[x*cn + k] = sx1 + k;
        }
    }

    ssize.width *= cn;
    dsize.width *= cn;
    width0 *= cn;

    for (x = 0; x < dsize.width; x++)
        tabM[x] = (x / cn) * 2 * cn + x % cn;

    for (int y = 0; y < dsize.height; y++)
    {
        T* dst = (T*)(_dst.data + _dst.step*y);
        WT *row0, *row1, *row2, *row3, *row4;

        // fill the ring buffer (horizontal convolution and decimation)
        for (; sy <= y * 2 + 2; sy++)
        {
            WT* row = buf + ((sy - sy0) % PD_SZ)*bufstep;
            int _sy = cv::borderInterpolate(sy, ssize.height, vertBorderType);
            const T* src = (const T*)(_src.data + _src.step*_sy);
            int limit = cn;
            const int* tab = tabL;

            for (x = 0;;)
            {
                for (; x < limit; x++)
                {
                    row[x] = src[tab[x + cn * 2]] * 6 + (src[tab[x + cn]] + src[tab[x + cn * 3]]) * 4 +
                        src[tab[x]] + src[tab[x + cn * 4]];
                }

                if (x == dsize.width)
                    break;

                if (cn == 1)
                {
                    for (; x < width0; x++)
                        row[x] = src[x * 2] * 6 + (src[x * 2 - 1] + src[x * 2 + 1]) * 4 +
                        src[x * 2 - 2] + src[x * 2 + 2];
                }
                else if (cn == 3)
                {
                    for (; x < width0; x += 3)
                    {
                        const T* s = src + x * 2;
                        WT t0 = s[0] * 6 + (s[-3] + s[3]) * 4 + s[-6] + s[6];
                        WT t1 = s[1] * 6 + (s[-2] + s[4]) * 4 + s[-5] + s[7];
                        WT t2 = s[2] * 6 + (s[-1] + s[5]) * 4 + s[-4] + s[8];
                        row[x] = t0; row[x + 1] = t1; row[x + 2] = t2;
                    }
                }
                else if (cn == 4)
                {
                    for (; x < width0; x += 4)
                    {
                        const T* s = src + x * 2;
                        WT t0 = s[0] * 6 + (s[-4] + s[4]) * 4 + s[-8] + s[8];
                        WT t1 = s[1] * 6 + (s[-3] + s[5]) * 4 + s[-7] + s[9];
                        row[x] = t0; row[x + 1] = t1;
                        t0 = s[2] * 6 + (s[-2] + s[6]) * 4 + s[-6] + s[10];
                        t1 = s[3] * 6 + (s[-1] + s[7]) * 4 + s[-5] + s[11];
                        row[x + 2] = t0; row[x + 3] = t1;
                    }
                }
                else
                {
                    for (; x < width0; x++)
                    {
                        int sx = tabM[x];
                        row[x] = src[sx] * 6 + (src[sx - cn] + src[sx + cn]) * 4 +
                            src[sx - cn * 2] + src[sx + cn * 2];
                    }
                }

                limit = dsize.width;
                tab = tabR - x;
            }
        }

        // do vertical convolution and decimation and write the result to the destination image
        for (k = 0; k < PD_SZ; k++)
            rows[k] = buf + ((y * 2 - PD_SZ / 2 + k - sy0) % PD_SZ)*bufstep;
        row0 = rows[0]; row1 = rows[1]; row2 = rows[2]; row3 = rows[3]; row4 = rows[4];

        x = vecOp(rows, dst, (int)_dst.step, dsize.width);
        for (; x < dsize.width; x++)
            dst[x] = castOp(row2[x] * 6 + (row1[x] + row3[x]) * 4 + row0[x] + row4[x]);
    }
}

template<class CastOp, class VecOp> void
pyrUp_(const cv::Mat& _src, cv::Mat& _dst, std::vector<unsigned char>& aux1, std::vector<unsigned char>& aux2, int horiBorderType, int vertBorderType)
{
    const int PU_SZ = 3;
    typedef typename CastOp::type1 WT;
    typedef typename CastOp::rtype T;

    cv::Size ssize = _src.size(), dsize = _dst.size();
    int cn = _src.channels();
    int bufstep = (int)cv::alignSize((dsize.width + 1)*cn, 16);
    //cv::AutoBuffer<WT> _buf(bufstep*PU_SZ + 16);
    //WT* buf = cv::alignPtr((WT*)_buf, 16);
    aux1.resize(sizeof(WT) * (bufstep*PU_SZ + 16));
    WT* buf = cv::alignPtr((WT*)aux1.data(), 16);
    //cv::AutoBuffer<int> _dtab(ssize.width*cn);
    //int* dtab = _dtab;
    aux2.resize(sizeof(int) * ssize.width*cn);
    int* dtab = (int*)aux2.data();
    WT* rows[PU_SZ];
    CastOp castOp;
    VecOp vecOp;

    CV_Assert(std::abs(dsize.width - ssize.width * 2) == dsize.width % 2 &&
        std::abs(dsize.height - ssize.height * 2) == dsize.height % 2);
    int k, x, sy0 = -PU_SZ / 2, sy = sy0;

    int lx = cv::borderInterpolate(-1, ssize.width, horiBorderType) * cn;
    int rx = cv::borderInterpolate(ssize.width, ssize.width, horiBorderType) * cn;
    //printf("srcWidth = %d, dstWidth = %d, cn = %d, ", ssize.width, dsize.width, cn);
    //printf("lx = %d, rx = %d\n", lx, rx);

    ssize.width *= cn;
    dsize.width *= cn;

    for (x = 0; x < ssize.width; x++)
        dtab[x] = (x / cn) * 2 * cn + x % cn;

    for (int y = 0; y < ssize.height; y++)
    {
        T* dst0 = (T*)(_dst.data + _dst.step*y * 2);
        T* dst1 = (T*)(_dst.data + _dst.step*(y * 2 + 1));
        WT *row0, *row1, *row2;

        if (y * 2 + 1 >= dsize.height)
            dst1 = dst0;

        // fill the ring buffer (horizontal convolution and decimation)
        for (; sy <= y + 1; sy++)
        {
            WT* row = buf + ((sy - sy0) % PU_SZ)*bufstep;
            //int _sy = cv::borderInterpolate(sy*2, dsize.height, cv::BORDER_REFLECT_101)/2;
            int _sy = cv::borderInterpolate(sy, ssize.height, vertBorderType/*cv::BORDER_REFLECT_101*/);
            const T* src = (const T*)(_src.data + _src.step*_sy);

            if (ssize.width == cn)
            {
                for (x = 0; x < cn; x++)
                    row[x] = row[x + cn] = src[x] * 8;
                continue;
            }

            for (x = 0; x < cn; x++)
            {
                int dx = dtab[x];
                WT t0 = src[lx + x] + src[x] * 6 + src[x + cn];
                WT t1 = (src[x] + src[x + cn]) * 4;
                row[dx] = t0; row[dx + cn] = t1;
                //printf("x = %d: [%d, %d, %d], [%d, %d], ", x, lx + x, x, x + cn, x, x + cn);
                dx = dtab[ssize.width - cn + x];
                int sx = ssize.width - cn + x;
                t0 = src[sx - cn] + src[sx] * 6 + src[rx + x];
                t1 = (src[sx] + src[rx + x]) * 4;
                row[dx] = t0; row[dx + cn] = t1;
                //printf("[%d, %d, %d], [%d, %d]\n", sx - cn, sx, rx + x, sx, rx + x);
            }

            for (x = cn; x < ssize.width - cn; x++)
            {
                int dx = dtab[x];
                WT t0 = src[x - cn] + src[x] * 6 + src[x + cn];
                WT t1 = (src[x] + src[x + cn]) * 4;
                row[dx] = t0;
                row[dx + cn] = t1;
            }
        }

        // do vertical convolution and decimation and write the result to the destination image
        for (k = 0; k < PU_SZ; k++)
            rows[k] = buf + ((y - PU_SZ / 2 + k - sy0) % PU_SZ)*bufstep;
        row0 = rows[0]; row1 = rows[1]; row2 = rows[2];

        x = vecOp(rows, dst0, (int)_dst.step, dsize.width);
        for (; x < dsize.width; x++)
        {
            T t1 = castOp((row1[x] + row2[x]) * 4);
            T t0 = castOp(row0[x] + row1[x] * 6 + row2[x]);
            dst1[x] = t1; dst0[x] = t0;
        }
    }
}

typedef void(*PyrFunc)(const cv::Mat&, cv::Mat&, std::vector<unsigned char>&, std::vector<unsigned char>&, int, int);

void pyramidDown(const cv::Mat& src, cv::Mat& dst, std::vector<unsigned char>& aux1, std::vector<unsigned char>& aux2, const cv::Size& _dsz,
    int horiBorderType, int vertBorderType)
{
    CV_Assert(horiBorderType == cv::BORDER_DEFAULT || horiBorderType == cv::BORDER_WRAP);
    CV_Assert(vertBorderType == cv::BORDER_DEFAULT || vertBorderType == cv::BORDER_WRAP);
    cv::Size dsz = _dsz == cv::Size() ? cv::Size((src.cols + 1) / 2, (src.rows + 1) / 2) : _dsz;
    dst.create(dsz, src.type());

    int depth = src.depth();
    PyrFunc func = 0;
    if (depth == CV_8U)
        func = pyrDown_<FixPtCast<uchar, 8>, PyrDownVec_32s8u>;
    else if (depth == CV_16S)
        func = pyrDown_<FixPtCast<short, 8>, NoVec<int, short> >;
    else if (depth == CV_16U)
        func = pyrDown_<FixPtCast<ushort, 8>, NoVec<int, ushort> >;
    else if (depth == CV_32S)
        func = pyrDown_<FixPtCast<int, 8>, NoVec<int, int> >;
    else if (depth == CV_32F)
        func = pyrDown_<FltCast<float, 8>, PyrDownVec_32f>;
    else if (depth == CV_64F)
        func = pyrDown_<FltCast<double, 8>, NoVec<double, double> >;
    else
        CV_Error(CV_StsUnsupportedFormat, "");

    func(src, dst, aux1, aux2, horiBorderType, vertBorderType);
}

void pyramidUp(const cv::Mat& src, cv::Mat& dst, std::vector<unsigned char>& aux1, std::vector<unsigned char>& aux2, const cv::Size& _dsz,
    int horiBorderType, int vertBorderType)
{
    CV_Assert(horiBorderType == cv::BORDER_DEFAULT || horiBorderType == cv::BORDER_WRAP);
    CV_Assert(vertBorderType == cv::BORDER_DEFAULT || vertBorderType == cv::BORDER_WRAP);
    cv::Size dsz = _dsz == cv::Size() ? cv::Size(src.cols * 2, src.rows * 2) : _dsz;
    dst.create(dsz, src.type());

    int depth = src.depth();
    PyrFunc func = 0;
    if (depth == CV_8U)
        func = pyrUp_<FixPtCast<uchar, 6>, NoVec<int, uchar> >;
    else if (depth == CV_16S)
        func = pyrUp_<FixPtCast<short, 6>, NoVec<int, short> >;
    else if (depth == CV_16U)
        func = pyrUp_<FixPtCast<ushort, 6>, NoVec<int, ushort> >;
    else if (depth == CV_32S)
        func = pyrUp_<FixPtCast<int, 6>, NoVec<int, int> >;
    else if (depth == CV_32F)
        func = pyrUp_<FltCast<float, 6>, NoVec<float, float> >;
    else if (depth == CV_64F)
        func = pyrUp_<FltCast<double, 6>, NoVec<double, double> >;
    else
        CV_Error(CV_StsUnsupportedFormat, "");

    func(src, dst, aux1, aux2, horiBorderType, vertBorderType);
}