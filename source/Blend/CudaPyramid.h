#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/cuda.hpp"

// CudaPyramid2.cu -- Main implementation of pyramid operations.
// CudaPyramid3.cu -- Src memory access is implemented as texture memory access.
void pyramidDown16SC1To16SC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void pyramidDown16SC1To32SC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void pyramidDown16SC4To32SC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void pyramidDown16SC4To16SC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void divide32SC4To16SC4(const cv::cuda::GpuMat& srcImage, const cv::cuda::GpuMat& srcAlpha,
    cv::cuda::GpuMat& dstImage, cv::cuda::GpuMat& dstAlpha, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void pyramidDown16SC4To16SC4(const cv::cuda::GpuMat& srcImage, const cv::cuda::GpuMat& srcAlpha, bool horiWrap,
    cv::cuda::GpuMat& dstImage, cv::cuda::GpuMat& dstAlpha, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void pyramidUp16SC4To16SC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void pyramidUp32SC4To32SC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void accumulate16SC4To32SC4(const cv::cuda::GpuMat& src, const cv::cuda::GpuMat& weight, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void normalize32SC4(cv::cuda::GpuMat& image, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void accumulate16SC4To32SC4(const cv::cuda::GpuMat& src, const cv::cuda::GpuMat& weight, 
    cv::cuda::GpuMat& dst, cv::cuda::GpuMat& dstWeight, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void normalize32SC4(cv::cuda::GpuMat& image, const cv::cuda::GpuMat& weight, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void scaledSet16SC1Mask16SC1(cv::cuda::GpuMat& image, short val, const cv::cuda::GpuMat& mask, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void scaledSet16SC1Mask32SC1(cv::cuda::GpuMat& image, short val, const cv::cuda::GpuMat& mask, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void divide32SC4To16SC4(const cv::cuda::GpuMat& srcImage, const cv::cuda::GpuMat& srcAlpha, cv::cuda::GpuMat& dstImage, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void subtract16SC4(const cv::cuda::GpuMat& a, const cv::cuda::GpuMat& b, cv::cuda::GpuMat& c, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void add32SC4(const cv::cuda::GpuMat& a, const cv::cuda::GpuMat& b, cv::cuda::GpuMat& c, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void accumulate16SC1To32SC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

// CudaPyramid4.cu -- Try to pad src image with border elements to reduce kernel's condition code.
// But this version of code actually runs slower than CudaPyramid2.cu.
void pyramidDownPad16SC1To32SC1(cv::cuda::GpuMat& padSrc, cv::cuda::GpuMat& padDst, cv::Size padDstSize, bool horiWrap);
void pyramidDownPad16SC1To16SC1(cv::cuda::GpuMat& padSrc, cv::cuda::GpuMat& padDst, cv::Size padDstSize, bool horiWrap);
void pyramidDownPad16SC4To32SC4(cv::cuda::GpuMat& padSrc, cv::cuda::GpuMat& padDst, cv::Size padDstSize, bool horiWrap);
void pyramidUpPad32SC4To32SC4(const cv::cuda::GpuMat& padSrc, cv::cuda::GpuMat& padDst, cv::Size padDstSize, bool horiWrap);
void pyramidUpPad16SC4To16SC4(const cv::cuda::GpuMat& padSrc, cv::cuda::GpuMat& padDst, cv::Size padDstSize, bool horiWrap);

// CudaPyramid5.cu -- Use traditional seperable filter method to implement pyramid operation.
// But still runs slower than CudaPyramid2.cu.
int inline alignAnySize(int val, int grain)
{
    return (val + grain - 1) / grain * grain;
}

inline void getIndexTab(int length, int pad, int borderType, cv::Mat& indexTab)
{
    int actualLength = alignAnySize(length, pad) + pad * 2;
    indexTab.create(1, actualLength, CV_32SC1);
    indexTab.setTo(0);
    int* ptr = (int*)indexTab.data + pad;
    for (int i = -2; i < length + 2; i++)
        ptr[i] = cv::borderInterpolate(i, length, borderType);
}

void pyramidDown16SC1To32SC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& aux,
    const cv::cuda::GpuMat& horiIndexTab, const cv::cuda::GpuMat& vertIndexTab);


void func(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);

// CudaPyramid6.cpp floating point implementation
void pyramidDown32FC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap);
void pyramidDown32FC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap);
void pyramidUp32FC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize, bool horiWrap);
void scaledSet32FC1Mask32FC1(cv::cuda::GpuMat& image, float val, const cv::cuda::GpuMat& mask);
void divide32FC4(const cv::cuda::GpuMat& srcImage, const cv::cuda::GpuMat& srcAlpha, cv::cuda::GpuMat& dstImage);
void subtract32FC4(const cv::cuda::GpuMat& a, const cv::cuda::GpuMat& b, cv::cuda::GpuMat& c);
void add32FC4(const cv::cuda::GpuMat& a, const cv::cuda::GpuMat& b, cv::cuda::GpuMat& c);
void accumulate32FC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);
void accumulate32FC4(const cv::cuda::GpuMat& src, const cv::cuda::GpuMat& weight, cv::cuda::GpuMat& dst);
void inverse32FC1(cv::cuda::GpuMat& mat);
void scale32FC4(cv::cuda::GpuMat& image, const cv::cuda::GpuMat& alpha);