#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

// CudaPyramid2.cu -- Main implementation of pyramid operations.
// CudaPyramid3.cu -- Src memory access is implemented as texture memory access.
void pyramidDown16SC1To16SC1(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
void pyramidDown16SC1To32SC1(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
void pyramidDown16SC4To32SC4(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
void pyramidDown16SC4To16SC4(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
void divide32SC4To16SC4(const cv::gpu::GpuMat& srcImage, const cv::gpu::GpuMat& srcAlpha,
    cv::gpu::GpuMat& dstImage, cv::gpu::GpuMat& dstAlpha, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
void pyramidDown16SC4To16SC4(const cv::gpu::GpuMat& srcImage, const cv::gpu::GpuMat& srcAlpha, bool horiWrap,
    cv::gpu::GpuMat& dstImage, cv::gpu::GpuMat& dstAlpha, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
void pyramidUp16SC4To16SC4(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
void pyramidUp32SC4To32SC4(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst, cv::Size dstSize, bool horiWrap, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
void accumulate16SC4To32SC4(const cv::gpu::GpuMat& src, const cv::gpu::GpuMat& weight, cv::gpu::GpuMat& dst, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
void normalize32SC4(cv::gpu::GpuMat& image, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
void scaledSet16SC1Mask16SC1(cv::gpu::GpuMat& image, short val, const cv::gpu::GpuMat& mask, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
void scaledSet16SC1Mask32SC1(cv::gpu::GpuMat& image, short val, const cv::gpu::GpuMat& mask, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
void divide32SC4To16SC4(const cv::gpu::GpuMat& srcImage, const cv::gpu::GpuMat& srcAlpha, cv::gpu::GpuMat& dstImage, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
void subtract16SC4(const cv::gpu::GpuMat& a, const cv::gpu::GpuMat& b, cv::gpu::GpuMat& c, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
void add32SC4(const cv::gpu::GpuMat& a, const cv::gpu::GpuMat& b, cv::gpu::GpuMat& c, cv::gpu::Stream& stream = cv::gpu::Stream::Null());

// CudaPyramid4.cu -- Try to pad src image with border elements to reduce kernel's condition code.
// But this version of code actually runs slower than CudaPyramid2.cu.
void pyramidDownPad16SC1To32SC1(cv::gpu::GpuMat& padSrc, cv::gpu::GpuMat& padDst, cv::Size padDstSize, bool horiWrap);
void pyramidDownPad16SC1To16SC1(cv::gpu::GpuMat& padSrc, cv::gpu::GpuMat& padDst, cv::Size padDstSize, bool horiWrap);
void pyramidDownPad16SC4To32SC4(cv::gpu::GpuMat& padSrc, cv::gpu::GpuMat& padDst, cv::Size padDstSize, bool horiWrap);
void pyramidUpPad32SC4To32SC4(const cv::gpu::GpuMat& padSrc, cv::gpu::GpuMat& padDst, cv::Size padDstSize, bool horiWrap);
void pyramidUpPad16SC4To16SC4(const cv::gpu::GpuMat& padSrc, cv::gpu::GpuMat& padDst, cv::Size padDstSize, bool horiWrap);

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

void pyramidDown16SC1To32SC1(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst, cv::gpu::GpuMat& aux,
    const cv::gpu::GpuMat& horiIndexTab, const cv::gpu::GpuMat& vertIndexTab);


void func(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst);