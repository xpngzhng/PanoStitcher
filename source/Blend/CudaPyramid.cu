#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda_devptrs.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/device/common.hpp>
#include <opencv2/gpu/device/border_interpolate.hpp>
#include <opencv2/gpu/device/vec_traits.hpp>
#include <opencv2/gpu/device/vec_math.hpp>
#include <opencv2/gpu/device/saturate_cast.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#define PYR_DOWN_BLOCK_SIZE 256

namespace cv { namespace gpu { namespace device
{

__device__ inline int3 operator<<(int3 vec, unsigned numShiftBits)
{
    int3 ret;
    ret.x = vec.x << numShiftBits;
    ret.y = vec.y << numShiftBits;
    ret.z = vec.z << numShiftBits;
    return ret;
}

__device__ inline int3 operator>>(int3 vec, unsigned numShiftBits)
{
    int3 ret;
    ret.x = vec.x >> numShiftBits;
    ret.y = vec.y >> numShiftBits;
    ret.z = vec.z >> numShiftBits;
    return ret;
}

__device__ inline int3 roundCast8(int3 vec)
{
    int3 ret;
    ret.x = (vec.x + 128) >> 8;
    ret.y = (vec.y + 128) >> 8;
    ret.z = (vec.z + 128) >> 8;
    return ret;
}

__device__ inline int roundCast8(int val)
{
    return (val + 128) >> 8;
}

__device__ inline int3 roundCast6(int3 vec)
{
    int3 ret;
    ret.x = (vec.x + 32) >> 6;
    ret.y = (vec.y + 32) >> 6;
    ret.z = (vec.z + 32) >> 6;
    return ret;
}

__device__ inline int roundCast6(int val)
{
    return (val + 32) >> 6;
}

__global__ void pyrDown32SC3(const PtrStepSz<int3> src, PtrStep<int3> dst, 
    const BrdColReflect101<int3> rb, const BrdRowReflect101<int3> cb, bool origScale, int dst_cols)
{
    __shared__ int3 smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    const int src_y = 2 * y;

    if (src_y >= 2 && src_y < src.rows - 2 && x >= 2 && x < src.cols - 2)
    {
        {
            int3 sum;
            sum =           src(src_y - 2, x);
            sum = sum + 4 * src(src_y - 1, x);
            sum = sum + 6 * src(src_y,     x);
            sum = sum + 4 * src(src_y + 1, x);
            sum = sum +     src(src_y + 2, x);
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int left_x = x - 2;
            int3 sum;
            sum =           src(src_y - 2, left_x);
            sum = sum + 4 * src(src_y - 1, left_x);
            sum = sum + 6 * src(src_y,     left_x);
            sum = sum + 4 * src(src_y + 1, left_x);
            sum = sum +     src(src_y + 2, left_x);
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int right_x = x + 2;
            int3 sum;
            sum =           src(src_y - 2, right_x);
            sum = sum + 4 * src(src_y - 1, right_x);
            sum = sum + 6 * src(src_y,     right_x);
            sum = sum + 4 * src(src_y + 1, right_x);
            sum = sum +     src(src_y + 2, right_x);
            smem[4 + threadIdx.x] = sum;
        }
    }
    else
    {
        {
            int3 sum;
            sum =           src(rb.idx_row_low(src_y - 2),  cb.idx_col_high(x));
            sum = sum + 4 * src(rb.idx_row_low(src_y - 1),  cb.idx_col_high(x));
            sum = sum + 6 * src(src_y,                      cb.idx_col_high(x));
            sum = sum + 4 * src(rb.idx_row_high(src_y + 1), cb.idx_col_high(x));
            sum = sum +     src(rb.idx_row_high(src_y + 2), cb.idx_col_high(x));
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int left_x = x - 2;
            int3 sum;
            sum =           src(rb.idx_row_low(src_y - 2),  cb.idx_col(left_x));
            sum = sum + 4 * src(rb.idx_row_low(src_y - 1),  cb.idx_col(left_x));
            sum = sum + 6 * src(src_y,                      cb.idx_col(left_x));
            sum = sum + 4 * src(rb.idx_row_high(src_y + 1), cb.idx_col(left_x));
            sum = sum +     src(rb.idx_row_high(src_y + 2), cb.idx_col(left_x));
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int right_x = x + 2;
            int3 sum;
            sum =           src(rb.idx_row_low(src_y - 2),  cb.idx_col_high(right_x));
            sum = sum + 4 * src(rb.idx_row_low(src_y - 1),  cb.idx_col_high(right_x));
            sum = sum + 6 * src(src_y,                      cb.idx_col_high(right_x));
            sum = sum + 4 * src(rb.idx_row_high(src_y + 1), cb.idx_col_high(right_x));
            sum = sum +     src(rb.idx_row_high(src_y + 2), cb.idx_col_high(right_x));
            smem[4 + threadIdx.x] = sum;
        }
    }

    __syncthreads();

    if (threadIdx.x < PYR_DOWN_BLOCK_SIZE / 2)
    {
        const int tid2 = threadIdx.x * 2;
        int3 sum;
        sum =           smem[2 + tid2 - 2];
        sum = sum + 4 * smem[2 + tid2 - 1];
        sum = sum + 6 * smem[2 + tid2    ];
        sum = sum + 4 * smem[2 + tid2 + 1];
        sum = sum +     smem[2 + tid2 + 2];

        const int dst_x = (blockIdx.x * blockDim.x + tid2) / 2;

        if (dst_x < dst_cols)
            dst.ptr(y)[dst_x] = origScale ? roundCast8(sum) : sum;
    }
}

__global__ void pyrDown32SC1(const PtrStepSz<int> src, PtrStep<int> dst, 
    const BrdColReflect101<int> rb, const BrdRowReflect101<int> cb, bool origScale, int dst_cols)
{
    __shared__ int smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    const int src_y = 2 * y;

    if (src_y >= 2 && src_y < src.rows - 2 && x >= 2 && x < src.cols - 2)
    {
        {
            int sum;
            sum =           src(src_y - 2, x);
            sum = sum + 4 * src(src_y - 1, x);
            sum = sum + 6 * src(src_y,     x);
            sum = sum + 4 * src(src_y + 1, x);
            sum = sum +     src(src_y + 2, x);
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int left_x = x - 2;
            int sum;
            sum =           src(src_y - 2, left_x);
            sum = sum + 4 * src(src_y - 1, left_x);
            sum = sum + 6 * src(src_y,     left_x);
            sum = sum + 4 * src(src_y + 1, left_x);
            sum = sum +     src(src_y + 2, left_x);
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int right_x = x + 2;
            int sum;
            sum =           src(src_y - 2, right_x);
            sum = sum + 4 * src(src_y - 1, right_x);
            sum = sum + 6 * src(src_y,     right_x);
            sum = sum + 4 * src(src_y + 1, right_x);
            sum = sum +     src(src_y + 2, right_x);
            smem[4 + threadIdx.x] = sum;
        }
    }
    else
    {
        {
            int sum;
            sum =           src(rb.idx_row_low(src_y - 2),  cb.idx_col_high(x));
            sum = sum + 4 * src(rb.idx_row_low(src_y - 1),  cb.idx_col_high(x));
            sum = sum + 6 * src(src_y,                      cb.idx_col_high(x));
            sum = sum + 4 * src(rb.idx_row_high(src_y + 1), cb.idx_col_high(x));
            sum = sum +     src(rb.idx_row_high(src_y + 2), cb.idx_col_high(x));
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int left_x = x - 2;
            int sum;
            sum =           src(rb.idx_row_low(src_y - 2),  cb.idx_col(left_x));
            sum = sum + 4 * src(rb.idx_row_low(src_y - 1),  cb.idx_col(left_x));
            sum = sum + 6 * src(src_y,                      cb.idx_col(left_x));
            sum = sum + 4 * src(rb.idx_row_high(src_y + 1), cb.idx_col(left_x));
            sum = sum +     src(rb.idx_row_high(src_y + 2), cb.idx_col(left_x));
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int right_x = x + 2;
            int sum;
            sum =           src(rb.idx_row_low(src_y - 2),  cb.idx_col_high(right_x));
            sum = sum + 4 * src(rb.idx_row_low(src_y - 1),  cb.idx_col_high(right_x));
            sum = sum + 6 * src(src_y,                      cb.idx_col_high(right_x));
            sum = sum + 4 * src(rb.idx_row_high(src_y + 1), cb.idx_col_high(right_x));
            sum = sum +     src(rb.idx_row_high(src_y + 2), cb.idx_col_high(right_x));
            smem[4 + threadIdx.x] = sum;
        }
    }

    __syncthreads();

    if (threadIdx.x < PYR_DOWN_BLOCK_SIZE / 2)
    {
        const int tid2 = threadIdx.x * 2;
        int sum;
        sum =           smem[2 + tid2 - 2];
        sum = sum + 4 * smem[2 + tid2 - 1];
        sum = sum + 6 * smem[2 + tid2    ];
        sum = sum + 4 * smem[2 + tid2 + 1];
        sum = sum +     smem[2 + tid2 + 2];

        const int dst_x = (blockIdx.x * blockDim.x + tid2) / 2;

        if (dst_x < dst_cols)
            dst.ptr(y)[dst_x] = origScale ? roundCast8(sum) : sum;
    }
}

__global__ void pyrUp32SC3(const PtrStepSz<int3> src, PtrStepSz<int3> dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int3 s_srcPatch[10][10];
    __shared__ int3 s_dstPatch[20][16];

    if (threadIdx.x < 10 && threadIdx.y < 10)
    {
        int srcx = static_cast<int>((blockIdx.x * blockDim.x) / 2 + threadIdx.x) - 1;
        int srcy = static_cast<int>((blockIdx.y * blockDim.y) / 2 + threadIdx.y) - 1;

        srcx = ::abs(srcx);
        srcx = ::min(src.cols - 1, srcx);

        srcy = ::abs(srcy);
        srcy = ::min(src.rows - 1, srcy);

        s_srcPatch[threadIdx.y][threadIdx.x] = src(srcy, srcx);
    }

    __syncthreads();

    int3 sum = VecTraits<int3>::all(0);

    const int evenFlag = static_cast<int>((threadIdx.x & 1) == 0);
    const int oddFlag  = static_cast<int>((threadIdx.x & 1) != 0);
    const bool eveny = ((threadIdx.y & 1) == 0);
    const int tidx = threadIdx.x;

    if (eveny)
    {
        sum = sum + (evenFlag    ) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx - 2) >> 1)];
        sum = sum + ( oddFlag * 4) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx - 1) >> 1)];
        sum = sum + (evenFlag * 6) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx    ) >> 1)];
        sum = sum + ( oddFlag * 4) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx + 1) >> 1)];
        sum = sum + (evenFlag    ) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx + 2) >> 1)];
    }

    s_dstPatch[2 + threadIdx.y][threadIdx.x] = sum;

    if (threadIdx.y < 2)
    {
        sum = VecTraits<int3>::all(0);

        if (eveny)
        {
            sum = sum + (evenFlag    ) * s_srcPatch[0][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[0][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 6) * s_srcPatch[0][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[0][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag    ) * s_srcPatch[0][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[threadIdx.y][threadIdx.x] = sum;
    }

    if (threadIdx.y > 13)
    {
        sum = VecTraits<int3>::all(0);

        if (eveny)
        {
            sum = sum + (evenFlag    ) * s_srcPatch[9][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[9][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 6) * s_srcPatch[9][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[9][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag    ) * s_srcPatch[9][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[4 + threadIdx.y][threadIdx.x] = sum;
    }

    __syncthreads();

    sum = VecTraits<int3>::all(0);

    const int tidy = threadIdx.y;

    sum = sum +     s_dstPatch[2 + tidy - 2][threadIdx.x];
    sum = sum + 4 * s_dstPatch[2 + tidy - 1][threadIdx.x];
    sum = sum + 6 * s_dstPatch[2 + tidy    ][threadIdx.x];
    sum = sum + 4 * s_dstPatch[2 + tidy + 1][threadIdx.x];
    sum = sum +     s_dstPatch[2 + tidy + 2][threadIdx.x];

    if (x < dst.cols && y < dst.rows)
        dst(y, x) = roundCast6(sum);
}

__global__ void divide(const PtrStepSz<int3> srcImage, const PtrStepSz<int> srcAlpha, 
    PtrStepSz<int3> dstImage, PtrStepSz<int> dstAlpha)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < srcImage.cols && y < srcImage.rows)
    {
        int3 srcImageVal = srcImage(y, x);
        int srcAlphaVal = srcAlpha(y, x);
        if (srcAlphaVal)
        {
            dstImage(y, x) = ((srcImageVal << 8) - srcImageVal) / srcAlphaVal;
            dstAlpha(y, x) = 256;
        }
        else
        {
            dstImage(y, x) = make_int3(0, 0, 0);
            dstAlpha(y, x) = 0;
        }
    }
}

__global__ void accumulate(const PtrStepSz<int3> image, const PtrStepSz<int> weight, PtrStepSz<int3> dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < image.cols && y < image.rows)
    {
        dst(y, x) = dst(y, x) + image(y, x) * weight(y, x);
    }
}

__global__ void normalize(PtrStepSz<int3> image)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < image.cols && y < image.rows)
    {
        image(y, x) = roundCast8(image(y, x));
    }
}

__global__ void add(const PtrStepSz<int3> a, const PtrStepSz<int3> b, PtrStepSz<int3> c)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < a.cols && y < a.rows)
    {
        c(y, x) = a(y, x) + b(y, x);
    }
}

__global__ void subtract(const PtrStepSz<int3> a, const PtrStepSz<int3> b, PtrStepSz<int3> c)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < a.cols && y < a.rows)
    {
        c(y, x) = a(y, x) - b(y, x);
    }
}

__global__ void set(PtrStepSz<int> image, const PtrStepSz<unsigned char> mask, int val)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < image.cols && y < image.rows && mask(y, x))
    {
        image(y, x) = val;
    }
}
}}}

void pyramidDown(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst, cv::Size dstSize, bool dstScaleBack)
{
    CV_Assert(src.data && (src.type() == CV_32SC3 || src.type() == CV_32SC1)); 

    int type = src.type();
    if (dstSize == cv::Size())
    {
        dstSize.width = ((src.cols + 1) >> 1);
        dstSize.height = ((src.rows + 1) >> 1);
    }
    dst.create(dstSize, type);

    const dim3 block(PYR_DOWN_BLOCK_SIZE);
    const dim3 grid(cv::gpu::divUp(src.cols, block.x), dst.rows);
    if (type == CV_32SC3)
    {
        cv::gpu::device::BrdColReflect101<int3> rb(src.rows);
        cv::gpu::device::BrdRowReflect101<int3> cb(src.cols);    
        cv::gpu::device::pyrDown32SC3<<<grid, block>>>(src, dst, rb, cb, dstScaleBack, dst.cols);
    }
    else
    {
        cv::gpu::device::BrdColReflect101<int> rb(src.rows);
        cv::gpu::device::BrdRowReflect101<int> cb(src.cols);    
        cv::gpu::device::pyrDown32SC1<<<grid, block>>>(src, dst, rb, cb, dstScaleBack, dst.cols);
    }
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidUp(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst, cv::Size dstSize)
{
    CV_Assert(src.data && src.type() == CV_32SC3);
    
    if (dstSize == cv::Size())
    {
        dstSize.width = (src.cols << 1);
        dstSize.height = (src.rows << 1);
    }
    dst.create(dstSize, CV_32SC3);

    const dim3 block(16, 16);
    const dim3 grid(cv::gpu::divUp(dst.cols, block.x), cv::gpu::divUp(dst.rows, block.y));
    cv::gpu::device::pyrUp32SC3<<<grid, block>>>(src, dst);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void divide(const cv::gpu::GpuMat& srcImage, const cv::gpu::GpuMat& srcAlpha,
    cv::gpu::GpuMat& dstImage, cv::gpu::GpuMat& dstAlpha)
{
    CV_Assert(srcImage.data && srcImage.type() == CV_32SC3 &&
        srcAlpha.data && srcAlpha.type() == CV_32SC1 &&
        srcImage.size() == srcAlpha.size());

    dstImage.create(srcImage.size(), CV_32SC3);
    dstAlpha.create(srcAlpha.size(), CV_32SC1);

    const dim3 block(256);
    const dim3 grid(cv::gpu::divUp(srcImage.cols, block.x), cv::gpu::divUp(srcImage.rows, block.y));
    cv::gpu::device::divide<<<grid, block>>>(srcImage, srcAlpha, dstImage, dstAlpha);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void accumulate(const cv::gpu::GpuMat& image, const cv::gpu::GpuMat& weight, cv::gpu::GpuMat& dst)
{
    CV_Assert(image.data && image.type() == CV_32SC3 &&
        weight.data && weight.type() == CV_32SC1 &&
        dst.data && dst.type() == CV_32SC3 &&
        image.size() == weight.size() && image.size() == dst.size());

    const dim3 block(32, 8);
    const dim3 grid(cv::gpu::divUp(image.cols, block.x), cv::gpu::divUp(image.rows, block.y));
    cv::gpu::device::accumulate<<<grid, block>>>(image, weight, dst);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void normalize(cv::gpu::GpuMat& image)
{
    CV_Assert(image.data && image.type() == CV_32SC3);
    const dim3 block(32, 8);
    const dim3 grid(cv::gpu::divUp(image.cols, block.x), cv::gpu::divUp(image.rows, block.y));
    cv::gpu::device::normalize<<<grid, block>>>(image);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void add(const cv::gpu::GpuMat& a, const cv::gpu::GpuMat& b, cv::gpu::GpuMat& c)
{
    CV_Assert(a.data && a.type() == CV_32SC3 
        && b.data && b.type() == CV_32SC3&&
        a.size() == b.size());
    c.create(a.size(), CV_32SC3);
    const dim3 block(32, 8);
    const dim3 grid(cv::gpu::divUp(a.cols, block.x), cv::gpu::divUp(a.rows, block.y));
    cv::gpu::device::add<<<grid, block>>>(a, b, c);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void subtract(const cv::gpu::GpuMat& a, const cv::gpu::GpuMat& b, cv::gpu::GpuMat& c)
{
    CV_Assert(a.data && a.type() == CV_32SC3 
        && b.data && b.type() == CV_32SC3&&
        a.size() == b.size());
    c.create(a.size(), CV_32SC3);
    const dim3 block(32, 8);
    const dim3 grid(cv::gpu::divUp(a.cols, block.x), cv::gpu::divUp(a.rows, block.y));
    cv::gpu::device::subtract<<<grid, block>>>(a, b, c);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void set(cv::gpu::GpuMat& image, const cv::gpu::GpuMat& mask, int val)
{
    CV_Assert(image.data && image.type() == CV_32SC1 &&
        mask.data && mask.type() == CV_8UC1 &&
        image.size() == mask.size());
    const dim3 block(32, 8);
    const dim3 grid(cv::gpu::divUp(image.cols, block.x), cv::gpu::divUp(image.rows, block.y));
    cv::gpu::device::set<<<grid, block>>>(image, mask, val);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}