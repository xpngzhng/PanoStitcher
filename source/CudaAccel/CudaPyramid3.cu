#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
//#include <opencv2/gpu/gpu.hpp>
//#include <opencv2/gpu/device/common.hpp>
//#include <opencv2/gpu/device/border_interpolate.hpp>
//#include <opencv2/gpu/device/vec_traits.hpp>
//#include <opencv2/gpu/device/vec_math.hpp>
//#include <opencv2/gpu/device/saturate_cast.hpp>

#define PYR_DOWN_BLOCK_SIZE 256

struct BrdRowReflect101
{
    explicit __host__ __device__ __forceinline__ BrdRowReflect101(int width) : last_col(width - 1) {}

    __device__ __forceinline__ int idx_col_low(int x) const
    {
        return ::abs(x) % (last_col + 1);
    }

    __device__ __forceinline__ int idx_col_high(int x) const
    {
        return ::abs(last_col - ::abs(last_col - x)) % (last_col + 1);
    }

    __device__ __forceinline__ int idx_col(int x) const
    {
        return idx_col_low(idx_col_high(x));
    }

    const int last_col;
};

struct BrdColReflect101
{
    explicit __host__ __device__ __forceinline__ BrdColReflect101(int height) : last_row(height - 1) {}

    __device__ __forceinline__ int idx_row_low(int y) const
    {
        return ::abs(y) % (last_row + 1);
    }

    __device__ __forceinline__ int idx_row_high(int y) const
    {
        return ::abs(last_row - ::abs(last_row - y)) % (last_row + 1);
    }

    __device__ __forceinline__ int idx_row(int y) const
    {
        return idx_row_low(idx_row_high(y));
    }

    const int last_row;
};

struct BrdRowWrap
{
    explicit __host__ __device__ __forceinline__ BrdRowWrap(int width_) : width(width_) {}

    __device__ __forceinline__ int idx_col_low(int x) const
    {
        return (x >= 0) * x + (x < 0) * (x - ((x - width + 1) / width) * width);
    }

    __device__ __forceinline__ int idx_col_high(int x) const
    {
        return (x < width) * x + (x >= width) * (x % width);
    }

    __device__ __forceinline__ int idx_col(int x) const
    {
        return idx_col_high(idx_col_low(x));
    }

    const int width;
};

struct BrdColWrap
{
    explicit __host__ __device__ __forceinline__ BrdColWrap(int height_) : height(height_) {}

    __device__ __forceinline__ int idx_row_low(int y) const
    {
        return (y >= 0) * y + (y < 0) * (y - ((y - height + 1) / height) * height);
    }

    __device__ __forceinline__ int idx_row_high(int y) const
    {
        return (y < height) * y + (y >= height) * (y % height);
    }

    __device__ __forceinline__ int idx_row(int y) const
    {
        return idx_row_high(idx_row_low(y));
    }

    const int height;
};

template<typename Type>
__device__ __forceinline__ Type getElem(const unsigned char* data, int step, int row, int col)
{
    return *((Type*)(data + row * step) + col);
}

template<typename Type>
__device__ __forceinline__ Type getElem(unsigned char* data, int step, int row, int col)
{
    return *((Type*)(data + row * step) + col);
}

template<typename Type>
__device__ __forceinline__ const Type* getRowPtr(const unsigned char* data, int step, int row)
{
    return (const Type*)(data + row * step);
}

template<typename Type>
__device__ __forceinline__ Type* getRowPtr(unsigned char* data, int step, int row)
{
    return (Type*)(data + row * step);
}

__device__ __forceinline__ int4 operator*(int scale, short4 val)
{
    int4 ret;
    ret.x = scale * val.x;
    ret.y = scale * val.y;
    ret.z = scale * val.z;
    return ret;
}

__device__ __forceinline__ int4 operator+(int4 a, int4 b)
{
    int4 ret;
    ret.x = a.x + b.x;
    ret.y = a.y + b.y;
    ret.z = a.z + b.z;
    return ret;
}

__device__ __forceinline__ int4 operator-(int4 a, int4 b)
{
    int4 ret;
    ret.x = a.x - b.x;
    ret.y = a.y - b.y;
    ret.z = a.z - b.z;
    return ret;
}

__device__ __forceinline__ int4 operator*(short scale, int4 val)
{
    int4 ret;
    ret.x = scale * val.x;
    ret.y = scale * val.y;
    ret.z = scale * val.z;
    return ret;
}

__device__ __forceinline__ int4 operator*(int scale, int4 val)
{
    int4 ret;
    ret.x = scale * val.x;
    ret.y = scale * val.y;
    ret.z = scale * val.z;
    return ret;
}

__device__ __forceinline__ short4 operator/(int4 val, int scale)
{
    short4 ret;
    ret.x = val.x / scale;
    ret.y = val.y / scale;
    ret.z = val.z / scale;
    return ret;
}

__device__ __forceinline__ int4& operator+=(int4& a, int4& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__device__ __forceinline__ int4 operator>>(int4 val, int amount)
{
    int4 ret;
    ret.x = val.x >> amount;
    ret.y = val.y >> amount;
    ret.z = val.z >> amount;
    return ret;
}

__device__ __forceinline__ int4 operator<<(int4 val, int amount)
{
    int4 ret;
    ret.x = val.x << amount;
    ret.y = val.y << amount;
    ret.z = val.z << amount;
    return ret;
}

__device__ __forceinline__ int4 roundCastShift6ToInt4(int4 vec)
{
    int4 ret;
    ret.x = (vec.x + 32) >> 6;
    ret.y = (vec.y + 32) >> 6;
    ret.z = (vec.z + 32) >> 6;
    return ret;
}

__device__ __forceinline__ short4 roundCastShift6ToShort4(int4 vec)
{
    short4 ret;
    ret.x = (vec.x + 32) >> 6;
    ret.y = (vec.y + 32) >> 6;
    ret.z = (vec.z + 32) >> 6;
    return ret;
}

texture<short, 2> tex16SC1;
texture<short4, 2> tex16SC4;
texture<int4, 2> tex32SC4;

__global__ void pyrDown16SC1To32SC1(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep, 
    const BrdColReflect101 rb, const BrdRowReflect101 cb)
{
    __shared__ int smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    const int srcy = 2 * y;

    if (srcy >= 2 && srcy < srcRows - 2 && x >= 2 && x < srcCols - 2)
    {
        {
            int sum;
            sum =           tex2D(tex16SC1, x, srcy - 2);
            sum = sum + 4 * tex2D(tex16SC1, x, srcy - 1);
            sum = sum + 6 * tex2D(tex16SC1, x, srcy    );
            sum = sum + 4 * tex2D(tex16SC1, x, srcy + 1);
            sum = sum +     tex2D(tex16SC1, x, srcy + 2);
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int leftx = x - 2;
            int sum;
            sum =           tex2D(tex16SC1, leftx, srcy - 2);
            sum = sum + 4 * tex2D(tex16SC1, leftx, srcy - 1);
            sum = sum + 6 * tex2D(tex16SC1, leftx, srcy    );
            sum = sum + 4 * tex2D(tex16SC1, leftx, srcy + 1);
            sum = sum +     tex2D(tex16SC1, leftx, srcy + 2);
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int rightx = x + 2;
            int sum;
            sum =           tex2D(tex16SC1, rightx, srcy - 2);
            sum = sum + 4 * tex2D(tex16SC1, rightx, srcy - 1);
            sum = sum + 6 * tex2D(tex16SC1, rightx, srcy    );
            sum = sum + 4 * tex2D(tex16SC1, rightx, srcy + 1);
            sum = sum +     tex2D(tex16SC1, rightx, srcy + 2);
            smem[4 + threadIdx.x] = sum;
        }
    }
    else
    {
        {
            int sum;
            sum =           tex2D(tex16SC1, cb.idx_col_high(x), rb.idx_row_low(srcy - 2) );
            sum = sum + 4 * tex2D(tex16SC1, cb.idx_col_high(x), rb.idx_row_low(srcy - 1) );
            sum = sum + 6 * tex2D(tex16SC1, cb.idx_col_high(x), srcy                     );
            sum = sum + 4 * tex2D(tex16SC1, cb.idx_col_high(x), rb.idx_row_high(srcy + 1));
            sum = sum +     tex2D(tex16SC1, cb.idx_col_high(x), rb.idx_row_high(srcy + 2));
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int leftx = x - 2;
            int sum;
            sum =           tex2D(tex16SC1, rb.idx_row_low(srcy - 2),  cb.idx_col(leftx));
            sum = sum + 4 * tex2D(tex16SC1, rb.idx_row_low(srcy - 1),  cb.idx_col(leftx));
            sum = sum + 6 * tex2D(tex16SC1, srcy,                      cb.idx_col(leftx));
            sum = sum + 4 * tex2D(tex16SC1, rb.idx_row_high(srcy + 1), cb.idx_col(leftx));
            sum = sum +     tex2D(tex16SC1, rb.idx_row_high(srcy + 2), cb.idx_col(leftx));
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int rightx = x + 2;
            int sum;
            sum =           tex2D(tex16SC1, cb.idx_col_high(rightx), rb.idx_row_low(srcy - 2) );
            sum = sum + 4 * tex2D(tex16SC1, cb.idx_col_high(rightx), rb.idx_row_low(srcy - 1) );
            sum = sum + 6 * tex2D(tex16SC1, cb.idx_col_high(rightx), srcy                     );
            sum = sum + 4 * tex2D(tex16SC1, cb.idx_col_high(rightx), rb.idx_row_high(srcy + 1));
            sum = sum +     tex2D(tex16SC1, cb.idx_col_high(rightx), rb.idx_row_high(srcy + 2));
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

        const int dstx = (blockIdx.x * blockDim.x + tid2) / 2;

        if (dstx < dstCols)
            getRowPtr<int>(dstData, dstStep, y)[dstx] = sum;
    }
}

__global__ void pyrDown16SC1To16SC1(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep, 
    const BrdColReflect101 rb, const BrdRowReflect101 cb)
{
    __shared__ int smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    const int srcy = 2 * y;

    if (srcy >= 2 && srcy < srcRows - 2 && x >= 2 && x < srcCols - 2)
    {
        {
            int sum;
            sum =           tex2D(tex16SC1, x, srcy - 2);
            sum = sum + 4 * tex2D(tex16SC1, x, srcy - 1);
            sum = sum + 6 * tex2D(tex16SC1, x, srcy    );
            sum = sum + 4 * tex2D(tex16SC1, x, srcy + 1);
            sum = sum +     tex2D(tex16SC1, x, srcy + 2);
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int leftx = x - 2;
            int sum;
            sum =           tex2D(tex16SC1, leftx, srcy - 2);
            sum = sum + 4 * tex2D(tex16SC1, leftx, srcy - 1);
            sum = sum + 6 * tex2D(tex16SC1, leftx, srcy    );
            sum = sum + 4 * tex2D(tex16SC1, leftx, srcy + 1);
            sum = sum +     tex2D(tex16SC1, leftx, srcy + 2);
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int rightx = x + 2;
            int sum;
            sum =           tex2D(tex16SC1, rightx, srcy - 2);
            sum = sum + 4 * tex2D(tex16SC1, rightx, srcy - 1);
            sum = sum + 6 * tex2D(tex16SC1, rightx, srcy    );
            sum = sum + 4 * tex2D(tex16SC1, rightx, srcy + 1);
            sum = sum +     tex2D(tex16SC1, rightx, srcy + 2);
            smem[4 + threadIdx.x] = sum;
        }
    }
    else
    {
        {
            int sum;
            sum =           tex2D(tex16SC1, cb.idx_col_high(x), rb.idx_row_low(srcy - 2) );
            sum = sum + 4 * tex2D(tex16SC1, cb.idx_col_high(x), rb.idx_row_low(srcy - 1) );
            sum = sum + 6 * tex2D(tex16SC1, cb.idx_col_high(x), srcy                     );
            sum = sum + 4 * tex2D(tex16SC1, cb.idx_col_high(x), rb.idx_row_high(srcy + 1));
            sum = sum +     tex2D(tex16SC1, cb.idx_col_high(x), rb.idx_row_high(srcy + 2));
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int leftx = x - 2;
            int sum;
            sum =           tex2D(tex16SC1, rb.idx_row_low(srcy - 2),  cb.idx_col(leftx));
            sum = sum + 4 * tex2D(tex16SC1, rb.idx_row_low(srcy - 1),  cb.idx_col(leftx));
            sum = sum + 6 * tex2D(tex16SC1, srcy,                      cb.idx_col(leftx));
            sum = sum + 4 * tex2D(tex16SC1, rb.idx_row_high(srcy + 1), cb.idx_col(leftx));
            sum = sum +     tex2D(tex16SC1, rb.idx_row_high(srcy + 2), cb.idx_col(leftx));
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int rightx = x + 2;
            int sum;
            sum =           tex2D(tex16SC1, cb.idx_col_high(rightx), rb.idx_row_low(srcy - 2) );
            sum = sum + 4 * tex2D(tex16SC1, cb.idx_col_high(rightx), rb.idx_row_low(srcy - 1) );
            sum = sum + 6 * tex2D(tex16SC1, cb.idx_col_high(rightx), srcy                     );
            sum = sum + 4 * tex2D(tex16SC1, cb.idx_col_high(rightx), rb.idx_row_high(srcy + 1));
            sum = sum +     tex2D(tex16SC1, cb.idx_col_high(rightx), rb.idx_row_high(srcy + 2));
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

        const int dstx = (blockIdx.x * blockDim.x + tid2) / 2;

        if (dstx < dstCols)
            getRowPtr<short>(dstData, dstStep, y)[dstx] = (sum + 128) >> 8;
    }
}

__global__ void pyrDown16SC4To32SC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep, 
    const BrdColReflect101 rb, const BrdRowReflect101 cb)
{
    __shared__ int4 smem[PYR_DOWN_BLOCK_SIZE + 4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    const int srcy = 2 * y;

    if (srcy >= 2 && srcy < srcRows - 2 && x >= 2 && x < srcCols - 2)
    {
        {
            int4 sum;
            sum =       1 * tex2D(tex16SC4, x, srcy - 2);
            sum = sum + 4 * tex2D(tex16SC4, x, srcy - 1);
            sum = sum + 6 * tex2D(tex16SC4, x, srcy    );
            sum = sum + 4 * tex2D(tex16SC4, x, srcy + 1);
            sum = sum + 1 * tex2D(tex16SC4, x, srcy + 2);
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int leftx = x - 2;
            int4 sum;
            sum =       1 * tex2D(tex16SC4, leftx, srcy - 2);
            sum = sum + 4 * tex2D(tex16SC4, leftx, srcy - 1);
            sum = sum + 6 * tex2D(tex16SC4, leftx, srcy    );
            sum = sum + 4 * tex2D(tex16SC4, leftx, srcy + 1);
            sum = sum + 1 * tex2D(tex16SC4, leftx, srcy + 2);
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int rightx = x + 2;
            int4 sum;
            sum =       1 * tex2D(tex16SC4, rightx, srcy - 2);
            sum = sum + 4 * tex2D(tex16SC4, rightx, srcy - 1);
            sum = sum + 6 * tex2D(tex16SC4, rightx, srcy    );
            sum = sum + 4 * tex2D(tex16SC4, rightx, srcy + 1);
            sum = sum + 1 * tex2D(tex16SC4, rightx, srcy + 2);
            smem[4 + threadIdx.x] = sum;
        }
    }
    else
    {
        {
            int4 sum;
            sum =       1 * tex2D(tex16SC4, cb.idx_col_high(x), rb.idx_row_low(srcy - 2) );
            sum = sum + 4 * tex2D(tex16SC4, cb.idx_col_high(x), rb.idx_row_low(srcy - 1) );
            sum = sum + 6 * tex2D(tex16SC4, cb.idx_col_high(x), srcy                     );
            sum = sum + 4 * tex2D(tex16SC4, cb.idx_col_high(x), rb.idx_row_high(srcy + 1));
            sum = sum + 1 * tex2D(tex16SC4, cb.idx_col_high(x), rb.idx_row_high(srcy + 2));
            smem[2 + threadIdx.x] = sum;
        }

        if (threadIdx.x < 2)
        {
            const int leftx = x - 2;
            int4 sum;
            sum =       1 * tex2D(tex16SC4, rb.idx_row_low(srcy - 2),  cb.idx_col(leftx));
            sum = sum + 4 * tex2D(tex16SC4, rb.idx_row_low(srcy - 1),  cb.idx_col(leftx));
            sum = sum + 6 * tex2D(tex16SC4, srcy,                      cb.idx_col(leftx));
            sum = sum + 4 * tex2D(tex16SC4, rb.idx_row_high(srcy + 1), cb.idx_col(leftx));
            sum = sum + 1 * tex2D(tex16SC4, rb.idx_row_high(srcy + 2), cb.idx_col(leftx));
            smem[threadIdx.x] = sum;
        }

        if (threadIdx.x > PYR_DOWN_BLOCK_SIZE - 3)
        {
            const int rightx = x + 2;
            int4 sum;
            sum =       1 * tex2D(tex16SC4, cb.idx_col_high(rightx), rb.idx_row_low(srcy - 2) );
            sum = sum + 4 * tex2D(tex16SC4, cb.idx_col_high(rightx), rb.idx_row_low(srcy - 1) );
            sum = sum + 6 * tex2D(tex16SC4, cb.idx_col_high(rightx), srcy                     );
            sum = sum + 4 * tex2D(tex16SC4, cb.idx_col_high(rightx), rb.idx_row_high(srcy + 1));
            sum = sum + 1 * tex2D(tex16SC4, cb.idx_col_high(rightx), rb.idx_row_high(srcy + 2));
            smem[4 + threadIdx.x] = sum;
        }
    }

    __syncthreads();

    if (threadIdx.x < PYR_DOWN_BLOCK_SIZE / 2)
    {
        const int tid2 = threadIdx.x * 2;
        int4 sum;
        sum =       1 * smem[2 + tid2 - 2];
        sum = sum + 4 * smem[2 + tid2 - 1];
        sum = sum + 6 * smem[2 + tid2    ];
        sum = sum + 4 * smem[2 + tid2 + 1];
        sum = sum + 1 * smem[2 + tid2 + 2];

        const int dstx = (blockIdx.x * blockDim.x + tid2) / 2;

        if (dstx < dstCols)
            getRowPtr<int4>(dstData, dstStep, y)[dstx] = sum;
    }
}

__global__ void pyrUp32SC4To32SC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep, 
    const BrdColReflect101 rb, const BrdRowReflect101 cb)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int4 s_srcPatch[10][10];
    __shared__ int4 s_dstPatch[20][16];

    if (threadIdx.x < 10 && threadIdx.y < 10)
    {
        int srcx = static_cast<int>((blockIdx.x * blockDim.x) / 2 + threadIdx.x) - 1;
        int srcy = static_cast<int>((blockIdx.y * blockDim.y) / 2 + threadIdx.y) - 1;

        srcx = srcx < 0 ? cb.idx_col_low(srcx) : (srcx >= srcCols) ? cb.idx_col_high(srcx) : srcx;
        srcy = srcy < 0 ? rb.idx_row_low(srcy) : (srcy >= srcRows) ? rb.idx_row_high(srcy) : srcy;

        s_srcPatch[threadIdx.y][threadIdx.x] = tex2D(tex32SC4, srcx, srcy);
    }

    __syncthreads();

    int4 sum;

    const int evenFlag = static_cast<int>((threadIdx.x & 1) == 0);
    const int oddFlag  = static_cast<int>((threadIdx.x & 1) != 0);
    const int eveny = ((threadIdx.y & 1) == 0);
    const int tidx = threadIdx.x;

    if (eveny)
    {
        sum =       (evenFlag    ) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx - 2) >> 1)];
        sum = sum + ( oddFlag * 4) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx - 1) >> 1)];
        sum = sum + (evenFlag * 6) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx    ) >> 1)];
        sum = sum + ( oddFlag * 4) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx + 1) >> 1)];
        sum = sum + (evenFlag    ) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx + 2) >> 1)];
    }

    s_dstPatch[2 + threadIdx.y][threadIdx.x] = sum;

    if (threadIdx.y < 2)
    {
        if (eveny)
        {
            sum =       (evenFlag    ) * s_srcPatch[0][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[0][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 6) * s_srcPatch[0][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[0][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag    ) * s_srcPatch[0][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[threadIdx.y][threadIdx.x] = sum;
    }

    if (threadIdx.y > 13)
    {
        if (eveny)
        {
            sum =       (evenFlag    ) * s_srcPatch[9][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[9][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 6) * s_srcPatch[9][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[9][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag    ) * s_srcPatch[9][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[4 + threadIdx.y][threadIdx.x] = sum;
    }

    __syncthreads();

    const int tidy = threadIdx.y;

    sum =           s_dstPatch[2 + tidy - 2][threadIdx.x];
    sum = sum + 4 * s_dstPatch[2 + tidy - 1][threadIdx.x];
    sum = sum + 6 * s_dstPatch[2 + tidy    ][threadIdx.x];
    sum = sum + 4 * s_dstPatch[2 + tidy + 1][threadIdx.x];
    sum = sum +     s_dstPatch[2 + tidy + 2][threadIdx.x];

    if (x < dstCols && y < dstRows)
        getRowPtr<int4>(dstData, dstStep, y)[x] = roundCastShift6ToInt4(sum);
}

__global__ void pyrUp16SC4To16SC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep, 
    const BrdColReflect101 rb, const BrdRowReflect101 cb)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ short4 s_srcPatch[10][10];
    __shared__ int4 s_dstPatch[20][16];

    if (threadIdx.x < 10 && threadIdx.y < 10)
    {
        int srcx = static_cast<int>((blockIdx.x * blockDim.x) / 2 + threadIdx.x) - 1;
        int srcy = static_cast<int>((blockIdx.y * blockDim.y) / 2 + threadIdx.y) - 1;

        srcx = srcx < 0 ? cb.idx_col_low(srcx) : (srcx >= srcCols) ? cb.idx_col_high(srcx) : srcx;
        srcy = srcy < 0 ? rb.idx_row_low(srcy) : (srcy >= srcRows) ? rb.idx_row_high(srcy) : srcy;

        s_srcPatch[threadIdx.y][threadIdx.x] = tex2D(tex16SC4, srcx, srcy);
    }

    __syncthreads();

    int4 sum;

    const int evenFlag = static_cast<int>((threadIdx.x & 1) == 0);
    const int oddFlag  = static_cast<int>((threadIdx.x & 1) != 0);
    const int eveny = ((threadIdx.y & 1) == 0);
    const int tidx = threadIdx.x;

    if (eveny)
    {
        sum =       (evenFlag    ) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx - 2) >> 1)];
        sum = sum + ( oddFlag * 4) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx - 1) >> 1)];
        sum = sum + (evenFlag * 6) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx    ) >> 1)];
        sum = sum + ( oddFlag * 4) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx + 1) >> 1)];
        sum = sum + (evenFlag    ) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx + 2) >> 1)];
    }

    s_dstPatch[2 + threadIdx.y][threadIdx.x] = sum;

    if (threadIdx.y < 2)
    {
        if (eveny)
        {
            sum =       (evenFlag    ) * s_srcPatch[0][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[0][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 6) * s_srcPatch[0][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[0][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag    ) * s_srcPatch[0][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[threadIdx.y][threadIdx.x] = sum;
    }

    if (threadIdx.y > 13)
    {
        if (eveny)
        {
            sum =       (evenFlag    ) * s_srcPatch[9][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[9][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 6) * s_srcPatch[9][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 4) * s_srcPatch[9][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag    ) * s_srcPatch[9][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[4 + threadIdx.y][threadIdx.x] = sum;
    }

    __syncthreads();

    const int tidy = threadIdx.y;

    sum =           s_dstPatch[2 + tidy - 2][threadIdx.x];
    sum = sum + 4 * s_dstPatch[2 + tidy - 1][threadIdx.x];
    sum = sum + 6 * s_dstPatch[2 + tidy    ][threadIdx.x];
    sum = sum + 4 * s_dstPatch[2 + tidy + 1][threadIdx.x];
    sum = sum +     s_dstPatch[2 + tidy + 2][threadIdx.x];

    if (x < dstCols && y < dstRows)
        getRowPtr<short4>(dstData, dstStep, y)[x] = roundCastShift6ToShort4(sum);
}

__global__ void divide32SC4To16SC4(const unsigned char* srcImageData, int srcImageRows, int srcImageCols, int srcImageStep,
    const unsigned char* srcAlphaData, int srcAlphaRows, int srcAlphaCols, int srcAlphaStep, 
    unsigned char* dstImageData, int dstImageRows, int dstImageCols, int dstImageStep,
    unsigned char* dstAlphaData, int dstAlphaRows, int dstAlphaCols, int dstAlphaStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < srcImageCols && y < srcImageRows)
    {
        int4 srcImageVal = getElem<int4>(srcImageData, srcImageStep, y, x);
        int srcAlphaVal = getElem<int>(srcAlphaData, srcAlphaStep, y, x);
        if (srcAlphaVal)
        {
            getRowPtr<short4>(dstImageData, dstImageStep, y)[x] = (srcImageVal << 8) / srcAlphaVal;
            getRowPtr<short>(dstAlphaData, dstAlphaStep, y)[x] = 256;
        }
        else
        {
            getRowPtr<short4>(dstImageData, dstImageStep, y)[x] = make_short4(0, 0, 0, 0);
            getRowPtr<short>(dstAlphaData, dstAlphaStep, y)[x] = 0;
        }
    }
}

__global__ void accumulate16SC4To32SC4(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    const unsigned char* weightData, int weightRows, int weightCols, int weightStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < srcCols && y < srcRows)
    {
        getRowPtr<int4>(dstData, dstStep, y)[x] = getRowPtr<int4>(dstData, dstStep, y)[x] + 
            getElem<short>(weightData, weightStep, y, x) * getElem<short4>(srcData, srcStep, y, x);
    }
}

__global__ void normalize32SC4(unsigned char* imageData, int imageRows, int imageCols, int imageStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < imageCols && y < imageRows)
    {
        getRowPtr<int4>(imageData, imageStep, y)[x] = (getElem<int4>(imageData, imageStep, y, x) + make_int4(128, 128, 128, 0)) >> 8;
    }
}

void pyramidDown16SC1To16SC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize)
{
    CV_Assert(src.data && src.type() == CV_16SC1); 

    if (dstSize == cv::Size())
    {
        dstSize.width = ((src.cols + 1) >> 1);
        dstSize.height = ((src.rows + 1) >> 1);
    }
    dst.create(dstSize, CV_16SC1);

    const dim3 block(PYR_DOWN_BLOCK_SIZE);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), dst.rows);
    BrdColReflect101 rb(src.rows);
    BrdRowReflect101 cb(src.cols);
    cudaChannelFormatDesc chanDescShort = cudaCreateChannelDesc<short>();
    cudaSafeCall(cudaBindTexture2D(NULL, tex16SC1, src.data, chanDescShort, src.cols, src.rows, src.step));
    pyrDown16SC1To16SC1<<<grid, block>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaUnbindTexture(tex16SC1));
    cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidDown16SC1To32SC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize)
{
    CV_Assert(src.data && src.type() == CV_16SC1); 

    if (dstSize == cv::Size())
    {
        dstSize.width = ((src.cols + 1) >> 1);
        dstSize.height = ((src.rows + 1) >> 1);
    }
    dst.create(dstSize, CV_32SC1);

    const dim3 block(PYR_DOWN_BLOCK_SIZE);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), dst.rows);
    BrdColReflect101 rb(src.rows);
    BrdRowReflect101 cb(src.cols);
    cudaChannelFormatDesc chanDescShort = cudaCreateChannelDesc<short>();
    cudaSafeCall(cudaBindTexture2D(NULL, tex16SC1, src.data, chanDescShort, src.cols, src.rows, src.step));
    pyrDown16SC1To32SC1<<<grid, block>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaUnbindTexture(tex16SC1));
    cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidDown16SC4To32SC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize)
{
    CV_Assert(src.data && src.type() == CV_16SC4); 

    if (dstSize == cv::Size())
    {
        dstSize.width = ((src.cols + 1) >> 1);
        dstSize.height = ((src.rows + 1) >> 1);
    }
    dst.create(dstSize, CV_32SC4);

    const dim3 block(PYR_DOWN_BLOCK_SIZE);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), dst.rows);
    BrdColReflect101 rb(src.rows);
    BrdRowReflect101 cb(src.cols);
    cudaChannelFormatDesc chanDescShort4 = cudaCreateChannelDesc<short4>();
    cudaSafeCall(cudaBindTexture2D(NULL, tex16SC4, src.data, chanDescShort4, src.cols, src.rows, src.step));
    pyrDown16SC4To32SC4<<<grid, block>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaUnbindTexture(tex16SC4));
    cudaSafeCall(cudaDeviceSynchronize());
}

void divide32SC4To16SC4(const cv::cuda::GpuMat& srcImage, const cv::cuda::GpuMat& srcAlpha,
    cv::cuda::GpuMat& dstImage, cv::cuda::GpuMat& dstAlpha)
{
    CV_Assert(srcImage.data && srcImage.type() == CV_32SC4 &&
        srcAlpha.data && srcAlpha.type() == CV_32SC1 &&
        srcImage.size() == srcAlpha.size());

    dstImage.create(srcImage.size(), CV_16SC4);
    dstAlpha.create(srcAlpha.size(), CV_16SC1);

    const dim3 block(256);
    const dim3 grid(cv::cuda::device::divUp(srcImage.cols, block.x), cv::cuda::device::divUp(srcImage.rows, block.y));
    divide32SC4To16SC4<<<grid, block>>>(srcImage.data, srcImage.rows, srcImage.cols, srcImage.step,
        srcAlpha.data, srcAlpha.rows, srcAlpha.cols, srcAlpha.step,
        dstImage.data, dstImage.rows, dstImage.cols, dstImage.step,
        dstAlpha.data, dstAlpha.rows, dstAlpha.cols, dstAlpha.step);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidDown16SC4To16SC4(const cv::cuda::GpuMat& srcImage, const cv::cuda::GpuMat& srcAlpha,
    cv::cuda::GpuMat& dstImage, cv::cuda::GpuMat& dstAlpha)
{
    CV_Assert(srcImage.data && srcImage.type() == CV_16SC4 &&
        srcAlpha.data && srcAlpha.type() == CV_16SC1 && srcImage.size() == srcAlpha.size());
    cv::cuda::GpuMat dstImage32S, dstAlpha32S;
    pyramidDown16SC4To32SC4(srcImage, dstImage32S, cv::Size());
    pyramidDown16SC1To32SC1(srcAlpha, dstAlpha32S, cv::Size());
    divide32SC4To16SC4(dstImage32S, dstAlpha32S, dstImage, dstAlpha);
}

void pyramidUp32SC4To32SC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize)
{
    CV_Assert(src.data && src.type() == CV_32SC4);
    
    if (dstSize == cv::Size())
    {
        dstSize.width = (src.cols << 1);
        dstSize.height = (src.rows << 1);
    }
    dst.create(dstSize, CV_32SC4);

    const dim3 block(16, 16);
    const dim3 grid(cv::cuda::device::divUp(dst.cols, block.x), cv::cuda::device::divUp(dst.rows, block.y));
    BrdColReflect101 rb(src.rows);
    BrdRowReflect101 cb(src.cols);
    cudaChannelFormatDesc chanDescInt4 = cudaCreateChannelDesc<int4>();
    cudaSafeCall(cudaBindTexture2D(NULL, tex32SC4, src.data, chanDescInt4, src.cols, src.rows, src.step));
    pyrUp32SC4To32SC4<<<grid, block>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaUnbindTexture(tex32SC4));
    cudaSafeCall(cudaDeviceSynchronize());
}

void pyramidUp16SC4To16SC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize)
{
    CV_Assert(src.data && src.type() == CV_16SC4);
    
    if (dstSize == cv::Size())
    {
        dstSize.width = (src.cols << 1);
        dstSize.height = (src.rows << 1);
    }
    dst.create(dstSize, CV_16SC4);

    const dim3 block(16, 16);
    const dim3 grid(cv::cuda::device::divUp(dst.cols, block.x), cv::cuda::device::divUp(dst.rows, block.y));
    BrdColReflect101 rb(src.rows);
    BrdRowReflect101 cb(src.cols);
    cudaChannelFormatDesc chanDescShort4 = cudaCreateChannelDesc<short4>();
    cudaSafeCall(cudaBindTexture2D(NULL, tex16SC4, src.data, chanDescShort4, src.cols, src.rows, src.step));
    pyrUp16SC4To16SC4<<<grid, block>>>(src.data, src.rows, src.cols, src.step, dst.data, dst.rows, dst.cols, dst.step, rb, cb);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaUnbindTexture(tex16SC4));
    cudaSafeCall(cudaDeviceSynchronize());
}

void accumulate16SC4To32SC4(const cv::cuda::GpuMat& src, const cv::cuda::GpuMat& weight, cv::cuda::GpuMat& dst)
{
    CV_Assert(src.data && src.type() == CV_16SC4 &&
        weight.data && weight.type() == CV_16SC1 &&
        dst.data && dst.type() == CV_32SC4 &&
        src.size() == weight.size() && src.size() == dst.size());

    const dim3 block(32, 8);
    const dim3 grid(cv::cuda::device::divUp(src.cols, block.x), cv::cuda::device::divUp(src.rows, block.y));
    accumulate16SC4To32SC4<<<grid, block>>>(src.data, src.rows, src.cols, src.step, 
        weight.data, weight.rows, weight.cols, weight.step,
        dst.data, dst.rows, dst.cols, dst.step);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void normalize32SC4(cv::cuda::GpuMat& image)
{
    CV_Assert(image.data && image.type() == CV_32SC4);
    const dim3 block(32, 8);
    const dim3 grid(cv::cuda::device::divUp(image.cols, block.x), cv::cuda::device::divUp(image.rows, block.y));
    normalize32SC4<<<grid, block>>>(image.data, image.rows, image.cols, image.step);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}