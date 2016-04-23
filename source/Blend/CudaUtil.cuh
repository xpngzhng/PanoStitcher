#pragma once
#include "cuda_runtime.h"
#include <cmath>

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
        //return (x >= 0) * x + (x < 0) * (x - ((x - width + 1) / width) * width);
        if (x >= 0) return x;
        else return (x < 0) * (x - ((x - width + 1) / width) * width);
    }

    __device__ __forceinline__ int idx_col_high(int x) const
    {
        //return (x < width) * x + (x >= width) * (x % width);
        if (x < width) return x;
        else return (x % width);
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
        //return (y >= 0) * y + (y < 0) * (y - ((y - height + 1) / height) * height);
        if (y >= 0) return y;
        else return (y - ((y - height + 1) / height) * height);
    }

    __device__ __forceinline__ int idx_row_high(int y) const
    {
        //return (y < height) * y + (y >= height) * (y % height);
        if (y < height) return y;
        else return (y % height);
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

__device__ __forceinline__ int4 operator*(int scale, uchar4 val)
{
    int4 ret;
    ret.x = scale * val.x;
    ret.y = scale * val.y;
    ret.z = scale * val.z;
    return ret;
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

__device__ __forceinline__ float4 operator+(float4 a, float4 b)
{
    float4 ret;
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

__device__ __forceinline__ float4 operator-(float4 a, float4 b)
{
    float4 ret;
    ret.x = a.x - b.x;
    ret.y = a.y - b.y;
    ret.z = a.z - b.z;
    return ret;
}

__device__ __forceinline__ short4 operator-(short4 a, short4 b)
{
    short4 ret;
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

__device__ __forceinline__ float4 operator*(float scale, float4 val)
{
    float4 ret;
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

__device__ __forceinline__ float4 operator/(float4 val, float scale)
{
    float4 ret;
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

__device__ __forceinline__ short4 roundCastShift8ToShort4(int4 vec)
{
    short4 ret;
    ret.x = (vec.x + 128) >> 8;
    ret.y = (vec.y + 128) >> 8;
    ret.z = (vec.z + 128) >> 8;
    return ret;
}