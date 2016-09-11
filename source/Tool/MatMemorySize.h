#pragma once

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"

struct MatMemorySize
{
    MatMemorySize(void* data_, int refcount_, long long int size_)
    : data(data_), refcount(refcount_), size(size_) {}
    void* data;
    int refcount;
    long long int size;
};

template<typename MatType>
inline long long int calcMemorySize(const MatType& mat, std::vector<MatMemorySize>& memSizes)
{
    long long int size = 0;
    if (mat.data && mat.refcount)
    {
        int arrLength = memSizes.size();
        bool shoudAdd = true;
        for (int i = 0; i < arrLength; i++)
        {
            if (memSizes[i].data == mat.datastart)
            {
                shoudAdd = false;
                break;
            }
        }
        size = mat.dataend - mat.datastart;
        if (shoudAdd)
            memSizes.push_back(MatMemorySize(mat.datastart, *mat.refcount, size));
    }
    return size;
}

template<typename MatType>
inline long long int calcMemorySize(const std::vector<MatType>& mats, std::vector<MatMemorySize>& memSizes)
{
    long long int size = 0;
    int num = mats.size();
    for (int i = 0; i < num; i++)
        size += calcMemorySize(mats[i], memSizes);
    return size;
}

template<typename MatType>
inline long long int calcMemorySize(const std::vector<std::vector<MatType> >& mats, std::vector<MatMemorySize>& memSizes)
{
    long long int size = 0;
    int num = mats.size();
    for (int i = 0; i < num; i++)
        size += calcMemorySize(mats[i], memSizes);
    return size;
}

inline long long int calcMemorySize(const std::vector<MatMemorySize>& memSizes)
{
    long long int size = 0;
    int numMems = memSizes.size();
    for (int i = 0; i < numMems; i++)
        size += memSizes[i].size;
    return size;
}

