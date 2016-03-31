#pragma once

#include "opencv2/core/cuda.hpp"
#include <vector>
#include <memory>
#include <mutex>

class PinnedMemoryPool
{
public:
    PinnedMemoryPool() :
        rows(0), cols(0), type(0), hasInit(0)
    {
    }

    bool init(int rows_, int cols_, int type_)
    {
        clear();

        std::lock_guard<std::mutex> lock(mtx);

        cv::cuda::HostMem test;
        try
        {
            test.create(rows_, cols_, type_);
        }
        catch (...)
        {
            return false;
        }
        if (!test.data)
            return false;

        rows = rows_;
        cols = cols_;
        type = type_;

        hasInit = 1;
        return true;
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mtx);
        pool.clear();
    }

    bool get(cv::cuda::HostMem& mem)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!hasInit)
        {
            mem = cv::cuda::HostMem();
            return false;
        }

        int size = pool.size();
        int index = -1;
        for (int i = 0; i < size; i++)
        {
            if (pool[i].refcount && *pool[i].refcount == 1)
            {
                index = i;
                break;
            }
        }
        if (index >= 0)
        {
            mem = pool[index];
            return true;
        }

        cv::cuda::HostMem newMem(rows, cols, type);
        if (!newMem.data)
        {
            mem = cv::cuda::HostMem();
            return false;
        }

        mem = newMem;
        pool.push_back(newMem);
        return true;
    }
private:
    int rows, cols, type;
    std::vector<cv::cuda::HostMem> pool;
    std::mutex mtx;
    int hasInit;
};