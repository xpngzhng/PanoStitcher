#pragma once

#include "AudioVideoProcessor.h"
#include "opencv2/gpu/gpu.hpp"
#include <vector>
#include <list>
#include <memory>
#include <mutex>

class StampedPinnedMemoryPool
{    
public:
    enum { DEFAULT_SIZE = 4, MAX_SIZE = 16 };

    StampedPinnedMemoryPool(int size = DEFAULT_SIZE) : 
        maxCapacity((size < DEFAULT_SIZE || size > MAX_SIZE) ? DEFAULT_SIZE : size), 
        currCapacity(0) 
    {}

    void clear()
    {
        {
            std::lock_guard<std::mutex> lock(mtxBuffer);
            pool.clear();
        }
        size = 0;
        currCapacity = 0;
    }

    bool push(const std::vector<avp::SharedAudioVideoFrame>& frames)
    {
        std::lock_guard<std::mutex> lock(mtxBuffer);
        std::shared_ptr<StampedPinnedMemory> ptrMemory;
        if (currCapacity < maxCapacity)
        {
            ptrMemory.reset(new StampedPinnedMemory);
            currCapacity++;
        }
        else
        {
            for (PoolType::iterator itr = pool.begin(); itr != pool.end(); ++itr)
            {
                ptrMemory = *itr;
                StampedPinnedMemory& stampedMem = *ptrMemory.get();
                // We only need to check buffer[0].refcount
                // because all CudaMem in buffer have the same (*refcount)
                if (stampedMem.buffer[0].refcount && (*stampedMem.buffer[0].refcount > 1))
                {
                    ptrMemory.reset();
                    continue;
                }
                else
                {
                    pool.erase(itr);
                    break;
                }
            }
            // The following case should never happen
            // since maxSize > 1 and only one thread use one item in pool
            if (!ptrMemory.get())
                return false;
        }
        
        if (size < currCapacity)
            size++;
        pool.push_back(ptrMemory);
        StampedPinnedMemory& stampedMem = *ptrMemory.get();
        int num = frames.size();
        stampedMem.buffer.resize(num);
        for (int i = 0; i < num; i++)
        {
            stampedMem.buffer[i].create(frames[i].height, frames[i].width, CV_8UC4);
            cv::Mat src(frames[i].height, frames[i].width, CV_8UC4, frames[i].data, frames[i].step);
            cv::Mat dst = stampedMem.buffer[i];
            src.copyTo(dst);
        }
        stampedMem.timeStamp = frames[0].timeStamp;
        
        return true;
    }

    bool pull(std::vector<cv::gpu::CudaMem>& mems, long long int& timeStamp)
    {
        if (!size)
        {
            mems.clear();
            timeStamp = -1LL;
            return false;
        }

        std::lock_guard<std::mutex> lock(mtxBuffer);
        StampedPinnedMemory& stampedMem = *(*pool.begin()).get();
        mems = stampedMem.buffer;
        timeStamp = stampedMem.timeStamp;
        size--;

        return true;
    }

private:
    struct StampedPinnedMemory
    {
        std::vector<cv::gpu::CudaMem> buffer;
        long long int timeStamp;
    };
    std::list<std::shared_ptr<StampedPinnedMemory> > pool;
    typedef std::list<std::shared_ptr<StampedPinnedMemory> > PoolType;
    int maxCapacity;
    int currCapacity;
    int size;
    std::mutex mtxBuffer;
};