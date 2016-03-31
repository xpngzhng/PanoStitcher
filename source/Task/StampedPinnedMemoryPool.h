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

        int c = 0;
        for (PoolType::iterator itr = pool.begin(); itr != pool.end(); ++itr)
            c++;

        if (c > maxCapacity)
            return false;

        std::shared_ptr<StampedPinnedMemory> ptrMemory;
        if (currCapacity < maxCapacity)
        {
            ptrMemory.reset(new StampedPinnedMemory);
            // There may be something wrong in the following line
            pool.push_back(ptrMemory);
            currCapacity++;
            size++;
            printf("not full\n");
        }
        else
        {
            // If currCapacity == maxCapacity, we cannot allocate more memory
            // We can only change allocated memory to put new frames
            if (size == maxCapacity)
            {
                // If size == maxCapacity, erase the smallest time stamp item
                // append the pointed memory to the end of the list
                // newly arriving frames will be copied to it
                ptrMemory = pool.front();
                pool.pop_front();
                pool.push_back(ptrMemory);
            }
            else
            {
                PoolType::iterator itr = pool.begin();
                // Go to the maybe available memory
                // The first item is just the size-th item (counted from zero) of the list
                for (int i = 0; i < size; i++)
                    ++itr;
                PoolType::iterator itrEnd = itr;
                bool foundAvailable = false;
                for (; itr != pool.end(); ++itr)
                {
                    ptrMemory = *itr;
                    StampedPinnedMemory& stampedMem = *ptrMemory.get();
                    if (!stampedMem.buffer[0].refcount || (*stampedMem.buffer[0].refcount == 1))
                    {
                        foundAvailable = true;
                        break;
                    }
                }
                if (foundAvailable)
                    size++;
                // If we cannot find available memory at the tail part of the list,
                // We should erase the smallset time stamp item
                if (!foundAvailable)
                {
                    ptrMemory = pool.front();
                    pool.pop_front();
                    pool.insert(itrEnd, ptrMemory);
                }
            }
            
            // The following case should never happen
            // since maxSize > 1 and only one thread use one item in pool
            if (!ptrMemory.get())
                return false;
        }

        //pool.push_back(ptrMemory);
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

        for (PoolType::iterator itr = pool.begin(); itr != pool.end(); ++itr)
            printf("%lld ", (*itr)->timeStamp);
        printf("\n");
        
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
        std::shared_ptr<StampedPinnedMemory> ptrMemory = *pool.begin();
        pool.erase(pool.begin());
        // Output memory is pushed to the end of the list
        pool.push_back(ptrMemory);
        StampedPinnedMemory& stampedMem = *ptrMemory.get();
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