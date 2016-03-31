#pragma once

#include "AudioVideoProcessor.h"
#include "opencv2/gpu/gpu.hpp"
#include <vector>
#include <list>
#include <memory>
#include <mutex>
#include <condition_variable>

class StampedPinnedMemoryPool
{
public:
    enum { DEFAULT_SIZE = 4, MAX_SIZE = 16 };

    StampedPinnedMemoryPool(int size = DEFAULT_SIZE) :
        maxCapacity((size < DEFAULT_SIZE || size > MAX_SIZE) ? DEFAULT_SIZE : size),
        currCapacity(0), 
        pass(0)
    {}

    void clear()
    {
        std::lock_guard<std::mutex> lock(mtxBuffer);
        pool.clear();
        indexes.clear();
        size = 0;
        currCapacity = 0;
        pass = 0;
    }

    bool push(const std::vector<avp::SharedAudioVideoFrame>& frames)
    {
        {
            std::lock_guard<std::mutex> lock(mtxBuffer);

            // First check whether pool has available memory
            int poolSize = pool.size();
            int availIndex = -1;
            for (int i = 0; i < poolSize; i++)
            {
                if (!pool[i].waiting &&
                    (!pool[i].buffer[0].refcount || (*pool[i].buffer[0].refcount == 1)))
                {
                    availIndex = i;
                    break;
                }
            }

            if (availIndex < 0)
            {
                if (poolSize < maxCapacity)
                {
                    pool.push_back(StampedPinnedMemory());
                    availIndex = poolSize;
                }
                else
                {
                    availIndex = indexes.front();
                    indexes.pop_front();
                }
            }
            indexes.push_back(availIndex);

            StampedPinnedMemory& mem = pool[availIndex];
            int num = frames.size();
            mem.buffer.resize(num);
            for (int i = 0; i < num; i++)
            {
                mem.buffer[i].create(frames[i].height, frames[i].width, CV_8UC4);
                cv::Mat src(frames[i].height, frames[i].width, CV_8UC4, frames[i].data, frames[i].step);
                cv::Mat dst = mem.buffer[i];
                src.copyTo(dst);
            }
            mem.timeStamp = frames[0].timeStamp;
            mem.waiting = 1;

            //for (std::list<int>::const_iterator itr = indexes.cbegin(), itrEnd = indexes.cend(); itr != itrEnd; ++itr)
            //    printf("%lld ", pool[*itr].timeStamp);
            //printf("\n");
        }

        cvNonEmpty.notify_one();

        return true;
    }

    bool pull(std::vector<cv::gpu::CudaMem>& mems, long long int& timeStamp)
    {
        std::unique_lock<std::mutex> lock(mtxBuffer);
        cvNonEmpty.wait(lock, [this] {return !indexes.empty() || pass; });

        if (pass)
        {
            mems = std::vector<cv::gpu::CudaMem>();
            timeStamp = -1LL;
            return false;
        }

        int index = indexes.front();
        StampedPinnedMemory& mem = pool[index];
        mems = mem.buffer;
        timeStamp = mem.timeStamp;
        mem.waiting = 0;
        indexes.pop_front();

        return true;
    }

    void stop()
    {
        pass = 1;
        cvNonEmpty.notify_one();
    }

    void resume()
    {
        pass = 0;
    }

private:
    struct StampedPinnedMemory
    {
        StampedPinnedMemory() : timeStamp(-1LL), waiting(0) {}
        std::vector<cv::gpu::CudaMem> buffer;
        long long int timeStamp;
        int waiting;
    };
    std::vector<StampedPinnedMemory> pool;
    std::list<int> indexes;
    int maxCapacity;
    int currCapacity;
    int size;
    std::mutex mtxBuffer;
    std::condition_variable cvNonEmpty;
    int pass;
};