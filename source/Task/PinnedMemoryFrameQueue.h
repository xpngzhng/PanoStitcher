#pragma once

#include "AudioVideoProcessor.h"
#include "opencv2/core/cuda.hpp"
#include <vector>
#include <list>
#include <memory>
#include <mutex>
#include <condition_variable>

class BoundedPinnedMemoryFrameQueue
{
public:
    enum { DEFAULT_SIZE = 4, MAX_SIZE = 16 };

    BoundedPinnedMemoryFrameQueue(int size = DEFAULT_SIZE) :
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
            mem.timeStamps.resize(num);
            for (int i = 0; i < num; i++)
            {
                mem.buffer[i].create(frames[i].height, frames[i].width, CV_8UC4);
                cv::Mat src(frames[i].height, frames[i].width, CV_8UC4, frames[i].data, frames[i].step);
                cv::Mat dst = mem.buffer[i].createMatHeader();
                src.copyTo(dst);
                mem.timeStamps[i] = frames[i].timeStamp;
            }            
            mem.waiting = 1;
        }

        cvNonEmpty.notify_one();

        return true;
    }

    bool pull(std::vector<cv::cuda::HostMem>& mems, std::vector<long long int>& timeStamps)
    {
        std::unique_lock<std::mutex> lock(mtxBuffer);
        cvNonEmpty.wait(lock, [this] {return !indexes.empty() || pass; });

        if (pass)
        {
            mems.clear();
            timeStamps.clear();
            return false;
        }

        int index = indexes.front();
        StampedPinnedMemory& mem = pool[index];
        mems = mem.buffer;
        timeStamps = mem.timeStamps;
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
        StampedPinnedMemory() : waiting(0) {}
        std::vector<cv::cuda::HostMem> buffer;
        std::vector<long long int> timeStamps;
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
