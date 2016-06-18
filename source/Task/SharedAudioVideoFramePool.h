#pragma once

#include "AudioVideoProcessor.h"
#include <vector>
#include <memory>
#include <mutex>

class AudioVideoFramePool
{
public:

    AudioVideoFramePool() :
        hasInit(0)
    {}

    void clear()
    {
        std::lock_guard<std::mutex> lock(mtx);
        pool.clear();
        deep = avp::AudioVideoFrame2();
        hasInit = 0;
    }

    bool initAsAudioFramePool(int sampleType, int numChannels, int channelLayout, int numSamples)
    {
        clear();

        std::lock_guard<std::mutex> lock(mtx);

        deep.create(sampleType, numChannels, channelLayout, numSamples, -1LL, -1);

        if (deep.data)
        {
            hasInit = 1;
            return true;
        }
        else
        {
            clear();
            return false;
        }
    }

    bool initAsVideoFramePool(int pixelType, int width, int height)
    {
        clear();

        std::lock_guard<std::mutex> lock(mtx);

        deep.create(pixelType, width, height, -1LL, -1);

        if (deep.data)
        {
            hasInit = 1;
            return true;
        }
        else
        {
            clear();
            return false;
        }
    }

    bool get(avp::AudioVideoFrame2& frame)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!hasInit)
        {
            frame = avp::AudioVideoFrame2();
            return false;
        }

        int size = pool.size();
        int index = -1;
        for (int i = 0; i < size; i++)
        {
            if (pool[i].sdata.use_count() == 1)
            {
                index = i;
                break;
            }
        }
        if (index >= 0)
        {
            frame = pool[index];
            return true;
        }

        avp::AudioVideoFrame2 newFrame;
        if (deep.mediaType == avp::AUDIO)
            newFrame.create(deep.sampleType, deep.numChannels, deep.channelLayout, deep.numSamples, -1LL, -1);
        else
            newFrame.create(deep.pixelType, deep.width, deep.height, -1LL, -1);

        if (!newFrame.data)
        {
            frame = avp::AudioVideoFrame2();
            return false;
        }

        frame = newFrame;
        pool.push_back(newFrame);
        return true;
    }

    void shrink(int minSize)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!hasInit)
            return;

        int size = pool.size();
        if (size < minSize)
            return;

        int numNotInUse = 0;        
        for (int i = 0; i < size; i++)
        {
            if (pool[i].sdata.use_count() == 1)
                numNotInUse++;
        }
        if (numNotInUse == 0)
            return;

        int numInUse = size - numNotInUse;
        int numDelete = 0;
        if (numInUse >= minSize)
            numDelete = numNotInUse;
        else
            numDelete = size - minSize;
        if (numDelete == 0)
            return;

        std::vector<avp::AudioVideoFrame2>::iterator itr = pool.begin();
        for (; itr != pool.end();)
        {
            if (itr->sdata.use_count() == 1)
            {
                itr = pool.erase(itr);
                numDelete--;
                if (numDelete == 0)
                    return;
            }
            else
                ++itr;
        }
    }

    int size()
    {
        std::lock_guard<std::mutex> lock(mtx);
        return pool.size();
    }

private:
    avp::AudioVideoFrame2 deep;
    std::vector<avp::AudioVideoFrame2> pool;
    std::mutex mtx;
    int hasInit;
};