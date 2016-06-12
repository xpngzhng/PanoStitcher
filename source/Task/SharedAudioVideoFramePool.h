#pragma once

#include "AudioVideoProcessor.h"
#include <vector>
#include <memory>
#include <mutex>

class SharedAudioVideoFramePool
{
public:

    SharedAudioVideoFramePool() :
        hasInit(0)
    {}

    void clear()
    {
        std::lock_guard<std::mutex> lock(mtx);
        pool.clear();
        deep = avp::SharedAudioVideoFrame();
        hasInit = 0;
    }

    bool initAsAudioFramePool(int sampleType, int numChannels, int channelLayout, int numSamples)
    {
        clear();

        std::lock_guard<std::mutex> lock(mtx);

        deep = avp::sharedAudioFrame(sampleType, numChannels, channelLayout, numSamples, -1LL);

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

        deep = avp::sharedVideoFrame(pixelType, width, height, -1LL);

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

    bool get(avp::SharedAudioVideoFrame& frame)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!hasInit)
        {
            frame = avp::SharedAudioVideoFrame();
            return false;
        }

        int size = pool.size();
        int index = -1;
        for (int i = 0; i < size; i++)
        {
            if (pool[i].sharedData.use_count() == 1)
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

        avp::SharedAudioVideoFrame newFrame;
        if (deep.mediaType == avp::AUDIO)
            newFrame = avp::sharedAudioFrame(deep.sampleType, deep.numChannels, deep.channelLayout, deep.numSamples, -1LL);
        else
            newFrame = avp::sharedVideoFrame(deep.pixelType, deep.width, deep.height, -1LL);

        if (!newFrame.data)
        {
            frame = avp::SharedAudioVideoFrame();
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
            if (pool[i].sharedData.use_count() == 1)
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

        std::vector<avp::SharedAudioVideoFrame>::iterator itr = pool.begin();
        for (; itr != pool.end();)
        {
            if (itr->sharedData.use_count() == 1)
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
    avp::SharedAudioVideoFrame deep;
    std::vector<avp::SharedAudioVideoFrame> pool;
    std::mutex mtx;
    int hasInit;
};