#pragma once

#include "CompileControl.h"
#if COMPILE_DISCRETE_OPENCL

#include "DiscreteOpenCLInterface.h"
#include "AudioVideoProcessor.h"
#include <mutex>

struct DOclWatermarkFilter
{
    DOclWatermarkFilter() : initSuccess(false), width(0), height(0) {}
    bool init(int width, int height);
    bool addWatermark(docl::GpuMat& image) const;
    void clear();

    int width, height;
    docl::GpuMat logo;
    bool initSuccess;
};

struct DOclLogoFilter
{
    DOclLogoFilter() : initSuccess(false), width(0), height(0) {}
    bool init(const std::string& logoFileName, int hFov, int width, int height);
    bool addLogo(docl::GpuMat& image) const;
    void clear();

    int width, height;
    docl::GpuMat logo;
    bool initSuccess;
};

struct DOclMixedAudioVideoFrame
{
    DOclMixedAudioVideoFrame() {}

    DOclMixedAudioVideoFrame(const avp::AudioVideoFrame2& audio) : frame(audio) {}

    DOclMixedAudioVideoFrame(int pixelType, int width, int height, int bufferFlag, int mapFlag)
    {
        if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0)
            return;

        if (pixelType == avp::PixelTypeYUV420P)
        {
            planes[0] = docl::HostMem(height, width, CV_8UC1, bufferFlag, mapFlag);
            planes[1] = docl::HostMem(height / 2, width / 2, CV_8UC1, bufferFlag, mapFlag);
            planes[2] = docl::HostMem(height / 2, width / 2, CV_8UC1, bufferFlag, mapFlag);
            unsigned char* data[4] = { planes[0].data, planes[1].data, planes[2].data, 0 };
            int steps[4] = { planes[0].step, planes[1].step, planes[2].step, 0 };
            frame = avp::AudioVideoFrame2(data, steps, pixelType, width, height, -1LL);
        }
        else if (pixelType == avp::PixelTypeNV12)
        {
            planes[0] = docl::HostMem(height, width, CV_8UC1, bufferFlag, mapFlag);
            planes[1] = docl::HostMem(height / 2, width, CV_8UC1, bufferFlag, mapFlag);
            unsigned char* data[4] = { planes[0].data, planes[1].data, 0, 0 };
            int steps[4] = { planes[0].step, planes[1].step, 0, 0 };
            frame = avp::AudioVideoFrame2(data, steps, pixelType, width, height, -1LL);
        }
        else if (pixelType == avp::PixelTypeBGR32)
        {
            planes[0] = docl::HostMem(height, width, CV_8UC4, bufferFlag, mapFlag);
            unsigned char* data[4] = { planes[0].data, 0, 0, 0 };
            int steps[4] = { planes[0].step, 0, 0, 0 };
            frame = avp::AudioVideoFrame2(data, steps, pixelType, width, height, -1LL);
        }
        else if (pixelType == avp::PixelTypeBGR24)
        {
            planes[0] = docl::HostMem(height, width, CV_8UC3, bufferFlag, mapFlag);
            unsigned char* data[4] = { planes[0].data, 0, 0, 0 };
            int steps[4] = { planes[0].step, 0, 0, 0 };
            frame = avp::AudioVideoFrame2(data, steps, pixelType, width, height, -1LL);
        }
    }

    docl::HostMem planes[3];
    avp::AudioVideoFrame2 frame;
};

class DOclHostMemVideoFrameMemoryPool
{
public:
    DOclHostMemVideoFrameMemoryPool() :
        pixelType(0), width(0), height(0), hasInit(0)
    {
    }

    bool init(int pixelType_, int width_, int height_, int bufferFlag_, int mapFlag_)
    {
        clear();

        std::lock_guard<std::mutex> lock(mtx);

        DOclMixedAudioVideoFrame test;
        try
        {
            test = DOclMixedAudioVideoFrame(pixelType_, width_, height_, bufferFlag_, mapFlag_);
        }
        catch (...)
        {
            return false;
        }
        if (!test.planes[0].data)
            return false;

        pixelType = pixelType_;
        width = width_;
        height = height_;
        bufferFlag = bufferFlag_;
        mapFlag = mapFlag_;

        hasInit = 1;
        return true;
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mtx);
        pool.clear();
    }

    bool get(DOclMixedAudioVideoFrame& mem)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!hasInit)
        {
            mem = DOclMixedAudioVideoFrame();
            return false;
        }

        int size = pool.size();
        int index = -1;
        for (int i = 0; i < size; i++)
        {
            if (pool[i].planes[0].mem && pool[i].planes[0].smem.use_count() == 1)
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

        DOclMixedAudioVideoFrame newMem(pixelType, width, height, bufferFlag, mapFlag);
        if (!newMem.planes[0].data)
        {
            mem = DOclMixedAudioVideoFrame();
            return false;
        }

        mem = newMem;
        pool.push_back(newMem);
        return true;
    }
private:
    int pixelType, width, height;
    int bufferFlag, mapFlag;
    std::vector<DOclMixedAudioVideoFrame> pool;
    std::mutex mtx;
    int hasInit;
};

#endif