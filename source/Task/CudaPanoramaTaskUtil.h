#pragma once

#include "AudioVideoProcessor.h"
#include "opencv2/core/cuda.hpp"
#include <mutex>

struct CudaWatermarkFilter
{
    CudaWatermarkFilter() : initSuccess(false), width(0), height(0) {}
    bool init(int width, int height);
    bool addWatermark(cv::cuda::GpuMat& image) const;
    void clear();

    int width, height;
    cv::cuda::GpuMat logo;
    bool initSuccess;
};

struct CudaLogoFilter
{
    CudaLogoFilter() : initSuccess(false), width(0), height(0) {}
    bool init(const std::string& logoFileName, int hFov, int width, int height);
    bool addLogo(cv::cuda::GpuMat& image) const;
    void clear();

    int width, height;
    cv::cuda::GpuMat logo;
    bool initSuccess;
};

void alphaBlend8UC4(cv::cuda::GpuMat& target, const cv::cuda::GpuMat& blender);

void cvtBGR32ToYUV420P(const cv::cuda::GpuMat& bgr32, cv::cuda::GpuMat& y, cv::cuda::GpuMat& u, cv::cuda::GpuMat& v,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void cvtBGR32ToNV12(const cv::cuda::GpuMat& bgr32, cv::cuda::GpuMat& y, cv::cuda::GpuMat& uv,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void cvtYUV420PToBGR32(const cv::cuda::GpuMat& y, const cv::cuda::GpuMat& u, const cv::cuda::GpuMat& v, cv::cuda::GpuMat& bgr32,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void cvtNV12ToBGR32(const cv::cuda::GpuMat& y, const cv::cuda::GpuMat& uv, cv::cuda::GpuMat& bgr32,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void resize8UC4(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dstSize);

struct CudaMixedAudioVideoFrame
{
    CudaMixedAudioVideoFrame() {}

    CudaMixedAudioVideoFrame(const avp::AudioVideoFrame2& audio) : frame(audio) {}

    CudaMixedAudioVideoFrame(int pixelType, int width, int height)
    {
        if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0)
            return;

        if (pixelType == avp::PixelTypeYUV420P)
        {
            planes[0] = cv::cuda::HostMem(height, width, CV_8UC1);
            planes[1] = cv::cuda::HostMem(height / 2, width / 2, CV_8UC1);
            planes[2] = cv::cuda::HostMem(height / 2, width / 2, CV_8UC1);
            unsigned char* data[4] = { planes[0].data, planes[1].data, planes[2].data, 0 };
            int steps[4] = { planes[0].step, planes[1].step, planes[2].step, 0 };
            frame = avp::AudioVideoFrame2(data, steps, pixelType, width, height, -1LL);
        }
        else if (pixelType == avp::PixelTypeNV12)
        {
            planes[0] = cv::cuda::HostMem(height, width, CV_8UC1);
            planes[1] = cv::cuda::HostMem(height / 2, width, CV_8UC1);
            unsigned char* data[4] = { planes[0].data, planes[1].data, 0, 0 };
            int steps[4] = { planes[0].step, planes[1].step, 0, 0 };
            frame = avp::AudioVideoFrame2(data, steps, pixelType, width, height, -1LL);
        }
        else if (pixelType == avp::PixelTypeBGR32)
        {
            planes[0] = cv::cuda::HostMem(height, width, CV_8UC4);
            unsigned char* data[4] = { planes[0].data, 0, 0, 0 };
            int steps[4] = { planes[0].step, 0, 0, 0 };
            frame = avp::AudioVideoFrame2(data, steps, pixelType, width, height, -1LL);
        }
        else if (pixelType == avp::PixelTypeBGR24)
        {
            planes[0] = cv::cuda::HostMem(height, width, CV_8UC3);
            unsigned char* data[4] = { planes[0].data, 0, 0, 0 };
            int steps[4] = { planes[0].step, 0, 0, 0 };
            frame = avp::AudioVideoFrame2(data, steps, pixelType, width, height, -1LL);
        }
    }

    cv::cuda::HostMem planes[3];
    avp::AudioVideoFrame2 frame;
};

class CudaHostMemVideoFrameMemoryPool
{
public:
    CudaHostMemVideoFrameMemoryPool() :
        pixelType(0), width(0), height(0), hasInit(0)
    {
    }

    bool init(int pixelType_, int width_, int height_)
    {
        clear();

        std::lock_guard<std::mutex> lock(mtx);

        CudaMixedAudioVideoFrame test;
        try
        {
            test = CudaMixedAudioVideoFrame(pixelType_, width_, height_);
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

        hasInit = 1;
        return true;
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mtx);
        pool.clear();
    }

    bool get(CudaMixedAudioVideoFrame& mem)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!hasInit)
        {
            mem = CudaMixedAudioVideoFrame();
            return false;
        }

        int size = pool.size();
        int index = -1;
        for (int i = 0; i < size; i++)
        {
            if (pool[i].planes[0].refcount && *pool[i].planes[0].refcount == 1)
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

        CudaMixedAudioVideoFrame newMem(pixelType, width, height);
        if (!newMem.planes[0].data)
        {
            mem = CudaMixedAudioVideoFrame();
            return false;
        }

        mem = newMem;
        pool.push_back(newMem);
        return true;
    }
private:
    int pixelType, width, height;
    std::vector<CudaMixedAudioVideoFrame> pool;
    std::mutex mtx;
    int hasInit;
};