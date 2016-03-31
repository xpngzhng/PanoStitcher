#pragma once

#include <opencv2/core/core.hpp>
#include <memory>
#include <string>

class RicohPanoramaRender
{
public:
    RicohPanoramaRender();
    bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize);
    void render(const cv::Mat& src, cv::Mat& dst);
private:
    struct Impl;
    std::shared_ptr<Impl> ptrImpl;
};

class DetuPanoramaRender
{
public:
    DetuPanoramaRender();
    bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize);
    void render(const cv::Mat& src, cv::Mat& dst);
private:
    struct Impl;
    std::shared_ptr<Impl> ptrImpl;
};

class PanoramaRender
{
public:
    virtual ~PanoramaRender() {};
    virtual bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize) = 0;
    virtual bool render(const std::vector<cv::Mat>& src, cv::Mat& dst) = 0;
};

class DualGoProPanoramaRender : public PanoramaRender
{
public:
    DualGoProPanoramaRender();
    ~DualGoProPanoramaRender() {};
    bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const std::vector<cv::Mat>& src, cv::Mat& dst);
private:
    struct Impl;
    std::shared_ptr<Impl> ptrImpl;
};

class CPUMultiCameraPanoramaRender : public PanoramaRender
{
public:
    CPUMultiCameraPanoramaRender();
    ~CPUMultiCameraPanoramaRender() {};
    bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const std::vector<cv::Mat>& src, cv::Mat& dst);
private:
    struct Impl;
    std::shared_ptr<Impl> ptrImpl;
};

class CudaMultiCameraPanoramaRender : public PanoramaRender
{
public:
    CudaMultiCameraPanoramaRender();
    ~CudaMultiCameraPanoramaRender() {};
    bool prepare(const std::string& path, const cv::Size& srcSize, const cv::Size& dstSize);
    bool render(const std::vector<cv::Mat>& src, cv::Mat& dst);
private:
    struct Impl;
    std::shared_ptr<Impl> ptrImpl;
};