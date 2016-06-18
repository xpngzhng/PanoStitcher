#include "opencv2/core/cuda.hpp"

struct CudaLogoFilter
{
    CudaLogoFilter() : initSuccess(false), width(0), height(0) {}
    bool init(int width, int height);
    bool addLogo(cv::cuda::GpuMat& image) const;

    int width, height;
    cv::cuda::GpuMat logo;
    bool initSuccess;
};

void alphaBlend8UC4(cv::cuda::GpuMat& target, const cv::cuda::GpuMat& blender);

void cvtBGR32ToYUV420P(const cv::cuda::GpuMat& bgr32, cv::cuda::GpuMat& y, cv::cuda::GpuMat& u, cv::cuda::GpuMat& v);

void cvtBGR32ToNV12(const cv::cuda::GpuMat& bgr32, cv::cuda::GpuMat& y, cv::cuda::GpuMat& uv);