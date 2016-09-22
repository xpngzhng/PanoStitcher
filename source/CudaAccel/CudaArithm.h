#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/cuda.hpp"

int countNonZero8UC1(const cv::cuda::GpuMat& mat);

// c(y, x) = a(y, x) | b(y, x)
void or8UC1(const cv::cuda::GpuMat& a, const cv::cuda::GpuMat& b, cv::cuda::GpuMat& c);

void and8UC1(const cv::cuda::GpuMat& a, const cv::cuda::GpuMat& b, cv::cuda::GpuMat& c);

void not8UC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);

void subtract8UC1(const cv::cuda::GpuMat& a, const cv::cuda::GpuMat& b, cv::cuda::GpuMat& c);