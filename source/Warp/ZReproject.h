#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <vector>
#include <string>
#include <memory>

struct PhotoParam
{
    int imageType;
    int cropMode;
    int cropX;
    int cropY;
    int cropWidth;
    int cropHeight;
    double hfov;
    double vfov;
    double alpha;
    double beta;
    double gamma;
    double shiftX;
    double shiftY;
    double shearX;
    double shearY;
    double yaw;
    double pitch;
    double roll;
};

void loadPhotoParamFromXML(const std::string& fileName, PhotoParam& param);

void loadPhotoParamFromXML(const std::string& fileName, std::vector<PhotoParam>& params);

void loadPhotoParamFromPTS(const std::string& fileName, std::vector<PhotoParam>& params);

void exportPhotoParamToXML(const std::string& fileName, const std::vector<PhotoParam>& params);

void rotateCamera(PhotoParam& param, double yaw, double pitch, double roll);

void rotateCameras(std::vector<PhotoParam>& params, double yaw, double pitch, double roll);

void rotatePhotoParamInXML(const std::string& src, const std::string& dst, double yaw, double pitch, double roll);

// NOTICE!!!!
// We want to remap fisheye image to equirect image.
// In this class, src refers to equirect image and dst refers to fisheye image.
// This kind of image expression is different from what we usually see in OpenCV.
struct Remap
{
public:
    void init(const PhotoParam& param, int srcWidth, int srcHeight);
    void initInverse(const PhotoParam& param, int srcWidth, int srcHeight);
    bool remapImage(double & x_dest, double & y_dest, double x_src, double y_src);
    bool inverseRemapImage(double x_dest, double y_dest, double & x_src, double & y_src);

public:
    void clear();
    double srcTX, srcTY;
    double destTX, destTY;
    struct MakeParam
    {
        double scale[2];
        double shear[2];
        double rot[2];
        void *perspect[2];
        double rad[6];
        double mt[3][3];
        double distance;
        double horizontal;
        double vertical;
    };
    MakeParam mp;
};

void getReprojectMapAndMask(const PhotoParam& param, 
    const cv::Size& srcSize, const cv::Size& dstSize, cv::Mat& dstSrcMap, cv::Mat& mask);

void getReprojectMapsAndMasks(const std::vector<PhotoParam>& params,
    const cv::Size& srcSize, const cv::Size& dstSize, std::vector<cv::Mat>& dstSrcMaps, std::vector<cv::Mat>& masks);

void reproject(const cv::Mat& src, cv::Mat& dst, cv::Mat& mask, 
    const PhotoParam& param, const cv::Size& dstSize);

void reproject(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, std::vector<cv::Mat>& masks, 
    const std::vector<PhotoParam>& params, const cv::Size& dstSize);

void reproject(const cv::Mat& src, cv::Mat& dst, const cv::Mat& dstSrcMap);

void reproject(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, const std::vector<cv::Mat>& dstSrcMaps);

void reprojectParallel(const cv::Mat& src, cv::Mat& dst, const cv::Mat& dstSrcMap);

void reprojectParallel(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, const std::vector<cv::Mat>& dstSrcMaps);

void reprojectParallelTo16S(const cv::Mat& src, cv::Mat& dst, const cv::Mat& dstSrcMap);

void reprojectParallelTo16S(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, const std::vector<cv::Mat>& dstSrcMaps);

void cudaGenerateReprojectMap(const PhotoParam& param,
    const cv::Size& srcSize, const cv::Size& dstSize, cv::gpu::GpuMat& xmap, cv::gpu::GpuMat& ymap);

void cudaGenerateReprojectMaps(const std::vector<PhotoParam>& params,
    const cv::Size& srcSize, const cv::Size& dstSize, std::vector<cv::gpu::GpuMat>& xmaps, std::vector<cv::gpu::GpuMat>& ymaps);

void cudaReproject(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst,
    const cv::gpu::GpuMat& xmap, const cv::gpu::GpuMat& ymap, cv::gpu::Stream& stream = cv::gpu::Stream::Null());

void cudaReprojectTo16S(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst,
    const cv::gpu::GpuMat& xmap, const cv::gpu::GpuMat& ymap, cv::gpu::Stream& stream = cv::gpu::Stream::Null());
