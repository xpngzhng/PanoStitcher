#pragma once

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include <vector>
#include <string>
#include <memory>

enum PanoToolsImageType
{
    PTImageTypeRectlinear = 0,
    PTImageTypePanoramic = 1,
    PTImageTypeCircularFishEye = 2,
    PTImageTypeFullFrameFishEye = 3,
    PTImageTypeEquiRect = 4
};

struct PhotoParam
{
    enum ImageType
    {
        ImageTypeRectlinear = 0,
        ImageTypeFullFrameFishEye = 1,
        ImageTypeDrumFishEye = 2,
        ImageTypeCircularFishEye = 3
    };

    PhotoParam();

    int imageType;
    int cropMode;
    int cropX;
    int cropY;
    int cropWidth;
    int cropHeight;
    int circleX;
    int circleY;
    int circleR;
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

void loadPhotoParamFromXML(const std::string& fileName, std::vector<PhotoParam>& params);

void loadPhotoParamFromPTS(const std::string& fileName, std::vector<PhotoParam>& params);

bool loadPhotoParams(const std::string& fileName, std::vector<PhotoParam>& params);

void exportPhotoParamToXML(const std::string& fileName, const std::vector<PhotoParam>& params);

void rotateCamera(PhotoParam& param, double yaw, double pitch, double roll);

void rotateCameras(std::vector<PhotoParam>& params, double yaw, double pitch, double roll);

void rotatePhotoParamInXML(const std::string& src, const std::string& dst, double yaw, double pitch, double roll);

// NOTICE!!!!
// We want to remap fisheye image to equirect image.
// In this class, src refers to equirectangular image and dst refers to fisheye image or rectlinear image.
// Currently initInverse and inverseRemapImage only support dst image of type fisheye,
// rectlinear is not supported
// This kind of image expression is different from what we usually see in OpenCV.
struct Remap
{
public:
    void init(const PhotoParam& param, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
    void initInverse(const PhotoParam& param, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
    bool remapImage(double & x_dest, double & y_dest, double x_src, double y_src);
    bool inverseRemapImage(double x_dest, double y_dest, double & x_src, double & y_src);

public:
    void clear();
    int srcImageType;
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

void getReprojectMap32FAndMask(const PhotoParam& param, const cv::Size& srcSize, const cv::Size& dstSize,
    cv::Mat& dstSrcXMap, cv::Mat& dstSrcYMap, cv::Mat& mask);

void getReprojectMaps32FAndMasks(const std::vector<PhotoParam>& params, const cv::Size& srcSize, const cv::Size& dstSize,
    std::vector<cv::Mat>& dstSrcXMaps, std::vector<cv::Mat>& dstSrcYMaps, std::vector<cv::Mat>& masks);

void getReprojectMap16SAndWeight32SAndMask(const PhotoParam& param, const cv::Size& srcSize, const cv::Size& dstSize,
    cv::Mat& dstSrcMap, cv::Mat& weight, cv::Mat& mask);

void getReprojectMaps16SAndWeights32SAndMasks(const std::vector<PhotoParam>& params, const cv::Size& srcSize, const cv::Size& dstSize,
    std::vector<cv::Mat>& dstSrcMaps, std::vector<cv::Mat>& weights, std::vector<cv::Mat>& masks);

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

void reprojectWeightedAccumulateTo32F(const cv::Mat& src, cv::Mat& dst,
    const cv::Mat& dstSrcMap, const cv::Mat& weight);

void reprojectWeightedAccumulateParallelTo32F(const cv::Mat& src, cv::Mat& dst,
    const cv::Mat& dstSrcMap, const cv::Mat& weight);

void cudaGenerateReprojectMap(const PhotoParam& param,
    const cv::Size& srcSize, const cv::Size& dstSize, cv::cuda::GpuMat& xmap, cv::cuda::GpuMat& ymap);

void cudaGenerateReprojectMaps(const std::vector<PhotoParam>& params,
    const cv::Size& srcSize, const cv::Size& dstSize, std::vector<cv::cuda::GpuMat>& xmaps, std::vector<cv::cuda::GpuMat>& ymaps);

void cudaReproject(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
    const cv::cuda::GpuMat& xmap, const cv::cuda::GpuMat& ymap, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void cudaReprojectTo16S(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
    const cv::cuda::GpuMat& xmap, const cv::cuda::GpuMat& ymap, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void cudaReprojectWeightedAccumulateTo32F(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
    const cv::cuda::GpuMat& xmap, const cv::cuda::GpuMat& ymap, const cv::cuda::GpuMat& weight,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null());


// right left top bottom front back
enum CubeType
{
    CubeType6x1,
    CubeType3x2,
    CubeType180
};

void getEquiRectToCubeMap(cv::Mat& dstSrcMap, int equiRectHeight, int cubeHeight, int cubeType);

void getEquiRectToCubeMap(cv::Mat& dstSrcXMap, cv::Mat& dstSrcYMap, int equiRectHeight, int cubeHeight, int cubeType);
