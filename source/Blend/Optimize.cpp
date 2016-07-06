#include "ZBlend.h"
#include "ZReproject.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

inline cv::Vec3b toVec3b(const cv::Vec3d v)
{
    return cv::Vec3b(cv::saturate_cast<unsigned char>(v[0]),
        cv::saturate_cast<unsigned char>(v[1]),
        cv::saturate_cast<unsigned char>(v[2]));
}

inline cv::Vec3d toVec3d(const cv::Vec3b v)
{
    return cv::Vec3d(v[0], v[1], v[2]);
}

void calcGradImage(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(src.type() == CV_8UC3);
    cv::Mat blurred, grad;
    cv::GaussianBlur(src, blurred, cv::Size(5, 5), 1.0);
    cv::Laplacian(blurred, grad, CV_16S);
    cv::convertScaleAbs(grad, dst);
}

struct ValuePair
{
    int i, j;
    cv::Vec3b iVal, jVal;
    cv::Vec3d iValD, jValD;
    cv::Point iPos, jPos;
    cv::Point equiRectPos;
};

void getPointPairs(const std::vector<cv::Mat>& src, const std::vector<PhotoParam>& photoParams, std::vector<ValuePair>& pairs)
{
    int numImages = src.size();
    CV_Assert(photoParams.size() == numImages);

    int erWidth = 2048, erHeight = 1024;
    std::vector<Remap> remaps(numImages);
    for (int i = 0; i < numImages; i++)
        remaps[i].init(photoParams[i], erWidth, erHeight, src[0].cols, src[0].rows);

    std::vector<cv::Mat> grads(numImages);
    for (int i = 0; i < numImages; i++)
        calcGradImage(src[i], grads[i]);

    for (int i = 0; i < numImages; i++)
    {
        double maxVal, minVal;
        cv::minMaxLoc(grads[i], &minVal, &maxVal);
        printf("min %f, max %f\n", minVal, maxVal);
    }

    cv::Rect validRect(0, 0, src[0].cols, src[0].rows);

    pairs.clear();

    int gradThresh = 10;
    cv::RNG_MT19937 rng(cv::getTickCount());
    int numTrials = 4000;
    int expectNumPairs = 1000;
    int numPairs = 0;
    for (int t = 0; t < numTrials; t++)
    {
        int erx = rng.uniform(0, erWidth);
        int ery = rng.uniform(0, erHeight);
        for (int i = 0; i < numImages; i++)
        {
            int getPair = 0;
            double srcxid, srcyid;
            remaps[i].remapImage(srcxid, srcyid, erx, ery);
            cv::Point pti(srcxid, srcyid);
            if (validRect.contains(pti))
            {
                int gradValI = grads[i].at<unsigned char>(pti);
                if (gradValI < gradThresh)
                {
                    for (int j = 0; j < numImages; j++)
                    {
                        if (i == j)
                            continue;

                        double srcxjd, srcyjd;
                        remaps[j].remapImage(srcxjd, srcyjd, erx, ery);
                        cv::Point ptj(srcxjd, srcyjd);
                        if (validRect.contains(ptj))
                        {
                            int gradValJ = grads[j].at<unsigned char>(ptj);
                            if (gradValJ < gradThresh)
                            {
                                ValuePair pair;
                                pair.i = i;
                                pair.j = j;
                                pair.iPos = pti;
                                pair.jPos = ptj;
                                pair.iVal = src[i].at<cv::Vec3b>(pti);
                                pair.jVal = src[j].at<cv::Vec3b>(ptj);
                                pair.iValD = toVec3d(pair.iVal);
                                pair.jValD = toVec3d(pair.jVal);
                                pair.equiRectPos = cv::Point(erx, ery);
                                getPair = 1;
                                numPairs++;
                                pairs.push_back(pair);
                                break;
                            }
                        }
                    }
                }
            }
            if (getPair)
                break;
        }
        if (numPairs >= expectNumPairs)
            break;
    }

    std::vector<int> appearCount(numImages, 0);

    cv::Mat mask = cv::Mat::zeros(erHeight, erWidth, CV_8UC1);
    for (int i = 0; i < numPairs; i++)
    {
        mask.at<unsigned char>(pairs[i].equiRectPos) = 255;
        for (int k = 0; k < numImages; k++)
        {
            if (pairs[i].i == k)
                appearCount[k]++;
            if (pairs[i].j == k)
                appearCount[k]++;
        }
    }

    printf("num pairs found %d\n", numPairs);
    for (int i = 0; i < numImages; i++)
    {
        printf("%d appear %d times\n", i, appearCount[i]);
    }
    cv::imshow("mask", mask);
    cv::waitKey(0);

    std::vector<cv::Mat> show(numImages);
    for (int i = 0; i < numImages; i++)
        show[i] = src[i].clone();

    for (int i = 0; i < numPairs; i++)
    {
        for (int k = 0; k < numImages; k++)
        {
            if (pairs[i].i == k)
                cv::circle(show[k], pairs[i].iPos, 4, cv::Scalar(255), -1);
            if (pairs[i].j == k)
                cv::circle(show[k], pairs[i].jPos, 4, cv::Scalar(255), -1);
        }
    }

    for (int i = 0; i < numImages; i++)
    {
        char buf[64];
        sprintf(buf, "%d", i);
        cv::imshow(buf, show[i]);
    }
    cv::waitKey(0);
}

enum { OPTIMIZE_VAL_NUM = 12 };
enum { EMOR_COEFF_LENGTH = 5 };
enum { VIGNETT_COEFF_LENGTH = 4 };

struct ImageInfo
{
    ImageInfo()
    {

    }

    ImageInfo(const PhotoParam& param_, const cv::Size& size_)
    {
        memset(emorCoeffs, 0, sizeof(emorCoeffs));
        memset(radialVignettCoeffs, 0, sizeof(radialVignettCoeffs));
        radialVignettCoeffs[0] = 1;
        exposureExponent = 0;
        gamma = 0;
        whiteBalanceRed = 1;
        whiteBalanceBlue = 1;
        param = param_;
        size = size_;
    }

    double getExposure() const
    {
        return 1.0 / pow(2.0, exposureExponent);
    }

    void setExposure(double e)
    {
        exposureExponent = log2(1 / e);
    }

    void fromOutside(const double* x)
    {
        int index = 0;
        for (; index < EMOR_COEFF_LENGTH; index++)
            emorCoeffs[index] = x[index];
        for (; index < EMOR_COEFF_LENGTH + VIGNETT_COEFF_LENGTH; index++)
            radialVignettCoeffs[index - EMOR_COEFF_LENGTH] = x[index];
        exposureExponent = x[index++];
        whiteBalanceRed = x[index++];
        whiteBalanceBlue = x[index];
    }

    void toOutside(double* x) const
    {
        int index = 0;
        for (; index < EMOR_COEFF_LENGTH; index++)
            x[index] = emorCoeffs[index];
        for (; index < EMOR_COEFF_LENGTH + VIGNETT_COEFF_LENGTH; index++)
            x[index] = radialVignettCoeffs[index - EMOR_COEFF_LENGTH];
        x[index++] = exposureExponent;
        x[index++] = whiteBalanceRed;
        x[index] = whiteBalanceBlue;
    }
    
    double emorCoeffs[EMOR_COEFF_LENGTH];
    double radialVignettCoeffs[VIGNETT_COEFF_LENGTH];
    double exposureExponent;
    double whiteBalanceRed;
    double whiteBalanceBlue;
    double gamma = 0;
    PhotoParam param;
    cv::Size size;
};

void readFrom(std::vector<ImageInfo>& infos, const double* x)
{
    int numInfos = infos.size();
    for (int i = 0; i < numInfos; i++)
        infos[i].fromOutside(x + i * OPTIMIZE_VAL_NUM);
}

void writeTo(const std::vector<ImageInfo>& infos, double* x)
{
    int numInfos = infos.size();
    for (int i = 0; i < numInfos; i++)
        infos[i].toOutside(x + i * OPTIMIZE_VAL_NUM);
}

#include "emor.h"
struct Transform
{
    Transform()
    {

    }

    Transform(const ImageInfo& imageInfo)
    {
        memset(lut, 0, sizeof(lut));
        for (int i = 0; i < 256; i++)
        {
            double t = EMoR::f0[i * 4];
            for (int k = 0; k < EMOR_COEFF_LENGTH; k++)
                t += EMoR::h[k][i * 4] * imageInfo.emorCoeffs[k];
            lut[i] = cv::saturate_cast<unsigned char>(t * 255);
        }

        vigCenterX = imageInfo.size.width / 2;
        vigCenterY = imageInfo.size.height / 2;
        memcpy(vigCoeffs, imageInfo.radialVignettCoeffs, sizeof(vigCoeffs));
        radiusScale = 1.0 / sqrt(vigCenterX * vigCenterX + vigCenterY * vigCenterY);

        exposure = imageInfo.getExposure();

        whiteBalanceRed = imageInfo.whiteBalanceRed;
        whiteBalanceBlue = imageInfo.whiteBalanceBlue;
    }

    cv::Vec3d apply(const cv::Point& p, const cv::Vec3b& val) const
    {
        double scale = calcVigFactor(p) * exposure;
        int b = cv::saturate_cast<unsigned char>(val[0] * scale * whiteBalanceBlue);
        int g = cv::saturate_cast<unsigned char>(val[1] * scale);
        int r = cv::saturate_cast<unsigned char>(val[2] * scale * whiteBalanceRed);
        return cv::Vec3d(lut[b], lut[g], lut[r]);
    }

    cv::Vec3d applyInverse(const cv::Point& p, const cv::Vec3b& val) const
    {
        double scale = 1.0 / (calcVigFactor(p) * exposure);
        double b = invLUT(val[0]) * scale;
        double g = invLUT(val[1]) * scale;
        double r = invLUT(val[2]) * scale;
        r /= whiteBalanceRed;
        b /= whiteBalanceBlue;
        return cv::Vec3d(b, g, r);
    }

    cv::Vec3d apply(const cv::Point& p, const cv::Vec3d& val) const
    {
        double scale = calcVigFactor(p) * exposure;
        int b = cv::saturate_cast<unsigned char>(val[0] * scale * whiteBalanceBlue);
        int g = cv::saturate_cast<unsigned char>(val[1] * scale);
        int r = cv::saturate_cast<unsigned char>(val[2] * scale * whiteBalanceRed);
        return cv::Vec3d(lut[b], lut[g], lut[r]);
    }

    cv::Vec3d applyInverse(const cv::Point& p, const cv::Vec3d& val) const
    {
        double scale = 1.0 / (calcVigFactor(p) * exposure);
        double b = invLUT(val[0]) * scale;
        double g = invLUT(val[1]) * scale;
        double r = invLUT(val[2]) * scale;
        r /= whiteBalanceRed;
        b /= whiteBalanceBlue;
        return cv::Vec3d(b, g, r);
    }

    double calcVigFactor(const cv::Point& p) const
    {
        double diffx = (p.x - vigCenterX) * radiusScale;
        double diffy = (p.y - vigCenterY) * radiusScale;
        double vig = vigCoeffs[0];
        double r2 = diffx * diffx + diffy * diffy;
        double r = r2;
        for (int i = 1; i < VIGNETT_COEFF_LENGTH; i++)
        {
            vig += vigCoeffs[i] * r;
            r *= r2;
        }
        return vig;
    }

    double invLUT(double val) const
    {
        if (val <= 0)
            return 0;
        if (val >= 255)
            return 255;

        int lowIdx = 0, upIdx = 255;
        for (int i = 0; i < 255; i++)
        {
            if (lut[i] < val && lut[i + 1] >= val)
            {
                lowIdx = i;
                break;
            }
        }
        for (int i = 255; i > 0; i--)
        {
            if (lut[i - 1] <= val && lut[i] > val)
            {
                upIdx = i;
                break;
            }
        }
        if (lowIdx == upIdx)
            return lowIdx;

        double diff = lut[upIdx] - lut[lowIdx];
        double lambda = (val - lut[lowIdx]) / diff;
        return lowIdx * lambda + upIdx * (1 - lambda);
    }

    void enforceMonotonicity()
    {
        double val = lut[255];
        for (int i = 0; i < 255; i++)
        {
            if (lut[i] > val)
                lut[i] = val;
            if (lut[i + 1] < lut[i])
                lut[i + 1] = lut[i];
        }
    }

    enum { LUT_LENGTH = 256 };
    double lut[LUT_LENGTH];
    double vigCenterX, vigCenterY;
    double vigCoeffs[VIGNETT_COEFF_LENGTH];
    double radiusScale;
    double exposure;
    double whiteBalanceRed;
    double whiteBalanceBlue;
};

struct ExternData
{
    ExternData(std::vector<ImageInfo>& infos_, const std::vector<ValuePair>& pairs_) 
    : imageInfos(infos_), pairs(pairs_)
    {}
    std::vector<ImageInfo>& imageInfos;
    const std::vector<ValuePair>& pairs;
    double huberSigma;
    int errorFuncCallCount;
};

inline double weightHuber(double x, double sigma)
{
    if (x > sigma) 
    {
        x = sqrt(sigma* (2 * x - sigma));
    }
    return x;
}



void errorFunc(double* p, double* hx, int m, int n, void* data)
{
    ExternData* edata = (ExternData*)data;
    const std::vector<ImageInfo>& infos = edata->imageInfos;
    const std::vector<ValuePair>& pairs = edata->pairs;

    readFrom(edata->imageInfos, p);
    int numImages = infos.size();

    std::vector<Transform> transforms(numImages);
    for (int i = 0; i < numImages; i++)
        transforms[i] = Transform(infos[i]);

    int index = 0;
    for (int i = 0; i < numImages; i++)
    {
        Transform& trans = transforms[i];
        double err = 0;
        for (int j = 0; j < 255; j++)
        {
            if (trans.lut[j] > trans.lut[j + 1])
            {
                double diff = trans.lut[j] - trans.lut[j + 1];
                err += diff * diff * 256;
            }
        }
        trans.enforceMonotonicity();
        hx[index++] = err;
    }

    double huberSigma = edata->huberSigma;

    double sqrErr = 0;
    int numPairs = pairs.size();
    for (int i = 0; i < numPairs; i++)
    {
        const ValuePair& pair = pairs[i];
        
        if ((double(pair.iVal[0]) - pair.jVal[0]) > 10 ||
            (double(pair.iVal[1]) - pair.jVal[1]) > 10 ||
            (double(pair.iVal[2]) - pair.jVal[2]) > 10)
        {
            //printf("diff large\n");
        }

        cv::Vec3d lightI = transforms[pair.i].applyInverse(pair.iPos, pair.iVal);
        cv::Vec3d valIInJ = transforms[pair.j].apply(pair.jPos, lightI);
        cv::Vec3d errI = toVec3d(pair.jVal) - valIInJ;

        cv::Vec3d lightJ = transforms[pair.j].applyInverse(pair.jPos, pair.jVal);
        cv::Vec3d valJInI = transforms[pair.i].apply(pair.iPos, lightJ);
        cv::Vec3d errJ = toVec3d(pair.iVal) - valJInI;

        for (int j = 0; j < 3; j++)
        {
            hx[index++] = weightHuber(abs(errI[j]), huberSigma);
            hx[index++] = weightHuber(abs(errJ[j]), huberSigma);
        }        

        sqrErr += errI.dot(errI);
        sqrErr += errJ.dot(errJ);
    }

    edata->errorFuncCallCount++;

    std::vector<double> pp(m), hxhx(n);
    memcpy(pp.data(), p, m * sizeof(double));
    memcpy(hxhx.data(), hx, n * sizeof(double));

    printf("call count %d, sqr err = %f\n", edata->errorFuncCallCount, sqrErr);
}

#include "levmar.h"

void optimize(const std::vector<PhotoParam>& photoParams, const std::vector<ValuePair>& valuePairs, const cv::Size& imageSize)
{
    int size = photoParams.size();
    std::vector<ImageInfo> imageInfos(size);
    for (int i = 0; i < size; i++)
    {
        ImageInfo info(photoParams[i], imageSize);
        imageInfos[i] = info;
    }

    int ret;
    //double opts[LM_OPTS_SZ];
    double info[LM_INFO_SZ];

    // parameters
    int m = size * OPTIMIZE_VAL_NUM;
    std::vector<double> p(m, 0.0);

    // vector for errors
    int n = 2 * 3 * valuePairs.size() + size;
    std::vector<double> x(n, 0.0);

    writeTo(imageInfos, p.data());

    // covariance matrix at solution
    cv::Mat cov(m, m, CV_64FC1);
    // TODO: setup optimisation options with some good defaults.
    double optimOpts[5];

    optimOpts[0] = 1E-03;  // init mu
    // stop thresholds
    optimOpts[1] = 1e-5;   // ||J^T e||_inf
    optimOpts[2] = 1e-5;   // ||Dp||_2
    optimOpts[3] = 1e-1;   // ||e||_2
    // difference mode
    optimOpts[4] = LM_DIFF_DELTA;

    int maxIter = 300;

    ExternData edata(imageInfos, valuePairs);
    edata.huberSigma = 5;
    edata.errorFuncCallCount = 0;

    ret = dlevmar_dif(&errorFunc, &(p[0]), &(x[0]), m, n, maxIter, optimOpts, info, NULL, (double*)cov.data, &edata);  // no jacobian
    // copy to source images (data.m_imgs)
    readFrom(imageInfos, p.data());

    int a = 0;
}

int main()
{
    //std::vector<std::string> paths;
    //paths.push_back("F:\\panoimage\\919-4\\snapshot0(2).bmp");
    //paths.push_back("F:\\panoimage\\919-4\\snapshot1(2).bmp");
    //paths.push_back("F:\\panoimage\\919-4\\snapshot2(2).bmp");
    //paths.push_back("F:\\panoimage\\919-4\\snapshot3(2).bmp");

    std::vector<std::string> imagePaths;
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\0.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\1.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\2.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\3.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\4.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\5.jpg");

    int numImages = imagePaths.size();
    std::vector<cv::Mat> src(numImages);
    for (int i = 0; i < numImages; i++)
        src[i] = cv::imread(imagePaths[i]);

    std::vector<PhotoParam> params;
    //loadPhotoParams("E:\\Projects\\GitRepo\\panoLive\\PanoLive\\PanoLive\\PanoLive\\201603260848.vrdl", params);
    //loadPhotoParamFromXML("F:\\panoimage\\919-4\\vrdl201606231708.xml", params);
    loadPhotoParamFromXML("F:\\panoimage\\zhanxiang\\zhanxiang.xml", params);
    double PI = 3.1415926;
    rotateCameras(params, 0, 35.264 / 180 * PI, PI / 4);

    std::vector<ValuePair> pairs;
    getPointPairs(src, params, pairs);

    optimize(params, pairs, src[0].size());

    return 0;

    std::vector<cv::Mat> srcGrad(numImages);
    for (int i = 0; i < numImages; i++)
        calcGradImage(src[i], srcGrad[i]);

    cv::Size dstSize(2048, 1024);
    std::vector<cv::Mat> maps, masks;
    getReprojectMapsAndMasks(params, src[0].size(), dstSize, maps, masks);

    std::vector<cv::Mat> images;
    reprojectParallel(src, images, maps);
    cv::Mat dst;
    for (int i = 0; i < numImages; i++)
    {
        cv::imshow("image", images[i]);
        calcGradImage(images[i], dst);
        cv::imshow("grad", dst);
        cv::waitKey(0);
    }

    

    return 0;
}
