#include "ZBlend.h"
#include "ZBlendAlgo.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <map>
#include <iostream>

void getIntsctMasksAroundDistTransSeams(const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& outMasks)
{
    std::vector<cv::Mat> extendMasks;
    getExtendedMasks(masks, 100, extendMasks);

    int numImages = masks.size();
    outMasks.resize(numImages);
    cv::Mat temp;
    for (int i = 0; i < numImages; i++)
    {
        outMasks[i] = cv::Mat::zeros(masks[i].size(), CV_8UC1);
        for (int j = 0; j < numImages; j++)
        {
            if (i == j)
                continue;
            temp = extendMasks[i] & extendMasks[j];
            outMasks[i] |= temp;
        }
    }
}

void isGradSmall(const cv::Mat& image, int thresh, cv::Mat& mask, cv::Mat& blurred, cv::Mat& grad16S)
{
    CV_Assert(image.type() == CV_8UC1 && thresh > 0);
    cv::GaussianBlur(image, blurred, cv::Size(3, 3), 1.0);
    cv::Laplacian(blurred, grad16S, CV_16S);

    int rows = image.rows, cols = image.cols;
    mask.create(rows, cols, CV_8UC1);
    int minusThresh = -thresh;
    for (int i = 0; i < rows; i++)
    {
        const short* ptrGrad = grad16S.ptr<short>(i);
        unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            short g = *(ptrGrad++);
            *(ptrMask++) = (g >= minusThresh && g <= thresh) ? 255 : 0;            
        }
    }
}

void getTransformsGrayPairWiseSiftPanoPaper(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    std::vector<double>& kt)
{
    int numImages = images.size();

    double invSigmaNSqr = 0.01;
    double invSigmaGSqr = 100;

    cv::Mat_<double> A(numImages, numImages); A.setTo(0);
    cv::Mat_<double> b(numImages, 1); b.setTo(0);
    cv::Mat_<double> gains(numImages, 1);
    cv::Mat intersect;
    int rows = images[0].rows, cols = images[0].cols;
    for (int i = 0; i < numImages; ++i)
    {
        for (int j = 0; j < numImages; ++j)
        {
            if (i == j)
                continue;

            intersect = masks[i] & masks[j];
            if (cv::countNonZero(intersect) == 0)
                continue;

            for (int u = 0; u < rows; u++)
            {
                const unsigned char* ptri = images[i].ptr<unsigned char>(u);
                const unsigned char* ptrj = images[j].ptr<unsigned char>(u);
                const unsigned char* ptrm = intersect.ptr<unsigned char>(u);
                for (int v = 0; v < cols; v++)
                {
                    if (ptrm[v])
                    {
                        double vali = ptri[v], valj = ptrj[v];
                        A(i, i) += vali * vali * invSigmaNSqr + invSigmaGSqr;
                        A(j, j) += valj * valj * invSigmaNSqr;
                        A(i, j) -= 2 * vali * valj * invSigmaNSqr;
                        b(i) += invSigmaGSqr;
                    }
                }
            }
        }
    }

    //std::cout << A << "\n" << b << "\n";
    bool success = cv::solve(A, b, gains);
    //std::cout << gains.t() << "\n";
    if (!success)
        gains.setTo(1);

    kt.resize(numImages);
    for (int i = 0; i < numImages; i++)
        kt[i] = gains(i);
}

void getTransformsGrayPairWiseMutualError(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    std::vector<double>& kt)
{
    int numImages = images.size();

    double invSigmaNSqr = 0.01;
    double invSigmaDSqr = 1;
    double invSigmaGSqr = 100;

    cv::Mat_<double> A(numImages, numImages); A.setTo(0);
    cv::Mat_<double> b(numImages, 1); b.setTo(0);
    cv::Mat_<double> gains(numImages, 1);
    cv::Mat intersect;
    int rows = images[0].rows, cols = images[0].cols;
    for (int i = 0; i < numImages; ++i)
    {
        for (int j = 0; j < numImages; ++j)
        {
            if (i == j)
                continue;

            intersect = masks[i] & masks[j];
            if (cv::countNonZero(intersect) == 0)
                continue;

            for (int u = 0; u < rows; u++)
            {
                const unsigned char* ptri = images[i].ptr<unsigned char>(u);
                const unsigned char* ptrj = images[j].ptr<unsigned char>(u);
                const unsigned char* ptrm = intersect.ptr<unsigned char>(u);
                for (int v = 0; v < cols; v++)
                {
                    if (ptrm[v])
                    {
                        double vali = ptri[v], valj = ptrj[v];
                        if (vali > 10 && vali < 245 && valj > 10 && valj < 245 &&
                            (abs(vali - valj) < 64 || (vali * 3 > valj && valj * 3 > vali)))
                        {
                            A(i, i) += vali * vali * (invSigmaNSqr + invSigmaDSqr) + invSigmaGSqr;
                            A(j, j) += valj * valj * (invSigmaNSqr + invSigmaDSqr);
                            A(i, j) -= 2 * vali * valj * invSigmaNSqr;
                            b(i) += invSigmaGSqr + vali * valj * invSigmaDSqr;
                            b(j) += vali * valj * invSigmaDSqr;
                        }
                    }
                }
            }
        }
    }

    //std::cout << A << "\n" << b << "\n";
    bool success = cv::solve(A, b, gains);
    //std::cout << gains.t() << "\n";
    if (!success)
        gains.setTo(1);

    kt.resize(numImages);
    for (int i = 0; i < numImages; i++)
        kt[i] = gains(i);
}

void getTransformsBGRPairWiseSiftPanoPaper(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    std::vector<std::vector<double> >& kts)
{
    int numImages = images.size();

    double invSigmaNSqr = 0.01;
    double invSigmaGSqr = 100;

    cv::Mat_<double> A[3]; 
    cv::Mat_<double> B[3];
    cv::Mat_<double> gains[3];
    for (int i = 0; i < 3; i++)
    {
        A[i].create(numImages, numImages); A[i].setTo(0);
        B[i].create(numImages, 1); B[i].setTo(0);
    }
    cv::Mat intersect;
    int rows = images[0].rows, cols = images[0].cols;
    for (int i = 0; i < numImages; ++i)
    {
        for (int j = 0; j < numImages; ++j)
        {
            if (i == j)
                continue;

            intersect = masks[i] & masks[j];
            if (cv::countNonZero(intersect) == 0)
                continue;

            for (int u = 0; u < rows; u++)
            {
                const unsigned char* ptri = images[i].ptr<unsigned char>(u);
                const unsigned char* ptrj = images[j].ptr<unsigned char>(u);
                const unsigned char* ptrm = intersect.ptr<unsigned char>(u);
                for (int v = 0; v < cols; v++)
                {
                    if (ptrm[v])
                    {
                        const unsigned char* pi = ptri + v * 3;
                        const unsigned char* pj = ptrj + v * 3;
                        for (int w = 0; w < 3; w++)
                        {
                            double vali = pi[w], valj = pj[w];
                            A[w](i, i) += vali * vali * invSigmaNSqr + invSigmaGSqr;
                            A[w](j, j) += valj * valj * invSigmaNSqr;
                            A[w](i, j) -= 2 * vali * valj * invSigmaNSqr;
                            B[w](i) += invSigmaGSqr;
                        }
                    }
                }
            }
        }
    }
    
    for (int i = 0; i < 3; i++)
    {
        //std::cout << A[i] << "\n" << B[i] << "\n";
        bool success = cv::solve(A[i], B[i], gains[i]);
        //std::cout << gains[i] << "\n";
        if (!success)
            gains[i].setTo(1);
    }

    kts.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        kts[i].resize(3);
        for (int j = 0; j < 3; j++)
            kts[i][j] = gains[j](i);
    }        
}

void getTransformsBGRPairWiseMutualError(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    std::vector<std::vector<double> >& kts)
{
    int numImages = images.size();

    double invSigmaNSqr = 0.01;
    double invSigmaDSqr = 1;
    double invSigmaGSqr = 100;

    cv::Mat_<double> A[3];
    cv::Mat_<double> B[3];
    cv::Mat_<double> gains[3];
    for (int i = 0; i < 3; i++)
    {
        A[i].create(numImages, numImages); A[i].setTo(0);
        B[i].create(numImages, 1); B[i].setTo(0);
    }
    cv::Mat intersect;
    int rows = images[0].rows, cols = images[0].cols;
    for (int i = 0; i < numImages; ++i)
    {
        for (int j = 0; j < numImages; ++j)
        {
            if (i == j)
                continue;

            intersect = masks[i] & masks[j];
            if (cv::countNonZero(intersect) == 0)
                continue;

            for (int u = 0; u < rows; u++)
            {
                const unsigned char* ptri = images[i].ptr<unsigned char>(u);
                const unsigned char* ptrj = images[j].ptr<unsigned char>(u);
                const unsigned char* ptrm = intersect.ptr<unsigned char>(u);
                for (int v = 0; v < cols; v++)
                {
                    if (ptrm[v])
                    {
                        const unsigned char* pi = ptri + v * 3;
                        const unsigned char* pj = ptrj + v * 3;
                        for (int w = 0; w < 3; w++)
                        {
                            double vali = pi[w], valj = pj[w];
                            //A[w](i, i) += vali * vali * invSigmaNSqr + invSigmaGSqr;
                            //A[w](j, j) += valj * valj * invSigmaNSqr;
                            //A[w](i, j) -= 2 * vali * valj * invSigmaNSqr;
                            //B[w](i) += invSigmaGSqr;
                            if (vali > 10 && vali < 245 && valj > 10 && valj < 245 &&
                                (abs(vali - valj) < 64 || (vali * 3 > valj && valj * 3 > vali)))
                            {
                                A[w](i, i) += vali * vali * (invSigmaNSqr + invSigmaDSqr) + invSigmaGSqr;
                                A[w](j, j) += valj * valj * (invSigmaNSqr + invSigmaDSqr);
                                A[w](i, j) -= 2 * vali * valj * invSigmaNSqr;
                                B[w](i) += invSigmaGSqr + vali * valj * invSigmaDSqr;
                                B[w](j) += vali * valj * invSigmaDSqr;
                            }
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < 3; i++)
    {
        //std::cout << A[i] << "\n" << B[i] << "\n";
        bool success = cv::solve(A[i], B[i], gains[i]);
        //std::cout << gains[i].t() << "\n";
        if (!success)
            gains[i].setTo(1);
    }

    kts.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        kts[i].resize(3);
        for (int j = 0; j < 3; j++)
            kts[i][j] = gains[j](i);
    }
}

void getLUTBezierSmooth(std::vector<unsigned char>& lut, double k)
{
    CV_Assert(k > 0);
    lut.resize(256);
    if (abs(k - 1) < 0.02/*k > 0.99*/)
    {
        for (int i = 0; i < 256; i++)
            lut[i] = cv::saturate_cast<unsigned char>(i * k);
    }
    else
    {
        cv::Point2d p0(0, 0), p1, p2(255, 255);
        if (k > 1)
            p1 = cv::Point2d(255 / k, 255);
        else
        {
            double x = 255.0 * (1 - 1 / k) / (k - 1 / k);
            p1.x = x;
            p1.y = x * k;
        }
        lut[0] = 0;
        lut[255] = 255;
        for (int i = 1; i < 255; i++)
        {
            double a = p0.x + p2.x - 2 * p1.x, b = 2 * (p1.x - p0.x), c = p0.x - i;
            double m = -b / (2 * a), n = sqrt(b * b - 4 * a * c) / (2 * a);
            double t0 = m - n, t1 = m + n, t;
            if (t0 < 1 && t0 > 0)
                t = t0;
            else if (t1 < 1 && t1 > 0)
                t = t1;
            else
                CV_Assert(0);
            double y = (1 - t) * (1 - t) * p0.y + 2 * (1 - t) * t * p1.y + t * t * p2.y + 0.5;
            y = y < 0 ? 0 : (y > 255 ? 255 : y);
            lut[i] = y;
        }
    }
}

static void adjust(cv::Mat& image, const unsigned char lut[256])
{
    CV_Assert(image.data && image.depth() == CV_8U);
    int rows = image.rows, cols = image.cols * image.channels();
    for (int i = 0; i < rows; i++)
    {
        unsigned char* ptr = image.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            *ptr = lut[*ptr];
            ptr++;
        }
    }
}

void adjust(const cv::Mat& src, cv::Mat& dst, const std::vector<unsigned char>& lut)
{
    CV_Assert(src.data && src.depth() == CV_8U && lut.size() == 256);
    dst.create(src.size(), src.type());
    int rows = src.rows, cols = src.cols * src.channels();
    const unsigned char* ptrLut = lut.data();
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrSrc = src.ptr<unsigned char>(i);
        unsigned char* ptrDst = dst.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            *ptrDst = ptrLut[*ptrSrc];
            ptrDst++;
            ptrSrc++;
        }
    }
}

void adjust(const cv::Mat& src, cv::Mat& dst, const std::vector<std::vector<unsigned char> >& luts)
{
    CV_Assert(src.data && src.depth() == CV_8U &&
        luts.size() == 3 && luts[0].size() == 256 && luts[1].size() == 256 && luts[2].size() == 256);
    dst.create(src.size(), src.type());
    int rows = src.rows, cols = src.cols;
    const unsigned char* blut = luts[0].data();
    const unsigned char* glut = luts[1].data();
    const unsigned char* rlut = luts[2].data();
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrSrc = src.ptr<unsigned char>(i);
        unsigned char* ptrDst = dst.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            *(ptrDst++) = blut[*(ptrSrc++)];
            *(ptrDst++) = glut[*(ptrSrc++)];
            *(ptrDst++) = rlut[*(ptrSrc++)];
        }
    }
}

void isDiffSmall(const cv::Mat& a, const cv::Mat& b, const cv::Mat& mask, cv::Mat& out)
{
    int rows = a.rows, cols = a.cols;
    out.create(rows, cols, CV_8UC1);
    out.setTo(0);
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrA = a.ptr<unsigned char>(i);
        const unsigned char* ptrB = b.ptr<unsigned char>(i);
        const unsigned char* ptrM = mask.ptr<unsigned char>(i);
        unsigned char* ptrO = out.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrM[j])
            {
                if (abs(int(ptrA[j]) - int(ptrB[j])) < 64)
                    ptrO[j] = 255;
            }
        }
    }
}

void compensate(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results)
{
    int numImages = images.size();

    std::vector<cv::Mat> grayImages(numImages);
    for (int i = 0; i < numImages; i++)
        cv::cvtColor(images[i], grayImages[i], CV_BGR2GRAY);

    //cv::Mat gradSmallMask, blur, grad;
    //for (int i = 0; i < numImages; i++)
    //{
    //    isGradSmall(grayImages[i], 4, gradSmallMask, blur, grad);
    //    outMasks[i] &= gradSmallMask;
    //}

    std::vector<double> gains;
    getTransformsGrayPairWiseMutualError(grayImages, masks, gains);

    results.resize(numImages);
    std::vector<unsigned char> lut;
    for (int i = 0; i < numImages; i++)
    {
        getLUTBezierSmooth(lut, gains[i]);
        adjust(images[i], results[i], lut);
    }
}

void compensateBGR(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results)
{
    int numImages = images.size();

    std::vector<std::vector<double> > kts;
    getTransformsBGRPairWiseMutualError(images, masks, kts);

    std::vector<std::vector<std::vector<unsigned char> > > luts(numImages);
    for (int i = 0; i < numImages; i++)
    {
        luts[i].resize(3);
        for (int j = 0; j < 3; j++)
            getLUTBezierSmooth(luts[i][j], kts[i][j]);
    }

    results.resize(numImages);
    for (int i = 0; i < numImages; i++)
        adjust(images[i], results[i], luts[i]);
}

class GainCompensate
{
public:
    GainCompensate() :numImages(0), maxMeanIndex(0), rows(0), cols(0), success(false) {}
    bool prepare(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks);
    bool compensate(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& results) const;
private:
    std::vector<double> gains;
    std::vector<std::vector<unsigned char> > LUTs;
    int numImages;
    int maxMeanIndex;
    int rows, cols;
    int success;
};

static void getTransformsMeanApproxSiftPanoPaper(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    int& maxIndex, std::vector<double>& kt)
{
    int numImages = images.size();

    cv::Mat_<double> N(numImages, numImages), I(numImages, numImages);
    for (int i = 0; i < numImages; i++)
    {
        for (int j = 0; j < numImages; j++)
        {
            if (i == j)
            {
                N(i, i) = cv::countNonZero(masks[i]);
                I(i, i) = cv::mean(images[i], masks[i])[0];
            }
            else
            {
                cv::Mat intersect = masks[i] & masks[j];
                N(i, j) = cv::countNonZero(intersect);
                I(i, j) = cv::mean(images[i], intersect)[0];
            }
        }
    }
    //std::cout << N << "\n" << I << "\n";

    double invSigmaNSqr = 0.01;
    double invSigmaGSqr = 100;

    cv::Mat_<double> A(numImages, numImages); A.setTo(0);
    cv::Mat_<double> b(numImages, 1); b.setTo(0);
    cv::Mat_<double> gains(numImages, 1);
    for (int i = 0; i < numImages; ++i)
    {
        for (int j = 0; j < numImages; ++j)
        {
            A(i, i) += N[i][j] * (I[i][j] * I[i][j] * invSigmaNSqr + invSigmaGSqr);
            A(j, j) += N[i][j] * (I[j][i] * I[j][i] * invSigmaNSqr);
            A(i, j) -= 2 * N[i][j] * (I[i][j] * I[j][i] * invSigmaNSqr);
            b(i) += N[i][j] * invSigmaGSqr;
        }
    }

    //std::cout << A << "\n" << b << "\n";
    bool success = cv::solve(A, b, gains);
    std::cout << gains << "\n";
    if (!success)
        gains.setTo(1);

    double maxMean = -1;
    int maxMeanIndex = -1;
    for (int i = 0; i < numImages; i++)
    {
        if (I[i][i] > maxMean)
        {
            maxMean = I[i][i];
            maxMeanIndex = i;
        }
    }
    maxIndex = maxMeanIndex;

    kt.resize(numImages);
    for (int i = 0; i < numImages; i++)
        kt[i] = gains(i);
}

static void rescale(std::vector<double>& kt, int index)
{
    int numImages = kt.size();
    double kscale = 1.0 / kt[index];

    for (int i = 0; i < numImages; i++)
    {
        kt[i] = kscale * kt[i];
        //printf("k = %f\n", kt[i]);
    }
}

static void getLUT(double k, unsigned char lut[256])
{
    CV_Assert(k > 0);
    if (k > 1)
    {
        cv::Point2d p0(0, 0), p1(255 / k, 255), p2(255, 255);
        lut[0] = 0;
        lut[255] = 255;
        for (int i = 1; i < 255; i++)
        {
            double a = p0.x + p2.x - 2 * p1.x, b = 2 * (p1.x - p0.x), c = p0.x - i;
            double m = -b / (2 * a), n = sqrt(b * b - 4 * a * c) / (2 * a);
            double t0 = m - n, t1 = m + n, t;
            if (t0 < 1 && t0 > 0)
                t = t0;
            else if (t1 < 1 && t1 > 0)
                t = t1;
            else
                CV_Assert(0);
            double y = (1 - t) * (1 - t) * p0.y + 2 * (1 - t) * t * p1.y + t * t * p2.y + 0.5;
            y = y < 0 ? 0 : (y > 255 ? 255 : y);
            lut[i] = y;
        }
    }
    else if (k < 1)
    {
        for (int i = 0; i < 256; i++)
            lut[i] = k * i + 0.5;
    }
    else
    {
        for (int i = 0; i < 256; i++)
            lut[i] = i;
    }
}

bool GainCompensate::prepare(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks)
{
    success = false;
    if (!checkSize(images, masks))
        return false;

    int currNumImages = images.size();
    for (int i = 0; i < currNumImages; i++)
    {
        if (images[i].type() != CV_8UC3 || masks[i].type() != CV_8UC1)
            return false;
    }

    numImages = currNumImages;
    rows = images[0].rows;
    cols = images[0].cols;

    std::vector<cv::Mat> grayImages(numImages);
    for (int i = 0; i < numImages; i++)
        cv::cvtColor(images[i], grayImages[i], CV_BGR2GRAY);

    getTransformsMeanApproxSiftPanoPaper(grayImages, masks, maxMeanIndex, gains);
    rescale(gains, maxMeanIndex);

    LUTs.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        LUTs[i].resize(256);
        getLUT(gains[i], &LUTs[i][0]);
    }

    success = true;
    return true;
}

bool GainCompensate::compensate(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& results) const
{
    if (!success)
        return false;

    if (images.size() != numImages)
        return false;

    for (int i = 0; i < numImages; i++)
    {
        if (images[i].type() != CV_8UC3 ||
            images[i].rows != rows || images[i].cols != cols)
            return false;
    }

    results.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        if (i == maxMeanIndex)
            images[i].copyTo(results[i]);
        else
        {
            //images[i].copyTo(results[i]);
            //adjust(results[i], &LUTs[i][0]);
            adjust(images[i], results[i], LUTs[i]);
        }
    }

    return true;
}

