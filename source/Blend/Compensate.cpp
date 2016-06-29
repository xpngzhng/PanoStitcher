#include "ZBlend.h"
#include "ZBlendAlgo.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

// Header files content

//void compensate(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results);
void compensate3(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results);

//class GainCompensate
//{
//public:
//    GainCompensate() :numImages(0), maxMeanIndex(0), rows(0), cols(0), success(false) {}
//    bool prepare(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks);
//    bool compensate(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& results) const;
//private:
//    std::vector<double> gains;
//    std::vector<std::vector<unsigned char> > LUTs;
//    int numImages;
//    int maxMeanIndex;
//    int rows, cols;
//    int success;
//};

void compensateGray(const std::vector<cv::Mat>& images,
    const std::vector<cv::Mat>& masks, int refIndex, std::vector<cv::Mat>& results);
void compensateLightAndSaturation(const std::vector<cv::Mat>& images,
    const std::vector<cv::Mat>& masks, int refIndex, std::vector<cv::Mat>& results);

// End header files content

static void getDependentMatrices(const std::vector<cv::Mat>& images, 
    const std::vector<cv::Mat>& masks, int channel, cv::Mat_<double>& N, cv::Mat_<double>& I)
{
    CV_Assert(channel >= 0 && channel < 3);
    int numImages = images.size();

    N.create(numImages, numImages);
    I.create(numImages, numImages);
    for (int i = 0; i < numImages; i++)
    {
        for (int j = 0; j < numImages; j++)
        {
            if (i == j)
            {
                N(i, i) = cv::countNonZero(masks[i]);
                I(i, i) = cv::mean(images[i], masks[i])[channel];
            }
            else
            {
                cv::Mat intersect = masks[i] & masks[j];
                N(i, j) = cv::countNonZero(intersect);
                I(i, j) = cv::mean(images[i], intersect)[channel];
            }
        }
    }
}

static void getDependentMatrices(const std::vector<cv::Mat>& images, 
    const std::vector<cv::Mat>& masks, cv::Mat_<double>& N, cv::Mat_<double> I[3])
{
    int numImages = images.size();

    N.create(numImages, numImages);
    I[0].create(numImages, numImages);
    I[1].create(numImages, numImages);
    I[2].create(numImages, numImages);
    for (int i = 0; i < numImages; i++)
    {
        for (int j = 0; j < numImages; j++)
        {
            if (i == j)
            {
                N(i, i) = cv::countNonZero(masks[i]);
                cv::Scalar meanVal = cv::mean(images[i], masks[i]);
                I[0](i, i) = meanVal[0];
                I[1](i, i) = meanVal[1];
                I[2](i, i) = meanVal[2];
            }
            else
            {
                cv::Mat intersect = masks[i] & masks[j];
                N(i, j) = cv::countNonZero(intersect);
                cv::Scalar meanVal = cv::mean(images[i], intersect);
                I[0](i, j) = meanVal[0];
                I[1](i, j) = meanVal[1];
                I[2](i, j) = meanVal[2];
            }
        }
    }
}

static void getLinearTransforms(const cv::Mat_<double>& N, const cv::Mat_<double>& I, 
    int& maxIndex, std::vector<double>& kt)
{
    int numImages = N.rows;

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

static void getLinearTransforms(const cv::Mat_<double>& N, const cv::Mat_<double>& I,
    int& maxIndex, std::vector<double>& kt, std::vector<double>& bt)
{
    int numImages = N.rows;

    double invSigmaNSqr = 0.01;
    double invSigmaGSqr = 100;

    cv::Mat_<double> A(2 * numImages, 2 * numImages); A.setTo(0);
    cv::Mat_<double> b(2 * numImages, 1); b.setTo(0);
    cv::Mat_<double> x(2 * numImages, 1);
    for (int i = 0; i < numImages; i++)
    {
        for (int j = 0; j < numImages; j++)
        {
            A(i, i) += N[i][j] * (I[i][j] * I[i][j] * invSigmaNSqr + invSigmaGSqr);
            A(j, j) += N[i][j] * I[j][i] * I[j][i] * invSigmaNSqr;
            A(i + numImages, i + numImages) += N[i][j] * invSigmaNSqr;
            A(j + numImages, j + numImages) += N[i][j] * invSigmaNSqr;
            A(i, j) -= 2 * N[i][j] * I[i][j] * I[j][i] * invSigmaNSqr;
            A(i + numImages, j + numImages) -= 2 * N[i][j] * invSigmaNSqr;

            A(i, i + numImages) += N[i][j] * I[i][j] * invSigmaNSqr;
            A(i + numImages, i) += N[i][j] * I[i][j] * invSigmaNSqr;
            A(i, j + numImages) -= N[i][j] * I[i][j] * invSigmaNSqr;
            A(j + numImages, i) -= N[i][j] * I[i][j] * invSigmaNSqr;
            A(j, i + numImages) -= N[i][j] * I[j][i] * invSigmaNSqr;
            A(i + numImages, j) -= N[i][j] * I[j][i] * invSigmaNSqr;
            A(j, j + numImages) += N[i][j] * I[j][i] * invSigmaNSqr;
            A(j + numImages, j) += N[i][j] * I[j][i] * invSigmaNSqr;

            b(i) += N[i][j] * invSigmaGSqr;
        }
    }

    //std::cout << A << "\n" << b << "\n";
    bool success = cv::solve(A, b, x);
    std::cout << x << "\n";
    if (!success)
    {
        x(cv::Range(0, numImages), cv::Range::all()).setTo(1);
        x(cv::Range(numImages, 2 * numImages), cv::Range::all()).setTo(0);
    }

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
    bt.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        kt[i] = x(i);
        bt[i] = x(i + numImages);
    }
}

static void getLinearTransformsEnhance(const cv::Mat_<double>& N, const cv::Mat_<double>& I,
    int& maxIndex, std::vector<double>& kt, std::vector<double>& bt)
{
    int numImages = N.rows;

    double globalMean = 0, globalNum = 0;
    for (int i = 0; i < numImages; i++)
    {
        globalMean += I[i][i] * N[i][i];
        globalNum += N[i][i];
    }
    globalMean /= globalNum;

    double invSigmaNSqr = 0.01;
    double invSigmaGSqr = 100;
    double invSigmaDSqr = 0.0001;

    cv::Mat_<double> A(2 * numImages, 2 * numImages); A.setTo(0);
    cv::Mat_<double> b(2 * numImages, 1); b.setTo(0);
    cv::Mat_<double> x(2 * numImages, 1);
    for (int i = 0; i < numImages; i++)
    {
        for (int j = 0; j < numImages; j++)
        {
            A(i, i) += N[i][j] * (I[i][j] * I[i][j] * invSigmaNSqr + globalMean * globalMean * invSigmaDSqr + invSigmaGSqr);
            A(j, j) += N[i][j] * I[j][i] * I[j][i] * invSigmaNSqr;
            A(i + numImages, i + numImages) += N[i][j] * (invSigmaNSqr + invSigmaDSqr);
            A(j + numImages, j + numImages) += N[i][j] * invSigmaNSqr;
            A(i, j) -= 2 * N[i][j] * I[i][j] * I[j][i] * invSigmaNSqr;
            A(i + numImages, j + numImages) -= 2 * N[i][j] * invSigmaNSqr;

            A(i, i + numImages) += N[i][j] * (I[i][j] * invSigmaNSqr + globalMean * invSigmaDSqr);
            A(i + numImages, i) += N[i][j] * (I[i][j] * invSigmaNSqr + globalMean * invSigmaDSqr);
            A(i, j + numImages) -= N[i][j] * I[i][j] * invSigmaNSqr;
            A(j + numImages, i) -= N[i][j] * I[i][j] * invSigmaNSqr;
            A(j, i + numImages) -= N[i][j] * I[j][i] * invSigmaNSqr;
            A(i + numImages, j) -= N[i][j] * I[j][i] * invSigmaNSqr;
            A(j, j + numImages) += N[i][j] * I[j][i] * invSigmaNSqr;
            A(j + numImages, j) += N[i][j] * I[j][i] * invSigmaNSqr;

            b(i) += N[i][j] * (invSigmaGSqr + globalMean * globalMean * invSigmaDSqr);
            b(i + numImages) += N[i][j] * globalMean * invSigmaDSqr;
        }
    }

    //std::cout << A << "\n" << b << "\n";
    bool success = cv::solve(A, b, x);
    std::cout << x << "\n";
    if (!success)
    {
        x(cv::Range(0, numImages), cv::Range::all()).setTo(1);
        x(cv::Range(numImages, 2 * numImages), cv::Range::all()).setTo(0);
    }

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
    bt.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        kt[i] = x(i);
        bt[i] = x(i + numImages);
    }
}

static void getLinearTransforms(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
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

static void getAccurateLinearTransforms(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
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

            //A(i, i) += N[i][j] * (I[i][j] * I[i][j] * invSigmaNSqr + invSigmaGSqr);
            //A(j, j) += N[i][j] * (I[j][i] * I[j][i] * invSigmaNSqr);
            //A(i, j) -= 2 * N[i][j] * (I[i][j] * I[j][i] * invSigmaNSqr);
            //b(i) += N[i][j] * invSigmaGSqr;
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

    std::cout << A << "\n" << b << "\n";
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

static void adjust(const cv::Mat& src, cv::Mat& dst, const unsigned char lut[256])
{
    CV_Assert(src.data && src.depth() == CV_8U);
    dst.create(src.size(), src.type());
    int rows = src.rows, cols = src.cols * src.channels();
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrSrc = src.ptr<unsigned char>(i);
        unsigned char* ptrDst = dst.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            *ptrDst = lut[*ptrSrc];
            ptrDst++;
            ptrSrc++;
        }
    }
}

static void adjust(const cv::Mat& src, cv::Mat& dst, 
    const unsigned char blut[256], const unsigned char glut[256], const unsigned char rlut[256])
{
    CV_Assert(src.data && src.depth() == CV_8U);
    dst.create(src.size(), src.type());
    int rows = src.rows, cols = src.cols;
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

void compensate(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results)
{
    int numImages = images.size();

    std::vector<cv::Mat> grayImages(numImages);
    for (int i = 0; i < numImages; i++)
        cv::cvtColor(images[i], grayImages[i], CV_BGR2GRAY);

    std::vector<cv::Mat> extendMasks;
    getExtendedMasks(masks, 100, extendMasks);

    std::vector<cv::Mat> outMasks(numImages);
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

    int maxMeanIndex;
    std::vector<double> gains;
    getAccurateLinearTransforms(grayImages, masks, maxMeanIndex, gains);
    //rescale(gains, maxMeanIndex);
    //for (int i = 0; i < numImages; i++)
    //    printf("%f ", gains[i]);
    //printf("\n");

    results.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        //if (i == maxMeanIndex)
        //{
        //    images[i].copyTo(results[i]);
        //    continue;
        //}

        //double gain = gains[i];
        //if (gain <= 1)
        //    results[i] = images[i] * gain;
        //else
        //{
        //    unsigned char lut[256];
        //    getLUT(gain, lut);
        //    results[i] = images[i].clone();
        //    adjust(results[i], lut);
        //}
        results[i] = images[i] * gains[i];
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

    getLinearTransforms(grayImages, masks, maxMeanIndex, gains);
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
            adjust(images[i], results[i], &LUTs[i][0]);
        }
    }

    return true;
}

static void getLinearTransforms(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    int& maxIndex, std::vector<double>& kt, std::vector<double>& bt)
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

    cv::Mat_<double> A(2 * numImages, 2 * numImages); A.setTo(0);
    cv::Mat_<double> b(2 * numImages, 1); b.setTo(0);
    cv::Mat_<double> x(2 * numImages, 1);
    for (int i = 0; i < numImages; i++)
    {
        for (int j = 0; j < numImages; j++)
        {
            A(i, i) += N[i][j] * (I[i][j] * I[i][j] * invSigmaNSqr + invSigmaGSqr);
            A(j, j) += N[i][j] * I[j][i] * I[j][i] * invSigmaNSqr;
            A(i + numImages, i + numImages) += N[i][j] * invSigmaNSqr;
            A(j + numImages, j + numImages) += N[i][j] * invSigmaNSqr;
            A(i, j) -= 2 * N[i][j] * I[i][j] * I[j][i] * invSigmaNSqr;
            A(i + numImages, j + numImages) -= 2 * N[i][j] * invSigmaNSqr;

            A(i, i + numImages) += N[i][j] * I[i][j] * invSigmaNSqr;
            A(i + numImages, i) += N[i][j] * I[i][j] * invSigmaNSqr;
            A(i, j + numImages) -= N[i][j] * I[i][j] * invSigmaNSqr;
            A(j + numImages, i) -= N[i][j] * I[i][j] * invSigmaNSqr;
            A(j, i + numImages) -= N[i][j] * I[j][i] * invSigmaNSqr;
            A(i + numImages, j) -= N[i][j] * I[j][i] * invSigmaNSqr;
            A(j, j + numImages) += N[i][j] * I[j][i] * invSigmaNSqr;
            A(j + numImages, j) += N[i][j] * I[j][i] * invSigmaNSqr;

            b(i) += N[i][j] * invSigmaGSqr;
        }
    }

    //std::cout << A << "\n" << b << "\n";
    bool success = cv::solve(A, b, x);
    std::cout << x << "\n";
    if (!success)
    {
        x(cv::Range(0, numImages), cv::Range::all()).setTo(1);
        x(cv::Range(numImages, 2 * numImages), cv::Range::all()).setTo(0);
    }

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
    bt.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        kt[i] = x(i);
        bt[i] = x(i + numImages);
    }
}

static void rescale(std::vector<double>& kt, std::vector<double>& bt, int index)
{
    int numImages = kt.size();
    double kscale = 1.0 / kt[index], bshift = -kscale * bt[index];

    for (int i = 0; i < numImages; i++)
    {
        kt[i] = kscale * kt[i];
        bt[i] = kscale * bt[i] + bshift;
        //printf("k = %f, b = %f\n", kt[i], bt[i]);
    }
}

static void adjust(cv::Mat& image, double k, double b, const cv::Mat& mask)
{
    CV_Assert(image.data && image.type() == CV_8UC3 &&
        mask.data && mask.type() == CV_8UC1 &&
        image.size() == mask.size());

    int rows = image.rows, cols = image.cols;
    for (int i = 0; i < rows; i++)
    {
        unsigned char* ptr = image.ptr<unsigned char>(i);
        const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrMask[j])
            {
                ptr[j * 3] = cv::saturate_cast<unsigned char>(ptr[j * 3] * k + b);
                ptr[j * 3 + 1] = cv::saturate_cast<unsigned char>(ptr[j * 3 + 1] * k + b);
                ptr[j * 3 + 2] = cv::saturate_cast<unsigned char>(ptr[j * 3 + 2] * k + b);
            }
        }
    }
}

static void adjustChannel(cv::Mat& image, int channelIndex, double k, double b, const cv::Mat& mask)
{
    CV_Assert(image.data && image.type() == CV_8UC3 &&
        mask.data && mask.type() == CV_8UC1 &&
        image.size() == mask.size());
    int rows = image.rows, cols = image.cols;
    for (int i = 0; i < rows; i++)
    {
        unsigned char* ptr = image.ptr<unsigned char>(i);
        const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrMask[j])
                ptr[j * 3 + channelIndex] = cv::saturate_cast<unsigned char>(ptr[j * 3 + channelIndex] * k + b);
        }
    }
}

void compensate3(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& results)
{
    int numImages = images.size();

    std::vector<cv::Mat> grayImages(numImages);
    for (int i = 0; i < numImages; i++)
        cv::cvtColor(images[i], grayImages[i], CV_BGR2GRAY);

    int maxMeanIndex;
    std::vector<double> kl, bl;
    getLinearTransforms(grayImages, masks, maxMeanIndex, kl, bl);
    printf("max mean index = %d\n", maxMeanIndex);
    rescale(kl, bl, maxMeanIndex);    

    results.resize(numImages);
    cv::Mat tmp;
    for (int i = 0; i < numImages; i++)
    {
        images[i].copyTo(results[i]);
        adjust(results[i], kl[i], bl[i], masks[i]);
    }

    grayImages.clear();

    int fromTo[] = { 2, 0 };
    cv::Mat hls;
    std::vector<cv::Mat> satuImages(numImages);
    for (int i = 0; i < numImages; i++)
    {
        cv::cvtColor(images[i], hls, CV_BGR2HLS);
        satuImages[i].create(images[i].size(), CV_8UC1);
        cv::mixChannels(&images[i], 1, &satuImages[i], 1, fromTo, 1);
    }

    int maxSatuIndex;
    std::vector<double> ks, bs;
    //getLinearTransforms(satuImages, masks, maxSatuIndex, ks, bs);
    getLinearTransforms(satuImages, masks, maxSatuIndex, ks);
    printf("max satu index = %d\n", maxSatuIndex);
    //rescale(ks, bs, maxSatuIndex);
    rescale(ks, maxSatuIndex);

    for (int i = 0; i < numImages; i++)
    {
        cv::cvtColor(results[i], hls, CV_BGR2HLS);
        adjustChannel(hls, 2, ks[i], 0/*bs[i]*/, masks[i]);
        cv::cvtColor(hls, results[i], CV_HLS2BGR);
    }

    //std::vector<cv::Mat> origHLS(numImages), procHLS(numImages);
    //std::vector<double> origS(numImages), procS(numImages);
    //for (int i = 0; i < numImages; i++)
    //{
    //    cv::cvtColor(images[i], origHLS[i], CV_BGR2HLS);
    //    cv::cvtColor(results[i], procHLS[i], CV_BGR2HLS);
    //    origS[i] = cv::mean(origHLS[i], masks[i])[2];
    //    procS[i] = cv::mean(procHLS[i], masks[i])[2];
    //}
    //for (int i = 0; i < numImages; i++)
    //    printf("%f - > %f\n", origS[i], procS[i]);
    //
    //cv::Mat its = masks[2] & masks[4];
    //double mean2 = cv::mean(procHLS[2], its)[2];
    //double mean4 = cv::mean(procHLS[4], its)[2];
    //printf("%f, %f\n", mean2, mean4);
}

void compensateGray(const std::vector<cv::Mat>& images,
    const std::vector<cv::Mat>& masks, int refIndex, std::vector<cv::Mat>& results)
{
    int numImages = images.size();

    std::vector<cv::Mat> grayImages(numImages);
    for (int i = 0; i < numImages; i++)
        cv::cvtColor(images[i], grayImages[i], CV_BGR2GRAY);

    int maxMeanIndex;
    std::vector<double> kl, bl;
    getLinearTransforms(grayImages, masks, maxMeanIndex, kl, bl);
    //cv::Mat_<double> N, I;
    //getDependentMatrices(grayImages, masks, 0, N, I);
    //getLinearTransformsEnhance(N, I, maxMeanIndex, kl, bl);
    printf("max mean index = %d\n", maxMeanIndex);
    rescale(kl, bl, (refIndex >= 0 && refIndex < numImages) ? refIndex : maxMeanIndex);

    results.resize(numImages);
    cv::Mat tmp;
    for (int i = 0; i < numImages; i++)
    {
        images[i].copyTo(results[i]);
        adjust(results[i], kl[i], bl[i], masks[i]);
    }
}

void compensateLightAndSaturation(const std::vector<cv::Mat>& images, 
    const std::vector<cv::Mat>& masks, int refIndex, std::vector<cv::Mat>& results)
{
    int numImages = images.size();

    std::vector<cv::Mat> hlsImages(numImages);
    for (int i = 0; i < numImages; i++)
        cv::cvtColor(images[i], hlsImages[i], CV_BGR2HLS);

    cv::Mat_<double> N, I[3];
    getDependentMatrices(images, masks, N, I);

    std::vector<double> kLight, bLight, kSaturate;
    int maxIndexLight, maxIndexSaturate;
    getLinearTransforms(N, I[1], maxIndexLight, kLight, bLight);
    printf("max light index = %d\n", maxIndexLight);
    rescale(kLight, bLight, (refIndex >= 0 && refIndex < numImages) ? refIndex : maxIndexLight);
    getLinearTransforms(N, I[2], maxIndexSaturate, kSaturate);
    printf("max sat index = %d\n", maxIndexSaturate);
    rescale(kSaturate, (refIndex >= 0 && refIndex < numImages) ? refIndex : maxIndexSaturate);

    results.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        // NOTICE: adjust lightness is not a good idea.
        // Perhaps it is because BGR to HLS is not a linear transform.
        // We had better not use the following two lines of commented code.
        //adjustChannel(hlsImages[i], 1, kLight[i], bLight[i], masks[i]);
        //cv::cvtColor(hlsImages[i], results[i], CV_HLS2BGR);
        images[i].copyTo(results[i]);
        adjust(results[i], kLight[i], bLight[i], masks[i]);
        cv::cvtColor(results[i], hlsImages[i], CV_BGR2HLS);
        adjustChannel(hlsImages[i], 2, kSaturate[i], 0, masks[i]);
        cv::cvtColor(hlsImages[i], results[i], CV_HLS2BGR);
    }        
}