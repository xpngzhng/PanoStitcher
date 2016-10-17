#include "VisualManip.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

void getLUT(unsigned char lut[256], double k)
{
    CV_Assert(k > 0);
    for (int i = 0; i < 256; i++)
        lut[i] = cv::saturate_cast<unsigned char>(k * i);
}

double calcMaxScale(const std::vector<double>& es, const std::vector<double>& rs, const std::vector<double>& bs)
{
    int size = es.size();
    CV_Assert(size > 0 && rs.size() == size && bs.size() == size);

    double scale = 0;
    for (int i = 0; i < size; i++)
    {
        scale = es[i] > scale ? es[i] : scale;
        double s;
        s = es[i] * rs[i];
        scale = s > scale ? s : scale;
        s = es[i] * bs[i];
        scale = s > scale ? s : scale;
    }
    return scale;
}

void getLUTMaxScale(unsigned char LUT[256], double k, double maxK)
{
    CV_Assert(k > 0);
    if (maxK <= 1.05)
    {
        for (int i = 0; i < 256; i++)
            LUT[i] = cv::saturate_cast<unsigned char>(k * i);
        return;
    }
    LUT[0] = 0;
    for (int i = 1; i < 256; i++)
    {
        double val = i / 255.0 * k;
        cv::Point2d p0(0, 0), p1(1, 1), p2(maxK, 1);
        double a = p0.x + p2.x - 2 * p1.x, b = 2 * (p1.x - p0.x), c = p0.x - val;
        double m = -b / (2 * a), n = sqrt(b * b - 4 * a * c) / (2 * a);
        double t0 = m - n, t1 = m + n, t;
        if (t0 <= 1 && t0 >= 0)
            t = t0;
        else if (t1 <= 1 && t1 >= 0)
            t = t1;
        else
            //CV_Assert(0);
        {
            if (i < 2)
                t = 0;
            if (i > 253)
                t = 1;
        }
        double y = (1 - t) * (1 - t) * p0.y + 2 * (1 - t) * t * p1.y + t * t * p2.y;
        LUT[i] = cv::saturate_cast<unsigned char>(y * 255);
    }
}

void correct(const std::vector<cv::Mat>& src, const std::vector<double>& es,
    const std::vector<double>& rs, const std::vector<double>& bs, std::vector<cv::Mat>& dst)
{
    int numImages = src.size();

    double maxE = 0;
    for (int i = 0; i < numImages; i++)
    {
        double e = es[i];
        maxE = e > maxE ? e : maxE;
    }
    double maxScale = calcMaxScale(es, rs, bs);

    dst.resize(numImages);
    char buf[64];
    unsigned char lutr[256], lutg[256], lutb[256];
    for (int i = 0; i < numImages; i++)
    {
        dst[i].create(src[i].size(), CV_8UC3);
        int rows = dst[i].rows, cols = dst[i].cols;

        double e = es[i];
        //e /= maxE;
        double r = rs[i];
        double b = bs[i];
        getLUT(lutr, e * r);
        getLUT(lutg, e);
        getLUT(lutb, e * b);
        //getLUTMaxScale(lutr, e * r, maxScale);
        //getLUTMaxScale(lutg, e, maxScale);
        //getLUTMaxScale(lutb, e * b, maxScale);
        for (int y = 0; y < rows; y++)
        {
            const unsigned char* ptrSrc = src[i].ptr<unsigned char>(y);
            unsigned char* ptrDst = dst[i].ptr<unsigned char>(y);
            for (int x = 0; x < cols; x++)
            {
                //ptrDst[0] = cv::saturate_cast<unsigned char>(ptrSrc[0] * e * b);
                //ptrDst[1] = cv::saturate_cast<unsigned char>(ptrSrc[1] * e);
                //ptrDst[2] = cv::saturate_cast<unsigned char>(ptrSrc[2] * e * r);

                ptrDst[0] = lutb[ptrSrc[0]];
                ptrDst[1] = lutg[ptrSrc[1]];
                ptrDst[2] = lutr[ptrSrc[2]];

                //ptrDst[0] = pow(ptrSrc[0] / 255.0 * e, 1.0 / 2.2) * 255;
                //ptrDst[1] = pow(ptrSrc[1] / 255.0 * e, 1.0 / 2.2) * 255;
                //ptrDst[2] = pow(ptrSrc[2] / 255.0 * e, 1.0 / 2.2) * 255;

                ptrSrc += 3;
                ptrDst += 3;
            }
        }

        sprintf(buf, "dst image %d", i);
        cv::Mat show;
        cv::resize(dst[i], show, cv::Size(), 0.5, 0.5);
        cv::imshow(buf, show);
    }
    cv::waitKey(0);
}

void run(const std::vector<cv::Mat>& images, const std::vector<PhotoParam>& params,
    const std::vector<int>& anchorIndexes, const std::vector<int>& optimizeOptions)
{
    std::vector<double> exposures, redRatios, blueRatios;
    exposureColorOptimize(images, params, anchorIndexes, optimizeOptions, exposures, redRatios, blueRatios);

    std::vector<cv::Mat> dstImages;
    correct(images, exposures, redRatios, blueRatios, dstImages);

    cv::Size dstSize(1200, 600);
    std::vector<cv::Mat> maps, masks, weights;
    getReprojectMapsAndMasks(params, images[0].size(), dstSize, maps, masks);

    std::vector<cv::Mat> reprojImages;
    reprojectParallel(dstImages, reprojImages, maps);

    cv::Mat blendImage;

    TilingLinearBlend blender;
    blender.prepare(masks, 50);
    blender.blend(reprojImages, blendImage);
    cv::imshow("blend", blendImage);
    cv::imwrite("out.bmp", blendImage);

    TilingMultibandBlendFast mbBlender;
    mbBlender.prepare(masks, 10, 8);
    mbBlender.blend(reprojImages, blendImage);
    cv::imshow("mb blend", blendImage);

    cv::waitKey(0);
}

void loadImages(const std::vector<std::string>& imagePaths, std::vector<cv::Mat>& images)
{
    int numImages = imagePaths.size();
    images.resize(numImages);
    for (int i = 0; i < numImages; i++)
        images[i] = cv::imread(imagePaths[i]);
}

int main()
{
    double PI = 3.1415926;

    std::vector<std::string> imagePaths;
    std::vector<PhotoParam> params;
    std::vector<cv::Mat> srcImages;

    std::vector<int> opts;
    opts.push_back(EXPOSURE | WHITE_BALANCE);
    //opts.push_back(WHITE_BALANCE);

    std::vector<int> anchors;

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-00.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-01.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-02.jpg");
    imagePaths.push_back("F:\\panoimage\\detuoffice\\input-03.jpg");
    loadPhotoParams("F:\\panoimage\\detuoffice\\detuoffice.xml", params);
    loadImages(imagePaths, srcImages);
    anchors.clear();
    //anchors.push_back(imagePaths.size() - 2);
    run(srcImages, params, anchors, opts);

    //xxxx
    //imagePaths.clear();
    //imagePaths.push_back("F:\\panoimage\\detuoffice2\\input-00.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice2\\input-01.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice2\\input-02.jpg");
    //imagePaths.push_back("F:\\panoimage\\detuoffice2\\input-03.jpg");
    //loadPhotoParamFromXML("F:\\panoimage\\detuoffice2\\detu.xml", params);
    //run(imagePaths, params, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\919-4\\snapshot0(2).bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\snapshot1(2).bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\snapshot2(2).bmp");
    imagePaths.push_back("F:\\panoimage\\919-4\\snapshot3(2).bmp");
    loadPhotoParamFromXML("F:\\panoimage\\919-4\\vrdl4.xml", params);
    loadImages(imagePaths, srcImages);
    anchors.clear();
    //anchors.push_back(imagePaths.size() - 2);
    run(srcImages, params, anchors, opts);

    //xxxx
    //imagePaths.clear();
    //imagePaths.push_back("F:\\panovideo\\ricoh m15\\image2-128.bmp");
    //imagePaths.push_back("F:\\panovideo\\ricoh m15\\image2-128.bmp");
    //loadPhotoParamFromXML("F:\\panovideo\\ricoh m15\\parambestcircle.xml", params);

    //xxxx
    //imagePaths.clear();
    //imagePaths.push_back("F:\\panoimage\\vrdlc\\2016_1011_153743_001.JPG");
    //imagePaths.push_back("F:\\panoimage\\vrdlc\\2016_1011_153743_001.JPG");
    //loadPhotoParamFromXML("F:\\panoimage\\vrdlc\\vrdl-201610112019.xml", params);
    //run(imagePaths, params, opts);

    //xxxx
    //imagePaths.clear();
    //imagePaths.push_back("F:\\panoimage\\vrdlc\\QQÍ¼Æ¬20161014101159.png");
    //imagePaths.push_back("F:\\panoimage\\vrdlc\\QQÍ¼Æ¬20161014101159.png");
    //loadPhotoParamFromXML("F:\\panoimage\\vrdlc\\vrdl-201610112019small.xml", params);
    //run(imagePaths, params, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\919-4-1\\snapshot0.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4-1\\snapshot1.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4-1\\snapshot2.bmp");
    imagePaths.push_back("F:\\panoimage\\919-4-1\\snapshot3.bmp");
    loadPhotoParamFromXML("F:\\panoimage\\919-4-1\\vrdl(4).xml", params);
    loadImages(imagePaths, srcImages);
    anchors.clear();
    //anchors.push_back(imagePaths.size() - 2);
    run(srcImages, params, anchors, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\0.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\1.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\2.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\3.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\4.jpg");
    imagePaths.push_back("F:\\panoimage\\zhanxiang\\5.jpg");
    loadPhotoParamFromXML("F:\\panoimage\\zhanxiang\\zhanxiang.xml", params);
    rotateCameras(params, 0, 35.264 / 180 * PI, PI / 4);
    loadImages(imagePaths, srcImages);
    anchors.clear();
    //anchors.push_back(imagePaths.size() - 2);
    run(srcImages, params, anchors, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang2\\image5.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\test6\\proj.pvs", params);
    loadImages(imagePaths, srcImages);
    anchors.clear();
    //anchors.push_back(imagePaths.size() - 2);
    run(srcImages, params, anchors, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang3\\image5.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\test6\\proj.pvs", params);
    loadImages(imagePaths, srcImages);
    anchors.clear();
    //anchors.push_back(imagePaths.size() - 2);
    run(srcImages, params, anchors, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang4\\image5.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\test6\\proj.pvs", params);
    loadImages(imagePaths, srcImages);
    anchors.clear();
    //anchors.push_back(imagePaths.size() - 2);
    run(srcImages, params, anchors, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\zhanxiang5\\image5.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\test6\\proj.pvs", params);
    loadImages(imagePaths, srcImages);
    anchors.clear();
    //anchors.push_back(imagePaths.size() - 2);
    run(srcImages, params, anchors, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\2\\1\\1.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\2.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\3.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\4.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\5.jpg");
    imagePaths.push_back("F:\\panoimage\\2\\1\\6.jpg");
    loadPhotoParamFromXML("F:\\panoimage\\2\\1\\distortnew.xml", params);
    rotateCameras(params, 0, -35.264 / 180 * PI, -PI / 4);
    loadImages(imagePaths, srcImages);
    anchors.clear();
    //anchors.push_back(imagePaths.size() - 2);
    //anchors.push_back(0);
    run(srcImages, params, anchors, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panoimage\\changtai\\image0.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image1.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image2.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image3.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image4.bmp");
    imagePaths.push_back("F:\\panoimage\\changtai\\image5.bmp");
    loadPhotoParamFromXML("F:\\panoimage\\changtai\\test_test5_cam_param.xml", params);
    loadImages(imagePaths, srcImages);
    anchors.clear();
    //anchors.push_back(imagePaths.size() - 2);
    //anchors.push_back(0);
    //anchors.push_back(1);
    //anchors.push_back(2);
    //anchors.push_back(4);
    run(srcImages, params, anchors, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\1.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\2.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\3.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\4.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\5.MP4.jpg");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\6.MP4.jpg");
    loadPhotoParamFromXML("F:\\panovideo\\test\\chengdu\\´¨Î÷VR-¹·Æ´ÐÜÃ¨4\\proj.pvs", params);
    loadImages(imagePaths, srcImages);
    anchors.clear();
    //anchors.push_back(3);
    //anchors.push_back(4);
    //anchors.push_back(5);
    run(srcImages, params, anchors, opts);

    imagePaths.clear();
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image0.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image1.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image2.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image3.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image4.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image5.bmp");
    imagePaths.push_back("F:\\panovideo\\test\\chengdu\\1\\image6.bmp");
    loadPhotoParamFromXML("F:\\panovideo\\test\\chengdu\\1\\proj.pvs", params);
    loadImages(imagePaths, srcImages);
    anchors.clear();
    //anchors.push_back(imagePaths.size() - 2);
    //anchors.push_back(1);
    //anchors.push_back(3);
    run(srcImages, params, anchors, opts);

    return 0;
}
