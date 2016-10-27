#include "Rotation.h"
#include "ConvertCoordinate.h"
#include "Stabilize.h"
#include "ZReproject.h"
#include "Blend/ZBlend.h"
#include "CudaAccel/CudaInterface.h"
#include "Tool/Timer.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <cmath>
#include <iostream>
#include <utility>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

void mapBilinear(const cv::Mat& src, cv::Mat& dst, const cv::Matx33d& rot);
void mapNearestNeighbor(const cv::Mat& src, cv::Mat& dst, const cv::Matx33d& rot);

static void toValidFileName(const std::string& path, std::string& result)
{
    result = path;
    std::replace_if(result.begin(), result.end(), [](char c)->bool {return !(isdigit(c) || isalpha(c)); }, '_');
}

static void toEquiRect(const PhotoParam& param, const cv::Size& srcSize, const cv::Size& dstSize, std::vector<cv::Point2d>& srcPts, std::vector<cv::Point2d>& dstPts)
{
    Remap t;
    t.initInverse(param, dstSize.width, dstSize.height, srcSize.width, srcSize.height);
    int length = srcPts.size();
    dstPts.resize(length);
    for (int i = 0; i < length; i++)
    {
        t.inverseRemapImage(srcPts[i].x, srcPts[i].y, dstPts[i].x, dstPts[i].y);
    }
}

static void smooth(const std::vector<std::vector<double> >& src, int radius, std::vector<std::vector<double> >& dst)
{
    dst.clear();
    if (src.empty())
        return;
    int size = src.size();
    int length = src[0].size();
    dst.resize(size);
    std::vector<double> sum(length);
    for (int i = 0; i < size; i++)
    {
        int beg = std::max(0, i - radius);
        int end = std::min(i + radius, size - 1);
        for (int k = 0; k < length; k++)
            sum[k] = 0;
        for (int j = beg; j <= end; j++)
        {
            for (int k = 0; k < length; k++)
                sum[k] += src[j][k];
        }
        dst[i].resize(length);
        for (int k = 0; k < length; k++)
            dst[i][k] = sum[k] * (1.0 / (end + 1 - beg));
    }
}

struct FeatureDetectAndMatch
{
    FeatureDetectAndMatch(std::vector<cv::Mat>& frames_, std::vector<PhotoParam>& params_,
    std::vector<cv::Mat>& descsPrev_, std::vector<cv::Mat>& descsCurr_,
    std::vector<std::vector<cv::KeyPoint> >& pointsPrev_, std::vector<std::vector<cv::KeyPoint> >& pointsCurr_,
    std::vector<std::vector<cv::DMatch> >& matches_,
    std::vector<std::vector<cv::Point2d> >& points1_, std::vector<std::vector<cv::Point2d> >& points2_,
    std::vector<std::vector<cv::Point2d> >& srcEquiRectPts_, std::vector<std::vector<cv::Point2d> >& dstEquiRectPts_,
    std::vector<std::vector<cv::Point3d> >& srcSpherePts_, std::vector<std::vector<cv::Point3d> >& dstSpherePts_,
    cv::Size srcSize, cv::Size dstSize)
    : frames(frames_), params(params_), descsPrev(descsPrev_), descsCurr(descsCurr_),
    pointsPrev(pointsPrev_), pointsCurr(pointsCurr_), matches(matches_),
    points1(points1_), points2(points2_), srcEquiRectPts(srcEquiRectPts_), dstEquiRectPts(dstEquiRectPts_),
    srcSpherePts(srcSpherePts_), dstSpherePts(dstSpherePts_)
    {
        frameSize = dstSize;
        width = srcSize.width;
        height = srcSize.height;

        numImages = frames.size();
        grays.resize(numImages);
        for (int i = 0; i < numImages; i++)
        {
            ptrOrbs.push_back(cv::ORB::create(250));
            ptrMatchers.push_back(new cv::BFMatcher(cv::NORM_L2, true));
        }

        pass = 0;
        atmVal.store(numImages);
        for (int i = 0; i < numImages; i++)
            threads.emplace_back(std::thread(&FeatureDetectAndMatch::runThread, this, i));
    }

    void start()
    {
        atmVal.store(0);
        condStart.notify_all();
    }

    void runThread(int i)
    {
        while (true)
        {
            std::unique_lock<std::mutex> lg(mtxStart);
            condStart.wait(lg);
            if (pass)
                break;

            cv::cvtColor(frames[i], grays[i], CV_BGR2GRAY);
            ptrOrbs[i]->detectAndCompute(grays[i], cv::Mat(), pointsCurr[i], descsCurr[i]);
            ptrMatchers[i]->match(descsPrev[i], descsCurr[i], matches[i]);
            filterMatches(pointsPrev[i], pointsCurr[i], matches[i], 50);
            extractMatchPoints(pointsPrev[i], pointsCurr[i], matches[i], points1[i], points2[i]);
            toEquiRect(params[i], frames[i].size(), frameSize, points1[i], srcEquiRectPts[i]);
            toEquiRect(params[i], frames[i].size(), frameSize, points2[i], dstEquiRectPts[i]);
            equirectToSphere(srcEquiRectPts[i], width, height, srcSpherePts[i]);
            equirectToSphere(dstEquiRectPts[i], width, height, dstSpherePts[i]);

            atmVal.fetch_add(1);
            if (atmVal.load() == numImages)
                condWait.notify_all();
        }
    }

    void waitForCompletion()
    {
        std::unique_lock<std::mutex> lg(mtxWait);
        condWait.wait(lg, [this]{ return atmVal.load() == numImages; });
    }

    void finish()
    {
        pass = 1;
        condStart.notify_all();
        for (int i = 0; i < numImages; i++)
            threads[i].join();
    }

    std::vector<cv::Ptr<cv::ORB> > ptrOrbs;
    std::vector<cv::Ptr<cv::BFMatcher> > ptrMatchers;
    std::vector<cv::Mat> grays;
    std::vector<PhotoParam> params;
    std::vector<cv::Mat>& frames;
    std::vector<cv::Mat>& descsPrev, &descsCurr;
    std::vector<std::vector<cv::KeyPoint> >& pointsPrev, &pointsCurr;
    std::vector<std::vector<cv::DMatch> >& matches;
    std::vector<std::vector<cv::Point2d> >& points1, &points2;
    std::vector<std::vector<cv::Point2d> >& srcEquiRectPts, &dstEquiRectPts;
    std::vector<std::vector<cv::Point3d> >& srcSpherePts, &dstSpherePts;
    int width, height;
    cv::Size frameSize;
    std::vector<std::thread> threads;
    std::mutex mtxStart, mtxWait;
    std::condition_variable condStart, condWait;
    std::atomic<int> atmVal;
    int pass;
    int numImages;
};

int main()
{
    cv::Ptr<cv::ORB> ptrOrb = cv::ORB::create(250);
    cv::BFMatcher matcher(cv::NORM_L2, true);
    const char* videoPath = "F:\\QQRecord\\452103256\\FileRecv\\mergetest1new.avi";
    cv::VideoCapture cap(videoPath);
    int numFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    cap.release();

    std::vector<cv::Vec3d> angles;
    angles.reserve(numFrames);

    std::vector<std::vector<double> > es, bs, rs;
    es.reserve(numFrames);
    bs.reserve(numFrames);
    rs.reserve(numFrames);

    cv::Size frameSize = cv::Size(1280, 640);

    std::vector<PhotoParam> params;
    loadPhotoParamFromXML("F:\\QQRecord\\452103256\\FileRecv\\test1\\changtai_cam_param.xml", params);
    //loadPhotoParamFromXML("F:\\QQRecord\\452103256\\FileRecv\\test2\\changtai.xml", params);

    std::vector<cv::Mat> dstMasks;
    std::vector<cv::Mat> dstSrcMaps;
    getReprojectMapsAndMasks(params, cv::Size(1280, 960), frameSize, dstSrcMaps, dstMasks);

    ExposureColorCorrect correct;
    correct.prepare(dstMasks);

	// changtai
    std::vector<std::string> srcVideoNames;
    srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0078.mp4");
    srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0081.mp4");
    srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0087.mp4");
    srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0108.mp4");
    srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0118.mp4");
    srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test1\\YDXJ0518.mp4");
    int numVideos = srcVideoNames.size();

    int offset[] = { 563, 0, 268, 651, 91, 412 };
    int numSkip = 2100;

	// zhanxiang
    //std::vector<std::string> srcVideoNames;
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0072.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0075.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0080.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0101.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0112.mp4");
    //srcVideoNames.push_back("F:\\QQRecord\\452103256\\FileRecv\\test2\\YDXJ0512.mp4");
    //int numVideos = srcVideoNames.size();

    //int offset[] = {554, 0, 436, 1064, 164, 785};
    //int numSkip = 3000;

    std::vector<cv::VideoCapture> caps(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        caps[i].open(srcVideoNames[i]);
        int count = offset[i] + numSkip;
        //cv::Mat frame;
        //for (int j = 0; j < count; j++)
        //    caps[i].read(frame);
        caps[i].set(cv::CAP_PROP_POS_FRAMES, count);
    }

    cv::Size srcSize(1280, 960);
    int width = srcSize.width, height = srcSize.height;
    cv::Mat gray;
    std::vector<cv::Mat> frames(numVideos), reprojFrames(numVideos);
    std::vector<cv::Mat> descsPrev(numVideos), descsCurr(numVideos);
    std::vector<std::vector<cv::KeyPoint> > pointsPrev(numVideos), pointsCurr(numVideos);
    std::vector<std::vector<cv::DMatch> > matches(numVideos);
    std::vector<std::vector<cv::Point2d> > points1(numVideos), points2(numVideos);
    std::vector<std::vector<cv::Point2d> > srcEquiRectPts(numVideos), dstEquiRectPts(numVideos);
    std::vector<std::vector<cv::Point3d> > srcSpherePts(numVideos), dstSpherePts(numVideos);
    std::vector<cv::Point3d> src, dst;
    std::vector<unsigned char> mask;

    for (int i = 0; i < numVideos; i++)
    {
        caps[i].read(frames[i]);
        cv::cvtColor(frames[i], gray, CV_BGR2GRAY);
        ptrOrb->detectAndCompute(gray, cv::Mat(), pointsPrev[i], descsPrev[i]);
    }
    angles.push_back(cv::Vec3d(0, 0, 0));

    reprojectParallel(frames, reprojFrames, dstSrcMaps);
    std::vector<double> e, b, r;
    correct.correctExposureAndWhiteBalance(reprojFrames, e, r, b);
    es.push_back(e);
    rs.push_back(r);
    bs.push_back(b);

    int count = 0;
    ztool::Timer t;
    std::vector<double> timeElapse;

    cv::Mat showCombined(frameSize.height, frameSize.width, CV_8UC3);
    while (true)
    {
        printf("count %d\n", count);
        bool success = true;
        for (int i = 0; i < numVideos; i++)
        {
            if (!caps[i].read(frames[i]))
            {
                success = false;
                break;
            }
        }
        ++count;
        if (/*count > 2500 ||*/ !success)
            break;

        timeElapse.clear();
        t.start();
        //showCombined.setTo(0);
        for (int i = 0; i < numVideos; i++)
        {
            cv::cvtColor(frames[i], gray, CV_BGR2GRAY);
            ptrOrb->detectAndCompute(gray, cv::Mat(), pointsCurr[i], descsCurr[i]);
            matcher.match(descsPrev[i], descsCurr[i], matches[i]);
            filterMatches(pointsPrev[i], pointsCurr[i], matches[i], 50);
            extractMatchPoints(pointsPrev[i], pointsCurr[i], matches[i], points1[i], points2[i]);
            toEquiRect(params[i], frames[i].size(), frameSize, points1[i], srcEquiRectPts[i]);
            toEquiRect(params[i], frames[i].size(), frameSize, points2[i], dstEquiRectPts[i]);
            //drawDirection(srcEquiRectPts[i], dstEquiRectPts[i], showCombined);
            equirectToSphere(srcEquiRectPts[i], width, height, srcSpherePts[i]);
            equirectToSphere(dstEquiRectPts[i], width, height, dstSpherePts[i]);
        }
        //cv::imshow("show", showCombined);
        //cv::waitKey(0);
        t.end();
        timeElapse.push_back(t.elapse());

        int pointCount = 0;
        for (int i = 0; i < numVideos; i++)
        {
            pointCount += srcSpherePts[i].size();
        }
        
        src.clear();
        dst.clear();
        src.resize(pointCount);
        dst.resize(pointCount);
        std::vector<cv::Point3d>::iterator itrSrc = src.begin(), itrDst = dst.begin();
        for (int i = 0; i < numVideos; i++)
        {
            itrSrc = std::copy(srcSpherePts[i].begin(), srcSpherePts[i].end(), itrSrc);
            itrDst = std::copy(dstSpherePts[i].begin(), dstSpherePts[i].end(), itrDst);
        }

        t.start();
        cv::Matx33d currRot;
        cv::Point3d currTranslation;
        double yaw, pitch, roll;
        //checkMatchedPointsDist(src, dst);
        getRigidTransformRANSAC(src, dst, currRot, currTranslation, mask);
        getRotationRM(currRot, yaw, pitch, roll);
        angles.push_back(cv::Vec3d(yaw, pitch, roll));
        t.end();
        timeElapse.push_back(t.elapse());
        //printf("yaw = %f, pitch = %f, roll = %f\n", yaw, pitch, roll);

        for (int i = 0; i < numVideos; i++)
        {
            pointsCurr[i].swap(pointsPrev[i]);
            cv::swap(descsCurr[i], descsPrev[i]);
        }

        t.start();
        reprojectParallel(frames, reprojFrames, dstSrcMaps);
        t.end();
        timeElapse.push_back(t.elapse());
        t.start();
        correct.correctExposureAndWhiteBalance(reprojFrames, e, r, b);
        es.push_back(e);
        rs.push_back(r);
        bs.push_back(b);
        t.end();

        printf("detect %f, estimate rotate %f, reproject %f, correct %f\n",
            timeElapse[0], timeElapse[1], timeElapse[2], t.elapse());
    }

    for (int i = 0; i < numVideos; i++)
    {
        caps[i].release();
    }

    std::vector<cv::Vec3d> anglesProc;
    smooth(angles, 30, anglesProc);

    std::vector<cv::Vec3d> anglesAccum, anglesProcAccum;
    accumulate(angles, anglesAccum);
    accumulate(anglesProc, anglesProcAccum);

    //int length = angles.size();
    //std::string prefix;
    //toValidFileName(videoPath, prefix);
    //prefix.append("_");
    //prefix.append(std::to_string((long long)length));
    //std::string dataFileName;
    //std::ofstream of;
    //dataFileName = prefix + "_angles.dat";
    //of.open(dataFileName.c_str(), std::ios_base::binary);
    //of.write((char*)&angles[0], length * sizeof(cv::Vec3d));
    //of.close();
    //of.clear();
    //dataFileName = prefix + "_anglesProc.dat";
    //of.open(dataFileName.c_str(), std::ios_base::binary);
    //of.write((char*)&anglesProc[0], length * sizeof(cv::Vec3d));
    //of.close();
    //return 0;
    //int length = 4800;
    //std::ifstream ifs;
    //ifs.open("F__QQRecord_452103256_FileRecv_mergetest2new_avi_4800_angles.dat", std::ios_base::binary);
    //angles.resize(length);
    //ifs.read((char*)&angles[0], length * sizeof(cv::Vec3d));
    //ifs.close();
    //ifs.clear();
    //ifs.open("F__QQRecord_452103256_FileRecv_mergetest2new_avi_4800_anglesProc.dat", std::ios_base::binary);
    //anglesProc.resize(length);
    //ifs.read((char*)&anglesProc[0], length * sizeof(cv::Vec3d));
    //ifs.close();

    //cv::Mat angleShow;
    //cv::Scalar colors[] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
    //draw(angles, colors, angleShow);
    //cv::imshow("angles", angleShow);
    //draw(anglesProc, colors, angleShow);
    //cv::imshow("angles proc", angleShow);
    //draw(anglesAccum, colors, angleShow);
    //cv::imshow("angles accum", angleShow);
    //draw(anglesProcAccum,colors, angleShow);
    //cv::imshow("angles proc accum", angleShow);
    //cv::waitKey(0);
    //return 0;  

    int num = es.size();
    std::vector<std::vector<double> > rsProc, bsProc;
    smooth(rs, 96, rsProc);
    smooth(bs, 96, bsProc);

    std::vector<std::vector<std::vector<unsigned char> > > luts;

    for (int i = 0; i < numVideos; i++)
    {
        caps[i].open(srcVideoNames[i]);
        int count = offset[i] + numSkip;
        //cv::Mat frame;
        //for (int j = 0; j < count; j++)
        //    caps[i].read(frame);
        caps[i].set(cv::CAP_PROP_POS_FRAMES, count);
    }

    //frameSize = cv::Size(800, 400);

    const char* outPath = "stab_exposure_color_correct_1.avi";
    cv::VideoWriter writer(outPath, CV_FOURCC('X', 'V', 'I', 'D'), 48, frameSize);

    // rotate reproject adjust blend
    /*
    TilingMultibandBlendFast blender;
    blender.prepare(dstMasks, 16, 2);

    cv::Mat blendImage;
    int frameCount = 0;
    int maxCount = angles.size();
    std::vector<cv::Mat> srcImages(numVideos);
    std::vector<cv::Mat> dstImages, compImages;
    ztool::Timer timer;
    cv::Vec3d accumOrig(0, 0, 0), accumProc(0, 0, 0);
    while (true)
    {
        printf("currCount = %d\n", frameCount);
        bool success = true;
        for (int i = 0; i < numVideos; i++)
        {
            if (!caps[i].read(srcImages[i]))
            {
                success = false;
                break;
            }
        }
        if (!success)
            break;

        accumOrig += angles[frameCount];
        accumProc += anglesProc[frameCount];
        printf("accumOrig = (%f, %f, %f), accumProc = (%f, %f, %f)\n",
            accumOrig[0], accumOrig[1], accumOrig[2],
            accumProc[0], accumProc[1], accumProc[2]);
        cv::Vec3d diff = accumProc - accumOrig;

        printf("reproject:\n");
        std::vector<PhotoParam> currParams = params;
        rotateCameras(currParams, diff[0], diff[1], diff[2]);
        getReprojectMapsAndMasks(currParams, cv::Size(1280, 960), frameSize, dstSrcMaps, dstMasks);
        reprojectParallel(srcImages, dstImages, dstSrcMaps);

        ExposureColorCorrect::getExposureAndWhiteBalanceLUTs(es[frameCount], rsProc[frameCount], bsProc[frameCount], luts);
        for (int i = 0; i < numVideos; i++)
            transform(dstImages[i], dstImages[i], luts[i], dstMasks[i]);

        //timer.start();
        printf("blend:\n");
        blender.blendAndCompensate(dstImages, dstMasks, blendImage);
        //timer.end();
        //printf("%f\n", timer.elapse());

        writer.write(blendImage);

        frameCount++;
        if (frameCount >= maxCount)
            break;
    }
    */
    
    // reproject adjust blend rotate
    /*
    TilingMultibandBlendFast blender;
    blender.prepare(dstMasks, 16, 2);

    cv::Mat blendImage;
    int frameCount = 0;
    int maxCount = angles.size();
    std::vector<cv::Mat> srcImages(numVideos);
    std::vector<cv::Mat> dstImages, compImages;
    ztool::Timer timer;
    cv::Vec3d accumOrig(0, 0, 0), accumProc(0, 0, 0);
    cv::Mat remapFrame;
    while (true)
    {
        printf("currCount = %d\n", frameCount);
        bool success = true;
        for (int i = 0; i < numVideos; i++)
        {
            if (!caps[i].read(srcImages[i]))
            {
                success = false;
                break;
            }
        }
        if (!success)
            break;

        accumOrig += angles[frameCount];
        accumProc += anglesProc[frameCount];
        printf("accumOrig = (%f, %f, %f), accumProc = (%f, %f, %f)\n",
            accumOrig[0], accumOrig[1], accumOrig[2],
            accumProc[0], accumProc[1], accumProc[2]);
        cv::Vec3d diff = accumProc - accumOrig;

        printf("reproject:\n");
        reprojectParallel(srcImages, dstImages, dstSrcMaps);

        ExposureColorCorrect::getExposureAndWhiteBalanceLUTs(es[frameCount], rsProc[frameCount], bsProc[frameCount], luts);
        for (int i = 0; i < numVideos; i++)
            transform(dstImages[i], dstImages[i], luts[i], dstMasks[i]);

        //timer.start();
        printf("blend:\n");
        blender.blend(dstImages, blendImage);
        //timer.end();
        //printf("%f\n", timer.elapse());

        cv::Matx33d rot;
        setRotationRM(rot, diff[0], diff[1], diff[2]);
        mapBilinear(blendImage, remapFrame, rot);
        writer.write(remapFrame);

        frameCount++;
        if (frameCount >= maxCount)
            break;
    }
    */

    // rotate reproject adjust blend
    /*
    BlendConfig config;
    config.setBlendMultiBand(8, 4);
    config.setSeamDistanceTransform();

    cv::Mat blendImage;
    int frameCount = 0;
    int maxCount = angles.size();
    std::vector<cv::Mat> srcImages(numVideos);
    std::vector<cv::Mat> dstImages, compImages;
    ztool::Timer timer;
    cv::Vec3d accumOrig(0, 0, 0), accumProc(0, 0, 0);
    while (true)
    {
        printf("currCount = %d\n", frameCount);
        bool success = true;
        for (int i = 0; i < numVideos; i++)
        {
            if (!caps[i].read(srcImages[i]))
            {
                success = false;
                break;
            }
        }
        if (!success)
            break;

        accumOrig += angles[frameCount];
        accumProc += anglesProc[frameCount];
        printf("accumOrig = (%f, %f, %f), accumProc = (%f, %f, %f)\n",
            accumOrig[0], accumOrig[1], accumOrig[2],
            accumProc[0], accumProc[1], accumProc[2]);
        cv::Vec3d diff = accumProc - accumOrig;

        printf("reproject:\n");
        std::vector<PhotoParam> currParams = params;
        rotateCameras(currParams, diff[0], diff[1], diff[2]);
        getReprojectMapsAndMasks(currParams, cv::Size(1280, 960), frameSize, dstSrcMaps, dstMasks);
        reprojectParallel(srcImages, dstImages, dstSrcMaps);

        ExposureColorCorrect::getExposureAndWhiteBalanceLUTs(es[frameCount], rsProc[frameCount], bsProc[frameCount], luts);
        for (int i = 0; i < numVideos; i++)
            transform(dstImages[i], dstImages[i], luts[i], dstMasks[i]);

        //timer.start();
        printf("blend:\n");
        parallelBlend(config, dstImages, dstMasks, blendImage);
        //timer.end();
        //printf("%f\n", timer.elapse());

        writer.write(blendImage);

        frameCount++;
        if (frameCount >= maxCount)
            break;
    }
    */

    // reproject adjust blend rotate
    /*
    TilingMultibandBlendFast blender;
    blender.prepare(dstMasks, 16, 2);

    cv::Mat blendImage;
    int frameCount = 0;
    int maxCount = angles.size();
    std::vector<cv::Mat> srcImages(numVideos);
    std::vector<cv::Mat> dstImages, compImages;
    ztool::Timer timer;
    cv::Vec3d accumOrig(0, 0, 0), accumProc(0, 0, 0);
    cv::Mat remapFrame;
    while (true)
    {
        printf("currCount = %d\n", frameCount);
        bool success = true;
        for (int i = 0; i < numVideos; i++)
        {
            if (!caps[i].read(srcImages[i]))
            {
                success = false;
                break;
            }
        }
        if (!success)
            break;

        accumOrig += angles[frameCount];
        accumProc += anglesProc[frameCount];
        printf("accumOrig = (%f, %f, %f), accumProc = (%f, %f, %f)\n",
            accumOrig[0], accumOrig[1], accumOrig[2],
            accumProc[0], accumProc[1], accumProc[2]);
        cv::Vec3d diff = accumProc - accumOrig;

        printf("reproject:\n");
        reprojectParallel(srcImages, dstImages, dstSrcMaps);

        ExposureColorCorrect::getExposureAndWhiteBalanceLUTs(es[frameCount], rsProc[frameCount], bsProc[frameCount], luts);
        for (int i = 0; i < numVideos; i++)
            transform(dstImages[i], dstImages[i], luts[i], dstMasks[i]);

        //timer.start();
        printf("blend:\n");
        blender.blend(dstImages, blendImage);
        //timer.end();
        //printf("%f\n", timer.elapse());

        cv::Matx33d rot;
        setRotationRM(rot, diff[0], diff[1], diff[2]);
        mapBilinear(blendImage, remapFrame, rot);
        writer.write(remapFrame);

        frameCount++;
        if (frameCount >= maxCount)
            break;
    }
    */

    // cuda reproject adjust blend rotate
    /*
    frameSize.width = 2048;
    frameSize.height = 1024;
    writer.open(outPath, CV_FOURCC('X', 'V', 'I', 'D'), 48, frameSize);
    CudaTilingMultibandBlendFast blender;
    getReprojectMapsAndMasks(params, srcSize, frameSize, dstSrcMaps, dstMasks);
    blender.prepare(dstMasks, 10, 4);

    std::vector<cv::cuda::HostMem> srcHostMems(numVideos);
    for (int i = 0; i < numVideos; i++)
        srcHostMems[i].create(srcSize, CV_8UC4);

    std::vector<cv::cuda::GpuMat> xmapsGpu, ymapsGpu;
    cudaGenerateReprojectMaps(params, srcSize, frameSize, xmapsGpu, ymapsGpu);

    std::vector<cv::cuda::GpuMat> srcImagesGpu(numVideos), procImagesGpu(numVideos), masksGpu(numVideos);
    cv::cuda::GpuMat blendImageGpu, rotateImageGpu;
    cv::cuda::HostMem blendMem(frameSize, CV_8UC4);
    std::vector<cv::cuda::Stream> streams(numVideos);

    cv::Mat blendImageC4 = blendMem.createMatHeader(), blendImage;
    cv::Mat showImage, showMask;
    int frameCount = 0;
    int maxCount = angles.size();
    ztool::Timer timer;
    cv::Vec3d accumOrig(0, 0, 0), accumProc(0, 0, 0);
    cv::Mat temp;
    while (true)
    {
        printf("currCount = %d\n", frameCount);
        bool success = true;
        for (int i = 0; i < numVideos; i++)
        {
            if (!caps[i].read(temp))
            {
                success = false;
                break;
            }
            cv::cvtColor(temp, srcHostMems[i].createMatHeader(), CV_BGR2BGRA);
        }
        if (!success)
            break;

        accumOrig += angles[frameCount];
        accumProc += anglesProc[frameCount];
        //printf("accumOrig = (%f, %f, %f), accumProc = (%f, %f, %f)\n",
        //    accumOrig[0], accumOrig[1], accumOrig[2],
        //    accumProc[0], accumProc[1], accumProc[2]);
        cv::Vec3d diff = accumProc - accumOrig;

        //printf("reproject:\n");
        ExposureColorCorrect::getExposureAndWhiteBalanceLUTs(es[frameCount], rsProc[frameCount], bsProc[frameCount], luts);
        for (int i = 0; i < numVideos; i++)
            srcImagesGpu[i].upload(srcHostMems[i].createMatHeader(), streams[i]);
        for (int i = 0; i < numVideos; i++)
            cudaTransform(srcImagesGpu[i], srcImagesGpu[i], luts[i], streams[i]);
        for (int i = 0; i < numVideos; i++)
            cudaReprojectTo16S(srcImagesGpu[i], procImagesGpu[i], xmapsGpu[i], ymapsGpu[i], streams[i]);
        for (int i = 0; i < numVideos; i++)
            streams[i].waitForCompletion();

        //timer.start();
        //printf("blend:\n");
        blender.blend(procImagesGpu, blendImageGpu);
        //timer.end();
        //printf("%f\n", timer.elapse());

        cv::Matx33d rot;
        setRotationRM(rot, diff[0], diff[1], diff[2]);
        cudaRotateEquiRect(blendImageGpu, rotateImageGpu, rot);

        rotateImageGpu.download(blendImageC4);
        cv::cvtColor(blendImageC4, blendImage, CV_BGRA2BGR);

        writer.write(blendImage);

        frameCount++;
        if (frameCount >= maxCount)
            break;
    }
    */

    // cuda rotate reproject adjust blend
    frameSize.width = 2048;
    frameSize.height = 1024;
    writer.open(outPath, CV_FOURCC('X', 'V', 'I', 'D'), 48, frameSize);
    CudaTilingMultibandBlend blender;
    getReprojectMapsAndMasks(params, srcSize, frameSize, dstSrcMaps, dstMasks);
    blender.prepare(dstMasks, 10, 4);

    std::vector<cv::cuda::HostMem> srcHostMems(numVideos);
    for (int i = 0; i < numVideos; i++)
        srcHostMems[i].create(srcSize, CV_8UC4);

    std::vector<cv::cuda::GpuMat> srcImagesGpu(numVideos), procImagesGpu(numVideos), masksGpu(numVideos);
    cv::cuda::GpuMat blendImageGpu;
    cv::cuda::HostMem blendMem(frameSize, CV_8UC4);
    std::vector<cv::cuda::Stream> streams(numVideos);

    cv::Mat blendImageC4 = blendMem.createMatHeader(), blendImage;
    cv::Mat showImage, showMask;
    int frameCount = 0;
    int maxCount = angles.size();
    ztool::Timer timer;
    cv::Vec3d accumOrig(0, 0, 0), accumProc(0, 0, 0);
    cv::Mat temp;
    while (true)
    {
        printf("currCount = %d\n", frameCount);
        bool success = true;
        for (int i = 0; i < numVideos; i++)
        {
            if (!caps[i].read(temp))
            {
                success = false;
                break;
            }
            cv::cvtColor(temp, srcHostMems[i].createMatHeader(), CV_BGR2BGRA);
        }
        if (!success)
            break;

        accumOrig += angles[frameCount];
        accumProc += anglesProc[frameCount];
        //printf("accumOrig = (%f, %f, %f), accumProc = (%f, %f, %f)\n",
        //    accumOrig[0], accumOrig[1], accumOrig[2],
        //    accumProc[0], accumProc[1], accumProc[2]);
        cv::Vec3d diff = accumProc - accumOrig;

        std::vector<PhotoParam> currParams = params;
        rotateCameras(currParams, diff[0], diff[1], diff[2]);

        //printf("reproject:\n");
        ExposureColorCorrect::getExposureAndWhiteBalanceLUTs(es[frameCount], rsProc[frameCount], bsProc[frameCount], luts);
        for (int i = 0; i < numVideos; i++)
            srcImagesGpu[i].upload(srcHostMems[i].createMatHeader(), streams[i]);
        for (int i = 0; i < numVideos; i++)
            cudaTransform(srcImagesGpu[i], srcImagesGpu[i], luts[i], streams[i]);
        for (int i = 0; i < numVideos; i++)
            cudaReprojectTo16S(srcImagesGpu[i], procImagesGpu[i], masksGpu[i], frameSize, currParams[i], streams[i]);
        for (int i = 0; i < numVideos; i++)
            streams[i].waitForCompletion();

        //timer.start();
        //printf("blend:\n");
        blender.blendAndCompensate(procImagesGpu, masksGpu, blendImageGpu);
        //timer.end();
        //printf("%f\n", timer.elapse());

        blendImageGpu.download(blendImageC4);
        cv::cvtColor(blendImageC4, blendImage, CV_BGRA2BGR);

        writer.write(blendImage);

        frameCount++;
        if (frameCount >= maxCount)
            break;
    }

    
    return 0;
}