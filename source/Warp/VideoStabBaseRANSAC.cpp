#include "Rotation.h"
#include "ConvertCoordinate.h"
#include "Stabilize.h"
#include "ZReproject.h"
#include "ZBlend.h"
#include "Timer.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <cmath>
#include <iostream>
#include <utility>
#include <fstream>

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

    cv::Size frameSize = cv::Size(2048, 1024);

    //ReprojectParam pi;
    //pi.LoadConfig("F:\\QQRecord\\452103256\\FileRecv\\test1\\changtai_cam_param.xml");
    //pi.SetPanoSize(frameSize);
    std::vector<PhotoParam> params;
    loadPhotoParamFromXML("F:\\QQRecord\\452103256\\FileRecv\\test1\\changtai_cam_param.xml", params);

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

    //ReprojectParam pi;
    //pi.LoadConfig("F:\\QQRecord\\452103256\\FileRecv\\test2\\changtai.xml");
    //pi.SetPanoSize(frameSize);

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
        cv::Mat frame;
        for (int j = 0; j < count; j++)
            caps[i].read(frame);
    }

    //std::vector<cv::Mat> dstMasks1;
    //std::vector<cv::Mat> dstSrcMaps1;
    //std::vector<cv::Mat> images1(numVideos), reprojImages1;
    //getReprojectMapsAndMasks(pi, cv::Size(1280, 960), dstSrcMaps1, dstMasks1);
    //for (int i = 0; i < numVideos; i++)
    //{
    //    caps[i].read(images1[i]);
    //}
    //reproject(images1, reprojImages1, dstSrcMaps1);
    //for (int i = 0; i < numVideos; i++)
    //{
    //    char buf[128];        
    //    sprintf(buf, "image%d.bmp", i);
    //    cv::imwrite(buf, images1[i]);
    //    sprintf(buf, "mask%d.bmp", i);
    //    cv::imwrite(buf, dstMasks1[i]);
    //    sprintf(buf, "reprojimage%d.bmp", i);
    //    cv::imwrite(buf, reprojImages1[i]);
    //}
    //return 0;

    cv::Size srcSize(1280, 960);
    int width = 2048, height = 1024;
    cv::Mat color, gray;
    std::vector<cv::Mat> frames(numVideos);
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
        caps[i].read(color);
        cv::cvtColor(color, gray, CV_BGR2GRAY);
        ptrOrb->detectAndCompute(gray, cv::Mat(), pointsPrev[i], descsPrev[i]);
    }
    angles.push_back(cv::Vec3d(0, 0, 0));

    int count = 0;

    cv::Mat showCombined(height, width, CV_8UC3);
    while (true)
    {
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
        if (/*count > 1000 ||*/ !success)
            break;

        //showCombined.setTo(0);
        for (int i = 0; i < numVideos; i++)
        {
            cv::cvtColor(frames[i], gray, CV_BGR2GRAY);
            ptrOrb->detectAndCompute(gray, cv::Mat(), pointsCurr[i], descsCurr[i]);
            matcher.match(descsPrev[i], descsCurr[i], matches[i]);
            extractMatchPoints(pointsPrev[i], pointsCurr[i], matches[i], points1[i], points2[i]);
            //toEquiRect(pi, i, srcSize, points1[i], srcEquiRectPts[i]);
            //toEquiRect(pi, i, srcSize, points2[i], dstEquiRectPts[i]);
            toEquiRect(params[i], frames[i].size(), frameSize, points1[i], srcEquiRectPts[i]);
            toEquiRect(params[i], frames[i].size(), frameSize, points2[i], dstEquiRectPts[i]);
            //drawDirection(srcEquiRectPts[i], dstEquiRectPts[i], showCombined);
            equirectToSphere(srcEquiRectPts[i], width, height, srcSpherePts[i]);
            equirectToSphere(dstEquiRectPts[i], width, height, dstSpherePts[i]);
        }
        //cv::imshow("show", showCombined);
        //cv::waitKey(15);

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

        cv::Matx33d currRot;
        cv::Point3d currTranslation;
        double yaw, pitch, roll;
        //checkMatchedPointsDist(src, dst);
        getRigidTransformRANSAC(src, dst, currRot, currTranslation, mask);
        getRotationRM(currRot, yaw, pitch, roll);
        angles.push_back(cv::Vec3d(yaw, pitch, roll));
        printf("yaw = %f, pitch = %f, roll = %f\n", yaw, pitch, roll);

        for (int i = 0; i < numVideos; i++)
        {
            pointsCurr[i].swap(pointsPrev[i]);
            cv::swap(descsCurr[i], descsPrev[i]);
        }
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

    for (int i = 0; i < numVideos; i++)
    {
        caps[i].open(srcVideoNames[i]);
        int count = offset[i] + numSkip;
        cv::Mat frame;
        for (int j = 0; j < count; j++)
            caps[i].read(frame);
    }

    frameSize = cv::Size(800, 400);

    std::vector<cv::Mat> dstMasks;
    std::vector<cv::Mat> dstSrcMaps;
    getReprojectMapsAndMasks(params, cv::Size(1280, 960), frameSize, dstSrcMaps, dstMasks);

    //TilingMultibandBlend blender;
    TilingMultibandBlendFast blender;
    blender.prepare(dstMasks, 16, 2);

    const char* outPath = "stab_merge_test1_compensate_new_long_fast.avi";
    cv::VideoWriter writer(outPath, CV_FOURCC('X', 'V', 'I', 'D'), 48, frameSize);

    /*
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

    return 0;

    
    return 0;
}