#include "PanoramaTask.h"
#include "PanoramaTaskUtil.h"
#include "Tool/Timer.h"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>

static void parseVideoPathsAndOffsets(const std::string& infoFileName, std::vector<std::string>& videoPath, std::vector<int>& offset)
{
    videoPath.clear();
    offset.clear();

    std::ifstream fstrm(infoFileName);
    std::string line;
    while (!fstrm.eof())
    {
        std::getline(fstrm, line);
        if (line.empty())
            continue;

        std::string::size_type pos = line.find(',');
        if (pos == std::string::npos)
            continue;

        videoPath.push_back(line.substr(0, pos));
        offset.push_back(atoi(line.substr(pos + 1).c_str()));
    }
}

int main(int argc, char* argv[])
{
    const char* keys =
        "{camera_param_file      |       | camera param file path}"
        "{video_path_offset_file |       | video path and offset file path}"
        "{pano_width             | 2048  | pano picture width}"
        "{pano_height            | 1024  | pano picture height}"
        "{use_cuda               | false | use gpu to accelerate computation}";

    cv::CommandLineParser parser(argc, argv, keys);

    cv::Size srcSize, dstSize;
    std::vector<std::string> srcVideoNames;
    std::vector<int> offset;
    int numSkip = 1500;
    std::string cameraParamFile, videoPathAndOffsetFile;
    std::string panoVideoName;

    cameraParamFile = "F:\\panovideo\\test\\test6\\changtai.xml"/*parser.get<std::string>("camera_param_file")*/;
    if (cameraParamFile.empty())
    {
        printf("Could not find camera_param_file\n");
        return 0;
    }

    dstSize.width = parser.get<int>("pano_width");
    dstSize.height = parser.get<int>("pano_height");

    dstSize = cv::Size(1024, 512);

    videoPathAndOffsetFile = "F:\\panovideo\\test\\test6\\synchro_param_copy.txt"/*parser.get<std::string>("video_path_offset_file")*/;
    if (videoPathAndOffsetFile.empty())
    {
        printf("Could not find video_path_offset_file\n");
        return 0;
    }
    parseVideoPathsAndOffsets(videoPathAndOffsetFile, srcVideoNames, offset);
    if (srcVideoNames.empty() || offset.empty())
    {
        printf("Could not parse video path and offset\n");
        return 0;
    }

    numSkip = parser.get<int>("num_frames_skip");
    if (numSkip < 0)
        numSkip = 0;
    for (int i = 0; i < offset.size(); i++)
        offset[i] += numSkip;

    panoVideoName = parser.get<std::string>("pano_video_name");

    std::string projFileName = "F:\\panovideo\\test\\colorgrid\\colorgrid.xml"
        /*"F:\\panovideo\\test\\test1\\haiyangguan.xml"*/;
    loadVideoFileNamesAndOffset(projFileName, srcVideoNames, offset);

    std::unique_ptr<PanoramaPreviewTask> task;
    //if (parser.get<bool>("use_cuda"))
    //    task.reset(new CudaPanoramaPreviewTask);
    //else
        task.reset(new CPUPanoramaPreviewTask);

    bool ok = task->init(srcVideoNames, projFileName, dstSize.width, dstSize.height);
    if (!ok)
    {
        printf("Could not init panorama local disk task\n");
        return 0;
    }

    CPUPanoramaPreviewTask* cpuTask = dynamic_cast<CPUPanoramaPreviewTask*>(task.get());
    if (cpuTask)
    {
        std::vector<std::vector<IntervaledContour> > contours;
        //getIntervaledContoursFromPreviewTask(*cpuTask, contours);
        loadIntervaledContours(projFileName, contours);

        double fps = cpuTask->getVideoFrameRate();
        int numVideos = cpuTask->getNumSourceVideos();
        std::vector<long long int> timeStamps(numVideos);
        for (int i = 0; i < numVideos; i++)
            timeStamps[i] = 1000000.0 / fps * offset[i];
        cpuTask->seek(timeStamps);

        std::vector<cv::Mat> masks, uniqueMasks;
        cpuTask->getMasks(masks);
        cpuTask->getUniqueMasks(uniqueMasks);

        //setIntervaledContoursToPreviewTask(contours, *cpuTask);
        //for (int i = 0; i < masks.size(); i++)
        //    cpuTask->setCustomMaskForOne(i, -200, 100, masks[i]);
        int videoIndex;
        int begIndex, endIndex;
        
        begIndex = 10, endIndex = 50;
        videoIndex = 0;
        cpuTask->setCustomMaskForOne(videoIndex, offset[videoIndex] + begIndex, offset[videoIndex] + endIndex, masks[videoIndex]);
        videoIndex = 1;
        cpuTask->setCustomMaskForOne(videoIndex, offset[videoIndex] + begIndex, offset[videoIndex] + endIndex, masks[videoIndex]);

        begIndex = 70, endIndex = 90;
        for (int i = 0; i < masks.size(); i++)
            cpuTask->setCustomMaskForOne(i, offset[i] + begIndex, offset[i] + endIndex, masks[i]);

        begIndex = 100, endIndex = 150;
        videoIndex = 0;
        cpuTask->setCustomMaskForOne(videoIndex, offset[videoIndex] + begIndex, offset[videoIndex] + endIndex, masks[videoIndex]);
        videoIndex = 1;
        cpuTask->setCustomMaskForOne(videoIndex, offset[videoIndex] + begIndex, offset[videoIndex] + endIndex, masks[videoIndex]);
        videoIndex = 3;
        cpuTask->setCustomMaskForOne(videoIndex, offset[videoIndex] + begIndex, offset[videoIndex] + endIndex, masks[videoIndex]);
        videoIndex = 4;
        cpuTask->setCustomMaskForOne(videoIndex, offset[videoIndex] + begIndex, offset[videoIndex] + endIndex, masks[videoIndex]);

        getIntervaledContoursFromPreviewTask(*cpuTask, offset, contours);

        int a = 0;
    }

    int numVideos = srcVideoNames.size();
    ztool::Timer t;
    int stitchCount = 0;

    char buf[64];
    std::vector<std::string> srcNames(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        sprintf(buf, "src%d", i);
        srcNames[i] = buf;
    }
    std::vector<cv::Mat> src;
    cv::Mat dst;
    std::vector<long long int> timeStamps, tempTimeStamps;
    ok = task->stitch(src, timeStamps, dst, 1);
    if (!ok)
    {
        printf("stitch failed\n");
        return 0;
    }
    t.start();
    while (task->stitch(src, timeStamps, dst, 1))
    {
        //for (int i = 0; i < numVideos; i++)
        //    cv::imshow(srcNames[i], src[i]);
        cv::imshow("render", dst);
        int key = cv::waitKey(0);
        if (key == 'q')
            break;
        stitchCount++;
        printf("stitch count %d\n", stitchCount);
        if (stitchCount % 20 == 0)
        {
            t.end();
            printf("fps %f\n", stitchCount / t.elapse());
        }
    }
    return 0;

    cv::imshow("render", dst);
    while (true)
    {
        int key = cv::waitKey(0);
        if (key == 'q')
            break;
        else if (key == 'a')
        {
            printf("<-\n");
            tempTimeStamps = timeStamps;
            for (int i = 0; i < numVideos; i++)
                tempTimeStamps[i] -= 500000;
            ok = task->seek(tempTimeStamps);
            if (ok)
            {
                timeStamps = tempTimeStamps;
                ok = task->stitch(src, timeStamps, dst, 1);
                if (ok)
                    cv::imshow("render", dst);
            }
        }
        else if (key == 'l')
        {
            printf("->\n");
            tempTimeStamps = timeStamps;
            for (int i = 0; i < numVideos; i++)
                tempTimeStamps[i] += 500000;
            ok = task->seek(tempTimeStamps);
            if (ok)
            {
                timeStamps = tempTimeStamps;
                ok = task->stitch(src, timeStamps, dst, 1);
                if (ok)
                    cv::imshow("render", dst);
            }
        }
    }

    return 0;
}