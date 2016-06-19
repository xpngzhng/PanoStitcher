#include "PanoramaTask.h"
#include "PanoramaTaskUtil.h"
#include "AudioVideoProcessor.h"
#include "Log.h"
#include "Timer.h"
#include <fstream>
#include <thread>

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

static void cancelTask(PanoramaLocalDiskTask* task)
{
    std::this_thread::sleep_for(std::chrono::seconds(105));
    if (task)
        task->cancel();
}

int main(int argc, char* argv[])
{
    const char* keys =
        "{camera_param_file      |             | camera param file path}"
        "{video_path_offset_file |             | video path and offset file path}"
        "{num_frames_skip        | 100         | number of frames to skip}"
        "{pano_width             | 2048        | pano picture width}"
        "{pano_height            | 1024        | pano picture height}"
        "{pano_video_name        | panogpu.mp4 | xml param file path}"
        "{pano_video_num_frames  | 1000        | number of frames to write}"
        "{use_cuda               | false       | use gpu to accelerate computation}";

    cv::CommandLineParser parser(argc, argv, keys);

    //initLog("", "");
    //avp::setFFmpegLogCallback(bstLogVlPrintf);
    //avp::setLogCallback(bstLogVlPrintf);
    //setPanoTaskLogCallback(bstLogVlPrintf);

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

    std::string projFileName = "F:\\panovideo\\test\\test1\\haiyangguansimple.xml";
    loadVideoFileNamesAndOffset(projFileName, srcVideoNames, offset);

    std::unique_ptr<PanoramaLocalDiskTask> task;
    if (parser.get<bool>("use_cuda"))
        task.reset(new CudaPanoramaLocalDiskTask);
    else
        task.reset(new CPUPanoramaLocalDiskTask);
    
    bool ok = task->init(srcVideoNames, offset, 0, projFileName, projFileName, panoVideoName,
        dstSize.width, dstSize.height, 8000000, "h264", "medium", 40 * 48);
    if (!ok)
    {
        printf("Could not init panorama local disk task\n");
        std::string msg;
        task->getLastSyncErrorMessage(msg);
        printf("Error message: %s\n", msg.c_str());
        return 0;
    }

    //std::thread t(cancelTask, task.get());
    ztool::Timer timer;
    task->start();
    int progress;
    while (true)
    {
        progress = task->getProgress();
        printf("percent %d\n", progress);
        if (progress == 100)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    task->waitForCompletion();
    printf("percent 100\n");
    timer.end();
    printf("%f\n", timer.elapse());

    //t.join();

    return 0;
}