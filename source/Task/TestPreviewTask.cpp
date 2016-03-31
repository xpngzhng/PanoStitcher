#include "PanoramaTask.h"
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

    cameraParamFile = parser.get<std::string>("camera_param_file");
    if (cameraParamFile.empty())
    {
        printf("Could not find camera_param_file\n");
        return 0;
    }

    dstSize.width = parser.get<int>("pano_width");
    dstSize.height = parser.get<int>("pano_height");

    dstSize = cv::Size(1024, 512);

    videoPathAndOffsetFile = parser.get<std::string>("video_path_offset_file");
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

    std::unique_ptr<PanoramaPreviewTask> task;
    if (parser.get<bool>("use_cuda"))
        task.reset(new CudaPanoramaPreviewTask);
    else
        task.reset(new CPUPanoramaPreviewTask);

    bool ok = task->init(srcVideoNames, cameraParamFile, dstSize.width, dstSize.height);
    if (!ok)
    {
        printf("Could not init panorama local disk task\n");
        return 0;
    }

    int numVideos = srcVideoNames.size();

    cv::Mat image;
    std::vector<long long int> timeStamps, tempTimeStamps;
    ok = task->stitch(image, timeStamps, 1);
    if (!ok)
    {
        printf("stitch failed\n");
        return 0;
    }
    while (task->stitch(image, timeStamps, 4))
    {
        cv::imshow("render", image);
        int key = cv::waitKey(1);
        if (key == 'q')
            break;
    }
    return 0;

    cv::imshow("render", image);
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
                ok = task->stitch(image, timeStamps, 1);
                if (ok)
                    cv::imshow("render", image);
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
                ok = task->stitch(image, timeStamps, 1);
                if (ok)
                    cv::imshow("render", image);
            }
        }
    }

    return 0;
}