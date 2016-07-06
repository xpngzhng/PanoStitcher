#include "Text.h"
#include <vector>

static int lang = 0;

void setLanguage(bool isChinese)
{
    lang = isChinese ? 0 : 1;
}

struct IndexTextsPair
{
    int index;
    std::string texts[2];
};

static IndexTextsPair indexTextsPairs[] =
{
    TI_PERIOD, { "。", "." },
    TI_COLON, { "：", ":" },
    TI_LINE_BREAK, { "\n", "\n" },
    TI_SPACE, { " ", " " },

    TI_PARAM_CHECK_FAIL, { "参数校验失败", "Parameters check failed" },
    TI_STITCH_INIT_FAIL, { "视频拼接初始化失败", "Could not initialize stitching" },
    TI_OPEN_VIDEO_FAIL, { "打开视频失败", "Could not open videos" },
    TI_CREATE_STITCH_VIDEO_FAIL, { "无法创建全景视频", "Could not create panorama video" },
    TI_WRITE_TO_VIDEO_FAIL_TASK_TERMINATE, { "写入视频失败，任务终止", "Could not write panorama video, task terminated" },
    TI_STITCH_FAIL_TASK_TERMINATE, { "视频拼接发生错误，任务终止", "Could not stitch panorama frame, task terminated" },

    TI_AUDIO_VIDEO_SOURCE_RUNNING_CLOSE_BEFORE_LANCH_NEW, { "音视频源任务正在运行中，先关闭当前运行的任务，再启动新的任务", "Audio video sources task is running, termiate current task before lanching a new one" },

    TI_VIDEO_SOURCE_RUNNING_CLOSE_BEFORE_LAUNCH_NEW, { "视频源任务正在运行中，先关闭当前运行的任务，再启动新的任务", "Video sources task is running, terminate current task before launching a new one" },
    TI_VIDEO_SOURCE_EMPTY, { "视频源地址为空，请重新设定", "Video sources URLs are empty, please reset" },
    TI_VIDEO_SOURCE_PROP_SHOULD_MATCH, { "所有视频源需要有同样的分辨率和帧率", "All video sources should share the same resolution and the same frame rate" },
    TI_VIDEO_SOURCE_OPEN_FAIL, { "视频源打开失败", "Could not open video sources" },
    TI_VIDEO_SOURCE_OPEN_SUCCESS, { "视频源打开成功", "Successfully opened video sources" },
    TI_VIDEO_SOURCE_TASK_LAUNCH_SUCCESS, { "视频源任务启动", "Video sources task launched" },
    TI_VIDEO_SOURCE_TASK_FINISH, { "视频源任务结束", "Video sources task finished" },
    TI_VIDEO_SOURCE_CLOSE, { "视频源关闭", "Video sources closed" },

    TI_AUDIO_SOURCE_RUNNING_CLOSE_BEFORE_LAUNCH_NEW, { "音频源任务正在运行中，先关闭当前运行的任务，再启动新的任务", "Audio source task is running, terminate current task before launching a new one" },
    TI_AUDIO_SOURCE_EMPTY, { "音频源地址为空，请重新设定", "Audio source URL is empty, please reset" },
    TI_AUDIO_SOURCE_OPEN_FAIL, { "音频源打开失败", "Could not open audio source" },
    TI_AUDIO_SOURCE_OPEN_SUCCESS, { "音频源打开成功", "Successfully opened audio source" },
    TI_AUDIO_SOURCE_TASK_LAUNCH_SUCCESS, { "音频源任务启动", "Audio source task launched" },
    TI_AUDIO_SOURCE_TASK_FINISH, { "音频源任务结束", "Audio source task finished" },
    TI_AUDIO_SOURCE_CLOSE, { "音频源关闭", "Audio source closed" },

    TI_SOURCE_NOT_OPENED_CANNOT_LAUNCH_STITCH, { "尚未打开音视频源，无法启动拼接任务", "Video (and audio) sources have not been opened, cannot launch stitching task" },
    TI_STITCH_RUNNING_CLOSE_BEFORE_LAUNCH_NEW, { "视频拼接任务正在进行中，请先关闭正在执行的任务，再启动新的任务", "Stitching task is running, terminate current task before launching a new one" },
    TI_STITCH_INIT_SUCCESS, { "视频拼接初始化成功", "Successfull initialized stitching" },
    TI_STITCH_TASK_LAUNCH_SUCCESS, { "视频拼接任务启动", "Stitching task launched" },
    TI_STITCH_TASK_FINISH, { "视频拼接任务结束", "Stitching task finished" },

    TI_SOURCE_NOT_OPENED_CANNOT_LAUNCH_LIVE, { "尚未打开音视频源，无法启动直播任务", "Video (and audio) sources have not been opened, cannot launch live stream task" },
    TI_STITCH_NOT_RUNNING_CANNOT_LAUNCH_LIVE, { "尚未启动拼接任务，无法启动直播任务", "Stitching task has not been lanched, cannot launch live stream task" },
    TI_LIVE_RUNNING_CLOSE_BEFORE_LAUNCH_NEW, { "直播任务正在进行中，请先关闭正在执行的任务，再启动新的任务。", "Live stream task is running, terminate current task before launching a new one" },
    TI_LIVE_PARAM_ERROR_CANNOT_LAUNCH_LIVE, { "参数错误，无法启动推流任务", "Invalid pamameters, cannot lanch live stream task" },
    TI_SERVER_CONNECT_FAIL, { "流媒体服务器连接失败", "Could not connect stream media server" },
    TI_SERVER_CONNECT_SUCCESS, { "流媒体服务器连接成功", "Sucessfully connected stream media server" },
    TI_LIVE_TASK_LAUNCH_SUCCESS, { "直播任务启动", "Live stream task launched" },
    TI_LIVE_TASK_FINISH, { "直播任务结束", "Live stream task finished" },
    TI_SERVER_DISCONNECT, { "流媒体服务器连接断开", "Disconnected stream media server" },

    TI_SOURCE_NOT_OPENED_CANNOT_LAUNCH_WRITE, { "尚未打开音视频源，无法启动保存任务", "Video (and audio) sources have not been opened, cannot launch saving to hard disk task" },
    TI_STITCH_NOT_RUNNING_CANNOT_LAUNCH_WRITE, { "尚未启动拼接任务，无法启动保存任务", "Stitching task has not been lanched, cannot launch saving to hard disk task" },
    TI_WRITE_RUNNING_CLOSE_BEFORE_LAUNCH_NEW, { "保存任务正在进行中，请先关闭正在执行的任务，再启动新的任务。", "Saving to hard disk task is running, terminate current task before launching a new one" },
    TI_WRITE_PARAM_ERROR_CANNOT_LAUNCH_WRITE, { "参数错误，无法启动保存任务", "Invalid pamameters, cannot lanch saving to hard disk task" },
    TI_WRITE_LAUNCH, { "保存任务启动", "Saving to hard disk task launched" },
    TI_WRITE_FINISH, { "保存任务结束", "Saving to hard disk task finished" },

    TI_ACQUIRE_VIDEO_SOURCE_FAIL_TASK_TERMINATE, { "获取视频源数据发生错误，任务终止", "Could not acquire data from video source(s), all the running tasks terminated" },
    TI_ACQUIRE_AUDIO_SOURCE_FAIL_TASK_TERMINATE, { "获取音频源数据发生错误，任务终止", "Could not acquire data from audio source, all the running tasks terminated" },
    TI_LIVE_FAIL_TASK_TERMINATE, { "推流发生错误，任务终止", "Could not send data to stream media server, all the running tasks terminated" },

    TI_FILE_OPEN_FAIL_TASK_TERMINATE, { "文件无法打开，任务终止", "file could not be opened, saving to hard disk task terminated" },
    TI_BEGIN_WRITE, { "开始写入", "began writing into this file" },
    TI_END_WRITE, { "写入结束", "finished writing into this file" },
    TI_WRITE_FAIL_TASK_TERMINATE, { "写入发生错误，任务终止", "could not write into this file, saving to hard disk task terminated" },
};

struct IndexedTexts
{
    IndexedTexts()
    {
        pairs.resize(TI_TEXT_NUM);
        for (int i = 0; i < TI_TEXT_NUM; i++)
        {
            int index = TI_TEXT_NUM;
            for (int j = 0; j < TI_TEXT_NUM; j++)
            {
                if (indexTextsPairs[j].index == i)
                {
                    index = i;
                    break;
                }
            }
            if (index < TI_TEXT_NUM)
                pairs[i] = indexTextsPairs[index];
        }
    }
    std::vector<IndexTextsPair> pairs;
};

static IndexedTexts indexedTexts;

std::string emptyText = "";

const std::string& getText(int index)
{
    if (index < 0 || index >= TI_TEXT_NUM)
        return  emptyText;
    else
        return indexedTexts.pairs[index].texts[lang];
}