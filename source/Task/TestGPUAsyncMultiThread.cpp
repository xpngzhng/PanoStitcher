#include "ZBlend.h"
#include "ZReproject.h"
#include "AudioVideoProcessor.h"
#include "Timer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/gpumat.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>

int numVideos;
std::vector<avp::AudioVideoReader> readers;
std::vector<cv::gpu::GpuMat> dstMasksGpu;
std::vector<cv::gpu::GpuMat> xmapsGpu, ymapsGpu;
CudaTilingMultibandBlendFast blender;
//CudaTilingLinearBlend blender;
std::vector<cv::gpu::Stream> streams;
std::vector<cv::gpu::CudaMem> pinnedMems;
std::vector<cv::gpu::GpuMat> imagesGpu, reprojImagesGpu;
cv::gpu::GpuMat blendImageGpu;
cv::Mat blendImageCpu;
ztool::Timer timerAll, timerTotal, timerDecode, timerUpload, timerReproject, timerBlend, timerDownload, timerEncode;
int frameCount;
int maxFrameCount = 600;
int actualWriteFrame = 0;
avp::AudioVideoWriter2 writer;
bool success;

int decodeCount = 0;
int procCount = 0;
int encodeCount = 0;

std::mutex mtxDecodedImages;
std::condition_variable cvDecodedImagesForWrite, cvDecodedImagesForRead;
bool decodedImagesOwnedByDecodeThread;
bool videoEnd = false;

std::mutex mtxEncodedImage;
std::condition_variable cvEncodedImageForWrite, cvEncodedImageForRead;
bool encodedImageOwnedByProcThread;
bool procEnd = false;

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

void decodeThread()
{
#if ENABLE_CALC_TIME
    ztool::Timer timer;
#endif
    while (true)
    {
#if ENABLE_CALC_TIME
        timer.start();
#endif
        {            
            std::unique_lock<std::mutex> ul(mtxDecodedImages);
            cvDecodedImagesForWrite.wait(ul, []{return decodedImagesOwnedByDecodeThread;});
            bool successRead = true;
            //timerDecode.start();
            for (int i = 0; i < numVideos; i++)
            {
                avp::AudioVideoFrame frame;
                if (!readers[i].read(frame))
                {
                    successRead = false;
                    break;
                }
                cv::Mat src(frame.height, frame.width, CV_8UC4, frame.data, frame.step);
                cv::Mat dst(pinnedMems[i]);
                src.copyTo(dst);
            }
            //timerDecode.end();
            
            if (!successRead || decodeCount >= maxFrameCount)
            {
                videoEnd = true;
                break;
            }

            // NOTICE!!!!!!
            // The following line had better be after break.
            // If not, before the lock at the end of the function, if proc() arrives at the first lock and wait,
            // it may find decodedImagesOwnedByDecodeThread == false, but procEnd == false, 
            // then the last frames will be reprojected and blends twice.
            // This file has not produces such result yet, but CPUPanoramaLocalDiskTask has. 
            decodedImagesOwnedByDecodeThread = false;
        }
        cvDecodedImagesForRead.notify_one();
        decodeCount++;
        //printf("decode count = %d\n", decodeCount);

#if ENABLE_CALC_TIME
        timer.end();
        printf("d %f %f\n", timerDecode.elapse(), timer.elapse());
#endif
    }

    {
        std::unique_lock<std::mutex> ul(mtxDecodedImages);
        cvDecodedImagesForWrite.wait(ul, []{return decodedImagesOwnedByDecodeThread; });
        decodedImagesOwnedByDecodeThread = false;
        videoEnd = true;
    }
    cvDecodedImagesForRead.notify_one();

    printf("decode thread end, %d\n", decodeCount);
}

void gpuProcThread()
{
    while (true)
    {
        {
            std::unique_lock<std::mutex> ul(mtxDecodedImages);
            cvDecodedImagesForRead.wait(ul, []{return !decodedImagesOwnedByDecodeThread;});
            if (videoEnd)
                break;

            //timerReproject.start();
            for (int i = 0; i < numVideos; i++)
                streams[i].enqueueUpload(pinnedMems[i], imagesGpu[i]);
            for (int i = 0; i < numVideos; i++)
                cudaReprojectTo16S(imagesGpu[i], reprojImagesGpu[i], xmapsGpu[i], ymapsGpu[i], streams[i]);
            for (int i = 0; i < numVideos; i++)
                streams[i].waitForCompletion();
            //timerReproject.end();

            decodedImagesOwnedByDecodeThread = true;
        }
        cvDecodedImagesForWrite.notify_one();

        blender.blend(reprojImagesGpu, blendImageGpu);
        cv::gpu::Stream::Null().waitForCompletion();

        {
            std::unique_lock<std::mutex> ul(mtxEncodedImage);
            cvEncodedImageForWrite.wait(ul, []{return encodedImageOwnedByProcThread;});

            blendImageGpu.download(blendImageCpu);
            encodedImageOwnedByProcThread = false;
        }
        cvEncodedImageForRead.notify_one();
        procCount++;
        //printf("proc count = %d\n", procCount);
    }

    {
        std::unique_lock<std::mutex> ul(mtxDecodedImages);
        cvDecodedImagesForRead.wait(ul, []{return !decodedImagesOwnedByDecodeThread; });
        decodedImagesOwnedByDecodeThread = true;
    }
    cvDecodedImagesForWrite.notify_one();

    // Without the following if statement, encodeThread will not terminate.
    {
        std::unique_lock<std::mutex> ul(mtxEncodedImage);
        cvEncodedImageForWrite.wait(ul, []{return encodedImageOwnedByProcThread; });
        encodedImageOwnedByProcThread = false;
        procEnd = true;
    }
    cvEncodedImageForRead.notify_one();

    printf("gpu proc thread end, %d\n", procCount);
}

void encodeThread()
{
    while (true)
    {
        {
            std::unique_lock<std::mutex> ul(mtxEncodedImage);
            cvEncodedImageForRead.wait(ul, []{return !encodedImageOwnedByProcThread;});
            if (procEnd)
                break;

            timerEncode.start();
            avp::AudioVideoFrame frame = avp::videoFrame(blendImageCpu.data, blendImageCpu.step, avp::PixelTypeBGR32, blendImageCpu.cols, blendImageCpu.rows, -1LL);
            writer.write(frame);
            timerEncode.end();
            
            encodeCount++;
            //printf("%d\n", encodeCount);
            //printf("%d, %d, %d\n", decodeCount, procCount, encodeCount);
            printf("frame %d finish, encode time = %f\n", encodeCount, timerEncode.elapse());

            encodedImageOwnedByProcThread = true;
        }
        cvEncodedImageForWrite.notify_one();
    }
    printf("encode thread end, %d\n", encodeCount);
}

int main(int argc, char* argv[])
{
    const char* keys =
        "{a | camera_param_file |  | camera param file path}"
        "{b | video_path_offset_file |  | video path and offset file path}"
        "{c | num_frames_skip | 100 | number of frames to skip}"
        "{d | pano_width | 2048 | pano picture width}"
        "{e | pano_height | 1024 | pano picture height}"
        "{h | pano_video_name | panoh264qsv_4k.mp4 | xml param file path}"
        "{g | pano_video_num_frames | 1000 | number of frames to write}";

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
    std::string::size_type pos = cameraParamFile.find_last_of(".");
    std::string ext = cameraParamFile.substr(pos + 1);

    std::vector<PhotoParam> params;
    if (ext == "pts")
        loadPhotoParamFromPTS(cameraParamFile, params);
    else
        loadPhotoParamFromXML(cameraParamFile, params);

    dstSize.width = parser.get<int>("pano_width");
    dstSize.height = parser.get<int>("pano_height");

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
    numVideos = srcVideoNames.size();

    numSkip = parser.get<int>("num_frames_skip");
    if (numSkip < 0)
        numSkip = 0;

    printf("Open videos and set to the correct frames\n");

    bool ok = false;
    readers.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
    {
        ok = readers[i].open(srcVideoNames[i], false, true, avp::PixelTypeBGR32/*avp::PixelTypeBGR24*/);
        if (!ok)
            break;
        int count = offset[i] + numSkip;
        avp::AudioVideoFrame frame;
        for (int j = 0; j < count; j++)
            readers[i].read(frame);
    }
    if (!ok)
    {
        printf("Could not open video file(s)\n");
        return 0;
    }

    printf("Open videos done\n");
    printf("Prepare for reproject and blend\n");

    srcSize.width = readers[0].getVideoWidth();
    srcSize.height = readers[0].getVideoHeight();

    std::vector<cv::Mat> dstMasks;
    std::vector<cv::Mat> dstSrcMaps, xmaps, ymaps;
    getReprojectMapsAndMasks(params, srcSize, dstSize, dstSrcMaps, dstMasks);

    dstMasksGpu.resize(numVideos);
    xmapsGpu.resize(numVideos);
    ymapsGpu.resize(numVideos);
    cv::Mat map32F;
    cv::Mat map64F[2];
    for (int i = 0; i < numVideos; i++)
    {
        dstMasksGpu[i].upload(dstMasks[i]);
        cv::split(dstSrcMaps[i], map64F);
        map64F[0].convertTo(map32F, CV_32F);
        xmapsGpu[i].upload(map32F);
        map64F[1].convertTo(map32F, CV_32F);
        ymapsGpu[i].upload(map32F);
    }
    dstSrcMaps.clear();
    map32F.release();
    map64F[0].release();
    map64F[1].release();

    success = blender.prepare(dstMasks, 16, 2);
    //success = blender.prepare(dstMasks, 50);
    if (!success)
    {
        printf("Blender prepare failed, exit.\n");
        return 0;
    }
    dstMasks.clear();
    //return 0;

    streams.resize(numVideos);
    pinnedMems.resize(numVideos);
    for (int i = 0; i < numVideos; i++)
        pinnedMems[i].create(srcSize, CV_8UC4);

    imagesGpu.resize(numVideos);
    reprojImagesGpu.resize(numVideos);

    printf("Prepare finish, begin stitching.\n");

    panoVideoName = parser.get<std::string>("pano_video_name");
    std::vector<avp::Option> options;
    options.push_back(std::make_pair("preset", "medium"));
    //success = writer.open(panoVideoName, avp::PixelTypeBGR32, dstSize.width, dstSize.height, 24, 12000000, avp::EncodeSpeedSlow);
    success = writer.open(panoVideoName, "", false, false, "", avp::SampleTypeUnknown, 0, 0, 0,
        true, "h264", avp::PixelTypeBGR32, dstSize.width, dstSize.height, readers[0].getVideoFps(), 48000000, options);
    if (!success)
    {
        printf("Video writer open failed, exit.\n");
        return 0;
    }

    maxFrameCount = parser.get<int>("pano_video_num_frames");

    timerAll.start();

    frameCount = 0;   
    decodedImagesOwnedByDecodeThread = true;
    encodedImageOwnedByProcThread = true;
    std::thread thrdDecode(decodeThread);
    std::thread thrdGpuProc(gpuProcThread);
    std::thread thrdEncode(encodeThread);

    // IMPORTANT!!! Forget to join may cause the program to abort. 
    // I do not know why now!!
    thrdDecode.join();
    thrdGpuProc.join();
    thrdEncode.join();

    for (int i = 0; i < numVideos; i++)
        readers[i].close();
    writer.close();
    
    timerAll.end();
    printf("finish, %d frames written\n", actualWriteFrame);
    printf("time elapsed total = %f seconds, average process time per frame = %f second\n", 
        timerAll.elapse(), timerAll.elapse() / (actualWriteFrame ? actualWriteFrame : 1));

    dstMasksGpu.clear();
    xmapsGpu.clear();
    ymapsGpu.clear();
    pinnedMems.clear();
    imagesGpu.clear();
    reprojImagesGpu.clear();
    blendImageGpu.release();
    blendImageCpu.release();
    return 0;
}