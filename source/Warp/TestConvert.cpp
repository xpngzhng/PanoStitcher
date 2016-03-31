#include "Sphere2Cube.h"
#include "AudioVideoProcessor.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

int main1()
{
    cv::Mat sphere = cv::imread("F:\\panoimage\\detuoffice\\blendmultiband.bmp");
    cv::Mat cubic = cv::Mat(600, 3600, CV_8UC3);
    CPSSphere2Cube transform(cubic.rows);
    transform.Convert2Cube(sphere.data, sphere.cols, sphere.rows, sphere.step, cubic.data, cubic.cols, cubic.rows, cubic.step);
    cv::imshow("cubic", cubic);
    cv::imwrite("cubic.bmp", cubic);
    cv::waitKey(0);
    return 0;
}

int main()
{
    avp::AudioVideoReader reader;
    reader.open("E:\\Projects\\PanoVideo\\build\\GeneratePanoVideo\\pano.mp4"/*"E:\\Projects\\PanoVideo\\build\\GeneratePanoVideo\\R0010008_our.mp4"*/, false, true, avp::PixelTypeBGR24);
    int width = reader.getVideoWidth();
    int height = reader.getVideoHeight();
    int cubeHeight = (2 * (double)height / 3.14169265359 + 0.5);
    if (cubeHeight & 1)
        cubeHeight += 1;
    int cubeWidth = 6 * cubeHeight;
    cv::Mat cube(cubeHeight, cubeWidth, CV_8UC3);
    cv::Mat cubeHori = cube(cv::Rect(0, 0, cubeHeight * 4, cubeHeight));
    cv::Mat cubeTop = cube(cv::Rect(cubeHeight * 4, 0, cubeHeight, cubeHeight));
    cv::Mat cubeBot = cube(cv::Rect(cubeHeight * 5, 0, cubeHeight, cubeHeight));
    int compCubeHeight = cubeHeight;
    int compCubeWidth = cubeHeight * 4.5;
    cv::Mat compCube(compCubeHeight, compCubeWidth, CV_8UC3);
    cv::Mat compCubeHori = compCube(cv::Rect(0, 0, cubeHeight * 4, cubeHeight));
    cv::Mat compCubeTop = compCube(cv::Rect(cubeHeight * 4, 0, cubeHeight / 2, cubeHeight / 2));
    cv::Mat compCubeBot = compCube(cv::Rect(cubeHeight * 4, cubeHeight / 2, cubeHeight / 2, cubeHeight / 2));

    CPSSphere2Cube transform(cubeHeight);

    avp::AudioVideoWriter writer;
    writer.open("compcubeout.mp4", "", false, false, "", avp::SampleTypeUnknown, -1, 0, 0,
        true, "", avp::PixelTypeBGR24, compCubeWidth, compCubeHeight, 30, 20000000);

    avp::AudioVideoFrame frame;
    while (reader.read(frame))
    {
        if (frame.mediaType == avp::VIDEO)
        {
            transform.Convert2Cube(frame.data, frame.width, frame.height, frame.step, cube.data, cube.cols, cube.rows, cube.step);
            cubeHori.copyTo(compCubeHori);
            cv::resize(cubeTop, compCubeTop, cv::Size(cubeHeight / 2, cubeHeight / 2));
            cv::resize(cubeBot, compCubeBot, cv::Size(cubeHeight / 2, cubeHeight / 2));
            avp::AudioVideoFrame dst = avp::videoFrame(compCube.data, compCube.step, avp::PixelTypeBGR24, compCube.cols, compCube.rows, frame.timeStamp);
            writer.write(dst);
        }
    }

    reader.close();
    writer.close();
}