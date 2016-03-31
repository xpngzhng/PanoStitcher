#include <opencv2/core/core.hpp>

int main1(int argc, char** argv)
{
    const char* keys =
        "{a | src_video_width | 1920 | camera picture width}"
        "{b | src_video_height | 1080 | camera picture height}"
        "{c | src_frames_per_second | 30 | camera frame rate}"
        "{d | dst_video_width | 2048 | stream picture width}"
        "{e | dst_video_height | 1024 | stream picture height}"
        "{f | dst_bits_per_second | 1000000 | stream bits per second}"
        "{g | dst_url | rtmp://pili-publish.live.detu.com/detulive/detudemov550?key=detukey | live stream address}";
    cv::CommandLineParser parser(argc, argv, keys);
    parser.printParams();
    int srcWidth = parser.get<int>("src_video_width");
    int srcHeight = parser.get<int>("src_video_height");
    std::string url = parser.get<std::string>("dst_url");

    return 0;
}