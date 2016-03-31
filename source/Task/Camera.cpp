#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main1()
{
    cv::VideoCapture cap;
    cv::Mat frame;
    bool ok = cap.open(0);
    if (!ok)
    {
        printf("failed to open camera\n");
        return 0;
    }
    printf("open camera success\n");
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
    while (true)
    {
        ok = cap.read(frame);
        if (ok)
        {
            cv::imshow("frame", frame);
            cv::waitKey(30);
        }
        else
            printf("failed to read\n");
    }
    return 0;
}

int main2()
{
    cv::VideoCapture cap;
    cap.open("F:\\panovideo\\ricoh\\R0010113.MP4");
    int numImages = 0;
    char buf[256];
    cv::Mat frame;
    while (true)
    {
        if (cap.read(frame))
        {
            cv::imshow("frame", frame);
            int key = cv::waitKey(0);
            if (key == 's')
            {
                sprintf(buf, "F:\\panovideo\\ricoh\\R0010113\\image%d.bmp", ++numImages);
                cv::imwrite(buf, frame);
            }
            else if (key == 'q')
                break;
        }
    }
    return 0;
}

int main3()
{
    //cv::Rect srcRectLeft(10, 17, 922, 922);
    //cv::Rect srcRectRight(25, 19, 912, 912);
    //for (int i = 1; i < 8; i++)
    //{
    //    char buf[256];
    //    sprintf(buf, "F:\\panovideo\\ricoh\\image%d.bmp", i);
    //    cv::Mat image = cv::imread(buf);
    //    int rows = image.rows, cols = image.cols;
    //    cv::Mat left = image(cv::Rect(0, 0, cols / 2, rows))(srcRectLeft);
    //    cv::Mat right = image(cv::Rect(cols / 2, 0, cols / 2, rows))(srcRectRight);
    //    sprintf(buf, "F:\\panovideo\\ricoh\\image%dleftcrop.bmp", i);
    //    cv::imwrite(buf, left);
    //    sprintf(buf, "F:\\panovideo\\ricoh\\image%drightcrop.bmp", i);
    //    cv::imwrite(buf, right);
    //}
    //return 0;
    for (int i = 1; i < 6; i++)
    {
        char buf[256];
        sprintf(buf, "F:\\panovideo\\ricoh\\R0010113\\image%d.bmp", i);
        cv::Mat image = cv::imread(buf);
        int rows = image.rows, cols = image.cols;
        cv::Mat left = image(cv::Rect(0, 0, cols / 2, rows));
        cv::Mat right = image(cv::Rect(cols / 2, 0, cols / 2, rows));
        sprintf(buf, "F:\\panovideo\\ricoh\\R0010113\\image%dleft.bmp", i);
        cv::imwrite(buf, left);
        sprintf(buf, "F:\\panovideo\\ricoh\\R0010113\\image%dright.bmp", i);
        cv::imwrite(buf, right);
    }
    return 0;
    cv::Mat image = cv::imread("F:\\panovideo\\ricoh\\image3.bmp");
    int rows = image.rows, cols = image.cols;
    cv::Mat left = image(cv::Rect(0, 0, cols / 2, rows));
    cv::Mat right = image(cv::Rect(cols / 2, 0, cols / 2, rows));
    cv::Mat rightFlip;
    cv::flip(right, rightFlip, 1);
    cv::Mat diff = left - rightFlip;
    cv::imshow("image", image);
    cv::imshow("diff", diff);
    cv::waitKey(0);
    return 0;
}

int main()
{
    char* path[] = { "F:\\QQRecord\\2710916451\\FileRecv\\1.mp4",
        "F:\\QQRecord\\2710916451\\FileRecv\\2.mp4",
        "F:\\QQRecord\\2710916451\\FileRecv\\3.mp4",
        "F:\\QQRecord\\2710916451\\FileRecv\\4.mp4" };
    cv::VideoCapture cap;
    for (int i = 0; i < 4; i++)
    {
        cap.open(path[i]);
        cv::Mat img;
        cap.read(img);
        char buf[64];
        sprintf(buf, "image%d.bmp", i);
        cv::imwrite(buf, img);
    }
    return 0;
}