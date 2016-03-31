#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

//! smooths and downsamples the image
void pyramidDownTo32S(const cv::Mat& src, cv::Mat& dst, const cv::Size& dstsize = cv::Size(), 
    int horiBorderType = cv::BORDER_DEFAULT, int vertBorderType = cv::BORDER_DEFAULT);

//! smooths and downsamples the image
void pyramidDown(const cv::Mat& src, cv::Mat& dst, const cv::Size& dstsize = cv::Size(),
    int horiBorderType = cv::BORDER_DEFAULT, int vertBorderType = cv::BORDER_DEFAULT);

//! upsamples and smoothes the image
void pyramidUp(const cv::Mat& src, cv::Mat& dst, const cv::Size& dstsize = cv::Size(), 
    int horiBorderType = cv::BORDER_DEFAULT, int vertBorderType = cv::BORDER_DEFAULT);

//! smooths and downsamples the image
void pyramidDownTo32S(const cv::Mat& src, cv::Mat& dst, void* aux1, void* aux2, const cv::Size& dstsize = cv::Size(),
    int horiBorderType = cv::BORDER_DEFAULT, int vertBorderType = cv::BORDER_DEFAULT);

//! smooths and downsamples the image
void pyramidDown(const cv::Mat& src, cv::Mat& dst, void* aux1, void* aux2, const cv::Size& dstsize = cv::Size(),
    int horiBorderType = cv::BORDER_DEFAULT, int vertBorderType = cv::BORDER_DEFAULT);

//! upsamples and smoothes the image
void pyramidUp(const cv::Mat& src, cv::Mat& dst, void* aux1, void* aux2, const cv::Size& dstsize = cv::Size(),
    int horiBorderType = cv::BORDER_DEFAULT, int vertBorderType = cv::BORDER_DEFAULT);

//! smooths and downsamples the image
void pyramidDownTo32S(const cv::Mat& src, cv::Mat& dst, 
    std::vector<unsigned char>& aux1, std::vector<unsigned char>& aux2, const cv::Size& dstsize = cv::Size(),
    int horiBorderType = cv::BORDER_DEFAULT, int vertBorderType = cv::BORDER_DEFAULT);

//! smooths and downsamples the image
void pyramidDown(const cv::Mat& src, cv::Mat& dst, 
    std::vector<unsigned char>& aux1, std::vector<unsigned char>& aux2, const cv::Size& dstsize = cv::Size(),
    int horiBorderType = cv::BORDER_DEFAULT, int vertBorderType = cv::BORDER_DEFAULT);

//! upsamples and smoothes the image
void pyramidUp(const cv::Mat& src, cv::Mat& dst, 
    std::vector<unsigned char>& aux1, std::vector<unsigned char>& aux2, const cv::Size& dstsize = cv::Size(),
    int horiBorderType = cv::BORDER_DEFAULT, int vertBorderType = cv::BORDER_DEFAULT);