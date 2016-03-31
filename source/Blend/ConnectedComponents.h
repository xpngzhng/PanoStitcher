#pragma once
#include <opencv2/core/core.hpp>

//! connected components algorithm output formats
enum ConnectedComponentsTypes {
    CC_STAT_LEFT   = 0, //!< The leftmost (x) coordinate which is the inclusive start of the bounding
                        //!< box in the horizontal direction.
    CC_STAT_TOP    = 1, //!< The topmost (y) coordinate which is the inclusive start of the bounding
                        //!< box in the vertical direction.
    CC_STAT_WIDTH  = 2, //!< The horizontal size of the bounding box
    CC_STAT_HEIGHT = 3, //!< The vertical size of the bounding box
    CC_STAT_AREA   = 4, //!< The total area (in pixels) of the connected component
    CC_STAT_MAX    = 5
};

/** @brief computes the connected components labeled image of boolean image

image with 4 or 8 way connectivity - returns N, the total number of labels [0, N-1] where 0
represents the background label. ltype specifies the output label image type, an important
consideration based on the total number of labels or alternatively the total number of pixels in
the source image.

@param image the image to be labeled
@param labels destination labeled image
@param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
@param ltype output image label type. Currently CV_32S and CV_16U are supported.
 */
int connectedComponents(const cv::Mat& image, cv::Mat& labels,
                        int connectivity = 8, int ltype = CV_32S);

/** @overload
@param image the image to be labeled
@param labels destination labeled image
@param stats statistics output for each label, including the background label, see below for
available statistics. Statistics are accessed via stats(label, COLUMN) where COLUMN is one of
cv::ConnectedComponentsTypes
@param centroids floating point centroid (x,y) output for each label, including the background label
@param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
@param ltype output image label type. Currently CV_32S and CV_16U are supported.
*/
int connectedComponentsWithStats(const cv::Mat& image, cv::Mat& labels,
                                 cv::Mat& stats, int connectivity = 8, int ltype = CV_32S);

