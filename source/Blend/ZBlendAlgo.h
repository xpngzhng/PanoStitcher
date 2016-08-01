#pragma once

#include "opencv2/core.hpp"
#include <string>
#include <vector>

// If imagePaths.size() should equal maskPaths.size(), 
// read temporary images from imagePaths, read temporary masks from maskPaths.
// If all temporary images and masks share the same width and height,
// set imageSize such width and height, and set images and masks to the
// temporary images and masks, respectively.
// images have 8UC3 type, and masks have 8UC1 type.
// Otherwise, images and masks are set to empty.
void getImagesAndMasks(const std::vector<std::string>& imagePaths, 
    const std::vector<std::string>& maskPaths, cv::Size& imageSize, 
    std::vector<cv::Mat>& images, std::vector<cv::Mat>& masks);

// Shrink and merge images and masks to minimum non zero mask size.
// masks should be of type CV_8UC1.
// images.size() should equal to masks.size(), and
// images[i].size() should equal to masks[i].size().
// Then imageSize is set to the common size of the images and masks.
// Get non zero bounding rect of masks[i],
// images[i] and masks[i] inside this bounding rect are deep copied to
// imageParts[i] and maskParts[i], respectively.
// The non zero bounding rect of masks[i] is tricky.
// This function is used for reprojected images and correponding masks.
// The reprojection model is rectlinear to equirectangle or fishey to equirectangle.
// images[i] is one reprojected equirectagular image.
// Due to the horizontal cirular property of equirectangular reprojection,
// one source image will be split to several parts in the resulting images[i].
// This function tries to merge the split parts together.
// If images[i] and masks[i] are split in the left part and right part,
// left part will be moved to combine with the right part.
// And the resulting rects[i] will not totally locates inside
// cv::Rect(0, 0, imageSize.width, imageSize.height).
void getParts(const std::vector<cv::Mat>& images, 
    const std::vector<cv::Mat>& masks, cv::Size& imageSize, 
    std::vector<cv::Mat>& imageParts, std::vector<cv::Mat>& maskParts,
    std::vector<cv::Rect>& rects);

// Combine the above two functions into sigle one.
void getParts(const std::vector<std::string>& contentPaths, 
    const std::vector<std::string>& maskPaths, cv::Size& imageSize, 
    std::vector<cv::Mat>& imageParts, std::vector<cv::Mat>& maskParts,
    std::vector<cv::Rect>& rects);

// Checke whether images are non empty and all entries in images share the same type.
bool checkType(const std::vector<cv::Mat>& images, int type);

// Check whether images are non empty and all entries in images share the same size.
bool checkSize(const std::vector<cv::Mat>& images);

// Check whether images and masks are both non empty and images.size() equals masks.size(),
// and all images[i]s and masks[i]s share the same size.
// Satisfaction of the above requirements yields true return, otherwise false.
bool checkSize(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks);

// Check whether images, masks and rects are all non empty and 
// images.size(), masks.size() and rects.size() are the same,
// and images[i].size(), masks[i].size() and rects[i].size() are the same,
// and rects[i] are totally inside blendSize.
// Satisfaction of the above requirements yields true return, otherwise false.
bool checkSize(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    const std::vector<cv::Rect>& rects, const cv::Size& blendSize);

// Examine if masks is not empty and all cv::Mat inside masks share the same size.
// Satisfaction of the above requirements will make this function return true,
// and masks.size() is set to numMasks, and the size of each cv::Mat in masks is set to rows and cols,
// otherwise this function will return false.
bool probeMasks(const std::vector<cv::Mat>& masks, int& numMasks, int& rows, int& cols);

// Simple implementation of finding non zero bounding rect of mask,
// using horizontal scanning and vertical scanning, cv::findContours or 
// other connected component analysis algorithms are not used.
// mask should be of type CV_8UC1.
cv::Rect getNonZeroBoundingRect(const cv::Mat& mask);

// mask1 and mask2 should be the same width and height and be of type CV_8UC1.
// The intersect non zero area is set to 255 in intersectMask, the type of which is also CV_8UC1.
// The non zero bounding rect of intersectMask is set to intersectRect.
void getIntersect(const cv::Mat& mask1, const cv::Mat& mask2, 
    cv::Mat& intersectMask, cv::Rect& intersectRect);

// Use distance transform to split the intersect region of mask1 and mask2.
// After calling this function, mask1 & mask2 is empty set,
// and mask1 | mask2 remains the same.
// mask1, mask2 and intersect should be the same size and type CV_8UC1.
// intersect should be the true intersection of mask1 and mask2,
// otherwise you will not get correct result.
// mask1 and mask2 will change after calling this function.
// If you want to keep the original mask1 and mask2,
// make deep copies of them before calling this function.
void separateMask(cv::Mat& mask1, cv::Mat& mask2, const cv::Mat& intersect);

// Use distance transform to split the entire region to two parts.
// mask1 and mask2 should be the same size and type CV_8UC1.
// mask1 - (mask1 & mask2) will be inside region1, mask2 - (mask1 & mask2) will be in region2.
// mask1 & mask2 and ~(mask1 | mask2) will be split using distance transform
// and assigned to region1 and region2.
// The result is that region1 & region2 is empty set,
// and region1 | region2 equals the whole image.
void splitRegion(const cv::Mat& mask1, const cv::Mat& mask2, cv::Mat& region1, cv::Mat& region2);

// Use distance transform to split the entire region to two parts.
// This function is similar to the above one except that the intersection of mask1 and mask2,
// i.e., mask1 & mask2, should be provided in intersect.
void splitRegion(const cv::Mat& mask1, const cv::Mat& mask2, const cv::Mat& intersect, cv::Mat& region1, cv::Mat& region2);

// Use distance transform to fill the area that is not belong to either mask1 or mask2.
// mask1, mask2, intersect, fill1 and fill2 should be the same size and type CV_8UC1.
// Notice that memories of fill1 and fill2 should be allocated before calling this function,
// and intersect should be the true intersection of mask1 and mask2.
// ~(mask1 | mask2) will be splite into two parts according to distance transform,
// and respectively assigned to fill1 and fill2.
void fillExclusiveRegion(const cv::Mat& mask1, const cv::Mat& mask2, 
    const cv::Mat& intersect, cv::Mat& fill1, cv::Mat& fill2);

inline bool contains(const cv::Rect& base, const cv::Rect& test)
{
    return (base.area() && test.area() && (base & test) == test);
}

inline cv::Rect padRect(const cv::Rect& rect, int pad)
{
    return cv::Rect(rect.x - pad, rect.y - pad, rect.width + 2 * pad, rect.height + 2 * pad);
}

// Copy part or whole of src to dst, according to srcRect and dstRect, if they intersect.
// src and dst should be the same type.
// srcRect and dstRect should be in the same coordinate system.
// src.size() should equal to srcRect.size(), the same applies to dst and dstRect.
void copyIfIntersect(const cv::Mat& src, cv::Mat& dst, 
    const cv::Rect& srcRect, const cv::Rect& dstRect);

// Put image top left corner in the origin, 
// fill the part of image whose x < xBegin or x >= xBegin + period
// using the part of image whose x >= xBegin and x < xBegin + period,
// in a horizontally periodical fashion.
void horiCircularRepeat(cv::Mat& image, int xBegin, int period);

// Put src's top left corner in the origin, assume dst locates in area defined by dstRect,
// copy src to cv::Rect(0, 0, src.cols, src.rows) part of dst,
// use this part of dst to horizontally periodically fill the rest part of dst.
// It is not necessary to allocate memory for dst before calling this function.
void horiCircularExpand(const cv::Mat& src, const cv::Rect& dstRect, cv::Mat& dst);

// Put dst's top left corner in the origin, 
// dst's memory should be allocated before calling this function.
// Assume src locates in area defined by srcRect.
// If currRect = cv::Rect(k * dst.cols, 0, dst.cols, dst.height) & srcRect is not empty,
// copy currRect part of src to dst.
// The copy begins from small k and increases to large k.
// Older copied part will be replaced by newer copied part.
void horiCircularFold(const cv::Mat& src, const cv::Rect& srcRect, cv::Mat& dst);

// Put dst's top left corner in the origin, 
// dst's memory should be allocated before calling this function.
// Assume src locates in area defined by srcRect.
// If currRect = cv::Rect(k * dst.cols, 0, dst.cols, dst.height) & srcRect is not empty,
// copy currRect part of src with non zero value in srcMask to dst.
// The copy begins from small k and increases to large k.
// Older copied part will be replaced by newer copied part.
void horiCircularFold(const cv::Mat& src, const cv::Mat& srcMask, const cv::Rect& srcRect, cv::Mat& dst);

// Split src periodically and push them into dst.
// If currRect = src & cv::Rect(period * k, srcRect.y, period, srcRect.height) is not empty,
// push currRect to dst. Before the push begins, dst.clear() is called.
void horiCircularFold(const cv::Rect& src, int period, std::vector<cv::Rect>& dst);

// Find the widest zero vertical band in mask, and set the left and right bound
// to zeroBegInc and zeroEndExc, and return true.
// If all columns of mask contains non zero pixels, return false.
bool horiSplit(const cv::Mat& mask, int* zeroBegInc, int* zeroEndExc);

// Use graphcut algorithm to find a seam that splits the intersection of image1 and image2.
// mask1 and mask2 respectively label the area in image1 and image2 that should be considered in graphcut.
// image1 and image2 should be type CV_8UC3, and mask1 and mask2 should be type CV_8UC1, 
// and the four images should share the same size.
// The graphcut pixel weight is computed by the difference of image1 and image2.
// After calling this function, mask1 and mask2 change, 
// mask1 & mask2 becomes empty set, while mask1 | mask2 remains the same.
// IMPORTANT NOTICE: To make graphcut work, mask1 and mask2 should contain some zero boundaries,
// If horiWrap is true, graphcut will work as if the left most column and the right most column
// were joined together.
void findSeam(const cv::Mat& image1, const cv::Mat& image2,
    cv::Mat& mask1, cv::Mat& mask2, bool horiWrap);

// This variant of findSeam have a different pixel weight computation method for graphcut.
// Besides the difference of image1 and image2, 
// distance to the middle line of mask1 & mask2 is also taken into account.
// Before calling this function, call splitRegion(mask1, mask2, region1, region2) to get region1 and region2.
// region1 and region2 should share the same size with image1, image2, mask1 and mask2,
// and should be type CV_8UC1.
void findSeamByRegionSplitLine(const cv::Mat& image1, const cv::Mat& image2, 
    const cv::Mat& region1, const cv::Mat& region2, cv::Mat& mask1, cv::Mat& mask2, bool horiWrap);

// This variant of findSeam only use pixels marked non zero in roi 
// to build graph for graphcut and find seam.
// Usually roi can be set to mask1 & mask2, or a little larger than this intersection.
// Set roi to mask1 & mask2 is OK, since this function internally dilate roi by several pixels.
// The basic version of findSeam use all the pixel in the whole size to build graph.
// If mask1 & mask2 is much smaller than the whole image size,
// calling this function will save you a lot of time.
void findSeamInROI(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& roi,
    cv::Mat& mask1, cv::Mat& mask2, bool horiWrap);

// This variant of findSeam first resize image1, image2, mask1 and mask2
// to a smaller size and find seam at that scale, then transform the seam back
// to the original scale. The width and height of resized images and masks 
// equals to width / scale and height / scale. 
// The transformed seam will look rather coarse in the original scale.
// If refine is true, then findSeamInROI is called internally,
// and new seam will be found around the neighborhood of the coarse seam.
void findSeamScaleDown(const cv::Mat& image1, const cv::Mat& image2,
    cv::Mat& mask1, cv::Mat& mask2, bool horiWrap, int scale, bool refine);

// This variant of findSeam is used when middle part of the whole size has only zero mask,
// and the remaining left part and right part should be joined together to build graph.
// The consecutive zero mask columns are defined by zeroBegInc and zeroEndExc.
void findSeamLeftRightWrap(const cv::Mat& image1, const cv::Mat& image2,
    cv::Mat& mask1, cv::Mat& mask2, int zeroBegInc, int zeroEndExc);

// Use graphcut to split images whose effective regions are defined by masks,
// results are stored in resultMasks.
// images.size() should equal to masks.size(),
// images[i].type() should be CV_8UC3, and masks[i].type() should be CV_8UC1,
// all entries in images and masks should share the same cv::Mat::size() value.
// For any different i and j, resultMasks[i] & resultMasks[j] is empty set.
// Unions of all entries in resultMasks is the same as that of masks.
// This function internally calls the two-image findSeam variants.
// The rest of the input parameters controls which variant to call.
// pad means how much pixel wide are padded before calling a findSeam function.
// If scale is not 1, findSeamScaleDown will be called, 
// and refine controls whether the coarse seam will be refined at the original scale.
// If scale is 1, and cv::countNonZero(masks[i] & masks[j]) is more than 
// ratio multiplies the number of pixles in the bounding rect of the masks[i] & masks[j],
// the basic findSeam is called, otherwise findSeamInROI is called.
void findSeams(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks, 
    std::vector<cv::Mat>& resultMasks, int pad, int scale, double ratio, bool refine);

// Generate non-intersecting masks.
// masks should be non empty, masks.size() should not be more than 255,
// and all masks[i] should be type CV_8UC1, and share the same size.
// The resulting uniqueMasks have the folling two properties:
// First, for any different i and j, uniqueMasks[i] & uniqueMasks[j] is empty set.
// Second, unions of unqieMasks[i] equals unions of masks[i].
// Maybe the name of this function should change for a better one in the future.
void getUniqueMasks(const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& uniqueMasks);

// Generate non-intersecting masks.
// masks should be non empty, and all masks[i] should be type CV_8UC1, and share the same size.
// This function does not impose masks.size() limit.
// The resulting uniqueMasks have the folling two properties:
// First, for any different i and j, uniqueMasks[i] & uniqueMasks[j] is empty set.
// Second, unions of unqieMasks[i] equals unions of masks[i].
void getNonIntersectingMasks(const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& nonIntersectingMasks);

// According to image width and height, given maxLevels and minLength, 
// determine how many levels (the lowest, original resolution level not included) can be 
// for a image pyramid.
int getTrueNumLevels(int width, int height, int maxLevels, int minLength);

// Blend two images using multiband blend.
// image1 and image2 should be the same size, and type CV_8UC3.
// alpha1 and alpha2 act as the alpha channels for image1 and image2, respectively.
// alpha1 and alpha2 should be type CV_8UC1.
// Pixels having zero value in alpha1 should also be zero in image1, 
// the same applies to alpha2 and image2.
// mask1 and mask2 tells which pixel should come from image1 and which image2.
// mask1 and mask2 should be type CV_8UC1, 
// mask1 & mask2 should be empty set, and mask1 | mask2 should cover the whole image size.
// If horiWrap is true, left most columns and right most colums will joined together
// when buiding pyramids.
// maxLevels defines the most number of levels except the original scale level.
// minLength defines the smallest side length of images in the pyramid.
// The blended image is output to result.
void multibandBlend(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& alpha1, const cv::Mat& alpha2,
    cv::Mat& mask1, const cv::Mat& mask2, bool horiWrap, int maxLevels, int minLength, cv::Mat& result);

// This variant of blend also implements multiband blend algorithm,
// but mask1 & mask2 can be non empty set, and mask1 | mask2 does not need to equal to whole size.
void multibandBlendAnyMask(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& alpha1, const cv::Mat& alpha2,
    cv::Mat& mask1, const cv::Mat& mask2, bool horiWrap, int maxLevels, int minLength, cv::Mat& result);

// This is a multi-image variant of the above multibandBlend, parameter requirements are similar.
// In this variant, for any different i and j, masks[i] & masks[j] should be empty set,
// and the union of all the entries in masks should equal to the whole image size.
void multibandBlend(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& alphas,
    const std::vector<cv::Mat>& masks, int maxLevels, int minLength, cv::Mat& result);

// This is a multi-image variant of the above multibandBlendAnyMask, parameter requirements are similar.
// This variant does not impose requirements on the masks.
void multibandBlendAnyMask(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& alphas,
    const std::vector<cv::Mat>& masks, int maxLevels, int minLength, cv::Mat& result);

// Get weights for linear blend given several masks having intersections
// The function first apply linear transform to masks to get non-intersecting masks,
// then apply Gaussian blur with radius to non-intersecting masks.
// For every position in the blurred non-intersecting mask, the value is set to zero 
// if the value in the corresponding position of the original mask is zero.
// The resulting weights have integral type CV_32SC1
void getWeightsLinearBlend(const std::vector<cv::Mat>& masks, int radius, std::vector<cv::Mat>& weights);

// This is a variant of getWeightsLinearBlend except that the weight have floating point type CV_32FC1
void getWeightsLinearBlend32F(const std::vector<cv::Mat>& masks, int radius, std::vector<cv::Mat>& weights);

// Given masks, uniqueMasks(non intersecting version of masks) and dists(distance transforms of masks),
// try to find out how many pixels(n) wide we can expand uniqueMasks[i] so that the expanded mask 
// does not expand beyond the corresponding masks[i].
// We check the value n from distBound towards 0.
// It seams that checking value n from 1 to the maximum would be better, 
// then distBound is not needed.
int getMaxRadius(const std::vector<cv::Mat>& masks, const std::vector<cv::Mat>& uniqueMasks,
    const std::vector<cv::Mat>& dists, int distBound);

// Given masks, we compute the non intersecting version of masks and expand them by a certain number of 
// pixels which is determined by calling getMaxRadius with distBound set to radius.
void getExtendedMasks(const std::vector<cv::Mat>& masks, int radius, std::vector<cv::Mat>& extendedMasks);

// This is an overloaded getWeightsLinearBlend without assigning radius.
// This function internally computes the maximum valid radius and then compute weights.
void getWeightsLinearBlendBoundedRadius(const std::vector<cv::Mat>& masks, int maxRadius, std::vector<cv::Mat>& weights);

// This is a variant of getWeightsLinearBlend except that the weight have floating point type CV_32FC1
void getWeightsLinearBlendBoundedRadius32F(const std::vector<cv::Mat>& masks, int maxRadius, std::vector<cv::Mat>& weights);

// Blend two images using linear blend.
// image1 and image2 should be the same size, and type CV_8UC3.
// alpha1 and alpha2 act as the alpha channels for image1 and image2, respectively.
// alpha1 and alpha2 should be type CV_8UC1.
// Pixels having zero value in alpha1 should also be zero in image1, 
// the same applies to alpha2 and image2.
// mask1 and mask2 tells which pixel should come from image1 and which image2.
// mask1 and mask2 should be type CV_8UC1.
// radius controls how wide the transition area is.
void linearBlend(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& alpha1, const cv::Mat& alpha2,
    cv::Mat& mask1, const cv::Mat& mask2, int radius, cv::Mat& result);

// This is a multi-image variant of the above linearBlend, parameter requirements are similar.
void linearBlend(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& alphas,
    const std::vector<cv::Mat>& masks, int radius, cv::Mat& result);

// Get look up table for transform given slope k.
// If k is around 1, we obtain an approximately identity transform.
// If k > 1, we use three points (0, 0), (255 / k, 255) and (255, 255) to calculate an Bezier curve
// in place of the piece wise linear transform.
// If k < 1, we also use three points to calculate an Bezier curve.
// If we use three points (0, 0), (255 , k * 255) and (255, 255), the increase of the input around 255
// may cause significant increase of output, resulting in bad visual quality of the transformed image.
// Besides (0, 0) and (255, 255), the third point for transform is the intersection of lines
// y = k * x and y = (1 / k) * (x - 255) + 255.
void getLUT(std::vector<unsigned char>& lut, double k);

// Get intersecting masks around seams generated by distance transforms
void getIntsctMasksAroundDistTransSeams(const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& outMasks);

// Exposure/gain correction using gray images, the error minimization expression is the same as the one
// presented in the paper of panorama stitching algorithm using SIFT features except that we replace the mean approximation
// by real gray value pairs
void getTransformsGrayPairWiseSiftPanoPaper(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    std::vector<double>& kt);

// Exposure/gain correction using gray images, the error minimization expression considers not only 
// (k_i * I_i_m - k_j * I_j_n)^2, (k_i - 1)^2 and (k_j - 1)^2, where i and j are image indexes, m and n
// are the indexes in image i and j and refer to the same global position,
// as in the paper of panorama stitching algorithm using SIFT features,
// but also (k_i * I_i_m - I_j_n)^2 + (k_j * I_j_n - I_i_m)^2.
void getTransformsGrayPairWiseMutualError(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    std::vector<double>& kt);

// Exposure correction using color images, not gray images, similar to getTransformsGrayPairWiseSiftPanoPaper.
void getTransformsBGRPairWiseSiftPanoPaper(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    std::vector<std::vector<double> >& kts);

// Exposure correction using color images, not gray images, similar to getTransformsGrayPairWiseMutualError.
void getTransformsBGRPairWiseMutualError(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    std::vector<std::vector<double> >& kts);

// White balance correction using color images.
void getTintTransformsPairWiseMimicSiftPanoPaper(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks,
    std::vector<double>& rgRatioGains, std::vector<double>& bgRatioGains);

void adjust(const cv::Mat& src, cv::Mat& dst, const std::vector<unsigned char>& lut);

void adjust(const cv::Mat& src, cv::Mat& dst, const std::vector<std::vector<unsigned char> >& luts);

void calcHist(const cv::Mat& image, std::vector<int>& hist);

void calcHist(const cv::Mat& image, const cv::Mat& mask, std::vector<int>& hist);

int countNonZeroHistBins(const std::vector<int>& hist);