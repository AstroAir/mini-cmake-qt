#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Embeds a watermark into a host image using DCT.
 * @param host The host image.
 * @param watermark The watermark image.
 * @param alpha The embedding strength (default is 0.1).
 * @return The image with the embedded watermark.
 */
cv::Mat embedWatermark(const cv::Mat &host, const cv::Mat &watermark, double alpha = 0.1);

/**
 * @brief Embeds a watermark into multiple channels of a host image using DCT.
 * @param host The host image.
 * @param watermark The watermark image.
 * @param channelsToEmbed The channels to embed the watermark into.
 * @param alpha The embedding strength (default is 0.1).
 * @return The image with the embedded watermark.
 */
cv::Mat embedWatermarkMultiChannel(const cv::Mat &host, const cv::Mat &watermark,
                                   const std::vector<int> &channelsToEmbed,
                                   double alpha = 0.1);

/**
 * @brief Extracts a watermark from a watermarked image using DCT.
 * @param watermarked The watermarked image.
 * @param alpha The embedding strength used during embedding (default is 0.1).
 * @param wmSize The size of the extracted watermark (default is 64).
 * @return The extracted watermark image.
 */
cv::Mat extractWatermark(const cv::Mat &watermarked, double alpha = 0.1, int wmSize = 64);

/**
 * @brief Estimates the capacity of a host image for watermark embedding.
 * @param host The host image.
 * @param blockSize The size of the blocks used for embedding (default is 8).
 * @return The estimated capacity in number of blocks.
 */
size_t estimateWatermarkCapacity(const cv::Mat &host, int blockSize = 8);

/**
 * @brief Compares two watermarks using normalized cross-correlation.
 * @param wm1 The first watermark image.
 * @param wm2 The second watermark image.
 * @return The similarity score between the two watermarks.
 */
double compareWatermarks(const cv::Mat &wm1, const cv::Mat &wm2);