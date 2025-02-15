#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Extracts the Most Significant Bit (MSB) plane from an image.
 * @param image The input image.
 * @return The MSB plane image.
 */
cv::Mat extractMSBPlane(const cv::Mat &image);

/**
 * @brief Modifies the Most Significant Bit (MSB) of an image.
 * @param image The input image to modify.
 * @param setToOne If true, sets the MSB to 1; otherwise, sets it to 0.
 */
void modifyMSB(cv::Mat &image, bool setToOne);

/**
 * @brief Class for compressing images by keeping only the most significant
 * bits.
 */
class MSBCompressor {
public:
  /**
   * @brief Compresses an image by keeping only the specified number of most
   * significant bits.
   * @param image The input image.
   * @param keepBits The number of most significant bits to keep (default is 4).
   * @return The compressed image.
   */
  static cv::Mat compress(const cv::Mat &image, int keepBits = 4);

  /**
   * @brief Compresses a batch of images by keeping only the specified number of
   * most significant bits.
   * @param images The input vector of images.
   * @param keepBits The number of most significant bits to keep (default is 4).
   * @return A vector of compressed images.
   */
  static std::vector<cv::Mat> compressBatch(const std::vector<cv::Mat> &images,
                                            int keepBits = 4);
};
