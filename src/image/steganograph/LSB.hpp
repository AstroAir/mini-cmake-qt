#pragma once

#include <opencv2/opencv.hpp>
#include <span>
#include <string>
#include <vector>

/**
 * @brief Class for buffering a string message as a bit stream.
 */
class BitStreamBuffer {
public:
  /**
   * @brief Constructs a BitStreamBuffer from a string message.
   * @param message The input string message.
   * @param add_terminator Whether to add a terminator (default is true).
   */
  explicit BitStreamBuffer(const std::string &message,
                           bool add_terminator = true);

  /**
   * @brief Gets the buffered bits.
   * @return A reference to the vector of bits.
   */
  const std::vector<bool> &getBits() const;

  /**
   * @brief Gets the size of the buffered bits.
   * @return The size of the buffered bits.
   */
  size_t size() const;

private:
  std::vector<bool> bits; ///< The buffered bits
};

/**
 * @brief Embeds a message into an image using the LSB method.
 * @param image The input image.
 * @param message The message to embed.
 */
void embedLSB(cv::Mat &image, const std::string &message);

/**
 * @brief Extracts a hidden message from an image using the LSB method.
 * @param image The input image.
 * @return The extracted message.
 */
std::string extractLSB(const cv::Mat &image);

/**
 * @brief Gets the bit plane of an image at a specified bit position.
 * @param image The input image.
 * @param bitPosition The bit position (0-7).
 * @return The bit plane image.
 */
cv::Mat getBitPlane(const cv::Mat &image, int bitPosition);

/**
 * @brief Class for streaming LSB encoding.
 */
class StreamingLSBEncoder {
public:
  /**
   * @brief Constructs a StreamingLSBEncoder for a given image.
   * @param image The input image.
   */
  explicit StreamingLSBEncoder(cv::Mat &image);

  /**
   * @brief Embeds a chunk of data into the image.
   * @param data The chunk of data to embed.
   * @return True if the chunk was successfully embedded, false otherwise.
   */
  bool embedChunk(std::span<const char> data);

private:
  cv::Mat &image_;     ///< The image to embed data into
  size_t current_pos_; ///< The current position in the image
};