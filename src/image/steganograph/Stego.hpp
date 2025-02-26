#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

// Enumeration for steganography methods
enum class StegoMethod {
  DFT, // Discrete Fourier Transform
  DCT, // Discrete Cosine Transform
  DWT, // Discrete Wavelet Transform
  LSB  // Least Significant Bit
};

// Configuration for steganography
struct StegoConfig {
  StegoMethod method = StegoMethod::DFT;
  double alpha = 0.1;                  // Embedding strength
  int min_radius_ratio = 8;            // Minimum radius ratio
  int max_radius_ratio = 4;            // Maximum radius ratio
  bool use_cuda = false;               // Use CUDA for acceleration
  bool use_parallel = true;            // Use parallel processing
  int block_size = 32;                 // Block size for processing
  std::optional<std::string> password; // Encryption password
};

/**
 * @brief Converts a string message to a vector of bits.
 * @param message The input string message.
 * @return A vector of bits representing the message.
 */
std::vector<bool> str_to_bits(const std::string &message);

/**
 * @brief Converts a vector of bits to a string message.
 * @param bits The input vector of bits.
 * @return The decoded string message.
 */
std::string bits_to_str(const std::vector<bool> &bits);

/**
 * @brief Embeds a message into an image using Fourier Transform steganography.
 * @param carrier The input carrier image.
 * @param message The message to embed.
 * @param alpha The embedding strength (default is 0.1).
 * @return The image with the embedded message.
 */
cv::Mat embed_message(cv::Mat carrier, const std::string &message,
                      double alpha = 0.1);

/**
 * @brief Extracts a hidden message from an image using Fourier Transform
 * steganography.
 * @param stego The input stego image.
 * @param msg_length The length of the hidden message.
 * @param alpha The embedding strength used during embedding (default is 0.1).
 * @return The extracted message.
 */
std::string extract_message(cv::Mat stego, int msg_length, double alpha = 0.1);

/**
 * @brief Embeds a message into an image using the specified steganography
 * configuration.
 * @param carrier The input carrier image.
 * @param message The message to embed.
 * @param config The steganography configuration.
 * @return The image with the embedded message.
 */
cv::Mat embed_message(cv::Mat carrier, const std::string &message,
                      const StegoConfig &config);

/**
 * @brief Extracts a hidden message from an image using the specified
 * steganography configuration.
 * @param stego The input stego image.
 * @param msg_length The length of the hidden message.
 * @param config The steganography configuration.
 * @return The extracted message.
 */
std::string extract_message(cv::Mat stego, int msg_length,
                            const StegoConfig &config);

/**
 * @brief Estimates the compression ratio of an image based on the steganography
 * configuration.
 * @param image The input image.
 * @param config The steganography configuration.
 * @return The estimated compression ratio.
 */
double estimate_compression_ratio(const cv::Mat &image,
                                  const StegoConfig &config);

/**
 * @brief Evaluates the quality of the stego image compared to the original
 * image.
 * @param original The original image.
 * @param stego The stego image.
 * @return The quality score of the stego image.
 */
double evaluate_image_quality(const cv::Mat &original, const cv::Mat &stego);

/**
 * @brief Tests the robustness of the stego image by attempting to extract the
 * embedded message.
 * @param stego The stego image.
 * @param message The embedded message.
 * @param config The steganography configuration.
 * @return True if the message can be successfully extracted, false otherwise.
 */
bool test_robustness(const cv::Mat &stego, const std::string &message,
                     const StegoConfig &config);

/**
 * @brief Computes the Mean Structural Similarity Index (MSSIM) between two
 * images.
 * @param i1 The first image.
 * @param i2 The second image.
 * @return The MSSIM value as a cv::Scalar.
 */
cv::Scalar MSSIM(const cv::Mat &i1, const cv::Mat &i2);

/**
 * @brief Calculates similarity score between two images using multiple metrics
 * @param img1 First input image
 * @param img2 Second input image
 * @param metrics Optional vector to store individual metric scores
 * @return Overall similarity score between 0.0 and 1.0
 */
double calculateSimilarity(const cv::Mat &img1, const cv::Mat &img2, 
                         std::vector<double> *metrics = nullptr);