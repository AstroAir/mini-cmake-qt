#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

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