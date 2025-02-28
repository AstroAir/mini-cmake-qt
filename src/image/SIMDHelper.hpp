#pragma once
#include <immintrin.h>
#include <opencv2/core.hpp>
#include <span>
#include <type_traits>

// Concept to constrain numeric types that can be used with SIMD operations
template <typename T>
concept SIMDCompatible = std::is_arithmetic_v<T> &&
                         (std::is_same_v<T, float> || std::is_same_v<T, int> ||
                          std::is_same_v<T, double>);

class SIMDHelper {
public:
  /**
   * @brief Process an image tile with convolution using SIMD instructions if
   * available
   * @param input Input image
   * @param output Output image
   * @param kernel Convolution kernel
   * @param roi Region of interest to process
   * @throws std::invalid_argument If input parameters are invalid
   */
  static void processImageTile(const cv::Mat &input, cv::Mat &output,
                               const cv::Mat &kernel, const cv::Rect &roi);

  // Check if SIMD extensions are supported
  static bool isSSESupported() { return hasSSE; }
  static bool isAVXSupported() { return hasAVX; }

private:
  template <SIMDCompatible T>
  static void convolve2DSSE(std::span<const T> input, std::span<T> output,
                            std::span<const float> kernel, int rows, int cols,
                            int kernelSize, int stride);

  template <SIMDCompatible T>
  static void convolve2DAVX(std::span<const T> input, std::span<T> output,
                            std::span<const float> kernel, int rows, int cols,
                            int kernelSize, int stride);

  // Validate input parameters for convolution
  static bool validateConvolutionParams(const cv::Mat &input,
                                        const cv::Mat &output,
                                        const cv::Mat &kernel,
                                        const cv::Rect &roi);

  static bool hasSSE;
  static bool hasAVX;
  static void checkCPUSupport();
};
