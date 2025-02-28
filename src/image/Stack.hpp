#pragma once

#include "Calibration.hpp"
#include "Convolve.hpp"
#include "Denoise.hpp"
#include "Filter.hpp"
#include <concepts>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <span>
#include <vector>

enum class StackMode {
  MEAN,
  MEDIAN,
  MAXIMUM,
  MINIMUM,
  SIGMA_CLIPPING,
  WEIGHTED_MEAN,
  LIGHTEN,
  MODE,
  ENTROPY,
  FOCUS_STACK,
  TRIMMED_MEAN,
  WEIGHTED_MEDIAN,
  ADAPTIVE_FOCUS
};

/**
 * @brief Stack preprocessing configuration
 */
struct StackPreprocessConfig {
  // Calibration parameters
  bool enable_calibration{false};
  CalibrationParams calibration_params;
  cv::Mat response_function;
  cv::Mat flat_field;
  cv::Mat dark_frame;

  // Denoise parameters
  bool enable_denoise{false};
  DenoiseMethod denoise_method{DenoiseMethod::Auto};
  DenoiseParameters denoise_params;

  // Convolution parameters
  bool enable_convolution{false};
  ConvolutionConfig conv_config;

  // Filter parameters
  bool enable_filter{false};
  std::vector<std::unique_ptr<IFilterStrategy>> filters;

  // Parallel processing configuration
  bool parallel_preprocess{true};
  int thread_count{4};
};

// Concept for valid weight containers (C++20)
template <typename T>
concept WeightContainer = requires(T container) {
  { container.size() } -> std::convertible_to<std::size_t>;
  { container[0] } -> std::convertible_to<float>;
};

// Main stacking function with C++20 concepts
template <ImageContainer ImgCont, WeightContainer WCont = std::vector<float>>
auto stackImages(const ImgCont &images, StackMode mode, float sigma = 2.0f,
                 const WCont &weights = WCont{}) -> cv::Mat;

// Layered stacking function with C++20 concepts
template <ImageContainer ImgCont, WeightContainer WCont = std::vector<float>>
auto stackImagesByLayers(const ImgCont &images, StackMode mode,
                         float sigma = 2.0f, const WCont &weights = WCont{})
    -> cv::Mat;

// Add preprocessing stacking function with C++20 concepts
template <ImageContainer ImgCont, WeightContainer WCont = std::vector<float>>
auto stackImagesWithPreprocess(const ImgCont &images, StackMode mode,
                               const StackPreprocessConfig &preprocess_config,
                               float sigma = 2.0f, const WCont &weights = WCont{})
    -> cv::Mat;

// Helper function declarations
auto computeMeanAndStdDev(std::span<const cv::Mat> images) noexcept(false)
    -> std::pair<cv::Mat, cv::Mat>;
auto sigmaClippingStack(std::span<const cv::Mat> images,
                        float sigma) noexcept(false) -> cv::Mat;
auto computeMode(std::span<const cv::Mat> images) noexcept(false) -> cv::Mat;

// Additional function declarations
auto computeEntropy(const cv::Mat &image) noexcept -> double;
auto focusStack(std::span<const cv::Mat> images) noexcept(false) -> cv::Mat;
auto entropyStack(std::span<const cv::Mat> images) noexcept(false) -> cv::Mat;
auto trimmedMeanStack(std::span<const cv::Mat> images,
                      float trimRatio) noexcept(false) -> cv::Mat;
auto adaptiveFocusStack(std::span<const cv::Mat> images) noexcept(false)
    -> cv::Mat;
auto weightedMedianStack(std::span<const cv::Mat> images,
                         std::span<const float> weights) noexcept(false)
    -> cv::Mat;

// Concept-constrained functions to ensure correct usage
template <ImageContainer ImgCont>
auto validateStackInputs(const ImgCont &images, float sigma) {
  if (sigma < 0.0f) {
    throw std::invalid_argument("Sigma value must be non-negative");
  }

  if (images.empty()) {
    throw std::invalid_argument("Input image collection cannot be empty");
  }

  for (size_t i = 0; i < images.size(); ++i) {
    if (images[i].empty()) {
      throw std::invalid_argument("Image " + std::to_string(i) + " is empty");
    }
  }

  return true;
}

// Helper function to convert any image container to span
template <ImageContainer ImgCont>
auto toImagesSpan(const ImgCont &images) -> std::span<const cv::Mat> {
  return std::span<const cv::Mat>(images.data(), images.size());
}

template <WeightContainer WCont>
auto toWeightsSpan(const WCont &weights) -> std::span<const float> {
  if (weights.empty()) {
    return {};
  }
  return std::span<const float>(weights.data(), weights.size());
}
