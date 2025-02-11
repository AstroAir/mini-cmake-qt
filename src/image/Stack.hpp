#pragma once

#include "Calibration.hpp"
#include "Convolve.hpp"
#include "Denoise.hpp"
#include "Filter.hpp"
#include <opencv2/core/mat.hpp>
#include <vector>

enum StackMode {
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
 * @brief 堆叠前置处理配置
 */
struct StackPreprocessConfig {
  // 校准参数
  bool enable_calibration{false};
  CalibrationParams calibration_params;
  cv::Mat response_function;
  cv::Mat flat_field;
  cv::Mat dark_frame;

  // 降噪参数
  bool enable_denoise{false};
  DenoiseMethod denoise_method{DenoiseMethod::Auto};
  DenoiseParameters denoise_params;

  // 卷积参数
  bool enable_convolution{false};
  ConvolutionConfig conv_config;

  // 滤波参数
  bool enable_filter{false};
  std::vector<std::unique_ptr<IFilterStrategy>> filters;

  // 并行处理配置
  bool parallel_preprocess{true};
  int thread_count{4};
};

// 主要堆叠函数
auto stackImages(const std::vector<cv::Mat> &images, StackMode mode,
                 float sigma = 2.0f, const std::vector<float> &weights = {})
    -> cv::Mat;

// 按图层堆叠函数
auto stackImagesByLayers(const std::vector<cv::Mat> &images, StackMode mode,
                         float sigma = 2.0f,
                         const std::vector<float> &weights = {}) -> cv::Mat;

// 增加带前置处理的堆叠函数声明
auto stackImagesWithPreprocess(const std::vector<cv::Mat> &images,
                               StackMode mode,
                               const StackPreprocessConfig &preprocess_config,
                               float sigma = 2.0f,
                               const std::vector<float> &weights = {})
    -> cv::Mat;

// 辅助函数声明
auto computeMeanAndStdDev(const std::vector<cv::Mat> &images)
    -> std::pair<cv::Mat, cv::Mat>;
auto sigmaClippingStack(const std::vector<cv::Mat> &images, float sigma)
    -> cv::Mat;
auto computeMode(const std::vector<cv::Mat> &images) -> cv::Mat;

// 新增函数声明
auto computeEntropy(const cv::Mat &image) -> double;
auto focusStack(const std::vector<cv::Mat> &images) -> cv::Mat;
auto entropyStack(const std::vector<cv::Mat> &images) -> cv::Mat;
auto trimmedMeanStack(const std::vector<cv::Mat> &images, float trimRatio)
    -> cv::Mat;

auto adaptiveFocusStack(const std::vector<cv::Mat> &images) -> cv::Mat;
auto weightedMedianStack(const std::vector<cv::Mat> &images,
  const std::vector<float> &weights) -> cv::Mat;