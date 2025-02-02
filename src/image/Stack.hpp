#pragma once

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

// 主要堆叠函数
auto stackImages(const std::vector<cv::Mat> &images, StackMode mode,
                 float sigma = 2.0f, const std::vector<float> &weights = {})
    -> cv::Mat;

// 按图层堆叠函数
auto stackImagesByLayers(const std::vector<cv::Mat> &images, StackMode mode,
                         float sigma = 2.0f,
                         const std::vector<float> &weights = {}) -> cv::Mat;

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
