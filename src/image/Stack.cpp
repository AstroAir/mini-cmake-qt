#include "Stack.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>


// Compute the mean and standard deviation of images
auto computeMeanAndStdDev(const std::vector<cv::Mat> &images)
    -> std::pair<cv::Mat, cv::Mat> {
  if (images.empty()) {
    spdlog::error(
        "Input images are empty when computing mean and standard deviation.");
    throw std::runtime_error("Input images are empty");
  }

  spdlog::info(
      "Starting to compute mean and standard deviation. Number of images: {}",
      images.size());

  // Initialize mean and standard deviation matrices
  cv::Mat mean = cv::Mat::zeros(images[0].size(), CV_32F);
  cv::Mat accumSquare = cv::Mat::zeros(images[0].size(), CV_32F);

  // Accumulate pixel values
  for (const auto &img : images) {
    if (img.size() != mean.size() || img.type() != mean.type()) {
      spdlog::error("All images must have the same size and type.");
      throw std::runtime_error("Image size or type mismatch");
    }

    cv::Mat floatImg;
    img.convertTo(floatImg, CV_32F);
    mean += floatImg;
    accumSquare += floatImg.mul(floatImg);
  }

  // Compute mean
  mean /= static_cast<float>(images.size());

  // Compute standard deviation
  cv::Mat stdDev;
  cv::sqrt(accumSquare / static_cast<float>(images.size()) - mean.mul(mean),
           stdDev);

  spdlog::info("Mean and standard deviation computation completed.");

  return {mean, stdDev};
}

// Sigma clipping stack
auto sigmaClippingStack(const std::vector<cv::Mat> &images, float sigma)
    -> cv::Mat {
  if (images.empty()) {
    spdlog::error("No input images for sigma clipping stack.");
    throw std::runtime_error("No images to stack");
  }

  spdlog::info("Starting sigma clipping stack. Sigma value: {:.2f}", sigma);

  cv::Mat mean, stdDev;
  try {
    std::tie(mean, stdDev) = computeMeanAndStdDev(images);
  } catch (const std::exception &e) {
    spdlog::error("Failed to compute mean and standard deviation: {}",
                  e.what());
    throw;
  }

  std::vector<cv::Mat> layers;
  for (size_t i = 0; i < images.size(); ++i) {
    cv::Mat temp;
    images[i].convertTo(temp, CV_32F);
    cv::Mat mask = cv::abs(temp - mean) < (sigma * stdDev);
    temp.setTo(0, ~mask);
    layers.push_back(temp);
    spdlog::info("Processed image {}, applied sigma clipping mask.", i + 1);
  }

  cv::Mat sum = cv::Mat::zeros(images[0].size(), CV_32F);
  cv::Mat count = cv::Mat::zeros(images[0].size(), CV_32F);

  for (size_t i = 0; i < layers.size(); ++i) {
    cv::Mat mask = layers[i] != 0;
    sum += layers[i];
    count += mask;
    spdlog::info("Accumulated layer {}.", i + 1);
  }

  // Prevent division by zero
  cv::Mat nonZeroMask = count > 0;
  cv::Mat result = cv::Mat::zeros(images[0].size(), CV_32F);
  sum.copyTo(result, nonZeroMask);
  result /= count;

  // Convert result back to 8-bit image
  result.convertTo(result, CV_8U);

  spdlog::info("Sigma clipping stack completed.");

  return result;
}

// Compute the mode (most frequent value) of each pixel
auto computeMode(const std::vector<cv::Mat> &images) -> cv::Mat {
  if (images.empty()) {
    spdlog::error("Input images are empty when computing mode.");
    throw std::runtime_error("Input images are empty");
  }

  spdlog::info("Starting to compute image mode. Number of images: {}",
               images.size());

  cv::Mat modeImage = cv::Mat::zeros(images[0].size(), images[0].type());

  for (int row = 0; row < images[0].rows; ++row) {
    for (int col = 0; col < images[0].cols; ++col) {
      std::unordered_map<uchar, int> frequency;
      for (const auto &img : images) {
        uchar pixel = img.at<uchar>(row, col);
        frequency[pixel]++;
      }

      // Find the most frequent pixel value
      int maxFreq = 0;
      uchar modePixel = 0;
      for (const auto &[pixel, freq] : frequency) {
        if (freq > maxFreq) {
          maxFreq = freq;
          modePixel = pixel;
        }
      }

      modeImage.at<uchar>(row, col) = modePixel;
    }
  }

  spdlog::info("Image mode computation completed.");

  return modeImage;
}

// 计算图像熵
auto computeEntropy(const cv::Mat &image) -> double {
  std::vector<int> histogram(256, 0);
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      histogram[image.at<uchar>(i, j)]++;
    }
  }

  double entropy = 0.0;
  int totalPixels = image.rows * image.cols;
  for (int i = 0; i < 256; i++) {
    if (histogram[i] > 0) {
      double probability = static_cast<double>(histogram[i]) / totalPixels;
      entropy -= probability * std::log2(probability);
    }
  }
  return entropy;
}

// 基于熵的堆叠
auto entropyStack(const std::vector<cv::Mat> &images) -> cv::Mat {
  if (images.empty()) {
    throw std::runtime_error("No images to stack");
  }

  cv::Mat result = cv::Mat::zeros(images[0].size(), CV_8U);
  std::vector<cv::Mat> entropies;

  // 计算每个图像的局部熵
  for (const auto &img : images) {
    cv::Mat entropy = cv::Mat::zeros(img.size(), CV_32F);
    int windowSize = 9;
    int offset = windowSize / 2;

    for (int i = offset; i < img.rows - offset; i++) {
      for (int j = offset; j < img.cols - offset; j++) {
        cv::Mat window = img(cv::Range(i - offset, i + offset + 1),
                             cv::Range(j - offset, j + offset + 1));
        entropy.at<float>(i, j) = static_cast<float>(computeEntropy(window));
      }
    }
    entropies.push_back(entropy);
  }

  // 根据最大熵选择像素
  for (int i = 0; i < result.rows; i++) {
    for (int j = 0; j < result.cols; j++) {
      float maxEntropy = -1;
      int bestIndex = 0;
      for (size_t k = 0; k < images.size(); k++) {
        if (entropies[k].at<float>(i, j) > maxEntropy) {
          maxEntropy = entropies[k].at<float>(i, j);
          bestIndex = static_cast<int>(k);
        }
      }
      result.at<uchar>(i, j) = images[bestIndex].at<uchar>(i, j);
    }
  }

  return result;
}

// 焦点堆叠
auto focusStack(const std::vector<cv::Mat> &images) -> cv::Mat {
  if (images.empty()) {
    throw std::runtime_error("No images to stack");
  }

  cv::Mat result = cv::Mat::zeros(images[0].size(), CV_8U);
  std::vector<cv::Mat> laplacians;

  // 计算每个图像的Laplacian
  for (const auto &img : images) {
    cv::Mat laplacian;
    cv::Laplacian(img, laplacian, CV_32F);
    laplacian = cv::abs(laplacian);
    laplacians.push_back(laplacian);
  }

  // 根据Laplacian响应选择最清晰的像素
  for (int i = 0; i < result.rows; i++) {
    for (int j = 0; j < result.cols; j++) {
      float maxResponse = -1;
      int bestIndex = 0;
      for (size_t k = 0; k < images.size(); k++) {
        if (laplacians[k].at<float>(i, j) > maxResponse) {
          maxResponse = laplacians[k].at<float>(i, j);
          bestIndex = static_cast<int>(k);
        }
      }
      result.at<uchar>(i, j) = images[bestIndex].at<uchar>(i, j);
    }
  }

  // 应用高斯模糊以减少噪声
  cv::GaussianBlur(result, result, cv::Size(3, 3), 0);
  return result;
}

auto stackImages(const std::vector<cv::Mat> &images, StackMode mode,
                 float sigma, const std::vector<float> &weights) -> cv::Mat;

// Stack images by layers
auto stackImagesByLayers(const std::vector<cv::Mat> &images, StackMode mode,
                         float sigma, const std::vector<float> &weights)
    -> cv::Mat {
  if (images.empty()) {
    spdlog::error("No input images for stacking.");
    throw std::runtime_error("No images to stack");
  }

  spdlog::info("Starting image stacking by layers. Mode: {}",
               static_cast<int>(mode));

  std::vector<cv::Mat> channels;
  cv::split(images[0], channels);

  for (size_t i = 1; i < images.size(); ++i) {
    std::vector<cv::Mat> tempChannels;
    cv::split(images[i], tempChannels);
    for (size_t j = 0; j < channels.size(); ++j) {
      channels[j].push_back(tempChannels[j]);
    }
  }

  std::vector<cv::Mat> stackedChannels;
  for (auto &channel : channels) {
    stackedChannels.push_back(stackImages(channel, mode, sigma, weights));
  }

  cv::Mat stackedImage;
  cv::merge(stackedChannels, stackedImage);

  spdlog::info("Image stacking by layers completed.");

  return stackedImage;
}

// 新增：截断平均堆叠算法（提高算法精度和鲁棒性，利用OpenMP优化性能）
auto trimmedMeanStack(const std::vector<cv::Mat> &images, float trimRatio)
    -> cv::Mat {
  if (images.empty()) {
    throw std::runtime_error("No images to stack");
  }
  cv::Mat result = cv::Mat::zeros(images[0].size(), CV_32F);
  int totalImages = static_cast<int>(images.size());
  int trimCount = static_cast<int>(totalImages * trimRatio / 2);
#pragma omp parallel for
  for (int i = 0; i < images[0].rows; i++) {
    for (int j = 0; j < images[0].cols; j++) {
      std::vector<float> pixelValues;
      pixelValues.reserve(totalImages);
      for (const auto &img : images) {
        pixelValues.push_back(static_cast<float>(img.at<uchar>(i, j)));
      }
      std::sort(pixelValues.begin(), pixelValues.end());
      float sum = 0;
      int count = 0;
      for (int k = trimCount; k < totalImages - trimCount; k++) {
        sum += pixelValues[k];
        count++;
      }
      result.at<float>(i, j) = sum / count;
    }
  }
  result.convertTo(result, CV_8U);
  return result;
}

// 新增：加权中值堆叠（Weighted Median Stack）
auto weightedMedianStack(const std::vector<cv::Mat> &images,
                         const std::vector<float> &weights) -> cv::Mat {
  if (images.empty()) {
    spdlog::error("No input images for weighted median stack.");
    throw std::runtime_error("No images to stack");
  }
  if (weights.size() != images.size()) {
    spdlog::error("Number of weights does not match number of images for "
                  "weighted median stack.");
    throw std::runtime_error("Weights size mismatch");
  }

  spdlog::info("Starting weighted median stack for {} images.", images.size());
  cv::Mat result = cv::Mat::zeros(images[0].size(), images[0].type());
  int rows = images[0].rows, cols = images[0].cols;
  int n = static_cast<int>(images.size());

#pragma omp parallel for collapse(2)
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::vector<std::pair<uchar, float>> pixelWeights;
      pixelWeights.reserve(n);
      float totalWeight = 0.0f;
      for (int k = 0; k < n; k++) {
        uchar pixel = images[k].at<uchar>(i, j);
        float w = weights[k];
        pixelWeights.push_back({pixel, w});
        totalWeight += w;
      }
      std::sort(pixelWeights.begin(), pixelWeights.end(),
                [](auto a, auto b) { return a.first < b.first; });
      float halfWeight = totalWeight / 2.0f;
      float cumWeight = 0.0f;
      uchar medianPixel = 0;
      for (auto &pw : pixelWeights) {
        cumWeight += pw.second;
        if (cumWeight >= halfWeight) {
          medianPixel = pw.first;
          break;
        }
      }
      result.at<uchar>(i, j) = medianPixel;
    }
  }
  spdlog::info("Weighted median stack completed.");
  return result;
}

// 新增：自适应焦点堆叠（Adaptive Focus Stack）
// 利用每幅图像的 Laplacian 作为锐度权重计算加权平均
auto adaptiveFocusStack(const std::vector<cv::Mat> &images) -> cv::Mat {
  if (images.empty()) {
    spdlog::error("No input images for adaptive focus stack.");
    throw std::runtime_error("No images to stack");
  }

  spdlog::info("Starting adaptive focus stack for {} images.", images.size());
  int rows = images[0].rows, cols = images[0].cols;
  cv::Mat result = cv::Mat::zeros(images[0].size(), CV_32F);
  cv::Mat weightSum = cv::Mat::zeros(images[0].size(), CV_32F);
  int n = static_cast<int>(images.size());

  std::vector<cv::Mat> laplacianMats(n);
  for (int k = 0; k < n; k++) {
    cv::Laplacian(images[k], laplacianMats[k], CV_32F);
    laplacianMats[k] = cv::abs(laplacianMats[k]);
  }

#pragma omp parallel for collapse(2)
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      float weightedPixel = 0.0f;
      float totalWeight = 0.0f;
      for (int k = 0; k < n; k++) {
        float weight = laplacianMats[k].at<float>(i, j);
        float pixelVal = static_cast<float>(images[k].at<uchar>(i, j));
        weightedPixel += pixelVal * weight;
        totalWeight += weight;
      }
      if (totalWeight > 0)
        result.at<float>(i, j) = weightedPixel / totalWeight;
      else
        result.at<float>(i, j) = static_cast<float>(images[0].at<uchar>(i, j));
    }
  }
  result.convertTo(result, CV_8U);
  spdlog::info("Adaptive focus stack completed.");
  return result;
}

// Image stacking function
auto stackImages(const std::vector<cv::Mat> &images, StackMode mode,
                 float sigma, const std::vector<float> &weights) -> cv::Mat {
  if (images.empty()) {
    spdlog::error("No input images for stacking.");
    throw std::runtime_error("No images to stack");
  }

  spdlog::info("Starting image stacking. Mode: {}", static_cast<int>(mode));

  cv::Mat stackedImage;

  try {
    switch (mode) {
    case MEAN: {
      spdlog::info("Selected stacking mode: Mean stack (MEAN)");
      cv::Mat stdDev; // Declare stdDev variable
      std::tie(stackedImage, stdDev) = computeMeanAndStdDev(images);
      stackedImage.convertTo(stackedImage, CV_8U);
      break;
    }
    case MEDIAN: {
      spdlog::info("Selected stacking mode: Median stack (MEDIAN)");
      std::vector<cv::Mat> sortedImages;
      for (const auto &img : images) {
        cv::Mat floatImg;
        img.convertTo(floatImg, CV_32F);
        sortedImages.push_back(floatImg);
      }

      // Stack all images into a 4D matrix
      cv::Mat stacked4D;
      cv::merge(sortedImages, stacked4D);

      // Compute median
      cv::Mat medianImg = cv::Mat::zeros(images[0].size(), CV_32F);
      for (int row = 0; row < medianImg.rows; ++row) {
        for (int col = 0; col < medianImg.cols; ++col) {
          std::vector<float> pixelValues;
          pixelValues.reserve(sortedImages.size());
          for (const auto &sortedImg : sortedImages) {
            pixelValues.push_back(sortedImg.at<float>(row, col));
          }
          std::nth_element(pixelValues.begin(),
                           pixelValues.begin() + pixelValues.size() / 2,
                           pixelValues.end());
          medianImg.at<float>(row, col) = pixelValues[pixelValues.size() / 2];
        }
      }
      medianImg.convertTo(stackedImage, CV_8U);
      break;
    }
    case MAXIMUM: {
      spdlog::info("Selected stacking mode: Maximum stack (MAXIMUM)");
      stackedImage = images[0].clone();
      for (size_t i = 1; i < images.size(); ++i) {
        cv::max(stackedImage, images[i], stackedImage);
        spdlog::info("Applied maximum stack: Image {}", i + 1);
      }
      break;
    }
    case MINIMUM: {
      spdlog::info("Selected stacking mode: Minimum stack (MINIMUM)");
      stackedImage = images[0].clone();
      for (size_t i = 1; i < images.size(); ++i) {
        cv::min(stackedImage, images[i], stackedImage);
        spdlog::info("Applied minimum stack: Image {}", i + 1);
      }
      break;
    }
    case SIGMA_CLIPPING: {
      spdlog::info(
          "Selected stacking mode: Sigma clipping stack (SIGMA_CLIPPING)");
      stackedImage = sigmaClippingStack(images, sigma);
      break;
    }
    case WEIGHTED_MEAN: {
      spdlog::info(
          "Selected stacking mode: Weighted mean stack (WEIGHTED_MEAN)");
      if (weights.empty()) {
        spdlog::error("Weight vector is empty for weighted mean stack.");
        throw std::runtime_error("Weight vector cannot be empty");
      }
      if (weights.size() != images.size()) {
        spdlog::error("Number of weights does not match number of images.");
        throw std::runtime_error(
            "Number of weights does not match number of images");
      }

      cv::Mat weightedSum = cv::Mat::zeros(images[0].size(), CV_32F);
      float totalWeight = 0.0F;
      for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat floatImg;
        images[i].convertTo(floatImg, CV_32F);
        weightedSum += floatImg * weights[i];
        totalWeight += weights[i];
        spdlog::info("Applied weight {}: {:.2f}", i + 1, weights[i]);
      }

      weightedSum /= totalWeight;
      weightedSum.convertTo(stackedImage, CV_8U);
      break;
    }
    case LIGHTEN: {
      spdlog::info("Selected stacking mode: Lighten stack (LIGHTEN)");
      stackedImage = images[0].clone();
      for (size_t i = 1; i < images.size(); ++i) {
        cv::Mat mask = images[i] > stackedImage;
        images[i].copyTo(stackedImage, mask);
        spdlog::info("Applied lighten stack: Image {}", i + 1);
      }
      break;
    }
    case MODE: {
      spdlog::info("Selected stacking mode: Mode stack (MODE)");
      stackedImage = computeMode(images);
      break;
    }
    case ENTROPY: {
      spdlog::info("Selected stacking mode: Entropy stack (ENTROPY)");
      stackedImage = entropyStack(images);
      break;
    }
    case FOCUS_STACK: {
      spdlog::info("Selected stacking mode: Focus stack (FOCUS_STACK)");
      stackedImage = focusStack(images);
      break;
    }
    case TRIMMED_MEAN: {
      spdlog::info("Selected stacking mode: Trimmed mean stack (TRIMMED_MEAN)");
      float trimRatio = 0.2f; // 可调整的修剪比例，根据需要更改数值
      stackedImage = trimmedMeanStack(images, trimRatio);
      break;
    }
    case WEIGHTED_MEDIAN: {
      spdlog::info(
          "Selected stacking mode: Weighted median stack (WEIGHTED_MEDIAN)");
      if (weights.empty()) {
        spdlog::error("Weight vector is empty for weighted median stack.");
        throw std::runtime_error("Weight vector cannot be empty");
      }
      if (weights.size() != images.size()) {
        spdlog::error("Number of weights does not match number of images for "
                      "weighted median stack.");
        throw std::runtime_error("Number of weights does not match");
      }
      stackedImage = weightedMedianStack(images, weights);
      break;
    }
    case ADAPTIVE_FOCUS: {
      spdlog::info(
          "Selected stacking mode: Adaptive focus stack (ADAPTIVE_FOCUS)");
      stackedImage = adaptiveFocusStack(images);
      break;
    }
    default: {
      spdlog::error("Unknown stacking mode: {}", static_cast<int>(mode));
      throw std::invalid_argument("Unknown stacking mode");
    }
    }

    spdlog::info("Image stacking completed.");
  } catch (const std::exception &e) {
    spdlog::error("Exception occurred during image stacking: {}", e.what());
    throw;
  }

  return stackedImage;
}