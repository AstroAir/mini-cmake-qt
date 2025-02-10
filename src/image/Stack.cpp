#include "Stack.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
std::shared_ptr<spdlog::logger> stackLogger =
    spdlog::basic_logger_mt("StackLogger", "logs/stack.log");

template <typename T>
void transfer_unique_ptrs(std::vector<std::unique_ptr<T>> &dest,
                          std::vector<std::unique_ptr<T>> &src,
                          size_t start_idx = 0,
                          size_t count = std::numeric_limits<size_t>::max()) {
  if (&dest == &src) {
    throw std::invalid_argument(
        "Destination and source vectors cannot be the same");
  }

  if (src.empty()) {
    throw std::out_of_range("Source vector is empty");
  }

  // 计算实际要移动的数量
  const size_t available = src.size() - start_idx;
  const size_t actual_count = (count == std::numeric_limits<size_t>::max())
                                  ? available
                                  : std::min(count, available);

  if (start_idx >= src.size()) {
    throw std::out_of_range("Start index exceeds source vector size");
  }

  if (actual_count == 0) {
    throw std::logic_error("No elements to transfer");
  }

  auto start = src.begin() + start_idx;
  auto end = start + actual_count;

  dest.insert(dest.end(), std::make_move_iterator(start),
              std::make_move_iterator(end));

  std::for_each(start, end, [](auto &ptr) { ptr.release(); });

  src.erase(start, end);
}

// 辅助函数:执行图像预处理流水线
cv::Mat preprocessImage(const cv::Mat &input,
                        const StackPreprocessConfig &config) {
  cv::Mat processed = input.clone();

  try {
    // 1. 校准处理
    if (config.enable_calibration) {
      auto [calibrated, _, __, ___] = flux_calibration_ex(
          processed, config.calibration_params,
          config.response_function.empty() ? nullptr
                                           : &config.response_function,
          config.flat_field.empty() ? nullptr : &config.flat_field,
          config.dark_frame.empty() ? nullptr : &config.dark_frame);
      processed = calibrated;
      stackLogger->debug("Applied calibration preprocessing");
    }

    // 2. 降噪处理
    if (config.enable_denoise) {
      ImageDenoiser denoiser;
      processed = denoiser.denoise(processed, config.denoise_params);
      stackLogger->debug("Applied denoise preprocessing");
    }

    // 3. 卷积处理
    if (config.enable_convolution) {
      processed = Convolve::process(processed, config.conv_config);
      stackLogger->debug("Applied convolution preprocessing");
    }

    // 4. 滤波处理
    if (config.enable_filter && !config.filters.empty()) {
      std::vector<std::unique_ptr<IFilterStrategy>> strategies;
      // TODO: Uncomment the following line after implementing
      // transfer_unique_ptrs transfer_unique_ptrs<IFilterStrategy>(strategies,
      // config.filters);
      ChainImageFilterProcessor filter_processor(std::move(strategies));
      QImage qimg = ImageUtils::matToQtImage(processed);
      qimg = filter_processor.process(qimg);
      processed = ImageUtils::qtImageToMat(qimg);
      stackLogger->debug("Applied filter chain preprocessing");
    }

    stackLogger->info("Processed image size: {} x {}", processed.cols,
                      processed.rows);

    return processed;
  } catch (const std::exception &e) {
    stackLogger->error("Preprocessing failed: {}", e.what());
    throw;
  }
}
} // namespace

// Compute the mean and standard deviation of images
// 优化 computeMeanAndStdDev 函数，使用 OpenMP 并行计算
auto computeMeanAndStdDev(const std::vector<cv::Mat> &images)
    -> std::pair<cv::Mat, cv::Mat> {
  if (images.empty()) {
    stackLogger->error("Input images are empty.");
    throw std::runtime_error("Input images are empty");
  }

  cv::Mat mean = cv::Mat::zeros(images[0].size(), CV_32F);
  cv::Mat accumSquare = cv::Mat::zeros(images[0].size(), CV_32F);
  const int numImages = static_cast<int>(images.size());

#pragma omp parallel
  {
    cv::Mat localMean = cv::Mat::zeros(images[0].size(), CV_32F);
    cv::Mat localAccumSquare = cv::Mat::zeros(images[0].size(), CV_32F);

#pragma omp for nowait
    for (int i = 0; i < numImages; ++i) {
      cv::Mat floatImg;
      images[i].convertTo(floatImg, CV_32F);
      localMean += floatImg;
      localAccumSquare += floatImg.mul(floatImg);
    }

#pragma omp critical
    {
      mean += localMean;
      accumSquare += localAccumSquare;
    }
  }

  mean /= static_cast<float>(numImages);
  cv::Mat stdDev;
  cv::sqrt(accumSquare / static_cast<float>(numImages) - mean.mul(mean),
           stdDev);

  return {mean, stdDev};
}

// Sigma clipping stack
// 优化 sigmaClippingStack 函数，使用向量化操作
auto sigmaClippingStack(const std::vector<cv::Mat> &images, float sigma)
    -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for sigma clipping stack.");
    throw std::runtime_error("No images to stack");
  }

  stackLogger->info("Starting sigma clipping stack. Sigma value: {:.2f}",
                    sigma);

  cv::Mat mean, stdDev;
  try {
    std::tie(mean, stdDev) = computeMeanAndStdDev(images);
  } catch (const std::exception &e) {
    stackLogger->error("Failed to compute mean and standard deviation: {}",
                       e.what());
    throw;
  }

  // 使用向量化操作代替循环
  cv::Mat allImages;
  cv::vconcat(images, allImages);

  cv::Mat masks;
  cv::Mat temp = cv::abs(allImages - cv::repeat(mean, images.size(), 1));
  cv::compare(temp, cv::repeat(sigma * stdDev, images.size(), 1), masks,
              cv::CMP_LT);

  cv::Mat result;
  cv::reduce(allImages.mul(masks), result, 0, cv::REDUCE_SUM);
  cv::Mat counts;
  cv::reduce(masks, counts, 0, cv::REDUCE_SUM);

  // 防止除零
  counts.setTo(1, counts == 0);
  result /= counts;
  result.convertTo(result, CV_8U);

  stackLogger->info("Sigma clipping stack completed.");

  return result;
}

// Compute the mode (most frequent value) of each pixel
auto computeMode(const std::vector<cv::Mat> &images) -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("Input images are empty when computing mode.");
    throw std::runtime_error("Input images are empty");
  }

  stackLogger->info("Starting to compute image mode. Number of images: {}",
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

  stackLogger->info("Image mode computation completed.");

  return modeImage;
}

// 计算图像熵
// 优化 computeEntropy 函数，使用查找表加速计算
namespace {
// 预计算对数查找表
const int LOOKUP_TABLE_SIZE = 256;
std::vector<double> logLookupTable;

void initLogLookupTable() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    logLookupTable.resize(LOOKUP_TABLE_SIZE);
    for (int i = 1; i < LOOKUP_TABLE_SIZE; ++i) {
      logLookupTable[i] = std::log2(static_cast<double>(i));
    }
  });
}
} // namespace

auto computeEntropy(const cv::Mat &image) -> double {
  initLogLookupTable();

  std::array<int, 256> histogram{};
  const int totalPixels = image.rows * image.cols;

  // 使用向量化操作计算直方图
  for (int i = 0; i < image.rows; i++) {
    const uchar *row = image.ptr<uchar>(i);
    for (int j = 0; j < image.cols; j++) {
      histogram[row[j]]++;
    }
  }

  double entropy = 0.0;
  const double invTotal = 1.0 / totalPixels;

// 使用查找表计算熵
#pragma omp simd reduction(+ : entropy)
  for (int i = 0; i < 256; i++) {
    if (histogram[i] > 0) {
      double probability = histogram[i] * invTotal;
      entropy -= probability * logLookupTable[histogram[i]];
    }
  }

  return entropy;
}

// 基于熵的堆叠
auto entropyStack(const std::vector<cv::Mat> &images) -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for entropy stack.");
    throw std::runtime_error("No images to stack");
  }

  stackLogger->info("Starting entropy stack for {} images.", images.size());

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

  stackLogger->info("Entropy stack completed.");
  return result;
}

// 焦点堆叠
// 优化 focusStack 函数，增加并行处理和内存预分配
auto focusStack(const std::vector<cv::Mat> &images) -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for focus stack.");
    throw std::runtime_error("No images to stack");
  }

  stackLogger->info("Starting focus stack for {} images.", images.size());

  const int rows = images[0].rows;
  const int cols = images[0].cols;
  const size_t n = images.size();

  cv::Mat result(rows, cols, CV_8U);
  std::vector<cv::Mat> laplacians(n);

// 并行计算 Laplacian
#pragma omp parallel for
  for (int k = 0; k < static_cast<int>(n); ++k) {
    cv::Mat lap;
    cv::Laplacian(images[k], lap, CV_32F, 3);
    cv::convertScaleAbs(lap, laplacians[k]);
  }

// 使用向量化操作找到最大响应
#pragma omp parallel for collapse(2)
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      float maxResponse = -1;
      int bestIndex = 0;

// 使用 SIMD 指令优化内部循环
#pragma omp simd reduction(max : maxResponse)
      for (size_t k = 0; k < n; k++) {
        float response = laplacians[k].at<uchar>(i, j);
        if (response > maxResponse) {
          maxResponse = response;
          bestIndex = k;
        }
      }

      result.at<uchar>(i, j) = images[bestIndex].at<uchar>(i, j);
    }
  }

  stackLogger->info("Focus stack completed.");
  return result;
}

auto stackImages(const std::vector<cv::Mat> &images, StackMode mode,
                 float sigma, const std::vector<float> &weights) -> cv::Mat;

// Stack images by layers
auto stackImagesByLayers(const std::vector<cv::Mat> &images, StackMode mode,
                         float sigma, const std::vector<float> &weights)
    -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for stacking by layers.");
    throw std::runtime_error("No images to stack");
  }

  stackLogger->info("Starting image stacking by layers. Mode: {}",
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

  stackLogger->info("Image stacking by layers completed.");

  return stackedImage;
}

// 新增：截断平均堆叠算法（提高算法精度和鲁棒性，利用OpenMP优化性能）
auto trimmedMeanStack(const std::vector<cv::Mat> &images, float trimRatio)
    -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for trimmed mean stack.");
    throw std::runtime_error("No images to stack");
  }

  stackLogger->info("Starting trimmed mean stack with trim ratio: {:.2f}",
                    trimRatio);

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

  stackLogger->info("Trimmed mean stack completed.");
  return result;
}

// 新增：加权中值堆叠（Weighted Median Stack）
auto weightedMedianStack(const std::vector<cv::Mat> &images,
                         const std::vector<float> &weights) -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for weighted median stack.");
    throw std::runtime_error("No images to stack");
  }
  if (weights.size() != images.size()) {
    stackLogger->error("Number of weights does not match number of images for "
                       "weighted median stack.");
    throw std::runtime_error("Weights size mismatch");
  }

  stackLogger->info("Starting weighted median stack for {} images.",
                    images.size());
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

  stackLogger->info("Weighted median stack completed.");
  return result;
}

// 新增：自适应焦点堆叠（Adaptive Focus Stack）
// 利用每幅图像的 Laplacian 作为锐度权重计算加权平均
auto adaptiveFocusStack(const std::vector<cv::Mat> &images) -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for adaptive focus stack.");
    throw std::runtime_error("No images to stack");
  }

  stackLogger->info("Starting adaptive focus stack for {} images.",
                    images.size());
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

  stackLogger->info("Adaptive focus stack completed.");
  return result;
}

// Image stacking function
auto stackImages(const std::vector<cv::Mat> &images, StackMode mode,
                 float sigma, const std::vector<float> &weights) -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for stacking.");
    throw std::runtime_error("No images to stack");
  }

  stackLogger->info("Starting image stacking. Mode: {}",
                    static_cast<int>(mode));

  cv::Mat stackedImage;

  try {
    switch (mode) {
    case MEAN: {
      stackLogger->info("Selected stacking mode: Mean stack (MEAN)");
      cv::Mat stdDev; // Declare stdDev variable
      std::tie(stackedImage, stdDev) = computeMeanAndStdDev(images);
      stackedImage.convertTo(stackedImage, CV_8U);
      break;
    }
    case MEDIAN: {
      stackLogger->info("Selected stacking mode: Median stack (MEDIAN)");
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
      stackLogger->info("Selected stacking mode: Maximum stack (MAXIMUM)");
      stackedImage = images[0].clone();
      for (size_t i = 1; i < images.size(); ++i) {
        cv::max(stackedImage, images[i], stackedImage);
        stackLogger->info("Applied maximum stack: Image {}", i + 1);
      }
      break;
    }
    case MINIMUM: {
      stackLogger->info("Selected stacking mode: Minimum stack (MINIMUM)");
      stackedImage = images[0].clone();
      for (size_t i = 1; i < images.size(); ++i) {
        cv::min(stackedImage, images[i], stackedImage);
        stackLogger->info("Applied minimum stack: Image {}", i + 1);
      }
      break;
    }
    case SIGMA_CLIPPING: {
      stackLogger->info(
          "Selected stacking mode: Sigma clipping stack (SIGMA_CLIPPING)");
      stackedImage = sigmaClippingStack(images, sigma);
      break;
    }
    case WEIGHTED_MEAN: {
      stackLogger->info(
          "Selected stacking mode: Weighted mean stack (WEIGHTED_MEAN)");
      if (weights.empty()) {
        stackLogger->error("Weight vector is empty for weighted mean stack.");
        throw std::runtime_error("Weight vector cannot be empty");
      }
      if (weights.size() != images.size()) {
        stackLogger->error(
            "Number of weights does not match number of images.");
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
        stackLogger->info("Applied weight {}: {:.2f}", i + 1, weights[i]);
      }

      weightedSum /= totalWeight;
      weightedSum.convertTo(stackedImage, CV_8U);
      break;
    }
    case LIGHTEN: {
      stackLogger->info("Selected stacking mode: Lighten stack (LIGHTEN)");
      stackedImage = images[0].clone();
      for (size_t i = 1; i < images.size(); ++i) {
        cv::Mat mask = images[i] > stackedImage;
        images[i].copyTo(stackedImage, mask);
        stackLogger->info("Applied lighten stack: Image {}", i + 1);
      }
      break;
    }
    case MODE: {
      stackLogger->info("Selected stacking mode: Mode stack (MODE)");
      stackedImage = computeMode(images);
      break;
    }
    case ENTROPY: {
      stackLogger->info("Selected stacking mode: Entropy stack (ENTROPY)");
      stackedImage = entropyStack(images);
      break;
    }
    case FOCUS_STACK: {
      stackLogger->info("Selected stacking mode: Focus stack (FOCUS_STACK)");
      stackedImage = focusStack(images);
      break;
    }
    case TRIMMED_MEAN: {
      stackLogger->info(
          "Selected stacking mode: Trimmed mean stack (TRIMMED_MEAN)");
      float trimRatio = 0.2f; // 可调整的修剪比例，根据需要更改数值
      stackedImage = trimmedMeanStack(images, trimRatio);
      break;
    }
    case WEIGHTED_MEDIAN: {
      stackLogger->info(
          "Selected stacking mode: Weighted median stack (WEIGHTED_MEDIAN)");
      if (weights.empty()) {
        stackLogger->error("Weight vector is empty for weighted median stack.");
        throw std::runtime_error("Weight vector cannot be empty");
      }
      if (weights.size() != images.size()) {
        stackLogger->error(
            "Number of weights does not match number of images for "
            "weighted median stack.");
        throw std::runtime_error("Number of weights does not match");
      }
      stackedImage = weightedMedianStack(images, weights);
      break;
    }
    case ADAPTIVE_FOCUS: {
      stackLogger->info(
          "Selected stacking mode: Adaptive focus stack (ADAPTIVE_FOCUS)");
      stackedImage = adaptiveFocusStack(images);
      break;
    }
    default: {
      stackLogger->error("Unknown stacking mode: {}", static_cast<int>(mode));
      throw std::invalid_argument("Unknown stacking mode");
    }
    }

    stackLogger->info("Image stacking completed.");
  } catch (const std::exception &e) {
    stackLogger->error("Exception occurred during image stacking: {}",
                       e.what());
    throw;
  }

  return stackedImage;
}

auto stackImagesWithPreprocess(const std::vector<cv::Mat> &images,
                               StackMode mode,
                               const StackPreprocessConfig &preprocess_config,
                               float sigma, const std::vector<float> &weights)
    -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for stacking with preprocessing");
    throw std::runtime_error("No images to stack");
  }

  stackLogger->info("Starting image stacking with preprocessing. Mode: {}",
                    static_cast<int>(mode));

  std::vector<cv::Mat> preprocessed_images;
  preprocessed_images.reserve(images.size());

  if (preprocess_config.parallel_preprocess) {
// 并行处理每张图片
#pragma omp parallel for num_threads(preprocess_config.thread_count)
    for (int i = 0; i < static_cast<int>(images.size()); i++) {
      cv::Mat processed = preprocessImage(images[i], preprocess_config);
#pragma omp critical
      {
        preprocessed_images.push_back(processed);
      }
    }
  } else {
    // 串行处理
    for (const auto &img : images) {
      preprocessed_images.push_back(preprocessImage(img, preprocess_config));
    }
  }

  stackLogger->info("Preprocessing completed, starting stacking");

  // 使用处理后的图像进行堆叠
  return stackImages(preprocessed_images, mode, sigma, weights);
}