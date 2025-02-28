#include "Stack.hpp"
#include "src/image/ImageUtils.hpp"

#include <algorithm>
#include <cmath>
#include <functional> // std::invoke
#include <future>     // std::async
#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <utility> // std::move
#include <vector>


namespace {
std::shared_ptr<spdlog::logger> stackLogger =
    spdlog::basic_logger_mt("StackLogger", "logs/stack.log");

// Helper function to validate images before stacking
void validateImages(std::span<const cv::Mat> images) {
  if (images.empty()) {
    throw std::invalid_argument("Input images are empty");
  }

  const cv::Size firstSize = images[0].size();
  const int firstType = images[0].type();

  for (size_t i = 1; i < images.size(); ++i) {
    if (images[i].size() != firstSize) {
      throw std::invalid_argument("Images have inconsistent dimensions");
    }
    if (images[i].type() != firstType) {
      throw std::invalid_argument("Images have inconsistent types");
    }
    if (images[i].empty()) {
      throw std::invalid_argument("Empty image at index " + std::to_string(i));
    }
  }
}

// Optimized implementation using std::move_iterators
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

  // Calculate actual count to move
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

  // Reserve space in destination to avoid reallocations
  dest.reserve(dest.size() + actual_count);

  auto start = src.begin() + start_idx;
  auto end = start + actual_count;

  // Move elements using move iterators
  dest.insert(dest.end(), std::make_move_iterator(start),
              std::make_move_iterator(end));

  // Clear moved-from pointers
  src.erase(start, end);
}

// Helper function: perform image preprocessing pipeline
cv::Mat preprocessImage(const cv::Mat &input,
                        const StackPreprocessConfig &config) {
  if (input.empty()) {
    throw std::invalid_argument("Input image is empty");
  }

  cv::Mat processed = input.clone();

  try {
    // 1. Calibration processing
    if (config.enable_calibration) {
      auto result = flux_calibration_ex(
          processed, config.calibration_params,
          config.response_function.empty() ? nullptr
                                           : &config.response_function,
          config.flat_field.empty() ? nullptr : &config.flat_field,
          config.dark_frame.empty() ? nullptr : &config.dark_frame);
      if (result.has_value()) {
        processed = result.value().image;
      }
      stackLogger->debug("Applied calibration preprocessing");
    }

    // 2. Denoise processing
    if (config.enable_denoise) {
      ImageDenoiser denoiser;
      processed = denoiser.denoise(processed, config.denoise_params);
      stackLogger->debug("Applied denoise preprocessing");
    }

    // 3. Convolution processing
    if (config.enable_convolution) {
      processed = Convolve::process(processed, config.conv_config);
      stackLogger->debug("Applied convolution preprocessing");
    }

    // 4. Filter processing
    if (config.enable_filter && !config.filters.empty()) {
      std::vector<std::unique_ptr<IFilterStrategy>> strategies;
      transfer_unique_ptrs<IFilterStrategy>(
          strategies,
          const_cast<std::vector<std::unique_ptr<IFilterStrategy>> &>(
              config.filters));

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

// Optimized log lookup table with thread safety
class LogLookupTable {
public:
  static constexpr int TABLE_SIZE = 256;

  static double get(int index) noexcept {
    static LogLookupTable instance;
    if (index <= 0 || index >= TABLE_SIZE)
      return 0.0;
    return instance.table_[index];
  }

private:
  LogLookupTable() noexcept {
    for (int i = 1; i < TABLE_SIZE; ++i) {
      table_[i] = std::log2(static_cast<double>(i));
    }
  }

  double table_[TABLE_SIZE]{};
};

} // namespace

// Compute the mean and standard deviation of images
// Optimized with OpenMP and vectorization
auto computeMeanAndStdDev(std::span<const cv::Mat> images)
    -> std::pair<cv::Mat, cv::Mat> {
  if (images.empty()) {
    stackLogger->error("Input images are empty.");
    throw std::runtime_error("Input images are empty");
  }

  cv::Mat mean = cv::Mat::zeros(images[0].size(), CV_32F);
  cv::Mat squareSum = cv::Mat::zeros(images[0].size(), CV_32F);
  const int numImages = static_cast<int>(images.size());

  // Using std::vector of futures for async processing
  std::vector<std::future<std::pair<cv::Mat, cv::Mat>>> futures;
  const int numThreads =
      std::min(omp_get_max_threads(), 16); // Limit max threads
  const int imagesPerThread = (numImages + numThreads - 1) / numThreads;

  for (int t = 0; t < numThreads; ++t) {
    futures.push_back(std::async(std::launch::async, [&, t]() {
      cv::Mat localMean = cv::Mat::zeros(images[0].size(), CV_32F);
      cv::Mat localSquareSum = cv::Mat::zeros(images[0].size(), CV_32F);

      const int startIdx = t * imagesPerThread;
      const int endIdx = std::min(startIdx + imagesPerThread, numImages);

      for (int i = startIdx; i < endIdx; ++i) {
        cv::Mat floatImg;
        images[i].convertTo(floatImg, CV_32F);
        localMean += floatImg;
        cv::multiply(floatImg, floatImg, floatImg);
        localSquareSum += floatImg;
      }

      return std::make_pair(std::move(localMean), std::move(localSquareSum));
    }));
  }

  // Accumulate results from all threads
  for (auto &future : futures) {
    auto [threadMean, threadSquareSum] = future.get();
    mean += threadMean;
    squareSum += threadSquareSum;
  }

  mean /= static_cast<float>(numImages);
  cv::Mat stdDev;
  cv::sqrt(squareSum / static_cast<float>(numImages) - mean.mul(mean), stdDev);

  return {mean, stdDev};
}

// Sigma clipping stack with optimized vectorized operations
auto sigmaClippingStack(std::span<const cv::Mat> images, float sigma)
    -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for sigma clipping stack.");
    throw std::runtime_error("No images to stack");
  }

  if (sigma < 0) {
    stackLogger->error("Sigma value must be non-negative but got {}", sigma);
    throw std::invalid_argument("Sigma value cannot be negative");
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

  const int rows = images[0].rows;
  const int cols = images[0].cols;
  const size_t numImages = images.size();

  // Optimized implementation using vector operations
  cv::Mat result(rows, cols, CV_32F, cv::Scalar(0));
  cv::Mat counts(rows, cols, CV_32F, cv::Scalar(0));

// Use multiple threads for processing
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      float meanVal = mean.at<float>(r, c);
      float stdDevVal = stdDev.at<float>(r, c);
      float threshold = sigma * stdDevVal;

      // Process each pixel across all images
      float sum = 0.0f;
      int count = 0;

      for (size_t i = 0; i < numImages; ++i) {
        float pixelVal = static_cast<float>(images[i].at<uchar>(r, c));
        if (std::abs(pixelVal - meanVal) <= threshold) {
          sum += pixelVal;
          count++;
        }
      }

      if (count > 0) {
        result.at<float>(r, c) = sum / count;
      } else {
        // Fall back to mean if no pixels within threshold
        result.at<float>(r, c) = meanVal;
      }
    }
  }

  // Convert to 8-bit
  cv::Mat resultU8;
  result.convertTo(resultU8, CV_8U);

  stackLogger->info("Sigma clipping stack completed.");
  return resultU8;
}

// Optimized mode computation using histogram approach
auto computeMode(std::span<const cv::Mat> images) -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("Input images are empty when computing mode.");
    throw std::runtime_error("Input images are empty");
  }

  stackLogger->info("Starting to compute image mode. Number of images: {}",
                    images.size());

  const int rows = images[0].rows;
  const int cols = images[0].cols;
  cv::Mat modeImage(rows, cols, CV_8U);

// Process rows in parallel
#pragma omp parallel for schedule(dynamic)
  for (int r = 0; r < rows; ++r) {
    // Reuse histograms for each row to minimize memory allocations
    std::array<int, 256> histogram;

    for (int c = 0; c < cols; ++c) {
      // Clear histogram
      histogram.fill(0);

      // Build histogram for current pixel across all images
      for (const auto &img : images) {
        const uchar pixelValue = img.at<uchar>(r, c);
        histogram[pixelValue]++;
      }

      // Find most frequent value (mode)
      int maxFreq = 0;
      uchar modeValue = 0;

#pragma omp simd reduction(max : maxFreq)
      for (int i = 0; i < 256; ++i) {
        if (histogram[i] > maxFreq) {
          maxFreq = histogram[i];
          modeValue = static_cast<uchar>(i);
        }
      }

      modeImage.at<uchar>(r, c) = modeValue;
    }
  }

  stackLogger->info("Image mode computation completed.");
  return modeImage;
}

// Optimized entropy calculation with SIMD support
auto computeEntropy(const cv::Mat &image) noexcept -> double {
  if (image.empty()) {
    stackLogger->warn("Empty image provided to entropy calculation");
    return 0.0;
  }

  std::array<int, 256> histogram{};
  const int totalPixels = image.rows * image.cols;
  if (totalPixels == 0)
    return 0.0;

  // Calculate histogram using SIMD operations where possible
  for (int i = 0; i < image.rows; i++) {
    const uchar *row = image.ptr<uchar>(i);
    for (int j = 0; j < image.cols; j++) {
      histogram[row[j]]++;
    }
  }

  double entropy = 0.0;
  const double invTotal = 1.0 / totalPixels;

// Calculate entropy with SIMD acceleration
#pragma omp simd reduction(+ : entropy)
  for (int i = 0; i < 256; i++) {
    if (histogram[i] > 0) {
      double probability = histogram[i] * invTotal;
      entropy -= probability * LogLookupTable::get(histogram[i]);
    }
  }

  return entropy;
}

// Entropy-based stacking with improved parallelism
auto entropyStack(std::span<const cv::Mat> images) -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for entropy stack.");
    throw std::runtime_error("No images to stack");
  }

  stackLogger->info("Starting entropy stack for {} images.", images.size());

  const int rows = images[0].rows;
  const int cols = images[0].cols;
  cv::Mat result(rows, cols, CV_8U);

  const int windowSize = 9; // Must be odd
  const int offset = windowSize / 2;

  // Pre-allocate entropy matrices
  std::vector<cv::Mat> entropies(images.size());

// Calculate local entropy for each image in parallel
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < static_cast<int>(images.size()); ++i) {
    entropies[i] = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    cv::Mat &entropy = entropies[i];

    // Skip border pixels
    for (int r = offset; r < rows - offset; ++r) {
      for (int c = offset; c < cols - offset; ++c) {
        cv::Mat window = images[i](cv::Range(r - offset, r + offset + 1),
                                   cv::Range(c - offset, c + offset + 1));
        entropy.at<float>(r, c) = static_cast<float>(computeEntropy(window));
      }
    }
  }

// Select pixels with maximum entropy
#pragma omp parallel for collapse(2)
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      float maxEntropy = -1.0f;
      int bestIndex = 0;

      for (size_t i = 0; i < images.size(); ++i) {
        float entropyValue = entropies[i].at<float>(r, c);
        if (entropyValue > maxEntropy) {
          maxEntropy = entropyValue;
          bestIndex = static_cast<int>(i);
        }
      }

      result.at<uchar>(r, c) = images[bestIndex].at<uchar>(r, c);
    }
  }

  stackLogger->info("Entropy stack completed.");
  return result;
}

// Focus stacking with optimized Laplacian calculation
auto focusStack(std::span<const cv::Mat> images) -> cv::Mat {
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

  // Calculate Laplacian for each image in parallel
  std::vector<std::future<cv::Mat>> futures;
  for (size_t k = 0; k < n; ++k) {
    futures.push_back(std::async(std::launch::async, [&images, k]() {
      cv::Mat lap;
      cv::Laplacian(images[k], lap, CV_32F, 3);
      cv::Mat abs_lap;
      cv::convertScaleAbs(lap, abs_lap);
      return abs_lap;
    }));
  }

  // Get results from futures
  for (size_t k = 0; k < n; ++k) {
    laplacians[k] = futures[k].get();
  }

// Use best focus pixels from all images
#pragma omp parallel for collapse(2)
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      float maxResponse = -1.0f;
      int bestIndex = 0;

// Find image with maximum Laplacian response (sharpest)
#pragma omp simd reduction(max : maxResponse)
      for (size_t k = 0; k < n; ++k) {
        float response = static_cast<float>(laplacians[k].at<uchar>(r, c));
        if (response > maxResponse) {
          maxResponse = response;
          bestIndex = static_cast<int>(k);
        }
      }

      // Use pixel from sharpest image
      result.at<uchar>(r, c) = images[bestIndex].at<uchar>(r, c);
    }
  }

  stackLogger->info("Focus stack completed.");
  return result;
}

// Layered image stacking with improved memory management
template <ImageContainer ImgCont, WeightContainer WCont>
auto stackImagesByLayers(const ImgCont &images, StackMode mode, float sigma,
                         const WCont &weights) -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for stacking by layers.");
    throw std::runtime_error("No images to stack");
  }

  stackLogger->info("Starting image stacking by layers. Mode: {}",
                    static_cast<int>(mode));

  // Extract channels from first image to determine format
  std::vector<cv::Mat> firstImageChannels;
  cv::split(images[0], firstImageChannels);
  const int numChannels = static_cast<int>(firstImageChannels.size());

  // Create vector of channels for each color channel
  std::vector<std::vector<cv::Mat>> channelsVec(numChannels);
  for (int c = 0; c < numChannels; ++c) {
    channelsVec[c].reserve(images.size());
    channelsVec[c].push_back(std::move(firstImageChannels[c]));
  }

  // Extract channels from remaining images
  for (size_t i = 1; i < images.size(); ++i) {
    std::vector<cv::Mat> channels;
    cv::split(images[i], channels);

    if (channels.size() != static_cast<size_t>(numChannels)) {
      stackLogger->error("Inconsistent number of channels in image {}", i);
      throw std::runtime_error("Inconsistent number of channels in images");
    }

    for (int c = 0; c < numChannels; ++c) {
      channelsVec[c].push_back(std::move(channels[c]));
    }
  }

  // Stack each channel separately
  std::vector<cv::Mat> stackedChannels;
  stackedChannels.reserve(numChannels);

  for (const auto &channelImages : channelsVec) {
    // Create span from vector for stackImages call
    std::span<const cv::Mat> channelSpan(channelImages);
    stackedChannels.push_back(stackImages(channelSpan, mode, sigma, weights));
  }

  // Merge channels back
  cv::Mat stackedImage;
  cv::merge(stackedChannels, stackedImage);

  stackLogger->info("Image stacking by layers completed.");
  return stackedImage;
}

// Trimmed mean stack with better trimming algorithm
auto trimmedMeanStack(std::span<const cv::Mat> images, float trimRatio)
    -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for trimmed mean stack.");
    throw std::runtime_error("No images to stack");
  }

  if (trimRatio < 0.0f || trimRatio >= 1.0f) {
    stackLogger->error("Invalid trim ratio: {}", trimRatio);
    throw std::invalid_argument("Trim ratio must be in range [0.0, 1.0)");
  }

  stackLogger->info("Starting trimmed mean stack with trim ratio: {:.2f}",
                    trimRatio);

  const int rows = images[0].rows;
  const int cols = images[0].cols;
  cv::Mat result(rows, cols, CV_32F);
  const int totalImages = static_cast<int>(images.size());
  const int trimCount = static_cast<int>(totalImages * trimRatio / 2.0f);

  // Early optimization if no trimming needed
  if (trimCount == 0) {
    // Just compute mean
    cv::Mat stdDev; // Unused
    std::tie(result, stdDev) = computeMeanAndStdDev(images);
  } else {
// Use parallel processing for trimmed mean calculation
#pragma omp parallel for collapse(2)
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        // Collect pixel values across all images
        std::vector<float> pixelValues;
        pixelValues.reserve(totalImages);

        for (const auto &img : images) {
          pixelValues.push_back(static_cast<float>(img.at<uchar>(r, c)));
        }

        // Sort using partial_sort for better performance
        std::partial_sort(pixelValues.begin(), pixelValues.begin() + trimCount,
                          pixelValues.end());

        std::partial_sort(pixelValues.rbegin(),
                          pixelValues.rbegin() + trimCount, pixelValues.rend(),
                          std::greater<float>());

        // Compute mean of remaining values
        float sum = 0.0f;
        for (int i = trimCount; i < totalImages - trimCount; ++i) {
          sum += pixelValues[i];
        }

        int count = totalImages - 2 * trimCount;
        result.at<float>(r, c) =
            count > 0 ? sum / count : pixelValues[trimCount];
      }
    }
  }

  // Convert to 8-bit
  cv::Mat resultU8;
  result.convertTo(resultU8, CV_8U);

  stackLogger->info("Trimmed mean stack completed.");
  return resultU8;
}

// Weighted median stack with optimized sorting
auto weightedMedianStack(std::span<const cv::Mat> images,
                         std::span<const float> weights) -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for weighted median stack.");
    throw std::runtime_error("No images to stack");
  }

  if (weights.size() != images.size()) {
    stackLogger->error(
        "Number of weights ({}) does not match number of images ({}).",
        weights.size(), images.size());
    throw std::invalid_argument("Weights size mismatch");
  }

  // Validate weights
  float totalWeight = 0.0f;
  for (float w : weights) {
    if (w < 0.0f) {
      throw std::invalid_argument("Weights cannot be negative");
    }
    totalWeight += w;
  }

  if (totalWeight <= 0.0f) {
    throw std::invalid_argument("Total weight must be positive");
  }

  stackLogger->info("Starting weighted median stack for {} images.",
                    images.size());

  const int rows = images[0].rows;
  const int cols = images[0].cols;
  const int n = static_cast<int>(images.size());

  cv::Mat result(rows, cols, CV_8U);

// Process rows in parallel for better cache utilization
#pragma omp parallel for schedule(dynamic)
  for (int r = 0; r < rows; ++r) {
    // Pre-allocate vector once per row for efficiency
    std::vector<std::pair<uchar, float>> pixelWeights;
    pixelWeights.reserve(n);

    for (int c = 0; c < cols; ++c) {
      pixelWeights.clear();
      float cumulativeWeight = 0.0f;

      // Collect weighted pixels
      for (int k = 0; k < n; ++k) {
        uchar pixel = images[k].at<uchar>(r, c);
        float w = weights[k];
        pixelWeights.push_back({pixel, w});
        cumulativeWeight += w;
      }

      const float halfWeight = cumulativeWeight / 2.0f;

      // Sort by pixel value for weighted median calculation
      std::sort(pixelWeights.begin(), pixelWeights.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });

      // Find weighted median
      float weightSum = 0.0f;
      uchar medianPixel = 0;

      for (const auto &[pixel, weight] : pixelWeights) {
        weightSum += weight;
        if (weightSum >= halfWeight) {
          medianPixel = pixel;
          break;
        }
      }

      result.at<uchar>(r, c) = medianPixel;
    }
  }

  stackLogger->info("Weighted median stack completed.");
  return result;
}

// Adaptive focus stacking with advanced focus measure
auto adaptiveFocusStack(std::span<const cv::Mat> images) -> cv::Mat {
  if (images.empty()) {
    stackLogger->error("No input images for adaptive focus stack.");
    throw std::runtime_error("No images to stack");
  }

  stackLogger->info("Starting adaptive focus stack for {} images.",
                    images.size());

  const int rows = images[0].rows;
  const int cols = images[0].cols;
  const size_t numImages = images.size();

  // Allocate result image
  cv::Mat result(rows, cols, CV_32F, cv::Scalar(0.0f));
  cv::Mat weightSum(rows, cols, CV_32F, cv::Scalar(0.0f));

  // Calculate Laplacian response for each image - a measure of local sharpness
  std::vector<cv::Mat> laplacians(numImages);

  // Process images in parallel using std::async
  std::vector<std::future<cv::Mat>> futures;
  for (size_t i = 0; i < numImages; ++i) {
    futures.push_back(std::async(std::launch::async, [&images, i]() {
      cv::Mat laplacian;
      cv::Mat grayImg;

      // Convert to grayscale if needed
      if (images[i].channels() > 1) {
        cv::cvtColor(images[i], grayImg, cv::COLOR_BGR2GRAY);
      } else {
        grayImg = images[i];
      }

      // Apply Laplacian filter
      cv::Laplacian(grayImg, laplacian, CV_32F, 3);
      cv::Mat absLaplacian;
      cv::convertScaleAbs(laplacian, absLaplacian);

      // Apply Gaussian blur to create smoother weights
      cv::Mat smoothed;
      cv::GaussianBlur(absLaplacian, smoothed, cv::Size(5, 5), 0);

      // Convert to floating point for weight calculation
      cv::Mat response;
      smoothed.convertTo(response, CV_32F);

      // Square the response to emphasize areas of high frequency
      cv::multiply(response, response, response);

      return response;
    }));
  }

  // Collect results
  for (size_t i = 0; i < numImages; ++i) {
    laplacians[i] = futures[i].get();
  }

// Process pixel by pixel in parallel
#pragma omp parallel for collapse(2)
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      // Get weights for current pixel across all images
      std::vector<std::pair<float, int>> pixelWeights;
      pixelWeights.reserve(numImages);

      for (size_t i = 0; i < numImages; ++i) {
        float weight = laplacians[i].at<float>(r, c);
        pixelWeights.push_back({weight, static_cast<int>(i)});
      }

      // Sort weights in descending order
      std::sort(pixelWeights.begin(), pixelWeights.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });

      // Take top contributors (adaptive weighting)
      const size_t topCount = std::min<size_t>(3, numImages);
      float pixelSum = 0.0f;
      float totalWeight = 0.0f;

      for (size_t i = 0; i < topCount; ++i) {
        const auto &[weight, imgIdx] = pixelWeights[i];
        if (weight > 0) {
          pixelSum += images[imgIdx].at<uchar>(r, c) * weight;
          totalWeight += weight;
        }
      }

      if (totalWeight > 0) {
        result.at<float>(r, c) = pixelSum / totalWeight;
      } else {
        // Fallback to maximum weight if all weights are zero
        result.at<float>(r, c) =
            static_cast<float>(images[pixelWeights[0].second].at<uchar>(r, c));
      }
    }
  }

  // Convert to 8-bit
  cv::Mat resultU8;
  result.convertTo(resultU8, CV_8U);

  stackLogger->info("Adaptive focus stack completed.");
  return resultU8;
}

// Main stacking function with span interface
template <typename ImgCont, typename WCont>
auto stackImages(const ImgCont &img_container, StackMode mode, float sigma,
                 const WCont &weight_container) -> cv::Mat {
  // Convert containers to spans for unified processing
  std::span<const cv::Mat> images(img_container.data(), img_container.size());
  std::span<const float> weights;
  if constexpr (std::is_same_v<WCont, std::vector<float>>) {
    if (!weight_container.empty()) {
      weights = std::span<const float>(weight_container.data(),
                                       weight_container.size());
    }
  }

  try {
    validateImages(images);
  } catch (const std::exception &e) {
    stackLogger->error("Image validation failed: {}", e.what());
    throw;
  }

  stackLogger->info("Starting image stacking. Mode: {}",
                    static_cast<int>(mode));
  cv::Mat stackedImage;

  try {
    switch (mode) {
    case StackMode::MEAN: {
      stackLogger->info("Selected stacking mode: Mean stack (MEAN)");
      cv::Mat stdDev;
      std::tie(stackedImage, stdDev) = computeMeanAndStdDev(images);
      stackedImage.convertTo(stackedImage, CV_8U);
      break;
    }
    case StackMode::MEDIAN: {
      stackLogger->info("Selected stacking mode: Median stack (MEDIAN)");

      // Optimize median calculation for large image sets
      const int rows = images[0].rows;
      const int cols = images[0].cols;
      cv::Mat medianImg(rows, cols, CV_8U);

#pragma omp parallel for collapse(2)
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          std::vector<uchar> values;
          values.reserve(images.size());

          for (const auto &img : images) {
            values.push_back(img.at<uchar>(r, c));
          }

          std::nth_element(values.begin(), values.begin() + values.size() / 2,
                           values.end());
          medianImg.at<uchar>(r, c) = values[values.size() / 2];
        }
      }

      stackedImage = medianImg;
      break;
    }
    case StackMode::MAXIMUM: {
      stackLogger->info("Selected stacking mode: Maximum stack (MAXIMUM)");
      stackedImage = images[0].clone();
      for (size_t i = 1; i < images.size(); ++i) {
        cv::max(stackedImage, images[i], stackedImage);
      }
      break;
    }
    case StackMode::MINIMUM: {
      stackLogger->info("Selected stacking mode: Minimum stack (MINIMUM)");
      stackedImage = images[0].clone();
      for (size_t i = 1; i < images.size(); ++i) {
        cv::min(stackedImage, images[i], stackedImage);
      }
      break;
    }
    case StackMode::SIGMA_CLIPPING: {
      stackLogger->info(
          "Selected stacking mode: Sigma clipping stack (SIGMA_CLIPPING)");
      stackedImage = sigmaClippingStack(images, sigma);
      break;
    }
    case StackMode::WEIGHTED_MEAN: {
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
      float totalWeight = 0.0f;

#pragma omp parallel for reduction(+ : totalWeight)
      for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
        totalWeight += weights[i];
      }

      if (totalWeight <= 0) {
        throw std::runtime_error("Total weight must be positive");
      }

      for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat floatImg;
        images[i].convertTo(floatImg, CV_32F);
        weightedSum += floatImg * weights[i] / totalWeight;
      }

      weightedSum.convertTo(stackedImage, CV_8U);
      break;
    }
    case StackMode::LIGHTEN: {
      stackLogger->info("Selected stacking mode: Lighten stack (LIGHTEN)");
      stackedImage = images[0].clone();

      // Use parallel processing for large image sets
      if (images.size() > 10) {
        cv::parallel_for_(
            cv::Range(0, stackedImage.rows), [&](const cv::Range &range) {
              for (int r = range.start; r < range.end; ++r) {
                for (int c = 0; c < stackedImage.cols; ++c) {
                  uchar maxVal = stackedImage.at<uchar>(r, c);
                  for (size_t i = 1; i < images.size(); ++i) {
                    maxVal = std::max(maxVal, images[i].at<uchar>(r, c));
                  }
                  stackedImage.at<uchar>(r, c) = maxVal;
                }
              }
            });
      } else {
        for (size_t i = 1; i < images.size(); ++i) {
          cv::Mat mask = images[i] > stackedImage;
          images[i].copyTo(stackedImage, mask);
        }
      }
      break;
    }
    case StackMode::MODE: {
      stackLogger->info("Selected stacking mode: Mode stack (MODE)");
      stackedImage = computeMode(images);
      break;
    }
    case StackMode::ENTROPY: {
      stackLogger->info("Selected stacking mode: Entropy stack (ENTROPY)");
      stackedImage = entropyStack(images);
      break;
    }
    case StackMode::FOCUS_STACK: {
      stackLogger->info("Selected stacking mode: Focus stack (FOCUS_STACK)");
      stackedImage = focusStack(images);
      break;
    }
    case StackMode::TRIMMED_MEAN: {
      stackLogger->info(
          "Selected stacking mode: Trimmed mean stack (TRIMMED_MEAN)");
      const float trimRatio = 0.2f; // Configurable trimming ratio
      stackedImage = trimmedMeanStack(images, trimRatio);
      break;
    }
    case StackMode::WEIGHTED_MEDIAN: {
      stackLogger->info(
          "Selected stacking mode: Weighted median stack (WEIGHTED_MEDIAN)");
      if (weights.empty()) {
        stackLogger->error("Weight vector is empty for weighted median stack.");
        throw std::runtime_error("Weight vector cannot be empty");
      }
      if (weights.size() != images.size()) {
        stackLogger->error(
            "Number of weights does not match number of images.");
        throw std::runtime_error("Number of weights does not match");
      }
      stackedImage = weightedMedianStack(images, weights);
      break;
    }
    case StackMode::ADAPTIVE_FOCUS: {
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

    stackLogger->info("Image stacking completed successfully.");
    return stackedImage;

  } catch (const std::exception &e) {
    stackLogger->error("Exception occurred during image stacking: {}",
                       e.what());
    throw;
  }
}

// Stacking with preprocessing
template <typename ImgCont, typename WCont>
auto stackImagesWithPreprocess(const ImgCont &img_container, StackMode mode,
                               const StackPreprocessConfig &preprocess_config,
                               float sigma, const WCont &weight_container)
    -> cv::Mat {
  // Convert containers to spans
  std::span<const cv::Mat> images(img_container.data(), img_container.size());
  std::span<const float> weights;

  if constexpr (std::is_same_v<WCont, std::vector<float>>) {
    if (!weight_container.empty()) {
      weights = std::span<const float>(weight_container);
    }
  }

  try {
    validateImages(images);
  } catch (const std::exception &e) {
    stackLogger->error("Image validation failed during preprocessing: {}",
                       e.what());
    throw;
  }

  stackLogger->info("Starting image stacking with preprocessing. Mode: {}",
                    static_cast<int>(mode));

  std::vector<cv::Mat> preprocessed_images;
  preprocessed_images.reserve(images.size());

  if (preprocess_config.parallel_preprocess) {
    // Calculate optimal thread count
    const int max_threads = std::min(preprocess_config.thread_count,
                                     static_cast<int>(images.size()));

    // Parallel processing with thread pool
    std::vector<std::future<cv::Mat>> futures;
    futures.reserve(images.size());

    for (const auto &img : images) {
      futures.push_back(
          std::async(std::launch::async, [&img, &preprocess_config]() {
            return preprocessImage(img, preprocess_config);
          }));
    }

    // Collect processed images
    for (auto &future : futures) {
      try {
        preprocessed_images.push_back(future.get());
      } catch (const std::exception &e) {
        stackLogger->error("Preprocessing failed for an image: {}", e.what());
        throw;
      }
    }
  } else {
    // Sequential processing
    for (const auto &img : images) {
      try {
        preprocessed_images.push_back(preprocessImage(img, preprocess_config));
      } catch (const std::exception &e) {
        stackLogger->error("Sequential preprocessing failed: {}", e.what());
        throw;
      }
    }
  }

  stackLogger->info("Preprocessing completed for {} images",
                    preprocessed_images.size());

  // Use preprocessed images for stacking
  return stackImages(preprocessed_images, mode, sigma, weight_container);
}
