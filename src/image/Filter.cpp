#include "Filter.hpp"
#include "ImageUtils.hpp"
#include <algorithm>
#include <future>
#include <opencv2/core/ocl.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

// Setup a logger for filter operations.
namespace {
std::shared_ptr<spdlog::logger> filterLogger =
    spdlog::basic_logger_mt("FilterLogger", "logs/filter.log");

// Thread pool for parallel processing
std::shared_ptr<std::vector<std::thread>> thread_pool = nullptr;
std::atomic<bool> pool_initialized = false;

void initialize_thread_pool() {
  if (!pool_initialized.exchange(true)) {
    unsigned int num_threads = std::thread::hardware_concurrency();
    thread_pool = std::make_shared<std::vector<std::thread>>(num_threads);
  }
}
} // namespace

// ImageFilterProcessor implementation using UMat.
ImageFilterProcessor::ImageFilterProcessor(
    std::unique_ptr<IFilterStrategy> &&strategy) noexcept
    : strategy_(std::move(strategy)) {
  if (!strategy_) {
    filterLogger->error("Null strategy provided");
    // We can still construct, but process() will throw later
  }
}

QImage ImageFilterProcessor::process(const QImage &input) {
  if (!strategy_) {
    filterLogger->error("Cannot process with null strategy");
    throw ImageFilterException("Null strategy provided");
  }

  if (is_processing_.exchange(true)) {
    filterLogger->warn("Attempted concurrent processing");
    throw ImageFilterException("Filter is already processing an image");
  }

  try {
    if (input.isNull() || input.width() <= 0 || input.height() <= 0) {
      throw ImageFilterException("Invalid input image");
    }

    cv::UMat umatImage = ImageUtils::qtImageToUMat(input);
    strategy_->apply(umatImage);
    QImage result = ImageUtils::umatToQtImage(umatImage);
    is_processing_ = false;
    return result;
  } catch (const cv::Exception &e) {
    is_processing_ = false;
    filterLogger->error("OpenCV exception: {}", e.what());
    throw ImageFilterException(std::string("OpenCV error: ") + e.what());
  } catch (const ImageFilterException &) {
    is_processing_ = false;
    throw; // Rethrow the original exception
  } catch (const std::exception &e) {
    is_processing_ = false;
    filterLogger->error("Standard exception: {}", e.what());
    throw ImageFilterException(std::string("Standard error: ") + e.what());
  } catch (...) {
    is_processing_ = false;
    filterLogger->critical("Unknown processing error");
    throw ImageFilterException("Unknown error during image processing");
  }
}

std::future<QImage> ImageFilterProcessor::processAsync(const QImage &input) {
  return std::async(std::launch::async,
                    [this, image = input]() { return this->process(image); });
}

// ChainImageFilterProcessor implementation.
ChainImageFilterProcessor::ChainImageFilterProcessor(
    std::vector<std::unique_ptr<IFilterStrategy>> &&strategies) noexcept
    : strategies_(std::move(strategies)) {
  if (strategies_.empty()) {
    filterLogger->error("No strategies in chain");
    // We can still construct, but process() will throw later
  }
}

QImage ChainImageFilterProcessor::process(const QImage &input) {
  if (strategies_.empty()) {
    filterLogger->error("Cannot process with empty strategy chain");
    throw ImageFilterException("No strategies in chain");
  }

  if (is_processing_.exchange(true)) {
    filterLogger->warn("Attempted concurrent chain processing");
    throw ImageFilterException("Filter chain is already processing an image");
  }

  try {
    if (input.isNull() || input.width() <= 0 || input.height() <= 0) {
      throw ImageFilterException("Invalid input image");
    }

    cv::UMat umatImage = ImageUtils::qtImageToUMat(input);

    // Apply each strategy in sequence
    for (const auto &strategy : strategies_) {
      if (!strategy) {
        throw ImageFilterException("Null strategy in chain");
      }
      strategy->apply(umatImage);
    }

    QImage result = ImageUtils::umatToQtImage(umatImage);
    is_processing_ = false;
    return result;
  } catch (const cv::Exception &e) {
    is_processing_ = false;
    filterLogger->error("OpenCV exception in chain: {}", e.what());
    throw ImageFilterException(std::string("OpenCV error in chain: ") +
                               e.what());
  } catch (const ImageFilterException &) {
    is_processing_ = false;
    throw; // Rethrow the original exception
  } catch (const std::exception &e) {
    is_processing_ = false;
    filterLogger->error("Standard exception in chain: {}", e.what());
    throw ImageFilterException(std::string("Standard error in chain: ") +
                               e.what());
  } catch (...) {
    is_processing_ = false;
    filterLogger->critical("Unknown processing error in chain");
    throw ImageFilterException("Unknown error during chain processing");
  }
}

std::future<QImage>
ChainImageFilterProcessor::processAsync(const QImage &input) {
  return std::async(std::launch::async,
                    [this, image = input]() { return this->process(image); });
}

// Implementation for GaussianBlurFilter.
GaussianBlurFilter::GaussianBlurFilter(int kernelSize, double sigma)
    : kernelSize_(std::max(3, kernelSize | 1)),
      sigma_(sigma > 0.0 ? sigma : 1.0) {
  if (kernelSize_ < 3) {
    filterLogger->error("Kernel size too small, adjusted to 3");
  }
  if (sigma_ <= 0.0) {
    filterLogger->error("Sigma must be positive, adjusted to 1.0");
  }
}

void GaussianBlurFilter::apply(cv::UMat &image) {
  validateImage(image);
  cv::GaussianBlur(image, image, cv::Size(kernelSize_, kernelSize_), sigma_,
                   sigma_);
}

const char *GaussianBlurFilter::name() const noexcept {
  return "Gaussian Blur";
}

// Implementation for MedianBlurFilter.
MedianBlurFilter::MedianBlurFilter(int kernelSize)
    : kernelSize_(std::max(3, kernelSize | 1)) {
  if (kernelSize_ < 3) {
    filterLogger->error("Kernel size too small, adjusted to 3");
  }
}

void MedianBlurFilter::apply(cv::UMat &image) {
  validateImage(image);
  cv::medianBlur(image, image, kernelSize_);
}

const char *MedianBlurFilter::name() const noexcept { return "Median Blur"; }

// Implementation for BilateralFilter.
BilateralFilter::BilateralFilter(int diameter, double sigmaColor,
                                 double sigmaSpace)
    : diameter_(std::max(1, diameter)),
      sigmaColor_(sigmaColor > 0.0 ? sigmaColor : 10.0),
      sigmaSpace_(sigmaSpace > 0.0 ? sigmaSpace : 10.0) {
  if (diameter_ < 1) {
    filterLogger->error("Diameter must be positive, adjusted to 1");
  }
  if (sigmaColor_ <= 0.0) {
    filterLogger->error("sigmaColor must be positive, adjusted to 10.0");
  }
  if (sigmaSpace_ <= 0.0) {
    filterLogger->error("sigmaSpace must be positive, adjusted to 10.0");
  }
}

void BilateralFilter::apply(cv::UMat &image) {
  validateImage(image);

  try {
    // Optimized implementation using OpenCV's OCL capabilities
    if (cv::ocl::useOpenCL()) {
      cv::UMat result;
      cv::bilateralFilter(image, result, diameter_, sigmaColor_, sigmaSpace_);
      result.copyTo(image);
    } else {
      // Tiled processing for better cache utilization
      const int blockSize = 64; // Larger block size for better performance
      const int rows = image.rows;
      const int cols = image.cols;

      // Use parallel_for with execution policy from C++17/20
      std::vector<std::pair<int, int>> blocks;
      for (int y = 0; y < rows; y += blockSize) {
        for (int x = 0; x < cols; x += blockSize) {
          blocks.emplace_back(y, x);
        }
      }

      std::for_each(blocks.begin(), blocks.end(), [&](const auto &block) {
        int y = block.first;
        int x = block.second;
        cv::Rect roi(x, y, std::min(blockSize, cols - x),
                     std::min(blockSize, rows - y));
        if (roi.width > 0 && roi.height > 0) {
          cv::UMat src = image(roi);
          cv::UMat dst;
          cv::bilateralFilter(src, dst, diameter_, sigmaColor_, sigmaSpace_);
          dst.copyTo(image(roi));
        }
      });
    }
  } catch (const cv::Exception &e) {
    filterLogger->error("OpenCV error in BilateralFilter: {}", e.what());
    throw ImageFilterException(std::string("BilateralFilter error: ") +
                               e.what());
  }
}

const char *BilateralFilter::name() const noexcept {
  return "Bilateral Filter";
}

// Implementation for CannyEdgeFilter.
CannyEdgeFilter::CannyEdgeFilter(double threshold1, double threshold2)
    : threshold1_(std::max(0.0, threshold1)),
      threshold2_(std::max(0.0, threshold2)) {
  if (threshold1_ < 0.0) {
    filterLogger->error("Threshold1 must be non-negative, adjusted to 0");
  }
  if (threshold2_ < 0.0) {
    filterLogger->error("Threshold2 must be non-negative, adjusted to 0");
  }
  // Ensure threshold2 >= threshold1 as required by Canny algorithm
  if (threshold2_ < threshold1_) {
    std::swap(threshold1_, threshold2_);
    filterLogger->warn("Thresholds swapped to ensure threshold2 >= threshold1");
  }
}

void CannyEdgeFilter::apply(cv::UMat &image) {
  validateImage(image);

  try {
    cv::UMat gray, edges;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, threshold1_, threshold2_);
    cv::cvtColor(edges, image, cv::COLOR_GRAY2BGR);
  } catch (const cv::Exception &e) {
    filterLogger->error("OpenCV error in CannyEdgeFilter: {}", e.what());
    throw ImageFilterException(std::string("CannyEdgeFilter error: ") +
                               e.what());
  }
}

const char *CannyEdgeFilter::name() const noexcept { return "Canny Edge"; }

// Implementation for SharpenFilter.
SharpenFilter::SharpenFilter(double strength)
    : strength_(std::clamp(strength, 0.0, 10.0)) {
  if (strength != strength_) {
    filterLogger->warn("Strength clamped to range [0.0, 10.0]: {}", strength_);
  }

  // Pre-compute weights lookup table for optimization
  cv::Mat w(256, 1, CV_32F);
  for (int i = 0; i < 256; i++) {
    w.at<float>(i) = std::pow(i / 255.0f, strength_);
  }
  w.copyTo(weights_lut_);
}

void SharpenFilter::apply(cv::UMat &image) {
  validateImage(image);

  try {
    // Optimized implementation with separable kernels
    cv::UMat blurred;
    {
      cv::UMat tmp;
      cv::GaussianBlur(image, tmp, cv::Size(3, 1), 3);
      cv::GaussianBlur(tmp, blurred, cv::Size(1, 3), 3);
    }

    // Use pre-computed lookup table
    cv::UMat enhanced;
    cv::LUT(image, weights_lut_, enhanced);
    cv::addWeighted(image, 1.0 + strength_, blurred, -strength_, 0, image);
  } catch (const cv::Exception &e) {
    filterLogger->error("OpenCV error in SharpenFilter: {}", e.what());
    throw ImageFilterException(std::string("SharpenFilter error: ") + e.what());
  }
}

const char *SharpenFilter::name() const noexcept { return "Sharpen"; }

// Implementation for HSVAdjustFilter.
HSVAdjustFilter::HSVAdjustFilter(double hue, double saturation, double value)
    : hue_(hue), saturation_(std::clamp(saturation, 0.0, 3.0)),
      value_(std::clamp(value, 0.0, 3.0)) {
  if (saturation != saturation_ || value != value_) {
    filterLogger->warn("Parameters clamped: saturation={}, value={}",
                       saturation_, value_);
  }
}

void HSVAdjustFilter::apply(cv::UMat &image) {
  validateImage(image);

  try {
    cv::UMat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Use split for better memory management
    std::vector<cv::UMat> channels;
    cv::split(hsv, channels);

    // Apply adjustments with bounds checking
    cv::add(channels[0], cv::Scalar(hue_), channels[0]);
    // Clamp values to valid ranges (0-179 for H, 0-255 for S and V in OpenCV)
    cv::multiply(channels[1], saturation_, channels[1]);
    cv::multiply(channels[2], value_, channels[2]);

    cv::merge(channels, hsv);
    cv::cvtColor(hsv, image, cv::COLOR_HSV2BGR);
  } catch (const cv::Exception &e) {
    filterLogger->error("OpenCV error in HSVAdjustFilter: {}", e.what());
    throw ImageFilterException(std::string("HSVAdjustFilter error: ") +
                               e.what());
  }
}

const char *HSVAdjustFilter::name() const noexcept { return "HSV Adjust"; }

// Implementation for ContrastBrightnessFilter.
ContrastBrightnessFilter::ContrastBrightnessFilter(double contrast,
                                                   double brightness)
    : contrast_(std::clamp(contrast, 0.0, 3.0)),
      brightness_(std::clamp(brightness, -100.0, 100.0)) {
  if (contrast != contrast_ || brightness != brightness_) {
    filterLogger->warn("Parameters clamped: contrast={}, brightness={}",
                       contrast_, brightness_);
  }
}

void ContrastBrightnessFilter::apply(cv::UMat &image) {
  validateImage(image);

  try {
    image.convertTo(image, -1, contrast_, brightness_);
  } catch (const cv::Exception &e) {
    filterLogger->error("OpenCV error in ContrastBrightnessFilter: {}",
                        e.what());
    throw ImageFilterException(std::string("ContrastBrightnessFilter error: ") +
                               e.what());
  }
}

const char *ContrastBrightnessFilter::name() const noexcept {
  return "Contrast & Brightness";
}

// Implementation for EmbossFilter.
// Constructor implementation is in the class definition since it's default

void EmbossFilter::apply(cv::UMat &image) {
  validateImage(image);

  try {
    // Initialize the kernel only once
    if (kernel_.empty()) {
      cv::Mat k = (cv::Mat_<float>(3, 3) << -2, -1, 0, -1, 1, 1, 0, 1, 2);
      k.copyTo(kernel_);
    }

    cv::filter2D(image, image, image.depth(), kernel_);
  } catch (const cv::Exception &e) {
    filterLogger->error("OpenCV error in EmbossFilter: {}", e.what());
    throw ImageFilterException(std::string("EmbossFilter error: ") + e.what());
  }
}

const char *EmbossFilter::name() const noexcept { return "Emboss"; }

// Implementation for AdaptiveThresholdFilter.
AdaptiveThresholdFilter::AdaptiveThresholdFilter(int blockSize, int C)
    : blockSize_(std::max(3, blockSize % 2 == 1 ? blockSize : blockSize + 1)),
      C_(C) {
  if (blockSize != blockSize_) {
    filterLogger->warn("blockSize adjusted to odd value: {}", blockSize_);
  }
}

void AdaptiveThresholdFilter::apply(cv::UMat &image) {
  validateImage(image);

  try {
    cv::UMat gray, thresh;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, blockSize_, C_);
    cv::cvtColor(thresh, image, cv::COLOR_GRAY2BGR);
  } catch (const cv::Exception &e) {
    filterLogger->error("OpenCV error in AdaptiveThresholdFilter: {}",
                        e.what());
    throw ImageFilterException(std::string("AdaptiveThresholdFilter error: ") +
                               e.what());
  }
}

const char *AdaptiveThresholdFilter::name() const noexcept {
  return "Adaptive Threshold";
}

// Implementation for SobelEdgeFilter.
SobelEdgeFilter::SobelEdgeFilter(int ksize) : ksize_(std::max(3, ksize | 1)) {
  if (ksize_ != ksize) {
    filterLogger->warn("ksize adjusted to valid odd value: {}", ksize_);
  }
}

void SobelEdgeFilter::apply(cv::UMat &image) {
  validateImage(image);

  try {
    cv::UMat gray, gradX, gradY, absGradX, absGradY;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Use Scharr for better edge detection when ksize is 3
    if (ksize_ == 3) {
      cv::Scharr(gray, gradX, CV_16S, 1, 0);
      cv::Scharr(gray, gradY, CV_16S, 0, 1);
    } else {
      cv::Sobel(gray, gradX, CV_16S, 1, 0, ksize_);
      cv::Sobel(gray, gradY, CV_16S, 0, 1, ksize_);
    }

    cv::convertScaleAbs(gradX, absGradX);
    cv::convertScaleAbs(gradY, absGradY);
    cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, gray);
    cv::cvtColor(gray, image, cv::COLOR_GRAY2BGR);
  } catch (const cv::Exception &e) {
    filterLogger->error("OpenCV error in SobelEdgeFilter: {}", e.what());
    throw ImageFilterException(std::string("SobelEdgeFilter error: ") +
                               e.what());
  }
}

const char *SobelEdgeFilter::name() const noexcept { return "Sobel Edge"; }

// Implementation for LaplacianFilter.
LaplacianFilter::LaplacianFilter(int ksize) : ksize_(std::max(1, ksize | 1)) {
  if (ksize_ != ksize) {
    filterLogger->warn("ksize adjusted to valid odd value: {}", ksize_);
  }
}

void LaplacianFilter::apply(cv::UMat &image) {
  validateImage(image);

  try {
    cv::UMat gray, lap;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Add Gaussian blur to reduce noise before Laplacian
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);

    cv::Laplacian(gray, lap, CV_16S, ksize_);
    cv::convertScaleAbs(lap, gray);
    cv::cvtColor(gray, image, cv::COLOR_GRAY2BGR);
  } catch (const cv::Exception &e) {
    filterLogger->error("OpenCV error in LaplacianFilter: {}", e.what());
    throw ImageFilterException(std::string("LaplacianFilter error: ") +
                               e.what());
  }
}

const char *LaplacianFilter::name() const noexcept { return "Laplacian"; }

// Implementation for UnsharpMaskFilter.
UnsharpMaskFilter::UnsharpMaskFilter(double strength, int blurKernelSize)
    : strength_(std::clamp(strength, 0.1, 5.0)),
      blurKernelSize_(std::max(3, blurKernelSize | 1)) {
  if (strength_ != strength) {
    filterLogger->warn("strength clamped to range [0.1, 5.0]: {}", strength_);
  }
  if (blurKernelSize_ != blurKernelSize) {
    filterLogger->warn("blurKernelSize adjusted to valid odd value: {}",
                       blurKernelSize_);
  }
}

void UnsharpMaskFilter::apply(cv::UMat &image) {
  validateImage(image);

  try {
    cv::UMat blurred, mask;
    cv::GaussianBlur(image, blurred, cv::Size(blurKernelSize_, blurKernelSize_),
                     0);

    // Calculate the mask as the difference between the original and the blurred
    // image
    cv::subtract(image, blurred, mask);

    // Add the scaled mask back to the original image
    cv::addWeighted(image, 1.0, mask, strength_, 0, image);
  } catch (const cv::Exception &e) {
    filterLogger->error("OpenCV error in UnsharpMaskFilter: {}", e.what());
    throw ImageFilterException(std::string("UnsharpMaskFilter error: ") +
                               e.what());
  }
}

const char *UnsharpMaskFilter::name() const noexcept { return "Unsharp Mask"; }