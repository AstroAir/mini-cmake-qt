#include "Filter.hpp"
#include "ImageUtils.hpp"
#include <opencv2/core/ocl.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>


// Setup a logger for filter operations.
namespace {
std::shared_ptr<spdlog::logger> filterLogger =
    spdlog::basic_logger_mt("FilterLogger", "logs/filter.log");
} // namespace

// ImageFilterProcessor implementation using UMat.
ImageFilterProcessor::ImageFilterProcessor(
    std::unique_ptr<IFilterStrategy> &&strategy)
    : strategy_(std::move(strategy)) {
  if (!strategy_) {
    filterLogger->error("Null strategy provided");
    throw ImageFilterException("Null strategy provided");
  }
}

QImage ImageFilterProcessor::process(const QImage &input) {
  try {
    cv::UMat umatImage = ImageUtils::qtImageToUMat(input);
    // Use OpenCV's parallel framework
    cv::parallel_for_(cv::Range(0, 1),
                      [&](const cv::Range &) { strategy_->apply(umatImage); });
    return ImageUtils::umatToQtImage(umatImage);
  } catch (const cv::Exception &e) {
    filterLogger->error("OpenCV exception: {}", e.what());
    throw ImageFilterException(e.what());
  } catch (...) {
    filterLogger->critical("Unknown processing error");
    throw;
  }
}

// ChainImageFilterProcessor implementation.
ChainImageFilterProcessor::ChainImageFilterProcessor(
    std::vector<std::unique_ptr<IFilterStrategy>> &&strategies)
    : strategies_(std::move(strategies)) {
  if (strategies_.empty()) {
    filterLogger->error("No strategies in chain");
    throw ImageFilterException("No strategies in chain");
  }
}

QImage ChainImageFilterProcessor::process(const QImage &input) {
  try {
    cv::UMat umatImage = ImageUtils::qtImageToUMat(input);
    for (const auto &strategy : strategies_) {
      cv::parallel_for_(cv::Range(0, 1),
                        [&](const cv::Range &) { strategy->apply(umatImage); });
    }
    return ImageUtils::umatToQtImage(umatImage);
  } catch (const cv::Exception &e) {
    filterLogger->error("OpenCV exception: {}", e.what());
    throw ImageFilterException(e.what());
  } catch (...) {
    filterLogger->critical("Unknown processing error");
    throw;
  }
}

// Implementation for GaussianBlurFilter.
GaussianBlurFilter::GaussianBlurFilter(int kernelSize, double sigma)
    : kernelSize_(kernelSize | 1), sigma_(sigma) // ensure odd kernel
{
  if (kernelSize_ < 3) {
    filterLogger->error("Kernel size too small");
    throw ImageFilterException("Kernel size too small");
  }
}

void GaussianBlurFilter::apply(cv::UMat &image) {
  validateImage(image);
  cv::GaussianBlur(image, image, cv::Size(kernelSize_, kernelSize_), sigma_,
                   sigma_);
}

const char *GaussianBlurFilter::name() const { return "Gaussian Blur"; }

// Implementation for MedianBlurFilter.
MedianBlurFilter::MedianBlurFilter(int kernelSize)
    : kernelSize_(kernelSize | 1) {
  if (kernelSize_ < 3) {
    filterLogger->error("Kernel size too small");
    throw ImageFilterException("Kernel size too small");
  }
}

void MedianBlurFilter::apply(cv::UMat &image) {
  validateImage(image);
  cv::medianBlur(image, image, kernelSize_);
}

const char *MedianBlurFilter::name() const { return "Median Blur"; }

// Implementation for BilateralFilter.
BilateralFilter::BilateralFilter(int diameter, double sigmaColor,
                                 double sigmaSpace)
    : diameter_(diameter), sigmaColor_(sigmaColor), sigmaSpace_(sigmaSpace) {}

void BilateralFilter::apply(cv::UMat &image) {
  validateImage(image);

  // 使用OpenCV的GPU加速
  if (cv::ocl::useOpenCL()) {
    cv::UMat result;
    cv::bilateralFilter(image, result, diameter_, sigmaColor_, sigmaSpace_);
    result.copyTo(image);
  } else {
    // 分块处理以提高缓存利用率
    const int blockSize = 32;
    cv::parallel_for_(cv::Range(0, image.rows), [&](const cv::Range &range) {
      for (int y = range.start; y < range.end; y += blockSize) {
        for (int x = 0; x < image.cols; x += blockSize) {
          cv::Rect block(x, y, std::min(blockSize, image.cols - x),
                         std::min(blockSize, image.rows - y));
          cv::UMat roi = image(block);
          cv::bilateralFilter(roi, roi, diameter_, sigmaColor_, sigmaSpace_);
        }
      }
    });
  }
}

const char *BilateralFilter::name() const { return "Bilateral Filter"; }

// Implementation for CannyEdgeFilter.
CannyEdgeFilter::CannyEdgeFilter(double threshold1, double threshold2)
    : threshold1_(threshold1), threshold2_(threshold2) {}

void CannyEdgeFilter::apply(cv::UMat &image) {
  validateImage(image);
  cv::UMat gray, edges;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::Canny(gray, edges, threshold1_, threshold2_);
  cv::cvtColor(edges, image, cv::COLOR_GRAY2BGR);
}

const char *CannyEdgeFilter::name() const { return "Canny Edge"; }

// Implementation for SharpenFilter.
SharpenFilter::SharpenFilter(double strength) : strength_(strength) {}

void SharpenFilter::apply(cv::UMat &image) {
  validateImage(image);

  // 使用OpenCV内置的SIMD优化
  cv::UMat blurred;
  {
    cv::UMat tmp;
    // 分离高斯核以加快计算
    cv::GaussianBlur(image, tmp, cv::Size(3, 1), 3);
    cv::GaussianBlur(tmp, blurred, cv::Size(1, 3), 3);
  }

  // 使用查找表优化权重计算
  static const cv::UMat weights = [this] {
    cv::Mat w(256, 1, CV_32F);
    for (int i = 0; i < 256; i++) {
      w.at<float>(i) = std::pow(i / 255.0f, strength_);
    }
    cv::UMat umat;
    w.copyTo(umat);
    return umat;
  }();

  cv::UMat enhanced;
  cv::LUT(image, weights, enhanced);
  cv::addWeighted(image, 1.0 + strength_, blurred, -strength_, 0, image);
}

const char *SharpenFilter::name() const { return "Sharpen"; }

// Implementation for HSVAdjustFilter.
HSVAdjustFilter::HSVAdjustFilter(double hue, double saturation, double value)
    : hue_(hue), saturation_(saturation), value_(value) {}

void HSVAdjustFilter::apply(cv::UMat &image) {
  validateImage(image);
  cv::UMat hsv;
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::UMat> channels;
  cv::split(hsv, channels);
  cv::add(channels[0], hue_, channels[0]);             // adjust hue
  cv::multiply(channels[1], saturation_, channels[1]); // adjust saturation
  cv::multiply(channels[2], value_, channels[2]); // adjust brightness/value
  cv::merge(channels, hsv);
  cv::cvtColor(hsv, image, cv::COLOR_HSV2BGR);
}

const char *HSVAdjustFilter::name() const { return "HSV Adjust"; }

// Implementation for ContrastBrightnessFilter.
ContrastBrightnessFilter::ContrastBrightnessFilter(double contrast,
                                                   double brightness)
    : contrast_(contrast), brightness_(brightness) {}

void ContrastBrightnessFilter::apply(cv::UMat &image) {
  validateImage(image);
  image.convertTo(image, -1, contrast_, brightness_);
}

const char *ContrastBrightnessFilter::name() const {
  return "Contrast & Brightness";
}

// Implementation for EmbossFilter.
void EmbossFilter::apply(cv::UMat &image) {
  validateImage(image);
  cv::Mat kernel = (cv::Mat_<float>(3, 3) << -2, -1, 0, -1, 1, 1, 0, 1, 2);
  cv::filter2D(image, image, image.depth(), kernel);
}

const char *EmbossFilter::name() const { return "Emboss"; }

// Implementation for AdaptiveThresholdFilter.
AdaptiveThresholdFilter::AdaptiveThresholdFilter(int blockSize, int C)
    : blockSize_(blockSize % 2 == 1 ? blockSize : blockSize + 1), C_(C) {}

void AdaptiveThresholdFilter::apply(cv::UMat &image) {
  validateImage(image);
  cv::UMat gray, thresh;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY, blockSize_, C_);
  cv::cvtColor(thresh, image, cv::COLOR_GRAY2BGR);
}

const char *AdaptiveThresholdFilter::name() const {
  return "Adaptive Threshold";
}

// Implementation for SobelEdgeFilter.
SobelEdgeFilter::SobelEdgeFilter(int ksize) : ksize_(ksize | 1) {
  if (ksize_ < 3) {
    ksize_ = 3;
  }
}

void SobelEdgeFilter::apply(cv::UMat &image) {
  validateImage(image);
  cv::UMat gray, gradX, gradY, absGradX, absGradY;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::Sobel(gray, gradX, CV_16S, 1, 0, ksize_);
  cv::Sobel(gray, gradY, CV_16S, 0, 1, ksize_);
  cv::convertScaleAbs(gradX, absGradX);
  cv::convertScaleAbs(gradY, absGradY);
  cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, gray);
  cv::cvtColor(gray, image, cv::COLOR_GRAY2BGR);
}

const char *SobelEdgeFilter::name() const { return "Sobel Edge"; }

// Implementation for LaplacianFilter.
LaplacianFilter::LaplacianFilter(int ksize) : ksize_(ksize | 1) {
  if (ksize_ < 3) {
    ksize_ = 3;
  }
}

void LaplacianFilter::apply(cv::UMat &image) {
  validateImage(image);
  cv::UMat gray, lap;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::Laplacian(gray, lap, CV_16S, ksize_);
  cv::convertScaleAbs(lap, gray);
  cv::cvtColor(gray, image, cv::COLOR_GRAY2BGR);
}

const char *LaplacianFilter::name() const { return "Laplacian"; }

// Implementation for UnsharpMaskFilter.
UnsharpMaskFilter::UnsharpMaskFilter(double strength, int blurKernelSize)
    : strength_(strength), blurKernelSize_(blurKernelSize | 1) {
  if (blurKernelSize_ < 3) {
    blurKernelSize_ = 3;
  }
}

void UnsharpMaskFilter::apply(cv::UMat &image) {
  validateImage(image);
  cv::UMat blurred, mask;
  cv::GaussianBlur(image, blurred, cv::Size(blurKernelSize_, blurKernelSize_),
                   0);
  // Calculate the mask as the difference between the original and the blurred
  // image.
  cv::subtract(image, blurred, mask);
  // Add the scaled mask back to the original image.
  cv::addWeighted(image, 1.0, mask, strength_, 0, image);
}

const char *UnsharpMaskFilter::name() const { return "Unsharp Mask"; }