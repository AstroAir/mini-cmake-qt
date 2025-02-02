#include "Filter.hpp"

#include <QImage>
#include <memory>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <vector>

namespace ImageUtils {
cv::Mat qtImageToMat(const QImage &img) {
  try {
    return cv::Mat(img.height(), img.width(),
                   img.format() == QImage::Format_RGB32 ? CV_8UC4 : CV_8UC3,
                   const_cast<uchar *>(img.bits()),
                   static_cast<size_t>(img.bytesPerLine()));
  } catch (...) {
    throw ImageFilterException("Image conversion failed");
  }
}

QImage matToQtImage(const cv::Mat &mat) {
  try {
    return QImage(mat.data, mat.cols, mat.rows, mat.step,
                  mat.channels() == 4 ? QImage::Format_RGB32
                                      : QImage::Format_RGB888)
        .copy(); // 确保深拷贝
  } catch (...) {
    throw ImageFilterException("Mat to QImage conversion failed");
  }
}

}

void IFilterStrategy::validateImage(const cv::Mat &img) const {
  if (img.empty())
    throw ImageFilterException("Empty input image");
  if (img.depth() != CV_8U)
    throw ImageFilterException("Unsupported image depth");
}
ImageFilterProcessor::ImageFilterProcessor(
    std::unique_ptr<IFilterStrategy> &&strategy)
    : strategy_(std::move(strategy)) {
  if (!strategy_) {
    throw ImageFilterException("Null strategy provided");
  }
}

// 并行处理接口
QImage ImageFilterProcessor::process(const QImage &input) {
  try {
    cv::Mat cvImage = ImageUtils::qtImageToMat(input);

    // 使用OpenCV的并行框架
    cv::parallel_for_(cv::Range(0, 1), [&](const cv::Range &range) {
      strategy_->apply(cvImage);
    });

    return ImageUtils::matToQtImage(cvImage);
  } catch (const cv::Exception &e) {
    spdlog::error("OpenCV exception: {}", e.what());
    throw ImageFilterException(e.what());
  } catch (...) {
    spdlog::critical("Unknown processing error");
    throw;
  }
}

ChainImageFilterProcessor::ChainImageFilterProcessor(
    std::vector<std::unique_ptr<IFilterStrategy>> &&strategies)
    : strategies_(std::move(strategies)) {
  if (strategies_.empty()) {
    throw ImageFilterException("No strategies in chain");
  }
}

QImage ChainImageFilterProcessor::process(const QImage &input) {
  try {
    cv::Mat cvImage = ImageUtils::qtImageToMat(input);
    // 顺序应用每个滤镜
    for (const auto &strategy : strategies_) {
      // 每个滤镜内部也可以使用并行处理
      cv::parallel_for_(cv::Range(0, 1), [&](const cv::Range &range) {
        strategy->apply(cvImage);
      });
    }
    return ImageUtils::matToQtImage(cvImage);
  } catch (const cv::Exception &e) {
    spdlog::error("OpenCV exception: {}", e.what());
    throw ImageFilterException(e.what());
  } catch (...) {
    spdlog::critical("Unknown processing error");
    throw;
  }
}

GaussianBlurFilter::GaussianBlurFilter(int kernelSize, double sigma)
    : kernelSize_(kernelSize | 1), sigma_(sigma) // 确保奇数核
{
  if (kernelSize_ < 3)
    throw ImageFilterException("Kernel size too small");
}

void GaussianBlurFilter::apply(cv::Mat &image) {
  validateImage(image);
  cv::GaussianBlur(image, image, cv::Size(kernelSize_, kernelSize_), sigma_,
                   sigma_);
}

const char *GaussianBlurFilter::name() const { return "Gaussian Blur"; }

MedianBlurFilter::MedianBlurFilter(int kernelSize)
    : kernelSize_(kernelSize | 1) { // 确保奇数核
  if (kernelSize_ < 3)
    throw ImageFilterException("Kernel size too small");
}

void MedianBlurFilter::apply(cv::Mat &image) {
  validateImage(image);
  cv::medianBlur(image, image, kernelSize_);
}

const char *MedianBlurFilter::name() const { return "Median Blur"; }

BilateralFilter::BilateralFilter(int diameter, double sigmaColor,
                                 double sigmaSpace)
    : diameter_(diameter), sigmaColor_(sigmaColor), sigmaSpace_(sigmaSpace) {}

void BilateralFilter::apply(cv::Mat &image) {
  validateImage(image);
  cv::bilateralFilter(image, image, diameter_, sigmaColor_, sigmaSpace_);
}

const char *BilateralFilter::name() const { return "Bilateral Filter"; }

CannyEdgeFilter::CannyEdgeFilter(double threshold1, double threshold2)
    : threshold1_(threshold1), threshold2_(threshold2) {}

void CannyEdgeFilter::apply(cv::Mat &image) {
  validateImage(image);
  cv::Mat edges;
  cv::cvtColor(image, edges, cv::COLOR_BGR2GRAY);
  cv::Canny(edges, edges, threshold1_, threshold2_);
  cv::cvtColor(edges, image, cv::COLOR_GRAY2BGR);
}

const char *CannyEdgeFilter::name() const { return "Canny Edge"; }

SharpenFilter::SharpenFilter(double strength) : strength_(strength) {}

void SharpenFilter::apply(cv::Mat &image) {
  validateImage(image);
  cv::Mat blurred;
  cv::GaussianBlur(image, blurred, cv::Size(0, 0), 3);
  cv::addWeighted(image, 1.0 + strength_, blurred, -strength_, 0, image);
}

const char *SharpenFilter::name() const { return "Sharpen"; }

HSVAdjustFilter::HSVAdjustFilter(double hue, double saturation, double value)
    : hue_(hue), saturation_(saturation), value_(value) {}

void HSVAdjustFilter::apply(cv::Mat &image) {
  validateImage(image);
  cv::Mat hsv;
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
  
  std::vector<cv::Mat> channels;
  cv::split(hsv, channels);
  
  channels[0] = channels[0] + hue_;  // 色相调整
  channels[1] = channels[1] * saturation_;  // 饱和度调整
  channels[2] = channels[2] * value_;  // 明度调整
  
  cv::merge(channels, hsv);
  cv::cvtColor(hsv, image, cv::COLOR_HSV2BGR);
}

const char *HSVAdjustFilter::name() const { return "HSV Adjust"; }

ContrastBrightnessFilter::ContrastBrightnessFilter(double contrast, double brightness)
    : contrast_(contrast), brightness_(brightness) {}

void ContrastBrightnessFilter::apply(cv::Mat &image) {
  validateImage(image);
  image.convertTo(image, -1, contrast_, brightness_);
}

const char *ContrastBrightnessFilter::name() const { return "Contrast & Brightness"; }
