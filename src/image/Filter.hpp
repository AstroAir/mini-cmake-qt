#ifndef FILTER_HPP
#define FILTER_HPP

#include <QImage>
#include <exception>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


// Common exception class used by filter implementations.
class ImageFilterException : public std::exception {
public:
  explicit ImageFilterException(const std::string &msg) : msg_(msg) {}
  const char *what() const noexcept override { return msg_.c_str(); }

private:
  std::string msg_;
};

// Base interface for all filter strategies.
class IFilterStrategy {
public:
  virtual ~IFilterStrategy() = default;
  virtual void apply(cv::UMat &image) = 0;
  virtual const char *name() const = 0;

protected:
  void validateImage(const cv::UMat &img) const {
    if (img.empty()) {
      throw ImageFilterException("Empty input image");
    }
    if (img.depth() != CV_8U) {
      throw ImageFilterException("Unsupported image depth");
    }
  }
};

// Basic image filter processor using a single strategy.
class ImageFilterProcessor {
public:
  explicit ImageFilterProcessor(std::unique_ptr<IFilterStrategy> &&strategy);
  QImage process(const QImage &input);

private:
  std::unique_ptr<IFilterStrategy> strategy_;
};

// Chain processor to apply multiple filters in sequence.
class ChainImageFilterProcessor {
public:
  explicit ChainImageFilterProcessor(
      std::vector<std::unique_ptr<IFilterStrategy>> &&strategies);
  QImage process(const QImage &input);

private:
  std::vector<std::unique_ptr<IFilterStrategy>> strategies_;
};

// Existing Filter Strategies:
class GaussianBlurFilter : public IFilterStrategy {
public:
  GaussianBlurFilter(int kernelSize, double sigma);
  void apply(cv::UMat &image) override;
  const char *name() const override;

private:
  int kernelSize_;
  double sigma_;
};

class MedianBlurFilter : public IFilterStrategy {
public:
  explicit MedianBlurFilter(int kernelSize);
  void apply(cv::UMat &image) override;
  const char *name() const override;

private:
  int kernelSize_;
};

class BilateralFilter : public IFilterStrategy {
public:
  BilateralFilter(int diameter, double sigmaColor, double sigmaSpace);
  void apply(cv::UMat &image) override;
  const char *name() const override;

private:
  int diameter_;
  double sigmaColor_;
  double sigmaSpace_;
};

class CannyEdgeFilter : public IFilterStrategy {
public:
  CannyEdgeFilter(double threshold1, double threshold2);
  void apply(cv::UMat &image) override;
  const char *name() const override;

private:
  double threshold1_;
  double threshold2_;
};

class SharpenFilter : public IFilterStrategy {
public:
  explicit SharpenFilter(double strength);
  void apply(cv::UMat &image) override;
  const char *name() const override;

private:
  double strength_;
};

class HSVAdjustFilter : public IFilterStrategy {
public:
  HSVAdjustFilter(double hue, double saturation, double value);
  void apply(cv::UMat &image) override;
  const char *name() const override;

private:
  double hue_;
  double saturation_;
  double value_;
};

class ContrastBrightnessFilter : public IFilterStrategy {
public:
  ContrastBrightnessFilter(double contrast, double brightness);
  void apply(cv::UMat &image) override;
  const char *name() const override;

private:
  double contrast_;
  double brightness_;
};

class EmbossFilter : public IFilterStrategy {
public:
  EmbossFilter() = default;
  void apply(cv::UMat &image) override;
  const char *name() const override;
};

class AdaptiveThresholdFilter : public IFilterStrategy {
public:
  AdaptiveThresholdFilter(int blockSize, int C);
  void apply(cv::UMat &image) override;
  const char *name() const override;

private:
  int blockSize_;
  int C_;
};

// New Filter Strategies:

// Sobel Edge Filter: Computes image gradients using Sobel operators.
class SobelEdgeFilter : public IFilterStrategy {
public:
  SobelEdgeFilter(int ksize = 3);
  void apply(cv::UMat &image) override;
  const char *name() const override;

private:
  int ksize_; // kernel size for Sobel operator, must be odd.
};

// Laplacian Filter: Uses the Laplacian operator to detect edges.
class LaplacianFilter : public IFilterStrategy {
public:
  LaplacianFilter(int ksize = 3);
  void apply(cv::UMat &image) override;
  const char *name() const override;

private:
  int ksize_;
};

// Unsharp Mask Filter: Enhances edges by subtracting a blurred version.
class UnsharpMaskFilter : public IFilterStrategy {
public:
  UnsharpMaskFilter(double strength, int blurKernelSize = 5);
  void apply(cv::UMat &image) override;
  const char *name() const override;

private:
  double strength_;
  int blurKernelSize_;
};

// Utility functions for converting between QImage and cv::Mat / cv::UMat.
namespace ImageUtils {
cv::Mat qtImageToMat(const QImage &img);
QImage matToQtImage(const cv::Mat &mat);

cv::UMat qtImageToUMat(const QImage &img);
QImage umatToQtImage(const cv::UMat &umat);
} // namespace ImageUtils

#endif // FILTER_HPP