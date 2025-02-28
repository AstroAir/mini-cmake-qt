#ifndef FILTER_HPP
#define FILTER_HPP

#include <QImage>
#include <atomic>
#include <concepts>
#include <exception>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


// Common exception class used by filter implementations.
class ImageFilterException : public std::exception {
public:
  explicit ImageFilterException(const std::string &msg) noexcept : msg_(msg) {}
  [[nodiscard]] const char *what() const noexcept override {
    return msg_.c_str();
  }

private:
  std::string msg_;
};

// Image concept to validate image types at compile time
template <typename T>
concept ImageType = requires(T img) {
  { img.empty() } -> std::convertible_to<bool>;
  { img.depth() } -> std::convertible_to<int>;
  { img.channels() } -> std::convertible_to<int>;
};

// Base interface for all filter strategies.
class IFilterStrategy {
public:
  virtual ~IFilterStrategy() = default;
  virtual void apply(cv::UMat &image) = 0;
  [[nodiscard]] virtual const char *name() const noexcept = 0;

protected:
  template <ImageType T> void validateImage(const T &img) const {
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
  explicit ImageFilterProcessor(
      std::unique_ptr<IFilterStrategy> &&strategy) noexcept;
  [[nodiscard]] QImage process(const QImage &input);
  [[nodiscard]] std::future<QImage> processAsync(const QImage &input);

private:
  std::unique_ptr<IFilterStrategy> strategy_;
  std::atomic<bool> is_processing_ = false;
};

// Chain processor to apply multiple filters in sequence.
class ChainImageFilterProcessor {
public:
  explicit ChainImageFilterProcessor(
      std::vector<std::unique_ptr<IFilterStrategy>> &&strategies) noexcept;
  [[nodiscard]] QImage process(const QImage &input);
  [[nodiscard]] std::future<QImage> processAsync(const QImage &input);

private:
  std::vector<std::unique_ptr<IFilterStrategy>> strategies_;
  std::atomic<bool> is_processing_ = false;
};

// Existing Filter Strategies:
class GaussianBlurFilter : public IFilterStrategy {
public:
  explicit GaussianBlurFilter(int kernelSize = 3, double sigma = 1.0);
  void apply(cv::UMat &image) override;
  [[nodiscard]] const char *name() const noexcept override;

private:
  int kernelSize_;
  double sigma_;
};

class MedianBlurFilter : public IFilterStrategy {
public:
  explicit MedianBlurFilter(int kernelSize);
  void apply(cv::UMat &image) override;
  [[nodiscard]] const char *name() const noexcept override;

private:
  int kernelSize_;
};

class BilateralFilter : public IFilterStrategy {
public:
  BilateralFilter(int diameter, double sigmaColor, double sigmaSpace);
  void apply(cv::UMat &image) override;
  [[nodiscard]] const char *name() const noexcept override;

private:
  int diameter_;
  double sigmaColor_;
  double sigmaSpace_;
};

class CannyEdgeFilter : public IFilterStrategy {
public:
  explicit CannyEdgeFilter(double threshold1 = 100.0,
                           double threshold2 = 200.0);
  void apply(cv::UMat &image) override;
  [[nodiscard]] const char *name() const noexcept override;

private:
  double threshold1_;
  double threshold2_;
};

class SharpenFilter : public IFilterStrategy {
public:
  explicit SharpenFilter(double strength = 1.0);
  void apply(cv::UMat &image) override;
  [[nodiscard]] const char *name() const noexcept override;

private:
  double strength_;
  cv::UMat weights_lut_; // Pre-computed lookup table for weights
};

class HSVAdjustFilter : public IFilterStrategy {
public:
  HSVAdjustFilter(double hue = 0.0, double saturation = 1.0,
                  double value = 1.0);
  void apply(cv::UMat &image) override;
  [[nodiscard]] const char *name() const noexcept override;

private:
  double hue_;
  double saturation_;
  double value_;
};

class ContrastBrightnessFilter : public IFilterStrategy {
public:
  ContrastBrightnessFilter(double contrast = 1.0, double brightness = 0.0);
  void apply(cv::UMat &image) override;
  [[nodiscard]] const char *name() const noexcept override;

private:
  double contrast_;
  double brightness_;
};

class EmbossFilter : public IFilterStrategy {
public:
  EmbossFilter() = default;
  void apply(cv::UMat &image) override;
  [[nodiscard]] const char *name() const noexcept override;

private:
  cv::UMat kernel_; // Pre-computed kernel
};

class AdaptiveThresholdFilter : public IFilterStrategy {
public:
  AdaptiveThresholdFilter(int blockSize, int C);
  void apply(cv::UMat &image) override;
  [[nodiscard]] const char *name() const noexcept override;

private:
  int blockSize_;
  int C_;
};

// New Filter Strategies:

// Sobel Edge Filter: Computes image gradients using Sobel operators.
class SobelEdgeFilter : public IFilterStrategy {
public:
  explicit SobelEdgeFilter(int ksize = 3);
  void apply(cv::UMat &image) override;
  [[nodiscard]] const char *name() const noexcept override;

private:
  int ksize_; // kernel size for Sobel operator, must be odd.
};

// Laplacian Filter: Uses the Laplacian operator to detect edges.
class LaplacianFilter : public IFilterStrategy {
public:
  explicit LaplacianFilter(int ksize = 3);
  void apply(cv::UMat &image) override;
  [[nodiscard]] const char *name() const noexcept override;

private:
  int ksize_;
};

// Unsharp Mask Filter: Enhances edges by subtracting a blurred version.
class UnsharpMaskFilter : public IFilterStrategy {
public:
  UnsharpMaskFilter(double strength, int blurKernelSize = 5);
  void apply(cv::UMat &image) override;
  [[nodiscard]] const char *name() const noexcept override;

private:
  double strength_;
  int blurKernelSize_;
};

#endif // FILTER_HPP