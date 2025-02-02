#pragma once

#include <QImage>
#include <concepts>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ImageUtils {
cv::Mat qtImageToMat(const QImage &img);
QImage matToQtImage(const cv::Mat &mat);
}

// 使用C++20 concept定义滤镜策略约束
template <typename T>
concept FilterStrategy = requires(T t, cv::Mat &img) {
  { t.apply(img) } -> std::same_as<void>;
  { T::name() } -> std::same_as<const char *>;
};

// 异常安全基类
class ImageFilterException : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// 滤镜策略接口
class IFilterStrategy {
public:
  virtual ~IFilterStrategy() = default;
  virtual void apply(cv::Mat &image) = 0;
  virtual const char *name() const = 0;

protected:
  void validateImage(const cv::Mat &img) const;
};

// 图像处理器类
class ImageFilterProcessor {
public:
  explicit ImageFilterProcessor(std::unique_ptr<IFilterStrategy> &&strategy);
  QImage process(const QImage &input);

private:
  std::unique_ptr<IFilterStrategy> strategy_;
};

// 滤镜链式处理器
class ChainImageFilterProcessor {
public:
  explicit ChainImageFilterProcessor(
      std::vector<std::unique_ptr<IFilterStrategy>> &&strategies);
  QImage process(const QImage &input);

private:
  std::vector<std::unique_ptr<IFilterStrategy>> strategies_;
};

// 具体滤镜类声明
class GaussianBlurFilter : public IFilterStrategy {
public:
  explicit GaussianBlurFilter(int kernelSize = 5, double sigma = 1.5);
  void apply(cv::Mat &image) override;
  const char *name() const override;

private:
  int kernelSize_;
  double sigma_;
};

class MedianBlurFilter : public IFilterStrategy {
public:
  explicit MedianBlurFilter(int kernelSize = 5);
  void apply(cv::Mat &image) override;
  const char *name() const override;

private:
  int kernelSize_;
};

class BilateralFilter : public IFilterStrategy {
public:
  BilateralFilter(int diameter = 9, double sigmaColor = 75.0,
                  double sigmaSpace = 75.0);
  void apply(cv::Mat &image) override;
  const char *name() const override;

private:
  int diameter_;
  double sigmaColor_;
  double sigmaSpace_;
};

// 边缘检测滤镜
class CannyEdgeFilter : public IFilterStrategy {
public:
  CannyEdgeFilter(double threshold1 = 100, double threshold2 = 200);
  void apply(cv::Mat &image) override;
  const char *name() const override;

private:
  double threshold1_;
  double threshold2_;
};

// 图像锐化滤镜
class SharpenFilter : public IFilterStrategy {
public:
  explicit SharpenFilter(double strength = 1.0);
  void apply(cv::Mat &image) override;
  const char *name() const override;

private:
  double strength_;
};

// 色相饱和度滤镜
class HSVAdjustFilter : public IFilterStrategy {
public:
  HSVAdjustFilter(double hue = 0, double saturation = 1.0, double value = 1.0);
  void apply(cv::Mat &image) override;
  const char *name() const override;

private:
  double hue_;
  double saturation_;
  double value_;
};

// 对比度亮度滤镜
class ContrastBrightnessFilter : public IFilterStrategy {
public:
  ContrastBrightnessFilter(double contrast = 1.0, double brightness = 0);
  void apply(cv::Mat &image) override;
  const char *name() const override;

private:
  double contrast_;
  double brightness_;
};
