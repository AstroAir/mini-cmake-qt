#pragma once

#include <functional>
#include <opencv2/opencv.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <variant>
#include <vector>

#ifdef HAVE_OPENCV_CUDA
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

struct CircleCrop {
  cv::Point center;
  int radius;
};

struct EllipseCrop {
  cv::Point center;
  cv::Size axes;
  double angle;
};

// 按比例裁剪的参数结构体
struct RatioCrop {
  double ratio; // 期望的宽高比 (width/height)

  RatioCrop(double r) : ratio(r) {
    if (ratio <= 0) {
      throw std::invalid_argument("比例必须大于0");
    }
  }
};

using CropStrategy =
    std::variant<cv::Rect, CircleCrop, EllipseCrop, std::vector<cv::Point>>;

struct AdaptiveParams {
  double cannyThreshold1 = 50;
  double cannyThreshold2 = 150;
  int blurSize = 5;
  int margin = 10;
  bool autoAdjust = true;
  double minArea = 100.0;
  double maxArea = 1000000.0;
};

struct CropperConfig {
  bool useCuda = false;   // 是否使用CUDA加速
  int threads = 4;        // 并行线程数
  bool useCache = true;   // 是否使用缓存
  size_t cacheSize = 100; // 缓存大小(MB)
};

class ImageCropper {
public:
  explicit ImageCropper(const CropperConfig &config = CropperConfig{});
  static bool hasCUDASupport();

  // 公共裁剪方法
  cv::Mat cropRect(const cv::Mat &image, const cv::Rect &rect);
  cv::Mat cropPolygon(const cv::Mat &image,
                      const std::vector<cv::Point> &points);
  cv::Mat cropCircle(const cv::Mat &image, const CircleCrop &circle);
  cv::Mat cropEllipse(const cv::Mat &image, const EllipseCrop &ellipse);
  cv::Mat cropRatio(const cv::Mat &image, const RatioCrop &ratio);

  std::optional<cv::Mat> crop(const cv::Mat &src, const CropStrategy &strategy,
                              const AdaptiveParams &adaptive_params = {});

  std::optional<cv::Mat> cropAuto(const cv::Mat &src);

private:
  CropperConfig config;
  bool autoAdjust;

  // 私有辅助方法
  std::optional<cv::Mat> crop_rectangular(const cv::Mat &src, cv::Rect roi);
  std::optional<cv::Mat> crop_polygon(const cv::Mat &src,
                                      const std::vector<cv::Point> &polygon);
  std::optional<cv::Mat> crop_rotated(const cv::Mat &src, cv::Point2f center,
                                      float angle, float scale = 1.0);
  std::optional<cv::Mat> crop_circle(const cv::Mat &src,
                                     const CircleCrop &params);
  std::optional<cv::Mat> crop_ellipse(const cv::Mat &src,
                                      const EllipseCrop &params);
  std::optional<cv::Mat> crop_ratio(const cv::Mat &src,
                                    const RatioCrop &params);
  std::optional<cv::Mat>
  cropPerspective(const cv::Mat &src,
                  const std::vector<cv::Point2f> &srcPoints);

  std::optional<cv::Mat> cropMasked(const cv::Mat &src, const cv::Mat &mask);

  bool validateRect(const cv::Rect &rect, const cv::Size &imageSize);
  void adjustRectToImage(cv::Rect &rect, const cv::Size &imageSize);

  template <typename T> static constexpr bool always_false_v = false;

  // CUDA优化方法
#ifdef HAVE_OPENCV_CUDA
  cv::Ptr<cv::cuda::Filter> createGaussianFilter();
  cv::cuda::GpuMat preprocessGPU(const cv::cuda::GpuMat &src);
  std::optional<cv::Mat> cropWithCUDA(const cv::Mat &src,
                                      const CropStrategy &strategy);
#endif

  // 缓存相关
  std::unordered_map<size_t, cv::Mat> resultCache;
  void updateCache(size_t hash, const cv::Mat &result);
  std::optional<cv::Mat> getFromCache(size_t hash);

  // 并行处理辅助方法
  cv::Mat parallelProcess(const cv::Mat &src,
                          const std::function<void(cv::Mat &)> &processor);
};
