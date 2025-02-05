#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <spdlog/spdlog.h>
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

struct RatioCrop {
  double ratio;
  bool keepLargest = true;
};

using CropStrategy = std::variant<cv::Rect, std::vector<cv::Point>,
                                  std::tuple<cv::Point2f, float, float>,
                                  CircleCrop, EllipseCrop, RatioCrop>;

struct AdaptiveParams {
  double cannyThreshold1 = 50;
  double cannyThreshold2 = 150;
  int blurSize = 5;
  int margin = 10;
  bool autoAdjust = true;
  double minArea = 100.0;
  double maxArea = 1000000.0;
};

class ImageCropper {
public:
  explicit ImageCropper(bool autoAdjust = false);
  static bool hasCUDASupport();
  
  // 公共裁剪方法
  cv::Mat cropRect(const cv::Mat& image, const cv::Rect& rect);
  cv::Mat cropPolygon(const cv::Mat& image, const std::vector<cv::Point>& points);
  cv::Mat cropCircle(const cv::Mat& image, const CircleCrop& circle);
  cv::Mat cropEllipse(const cv::Mat& image, const EllipseCrop& ellipse);
  cv::Mat cropRatio(const cv::Mat& image, const RatioCrop& ratio);

  std::optional<cv::Mat> crop(const cv::Mat &src, const CropStrategy &strategy,
                             const AdaptiveParams &adaptive_params = {});

private:
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

  bool validateRect(const cv::Rect& rect, const cv::Size& imageSize);
  void adjustRectToImage(cv::Rect& rect, const cv::Size& imageSize);
  
  template <typename T> static constexpr bool always_false_v = false;
};
