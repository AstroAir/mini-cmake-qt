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
};

class ImageCropper {
public:
  explicit ImageCropper(bool enable_gpu = false);
  static bool hasCUDASupport();
  std::optional<cv::Mat> crop(const cv::Mat &src, const CropStrategy &strategy,
                              const AdaptiveParams &adaptive_params = {});
  std::optional<cv::Mat> adaptive_crop(const cv::Mat &src,
                                       const AdaptiveParams &params = {});

private:
  bool use_gpu_;
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
  template <typename T> static constexpr bool always_false_v = false;
};
