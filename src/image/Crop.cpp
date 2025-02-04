#include <opencv2/opencv.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <variant>
#include <vector>

// 检查是否启用CUDA
#ifdef HAVE_OPENCV_CUDA
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

#include "Crop.h"

namespace detail {
// 验证图像有效性
inline bool validate_image(const cv::Mat &img) noexcept {
  if (img.empty()) {
    spdlog::error("Invalid image: empty input");
    return false;
  }
  return true;
}

// 自动调整ROI到图像边界内
cv::Rect adjust_roi(cv::Rect roi, const cv::Size &img_size) {
  roi.x = std::clamp(roi.x, 0, img_size.width);
  roi.y = std::clamp(roi.y, 0, img_size.height);
  roi.width = std::clamp(roi.width, 0, img_size.width - roi.x);
  roi.height = std::clamp(roi.height, 0, img_size.height - roi.y);
  return roi;
}
} // namespace detail

// 扩展裁切策略变体
using CropStrategy =
    std::variant<cv::Rect,                              // 矩形区域
                 std::vector<cv::Point>,                // 多边形区域
                 std::tuple<cv::Point2f, float, float>, // 旋转裁切
                 CircleCrop,                            // 圆形裁切
                 EllipseCrop,                           // 椭圆形裁切
                 RatioCrop>;                            // 按比例裁切

ImageCropper::ImageCropper(bool enable_gpu)
    : use_gpu_(enable_gpu && hasCUDASupport()) {}

static bool hasCUDASupport() {
#ifdef HAVE_OPENCV_CUDA
  return cv::cuda::getCudaEnabledDeviceCount() > 0;
#else
  return false;
#endif
}

// 通用裁切入口
std::optional<cv::Mat>
ImageCropper::crop(const cv::Mat &src, const CropStrategy &strategy,
                   const AdaptiveParams &adaptive_params) {
  if (!detail::validate_image(src))
    return std::nullopt;

  try {
    return std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, cv::Rect>) {
            return crop_rectangular(src, arg);
          } else if constexpr (std::is_same_v<T, std::vector<cv::Point>>) {
            return crop_polygon(src, arg);
          } else if constexpr (std::is_same_v<
                                   T, std::tuple<cv::Point2f, float, float>>) {
            auto &[center, angle, scale] = arg;
            return crop_rotated(src, center, angle, scale);
          } else if constexpr (std::is_same_v<T, CircleCrop>) {
            return crop_circle(src, arg);
          } else if constexpr (std::is_same_v<T, EllipseCrop>) {
            return crop_ellipse(src, arg);
          } else if constexpr (std::is_same_v<T, RatioCrop>) {
            return crop_ratio(src, arg);
          } else {
            static_assert(always_false_v<T>,
                          "Non-exhaustive strategy handling");
          }
        },
        strategy);
  } catch (const cv::Exception &e) {
    spdlog::error("OpenCV exception: {}", e.what());
    return std::nullopt;
  } catch (const std::exception &e) {
    spdlog::error("General exception: {}", e.what());
    return std::nullopt;
  }
}

// 自适应内容感知裁切
std::optional<cv::Mat>
ImageCropper::adaptive_crop(const cv::Mat &src, const AdaptiveParams &params) {
  if (!detail::validate_image(src))
    return std::nullopt;

  try {
    cv::Mat processed;
#ifdef HAVE_OPENCV_CUDA
    if (use_gpu_) {
      cv::cuda::GpuMat gpu_src(src);
      cv::cuda::GpuMat gpu_processed;

      // 创建CUDA流和滤波器
      cv::cuda::Stream stream;
      auto color_cvt = cv::cuda::createColorMap(cv::COLOR_BGR2GRAY);
      auto blur = cv::cuda::createGaussianFilter(
          CV_8UC1, CV_8UC1, cv::Size(params.blurSize, params.blurSize), 0);

      // 执行处理
      color_cvt->apply(gpu_src, gpu_processed, stream);
      blur->apply(gpu_processed, gpu_processed, stream);
      gpu_processed.download(processed);
    } else
#endif
    {
      cv::cvtColor(src, processed, cv::COLOR_BGR2GRAY);
      cv::GaussianBlur(processed, processed,
                       cv::Size(params.blurSize, params.blurSize), 0);
    }

    cv::Canny(processed, processed, params.cannyThreshold1,
              params.cannyThreshold2);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(processed, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
      spdlog::warn("No contours found, returning original image");
      return src.clone();
    }

    auto max_contour = std::max_element(
        contours.begin(), contours.end(), [](const auto &a, const auto &b) {
          return cv::contourArea(a) < cv::contourArea(b);
        });

    cv::Rect bbox = cv::boundingRect(*max_contour);
    bbox += cv::Size(params.margin, params.margin);
    bbox -= cv::Point(params.margin, params.margin);

    return crop_rectangular(src, detail::adjust_roi(bbox, src.size()));
  } catch (const cv::Exception &e) {
    spdlog::error("Adaptive crop failed: {}", e.what());
    return std::nullopt;
  }
}

std::optional<cv::Mat> ImageCropper::crop_rectangular(const cv::Mat &src,
                                                      cv::Rect roi) {
  roi = detail::adjust_roi(roi, src.size());
  if (roi.area() <= 0) {
    spdlog::warn("Empty ROI, returning empty image");
    return cv::Mat();
  }
  return src(roi).clone();
}

std::optional<cv::Mat>
ImageCropper::crop_polygon(const cv::Mat &src,
                           const std::vector<cv::Point> &polygon) {
  if (polygon.size() < 3) {
    spdlog::error("Invalid polygon: requires at least 3 points");
    return std::nullopt;
  }

  cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
  cv::fillConvexPoly(mask, polygon, cv::Scalar(255));

  cv::Mat result;
  src.copyTo(result, mask);
  return result;
}

std::optional<cv::Mat> ImageCropper::crop_rotated(const cv::Mat &src,
                                                  cv::Point2f center,
                                                  float angle, float scale) {
  if (src.empty())
    return std::nullopt;

  cv::Mat rotation_mat = cv::getRotationMatrix2D(center, angle, scale);

  cv::Rect2f bbox = cv::RotatedRect(center, src.size(), angle).boundingRect2f();

  rotation_mat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
  rotation_mat.at<double>(1, 2) += bbox.height / 2.0 - center.y;

  cv::Mat rotated;
  cv::warpAffine(src, rotated, rotation_mat, bbox.size(), cv::INTER_LANCZOS4,
                 cv::BORDER_CONSTANT);

  return rotated;
}

// 新增圆形裁切
std::optional<cv::Mat> ImageCropper::crop_circle(const cv::Mat &src,
                                                 const CircleCrop &params) {
  cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
  cv::circle(mask, params.center, params.radius, cv::Scalar(255), -1);

  cv::Mat result;
  src.copyTo(result, mask);
  return result;
}

// 新增椭圆形裁切
std::optional<cv::Mat> ImageCropper::crop_ellipse(const cv::Mat &src,
                                                  const EllipseCrop &params) {
  cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
  cv::ellipse(mask, params.center, params.axes, params.angle, 0, 360,
              cv::Scalar(255), -1);

  cv::Mat result;
  src.copyTo(result, mask);
  return result;
}

// 新增按比例裁切
std::optional<cv::Mat> ImageCropper::crop_ratio(const cv::Mat &src,
                                                const RatioCrop &params) {
  double curr_ratio = static_cast<double>(src.cols) / src.rows;
  cv::Rect roi;

  if (curr_ratio > params.ratio) {
    // 图像较宽，需要裁剪宽度
    int new_width = static_cast<int>(src.rows * params.ratio);
    int x_offset = (src.cols - new_width) / 2;
    roi = cv::Rect(x_offset, 0, new_width, src.rows);
  } else {
    // 图像较高，需要裁剪高度
    int new_height = static_cast<int>(src.cols / params.ratio);
    int y_offset = (src.rows - new_height) / 2;
    roi = cv::Rect(0, y_offset, src.cols, new_height);
  }

  return crop_rectangular(src, roi);
}