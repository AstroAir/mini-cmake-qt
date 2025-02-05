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

ImageCropper::ImageCropper(bool autoAdjust) : autoAdjust(autoAdjust) {}

bool ImageCropper::hasCUDASupport() {
#ifdef HAVE_OPENCV_CUDA
  return cv::cuda::getCudaEnabledDeviceCount() > 0;
#else
  return false;
#endif
}

// 通用裁切入口
std::optional<cv::Mat> ImageCropper::crop(const cv::Mat &image,
                                          const CropStrategy &strategy,
                                          const AdaptiveParams &params) {
  if (!detail::validate_image(image))
    return std::nullopt;

  try {
    return std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, cv::Rect>) {
            return std::optional<cv::Mat>(cropRect(image, arg));
          } else if constexpr (std::is_same_v<T, std::vector<cv::Point>>) {
            return std::optional<cv::Mat>(cropPolygon(image, arg));
          } else if constexpr (std::is_same_v<
                                   T, std::tuple<cv::Point2f, float, float>>) {
            auto &[center, angle, scale] = arg;
            return crop_rotated(image, center, angle, scale);
          } else if constexpr (std::is_same_v<T, CircleCrop>) {
            return std::optional<cv::Mat>(cropCircle(image, arg));
          } else if constexpr (std::is_same_v<T, EllipseCrop>) {
            return std::optional<cv::Mat>(cropEllipse(image, arg));
          } else if constexpr (std::is_same_v<T, RatioCrop>) {
            return std::optional<cv::Mat>(cropRatio(image, arg));
          } else {
            static_assert(always_false_v<T>,
                          "Non-exhaustive strategy handling");
            return std::optional<cv::Mat>();
          }
        },
        strategy);
  } catch (const std::exception &e) {
    spdlog::error("Crop error: {}", e.what());
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

cv::Mat ImageCropper::cropRect(const cv::Mat &image, const cv::Rect &rect) {
  cv::Rect adjustedRect = rect;
  if (autoAdjust) {
    adjustRectToImage(adjustedRect, image.size());
  } else if (!validateRect(rect, image.size())) {
    throw std::runtime_error("裁剪区域超出图像范围");
  }
  return image(adjustedRect).clone();
}

cv::Mat ImageCropper::cropPolygon(const cv::Mat &image,
                                  const std::vector<cv::Point> &points) {
  cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
  std::vector<std::vector<cv::Point>> contours = {points};
  cv::fillPoly(mask, contours, cv::Scalar(255));

  cv::Mat result;
  image.copyTo(result, mask);

  // 获取多边形的边界框
  cv::Rect boundRect = cv::boundingRect(points);
  return result(boundRect).clone();
}

cv::Mat ImageCropper::cropCircle(const cv::Mat &image,
                                 const CircleCrop &circle) {
  cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
  cv::circle(mask, circle.center, circle.radius, cv::Scalar(255), -1);

  cv::Mat result;
  image.copyTo(result, mask);

  // 获取圆形的边界框
  cv::Rect boundRect(circle.center.x - circle.radius,
                     circle.center.y - circle.radius, 2 * circle.radius,
                     2 * circle.radius);
  adjustRectToImage(boundRect, image.size());
  return result(boundRect).clone();
}

cv::Mat ImageCropper::cropEllipse(const cv::Mat &image,
                                  const EllipseCrop &ellipse) {
  cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
  cv::ellipse(mask, ellipse.center, ellipse.axes, ellipse.angle, 0, 360,
              cv::Scalar(255), -1);

  cv::Mat result;
  image.copyTo(result, mask);

  // 获取椭圆的边界框
  cv::Rect boundRect(ellipse.center.x - ellipse.axes.width,
                     ellipse.center.y - ellipse.axes.height,
                     2 * ellipse.axes.width, 2 * ellipse.axes.height);
  adjustRectToImage(boundRect, image.size());
  return result(boundRect).clone();
}

cv::Mat ImageCropper::cropRatio(const cv::Mat &image, const RatioCrop &ratio) {
  double currentRatio = static_cast<double>(image.cols) / image.rows;
  cv::Rect roi;

  if (currentRatio > ratio.ratio) {
    // 图像过宽，需要在宽度上裁剪
    int newWidth = static_cast<int>(image.rows * ratio.ratio);
    int x = (image.cols - newWidth) / 2;
    roi = cv::Rect(x, 0, newWidth, image.rows);
  } else {
    // 图像过高，需要在高度上裁剪
    int newHeight = static_cast<int>(image.cols / ratio.ratio);
    int y = (image.rows - newHeight) / 2;
    roi = cv::Rect(0, y, image.cols, newHeight);
  }

  return this->cropRect(image, roi);
}

bool ImageCropper::validateRect(const cv::Rect &rect,
                                const cv::Size &imageSize) {
  return rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= imageSize.width &&
         rect.y + rect.height <= imageSize.height;
}

void ImageCropper::adjustRectToImage(cv::Rect &rect,
                                     const cv::Size &imageSize) {
  rect.x = std::max(0, std::min(rect.x, imageSize.width - 1));
  rect.y = std::max(0, std::min(rect.y, imageSize.height - 1));
  rect.width = std::min(rect.width, imageSize.width - rect.x);
  rect.height = std::min(rect.height, imageSize.height - rect.y);
}