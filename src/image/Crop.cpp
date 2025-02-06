#include <opencv2/opencv.hpp>
#include <optional>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <variant>
#include <vector>

// 检查是否启用CUDA
#ifdef HAVE_OPENCV_CUDA
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

#include "Crop.h"

namespace {
std::shared_ptr<spdlog::logger> cropLogger =
    spdlog::basic_logger_mt("CropLogger", "logs/crop.log");
} // namespace

namespace detail {
// 验证图像有效性
inline bool validate_image(const cv::Mat &img) noexcept {
  if (img.empty()) {
    cropLogger->error("Invalid image: empty input");
    return false;
  }
  cropLogger->debug("Image validation successful");
  return true;
}

// 自动调整ROI到图像边界内
cv::Rect adjust_roi(cv::Rect roi, const cv::Size &img_size) {
  cropLogger->debug("Adjusting ROI to image boundaries");
  roi.x = std::clamp(roi.x, 0, img_size.width);
  roi.y = std::clamp(roi.y, 0, img_size.height);
  roi.width = std::clamp(roi.width, 0, img_size.width - roi.x);
  roi.height = std::clamp(roi.height, 0, img_size.height - roi.y);
  cropLogger->debug("Adjusted ROI: x={}, y={}, width={}, height={}", roi.x,
                    roi.y, roi.width, roi.height);
  return roi;
}
} // namespace detail

ImageCropper::ImageCropper(bool autoAdjust) : autoAdjust(autoAdjust) {
  cropLogger->info("ImageCropper initialized with autoAdjust={}", autoAdjust);
}

bool ImageCropper::hasCUDASupport() {
#ifdef HAVE_OPENCV_CUDA
  bool cudaSupport = cv::cuda::getCudaEnabledDeviceCount() > 0;
  cropLogger->info("CUDA support: {}", cudaSupport);
  return cudaSupport;
#else
  cropLogger->info("CUDA support: disabled (OpenCV CUDA not built)");
  return false;
#endif
}

// 通用裁切入口
std::optional<cv::Mat> ImageCropper::crop(const cv::Mat &image,
                                          const CropStrategy &strategy,
                                          const AdaptiveParams &params) {
  cropLogger->info("Starting crop operation");
  if (!detail::validate_image(image)) {
    cropLogger->warn("Image validation failed, returning nullopt");
    return std::nullopt;
  }

  try {
    return std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, cv::Rect>) {
            cropLogger->info("Using Rect crop strategy");
            return std::optional<cv::Mat>(cropRect(image, arg));
          } else if constexpr (std::is_same_v<T, std::vector<cv::Point>>) {
            cropLogger->info("Using Polygon crop strategy");
            return std::optional<cv::Mat>(cropPolygon(image, arg));
          } else if constexpr (std::is_same_v<
                                   T, std::tuple<cv::Point2f, float, float>>) {
            cropLogger->info("Using Rotated crop strategy");
            auto &[center, angle, scale] = arg;
            return crop_rotated(image, center, angle, scale);
          } else if constexpr (std::is_same_v<T, CircleCrop>) {
            cropLogger->info("Using Circle crop strategy");
            return std::optional<cv::Mat>(cropCircle(image, arg));
          } else if constexpr (std::is_same_v<T, EllipseCrop>) {
            cropLogger->info("Using Ellipse crop strategy");
            return std::optional<cv::Mat>(cropEllipse(image, arg));
          } else if constexpr (std::is_same_v<T, RatioCrop>) {
            cropLogger->info("Using Ratio crop strategy");
            return std::optional<cv::Mat>(cropRatio(image, arg));
          } else {
            static_assert(always_false_v<T>,
                          "Non-exhaustive strategy handling");
            cropLogger->error("Non-exhaustive strategy handling");
            return std::optional<cv::Mat>();
          }
        },
        strategy);
  } catch (const std::exception &e) {
    cropLogger->error("Crop error: {}", e.what());
    return std::nullopt;
  }
}

std::optional<cv::Mat> ImageCropper::crop_rectangular(const cv::Mat &src,
                                                      cv::Rect roi) {
  cropLogger->info("Cropping rectangular with ROI: x={}, y={}, width={}, "
                   "height={}",
                   roi.x, roi.y, roi.width, roi.height);
  roi = detail::adjust_roi(roi, src.size());
  if (roi.area() <= 0) {
    cropLogger->warn("Empty ROI, returning empty image");
    return cv::Mat();
  }
  cv::Mat cropped = src(roi).clone();
  cropLogger->info("Rectangular crop successful: size={}x{}", cropped.cols,
                   cropped.rows);
  return cropped;
}

std::optional<cv::Mat>
ImageCropper::crop_polygon(const cv::Mat &src,
                           const std::vector<cv::Point> &polygon) {
  cropLogger->info("Cropping polygon with {} points", polygon.size());
  if (polygon.size() < 3) {
    cropLogger->error("Invalid polygon: requires at least 3 points");
    return std::nullopt;
  }

  cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
  cv::fillConvexPoly(mask, polygon, cv::Scalar(255));

  cv::Mat result;
  src.copyTo(result, mask);
  cropLogger->info("Polygon crop successful");
  return result;
}

std::optional<cv::Mat> ImageCropper::crop_rotated(const cv::Mat &src,
                                                  cv::Point2f center,
                                                  float angle, float scale) {
  cropLogger->info("Cropping rotated with center=({}, {}), angle={}, scale={}",
                   center.x, center.y, angle, scale);
  if (src.empty()) {
    cropLogger->warn("Source image is empty, returning nullopt");
    return std::nullopt;
  }

  cv::Mat rotation_mat = cv::getRotationMatrix2D(center, angle, scale);

  cv::Rect2f bbox = cv::RotatedRect(center, src.size(), angle).boundingRect2f();

  rotation_mat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
  rotation_mat.at<double>(1, 2) += bbox.height / 2.0 - center.y;

  cv::Mat rotated;
  cv::warpAffine(src, rotated, rotation_mat, bbox.size(), cv::INTER_LANCZOS4,
                 cv::BORDER_CONSTANT);

  cropLogger->info("Rotated crop successful: size={}x{}", rotated.cols,
                   rotated.rows);
  return rotated;
}

// 新增圆形裁切
std::optional<cv::Mat> ImageCropper::crop_circle(const cv::Mat &src,
                                                 const CircleCrop &params) {
  cropLogger->info("Cropping circle with center=({}, {}), radius={}",
                   params.center.x, params.center.y, params.radius);
  cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
  cv::circle(mask, params.center, params.radius, cv::Scalar(255), -1);

  cv::Mat result;
  src.copyTo(result, mask);
  cropLogger->info("Circle crop successful");
  return result;
}

// 新增椭圆形裁切
std::optional<cv::Mat> ImageCropper::crop_ellipse(const cv::Mat &src,
                                                  const EllipseCrop &params) {
  cropLogger->info(
      "Cropping ellipse with center=({}, {}), axes=({}, {}), angle={}",
      params.center.x, params.center.y, params.axes.width, params.axes.height,
      params.angle);
  cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
  cv::ellipse(mask, params.center, params.axes, params.angle, 0, 360,
              cv::Scalar(255), -1);

  cv::Mat result;
  src.copyTo(result, mask);
  cropLogger->info("Ellipse crop successful");
  return result;
}

// 新增按比例裁切
std::optional<cv::Mat> ImageCropper::crop_ratio(const cv::Mat &src,
                                                const RatioCrop &params) {
  cropLogger->info("Cropping ratio with ratio={}", params.ratio);
  double curr_ratio = static_cast<double>(src.cols) / src.rows;
  cv::Rect roi;

  if (curr_ratio > params.ratio) {
    // 图像较宽，需要裁剪宽度
    int new_width = static_cast<int>(src.rows * params.ratio);
    int x_offset = (src.cols - new_width) / 2;
    roi = cv::Rect(x_offset, 0, new_width, src.rows);
    cropLogger->info("Image is wider, cropping width");
  } else {
    // 图像较高，需要裁剪高度
    int new_height = static_cast<int>(src.cols / params.ratio);
    int y_offset = (src.rows - new_height) / 2;
    roi = cv::Rect(0, y_offset, src.cols, new_height);
    cropLogger->info("Image is taller, cropping height");
  }

  return crop_rectangular(src, roi);
}

cv::Mat ImageCropper::cropRect(const cv::Mat &image, const cv::Rect &rect) {
  cropLogger->info("Cropping Rect with x={}, y={}, width={}, height={}", rect.x,
                   rect.y, rect.width, rect.height);
  cv::Rect adjustedRect = rect;
  if (autoAdjust) {
    adjustRectToImage(adjustedRect, image.size());
    cropLogger->info("Auto-adjust enabled, adjusted Rect to x={}, y={}, "
                     "width={}, height={}",
                     adjustedRect.x, adjustedRect.y, adjustedRect.width,
                     adjustedRect.height);
  } else if (!validateRect(rect, image.size())) {
    cropLogger->error("Rect is out of image bounds");
    throw std::runtime_error("裁剪区域超出图像范围");
  }
  cv::Mat cropped = image(adjustedRect).clone();
  cropLogger->info("Rect crop successful: size={}x{}", cropped.cols,
                   cropped.rows);
  return cropped;
}

cv::Mat ImageCropper::cropPolygon(const cv::Mat &image,
                                  const std::vector<cv::Point> &points) {
  cropLogger->info("Cropping Polygon with {} points", points.size());
  cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
  std::vector<std::vector<cv::Point>> contours = {points};
  cv::fillPoly(mask, contours, cv::Scalar(255));

  cv::Mat result;
  image.copyTo(result, mask);

  // 获取多边形的边界框
  cv::Rect boundRect = cv::boundingRect(points);
  cv::Mat cropped = result(boundRect).clone();
  cropLogger->info("Polygon crop successful: size={}x{}", cropped.cols,
                   cropped.rows);
  return cropped;
}

cv::Mat ImageCropper::cropCircle(const cv::Mat &image,
                                 const CircleCrop &circle) {
  cropLogger->info("Cropping Circle with center=({}, {}), radius={}",
                   circle.center.x, circle.center.y, circle.radius);
  cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
  cv::circle(mask, circle.center, circle.radius, cv::Scalar(255), -1);

  cv::Mat result;
  image.copyTo(result, mask);

  // 获取圆形的边界框
  cv::Rect boundRect(circle.center.x - circle.radius,
                     circle.center.y - circle.radius, 2 * circle.radius,
                     2 * circle.radius);
  adjustRectToImage(boundRect, image.size());
  cv::Mat cropped = result(boundRect).clone();
  cropLogger->info("Circle crop successful: size={}x{}", cropped.cols,
                   cropped.rows);
  return cropped;
}

cv::Mat ImageCropper::cropEllipse(const cv::Mat &image,
                                  const EllipseCrop &ellipse) {
  cropLogger->info(
      "Cropping Ellipse with center=({}, {}), axes=({}, {}), angle={}",
      ellipse.center.x, ellipse.center.y, ellipse.axes.width,
      ellipse.axes.height, ellipse.angle);
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
  cv::Mat cropped = result(boundRect).clone();
  cropLogger->info("Ellipse crop successful: size={}x{}", cropped.cols,
                   cropped.rows);
  return cropped;
}

cv::Mat ImageCropper::cropRatio(const cv::Mat &image, const RatioCrop &ratio) {
  cropLogger->info("Cropping Ratio with ratio={}", ratio.ratio);
  double currentRatio = static_cast<double>(image.cols) / image.rows;
  cv::Rect roi;

  if (currentRatio > ratio.ratio) {
    // 图像过宽，需要在宽度上裁剪
    int newWidth = static_cast<int>(image.rows * ratio.ratio);
    int x = (image.cols - newWidth) / 2;
    roi = cv::Rect(x, 0, newWidth, image.rows);
    cropLogger->info("Image is wider, cropping width");
  } else {
    // 图像过高，需要在高度上裁剪
    int newHeight = static_cast<int>(image.cols / ratio.ratio);
    int y = (image.rows - newHeight) / 2;
    roi = cv::Rect(0, y, image.cols, newHeight);
    cropLogger->info("Image is taller, cropping height");
  }

  return this->cropRect(image, roi);
}

bool ImageCropper::validateRect(const cv::Rect &rect,
                                const cv::Size &imageSize) {
  bool isValid = rect.x >= 0 && rect.y >= 0 &&
                 rect.x + rect.width <= imageSize.width &&
                 rect.y + rect.height <= imageSize.height;
  cropLogger->debug("Validating Rect: x={}, y={}, width={}, height={}, "
                    "isValid={}",
                    rect.x, rect.y, rect.width, rect.height, isValid);
  return isValid;
}

void ImageCropper::adjustRectToImage(cv::Rect &rect,
                                     const cv::Size &imageSize) {
  cropLogger->debug("Adjusting Rect to image size: width={}, height={}",
                    imageSize.width, imageSize.height);
  cv::Rect originalRect = rect;
  rect.x = std::max(0, std::min(rect.x, imageSize.width - 1));
  rect.y = std::max(0, std::min(rect.y, imageSize.height - 1));
  rect.width = std::min(rect.width, imageSize.width - rect.x);
  rect.height = std::min(rect.height, imageSize.height - rect.y);
  cropLogger->debug("Original Rect: x={}, y={}, width={}, height={}",
                    originalRect.x, originalRect.y, originalRect.width,
                    originalRect.height);
  cropLogger->debug("Adjusted Rect: x={}, y={}, width={}, height={}", rect.x,
                    rect.y, rect.width, rect.height);
}