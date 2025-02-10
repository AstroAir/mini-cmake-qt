#include <algorithm>
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

// 构造函数，设置是否自动调整裁剪区域
ImageCropper::ImageCropper(const CropperConfig &config) : config(config) {
  if (config.useCache) {
    resultCache.reserve(100); // 预分配缓存空间
  }

  if (config.useCuda && !hasCUDASupport()) {
    cropLogger->warn("CUDA requested but not available, falling back to CPU");
    this->config.useCuda = false;
  }
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

// 通用裁切入口，根据传入的策略选择不同的裁剪方式
std::optional<cv::Mat> ImageCropper::crop(const cv::Mat &src,
                                          const CropStrategy &strategy,
                                          const AdaptiveParams &params) {
  // 计算输入的哈希值用于缓存
  size_t inputHash = 0;
  if (config.useCache) {
    inputHash = std::hash<std::string>{}(
        std::string(reinterpret_cast<const char *>(src.data),
                    src.total() * src.elemSize()));
    if (auto cached = getFromCache(inputHash)) {
      return cached;
    }
  }

#ifdef HAVE_OPENCV_CUDA
  if (config.useCuda) {
    auto result = cropWithCUDA(src, strategy);
    if (result && config.useCache) {
      updateCache(inputHash, *result);
    }
    return result;
  }
#endif

  // 原有的CPU处理逻辑
  cropLogger->info("Starting crop operation");
  if (!detail::validate_image(src)) {
    cropLogger->warn("Image validation failed, returning nullopt");
    return std::nullopt;
  }

  try {
    auto result = std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, cv::Rect>) {
            cropLogger->info("Using Rect crop strategy");
            return std::optional<cv::Mat>(cropRect(src, arg));
          } else if constexpr (std::is_same_v<T, std::vector<cv::Point>>) {
            cropLogger->info("Using Polygon crop strategy");
            return std::optional<cv::Mat>(cropPolygon(src, arg));
          } else if constexpr (std::is_same_v<
                                   T, std::tuple<cv::Point2f, float, float>>) {
            cropLogger->info("Using Rotated crop strategy");
            auto &[center, angle, scale] = arg;
            return crop_rotated(src, center, angle, scale);
          } else if constexpr (std::is_same_v<T, CircleCrop>) {
            cropLogger->info("Using Circle crop strategy");
            return std::optional<cv::Mat>(cropCircle(src, arg));
          } else if constexpr (std::is_same_v<T, EllipseCrop>) {
            cropLogger->info("Using Ellipse crop strategy");
            return std::optional<cv::Mat>(cropEllipse(src, arg));
          } else if constexpr (std::is_same_v<T, RatioCrop>) {
            cropLogger->info("Using Ratio crop strategy");
            return std::optional<cv::Mat>(cropRatio(src, arg));
          } else {
            static_assert(always_false_v<T>,
                          "Non-exhaustive strategy handling");
            cropLogger->error("Non-exhaustive strategy handling");
            return std::optional<cv::Mat>();
          }
        },
        strategy);
    if (result && config.useCache) {
      updateCache(inputHash, *result);
    }
    return result;
  } catch (const std::exception &e) {
    cropLogger->error("Crop error: {}", e.what());
    return std::nullopt;
  }
}

std::optional<cv::Mat> ImageCropper::crop_rectangular(const cv::Mat &src,
                                                      cv::Rect roi) {
  cropLogger->info(
      "Cropping rectangular with ROI: x={}, y={}, width={}, height={}", roi.x,
      roi.y, roi.width, roi.height);
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

std::optional<cv::Mat> ImageCropper::crop_ratio(const cv::Mat &src,
                                                const RatioCrop &params) {
  cropLogger->info("Cropping ratio with ratio={}", params.ratio);
  double curr_ratio = static_cast<double>(src.cols) / src.rows;
  cv::Rect roi;

  if (curr_ratio > params.ratio) {
    int new_width = static_cast<int>(src.rows * params.ratio);
    int x_offset = (src.cols - new_width) / 2;
    roi = cv::Rect(x_offset, 0, new_width, src.rows);
    cropLogger->info("Image is wider, cropping width");
  } else {
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
    int newWidth = static_cast<int>(image.rows * ratio.ratio);
    int x = (image.cols - newWidth) / 2;
    roi = cv::Rect(x, 0, newWidth, image.rows);
    cropLogger->info("Image is wider, cropping width");
  } else {
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
  cropLogger->debug(
      "Validating Rect: x={}, y={}, width={}, height={}, isValid={}", rect.x,
      rect.y, rect.width, rect.height, isValid);
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

std::optional<cv::Mat>
ImageCropper::cropPerspective(const cv::Mat &src,
                              const std::vector<cv::Point2f> &srcPoints) {
  cropLogger->info("Cropping perspective with {} points", srcPoints.size());
  if (srcPoints.size() != 4) {
    cropLogger->error("Perspective crop requires exactly 4 points");
    return std::nullopt;
  }

  // 计算目标矩形大小（宽高）——假设为最终的裁切区域
  double widthA = cv::norm(srcPoints[0] - srcPoints[1]);
  double widthB = cv::norm(srcPoints[2] - srcPoints[3]);
  double maxWidth = std::max(widthA, widthB);

  double heightA = cv::norm(srcPoints[0] - srcPoints[3]);
  double heightB = cv::norm(srcPoints[1] - srcPoints[2]);
  double maxHeight = std::max(heightA, heightB);

  std::vector<cv::Point2f> dstPoints = {
      cv::Point2f(0, 0), cv::Point2f(static_cast<float>(maxWidth - 1), 0),
      cv::Point2f(static_cast<float>(maxWidth - 1),
                  static_cast<float>(maxHeight - 1)),
      cv::Point2f(0, static_cast<float>(maxHeight - 1))};

  cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);
  cv::Mat warped;
  cv::warpPerspective(
      src, warped, M,
      cv::Size(static_cast<int>(maxWidth), static_cast<int>(maxHeight)),
      cv::INTER_LINEAR);
  cropLogger->info("Perspective crop successful: size={}x{}", warped.cols,
                   warped.rows);
  return warped;
}

std::optional<cv::Mat> ImageCropper::cropMasked(const cv::Mat &src,
                                                const cv::Mat &mask) {
  cropLogger->info("Cropping using user provided mask");
  if (mask.empty() || mask.size() != src.size() || mask.type() != CV_8UC1) {
    cropLogger->error("Invalid mask provided for cropping");
    return std::nullopt;
  }
  cv::Mat result;
  src.copyTo(result, mask);

  // 计算掩模非零区域的最小外接矩形
  cv::Rect roi = cv::boundingRect(mask);
  if (roi.area() <= 0) {
    cropLogger->warn("Mask has no valid region, returning empty image");
    return cv::Mat();
  }
  cv::Mat cropped = result(roi).clone();
  cropLogger->info(
      "Masked crop successful: ROI x={}, y={}, width={}, height={}", roi.x,
      roi.y, roi.width, roi.height);
  return cropped;
}

std::optional<cv::Mat> ImageCropper::cropAuto(const cv::Mat &src) {
  cropLogger->info("Performing auto crop using edge detection and contours");
  if (!detail::validate_image(src))
    return std::nullopt;

  cv::Mat gray, blurred, edged;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
  cv::Canny(blurred, edged, 50, 150);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(edged, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  if (contours.empty()) {
    cropLogger->error("No contours detected for auto cropping");
    return std::nullopt;
  }

  // 找到最大的轮廓
  auto maxContour = std::max_element(
      contours.begin(), contours.end(),
      [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
        return cv::contourArea(a) < cv::contourArea(b);
      });

  cv::Rect roi = cv::boundingRect(*maxContour);
  cropLogger->info("Auto crop detected ROI: x={}, y={}, width={}, height={}",
                   roi.x, roi.y, roi.width, roi.height);
  if (roi.area() <= 0) {
    cropLogger->warn("Auto crop ROI is empty, returning empty image");
    return cv::Mat();
  }
  cv::Mat cropped = src(roi).clone();
  cropLogger->info("Auto crop successful: size={}x{}", cropped.cols,
                   cropped.rows);
  return cropped;
}

#ifdef HAVE_OPENCV_CUDA
cv::Ptr<cv::cuda::Filter> ImageCropper::createGaussianFilter() {
  static cv::Ptr<cv::cuda::Filter> gaussianFilter =
      cv::cuda::createGaussianFilter(CV_8UC3, CV_8UC3, cv::Size(5, 5), 0);
  return gaussianFilter;
}

cv::cuda::GpuMat ImageCropper::preprocessGPU(const cv::cuda::GpuMat &src) {
  cv::cuda::GpuMat result;
  auto filter = createGaussianFilter();
  filter->apply(src, result);
  return result;
}

std::optional<cv::Mat>
ImageCropper::cropWithCUDA(const cv::Mat &src, const CropStrategy &strategy) {
  try {
    cv::cuda::GpuMat gpuSrc(src);
    cv::cuda::GpuMat gpuResult = preprocessGPU(gpuSrc);

    // 根据不同的裁剪策略使用CUDA加速
    std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, cv::Rect>) {
            gpuResult = gpuResult(arg);
          } else if constexpr (std::is_same_v<T, CircleCrop>) {
            cv::cuda::GpuMat mask;
            // 在GPU上创建圆形mask
            // ...
          }
          // ... 其他策略的CUDA实现
        },
        strategy);

    cv::Mat result;
    gpuResult.download(result);
    return result;
  } catch (const cv::Exception &e) {
    cropLogger->error("CUDA processing failed: {}", e.what());
    return std::nullopt;
  }
}
#endif

void ImageCropper::updateCache(size_t hash, const cv::Mat &result) {
  if (resultCache.size() >= config.cacheSize) {
    resultCache.erase(resultCache.begin()); // 移除最旧的缓存
  }
  resultCache[hash] = result.clone();
}

std::optional<cv::Mat> ImageCropper::getFromCache(size_t hash) {
  auto it = resultCache.find(hash);
  if (it != resultCache.end()) {
    return it->second.clone();
  }
  return std::nullopt;
}

cv::Mat
ImageCropper::parallelProcess(const cv::Mat &src,
                              const std::function<void(cv::Mat &)> &processor) {
  cv::Mat result = src.clone();

  int rows_per_thread = src.rows / config.threads;
  std::vector<std::thread> threads;

  for (int i = 0; i < config.threads; ++i) {
    int start_row = i * rows_per_thread;
    int end_row =
        (i == config.threads - 1) ? src.rows : start_row + rows_per_thread;

    threads.emplace_back([&, start_row, end_row]() {
      cv::Mat region = result(cv::Range(start_row, end_row), cv::Range::all());
      processor(region);
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  return result;
}