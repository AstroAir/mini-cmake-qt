#pragma once

#include <QImage>
#include <concepts>
#include <opencv2/opencv.hpp>
#include <type_traits>


namespace ImageUtils {

// Concepts for type constraints
template <typename T>
concept ImageConvertible =
    std::is_same_v<T, QImage> || std::is_same_v<T, cv::Mat> ||
    std::is_same_v<T, cv::UMat> || std::is_same_v<T, QPixmap>;

// Conversion functions with better error handling
cv::Mat qtImageToMat(const QImage &img, bool deepCopy = false) noexcept(false);
QImage matToQtImage(const cv::Mat &mat) noexcept(false);

cv::UMat qtImageToUMat(const QImage &img) noexcept(false);
QImage umatToQtImage(const cv::UMat &umat) noexcept(false);

// RGB/BGR conversions with parallel processing option
cv::Mat bgrToRgb(const cv::Mat &bgr, bool useParallel = false) noexcept(false);
cv::Mat rgbToBgr(const cv::Mat &rgb, bool useParallel = false) noexcept(false);

// QPixmap conversions with validation
QPixmap matToQPixmap(const cv::Mat &mat) noexcept(false);
cv::Mat qPixmapToMat(const QPixmap &pixmap) noexcept(false);

// Additional format conversions with enhanced robustness
QImage grayToQImage(const cv::Mat &gray) noexcept(false);
cv::Mat qImageToGray(const QImage &img,
                     bool useParallel = false) noexcept(false);

// Image validation functions
bool isValidQImage(const QImage &img) noexcept;
bool isValidMat(const cv::Mat &mat) noexcept;

// Optimized batch processing using C++20 ranges
template <std::ranges::input_range R>
  requires std::convertible_to<std::ranges::range_value_t<R>, cv::Mat>
std::vector<QImage> batchMatToQImage(const R &mats) noexcept(false);

} // namespace ImageUtils