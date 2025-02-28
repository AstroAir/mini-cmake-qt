#include "ImageUtils.hpp"

#include <QElapsedTimer>
#include <QPixmap>
#include <future>
#include <spdlog/spdlog.h>
#include <stdexcept>


namespace ImageUtils {

bool isValidQImage(const QImage &img) noexcept {
  return !img.isNull() && img.width() > 0 && img.height() > 0 &&
         (img.format() == QImage::Format_RGB32 ||
          img.format() == QImage::Format_RGB888 ||
          img.format() == QImage::Format_ARGB32);
}

bool isValidMat(const cv::Mat &mat) noexcept {
  return !mat.empty() && mat.cols > 0 && mat.rows > 0 &&
         (mat.type() == CV_8UC1 || mat.type() == CV_8UC3 ||
          mat.type() == CV_8UC4);
}

cv::Mat qtImageToMat(const QImage &img, bool deepCopy) noexcept(false) {
  if (!isValidQImage(img)) {
    spdlog::error("Invalid QImage for conversion");
    throw std::invalid_argument("Invalid QImage: null or incorrect format");
  }

  try {
    cv::Mat result(img.height(), img.width(),
                   img.format() == QImage::Format_RGB32 ||
                           img.format() == QImage::Format_ARGB32
                       ? CV_8UC4
                       : CV_8UC3,
                   const_cast<uchar *>(img.bits()),
                   static_cast<size_t>(img.bytesPerLine()));

    // Create a deep copy if requested
    if (deepCopy) {
      return result.clone();
    }
    return result;
  } catch (const std::exception &e) {
    spdlog::error("Image conversion failed: {}", e.what());
    throw std::runtime_error(std::string("Image conversion failed: ") +
                             e.what());
  } catch (...) {
    spdlog::error("Image conversion failed with unknown error");
    throw std::runtime_error("Image conversion failed with unknown error");
  }
}

QImage matToQtImage(const cv::Mat &mat) noexcept(false) {
  if (!isValidMat(mat)) {
    spdlog::error("Invalid Mat for conversion");
    throw std::invalid_argument("Invalid Mat: empty or incorrect type");
  }

  try {
    // Handle different Mat formats appropriately
    QImage::Format format;
    cv::Mat converted;

    if (mat.channels() == 1) {
      // Convert grayscale to RGB for proper display
      cv::cvtColor(mat, converted, cv::COLOR_GRAY2RGB);
      format = QImage::Format_RGB888;
    } else if (mat.channels() == 3) {
      // Convert BGR to RGB if needed
      cv::cvtColor(mat, converted, cv::COLOR_BGR2RGB);
      format = QImage::Format_RGB888;
    } else if (mat.channels() == 4) {
      // Convert BGRA to RGBA
      cv::cvtColor(mat, converted, cv::COLOR_BGRA2RGBA);
      format = QImage::Format_RGBA8888;
    } else {
      throw std::runtime_error("Unsupported number of channels");
    }

    // Create a copy to ensure data ownership
    QImage result(converted.data, converted.cols, converted.rows,
                  static_cast<int>(converted.step), format);
    return result.copy();
  } catch (const std::exception &e) {
    spdlog::error("Mat to QImage conversion failed: {}", e.what());
    throw std::runtime_error(std::string("Mat to QImage conversion failed: ") +
                             e.what());
  } catch (...) {
    spdlog::error("Mat to QImage conversion failed with unknown error");
    throw std::runtime_error(
        "Mat to QImage conversion failed with unknown error");
  }
}

cv::UMat qtImageToUMat(const QImage &img) noexcept(false) {
  try {
    cv::Mat mat =
        qtImageToMat(img, true); // Always deep copy for UMat conversion
    cv::UMat umat;
    mat.copyTo(umat);
    return umat;
  } catch (const std::exception &e) {
    spdlog::error("QImage to UMat conversion failed: {}", e.what());
    throw std::runtime_error(std::string("QImage to UMat conversion failed: ") +
                             e.what());
  }
}

QImage umatToQtImage(const cv::UMat &umat) noexcept(false) {
  if (umat.empty()) {
    spdlog::error("Empty UMat provided for conversion");
    throw std::invalid_argument("Empty UMat cannot be converted");
  }

  try {
    cv::Mat mat;
    umat.copyTo(mat);
    return matToQtImage(mat);
  } catch (const std::exception &e) {
    spdlog::error("UMat to QImage conversion failed: {}", e.what());
    throw std::runtime_error(std::string("UMat to QImage conversion failed: ") +
                             e.what());
  }
}

cv::Mat bgrToRgb(const cv::Mat &bgr, bool useParallel) noexcept(false) {
  if (!isValidMat(bgr)) {
    spdlog::error("Invalid BGR Mat for conversion");
    throw std::invalid_argument("Invalid BGR Mat");
  }

  try {
    cv::Mat rgb;
    if (useParallel &&
        bgr.rows * bgr.cols > 1000000) { // Only use parallel for large images
      // Use OpenCV's parallel processing
      cv::setNumThreads(std::thread::hardware_concurrency());
      cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    } else {
      cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    }
    return rgb;
  } catch (const std::exception &e) {
    spdlog::error("BGR to RGB conversion failed: {}", e.what());
    throw std::runtime_error(std::string("BGR to RGB conversion failed: ") +
                             e.what());
  }
}

cv::Mat rgbToBgr(const cv::Mat &rgb, bool useParallel) noexcept(false) {
  if (!isValidMat(rgb)) {
    spdlog::error("Invalid RGB Mat for conversion");
    throw std::invalid_argument("Invalid RGB Mat");
  }

  try {
    cv::Mat bgr;
    if (useParallel &&
        rgb.rows * rgb.cols > 1000000) { // Only use parallel for large images
      cv::setNumThreads(std::thread::hardware_concurrency());
      cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    } else {
      cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    }
    return bgr;
  } catch (const std::exception &e) {
    spdlog::error("RGB to BGR conversion failed: {}", e.what());
    throw std::runtime_error(std::string("RGB to BGR conversion failed: ") +
                             e.what());
  }
}

QPixmap matToQPixmap(const cv::Mat &mat) noexcept(false) {
  try {
    QImage img = matToQtImage(mat);
    return QPixmap::fromImage(img);
  } catch (const std::exception &e) {
    spdlog::error("Mat to QPixmap conversion failed: {}", e.what());
    throw std::runtime_error(std::string("Mat to QPixmap conversion failed: ") +
                             e.what());
  }
}

cv::Mat qPixmapToMat(const QPixmap &pixmap) noexcept(false) {
  if (pixmap.isNull()) {
    spdlog::error("Null QPixmap provided for conversion");
    throw std::invalid_argument("Null QPixmap cannot be converted");
  }

  try {
    QImage img = pixmap.toImage();
    return qtImageToMat(img, true); // Use deep copy for safety
  } catch (const std::exception &e) {
    spdlog::error("QPixmap to Mat conversion failed: {}", e.what());
    throw std::runtime_error(std::string("QPixmap to Mat conversion failed: ") +
                             e.what());
  }
}

QImage grayToQImage(const cv::Mat &gray) noexcept(false) {
  if (!isValidMat(gray)) {
    spdlog::error("Invalid Mat for grayscale conversion");
    throw std::invalid_argument("Invalid Mat for grayscale conversion");
  }

  try {
    cv::Mat temp;
    if (gray.channels() == 1) {
      cv::cvtColor(gray, temp, cv::COLOR_GRAY2RGB);
      return matToQtImage(temp);
    }

    spdlog::warn(
        "Input Mat is not grayscale, continuing with regular conversion");
    return matToQtImage(gray);
  } catch (const std::exception &e) {
    spdlog::error("Gray to QImage conversion failed: {}", e.what());
    throw std::runtime_error(std::string("Gray to QImage conversion failed: ") +
                             e.what());
  }
}

cv::Mat qImageToGray(const QImage &img, bool useParallel) noexcept(false) {
  if (!isValidQImage(img)) {
    spdlog::error("Invalid QImage for grayscale conversion");
    throw std::invalid_argument("Invalid QImage for grayscale conversion");
  }

  try {
    cv::Mat mat = qtImageToMat(img);
    cv::Mat gray;

    if (useParallel && mat.rows * mat.cols > 1000000) {
      // Use OpenCV's parallel processing
      cv::setNumThreads(std::thread::hardware_concurrency());
      cv::cvtColor(mat, gray, cv::COLOR_RGB2GRAY);
    } else {
      cv::cvtColor(mat, gray, cv::COLOR_RGB2GRAY);
    }

    return gray;
  } catch (const std::exception &e) {
    spdlog::error("QImage to Gray conversion failed: {}", e.what());
    throw std::runtime_error(std::string("QImage to Gray conversion failed: ") +
                             e.what());
  }
}

// Implementation of the batch processing template function
template <std::ranges::input_range R>
  requires std::convertible_to<std::ranges::range_value_t<R>, cv::Mat>
std::vector<QImage> batchMatToQImage(const R &mats) noexcept(false) {
  std::vector<QImage> results;
  results.reserve(std::distance(std::begin(mats), std::end(mats)));

  try {
    // Process in parallel for large batches
    if (std::distance(std::begin(mats), std::end(mats)) > 8) {
      std::vector<std::future<QImage>> futures;
      futures.reserve(std::distance(std::begin(mats), std::end(mats)));

      for (const auto &mat : mats) {
        futures.push_back(std::async(std::launch::async,
                                     [&mat]() { return matToQtImage(mat); }));
      }

      for (auto &future : futures) {
        results.push_back(future.get());
      }
    } else {
      // Process sequentially for small batches
      for (const auto &mat : mats) {
        results.push_back(matToQtImage(mat));
      }
    }

    return results;
  } catch (const std::exception &e) {
    spdlog::error("Batch conversion failed: {}", e.what());
    throw std::runtime_error(std::string("Batch conversion failed: ") +
                             e.what());
  }
}

} // namespace ImageUtils