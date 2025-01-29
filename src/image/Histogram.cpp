#include "Histogram.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <execution>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>


namespace {
const cv::Scalar DEFAULT_HIST_COLOR(255, 0, 0);
constexpr int DEFAULT_LINE_TYPE = cv::LINE_AA;
constexpr float DEFAULT_THRESHOLD = 4.0f;
} // namespace

auto calculateHist(const cv::Mat &img, int histSize, bool normalize,
                   float threshold) -> std::vector<cv::Mat> {
  spdlog::info("Calculating BGR histograms with histSize: {}", histSize);
  if (img.empty()) {
    spdlog::error("Input image for calculateHist is empty.");
    throw std::invalid_argument("Input image for calculateHist is empty.");
  }
  if (img.channels() != 3) {
    spdlog::error("Input image does not have 3 channels.");
    throw std::invalid_argument("Input image does not have 3 channels.");
  }
  if (histSize <= 0) {
    spdlog::error("Histogram size must be positive.");
    throw std::invalid_argument("Histogram size must be positive.");
  }

  std::array<cv::Mat, 3> bgrPlanes;
  cv::split(img, bgrPlanes);

  const std::array<int, 1> channels{0};
  const std::array<float, 2> range{0.0f, static_cast<float>(histSize)};
  const float *ranges[] = {range.data()};
  const int dims = 1;

  std::vector<cv::Mat> histograms(3);
  std::atomic<bool> calcHistFailed{false};

  std::for_each(std::execution::par_unseq, bgrPlanes.begin(), bgrPlanes.end(),
                [&](const cv::Mat &plane) {
                  const int idx = &plane - bgrPlanes.data();
                  try {
                    cv::calcHist(&plane, 1, channels.data(), cv::Mat(),
                                 histograms[idx], dims, &histSize, ranges);
                  } catch (...) {
                    calcHistFailed.store(true, std::memory_order_relaxed);
                  }
                });

  if (calcHistFailed.load(std::memory_order_relaxed)) {
    spdlog::error("Failed to calculate histogram for some channels.");
    throw std::runtime_error("Histogram calculation failed.");
  }

  const auto applyPostProcessing = [&](cv::Mat &hist) {
    if (threshold > 0) {
      cv::threshold(hist, hist, threshold, 0, cv::THRESH_TOZERO);
    }
    if (normalize) {
      cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    }
  };

  std::for_each(std::execution::par_unseq, histograms.begin(), histograms.end(),
                applyPostProcessing);

  spdlog::info("Completed BGR histogram calculation.");
  return histograms;
}

auto calculateGrayHist(const cv::Mat &img, int histSize, bool normalize,
                       float threshold) -> cv::Mat {
  spdlog::info("Calculating grayscale histogram with histSize: {}", histSize);
  if (img.empty()) {
    spdlog::error("Input image for calculateGrayHist is empty.");
    throw std::invalid_argument("Input image for calculateGrayHist is empty.");
  }
  if (img.channels() != 1) {
    spdlog::error("Input image is not grayscale.");
    throw std::invalid_argument("Input image is not grayscale.");
  }
  if (histSize <= 0) {
    spdlog::error("Histogram size must be positive.");
    throw std::invalid_argument("Histogram size must be positive.");
  }

  const std::array<int, 1> channels{0};
  const std::array<float, 2> range{0.0f, static_cast<float>(histSize)};
  const float *ranges[] = {range.data()};

  cv::Mat grayHist;
  cv::calcHist(&img, 1, channels.data(), cv::Mat(), grayHist, 1, &histSize,
               ranges);

  if (threshold > 0) {
    cv::threshold(grayHist, grayHist, threshold, 0, cv::THRESH_TOZERO);
  }
  if (normalize) {
    cv::normalize(grayHist, grayHist, 0, 1, cv::NORM_MINMAX);
  }

  spdlog::info("Completed grayscale histogram calculation.");
  return grayHist;
}

auto calculateCDF(const cv::Mat &hist) -> cv::Mat {
  spdlog::info("Calculating CDF.");
  if (hist.empty()) {
    spdlog::error("Input histogram for calculateCDF is empty.");
    throw std::invalid_argument("Input histogram for calculateCDF is empty.");
  }
  if (hist.rows == 0) {
    spdlog::error("Input histogram has no data.");
    throw std::invalid_argument("Input histogram has no data.");
  }

  cv::Mat cdf;
  hist.copyTo(cdf);

  auto *cdf_ptr = cdf.ptr<float>();
  for (int i = 1; i < hist.rows; ++i) {
    cdf_ptr[i] += cdf_ptr[i - 1];
  }

  cv::normalize(cdf, cdf, 0, 1, cv::NORM_MINMAX);
  spdlog::info("Completed CDF calculation.");
  return cdf;
}

auto equalizeHistogram(const cv::Mat &img, bool preserveColor) -> cv::Mat {
  spdlog::info("Starting histogram equalization.");
  if (img.empty()) {
    spdlog::error("Input image for equalizeHistogram is empty.");
    throw std::invalid_argument("Input image for equalizeHistogram is empty.");
  }

  cv::Mat equalized;
  if (img.channels() == 1) {
    cv::equalizeHist(img, equalized);
  } else {
    if (preserveColor) {
      cv::Mat colorSpace;
      cv::cvtColor(img, colorSpace, cv::COLOR_BGR2YCrCb);
      std::vector<cv::Mat> channels;
      cv::split(colorSpace, channels);
      cv::equalizeHist(channels[0], channels[0]);
      cv::merge(channels, colorSpace);
      cv::cvtColor(colorSpace, equalized, cv::COLOR_YCrCb2BGR);
    } else {
      std::vector<cv::Mat> bgrPlanes;
      cv::split(img, bgrPlanes);
      std::for_each(std::execution::par_unseq, bgrPlanes.begin(),
                    bgrPlanes.end(),
                    [](cv::Mat &plane) { cv::equalizeHist(plane, plane); });
      cv::merge(bgrPlanes, equalized);
    }
  }
  spdlog::info("Completed histogram equalization.");
  return equalized;
}

auto drawHistogram(const cv::Mat &hist, int histSize, int width, int height,
                   cv::Scalar color) -> cv::Mat {
  spdlog::info("Drawing histogram.");
  if (hist.empty()) {
    spdlog::error("Input histogram for drawHistogram is empty.");
    throw std::invalid_argument("Input histogram for drawHistogram is empty.");
  }
  if (width <= 0 || height <= 0) {
    spdlog::error("Invalid output dimensions.");
    throw std::invalid_argument("Invalid output dimensions.");
  }

  cv::Mat histImage(height, width, CV_8UC3, cv::Scalar::all(0));
  cv::Mat histNorm;
  cv::normalize(hist, histNorm, 0, height, cv::NORM_MINMAX);

  const int binWidth = cvRound(static_cast<double>(width) / histSize);
  std::vector<cv::Point> points;
  points.reserve(histSize + 1);

  points.emplace_back(0, height);
  for (int i = 0; i < histSize; ++i) {
    points.emplace_back(binWidth * i, height - cvRound(histNorm.at<float>(i)));
  }
  points.emplace_back(width, height);

  cv::polylines(histImage, points, false, color, 2, DEFAULT_LINE_TYPE);
  spdlog::info("Completed drawing histogram.");
  return histImage;
}