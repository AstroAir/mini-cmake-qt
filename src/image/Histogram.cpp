#include "Histogram.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

namespace {
const cv::Scalar DEFAULT_HIST_COLOR(255, 0, 0);
constexpr int DEFAULT_LINE_TYPE = cv::LINE_AA;
constexpr float DEFAULT_THRESHOLD = 4.0f;
} // namespace

auto calculateHist(const cv::Mat &img, const HistogramConfig &config)
    -> std::vector<cv::Mat> {
  spdlog::info("Calculating BGR histograms with histSize: {}", config.histSize);
  if (img.empty()) {
    spdlog::error("Input image for calculateHist is empty.");
    throw std::invalid_argument("Input image for calculateHist is empty.");
  }
  if (img.channels() != 3) {
    spdlog::error("Input image does not have 3 channels.");
    throw std::invalid_argument("Input image does not have 3 channels.");
  }
  if (config.histSize <= 0) {
    spdlog::error("Histogram size must be positive.");
    throw std::invalid_argument("Histogram size must be positive.");
  }

  std::array<cv::Mat, 3> bgrPlanes;
  cv::split(img, bgrPlanes);

  const int dims = 1;
  const std::array<int, 1> channels{0};
  const std::array<float, 2> range{0.0f, 256.0f};
  const float *ranges[] = {range.data()};

  std::vector<cv::Mat> histograms(3);

// 使用OpenMP加速直方图计算
#pragma omp parallel for num_threads(                                          \
        config.numThreads) if (config.numThreads != 1)
  for (int i = 0; i < 3; ++i) {
    try {
      cv::calcHist(&bgrPlanes[i], 1, channels.data(), cv::Mat(), histograms[i],
                   dims, &config.histSize, ranges);

      if (config.threshold > 0) {
        cv::threshold(histograms[i], histograms[i], config.threshold, 0,
                      cv::THRESH_TOZERO);
      }
      if (config.normalize) {
        cv::normalize(histograms[i], histograms[i], 0, 1, cv::NORM_MINMAX);
      }
    } catch (const cv::Exception &e) {
      spdlog::error("OpenCV error in channel {}: {}", i, e.what());
      throw;
    }
  }

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

auto equalizeHistogram(const cv::Mat &img, const EqualizeConfig &config)
    -> cv::Mat {
  spdlog::info("Starting histogram equalization with clip limit: {}",
               config.clipLimit ? "enabled" : "disabled");

  if (img.empty()) {
    spdlog::error("Input image is empty");
    throw std::invalid_argument("Empty input image");
  }

  cv::Mat equalized;
  if (img.channels() == 1) {
    if (config.clipLimit) {
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(config.clipValue);
      clahe->apply(img, equalized);
    } else {
      cv::equalizeHist(img, equalized);
    }
  } else {
    if (config.preserveColor) {
      cv::Mat ycrcb;
      cv::cvtColor(img, ycrcb, cv::COLOR_BGR2YCrCb);
      std::vector<cv::Mat> channels;
      cv::split(ycrcb, channels);

      if (config.clipLimit) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(config.clipValue);
        clahe->apply(channels[0], channels[0]);
      } else {
        cv::equalizeHist(channels[0], channels[0]);
      }

      cv::merge(channels, ycrcb);
      cv::cvtColor(ycrcb, equalized, cv::COLOR_YCrCb2BGR);
    } else {
      std::vector<cv::Mat> channels;
      cv::split(img, channels);

#pragma omp parallel for
      for (int i = 0; i < 3; ++i) {
        if (config.clipLimit) {
          cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(config.clipValue);
          clahe->apply(channels[i], channels[i]);
        } else {
          cv::equalizeHist(channels[i], channels[i]);
        }
      }

      cv::merge(channels, equalized);
    }
  }

  return equalized;
}

auto drawHistogram(const cv::Mat &hist, int width, int height, cv::Scalar color,
                   bool cumulative) -> cv::Mat {
  cv::Mat histImage(height, width, CV_8UC3, cv::Scalar::all(0));

  cv::Mat displayHist;
  if (cumulative) {
    displayHist = calculateCDF(hist);
  } else {
    cv::normalize(hist, displayHist, 0, height, cv::NORM_MINMAX);
  }

  const int binWidth =
      std::max(1, cvRound(static_cast<double>(width) / hist.rows));
  std::vector<cv::Point> points;
  points.reserve(hist.rows + 2);

  points.emplace_back(0, height);
  for (int i = 0; i < hist.rows; ++i) {
    points.emplace_back(binWidth * i,
                        height - cvRound(displayHist.at<float>(i) * height));
  }
  points.emplace_back(width, height);

  const std::vector<std::vector<cv::Point>> contours{points};
  if (cumulative) {
    cv::fillPoly(histImage, contours, color * 0.5);
  }
  cv::polylines(histImage, points, false, color, 2, cv::LINE_AA);

  return histImage;
}

auto compareHistograms(const cv::Mat &hist1, const cv::Mat &hist2, int method)
    -> double {
  if (hist1.empty() || hist2.empty()) {
    throw std::invalid_argument("Empty histogram in comparison");
  }
  return cv::compareHist(hist1, hist2, method);
}

auto calculateHistogramStats(const cv::Mat &hist) -> HistogramStats {
  HistogramStats stats;
  if (hist.empty())
    return stats;

  cv::Mat normHist;
  if (cv::sum(hist)[0] != 1.0) {
    cv::normalize(hist, normHist, 0, 1, cv::NORM_L1);
  } else {
    normHist = hist;
  }

  // 计算均值
  double sum = 0.0;
  for (int i = 0; i < normHist.rows; ++i) {
    sum += i * normHist.at<float>(i);
  }
  stats.mean = sum;

  // 计算高阶矩
  double m2 = 0.0, m3 = 0.0, m4 = 0.0;
  for (int i = 0; i < normHist.rows; ++i) {
    double diff = i - stats.mean;
    double diff2 = diff * diff;
    m2 += diff2 * normHist.at<float>(i);
    m3 += diff * diff2 * normHist.at<float>(i);
    m4 += diff2 * diff2 * normHist.at<float>(i);
  }

  stats.stdDev = std::sqrt(m2);
  stats.skewness = m3 / (std::pow(stats.stdDev, 3));
  stats.kurtosis = (m4 / (m2 * m2)) - 3.0;

  stats.entropy = calculateEntropy(normHist);
  stats.uniformity = calculateUniformity(normHist);

  return stats;
}

auto calculateEntropy(const cv::Mat &hist) -> double {
  double entropy = 0.0;
  for (int i = 0; i < hist.rows; ++i) {
    float p = hist.at<float>(i);
    if (p > 0) {
      entropy -= p * std::log2(p);
    }
  }
  return entropy;
}

auto calculateUniformity(const cv::Mat &hist) -> double {
  double uniformity = 0.0;
  for (int i = 0; i < hist.rows; ++i) {
    float p = hist.at<float>(i);
    uniformity += p * p;
  }
  return uniformity;
}

auto matchHistograms(const cv::Mat &source, const cv::Mat &reference,
                     bool preserveColor) -> cv::Mat {
  if (source.empty() || reference.empty()) {
    throw std::invalid_argument("Empty input image");
  }

  cv::Mat result;
  if (source.channels() == 1 || !preserveColor) {
    cv::Mat srcHist = calculateGrayHist(source);
    cv::Mat refHist = calculateGrayHist(reference);

    cv::Mat srcCdf = calculateCDF(srcHist);
    cv::Mat refCdf = calculateCDF(refHist);

    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
      int j = 0;
      while (j < 256 && refCdf.at<float>(j) <= srcCdf.at<float>(i)) {
        ++j;
      }
      lut.at<uchar>(i) = static_cast<uchar>(j);
    }

    cv::LUT(source, lut, result);
  } else {
    cv::Mat ycrcb, refYcrcb;
    cv::cvtColor(source, ycrcb, cv::COLOR_BGR2YCrCb);
    cv::cvtColor(reference, refYcrcb, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels, refChannels;
    cv::split(ycrcb, channels);
    cv::split(refYcrcb, refChannels);

    // 仅匹配亮度通道
    cv::Mat srcHist = calculateGrayHist(channels[0]);
    cv::Mat refHist = calculateGrayHist(refChannels[0]);

    cv::Mat lut = cv::Mat::zeros(1, 256, CV_8U);
    cv::Mat srcCdf = calculateCDF(srcHist);
    cv::Mat refCdf = calculateCDF(refHist);

    for (int i = 0; i < 256; ++i) {
      int j = 0;
      while (j < 256 && refCdf.at<float>(j) <= srcCdf.at<float>(i)) {
        ++j;
      }
      lut.at<uchar>(i) = static_cast<uchar>(j);
    }

    cv::LUT(channels[0], lut, channels[0]);
    cv::merge(channels, ycrcb);
    cv::cvtColor(ycrcb, result, cv::COLOR_YCrCb2BGR);
  }

  return result;
}

auto backProjectHistogram(const cv::Mat &image, const cv::Mat &hist,
                          const HistogramConfig &config) -> cv::Mat {
  if (image.empty() || hist.empty()) {
    throw std::invalid_argument("Empty input image or histogram");
  }

  cv::Mat result;
  const float *ranges[] = {config.range.data()}; // 使用 config.range.data()
  cv::calcBackProject(&image, 1, &config.channel, hist, result, ranges, 1,
                      true);

  if (config.normalize) {
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
  }

  return result;
}