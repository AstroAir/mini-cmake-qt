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
    #pragma omp parallel for num_threads(config.numThreads) if(config.numThreads != 1)
    for (int i = 0; i < 3; ++i) {
        try {
            cv::calcHist(&bgrPlanes[i], 1, channels.data(), cv::Mat(),
                        histograms[i], dims, &config.histSize, ranges);
            
            if (config.threshold > 0) {
                cv::threshold(histograms[i], histograms[i], 
                            config.threshold, 0, cv::THRESH_TOZERO);
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

auto equalizeHistogram(const cv::Mat &img, const EqualizeConfig &config) -> cv::Mat {
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

auto drawHistogram(const cv::Mat &hist, int width, int height,
                  cv::Scalar color, bool cumulative) -> cv::Mat {
    cv::Mat histImage(height, width, CV_8UC3, cv::Scalar::all(0));
    
    cv::Mat displayHist;
    if (cumulative) {
        displayHist = calculateCDF(hist);
    } else {
        cv::normalize(hist, displayHist, 0, height, cv::NORM_MINMAX);
    }

    const int binWidth = std::max(1, cvRound(static_cast<double>(width) / hist.rows));
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

auto compareHistograms(const cv::Mat &hist1, const cv::Mat &hist2, int method) -> double {
    if (hist1.empty() || hist2.empty()) {
        throw std::invalid_argument("Empty histogram in comparison");
    }
    return cv::compareHist(hist1, hist2, method);
}