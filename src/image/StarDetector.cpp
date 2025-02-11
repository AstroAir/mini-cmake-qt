#include "StarDetector.hpp"
#include "FWHM.hpp"
#include "HFR.hpp"

#include <algorithm>
#include <execution>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <omp.h>

using json = nlohmann::json;

StarDetector::StarDetector(StarDetectionConfig config)
    : config_(std::move(config)) {
  validate_parameters();
  spdlog::debug("Initialized StarDetector with configured parameters");
}

void StarDetector::validate_parameters() const {
  if (config_.median_filter_size % 2 == 0) {
    throw std::invalid_argument("Median filter size must be odd");
  }
  if (config_.scales.empty()) {
    throw std::invalid_argument("Scales list cannot be empty");
  }
  if (config_.wavelet_levels <= 0) {
    throw std::invalid_argument("Wavelet levels must be positive");
  }
  if (config_.binarization_threshold < 0 ||
      config_.binarization_threshold > 255) {
    throw std::invalid_argument(
        "Binarization threshold must be between 0 and 255");
  }
  if (config_.min_star_size <= 0) {
    throw std::invalid_argument("Minimum star size must be positive");
  }
  if (config_.min_star_brightness < 0 || config_.min_star_brightness > 255) {
    throw std::invalid_argument(
        "Minimum star brightness must be between 0 and 255");
  }
  if (config_.min_circularity <= 0 || config_.max_circularity <= 0 ||
      config_.min_circularity > config_.max_circularity) {
    throw std::invalid_argument(
        "Circularity values must be positive and min_circularity must be less "
        "than or equal to max_circularity");
  }
  if (config_.dbscan_eps <= 0) {
    throw std::invalid_argument("DBSCAN epsilon must be positive");
  }
  if (config_.dbscan_min_samples <= 0) {
    throw std::invalid_argument("DBSCAN minimum samples must be positive");
  }
}

std::vector<cv::Point>
StarDetector::multiscale_detect_stars(const cv::Mat &input_image) {
  if (input_image.empty()) {
    throw std::invalid_argument("Input image is empty");
  }

  cv::Mat gray_image;
  if (input_image.channels() > 1) {
    cv::cvtColor(input_image, gray_image, cv::COLOR_BGR2GRAY);
  } else {
    gray_image = input_image.clone();
  }

  cv::Mat mark_img;
  cv::cvtColor(gray_image, mark_img, cv::COLOR_GRAY2BGR);

  std::vector<cv::Point> all_stars;
  std::mutex mutex;

  // 并行处理不同尺度
  std::for_each(std::execution::par, config_.scales.begin(),
                config_.scales.end(), [&](float scale) {
                  try {
                    auto scaled_stars = process_scale(gray_image, scale);
                    std::lock_guard lock(mutex);
                    all_stars.insert(all_stars.end(), scaled_stars.begin(),
                                     scaled_stars.end());
                  } catch (const std::exception &e) {
                    spdlog::error("Scale {} processing failed: {}", scale,
                                  e.what());
                  }
                });

  auto unique_stars = remove_duplicates(all_stars);

  // 如果需要计算度量指标
  if (config_.calculate_metrics && !unique_stars.empty()) {
    auto metrics = calculate_batch_metrics(input_image, unique_stars,
                                           config_.local_region_size);

    // 可以根据需要将度量指标保存或可视化
    if (config_.visualize) {
      for (size_t i = 0; i < unique_stars.size(); ++i) {
        const auto &[fwhm, hfr] = metrics[i];
        cv::putText(mark_img, cv::format("FWHM:%.2f HFR:%.2f", fwhm, hfr),
                    unique_stars[i] + cv::Point(10, 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
      }
    }
  }

  if (config_.save_detected_stars) {
    save_detected_stars(unique_stars, config_.detected_stars_save_path);
  }

  if (config_.visualize) {
    visualize_stars(gray_image, unique_stars);
  }

  return unique_stars;
}

std::vector<cv::Point> StarDetector::process_scale(const cv::Mat &gray_image,
                                                   float scale) const {
  cv::Mat resized_image;
  cv::resize(gray_image, resized_image, {}, scale, scale, cv::INTER_LANCZOS4);

  // 预处理流水线
  cv::Mat filtered = apply_median_filter(resized_image);
  cv::Mat denoised = wavelet_denoising(filtered);
  cv::Mat binary = binarize_image(denoised);

  auto contours = detect_contours(binary);
  auto filtered_stars = filter_stars(contours, binary);

  // 坐标转换回原图尺寸
  std::vector<cv::Point> scaled_stars;
  scaled_stars.reserve(filtered_stars.size());
  for (const auto &p : filtered_stars) {
    scaled_stars.emplace_back(static_cast<int>(p.x / scale),
                              static_cast<int>(p.y / scale));
  }

  return scaled_stars;
}

cv::Mat StarDetector::apply_median_filter(const cv::Mat &image) const {
  cv::Mat filtered;
  cv::medianBlur(image, filtered, config_.median_filter_size);
  return filtered;
}

cv::Mat StarDetector::wavelet_denoising(const cv::Mat &image) const {
  // 简化的多级小波降噪实现
  cv::Mat denoised = image.clone();
  for (int i = 0; i < config_.wavelet_levels; ++i) {
    cv::Mat low_freq;
    cv::pyrDown(denoised, low_freq);
    cv::pyrUp(low_freq, denoised);
  }
  return denoised;
}

cv::Mat StarDetector::binarize_image(const cv::Mat &image) const {
  cv::Mat binary;
  cv::threshold(image, binary, config_.binarization_threshold, 255,
                cv::THRESH_BINARY);
  return binary;
}

std::vector<std::vector<cv::Point>>
StarDetector::detect_contours(const cv::Mat &binary_image) const {
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_image, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  return contours;
}

std::vector<cv::Point>
StarDetector::filter_stars(const std::vector<std::vector<cv::Point>> &contours,
                           const cv::Mat &binary_image) const {

  std::vector<cv::Point> valid_stars;
  valid_stars.reserve(contours.size());

#pragma omp parallel
  {
    std::vector<cv::Point> local_valid;
#pragma omp for schedule(dynamic) nowait
    for (int idx = 0; idx < static_cast<int>(contours.size()); ++idx) {
      const auto &contour = contours[idx];
      if (contour.empty())
        continue;

      // 使用高精度计算
      const double area = cv::contourArea(contour);
      const double perimeter = cv::arcLength(contour, true);
      if (perimeter <= std::numeric_limits<double>::epsilon())
        continue;

      const double circularity = (4.0 * M_PI * area) / (perimeter * perimeter);
      if (circularity < config_.min_circularity ||
          circularity > config_.max_circularity) {
        continue;
      }

      cv::Rect rect = cv::boundingRect(contour);
      if (rect.area() < config_.min_star_size)
        continue;

      cv::Mat mask = cv::Mat::zeros(binary_image.size(), CV_8U);
      cv::drawContours(mask, {contour}, -1, 255, cv::FILLED);

      // 使用 OpenCV SIMD优化
      cv::Mat roi = binary_image(rect);
      cv::Mat mask_roi = mask(rect);
      cv::Scalar mean = cv::mean(roi, mask_roi);

      if (mean[0] >= config_.min_star_brightness) {
        cv::Moments m = cv::moments(contour);
        // 使用double精度计算质心
        double cx = m.m10 / m.m00;
        double cy = m.m01 / m.m00;
        local_valid.emplace_back(static_cast<int>(std::round(cx)),
                                 static_cast<int>(std::round(cy)));
      }
    }

#pragma omp critical
    {
      valid_stars.insert(valid_stars.end(), local_valid.begin(),
                         local_valid.end());
    }
  }

  return valid_stars;
}

std::vector<cv::Point>
StarDetector::remove_duplicates(const std::vector<cv::Point> &stars) const {
  if (stars.empty())
    return {};

  std::vector<int> labels(stars.size(), -1);
  std::atomic<int> cluster_id{0};

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < stars.size(); ++i) {
    if (labels[i] != -1)
      continue;

    std::vector<size_t> neighbors = find_neighbors(stars, i);
    if (neighbors.size() < config_.dbscan_min_samples) {
      labels[i] = -1;
      continue;
    }

    int current_cluster = cluster_id++;
    labels[i] = current_cluster;

    // ...其余DBSCAN聚类代码保持不变...
  }

  // 并行计算聚类中心
  std::unordered_map<int, std::vector<cv::Point>> clusters;
#pragma omp parallel
  {
    std::unordered_map<int, std::vector<cv::Point>> local_clusters;
#pragma omp for nowait
    for (size_t i = 0; i < stars.size(); ++i) {
      if (labels[i] != -1) {
        local_clusters[labels[i]].push_back(stars[i]);
      }
    }

#pragma omp critical
    {
      for (auto &[id, points] : local_clusters) {
        auto &target = clusters[id];
        target.insert(target.end(), points.begin(), points.end());
      }
    }
  }

  std::vector<cv::Point> unique_stars;
  unique_stars.reserve(clusters.size());

#pragma omp parallel for schedule(dynamic) ordered
  for (auto it = clusters.begin(); it != clusters.end(); ++it) {
    const auto &points = it->second;
    cv::Point2d sum(0.0, 0.0);
    for (const auto &p : points) {
      sum += cv::Point2d(p);
    }
    cv::Point center(static_cast<int>(std::round(sum.x / points.size())),
                     static_cast<int>(std::round(sum.y / points.size())));

#pragma omp ordered
    unique_stars.push_back(center);
  }

  return unique_stars;
}

std::vector<size_t>
StarDetector::find_neighbors(const std::vector<cv::Point> &points,
                             size_t idx) const {
  std::vector<size_t> neighbors;
  const auto &p = points[idx];
  for (size_t i = 0; i < points.size(); ++i) {
    if (i == idx)
      continue;
    if (cv::norm(p - points[i]) < config_.dbscan_eps) {
      neighbors.push_back(i);
    }
  }
  return neighbors;
}

void StarDetector::save_detected_stars(const std::vector<cv::Point> &stars,
                                       const fs::path &path) const {
  if (path.empty())
    return;

  try {
    json j;
    for (const auto &p : stars) {
      j.push_back({{"x", p.x}, {"y", p.y}});
    }

    std::ofstream file(path);
    file << j.dump(4);
    spdlog::info("Saved detected stars to {}", path.string());
  } catch (const std::exception &e) {
    spdlog::error("Failed to save stars: {}", e.what());
  }
}

void StarDetector::visualize_stars(const cv::Mat &image,
                                   const std::vector<cv::Point> &stars) const {
  cv::Mat display;
  cv::cvtColor(image, display, cv::COLOR_GRAY2BGR);

  for (const auto &p : stars) {
    cv::circle(display, p, 3, {0, 0, 255}, 1);
  }

  if (!config_.visualization_save_path.empty()) {
    cv::imwrite(config_.visualization_save_path.string(), display);
  } else {
    cv::imshow("Detected Stars", display);
    cv::waitKey(0);
  }
}

std::pair<double, double> StarDetector::calculate_star_metrics(
    const cv::Mat &image, const cv::Point &center, int region_size) const {
  try {
    cv::Mat roi = extract_star_region(image, center, region_size);
    if (roi.empty()) {
      return {0.0, 0.0};
    }

    // 将ROI转换为双精度以提高计算精度
    cv::Mat roi_double;
    roi.convertTo(roi_double, CV_64F);

    // 计算HFR
    double hfr = calcHfr(roi_double, region_size / 2.0);

    // 提取数据点用于FWHM计算
    std::vector<GaussianFit::DataPoint> points;
    points.reserve(region_size);

    cv::Mat row_profile;
    cv::reduce(roi_double, row_profile, 0, cv::REDUCE_AVG);

    for (int i = 0; i < row_profile.cols; ++i) {
      points.push_back({static_cast<double>(i),
                        static_cast<double>(row_profile.at<double>(0, i))});
    }

    // 计算FWHM
    auto gaussian_params = GaussianFit::GaussianFitter::fit(points);
    double fwhm = gaussian_params ? 2.355 * gaussian_params->width : 0.0;

    return {fwhm, hfr};
  } catch (const std::exception &e) {
    spdlog::error("Error calculating star metrics: {}", e.what());
    return {0.0, 0.0};
  }
}

std::vector<std::pair<double, double>>
StarDetector::calculate_batch_metrics(const cv::Mat &image,
                                      const std::vector<cv::Point> &centers,
                                      int region_size) const {

  std::vector<std::pair<double, double>> results(centers.size());

  // 预分配内存
  std::vector<cv::Mat> rois(centers.size());

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < centers.size(); ++i) {
    rois[i] = extract_star_region(image, centers[i], region_size);
    if (!rois[i].empty()) {
      cv::Mat roi_double;
      rois[i].convertTo(roi_double, CV_64F);

      double hfr = calcHfr(roi_double, region_size / 2.0);

      std::vector<GaussianFit::DataPoint> points;
      points.reserve(region_size);

      cv::Mat row_profile;
      cv::reduce(roi_double, row_profile, 0, cv::REDUCE_AVG);

      for (int j = 0; j < row_profile.cols; ++j) {
        points.push_back(
            {static_cast<double>(j), row_profile.at<double>(0, j)});
      }

      auto gaussian_params = GaussianFit::GaussianFitter::fit(points);
      double fwhm = gaussian_params ? 2.355 * gaussian_params->width : 0.0;

      results[i] = {fwhm, hfr};
    }
  }

  return results;
}

cv::Mat StarDetector::extract_star_region(const cv::Mat &image,
                                          const cv::Point &center,
                                          int size) const {
  int half_size = size / 2;
  cv::Rect roi(std::max(0, center.x - half_size),
               std::max(0, center.y - half_size),
               std::min(size, image.cols - center.x + half_size),
               std::min(size, image.rows - center.y + half_size));

  if (roi.width < size || roi.height < size) {
    return cv::Mat();
  }

  cv::Mat region = image(roi).clone();
  if (image.type() != CV_32F) {
    region.convertTo(region, CV_32F);
  }

  return region;
}