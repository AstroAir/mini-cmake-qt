#pragma once

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>

namespace fs = std::filesystem;

struct StarDetectionConfig {
  int median_filter_size = 3;
  int wavelet_levels = 4;
  int binarization_threshold = 30;
  int min_star_size = 10;
  int min_star_brightness = 20;
  float min_circularity = 0.7f;
  float max_circularity = 1.3f;
  std::vector<float> scales = {1.0f, 0.75f, 0.5f};
  float dbscan_eps = 10.0f;
  int dbscan_min_samples = 2;
  bool save_detected_stars = false;
  fs::path detected_stars_save_path;
  bool visualize = true;
  fs::path visualization_save_path;
};

class StarDetector {
public:
  explicit StarDetector(StarDetectionConfig config = {});

  std::vector<cv::Point> multiscale_detect_stars(const cv::Mat &input_image);

private:
  StarDetectionConfig config_;

  void validate_parameters() const;
  std::vector<cv::Point> process_scale(const cv::Mat &gray_image,
                                       float scale) const;
  cv::Mat apply_median_filter(const cv::Mat &image) const;
  cv::Mat wavelet_denoising(const cv::Mat &image) const;
  cv::Mat binarize_image(const cv::Mat &image) const;
  std::vector<std::vector<cv::Point>>
  detect_contours(const cv::Mat &binary_image) const;
  std::vector<cv::Point>
  filter_stars(const std::vector<std::vector<cv::Point>> &contours,
               const cv::Mat &binary_image) const;
  std::vector<cv::Point>
  remove_duplicates(const std::vector<cv::Point> &stars) const;
  std::vector<size_t> find_neighbors(const std::vector<cv::Point> &points,
                                     size_t idx) const;
  void save_detected_stars(const std::vector<cv::Point> &stars,
                           const fs::path &path) const;
  void visualize_stars(const cv::Mat &image,
                       const std::vector<cv::Point> &stars) const;
};