#pragma once

#include "FWHM.hpp"
#include <filesystem>
#include <vector>

namespace cv {
    template <typename T> class Point_;
    typedef Point_<int> Point;
    class Mat;
}

namespace fs = std::filesystem;

/**
 * @struct StarDetectionConfig
 * @brief Configuration structure for star detection.
 */
struct StarDetectionConfig {
  int median_filter_size = 3;      ///< Size of the median filter.
  int wavelet_levels = 4;          ///< Number of wavelet levels for denoising.
  int binarization_threshold = 30; ///< Threshold for binarization.
  int min_star_size = 10;          ///< Minimum size of detected stars.
  int min_star_brightness = 20;    ///< Minimum brightness of detected stars.
  float min_circularity = 0.7f;    ///< Minimum circularity of detected stars.
  float max_circularity = 1.3f;    ///< Maximum circularity of detected stars.
  std::vector<float> scales = {1.0f, 0.75f,
                               0.5f}; ///< Scales for multiscale detection.
  float dbscan_eps = 10.0f; ///< Epsilon parameter for DBSCAN clustering.
  int dbscan_min_samples =
      2; ///< Minimum samples parameter for DBSCAN clustering.
  bool save_detected_stars = false;  ///< Flag to save detected stars.
  fs::path detected_stars_save_path; ///< Path to save detected stars.
  bool visualize = true;             ///< Flag to visualize detected stars.
  fs::path visualization_save_path;  ///< Path to save visualization.
  int local_region_size = 32;     ///< Size of local region for metrics calculation
  bool calculate_metrics = true;   ///< Whether to calculate FWHM and HFR
  bool use_gaussian_fit = true;     ///< 是否使用高斯拟合进行星点分析
  double fit_convergence = 1e-6;    ///< 高斯拟合收敛阈值
  int fit_max_iterations = 100;     ///< 高斯拟合最大迭代次数
  bool parallel_processing = true;   ///< 是否启用并行处理
  int block_size = 16;              ///< 并行处理块大小
};

/**
 * @class StarDetector
 * @brief Class for detecting stars in astronomical images.
 */
class StarDetector {
public:
  /**
   * @brief Constructor for StarDetector.
   * @param config The configuration for star detection.
   */
  explicit StarDetector(StarDetectionConfig config = {});

  /**
   * @brief Detects stars in an image using multiscale detection.
   * @param input_image The input image.
   * @return A vector of points representing the detected stars.
   */
  std::vector<cv::Point> multiscale_detect_stars(const cv::Mat &input_image);

  /**
   * @brief Calculates metrics (FWHM & HFR) for a star at specific location
   * @param image Input image
   * @param center Star center position
   * @param region_size Size of the region around the star
   * @return Pair of FWHM and HFR values
   */
  std::pair<double, double> calculate_star_metrics(const cv::Mat& image, 
                                                   const cv::Point& center,
                                                   int region_size) const;

  /**
   * @brief Batch calculates metrics for multiple stars
   * @param image Input image
   * @param centers Vector of star centers
   * @param region_size Size of the region around each star
   * @return Vector of FWHM and HFR pairs
   */
  std::vector<std::pair<double, double>> calculate_batch_metrics(
      const cv::Mat& image,
      const std::vector<cv::Point>& centers,
      int region_size) const;

  /**
   * @brief 计算星点的综合质量指标
   * @param image 输入图像
   * @param center 星点中心
   * @param region_size 区域大小
   * @return 包含FWHM、HFR和高斯拟合参数的结构
   */
  struct StarMetrics {
    double fwhm;
    double hfr;
    std::optional<GaussianFit::GaussianParams> gaussian_params;
    double quality_score;
  };

  StarMetrics calculate_star_quality(const cv::Mat& image,
                                    const cv::Point& center,
                                    int region_size) const;

private:
  StarDetectionConfig config_; ///< Configuration for star detection.

  /**
   * @brief Validates the configuration parameters.
   */
  void validate_parameters() const;

  /**
   * @brief Processes the image at a given scale.
   * @param gray_image The grayscale image.
   * @param scale The scale to process the image at.
   * @return A vector of points representing the detected stars at the given
   * scale.
   */
  std::vector<cv::Point> process_scale(const cv::Mat &gray_image,
                                       float scale) const;

  /**
   * @brief Applies a median filter to the image.
   * @param image The input image.
   * @return The filtered image.
   */
  cv::Mat apply_median_filter(const cv::Mat &image) const;

  /**
   * @brief Applies wavelet denoising to the image.
   * @param image The input image.
   * @return The denoised image.
   */
  cv::Mat wavelet_denoising(const cv::Mat &image) const;

  /**
   * @brief Binarizes the image.
   * @param image The input image.
   * @return The binarized image.
   */
  cv::Mat binarize_image(const cv::Mat &image) const;

  /**
   * @brief Detects contours in the binary image.
   * @param binary_image The binary image.
   * @return A vector of contours detected in the image.
   */
  std::vector<std::vector<cv::Point>>
  detect_contours(const cv::Mat &binary_image) const;

  /**
   * @brief Filters the detected stars based on size, brightness, and
   * circularity.
   * @param contours The detected contours.
   * @param binary_image The binary image.
   * @return A vector of points representing the filtered stars.
   */
  std::vector<cv::Point>
  filter_stars(const std::vector<std::vector<cv::Point>> &contours,
               const cv::Mat &binary_image) const;

  /**
   * @brief Removes duplicate star detections.
   * @param stars The detected stars.
   * @return A vector of points representing the unique stars.
   */
  std::vector<cv::Point>
  remove_duplicates(const std::vector<cv::Point> &stars) const;

  /**
   * @brief Finds neighbors of a point using DBSCAN clustering.
   * @param points The points to cluster.
   * @param idx The index of the point to find neighbors for.
   * @return A vector of indices representing the neighbors of the point.
   */
  std::vector<size_t> find_neighbors(const std::vector<cv::Point> &points,
                                     size_t idx) const;

  /**
   * @brief Saves the detected stars to a file.
   * @param stars The detected stars.
   * @param path The path to save the stars.
   */
  void save_detected_stars(const std::vector<cv::Point> &stars,
                           const fs::path &path) const;

  /**
   * @brief Visualizes the detected stars on the image.
   * @param image The input image.
   * @param stars The detected stars.
   */
  void visualize_stars(const cv::Mat &image,
                       const std::vector<cv::Point> &stars) const;

  /**
   * @brief Extracts region of interest around a star
   * @param image Input image
   * @param center Star center
   * @param size Region size
   * @return ROI as Mat
   */
  cv::Mat extract_star_region(const cv::Mat& image, 
                              const cv::Point& center,
                              int size) const;

  /**
   * @brief 计算星点质量的综合得分
   */
  double calculate_quality_score(const StarMetrics& metrics) const;
  
  /**
   * @brief 使用高斯拟合优化星点检测
   */
  std::vector<cv::Point> refine_star_positions(
      const cv::Mat& image,
      const std::vector<cv::Point>& initial_positions) const;
};