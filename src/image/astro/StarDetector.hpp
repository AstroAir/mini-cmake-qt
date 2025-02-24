#pragma once

#include "FWHM.hpp"
#include <filesystem>
#include <vector>

namespace cv {
template <typename T> class Point_;
typedef Point_<int> Point;
class Mat;
} // namespace cv

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
  int local_region_size = 32; ///< Size of local region for metrics calculation.
  bool calculate_metrics = true; ///< Whether to calculate FWHM and HFR.
  bool use_gaussian_fit =
      true; ///< Whether to use Gaussian fit for star analysis.
  double fit_convergence = 1e-6;   ///< Convergence threshold for Gaussian fit.
  int fit_max_iterations = 100;    ///< Maximum iterations for Gaussian fit.
  bool parallel_processing = true; ///< Whether to enable parallel processing.
  int block_size = 16;             ///< Block size for parallel processing.
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
   * @brief Calculates metrics (FWHM & HFR) for a star at a specific location.
   * @param image The input image.
   * @param center The star center position.
   * @param region_size The size of the region around the star.
   * @return A pair of FWHM and HFR values.
   */
  std::pair<double, double> calculate_star_metrics(const cv::Mat &image,
                                                   const cv::Point &center,
                                                   int region_size) const;

  /**
   * @brief Batch calculates metrics for multiple stars.
   * @param image The input image.
   * @param centers A vector of star centers.
   * @param region_size The size of the region around each star.
   * @return A vector of FWHM and HFR pairs.
   */
  std::vector<std::pair<double, double>>
  calculate_batch_metrics(const cv::Mat &image,
                          const std::vector<cv::Point> &centers,
                          int region_size) const;

  /**
   * @brief Calculates the comprehensive quality metrics for a star.
   * @param image The input image.
   * @param center The star center position.
   * @param region_size The size of the region around the star.
   * @return A structure containing FWHM, HFR, Gaussian fit parameters, and
   * quality score.
   */
  struct StarMetrics {
    double fwhm; ///< Full Width at Half Maximum.
    double hfr;  ///< Half-Flux Radius.
    std::optional<GaussianFit::GaussianParams>
        gaussian_params;  ///< Gaussian fit parameters.
    double quality_score; ///< Quality score of the star.
  };

  StarMetrics calculate_star_quality(const cv::Mat &image,
                                     const cv::Point &center,
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
   * @brief Extracts the region of interest around a star.
   * @param image The input image.
   * @param center The star center position.
   * @param size The size of the region.
   * @return The region of interest as a Mat.
   */
  cv::Mat extract_star_region(const cv::Mat &image, const cv::Point &center,
                              int size) const;

  /**
   * @brief Calculates the comprehensive quality score of a star.
   * @param metrics The star metrics.
   * @return The quality score.
   */
  double calculate_quality_score(const StarMetrics &metrics) const;

  /**
   * @brief Refines star positions using Gaussian fit.
   * @param image The input image.
   * @param initial_positions The initial star positions.
   * @return A vector of refined star positions.
   */
  std::vector<cv::Point>
  refine_star_positions(const cv::Mat &image,
                        const std::vector<cv::Point> &initial_positions) const;
};