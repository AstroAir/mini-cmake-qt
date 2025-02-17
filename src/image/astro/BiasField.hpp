#pragma once

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @brief Configuration structure for bias processing.
 */
struct BiasConfig {
  int block_size = 128;           ///< Block size for processing
  bool use_simd = true;           ///< Enable SIMD optimization
  float outlier_threshold = 3.0;  ///< Outlier threshold (sigma)
  size_t cache_size = 1024;       ///< Cache size (KB)
  bool enable_correlation = true; ///< Enable correlation analysis
  bool save_debug_info = false;   ///< Save debug information
  int noise_analysis_bins =
      100; ///< Number of bins for noise analysis histogram
  double quality_threshold = 0.9; ///< Quality score threshold
};

/**
 * @brief Structure to hold quality metrics for bias frames.
 */
struct QualityMetrics {
  double noise_uniformity;   ///< Noise uniformity
  double temporal_stability; ///< Temporal stability
  double spatial_uniformity; ///< Spatial uniformity
  int outlier_count;         ///< Number of outliers
  double overall_score;      ///< Overall quality score

  /**
   * @brief Converts the quality metrics to a string representation.
   * @return A string representation of the quality metrics.
   */
  std::string to_string() const;
};

/**
 * @brief Class for processing bias frames.
 */
class BiasProcessor {
public:
  /**
   * @brief Constructor.
   * @param config Configuration for bias processing.
   */
  explicit BiasProcessor(const BiasConfig &config);

  /**
   * @brief Creates a master bias frame from a set of input bias frames.
   * @param frames Input bias frames.
   * @return The generated master bias frame.
   */
  cv::Mat create_master_bias(const std::vector<cv::Mat> &frames);

  /**
   * @brief Analyzes noise in the master bias frame.
   * @param master The master bias frame.
   * @param frames Input bias frames.
   * @return The quality metrics of the master bias frame.
   */
  QualityMetrics analyze_noise(const cv::Mat &master,
                               const std::vector<cv::Mat> &frames);

  /**
   * @brief Saves the results of the bias processing.
   * @param master The master bias frame.
   * @param output_dir The directory to save the results.
   */
  void save_results(const cv::Mat &master,
                    const std::filesystem::path &output_dir);

private:
  BiasConfig config_;              ///< Configuration for bias processing
  QualityMetrics quality_metrics_; ///< Quality metrics of the master bias frame
  std::vector<cv::Mat> frames_;    ///< 存储输入的偏置帧
  cv::Mat master_;                 ///< 存储主偏置帧
  cv::Mat temporal_std_;           ///< 存储时间标准差
  cv::Mat noise_histogram_;        ///< 存储噪声直方图
  cv::Mat correlation_matrix_;     ///< 存储相关性矩阵

  /**
   * @brief Validates the input bias frames.
   * @param frames Input bias frames.
   * @throws std::runtime_error if the input frames are invalid.
   */
  void validate_input(const std::vector<cv::Mat> &frames);

  /**
   * @brief Processes the input bias frames in blocks.
   * @param frames Input bias frames.
   * @param master The master bias frame.
   */
  void process_blocks(const std::vector<cv::Mat> &frames, cv::Mat &master);

  /**
   * @brief Processes a single block of the input bias frames.
   * @param frames Input bias frames.
   * @param master The master bias frame.
   * @param x The x-coordinate of the block.
   * @param y The y-coordinate of the block.
   * @param block_size The size of the block.
   * @param buffer A buffer for storing pixel values.
   */
  void process_block(const std::vector<cv::Mat> &frames, cv::Mat &master, int x,
                     int y, int block_size, std::vector<float> &buffer);

  /**
   * @brief Computes the median pixel value for a given position.
   * @param frames Input bias frames.
   * @param master The master bias frame.
   * @param x The x-coordinate of the pixel.
   * @param y The y-coordinate of the pixel.
   * @param buffer A buffer for storing pixel values.
   */
  void compute_median_pixel(const std::vector<cv::Mat> &frames, cv::Mat &master,
                            int x, int y, std::vector<float> &buffer);

  /**
   * @brief Evaluates the quality of the master bias frame.
   * @param frames Input bias frames.
   * @param master The master bias frame.
   * @return The quality metrics of the master bias frame.
   */
  QualityMetrics evaluate_quality(const std::vector<cv::Mat> &frames,
                                  const cv::Mat &master);

  /**
   * @brief Computes the temporal stability of the input bias frames.
   * @param frames Input bias frames.
   * @return The temporal stability.
   */
  double compute_temporal_stability(const std::vector<cv::Mat> &frames);

  /**
   * @brief Computes the spatial uniformity of the master bias frame.
   * @param master The master bias frame.
   * @return The spatial uniformity.
   */
  double compute_spatial_uniformity(const cv::Mat &master);

  /**
   * @brief Computes statistical data for the input bias frames.
   * @param frames Input bias frames.
   * @param master The master bias frame.
   * @param variance The variance of the pixel values.
   * @param temporal_std The temporal standard deviation of the pixel values.
   */
  void compute_statistics(const std::vector<cv::Mat> &frames,
                          const cv::Mat &master, cv::Mat &variance,
                          cv::Mat &temporal_std);

  /**
   * @brief Generates a noise report from the variance and temporal standard
   * deviation.
   * @param variance The variance of the pixel values.
   * @param temporal_std The temporal standard deviation of the pixel values.
   */
  void generate_noise_report(const cv::Mat &variance,
                             const cv::Mat &temporal_std);

  /**
   * @brief Computes the correlation matrix for the input bias frames.
   */
  void compute_correlation_matrix();

  /**
   * @brief Saves debug visualizations to the specified output directory.
   * @param output_dir The directory to save the debug visualizations.
   */
  void save_debug_visualizations(const std::filesystem::path &output_dir);

  /**
   * @brief Converts the master bias frame to the original type.
   * @param master The master bias frame.
   * @param original_type The original type of the input bias frames.
   * @return The converted master bias frame.
   */
  cv::Mat convert_output(const cv::Mat &master, int original_type);
};