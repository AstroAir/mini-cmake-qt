#pragma once

#include "ParallelConfig.hpp"
#include <filesystem>
#include <functional>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

/**
 * @struct TemperatureData
 * @brief Structure representing temperature-related data for dark field
 * processing.
 */
struct TemperatureData {
  double temperature;   ///< Temperature value (Celsius).
  double dark_current;  ///< Dark current value.
  double exposure_time; ///< Exposure time (seconds).
};

/**
 * @struct TemperatureCompensation
 * @brief Structure representing temperature compensation parameters.
 */
struct TemperatureCompensation {
  double baseline_temp;         ///< Baseline temperature.
  double temp_coefficient;      ///< Temperature coefficient.
  double dark_current_baseline; ///< Baseline dark current.
};

/**
 * @struct DefectDetectionConfig
 * @brief Configuration structure for defect detection algorithms.
 */
struct DefectDetectionConfig {
  /**
   * @enum Algorithm
   * @brief Enumeration of defect detection algorithms.
   */
  enum class Algorithm {
    THRESHOLD,   ///< Threshold method.
    STATISTICAL, ///< Statistical method.
    PATTERN,     ///< Pattern recognition method.
    HYBRID       ///< Hybrid method.
  };

  Algorithm method =
      Algorithm::HYBRID;       ///< Selected defect detection algorithm.
  int window_size = 5;         ///< Detection window size.
  float confidence = 0.95f;    ///< Confidence level.
  int min_cluster = 3;         ///< Minimum cluster size.
  bool detect_clusters = true; ///< Whether to detect defect clusters.
};

/**
 * @struct QualityMetrics
 * @brief Structure representing the quality metrics of a dark field image.
 */
struct QualityMetrics {
  float snr;                       ///< Signal-to-noise ratio.
  float uniformity;                ///< Uniformity.
  float defect_density;            ///< Defect density.
  std::vector<cv::Point> clusters; ///< Locations of defect clusters.

  /**
   * @brief Converts the quality metrics to a string representation.
   * @return A string representation of the quality metrics.
   */
  std::string to_string() const;
};

/**
 * @class DefectPixelMapper
 * @brief Class for mapping and correcting defect pixels in dark field images.
 */
class DefectPixelMapper {
public:
  /**
   * @struct Config
   * @brief Configuration structure for DefectPixelMapper.
   */
  struct Config {
    int warm_pixel_threshold = 5;           ///< Warm pixel threshold.
    float dead_pixel_value = 0.0f;          ///< Dead pixel threshold.
    int cache_size = 1024;                  ///< Cache size (KB).
    bool use_simd = true;                   ///< Enable SIMD optimizations.
    bool enable_debug = false;              ///< Enable debug mode.
    DefectDetectionConfig detection;        ///< Defect detection configuration.
    std::string log_file = "darkfield.log"; ///< Log file path.
    int batch_size = 100;                   ///< Batch size for processing.
    bool save_intermediates = false;        ///< Save intermediate results.
    bool enable_temp_compensation = false; ///< Enable temperature compensation.
    double temp_coefficient = 0.1;         ///< Default temperature coefficient.
    double baseline_temp = 20.0;           ///< Baseline temperature (Celsius).

    // Parallel processing configuration
    bool use_parallel = true; ///< Use parallel processing.
    int block_size =
        parallel_config::DEFAULT_BLOCK_SIZE; ///< Block size for processing.
    bool use_gpu = false;                    ///< Use GPU for processing.
  };

  /**
   * @brief Constructs a DefectPixelMapper with the given configuration.
   * @param config The configuration for defect pixel mapping.
   */
  explicit DefectPixelMapper(const Config &config);

  /**
   * @brief Builds the defect map from a set of dark frames.
   * @param dark_frames The input dark frames.
   * @param progress_cb Optional progress callback function.
   */
  void build_defect_map(const std::vector<cv::Mat> &dark_frames,
                        std::function<void(float)> progress_cb = nullptr);

  /**
   * @brief Corrects an image using the defect map.
   * @param raw_image The raw image to correct.
   * @param current_temp The current temperature (Celsius).
   * @return The corrected image.
   */
  cv::Mat correct_image(const cv::Mat &raw_image, double current_temp = 20.0);

  /**
   * @brief Saves the defect map to a file.
   * @param path The file path to save the defect map.
   */
  void save_map(const fs::path &path) const;

  /**
   * @brief Loads the defect map from a file.
   * @param path The file path to load the defect map from.
   */
  void load_map(const fs::path &path);

  /**
   * @brief Analyzes the quality of a dark field image.
   * @param image The image to analyze.
   * @return The quality metrics of the image.
   */
  QualityMetrics analyze_quality(const cv::Mat &image) const;

  /**
   * @brief Batch processes a set of input files and saves the results to an
   * output directory.
   * @param input_files The input files to process.
   * @param output_dir The directory to save the processed results.
   */
  void batch_process(const std::vector<std::string> &input_files,
                     const std::string &output_dir);

  /**
   * @brief Adds temperature data for temperature compensation.
   * @param temp The temperature (Celsius).
   * @param dark_current The dark current value.
   * @param exposure_time The exposure time (seconds).
   */
  void add_temperature_data(double temp, double dark_current,
                            double exposure_time);

  /**
   * @brief Enables or disables temperature compensation.
   * @param enable Whether to enable temperature compensation.
   */
  void enable_temperature_compensation(bool enable = true);

  /**
   * @brief Gets the temperature compensation parameters.
   * @return The temperature compensation parameters.
   */
  TemperatureCompensation get_temperature_compensation() const;

private:
  Config config_;                  ///< Configuration for defect pixel mapping.
  cv::Mat defect_map_;             ///< The defect map.
  std::unique_ptr<float[]> cache_; ///< Cache for processing.
  std::vector<TemperatureData> temp_history_; ///< History of temperature data.
  TemperatureCompensation temp_comp_; ///< Temperature compensation parameters.

  /**
   * @brief Validates the input frames.
   * @param frames The input frames to validate.
   */
  void validate_input(const std::vector<cv::Mat> &frames);

  /**
   * @brief Computes statistics (mean and standard deviation) from the input
   * frames.
   * @param frames The input frames.
   * @param progress_cb Optional progress callback function.
   * @return A pair of matrices representing the mean and standard deviation.
   */
  std::pair<cv::Mat, cv::Mat>
  compute_statistics(const std::vector<cv::Mat> &frames,
                     std::function<void(float)> progress_cb = nullptr);

  /**
   * @brief Detects defect pixels at a specific location.
   * @param x The x-coordinate of the pixel.
   * @param y The y-coordinate of the pixel.
   * @param mean The mean image.
   * @param stddev The standard deviation image.
   */
  void detect_defect_pixel(int x, int y, const cv::Mat &mean,
                           const cv::Mat &stddev);

  /**
   * @brief Interpolates the value of a pixel.
   * @param x The x-coordinate of the pixel.
   * @param y The y-coordinate of the pixel.
   * @param img The image to interpolate from.
   * @return The interpolated pixel value.
   */
  float interpolate_pixel(int x, int y, const cv::Mat &img) const;

  /**
   * @brief Performs bilinear interpolation for a pixel.
   * @param x The x-coordinate of the pixel.
   * @param y The y-coordinate of the pixel.
   * @param img The image to interpolate from.
   * @return The interpolated pixel value.
   */
  float bilinear_interpolate(int x, int y, const cv::Mat &img) const;

  /**
   * @brief Optimizes the defect map.
   */
  void optimize_defect_map();

  /**
   * @brief Saves debug information.
   * @param mean The mean image.
   * @param stddev The standard deviation image.
   */
  void save_debug_info(const cv::Mat &mean, const cv::Mat &stddev);

  /**
   * @brief Finds clusters of defect pixels.
   * @param clusters The output vector of defect cluster locations.
   */
  void find_defect_clusters(std::vector<cv::Point> &clusters) const;

  /**
   * @brief Detects defects using a statistical method.
   * @param mean The mean image.
   * @param stddev The standard deviation image.
   */
  void detect_defects_statistical(const cv::Mat &mean, const cv::Mat &stddev);

  /**
   * @brief Detects defects using a pattern recognition method.
   */
  void detect_defects_pattern();

  /**
   * @brief Saves intermediate results during processing.
   * @param original The original image.
   * @param corrected The corrected image.
   * @param basename The base name for the saved files.
   * @param output_dir The directory to save the files.
   */
  void save_intermediate_results(const cv::Mat &original,
                                 const cv::Mat &corrected,
                                 const std::string &basename,
                                 const std::string &output_dir);

  /**
   * @brief Calibrates temperature compensation parameters.
   * @param dark_frames The input dark frames.
   * @param temp_data The temperature data.
   */
  void calibrate_temperature_compensation(
      const std::vector<cv::Mat> &dark_frames,
      const std::vector<TemperatureData> &temp_data);

  /**
   * @brief Applies temperature compensation to an image.
   * @param image The image to apply compensation to.
   * @param current_temp The current temperature (Celsius).
   * @return The compensated image.
   */
  cv::Mat apply_temperature_compensation(const cv::Mat &image,
                                         double current_temp);

  // Parallel processing methods
  /**
   * @brief Processes a block of the image in parallel.
   * @param frame The input frame.
   * @param result The output processed frame.
   * @param start_row The starting row of the block.
   * @param end_row The ending row of the block.
   */
  void process_block_parallel(const cv::Mat &frame, cv::Mat &result,
                              int start_row, int end_row);

  /**
   * @brief Processes an image using GPU.
   * @param frame The input frame.
   */
  void process_gpu(const cv::Mat &frame);

  /**
   * @brief Checks if GPU is available for processing.
   * @return True if GPU is available, false otherwise.
   */
  bool check_gpu_available() const;

  // Parallel processing related methods
  /**
   * @brief Processes a block of the image.
   * @param frame The input frame.
   * @param result The output processed frame.
   * @param start_row The starting row of the block.
   * @param end_row The ending row of the block.
   */
  void process_block(const cv::Mat &frame, cv::Mat &result, int start_row,
                     int end_row);

  /**
   * @brief Processes an image in parallel.
   * @param frame The input frame.
   * @param result The output processed frame.
   */
  void process_image_parallel(const cv::Mat &frame, cv::Mat &result);

  // GPU related methods
  /**
   * @brief Initializes GPU resources for processing.
   * @return True if GPU resources were successfully initialized, false
   * otherwise.
   */
  bool init_gpu_resources();

  /**
   * @brief Releases GPU resources.
   */
  void release_gpu_resources();

  /**
   * @brief Processes an image using GPU.
   * @param frame The input frame.
   * @param result The output processed frame.
   */
  void process_image_gpu(const cv::Mat &frame, cv::Mat &result);

  // Helper methods
  /**
   * @brief Splits the work into blocks for parallel processing.
   * @param total_rows The total number of rows in the image.
   * @param num_blocks The output number of blocks.
   * @param blocks The output vector of block ranges.
   */
  void split_work_blocks(int total_rows, int &num_blocks,
                         std::vector<std::pair<int, int>> &blocks);

  // GPU resource handles
  /**
   * @struct GpuResources
   * @brief Structure representing GPU resources.
   */
  struct GpuResources {
    void *d_input = nullptr;      ///< Device input pointer.
    void *d_output = nullptr;     ///< Device output pointer.
    void *d_defect_map = nullptr; ///< Device defect map pointer.
    bool initialized = false; ///< Whether the GPU resources are initialized.
  } gpu_res_;
};