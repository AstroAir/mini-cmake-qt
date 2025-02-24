#pragma once

#include <fmt/format.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

/**
 * @brief Collects the neighboring pixels of a region of interest (ROI).
 * @param roi The region of interest.
 * @param neighbors The vector to store the neighboring pixels.
 */
void collectNeighbors(const Mat &roi, vector<uchar> &neighbors);

/**
 * @brief Applies a guided filter to an image.
 * @param src The source image.
 * @param guide The guide image.
 * @param radius The radius of the guided filter.
 * @param eps The regularization parameter.
 * @return The filtered image.
 */
Mat guidedFilter(const Mat &src, const Mat &guide, int radius, double eps);

/**
 * @struct BadPixelStats
 * @brief Structure to hold statistics about bad pixels.
 */
struct BadPixelStats {
  int total_pixels;        ///< Total number of pixels.
  int bad_pixels;          ///< Number of bad pixels.
  float bad_ratio;         ///< Ratio of bad pixels.
  double avg_correction;   ///< Average correction value.
  vector<Point> hot_spots; ///< Hot spot regions.

  /**
   * @brief Converts the statistics to a string representation.
   * @return A string representation of the statistics.
   */
  string to_string() const;
};

/**
 * @struct BadPixelConfig
 * @brief Configuration structure for bad pixel correction.
 */
struct BadPixelConfig {
  /**
   * @enum DetectionMethod
   * @brief Enumeration of detection methods.
   */
  enum DetectionMethod {
    MEDIAN,       ///< Median detection.
    MEAN,         ///< Mean detection.
    GAUSSIAN,     ///< Gaussian detection.
    ADAPTIVE,     ///< Adaptive threshold detection.
    TEMPORAL,     ///< Temporal detection.
    PATTERN_BASED ///< Pattern-based detection.
  };

  /**
   * @enum CorrectionMethod
   * @brief Enumeration of correction methods.
   */
  enum CorrectionMethod {
    MEDIAN_REPLACE, ///< Median replacement.
    BILINEAR,       ///< Bilinear interpolation.
    EDGE_AWARE,     ///< Edge-aware correction.
    INPAINTING,     ///< Inpainting using OpenCV.
    GUIDED_FILTER,  ///< Guided filter correction.
    DEEP_PRIOR      ///< Deep prior correction.
  };

  DetectionMethod detect_method = MEDIAN;           ///< Detection method.
  CorrectionMethod correct_method = MEDIAN_REPLACE; ///< Correction method.

  int detect_window = 5;                ///< Detection window size.
  int correct_window = 5;               ///< Correction window size.
  float detect_threshold = 35.0f;       ///< Detection threshold.
  bool use_dynamic_threshold = false;   ///< Use dynamic threshold.
  bool use_channel_correlation = false; ///< Use channel correlation.
  bool multi_thread = true;             ///< Enable multi-threading.
  bool save_debug = false;              ///< Save debug information.
  string preset_map;                    ///< Preset bad pixel map.

  bool enable_gpu = false;            ///< Enable GPU acceleration.
  int temporal_window = 3;            ///< Temporal window size.
  float confidence_threshold = 0.95f; ///< Confidence threshold.
  string pattern_file;                ///< Pattern file.
  bool auto_threshold = true;         ///< Enable automatic threshold.
  vector<float> channel_weights;      ///< Channel weights.

  static constexpr size_t CACHE_SIZE = 1024; ///< Cache size.
  static constexpr int BLOCK_SIZE = 32;      ///< Block size.
};

/**
 * @class BadPixelCorrector
 * @brief Class for correcting bad pixels in images.
 */
class BadPixelCorrector {
public:
  /**
   * @brief Constructs a BadPixelCorrector with the given configuration.
   * @param cfg The configuration for bad pixel correction.
   */
  explicit BadPixelCorrector(const BadPixelConfig &cfg = {});

  /**
   * @brief Processes the input image to correct bad pixels.
   * @param input The input image.
   * @return The corrected image.
   */
  Mat process(const Mat &input);

  /**
   * @brief Gets the statistics of the bad pixel correction.
   * @return The statistics of the bad pixel correction.
   */
  BadPixelStats getStats() const;

  /**
   * @brief Evaluates the quality of the correction.
   * @param original The original image.
   * @param corrected The corrected image.
   * @return The quality score of the correction.
   */
  float evaluateQuality(const Mat &original, const Mat &corrected);

  /**
   * @brief Visualizes the correction process.
   * @return An image visualizing the correction process.
   */
  Mat visualizeCorrection() const;

private:
  BadPixelConfig config_;          ///< Configuration for bad pixel correction.
  unique_ptr<float[]> cache_;      ///< Cache for calculations.
  Mat preset_mask_;                ///< Preset bad pixel mask.
  vector<float> gaussian_weights_; ///< Gaussian weights.
  BadPixelStats stats_;            ///< Statistics of the bad pixel correction.
  Mat last_input_;                 ///< Last input image.
  Mat last_output_;                ///< Last output image.
  Mat last_mask_;                  ///< Last mask image.
  Mat pattern_template_; ///< Pattern template for pattern-based detection.

  /**
   * @struct DeepPriorModel
   * @brief Structure representing a deep prior model for correction.
   */
  struct DeepPriorModel {
    int patch_size = 64;          ///< Patch size.
    float learning_rate = 0.001f; ///< Learning rate.
    int max_iterations = 1000;    ///< Maximum iterations.
    vector<Mat> weights;          ///< Weights of the model.

    /**
     * @brief Initializes the deep prior model.
     */
    void init();

    /**
     * @brief Performs a forward pass through the model.
     * @param input The input image.
     * @return The output image after the forward pass.
     */
    Mat forward(const Mat &input);
  };

  DeepPriorModel deep_prior_model_; ///< Deep prior model.
  vector<Mat> temporal_buffer_;     ///< Buffer for temporal detection.

  /**
   * @brief Initializes the bad pixel corrector.
   */
  void init();

  /**
   * @brief Loads the preset bad pixel map.
   */
  void load_preset_map();

  /**
   * @brief Precomputes the Gaussian weights.
   */
  void precompute_gaussian_weights();

  /**
   * @brief Processes a single-channel image.
   * @param image The input image.
   * @return The corrected image.
   */
  Mat process_single_channel(Mat &image);

  /**
   * @brief Processes a color image.
   * @param image The input image.
   * @return The corrected image.
   */
  Mat process_color(Mat &image);

  /**
   * @brief Detects bad pixels in the image.
   * @param image The input image.
   * @return A mask indicating the bad pixels.
   */
  Mat detect_bad_pixels(const Mat &image);

  /**
   * @brief Calculates the dynamic threshold for a region.
   * @param region The region of interest.
   * @return The dynamic threshold.
   */
  float calculate_dynamic_threshold(const Mat &region);

  /**
   * @brief Gets the threshold for bad pixel detection.
   * @param image The input image.
   * @param x The x-coordinate of the pixel.
   * @param y The y-coordinate of the pixel.
   * @return The threshold for bad pixel detection.
   */
  float get_threshold(const Mat &image, int x, int y);

  /**
   * @brief Corrects the bad pixels in the image.
   * @param image The input image.
   * @param mask The mask indicating the bad pixels.
   */
  void correct_bad_pixels(Mat &image, const Mat &mask);

  /**
   * @brief Calculates the Gaussian reference value for a pixel.
   * @param image The input image.
   * @param x The x-coordinate of the pixel.
   * @param y The y-coordinate of the pixel.
   * @param radius The radius of the Gaussian window.
   * @return The Gaussian reference value.
   */
  float calculateGaussianReference(const Mat &image, int x, int y, int radius);

  /**
   * @brief Detects temporal anomalies in the current frame.
   * @param current_frame The current frame.
   * @return A mask indicating the temporal anomalies.
   */
  Mat detect_temporal_anomalies(const Mat &current_frame);

  /**
   * @brief Corrects the bad pixels using a deep prior model.
   * @param image The input image.
   * @param mask The mask indicating the bad pixels.
   */
  void correct_with_deep_prior(Mat &image, const Mat &mask);

  /**
   * @brief Calculates the adaptive threshold using an integral image.
   * @param integral The integral image.
   * @param x The x-coordinate of the pixel.
   * @param y The y-coordinate of the pixel.
   * @param radius The radius of the window.
   * @return The adaptive threshold.
   */
  float calculate_adaptive_threshold(const Mat &integral, int x, int y,
                                     int radius);

  /**
   * @brief Detects pattern deviation for a pixel.
   * @param image The input image.
   * @param x The x-coordinate of the pixel.
   * @param y The y-coordinate of the pixel.
   * @param radius The radius of the window.
   * @return The pattern deviation value.
   */
  float detect_pattern_deviation(const Mat &image, int x, int y, int radius);
};
