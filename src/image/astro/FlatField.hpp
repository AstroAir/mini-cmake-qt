#pragma once

#include "ParallelConfig.hpp"
#include "utils/ThreadPool.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @struct FlatQualityMetrics
 * @brief Structure representing the quality metrics of a flat field image.
 */
struct FlatQualityMetrics {
  double uniformity;    ///< Uniformity of the flat field.
  double signal_noise;  ///< Signal-to-noise ratio.
  double vignetting;    ///< Degree of vignetting.
  int hot_pixels;       ///< Number of hot pixels.
  double overall_score; ///< Overall quality score.

  /**
   * @brief Converts the quality metrics to a string representation.
   * @return A string representation of the quality metrics.
   */
  std::string to_string() const;
};

/**
 * @struct FlatConfig
 * @brief Configuration structure for flat field processing.
 */
struct FlatConfig {
  bool use_median = true;      ///< Use median blending for flat field creation.
  float min_flat_value = 0.1f; ///< Minimum flat field value.
  bool use_simd = true;        ///< Enable SIMD optimizations.
  size_t cache_size = 1024;    ///< Cache size in KB.
  bool enable_debug = false;   ///< Enable debug mode.

  double hot_pixel_threshold = 3.0; ///< Hot pixel detection threshold (sigma).
  int block_size = 32;              ///< Block size for processing.
  bool save_debug_info = false;     ///< Save debug information.
  std::string output_dir = "debug"; ///< Directory for debug output.
  bool enable_auto_calibration = true; ///< Enable automatic calibration.

  /**
   * @brief Converts the configuration to a string representation.
   * @return A string representation of the configuration.
   */
  std::string to_string() const;
};

/**
 * @brief Blends multiple images using median blending.
 * @param images The input images to blend.
 * @param result The output blended image.
 */
void medianBlend(const std::vector<cv::Mat> &images, cv::Mat &result);

/**
 * @brief Blends multiple images using mean blending.
 * @param images The input images to blend.
 * @param result The output blended image.
 */
void meanBlend(const std::vector<cv::Mat> &images, cv::Mat &result);

/**
 * @class FlatFieldProcessor
 * @brief Class for processing flat field images.
 */
class FlatFieldProcessor {
public:
  /**
   * @brief Constructs a FlatFieldProcessor with the given configuration.
   * @param config The configuration for flat field processing.
   */
  FlatFieldProcessor(const FlatConfig &config);

  /**
   * @brief Processes the given flat frames to create a master flat field.
   * @param flat_frames The input flat frames.
   * @param master_bias The master bias frame (optional).
   * @param master_dark The master dark frame (optional).
   * @return The processed master flat field.
   */
  cv::Mat process(const std::vector<cv::Mat> &flat_frames,
                  const cv::Mat &master_bias = cv::Mat(),
                  const cv::Mat &master_dark = cv::Mat());

  /**
   * @brief Gets the quality metrics of the processed flat field.
   * @return The quality metrics of the flat field.
   */
  FlatQualityMetrics getQualityMetrics() const;

  /**
   * @brief Generates a visualization of the flat field quality.
   * @return An image visualizing the flat field quality.
   */
  cv::Mat visualizeQuality() const;

  /**
   * @struct ProcessConfig
   * @brief Configuration structure for parallel processing.
   */
  struct ProcessConfig {
    bool use_parallel = true; ///< Use parallel processing.
    int block_size =
        parallel_config::DEFAULT_BLOCK_SIZE; ///< Block size for parallel
                                             ///< processing.
    bool use_gpu = false;                    ///< Use GPU for processing.
    bool enable_simd = true;                 ///< Enable SIMD optimizations.
    int thread_count =
        parallel_config::DEFAULT_THREAD_COUNT; ///< Number of threads for
                                               ///< parallel processing.

    int min_block_size = 16;        ///< Minimum block size.
    int max_block_size = 256;       ///< Maximum block size.
    bool dynamic_block_size = true; ///< Use dynamic block size.
    int gpu_batch_size = 4;         ///< Batch size for GPU processing.
    bool use_thread_pool = true; ///< Use thread pool for parallel processing.
    int queue_size = 8;          ///< Task queue size for thread pool.
  };

private:
  FlatConfig config_;   ///< Configuration for flat field processing.
  cv::Mat master_flat_; ///< The master flat field.
  FlatQualityMetrics quality_metrics_; ///< Quality metrics of the flat field.
  ProcessConfig process_config_; ///< Configuration for parallel processing.

  /**
   * @brief Validates the input frames.
   * @param frames The input frames to validate.
   */
  void validateInputs(const std::vector<cv::Mat> &frames);

  /**
   * @brief Creates the master flat field from the input frames.
   * @param frames The input flat frames.
   * @param bias The master bias frame.
   * @param dark The master dark frame.
   * @return The created master flat field.
   */
  cv::Mat createMasterFlat(const std::vector<cv::Mat> &frames,
                           const cv::Mat &bias, const cv::Mat &dark);

  /**
   * @brief Evaluates the quality of the flat field.
   * @param flat The flat field to evaluate.
   * @return The quality metrics of the flat field.
   */
  FlatQualityMetrics evaluateQuality(const cv::Mat &flat);

  /**
   * @brief Calculates the degree of vignetting in the flat field.
   * @param flat The flat field to analyze.
   * @return The degree of vignetting.
   */
  double calculateVignetting(const cv::Mat &flat);

  /**
   * @brief Detects hot pixels in the flat field.
   * @param flat The flat field to analyze.
   * @return A mask image indicating the hot pixels.
   */
  cv::Mat detectHotPixels(const cv::Mat &flat);

  /**
   * @brief Calculates the overall quality score of the flat field.
   * @param metrics The quality metrics of the flat field.
   * @return The overall quality score.
   */
  double calculateOverallScore(const FlatQualityMetrics &metrics);

  /**
   * @brief Creates a uniformity heatmap for the flat field.
   * @param map The output uniformity heatmap.
   */
  void createUniformityMap(cv::Mat &map) const;

  /**
   * @brief Creates a vignetting analysis map for the flat field.
   * @param map The output vignetting analysis map.
   */
  void createVignettingMap(cv::Mat &map) const;

  /**
   * @brief Computes the radial profile of the flat field.
   * @param flat The flat field to analyze.
   * @param profile The output radial profile.
   */
  void computeRadialProfile(const cv::Mat &flat, cv::Mat &profile) const;

  /**
   * @brief Saves debug information to disk.
   */
  void saveDebugInfo();

  /**
   * @brief Processes the flat frames in parallel blocks.
   * @param frames The input flat frames.
   * @param result The output processed flat field.
   */
  void process_parallel_blocks(const std::vector<cv::Mat> &frames,
                               cv::Mat &result);

  /**
   * @brief Applies flat field correction in parallel.
   * @param image The image to correct.
   */
  void apply_correction_parallel(cv::Mat &image);

  /**
   * @brief Validates the GPU requirements for processing.
   * @return True if the GPU requirements are met, false otherwise.
   */
  bool validate_gpu_requirements() const;

  /**
   * @brief Initializes resources for parallel processing.
   */
  void initParallelResources();

  /**
   * @brief Releases resources used for parallel processing.
   */
  void releaseParallelResources();

  /**
   * @brief Processes a block of the image using SIMD optimizations.
   * @param src The source image.
   * @param dst The destination image.
   * @param range The range of the block to process.
   */
  void processSIMDBlock(const cv::Mat &src, cv::Mat &dst,
                        const cv::Range &range);

  /**
   * @brief Applies flat field correction using SIMD optimizations.
   * @param image The image to correct.
   * @param range The range of the image to process.
   */
  void applyFlatCorrectionSIMD(cv::Mat &image, const cv::Range &range);

  /**
   * @brief Initializes the GPU for processing.
   * @return True if the GPU was successfully initialized, false otherwise.
   */
  bool initializeGPU();

  /**
   * @brief Processes a batch of images on the GPU.
   * @param batch The batch of images to process.
   * @param result The output processed image.
   */
  void processGPUBatch(const std::vector<cv::Mat> &batch, cv::Mat &result);

  /**
   * @brief Uploads data from the host to the GPU.
   * @param host_data The data on the host.
   * @param device_data The data on the GPU.
   */
  void uploadToGPU(const cv::Mat &host_data, void *device_data);

  /**
   * @brief Downloads data from the GPU to the host.
   * @param device_data The data on the GPU.
   * @param host_data The data on the host.
   */
  void downloadFromGPU(void *device_data, cv::Mat &host_data);

  /**
   * @struct Block
   * @brief Structure representing a block of the image for processing.
   */
  struct Block {
    cv::Range row_range; ///< The row range of the block.
    cv::Range col_range; ///< The column range of the block.
    int priority;        ///< The priority of the block.
  };

  /**
   * @brief Generates blocks for parallel processing.
   * @param image The image to process.
   * @return A vector of blocks for parallel processing.
   */
  std::vector<Block> generateBlocks(const cv::Mat &image) const;

  /**
   * @brief Processes a block of the image.
   * @param block The block to process.
   * @param result The output processed image.
   */
  void processBlock(const Block &block, cv::Mat &result);

  /**
   * @brief Initializes the thread pool for parallel processing.
   */
  void initThreadPool();

  /**
   * @brief Submits a task to the thread pool.
   * @param block The block to process.
   * @param result The output processed image.
   * @param priority The priority of the task.
   */
  void submitTask(const Block &block, cv::Mat &result,
                  DynamicThreadPool::Priority priority =
                      DynamicThreadPool::Priority::Normal);

  /**
   * @brief Waits for all tasks in the thread pool to complete.
   */
  void waitForTasks();

  /**
   * @brief Preallocates buffers for processing.
   * @param size The size of the buffers to preallocate.
   */
  void preallocateBuffers(const cv::Size &size);

  /**
   * @brief Recycles buffers after processing.
   */
  void recycleBuffers();

  /**
   * @brief Calculates the optimal block size for processing.
   * @param image_size The size of the image.
   * @return The optimal block size.
   */
  int calculateOptimalBlockSize(const cv::Size &image_size) const;

  /**
   * @brief Determines if GPU processing is beneficial for the given image size.
   * @param size The size of the image.
   * @return True if GPU processing is beneficial, false otherwise.
   */
  bool isGPUBeneficial(const cv::Size &size) const;

  std::shared_ptr<DynamicThreadPool>
      thread_pool_; ///< Thread pool for parallel processing.
};

/**
 * @brief Applies flat field correction to a raw image.
 * @param raw_image The raw image to correct.
 * @param master_flat The master flat field.
 * @param master_bias The master bias frame (optional).
 * @param master_dark The master dark frame (optional).
 * @param config The configuration for flat field correction.
 * @return The corrected image.
 */
cv::Mat apply_flat_correction(const cv::Mat &raw_image,
                              const cv::Mat &master_flat,
                              const cv::Mat &master_bias = cv::Mat(),
                              const cv::Mat &master_dark = cv::Mat(),
                              const FlatConfig &config = FlatConfig());

/**
 * @brief Main function example.
 * @return Exit status.
 */
int main();