#pragma once

#include <chrono>
#include <concepts>
#include <expected>
#include <future>
#include <opencv2/opencv.hpp>
#include <variant>

/**
 * @enum BorderMode
 * @brief Enum representing different border handling modes.
 */
enum class BorderMode { ZERO_PADDING, MIRROR_REFLECT, REPLICATE, CIRCULAR };

/**
 * @enum DeconvMethod
 * @brief Enum representing different deconvolution methods.
 */
enum class DeconvMethod { RICHARDSON_LUCY, WIENER, TIKHONOV };

/**
 * @struct ProcessError
 * @brief Error codes and descriptions for processing operations
 */
struct ProcessError {
  enum class Code {
    INVALID_INPUT,
    INVALID_CONFIG,
    PROCESSING_FAILED,
    OUT_OF_MEMORY,
    UNSUPPORTED_OPERATION
  };

  Code code;
  std::string message;
};

/**
 * @struct ConvolutionConfig
 * @brief Configuration structure for convolution operations.
 */
struct ConvolutionConfig {
  std::vector<float> kernel; ///< Convolution kernel.
  int kernel_size;           ///< Size of the convolution kernel.
  BorderMode border_mode = BorderMode::REPLICATE; ///< Border handling mode.
  bool normalize_kernel = true;   ///< Flag to normalize the kernel.
  bool parallel_execution = true; ///< Flag to enable parallel execution.
  bool per_channel = false;       ///< Flag to process each channel separately.
  bool use_simd = true;           ///< Enable SIMD optimization
  bool use_memory_pool = true;    ///< Use memory pool
  int tile_size = 256;            ///< Tile size for cache optimization
  bool use_fft = false;           ///< Use FFT for large kernels
  int thread_count = 0;           ///< Thread count (0 means auto)
  bool use_avx = true;            ///< Use AVX instruction set
  int block_size = 32;            ///< Cache block size
};

/**
 * @struct DeconvolutionConfig
 * @brief Configuration structure for deconvolution operations.
 */
struct DeconvolutionConfig {
  DeconvMethod method =
      DeconvMethod::RICHARDSON_LUCY; ///< Deconvolution method.
  int iterations = 30;      ///< Number of iterations for iterative methods.
  double noise_power = 0.0; ///< Noise power for Wiener deconvolution.
  double regularization =
      1e-6; ///< Regularization parameter for Tikhonov deconvolution.
  BorderMode border_mode = BorderMode::REPLICATE; ///< Border handling mode.
  bool per_channel = false;    ///< Flag to process each channel separately.
  bool use_simd = true;        ///< Enable SIMD optimization
  bool use_memory_pool = true; ///< Use memory pool
  int tile_size = 256;         ///< Tile size for cache optimization
  bool use_fft = true;         ///< Use FFT acceleration
  int thread_count = 0;        ///< Thread count (0 means auto)
  bool use_avx = true;         ///< Use AVX instruction set
  int block_size = 32;         ///< Cache block size
};

// C++20 concept for validating config types
template <typename T>
concept ConfigType =
    std::same_as<T, ConvolutionConfig> || std::same_as<T, DeconvolutionConfig>;

/**
 * @class Convolve
 * @brief Class for performing image processing operations such as convolution
 * and deconvolution.
 */
class Convolve {
public:
  /**
   * @brief Processes an image using the specified configuration.
   * @param input The input image.
   * @param config The configuration for processing (convolution or
   * deconvolution).
   * @return The processed image or an error.
   */
  static std::expected<cv::Mat, ProcessError>
  process(const cv::Mat &input,
          const std::variant<ConvolutionConfig, DeconvolutionConfig> &config);

  /**
   * @brief Async version of process that returns a coroutine
   */
  static std::future<std::expected<cv::Mat, ProcessError>> processAsync(
      const cv::Mat &input,
      const std::variant<ConvolutionConfig, DeconvolutionConfig> &config);

  /**
   * @brief Cleans up resources used by the Convolve class.
   */
  static void cleanup();

private:
  /**
   * @struct ScopedTimer
   * @brief Utility structure for measuring the duration of operations.
   */
  struct ScopedTimer {
    std::string operation; ///< Name of the operation being timed.
    std::chrono::steady_clock::time_point
        start; ///< Start time of the operation.

    /**
     * @brief Constructor for ScopedTimer.
     * @param op The name of the operation.
     */
    explicit ScopedTimer(std::string_view op);

    /**
     * @brief Destructor for ScopedTimer.
     */
    ~ScopedTimer();

    // No copy or move
    ScopedTimer(const ScopedTimer &) = delete;
    ScopedTimer &operator=(const ScopedTimer &) = delete;
  };

  /**
   * @brief Processes each channel of a multi-channel image separately.
   * @param input The input image.
   * @param processor The function to process each channel.
   * @return The processed image or an error.
   */
  static std::expected<cv::Mat, ProcessError> processMultiChannel(
      const cv::Mat &input,
      const std::function<std::expected<cv::Mat, ProcessError>(const cv::Mat &)>
          &processor);

  /**
   * @brief Performs convolution on an image.
   * @param input The input image.
   * @param cfg The convolution configuration.
   * @return The convolved image or an error.
   */
  static std::expected<cv::Mat, ProcessError>
  convolve(const cv::Mat &input, const ConvolutionConfig &cfg);

  /**
   * @brief Performs convolution on a single-channel image.
   * @param input The input image.
   * @param cfg The convolution configuration.
   * @return The convolved image or an error.
   */
  static std::expected<cv::Mat, ProcessError>
  convolveSingleChannel(const cv::Mat &input, const ConvolutionConfig &cfg);

  /**
   * @brief Performs deconvolution on an image.
   * @param input The input image.
   * @param cfg The deconvolution configuration.
   * @return The deconvolved image or an error.
   */
  static std::expected<cv::Mat, ProcessError>
  deconvolve(const cv::Mat &input, const DeconvolutionConfig &cfg);

  /**
   * @brief Performs deconvolution on a single-channel image.
   * @param input The input image.
   * @param cfg The deconvolution configuration.
   * @return The deconvolved image or an error.
   */
  static std::expected<cv::Mat, ProcessError>
  deconvolveSingleChannel(const cv::Mat &input, const DeconvolutionConfig &cfg);

  /**
   * @brief Performs Richardson-Lucy deconvolution.
   * @param input The input image.
   * @param psf The point spread function.
   * @param output The output image.
   * @param cfg The deconvolution configuration.
   * @return Error code if operation fails.
   */
  static std::expected<void, ProcessError>
  richardsonLucyDeconv(const cv::Mat &input, const cv::Mat &psf,
                       cv::Mat &output, const DeconvolutionConfig &cfg);

  /**
   * @brief Performs Wiener deconvolution.
   * @param input The input image.
   * @param psf The point spread function.
   * @param output The output image.
   * @param cfg The deconvolution configuration.
   * @return Error code if operation fails.
   */
  static std::expected<void, ProcessError>
  wienerDeconv(const cv::Mat &input, const cv::Mat &psf, cv::Mat &output,
               const DeconvolutionConfig &cfg);

  /**
   * @brief Performs Tikhonov regularization deconvolution.
   * @param input The input image.
   * @param psf The point spread function.
   * @param output The output image.
   * @param cfg The deconvolution configuration.
   * @return Error code if operation fails.
   */
  static std::expected<void, ProcessError>
  tikhonovDeconv(const cv::Mat &input, const cv::Mat &psf, cv::Mat &output,
                 const DeconvolutionConfig &cfg);

  /**
   * @brief Validates the convolution configuration.
   * @param input The input image.
   * @param cfg The convolution configuration.
   * @return Error if validation fails.
   */
  static std::expected<void, ProcessError>
  validateConvolutionConfig(const cv::Mat &input, const ConvolutionConfig &cfg);

  /**
   * @brief Validates the deconvolution configuration.
   * @param input The input image.
   * @param cfg The deconvolution configuration.
   * @return Error if validation fails.
   */
  static std::expected<void, ProcessError>
  validateDeconvolutionConfig(const cv::Mat &input,
                              const DeconvolutionConfig &cfg);

  /**
   * @brief Prepares the convolution kernel.
   * @param cfg The convolution configuration.
   * @return The prepared kernel or error.
   */
  static std::expected<cv::Mat, ProcessError>
  prepareKernel(const ConvolutionConfig &cfg);

  /**
   * @brief Converts the border mode to an OpenCV border type.
   * @param mode The border mode.
   * @return The OpenCV border type.
   */
  static int getOpenCVBorderType(BorderMode mode) noexcept;

  /**
   * @brief Estimates the point spread function (PSF).
   * @param imgSize The size of the image.
   * @return The estimated PSF or error.
   */
  static std::expected<cv::Mat, ProcessError> estimatePSF(cv::Size imgSize);

  // Improved memory pool management with shared pointers
  class MemoryPool {
  public:
    static cv::Mat allocate(int rows, int cols, int type);
    static void deallocate(cv::Mat &mat);
    static void clear();

  private:
    static std::vector<std::shared_ptr<cv::Mat>> pool_;
    static std::mutex pool_mutex_;
    static const int max_pool_size_ = 100;
  };

  // Optimized algorithms
  static std::expected<void, ProcessError>
  optimizedConvolveAVX(const cv::Mat &input, cv::Mat &output,
                       const cv::Mat &kernel, const ConvolutionConfig &cfg);

  static std::expected<void, ProcessError>
  fftConvolve(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);

  static std::expected<void, ProcessError>
  blockProcessing(const cv::Mat &input, cv::Mat &output,
                  const std::function<std::expected<void, ProcessError>(
                      const cv::Mat &, cv::Mat &)> &processor,
                  int blockSize);
};