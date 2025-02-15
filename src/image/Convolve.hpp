#pragma once

#include <chrono>
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
  bool use_simd = true;           ///< 启用SIMD优化
  bool use_memory_pool = true;    ///< 使用内存池
  int tile_size = 256;            ///< 分块处理大小，用于缓存优化
  bool use_fft = false;           ///< 使用FFT加速大型卷积
  int thread_count = 0;           ///< 线程数(0表示自动)
  bool use_avx = true;            ///< 启用AVX指令集
  int block_size = 32;            ///< 缓存块大小
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
  bool use_simd = true;        ///< 启用SIMD优化
  bool use_memory_pool = true; ///< 使用内存池
  int tile_size = 256;         ///< 分块处理大小
  bool use_fft = true;         ///< 使用FFT加速
  int thread_count = 0;        ///< 线程数(0表示自动)
  bool use_avx = true;         ///< 启用AVX指令集
  int block_size = 32;         ///< 缓存块大小
};

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
   * @return The processed image.
   */
  static cv::Mat
  process(const cv::Mat &input,
          const std::variant<ConvolutionConfig, DeconvolutionConfig> &config);

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
    ScopedTimer(const std::string &op);

    /**
     * @brief Destructor for ScopedTimer.
     */
    ~ScopedTimer();
  };

  /**
   * @brief Processes each channel of a multi-channel image separately.
   * @param input The input image.
   * @param processor The function to process each channel.
   * @return The processed image.
   */
  static cv::Mat
  processMultiChannel(const cv::Mat &input,
                      const std::function<cv::Mat(const cv::Mat &)> &processor);

  /**
   * @brief Performs convolution on an image.
   * @param input The input image.
   * @param cfg The convolution configuration.
   * @return The convolved image.
   */
  static cv::Mat convolve(const cv::Mat &input, const ConvolutionConfig &cfg);

  /**
   * @brief Performs convolution on a single-channel image.
   * @param input The input image.
   * @param cfg The convolution configuration.
   * @return The convolved image.
   */
  static cv::Mat convolveSingleChannel(const cv::Mat &input,
                                       const ConvolutionConfig &cfg);

  /**
   * @brief Performs deconvolution on an image.
   * @param input The input image.
   * @param cfg The deconvolution configuration.
   * @return The deconvolved image.
   */
  static cv::Mat deconvolve(const cv::Mat &input,
                            const DeconvolutionConfig &cfg);

  /**
   * @brief Performs deconvolution on a single-channel image.
   * @param input The input image.
   * @param cfg The deconvolution configuration.
   * @return The deconvolved image.
   */
  static cv::Mat deconvolveSingleChannel(const cv::Mat &input,
                                         const DeconvolutionConfig &cfg);

  /**
   * @brief Performs Richardson-Lucy deconvolution.
   * @param input The input image.
   * @param psf The point spread function.
   * @param output The output image.
   * @param cfg The deconvolution configuration.
   */
  static void richardsonLucyDeconv(const cv::Mat &input, const cv::Mat &psf,
                                   cv::Mat &output,
                                   const DeconvolutionConfig &cfg);

  /**
   * @brief Performs Wiener deconvolution.
   * @param input The input image.
   * @param psf The point spread function.
   * @param output The output image.
   * @param cfg The deconvolution configuration.
   */
  static void wienerDeconv(const cv::Mat &input, const cv::Mat &psf,
                           cv::Mat &output, const DeconvolutionConfig &cfg);

  /**
   * @brief Performs Tikhonov regularization deconvolution.
   * @param input The input image.
   * @param psf The point spread function.
   * @param output The output image.
   * @param cfg The deconvolution configuration.
   */
  static void tikhonovDeconv(const cv::Mat &input, const cv::Mat &psf,
                             cv::Mat &output, const DeconvolutionConfig &cfg);

  /**
   * @brief Validates the convolution configuration.
   * @param input The input image.
   * @param cfg The convolution configuration.
   */
  static void validateConvolutionConfig(const cv::Mat &input,
                                        const ConvolutionConfig &cfg);

  /**
   * @brief Validates the deconvolution configuration.
   * @param input The input image.
   * @param cfg The deconvolution configuration.
   */
  static void validateDeconvolutionConfig(const cv::Mat &input,
                                          const DeconvolutionConfig &cfg);

  /**
   * @brief Prepares the convolution kernel.
   * @param cfg The convolution configuration.
   * @return The prepared kernel.
   */
  static cv::Mat prepareKernel(const ConvolutionConfig &cfg);

  /**
   * @brief Converts the border mode to an OpenCV border type.
   * @param mode The border mode.
   * @return The OpenCV border type.
   */
  static int getOpenCVBorderType(BorderMode mode);

  /**
   * @brief Estimates the point spread function (PSF).
   * @param imgSize The size of the image.
   * @return The estimated PSF.
   */
  static cv::Mat estimatePSF(cv::Size imgSize);

  // 新增内存池管理
  class MemoryPool {
  public:
    static cv::Mat allocate(int rows, int cols, int type);
    static void deallocate(cv::Mat &mat);
    static void clear();

  private:
    static std::vector<cv::Mat> pool_;
    static std::mutex pool_mutex_;
    static const int max_pool_size_ = 100;
  };

  // 新增优化方法
  static void optimizedConvolveAVX(const cv::Mat &input, cv::Mat &output, 
                                  const cv::Mat &kernel, const ConvolutionConfig &cfg);
  static void fftConvolve(const cv::Mat &input, cv::Mat &output, 
                         const cv::Mat &kernel);
  static void blockProcessing(const cv::Mat &input, cv::Mat &output,
                            const std::function<void(const cv::Mat&, cv::Mat&)>& processor,
                            int blockSize);
};