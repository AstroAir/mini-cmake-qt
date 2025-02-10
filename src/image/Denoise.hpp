#pragma once

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

/**
 * @brief Enumeration of denoising methods.
 */
enum class DenoiseMethod {
  Auto,      ///< Automatically select based on noise analysis
  Median,    ///< Median filter for salt-and-pepper noise
  Gaussian,  ///< Gaussian filter for Gaussian noise
  Bilateral, ///< Bilateral filter to preserve edges
  NLM,       ///< Non-Local Means for uniform noise
  Wavelet    ///< Wavelet transform denoising
};

/**
 * @brief Enumeration of wavelet types.
 */
enum class WaveletType { Haar, Daubechies4, Coiflet, Biorthogonal };

/**
 * @brief Structure to hold denoising parameters.
 */
struct DenoiseParameters {
  DenoiseMethod method = DenoiseMethod::Auto; ///< Denoising method
  int threads = 4;                            ///< Number of parallel threads

  // Median filter parameters
  int median_kernel = 5; ///< Kernel size for median filter

  // Gaussian filter parameters
  cv::Size gaussian_kernel = {5, 5}; ///< Kernel size for Gaussian filter
  double sigma_x = 1.5;              ///< Sigma X for Gaussian filter
  double sigma_y = 1.5;              ///< Sigma Y for Gaussian filter

  // Bilateral filter parameters
  int bilateral_d = 9;       ///< Diameter of each pixel neighborhood
  double sigma_color = 75.0; ///< Filter sigma in the color space
  double sigma_space = 75.0; ///< Filter sigma in the coordinate space

  // NLM parameters
  float nlm_h = 3.0f;        ///< Parameter regulating filter strength
  int nlm_template_size = 7; ///< Size of the template patch
  int nlm_search_size = 21;  ///< Size of the window search area

  // Wavelet parameters
  int wavelet_level = 3;           ///< Number of decomposition levels
  float wavelet_threshold = 15.0f; ///< Threshold for wavelet denoising
  WaveletType wavelet_type = WaveletType::Haar; ///< Type of wavelet
  bool use_adaptive_threshold = true;           ///< Use adaptive thresholding
  double noise_estimate = 0.0;                  ///< Estimated noise level
  int block_size = 32;                          ///< Block size for processing

  // Optimization parameters
  bool use_simd = true;           ///< 使用SIMD优化
  bool use_opencl = false;        ///< 使用OpenCL GPU加速
  int tile_size = 256;            ///< 分块大小
  bool use_stream = true;         ///< 使用流水线处理
};

/**
 * @brief Class for wavelet-based denoising.
 */
class WaveletDenoiser {
public:
  /**
   * @brief Denoise an image using wavelet transform.
   * @param src Source image
   * @param dst Destination image
   * @param levels Number of decomposition levels
   * @param threshold Threshold for denoising
   */
  static void denoise(const cv::Mat &src, cv::Mat &dst, int levels,
                      float threshold);

private:
  static void wavelet_process_single_channel(const cv::Mat &src, cv::Mat &dst,
                                             int levels, float threshold);
  static cv::Mat decompose_one_level(const cv::Mat &src);
  static cv::Mat recompose_one_level(const cv::Mat &waveCoeffs,
                                     const cv::Size &originalSize);
  static void process_blocks(cv::Mat &img, int block_size,
                             const std::function<void(cv::Mat &)> &process_fn);
  static void wavelet_transform_simd(cv::Mat &data);
  static float compute_adaptive_threshold(const cv::Mat &coeffs,
                                          double noise_estimate);

  // 新增优化方法
  static void process_tile_simd(cv::Mat& tile);
  static void parallel_wavelet_transform(cv::Mat& data);
  static void optimize_memory_layout(cv::Mat& data);
  static void stream_process(const cv::Mat& src, cv::Mat& dst, 
                             const std::function<void(cv::Mat&)>& process_fn);

public:
  /**
   * @brief Denoise an image using specified parameters.
   * @param src Source image
   * @param dst Destination image
   * @param params Denoising parameters
   */
  static void denoise(const cv::Mat &src, cv::Mat &dst,
                      const DenoiseParameters &params);
};

/**
 * @brief Class for image denoising.
 */
class ImageDenoiser {
public:
  /**
   * @brief Constructor for ImageDenoiser.
   * @param logger Optional logger for logging
   */
  explicit ImageDenoiser();

  /**
   * @brief Denoise an image using specified parameters.
   * @param input Input image
   * @param params Denoising parameters
   * @return Denoised image
   */
  cv::Mat denoise(const cv::Mat &input, const DenoiseParameters &params);

private:
  DenoiseMethod analyze_noise(const cv::Mat &img);
  double detect_salt_pepper(const cv::Mat &gray);
  double detect_gaussian(const cv::Mat &gray);
  void process_nlm(const cv::Mat &src, cv::Mat &dst,
                   const DenoiseParameters &params);
  void validate_median(const DenoiseParameters &params);
  void validate_gaussian(const DenoiseParameters &params);
  void validate_bilateral(const DenoiseParameters &params);
  void process_bilateral(const cv::Mat &src, cv::Mat &dst,
                         const DenoiseParameters &params);
  const char *method_to_string(DenoiseMethod method);
};