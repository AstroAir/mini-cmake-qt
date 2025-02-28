#pragma once

#include <concepts>
#include <functional>
#include <future>
#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>


// C++20 concepts for image types
template <typename T>
concept ImageContainer = requires(T a) {
  { a.cols } -> std::convertible_to<int>;
  { a.rows } -> std::convertible_to<int>;
  { a.channels() } -> std::convertible_to<int>;
  { a.type() } -> std::convertible_to<int>;
};

struct CalibrationParams {
  double wavelength;         // Wavelength, unit: nanometer
  double aperture;           // Aperture diameter, unit: millimeter
  double obstruction;        // Obstruction diameter, unit: millimeter
  double filter_width;       // Filter bandwidth, unit: nanometer
  double transmissivity;     // Transmissivity
  double gain;               // Gain
  double quantum_efficiency; // Quantum efficiency
  double extinction;         // Extinction coefficient
  double exposure_time;      // Exposure time, unit: second

  // Validate parameters
  [[nodiscard]] bool isValid() const noexcept {
    return wavelength > 0 && aperture > 0 && obstruction >= 0 &&
           filter_width > 0 && transmissivity > 0 && transmissivity <= 1 &&
           gain > 0 && quantum_efficiency > 0 && quantum_efficiency <= 1 &&
           extinction >= 0 && extinction < 1 && exposure_time > 0;
  }
};

struct OptimizationParams {
  bool use_gpu{false};      // Whether to use GPU acceleration
  bool use_parallel{false}; // Whether to use parallel processing
  int num_threads{4};       // Number of parallel processing threads
  bool use_cache{false};    // Whether to use cache
  size_t cache_size{1024};  // Cache size (MB)
  bool use_simd{false};     // Whether to use SIMD instructions

  // Validate parameters
  [[nodiscard]] bool isValid() const noexcept {
    return num_threads > 0 && cache_size > 0;
  }
};

// Exception types
class CalibrationError : public std::runtime_error {
public:
  explicit CalibrationError(const std::string &message)
      : std::runtime_error(message) {}
};

class InvalidParameterError : public CalibrationError {
public:
  explicit InvalidParameterError(const std::string &message)
      : CalibrationError(message) {}
};

class ProcessingError : public CalibrationError {
public:
  explicit ProcessingError(const std::string &message)
      : CalibrationError(message) {}
};

// Enhanced function declarations with noexcept specifications and better error
// handling
cv::Mat instrument_response_correction(cv::InputArray &image,
                                       cv::InputArray &response_function);

cv::Mat background_noise_correction(cv::InputArray &image) noexcept;

cv::Mat apply_flat_field_correction(cv::InputArray &image,
                                    cv::InputArray &flat_field);

cv::Mat apply_dark_frame_subtraction(cv::InputArray &image,
                                     cv::InputArray &dark_frame);

[[nodiscard]] double compute_flx2dn(const CalibrationParams &params);

// Enhanced result type with std::optional for error handling
struct FluxCalibrationResult {
  cv::Mat image;
  double min_value;
  double range_value;
  double flx2dn_factor;
};

// Using std::optional for potential failure
[[nodiscard]] std::optional<FluxCalibrationResult>
flux_calibration_ex(const cv::Mat &image, const CalibrationParams &params,
                    const cv::Mat *response_function = nullptr,
                    const cv::Mat *flat_field = nullptr,
                    const cv::Mat *dark_frame = nullptr,
                    bool enable_optimization = false);

// Optimized versions with explicit optimization parameters
cv::Mat
instrument_response_correction_optimized(cv::InputArray &image,
                                         cv::InputArray &response_function,
                                         const OptimizationParams &params);

cv::Mat background_noise_correction_optimized(
    cv::InputArray &image, const OptimizationParams &params) noexcept;

// Batch processing capability
std::vector<cv::Mat>
batch_process_images(const std::vector<cv::Mat> &images,
                     const std::function<cv::Mat(const cv::Mat &)> &processor,
                     const OptimizationParams &params = {});

// Helper for async operations
template <typename Func, typename... Args>
[[nodiscard]] auto run_async(Func &&func, Args &&...args) {
  return std::async(std::launch::async, std::forward<Func>(func),
                    std::forward<Args>(args)...);
}

// Modern C++20 calibration class (forward declaration)
class CameraCalibrator;