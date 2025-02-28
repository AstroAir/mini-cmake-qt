#ifndef IMAGE_NOISE_H
#define IMAGE_NOISE_H

#include <functional>
#include <opencv2/opencv.hpp>
#include <span>
#include <variant>

// Concept for valid image types
template <typename T>
concept ValidImageType = std::is_same_v<T, cv::Mat>;

// Enum for different types of noise
enum class NoiseType {
  GAUSSIAN,
  SALT_AND_PEPPER,
  POISSON,
  SPECKLE,
  PERIODIC,
  LAPLACIAN,
  UNIFORM
};

// Parameter structures for noise functions
struct GaussianParams {
  double mean = 0.0;
  double stddev = 25.0;

  // Validate parameters
  [[nodiscard]] bool is_valid() const noexcept { return stddev >= 0.0; }
};

struct SaltAndPepperParams {
  double density = 0.05;
  double salt_prob = 0.5;

  // Validate parameters
  [[nodiscard]] bool is_valid() const noexcept {
    return density >= 0.0 && density <= 1.0 && salt_prob >= 0.0 &&
           salt_prob <= 1.0;
  }
};

struct SpeckleParams {
  double intensity = 0.1;

  // Validate parameters
  [[nodiscard]] bool is_valid() const noexcept { return intensity >= 0.0; }
};

struct PeriodicParams {
  double amplitude = 10.0;
  double frequency = 0.1;

  // Validate parameters
  [[nodiscard]] bool is_valid() const noexcept {
    return amplitude >= 0.0 && frequency > 0.0;
  }
};

struct LaplacianParams {
  double scale = 1.0;

  // Validate parameters
  [[nodiscard]] bool is_valid() const noexcept { return scale > 0.0; }
};

struct UniformParams {
  double low = -25.0; // Lower bound for uniform noise values
  double high = 25.0; // Upper bound for uniform noise values

  // Validate parameters
  [[nodiscard]] bool is_valid() const noexcept { return high > low; }
};

// Variant for passing noise parameters
using NoiseParams =
    std::variant<GaussianParams, SaltAndPepperParams, SpeckleParams,
                 PeriodicParams, LaplacianParams, UniformParams>;

// Progress callback type
using ProgressCallback = std::function<void(float)>;

namespace imgnoise {

// Exception class for noise operations
class NoiseException : public std::runtime_error {
public:
  explicit NoiseException(const std::string &what_arg)
      : std::runtime_error(what_arg) {}
  explicit NoiseException(const char *what_arg)
      : std::runtime_error(what_arg) {}
};

// Validate parameters based on noise type
[[nodiscard]] bool validate_params(NoiseType type,
                                   const NoiseParams &params) noexcept;

// Applies noise to the entire image based on the specified type and parameters.
void add_image_noise(const cv::Mat &input, cv::Mat &output, NoiseType type,
                     const NoiseParams &params,
                     ProgressCallback progress = nullptr);

// Applies a mix of noise types sequentially.
void add_mixed_noise(
    const cv::Mat &input, cv::Mat &output,
    std::span<const std::pair<NoiseType, NoiseParams>> noise_list,
    ProgressCallback progress = nullptr);

// Estimates the noise level of an image based on the standard deviation of the
// Laplacian.
[[nodiscard]] double estimate_noise_level(const cv::Mat &image) noexcept;

// Applies noise only to a specified region of interest (ROI)
// The noise is applied only to the ROI and the remaining image is left
// unchanged.
void add_roi_noise(const cv::Mat &input, cv::Mat &output, const cv::Rect &roi,
                   NoiseType type, const NoiseParams &params,
                   ProgressCallback progress = nullptr);

// Template function to validate input image
template <ValidImageType Image> void validate_image(const Image &img) {
  if (img.empty()) {
    throw NoiseException("Input image is empty");
  }
  if (img.depth() != CV_8U && img.depth() != CV_32F) {
    throw NoiseException("Unsupported image depth");
  }
}

// Template function to validate ROI
template <ValidImageType Image>
void validate_roi(const Image &img, const cv::Rect &roi) {
  validate_image(img);
  if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > img.cols ||
      roi.y + roi.height > img.rows) {
    throw NoiseException("ROI is out of image bounds");
  }
}

} // namespace imgnoise

#endif // IMAGE_NOISE_H