#ifndef IMAGE_NOISE_H
#define IMAGE_NOISE_H

#include <functional>
#include <opencv2/opencv.hpp>
#include <variant>
#include <vector>


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
};

struct SaltAndPepperParams {
  double density = 0.05;
  double salt_prob = 0.5;
};

struct SpeckleParams {
  double intensity = 0.1;
};

struct PeriodicParams {
  double amplitude = 10.0;
  double frequency = 0.1;
};

struct LaplacianParams {
  double scale = 1.0;
};

struct UniformParams {
  double low = -25.0; // Lower bound for uniform noise values
  double high = 25.0; // Upper bound for uniform noise values
};

// Variant for passing noise parameters
using NoiseParams =
    std::variant<GaussianParams, SaltAndPepperParams, SpeckleParams,
                 PeriodicParams, LaplacianParams, UniformParams>;

// Progress callback type
using ProgressCallback = std::function<void(float)>;

namespace imgnoise {

// Applies noise to the entire image based on the specified type and parameters.
void add_image_noise(const cv::Mat &input, cv::Mat &output, NoiseType type,
                     const NoiseParams &params,
                     ProgressCallback progress = nullptr);

// Applies a mix of noise types sequentially.
void add_mixed_noise(
    const cv::Mat &input, cv::Mat &output,
    const std::vector<std::pair<NoiseType, NoiseParams>> &noise_list,
    ProgressCallback progress = nullptr);

// Estimates the noise level of an image based on the standard deviation of the
// Laplacian.
double estimate_noise_level(const cv::Mat &image);

// Applies noise only to a specified region of interest (ROI)
// The noise is applied only to the ROI and the remaining image is left
// unchanged.
void add_roi_noise(const cv::Mat &input, cv::Mat &output, const cv::Rect &roi,
                   NoiseType type, const NoiseParams &params,
                   ProgressCallback progress = nullptr);

} // namespace imgnoise

#endif // IMAGE_NOISE_H