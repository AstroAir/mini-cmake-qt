#include "Noise.hpp"
#include <omp.h>
#include <random>
#include <spdlog/spdlog.h>


namespace {

// Internal helper functions in an anonymous namespace

namespace detail {

// Gaussian noise using OpenCV's randn (vectorized)
void add_gaussian_noise(const cv::Mat &input, cv::Mat &output,
                        const GaussianParams &params) {
  cv::Mat noise(input.size(), input.type());
  cv::randn(noise, params.mean, params.stddev);
  cv::add(input, noise, output);
  output.convertTo(output, input.type());
}

// Salt and Pepper noise using random pixel selection
void add_salt_and_pepper(const cv::Mat &input, cv::Mat &output,
                         const SaltAndPepperParams &params) {
  input.copyTo(output);
  const int num_pixels = input.rows * input.cols * input.channels();
  const int num_salt =
      static_cast<int>(num_pixels * params.density * params.salt_prob);
  const int num_pepper =
      static_cast<int>(num_pixels * params.density * (1 - params.salt_prob));
  static thread_local std::mt19937 gen{std::random_device{}()};
  std::uniform_int_distribution<> dis(0, num_pixels - 1);

  for (int i = 0; i < num_salt; ++i) {
    const int idx = dis(gen);
    output.data[idx] = 255;
  }
  for (int i = 0; i < num_pepper; ++i) {
    const int idx = dis(gen);
    output.data[idx] = 0;
  }
}

// Poisson noise (simplified version)
// Note: More advanced simulation would use the actual Poisson distribution.
void add_poisson_noise(const cv::Mat &input, cv::Mat &output) {
  cv::Mat noise(input.size(), input.type());
  cv::randn(noise, 0, 1);
  noise = noise * 0.5 + 0.5; // Normalize to [0,1)
  cv::Mat exp_input;
  cv::exp(-input, exp_input);
  cv::Mat thresholded;
  cv::threshold(exp_input, thresholded, 0.5, 255, cv::THRESH_BINARY);
  cv::multiply(input, thresholded, output);
  output.convertTo(output, input.type());
}

// Speckle noise using multiplicative noise
void add_speckle_noise(const cv::Mat &input, cv::Mat &output,
                       const SpeckleParams &params) {
  cv::Mat noise(input.size(), CV_32F);
  cv::randn(noise, 0, params.intensity);
  input.convertTo(output, CV_32F);
  output = output.mul(1.0f + noise);
  output.convertTo(output, input.type());
}

// Periodic noise with a sinusoidal pattern (parallelized with OpenMP)
void add_periodic_noise(const cv::Mat &input, cv::Mat &output,
                        const PeriodicParams &params) {
  output = input.clone();
#pragma omp parallel for
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      double noise = params.amplitude *
                     std::sin(2 * CV_PI * params.frequency * i) *
                     std::cos(2 * CV_PI * params.frequency * j);
      for (int c = 0; c < input.channels(); c++) {
        int value = cv::saturate_cast<uchar>(
            input.at<uchar>(i, j * input.channels() + c) + noise);
        output.at<uchar>(i, j * input.channels() + c) = value;
      }
    }
  }
}

// Laplacian noise using an exponential distribution (parallelized with OpenMP)
void add_laplacian_noise(const cv::Mat &input, cv::Mat &output,
                         const LaplacianParams &params) {
  cv::Mat noise(input.size(), input.type());
  static thread_local std::mt19937 gen{std::random_device{}()};
  std::exponential_distribution<> d(1.0 / params.scale);
#pragma omp parallel for
  for (int i = 0; i < noise.total(); i++) {
    noise.data[i] = cv::saturate_cast<uchar>(d(gen));
  }
  cv::add(input, noise, output);
}

// Uniform noise based on a uniform distribution.
void add_uniform_noise(const cv::Mat &input, cv::Mat &output,
                       const UniformParams &params) {
  cv::Mat noise(input.size(), input.type());
  cv::randu(noise, params.low, params.high);
  cv::add(input, noise, output);
  output.convertTo(output, input.type());
}

} // end namespace detail

} // end anonymous namespace

namespace imgnoise {

void add_image_noise(const cv::Mat &input, cv::Mat &output, NoiseType type,
                     const NoiseParams &params, ProgressCallback progress) {
  if (input.empty()) {
    spdlog::error("Input image is empty");
    throw std::invalid_argument("Input image is empty");
  }
  if (input.depth() != CV_8U && input.depth() != CV_32F) {
    spdlog::error("Unsupported image depth: {}", input.depth());
    throw std::invalid_argument("Unsupported image depth");
  }
  try {
    switch (type) {
    case NoiseType::GAUSSIAN: {
      const auto &p = std::get<GaussianParams>(params);
      if (p.stddev < 0) {
        spdlog::error("Invalid standard deviation: {}", p.stddev);
        throw std::invalid_argument("Negative standard deviation");
      }
      detail::add_gaussian_noise(input, output, p);
      break;
    }
    case NoiseType::SALT_AND_PEPPER: {
      const auto &p = std::get<SaltAndPepperParams>(params);
      if (p.density < 0 || p.density > 1) {
        spdlog::error("Invalid density: {}", p.density);
        throw std::invalid_argument("Density out of [0,1] range");
      }
      detail::add_salt_and_pepper(input, output, p);
      break;
    }
    case NoiseType::POISSON:
      detail::add_poisson_noise(input, output);
      break;
    case NoiseType::SPECKLE: {
      const auto &p = std::get<SpeckleParams>(params);
      detail::add_speckle_noise(input, output, p);
      break;
    }
    case NoiseType::PERIODIC: {
      const auto &p = std::get<PeriodicParams>(params);
      detail::add_periodic_noise(input, output, p);
      break;
    }
    case NoiseType::LAPLACIAN: {
      const auto &p = std::get<LaplacianParams>(params);
      detail::add_laplacian_noise(input, output, p);
      break;
    }
    case NoiseType::UNIFORM: {
      const auto &p = std::get<UniformParams>(params);
      detail::add_uniform_noise(input, output, p);
      break;
    }
    default:
      spdlog::error("Unknown noise type: {}", static_cast<int>(type));
      throw std::invalid_argument("Unknown noise type");
    }
    if (progress) {
      progress(1.0f);
    }
  } catch (const cv::Exception &e) {
    spdlog::error("OpenCV exception: {}", e.what());
    throw;
  } catch (const std::bad_variant_access &) {
    spdlog::error("Parameter type mismatch for noise type");
    throw;
  }

  // Normalize output if output memory differs from input memory.
  if (input.data != output.data) {
    cv::normalize(output, output, 0, 255, cv::NORM_MINMAX, input.type());
  }
}

void add_mixed_noise(
    const cv::Mat &input, cv::Mat &output,
    const std::vector<std::pair<NoiseType, NoiseParams>> &noise_list,
    ProgressCallback progress) {
  if (input.empty()) {
    spdlog::error("Input image is empty");
    throw std::invalid_argument("Input image is empty");
  }
  cv::Mat temp = input.clone();
  int total = static_cast<int>(noise_list.size());
  for (size_t i = 0; i < noise_list.size(); ++i) {
    NoiseType type = noise_list[i].first;
    NoiseParams params = noise_list[i].second;
    add_image_noise(temp, temp, type, params);
    if (progress) {
      progress(static_cast<float>(i + 1) / total);
    }
  }
  output = temp;
}

double estimate_noise_level(const cv::Mat &image) {
  cv::Mat gray;
  if (image.channels() > 1) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = image;
  }
  cv::Mat laplacian;
  cv::Laplacian(gray, laplacian, CV_64F);
  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);
  return stddev[0];
}

void add_roi_noise(const cv::Mat &input, cv::Mat &output, const cv::Rect &roi,
                   NoiseType type, const NoiseParams &params,
                   ProgressCallback progress) {
  if (input.empty()) {
    spdlog::error("Input image is empty");
    throw std::invalid_argument("Input image is empty");
  }
  if ((roi.x < 0) || (roi.y < 0) || (roi.x + roi.width > input.cols) ||
      (roi.y + roi.height > input.rows)) {
    spdlog::error("ROI is out of image bounds");
    throw std::invalid_argument("ROI is out of image bounds");
  }
  // Start with the full image copy.
  input.copyTo(output);
  // Extract ROI from output, apply noise on that area
  cv::Mat roiMat = output(roi);
  cv::Mat noisyRoi;
  add_image_noise(roiMat, noisyRoi, type, params, progress);
  noisyRoi.copyTo(roiMat);
}

} // namespace imgnoise