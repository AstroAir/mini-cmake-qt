#include "Noise.hpp"
#include <algorithm>
#include <execution>
#include <omp.h>
#include <random>
#include <spdlog/spdlog.h>

// Check if we can use SIMD
#if defined(__AVX__) || defined(__AVX2__) || defined(__SSE__) ||               \
    defined(__SSE2__)
#include <immintrin.h>
#define HAS_SIMD 1
#else
#define HAS_SIMD 0
#endif

namespace {

// Thread-local random generator with proper seeding
std::mt19937 &get_random_generator() {
  static thread_local std::mt19937 gen(
      std::random_device{}() ^
      static_cast<unsigned int>(
          std::hash<std::thread::id>{}(std::this_thread::get_id())));
  return gen;
}

// Internal helper functions
namespace detail {

// Gaussian noise using OpenCV's randn with optimizations
void add_gaussian_noise(const cv::Mat &input, cv::Mat &output,
                        const GaussianParams &params) {
  // Use preallocated output if possible
  cv::Mat noise = cv::Mat::zeros(input.size(), CV_32F);

// Use parallel execution of randn via OpenMP
#pragma omp parallel sections
  {
#pragma omp section
    {
      cv::randn(noise, params.mean, params.stddev);
    }
  }

  if (output.empty() || output.size() != input.size() ||
      output.type() != input.type()) {
    output.create(input.size(), input.type());
  }

  // Add noise using parallel algorithms
  cv::add(input, noise, output);
  output.convertTo(output, input.type());
}

// Salt and Pepper noise with SIMD and parallel optimizations
void add_salt_and_pepper(const cv::Mat &input, cv::Mat &output,
                         const SaltAndPepperParams &params) {
  input.copyTo(output);
  const int num_pixels = static_cast<int>(input.total() * input.channels());
  const int num_salt =
      static_cast<int>(num_pixels * params.density * params.salt_prob);
  const int num_pepper =
      static_cast<int>(num_pixels * params.density * (1 - params.salt_prob));

  // Vector of indices for parallel processing
  std::vector<int> salt_indices(num_salt);
  std::vector<int> pepper_indices(num_pepper);

  // Generate random indices in parallel
  auto gen = get_random_generator();
  std::uniform_int_distribution<> dis(0, num_pixels - 1);

#pragma omp parallel for
  for (int i = 0; i < num_salt; ++i) {
    auto local_gen = get_random_generator();
    salt_indices[i] = dis(local_gen);
  }

#pragma omp parallel for
  for (int i = 0; i < num_pepper; ++i) {
    auto local_gen = get_random_generator();
    pepper_indices[i] = dis(local_gen);
  }

// Apply salt (255) and pepper (0) values in parallel
#pragma omp parallel sections
  {
#pragma omp section
    {
      std::for_each(
          std::execution::par_unseq, salt_indices.begin(), salt_indices.end(),
          [&output](int idx) {
            if (idx >= 0 &&
                idx < static_cast<int>(output.total() * output.channels())) {
              output.data[idx] = 255;
            }
          });
    }

#pragma omp section
    {
      std::for_each(
          std::execution::par_unseq, pepper_indices.begin(),
          pepper_indices.end(), [&output](int idx) {
            if (idx >= 0 &&
                idx < static_cast<int>(output.total() * output.channels())) {
              output.data[idx] = 0;
            }
          });
    }
  }
}

// Improved Poisson noise implementation
void add_poisson_noise(const cv::Mat &input, cv::Mat &output) {
  // Create output if needed
  if (output.empty() || output.size() != input.size()) {
    output = cv::Mat(input.size(), input.type());
  }

  // Convert to float for processing
  cv::Mat float_input;
  input.convertTo(float_input, CV_32F);

  // Create a lambda for parallel noise application
  auto apply_poisson = [](const cv::Mat &src, cv::Mat &dst) {
    auto gen = get_random_generator();

#pragma omp parallel for collapse(2)
    for (int i = 0; i < src.rows; ++i) {
      for (int j = 0; j < src.cols; ++j) {
        auto local_gen = get_random_generator();
        for (int c = 0; c < src.channels(); ++c) {
          float pixel_val = src.at<float>(i, j * src.channels() + c);
          // Use actual Poisson distribution for better simulation
          std::poisson_distribution<int> d(std::max(1.0f, pixel_val));
          dst.at<float>(i, j * dst.channels() + c) =
              static_cast<float>(d(local_gen));
        }
      }
    }
  };

  // Apply noise
  cv::Mat float_output(float_input.size(), CV_32F);
  apply_poisson(float_input, float_output);

  // Convert back to original type
  float_output.convertTo(output, input.type());
}

// Optimized speckle noise implementation using SIMD when available
void add_speckle_noise(const cv::Mat &input, cv::Mat &output,
                       const SpeckleParams &params) {
  cv::Mat noise(input.size(), CV_32F);
  cv::randn(noise, 0, params.intensity);

  input.convertTo(output, CV_32F);

#if HAS_SIMD
  // Use SIMD for multiplication when possible
  if (input.isContinuous() && noise.isContinuous()) {
    const int total = static_cast<int>(input.total() * input.channels());
    const int step = 8; // process 8 floats at a time with AVX

#pragma omp parallel for
    for (int i = 0; i < total - step + 1; i += step) {
      __m256 in =
          _mm256_loadu_ps(reinterpret_cast<const float *>(output.data) + i);
      __m256 n =
          _mm256_loadu_ps(reinterpret_cast<const float *>(noise.data) + i);
      __m256 one = _mm256_set1_ps(1.0f);
      __m256 result = _mm256_mul_ps(in, _mm256_add_ps(one, n));
      _mm256_storeu_ps(reinterpret_cast<float *>(output.data) + i, result);
    }

    // Handle remaining elements
    for (int i = total - (total % step); i < total; ++i) {
      reinterpret_cast<float *>(output.data)[i] *=
          (1.0f + reinterpret_cast<float *>(noise.data)[i]);
    }
  } else {
    output = output.mul(1.0f + noise);
  }
#else
  output = output.mul(1.0f + noise);
#endif

  output.convertTo(output, input.type());
}

// Periodic noise with vectorization
void add_periodic_noise(const cv::Mat &input, cv::Mat &output,
                        const PeriodicParams &params) {
  if (output.empty() || output.size() != input.size() ||
      output.type() != input.type()) {
    output = input.clone();
  } else {
    input.copyTo(output);
  }

  // Pre-compute sin/cos values for better performance
  std::vector<float> sin_values(input.rows);
  std::vector<float> cos_values(input.cols);

#pragma omp parallel sections
  {
#pragma omp section
    {
      for (int i = 0; i < input.rows; i++) {
        sin_values[i] =
            static_cast<float>(std::sin(2 * CV_PI * params.frequency * i));
      }
    }

#pragma omp section
    {
      for (int j = 0; j < input.cols; j++) {
        cos_values[j] =
            static_cast<float>(std::cos(2 * CV_PI * params.frequency * j));
      }
    }
  }

#pragma omp parallel for collapse(2)
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      float noise =
          static_cast<float>(params.amplitude * sin_values[i] * cos_values[j]);
      for (int c = 0; c < input.channels(); c++) {
        int idx = i * input.step + j * input.elemSize() + c;
        int value = cv::saturate_cast<uchar>(input.data[idx] + noise);
        output.data[idx] = static_cast<uchar>(value);
      }
    }
  }
}

// Laplacian noise with improved random generation
void add_laplacian_noise(const cv::Mat &input, cv::Mat &output,
                         const LaplacianParams &params) {
  // Create output if needed
  if (output.empty() || output.size() != input.size() ||
      output.type() != input.type()) {
    output = cv::Mat(input.size(), input.type());
  }

  input.copyTo(output);

  // Using a more efficient approach with vectorization
  const int total_pixels = static_cast<int>(input.total() * input.channels());
  std::vector<float> noise_values(total_pixels);

// Generate Laplacian noise values
#pragma omp parallel
  {
    auto local_gen = get_random_generator();
    std::exponential_distribution<float> d(1.0f /
                                           static_cast<float>(params.scale));

#pragma omp for
    for (int i = 0; i < total_pixels; ++i) {
      float r = d(local_gen);
      noise_values[i] =
          (rand() % 2 == 0) ? r : -r; // Create Laplacian from exponential
    }
  }

// Apply noise
#pragma omp parallel for
  for (int i = 0; i < total_pixels; ++i) {
    int value = cv::saturate_cast<uchar>(input.data[i] + noise_values[i]);
    output.data[i] = static_cast<uchar>(value);
  }
}

// Uniform noise with parallel processing
void add_uniform_noise(const cv::Mat &input, cv::Mat &output,
                       const UniformParams &params) {
  // Create a uniform noise matrix
  cv::Mat noise(input.size(), input.type());
  cv::randu(noise, params.low, params.high);

  // Add the noise to the input
  cv::add(input, noise, output, cv::noArray(), input.type());
}

} // end namespace detail

} // end anonymous namespace

namespace imgnoise {

bool validate_params(NoiseType type, const NoiseParams &params) noexcept {
  try {
    switch (type) {
    case NoiseType::GAUSSIAN:
      return std::get<GaussianParams>(params).is_valid();
    case NoiseType::SALT_AND_PEPPER:
      return std::get<SaltAndPepperParams>(params).is_valid();
    case NoiseType::POISSON:
      return true; // No parameters to validate
    case NoiseType::SPECKLE:
      return std::get<SpeckleParams>(params).is_valid();
    case NoiseType::PERIODIC:
      return std::get<PeriodicParams>(params).is_valid();
    case NoiseType::LAPLACIAN:
      return std::get<LaplacianParams>(params).is_valid();
    case NoiseType::UNIFORM:
      return std::get<UniformParams>(params).is_valid();
    default:
      return false;
    }
  } catch (const std::bad_variant_access &) {
    return false;
  }
}

void add_image_noise(const cv::Mat &input, cv::Mat &output, NoiseType type,
                     const NoiseParams &params, ProgressCallback progress) {
  try {
    // Validate input
    validate_image(input);

    // Validate parameters
    if (!validate_params(type, params)) {
      throw NoiseException("Invalid parameters for the specified noise type");
    }

    switch (type) {
    case NoiseType::GAUSSIAN: {
      const auto &p = std::get<GaussianParams>(params);
      detail::add_gaussian_noise(input, output, p);
      break;
    }
    case NoiseType::SALT_AND_PEPPER: {
      const auto &p = std::get<SaltAndPepperParams>(params);
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
      throw NoiseException("Unknown noise type");
    }

    if (progress) {
      progress(1.0f);
    }
  } catch (const cv::Exception &e) {
    spdlog::error("OpenCV exception in add_image_noise: {}", e.what());
    throw NoiseException(std::string("OpenCV error: ") + e.what());
  } catch (const std::bad_variant_access &e) {
    spdlog::error("Parameter type mismatch for noise type");
    throw NoiseException("Parameter type mismatch for noise type");
  } catch (const NoiseException &) {
    throw; // Re-throw our exceptions
  } catch (const std::exception &e) {
    spdlog::error("Standard exception in add_image_noise: {}", e.what());
    throw NoiseException(std::string("Standard error: ") + e.what());
  }
}

void add_mixed_noise(
    const cv::Mat &input, cv::Mat &output,
    std::span<const std::pair<NoiseType, NoiseParams>> noise_list,
    ProgressCallback progress) {
  try {
    validate_image(input);

    if (noise_list.empty()) {
      input.copyTo(output);
      if (progress)
        progress(1.0f);
      return;
    }

    // Create a working copy
    cv::Mat temp = input.clone();
    const int total = static_cast<int>(noise_list.size());

    // Apply each noise type sequentially
    for (size_t i = 0; i < noise_list.size(); ++i) {
      const auto &[type, params] = noise_list[i];

      if (!validate_params(type, params)) {
        throw NoiseException("Invalid parameters for noise at index " +
                             std::to_string(i));
      }

      // Create a lambda for progress tracking
      auto sub_progress = [&progress, total, i](float p) {
        if (progress) {
          progress((static_cast<float>(i) + p) / total);
        }
      };

      add_image_noise(temp, temp, type, params, sub_progress);
    }

    output = temp;
  } catch (const cv::Exception &e) {
    spdlog::error("OpenCV exception in add_mixed_noise: {}", e.what());
    throw NoiseException(std::string("OpenCV error: ") + e.what());
  } catch (const NoiseException &) {
    throw; // Re-throw our exceptions
  } catch (const std::exception &e) {
    spdlog::error("Standard exception in add_mixed_noise: {}", e.what());
    throw NoiseException(std::string("Standard error: ") + e.what());
  }
}

double estimate_noise_level(const cv::Mat &image) noexcept {
  try {
    if (image.empty()) {
      spdlog::warn("Empty image provided to estimate_noise_level");
      return 0.0;
    }

    cv::Mat gray;
    if (image.channels() > 1) {
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
      gray = image.clone();
    }

    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);

    return stddev[0];
  } catch (const std::exception &e) {
    spdlog::error("Error in estimate_noise_level: {}", e.what());
    return 0.0;
  }
}

void add_roi_noise(const cv::Mat &input, cv::Mat &output, const cv::Rect &roi,
                   NoiseType type, const NoiseParams &params,
                   ProgressCallback progress) {
  try {
    // Validate input and ROI
    validate_roi(input, roi);

    // Validate parameters
    if (!validate_params(type, params)) {
      throw NoiseException("Invalid parameters for the specified noise type");
    }

    // Start with a copy of the input
    input.copyTo(output);

    // Extract ROI from output for processing
    cv::Mat roiMat = output(roi);
    cv::Mat noisyRoi;

    // Apply noise only to the ROI
    add_image_noise(roiMat, noisyRoi, type, params, progress);

    // Copy the noisy ROI back to the output
    noisyRoi.copyTo(roiMat);
  } catch (const cv::Exception &e) {
    spdlog::error("OpenCV exception in add_roi_noise: {}", e.what());
    throw NoiseException(std::string("OpenCV error: ") + e.what());
  } catch (const NoiseException &) {
    throw; // Re-throw our exceptions
  } catch (const std::exception &e) {
    spdlog::error("Standard exception in add_roi_noise: {}", e.what());
    throw NoiseException(std::string("Standard error: ") + e.what());
  }
}

} // namespace imgnoise