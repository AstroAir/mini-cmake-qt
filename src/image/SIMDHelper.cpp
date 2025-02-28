#include "SIMDHelper.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <thread>


bool SIMDHelper::hasSSE = false;
bool SIMDHelper::hasAVX = false;

void SIMDHelper::checkCPUSupport() {
  try {
#if defined(_MSC_VER)
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    hasSSE = (cpuInfo[3] & (1 << 25)) || (cpuInfo[3] & (1 << 26));
    hasAVX = (cpuInfo[2] & (1 << 28)) != 0;
#else
    hasSSE = __builtin_cpu_supports("sse") || __builtin_cpu_supports("sse2");
    hasAVX = __builtin_cpu_supports("avx2");
#endif

    spdlog::debug("SIMD support: SSE={}, AVX={}", hasSSE, hasAVX);
  } catch (const std::exception &e) {
    spdlog::error("Failed to check CPU support: {}", e.what());
    // Default to false for safety
    hasSSE = false;
    hasAVX = false;
  }
}

bool SIMDHelper::validateConvolutionParams(const cv::Mat &input,
                                           const cv::Mat &output,
                                           const cv::Mat &kernel,
                                           const cv::Rect &roi) {
  // Check if matrices are empty
  if (input.empty() || kernel.empty()) {
    spdlog::error("Input or kernel matrix is empty");
    return false;
  }

  // Check kernel properties
  if (kernel.rows != kernel.cols || kernel.rows % 2 == 0) {
    spdlog::error("Kernel must be square with odd dimensions");
    return false;
  }

  // Check ROI is within image bounds
  if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > input.cols ||
      roi.y + roi.height > input.rows) {
    spdlog::error("ROI is out of image bounds");
    return false;
  }

  // Check compatible types
  if (output.type() != input.type()) {
    spdlog::error("Input and output must have the same type");
    return false;
  }

  return true;
}

void SIMDHelper::processImageTile(const cv::Mat &input, cv::Mat &output,
                                  const cv::Mat &kernel, const cv::Rect &roi) {
  static bool initialized = false;
  if (!initialized) {
    checkCPUSupport();
    initialized = true;
  }

  try {
    // Validate inputs
    if (!validateConvolutionParams(input, output, kernel, roi)) {
      throw std::invalid_argument("Invalid convolution parameters");
    }

    const int kernelSize = kernel.rows;
    const int stride = input.step1();

    // Create sub-matrices for the ROI if needed
    cv::Mat inputROI = input(roi);
    cv::Mat outputROI = output(roi);

    // Choose the appropriate SIMD implementation based on hardware support and
    // data type
    if (hasAVX && input.depth() == CV_32F) {
      spdlog::debug("Using AVX implementation for convolution");
      convolve2DAVX<float>(
          std::span<const float>(inputROI.ptr<float>(), inputROI.total()),
          std::span<float>(outputROI.ptr<float>(), outputROI.total()),
          std::span<const float>(kernel.ptr<float>(), kernel.total()),
          roi.height, roi.width, kernelSize, stride);
    } else if (hasSSE && input.depth() == CV_32F) {
      spdlog::debug("Using SSE implementation for convolution");
      convolve2DSSE<float>(
          std::span<const float>(inputROI.ptr<float>(), inputROI.total()),
          std::span<float>(outputROI.ptr<float>(), outputROI.total()),
          std::span<const float>(kernel.ptr<float>(), kernel.total()),
          roi.height, roi.width, kernelSize, stride);
    } else {
      spdlog::debug("Using OpenCV implementation for convolution");
      cv::filter2D(inputROI, outputROI, -1, kernel);
    }
  } catch (const std::exception &e) {
    spdlog::error("Error in processImageTile: {}", e.what());
    throw; // Rethrow to allow caller to handle
  }
}

template <SIMDCompatible T>
void SIMDHelper::convolve2DSSE(std::span<const T> input, std::span<T> output,
                               std::span<const float> kernel, int rows,
                               int cols, int kernelSize, int stride) {
  const int radius = kernelSize / 2;
  const __m128 zeros = _mm_setzero_ps();

  // Validate sizes
  if (input.size() < static_cast<size_t>(rows * stride) ||
      output.size() < static_cast<size_t>(rows * stride) ||
      kernel.size() < static_cast<size_t>(kernelSize * kernelSize)) {
    throw std::invalid_argument(
        "Span sizes are insufficient for the specified dimensions");
  }

  // Calculate optimal number of threads based on hardware
  const unsigned int numThreads =
      std::min(static_cast<unsigned int>(std::thread::hardware_concurrency()),
               static_cast<unsigned int>(rows - 2 * radius));

#pragma omp parallel for num_threads(numThreads)
  for (int i = radius; i < rows - radius; i++) {
    // Process 4 pixels at once with SSE
    for (int j = radius; j < cols - radius - 3; j += 4) {
      __m128 sum = zeros;

      for (int ky = 0; ky < kernelSize; ky++) {
        for (int kx = 0; kx < kernelSize; kx++) {
          const float kval = kernel[ky * kernelSize + kx];
          const T *pixel =
              &input[(i + ky - radius) * stride + (j + kx - radius)];
          __m128 pix = _mm_loadu_ps(reinterpret_cast<const float *>(pixel));
          __m128 k = _mm_set1_ps(kval);
          sum = _mm_add_ps(sum, _mm_mul_ps(pix, k));
        }
      }

      _mm_storeu_ps(reinterpret_cast<float *>(&output[i * stride + j]), sum);
    }

    // Handle remaining pixels (edge case)
    for (int j = std::max(radius, cols - radius - 3); j < cols - radius; j++) {
      float sum = 0.0f;
      for (int ky = 0; ky < kernelSize; ky++) {
        for (int kx = 0; kx < kernelSize; kx++) {
          sum += static_cast<float>(
                     input[(i + ky - radius) * stride + (j + kx - radius)]) *
                 kernel[ky * kernelSize + kx];
        }
      }
      output[i * stride + j] = static_cast<T>(sum);
    }
  }
}

template <SIMDCompatible T>
void SIMDHelper::convolve2DAVX(std::span<const T> input, std::span<T> output,
                               std::span<const float> kernel, int rows,
                               int cols, int kernelSize, int stride) {
  const int radius = kernelSize / 2;

  // Validate sizes
  if (input.size() < static_cast<size_t>(rows * stride) ||
      output.size() < static_cast<size_t>(rows * stride) ||
      kernel.size() < static_cast<size_t>(kernelSize * kernelSize)) {
    throw std::invalid_argument(
        "Span sizes are insufficient for the specified dimensions");
  }

  // Calculate optimal number of threads based on hardware
  const unsigned int numThreads =
      std::min(static_cast<unsigned int>(std::thread::hardware_concurrency()),
               static_cast<unsigned int>(rows - 2 * radius));

#ifdef __AVX__
  const __m256 zeros = _mm256_setzero_ps();

#pragma omp parallel for num_threads(numThreads)
  for (int i = radius; i < rows - radius; i++) {
    // Process 8 pixels at once with AVX
    for (int j = radius; j < cols - radius - 7; j += 8) {
      __m256 sum = zeros;

      for (int ky = 0; ky < kernelSize; ky++) {
        for (int kx = 0; kx < kernelSize; kx++) {
          const float kval = kernel[ky * kernelSize + kx];
          const T *pixel =
              &input[(i + ky - radius) * stride + (j + kx - radius)];
          __m256 pix = _mm256_loadu_ps(reinterpret_cast<const float *>(pixel));
          __m256 k = _mm256_set1_ps(kval);
          sum = _mm256_add_ps(sum, _mm256_mul_ps(pix, k));
        }
      }

      _mm256_storeu_ps(reinterpret_cast<float *>(&output[i * stride + j]), sum);
    }

    // Handle remaining pixels (edge case)
    for (int j = std::max(radius, cols - radius - 7); j < cols - radius; j++) {
      float sum = 0.0f;
      for (int ky = 0; ky < kernelSize; ky++) {
        for (int kx = 0; kx < kernelSize; kx++) {
          sum += static_cast<float>(
                     input[(i + ky - radius) * stride + (j + kx - radius)]) *
                 kernel[ky * kernelSize + kx];
        }
      }
      output[i * stride + j] = static_cast<T>(sum);
    }
  }
#else
  // Fall back to SSE implementation if AVX is not supported
  convolve2DSSE<T>(input, output, kernel, rows, cols, kernelSize, stride);
#endif
}

// Explicitly instantiate the templates for common types
template void SIMDHelper::convolve2DSSE<float>(std::span<const float>,
                                               std::span<float>,
                                               std::span<const float>, int, int,
                                               int, int);
template void SIMDHelper::convolve2DAVX<float>(std::span<const float>,
                                               std::span<float>,
                                               std::span<const float>, int, int,
                                               int, int);
