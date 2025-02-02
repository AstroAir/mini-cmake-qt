#include "SIMDHelper.hpp"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

bool SIMDHelper::hasSSE = false;
bool SIMDHelper::hasAVX = false;

void SIMDHelper::checkCPUSupport() {
#if defined(_MSC_VER)
  int cpuInfo[4];
  __cpuid(cpuInfo, 1);
  hasSSE = (cpuInfo[3] & (1 << 25)) || (cpuInfo[3] & (1 << 26));
  hasAVX = (cpuInfo[2] & (1 << 28)) != 0;
#else
  hasSSE = __builtin_cpu_supports("sse") || __builtin_cpu_supports("sse2");
  hasAVX = __builtin_cpu_supports("avx2");
#endif
}

void SIMDHelper::processImageTile(const cv::Mat &input, cv::Mat &output,
                                  const cv::Mat &kernel, const cv::Rect &roi) {
  static bool initialized = false;
  if (!initialized) {
    checkCPUSupport();
    initialized = true;
  }

  const int kernelSize = kernel.rows;
  const int stride = input.step1();

  if (hasAVX && input.depth() == CV_32F) {
    convolve2DAVX<float>(input.ptr<float>(), output.ptr<float>(),
                         kernel.ptr<float>(), roi.height, roi.width, kernelSize,
                         stride);
  } else if (hasSSE && input.depth() == CV_32F) {
    convolve2DSSE<float>(input.ptr<float>(), output.ptr<float>(),
                         kernel.ptr<float>(), roi.height, roi.width, kernelSize,
                         stride);
  } else {
    cv::filter2D(input, output, -1, kernel);
  }
}

template <typename T>
void SIMDHelper::convolve2DSSE(const T *input, T *output, const float *kernel,
                               int rows, int cols, int kernelSize, int stride) {
  const int radius = kernelSize / 2;
  const __m128 zeros = _mm_setzero_ps();

#pragma omp parallel for
  for (int i = radius; i < rows - radius; i++) {
    for (int j = radius; j < cols - radius - 3; j += 4) {
      __m128 sum = zeros;

      for (int ky = 0; ky < kernelSize; ky++) {
        for (int kx = 0; kx < kernelSize; kx++) {
          const float kval = kernel[ky * kernelSize + kx];
          const T *pixel =
              input + (i + ky - radius) * stride + (j + kx - radius);
          __m128 pix = _mm_loadu_ps(reinterpret_cast<const float *>(pixel));
          __m128 k = _mm_set1_ps(kval);
          sum = _mm_add_ps(sum, _mm_mul_ps(pix, k));
        }
      }

      _mm_storeu_ps(reinterpret_cast<float *>(output + i * stride + j), sum);
    }
  }
}

template <typename T>
void SIMDHelper::convolve2DAVX(const T *input, T *output, const float *kernel,
                               int rows, int cols, int kernelSize, int stride) {
  const int radius = kernelSize / 2;
  
#ifdef __AVX__
  #pragma GCC target("avx")
  const __m256 zeros = _mm256_setzero_ps();
  
  #pragma omp parallel for
  for (int i = radius; i < rows - radius; i++) {
    for (int j = radius; j < cols - radius - 7; j += 8) {
      __m256 sum = zeros;
      
      for (int ky = 0; ky < kernelSize; ky++) {
        for (int kx = 0; kx < kernelSize; kx++) {
          const float kval = kernel[ky * kernelSize + kx];
          const T *pixel = input + (i + ky - radius) * stride + (j + kx - radius);
          __m256 pix = _mm256_loadu_ps(reinterpret_cast<const float *>(pixel));
          __m256 k = _mm256_set1_ps(kval);
          sum = _mm256_add_ps(sum, _mm256_mul_ps(pix, k));
        }
      }
      
      _mm256_storeu_ps(reinterpret_cast<float *>(output + i * stride + j), sum);
    }
  }
#else
  // 如果不支持AVX，回退到SSE实现
  convolve2DSSE<T>(input, output, kernel, rows, cols, kernelSize, stride);
#endif
}
