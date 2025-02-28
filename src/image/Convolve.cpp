#include "Convolve.hpp"
#include "../utils/ThreadPool.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <fmt/format.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <ranges>
#include <semaphore>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <thread>
#include <variant>

namespace {
std::shared_ptr<spdlog::logger> convolveLogger =
    spdlog::basic_logger_mt("ConvolveLogger", "logs/convolve.log");

// Exception-safe logging helper
template <typename Func> auto log_exceptions(Func &&func) {
  try {
    return std::forward<Func>(func)();
  } catch (const std::exception &e) {
    convolveLogger->error("Exception caught: {}", e.what());
    throw; // rethrow after logging
  } catch (...) {
    convolveLogger->error("Unknown exception caught");
    throw;
  }
}

// Helper to check if the system supports AVX
bool hasAVXSupport() noexcept {
#if defined(__AVX__) || defined(__AVX2__)
  return true;
#else
  return false;
#endif
}

// Helper to safely check if matrix is valid
bool isValidMatrix(const cv::Mat &mat) noexcept {
  return !mat.empty() && mat.data != nullptr;
}

// Template function to bound a value
template <typename T>
constexpr T clamp(T value, T min_val, T max_val) noexcept {
  return std::min(std::max(value, min_val), max_val);
}

// Parallel algorithm helper using C++20 ranges
template <std::ranges::range Range,
          std::invocable<std::ranges::range_value_t<Range>> Func>
void for_each_parallel(Range &&range, Func &&func) {
  std::for_each(std::ranges::begin(range), std::ranges::end(range),
                std::forward<Func>(func));
}
} // namespace

std::vector<std::shared_ptr<cv::Mat>> Convolve::MemoryPool::pool_;
std::mutex Convolve::MemoryPool::pool_mutex_;

std::expected<cv::Mat, ProcessError> Convolve::process(
    const cv::Mat &input,
    const std::variant<ConvolutionConfig, DeconvolutionConfig> &config) {
  if (!isValidMatrix(input)) {
    return std::unexpected(ProcessError{ProcessError::Code::INVALID_INPUT,
                                        "Input matrix is empty or invalid"});
  }

  return std::visit(
      [&](auto &&cfg) -> std::expected<cv::Mat, ProcessError> {
        using T = std::decay_t<decltype(cfg)>;

        if constexpr (std::is_same_v<T, ConvolutionConfig>) {
          return convolve(input, cfg);
        } else if constexpr (std::is_same_v<T, DeconvolutionConfig>) {
          return deconvolve(input, cfg);
        } else {
          return std::unexpected(
              ProcessError{ProcessError::Code::UNSUPPORTED_OPERATION,
                           "Unsupported configuration type"});
        }
      },
      config);
}

std::future<std::expected<cv::Mat, ProcessError>> Convolve::processAsync(
    const cv::Mat &input,
    const std::variant<ConvolutionConfig, DeconvolutionConfig> &config) {
  return std::async(std::launch::async, [input = input.clone(), config]() {
    return process(input, config);
  });
}

void Convolve::cleanup() { MemoryPool::clear(); }

Convolve::ScopedTimer::ScopedTimer(std::string_view op)
    : operation(op), start(std::chrono::steady_clock::now()) {
  convolveLogger->debug("Starting operation: {}", operation);
}

Convolve::ScopedTimer::~ScopedTimer() {
  auto duration = std::chrono::steady_clock::now() - start;
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
  convolveLogger->debug("{} took {} ms", operation, ms.count());
}

std::expected<cv::Mat, ProcessError> Convolve::processMultiChannel(
    const cv::Mat &input,
    const std::function<std::expected<cv::Mat, ProcessError>(const cv::Mat &)>
        &processor) {
  try {
    ScopedTimer timer("Multi-channel Processing");
    convolveLogger->info("Processing multi-channel image with {} channels",
                         input.channels());

    if (!isValidMatrix(input)) {
      return std::unexpected(ProcessError{ProcessError::Code::INVALID_INPUT,
                                          "Input matrix is empty or invalid"});
    }

    std::vector<cv::Mat> channels;
    cv::split(input, channels);

    // Use C++20 parallel algorithms with proper error handling
    std::vector<std::expected<cv::Mat, ProcessError>> results(channels.size());

    std::counting_semaphore<> completion(0);
    std::mutex error_mutex;
    std::optional<ProcessError> first_error;

    for (size_t i = 0; i < channels.size(); i++) {
      std::thread([&, i]() {
        try {
          results[i] = processor(channels[i]);
          if (!results[i].has_value()) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!first_error) {
              first_error = results[i].error();
            }
          }
        } catch (const std::exception &e) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!first_error) {
            first_error = ProcessError{
                ProcessError::Code::PROCESSING_FAILED,
                fmt::format("Exception in channel processing: {}", e.what())};
          }
        }
        completion.release();
      }).detach();
    }

    // Wait for all threads
    for (size_t i = 0; i < channels.size(); i++) {
      completion.acquire();
    }

    // If any channel had an error, return the first error
    if (first_error) {
      return std::unexpected(*first_error);
    }

    // Check all results
    for (size_t i = 0; i < results.size(); i++) {
      if (!results[i].has_value()) {
        return std::unexpected(results[i].error());
      }
      channels[i] = results[i].value();
    }

    cv::Mat result;
    cv::merge(channels, result);
    convolveLogger->info("Channels merged back into a single image");
    return result;
  } catch (const cv::Exception &e) {
    convolveLogger->error("OpenCV exception: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("OpenCV exception: {}", e.what())});
  } catch (const std::exception &e) {
    convolveLogger->error("Standard exception: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("Standard exception: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in processMultiChannel");
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     "Unknown exception in processMultiChannel"});
  }
}

std::expected<cv::Mat, ProcessError>
Convolve::convolve(const cv::Mat &input, const ConvolutionConfig &cfg) {
  try {
    convolveLogger->info("Starting convolution process");

    // Validate inputs
    auto validate_result = validateConvolutionConfig(input, cfg);
    if (!validate_result.has_value()) {
      return std::unexpected(validate_result.error());
    }

    if (input.channels() > 1 && cfg.per_channel) {
      convolveLogger->debug("Processing convolution per channel");
      return processMultiChannel(input, [&](const cv::Mat &channel) {
        return convolveSingleChannel(channel, cfg);
      });
    }

    convolveLogger->debug("Processing convolution on entire image");
    return convolveSingleChannel(input, cfg);
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in convolve: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("Exception in convolve: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in convolve");
    return std::unexpected(ProcessError{ProcessError::Code::PROCESSING_FAILED,
                                        "Unknown exception in convolve"});
  }
}

std::expected<cv::Mat, ProcessError>
Convolve::convolveSingleChannel(const cv::Mat &input,
                                const ConvolutionConfig &cfg) {
  try {
    ScopedTimer timer("Optimized Convolution");

    auto kernelResult = prepareKernel(cfg);
    if (!kernelResult.has_value()) {
      return std::unexpected(kernelResult.error());
    }

    cv::Mat kernel = kernelResult.value();
    cv::Mat output =
        cfg.use_memory_pool
            ? MemoryPool::allocate(input.rows, input.cols, input.type())
            : cv::Mat(input.rows, input.cols, input.type());

    // Choose appropriate algorithm based on kernel size and configuration
    std::expected<void, ProcessError> result;
    if (cfg.use_fft && kernel.rows > 15) {
      convolveLogger->debug("Using FFT-based convolution for large kernel");
      result = fftConvolve(input, output, kernel);
    } else if (cfg.use_avx && hasAVXSupport()) {
      convolveLogger->debug("Using AVX-optimized convolution");
      result = optimizedConvolveAVX(input, output, kernel, cfg);
    } else {
      convolveLogger->debug("Using block-based convolution");
      result = blockProcessing(
          input, output,
          [&](const cv::Mat &inBlock,
              cv::Mat &outBlock) -> std::expected<void, ProcessError> {
            try {
              cv::filter2D(inBlock, outBlock, -1, kernel);
              return {};
            } catch (const std::exception &e) {
              return std::unexpected(
                  ProcessError{ProcessError::Code::PROCESSING_FAILED,
                               fmt::format("Filter2D failed: {}", e.what())});
            }
          },
          cfg.block_size);
    }

    if (!result.has_value()) {
      return std::unexpected(result.error());
    }

    return output;
  } catch (const cv::Exception &e) {
    convolveLogger->error("OpenCV exception in convolveSingleChannel: {}",
                          e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("OpenCV exception: {}", e.what())});
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in convolveSingleChannel: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("Exception: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in convolveSingleChannel");
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     "Unknown exception in convolveSingleChannel"});
  }
}

std::expected<void, ProcessError>
Convolve::optimizedConvolveAVX(const cv::Mat &input, cv::Mat &output,
                               const cv::Mat &kernel,
                               const ConvolutionConfig &cfg) {
  try {
    // Check if AVX is supported
    if (!hasAVXSupport()) {
      convolveLogger->warn(
          "AVX not supported, falling back to standard convolution");
      cv::filter2D(input, output, -1, kernel);
      return {};
    }

#if defined(__AVX__)
    const int kCacheLineSize = 64;
    const int kRows = input.rows;
    const int kCols = input.cols;
    const int kKernelSize = kernel.rows;
    const int kRadius = kKernelSize / 2;

    // Ensure input and output matrices are valid
    if (!isValidMatrix(input) || !isValidMatrix(output) ||
        input.type() != CV_32F || output.size() != input.size()) {
      return std::unexpected(
          ProcessError{ProcessError::Code::INVALID_INPUT,
                       "Invalid input/output matrices for AVX convolution"});
    }

    // Determine optimal thread count
    int thread_count =
        cfg.thread_count <= 0
            ? std::thread::hardware_concurrency()
            : std::min(cfg.thread_count,
                       static_cast<int>(std::thread::hardware_concurrency()));

    // Use C++20 parallel algorithm
    std::vector<int> row_indices(kRows - 2 * kRadius);
    std::iota(row_indices.begin(), row_indices.end(), kRadius);

    std::for_each(row_indices.begin(), row_indices.end(), [&](int i) {
      for (int j = kRadius; j < kCols - kRadius; j += 8) {
        if (j + 8 <= kCols - kRadius) {
          // Process 8 elements at once with AVX
          __m256 sum = _mm256_setzero_ps();

          for (int ki = -kRadius; ki <= kRadius; ki++) {
            for (int kj = -kRadius; kj <= kRadius; kj++) {
              float k_val = kernel.at<float>(ki + kRadius, kj + kRadius);
              __m256 k = _mm256_set1_ps(k_val);
              __m256 in = _mm256_loadu_ps(&input.at<float>(i + ki, j + kj));
              sum = _mm256_add_ps(sum, _mm256_mul_ps(k, in));
            }
          }

          _mm256_storeu_ps(&output.at<float>(i, j), sum);
        } else {
          // Handle remaining elements
          for (int jj = j; jj < kCols - kRadius; jj++) {
            float sum = 0.0f;
            for (int ki = -kRadius; ki <= kRadius; ki++) {
              for (int kj = -kRadius; kj <= kRadius; kj++) {
                sum += kernel.at<float>(ki + kRadius, kj + kRadius) *
                       input.at<float>(i + ki, jj + kj);
              }
            }
            output.at<float>(i, jj) = sum;
          }
        }
      }
    });
#else
    // Fallback to standard OpenCV convolution
    cv::filter2D(input, output, -1, kernel);
#endif
    return {};
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in optimizedConvolveAVX: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("AVX convolution failed: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in optimizedConvolveAVX");
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     "Unknown exception in optimizedConvolveAVX"});
  }
}

std::expected<void, ProcessError> Convolve::fftConvolve(const cv::Mat &input,
                                                        cv::Mat &output,
                                                        const cv::Mat &kernel) {
  try {
    cv::Mat inputPadded, kernelPadded;

    // Check inputs
    if (!isValidMatrix(input) || !isValidMatrix(kernel)) {
      return std::unexpected(
          ProcessError{ProcessError::Code::INVALID_INPUT,
                       "Invalid input or kernel for FFT convolution"});
    }

    // Get optimal FFT sizes with error checking
    int m = cv::getOptimalDFTSize(input.rows + kernel.rows - 1);
    int n = cv::getOptimalDFTSize(input.cols + kernel.cols - 1);

    if (m <= 0 || n <= 0) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       "Failed to determine optimal DFT size"});
    }

    // Zero padding for FFT
    try {
      cv::copyMakeBorder(input, inputPadded, 0, m - input.rows, 0,
                         n - input.cols, cv::BORDER_CONSTANT, 0);
      cv::copyMakeBorder(kernel, kernelPadded, 0, m - kernel.rows, 0,
                         n - kernel.cols, cv::BORDER_CONSTANT, 0);
    } catch (const cv::Exception &e) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       fmt::format("Failed to pad matrices: {}", e.what())});
    }

    // Parallel FFT transform with proper error handling
    cv::Mat inputDFT, kernelDFT;
    std::mutex fft_mutex;
    std::exception_ptr input_exception = nullptr;
    std::exception_ptr kernel_exception = nullptr;

    std::thread input_thread([&]() {
      try {
        cv::dft(inputPadded, inputDFT, cv::DFT_COMPLEX_OUTPUT);
      } catch (...) {
        input_exception = std::current_exception();
      }
    });

    std::thread kernel_thread([&]() {
      try {
        cv::dft(kernelPadded, kernelDFT, cv::DFT_COMPLEX_OUTPUT);
      } catch (...) {
        kernel_exception = std::current_exception();
      }
    });

    input_thread.join();
    kernel_thread.join();

    // Check for exceptions
    if (input_exception) {
      std::rethrow_exception(input_exception);
    }

    if (kernel_exception) {
      std::rethrow_exception(kernel_exception);
    }

    // Domain multiplication
    cv::Mat productDFT;
    try {
      cv::mulSpectrums(inputDFT, kernelDFT, productDFT, 0);
    } catch (const cv::Exception &e) {
      return std::unexpected(ProcessError{
          ProcessError::Code::PROCESSING_FAILED,
          fmt::format("Failed to multiply spectrums: {}", e.what())});
    }

    // Inverse transform
    cv::Mat tempOutput;
    try {
      cv::dft(productDFT, tempOutput,
              cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    } catch (const cv::Exception &e) {
      return std::unexpected(ProcessError{
          ProcessError::Code::PROCESSING_FAILED,
          fmt::format("Failed to perform inverse DFT: {}", e.what())});
    }

    // Crop to original size
    cv::Rect roi(0, 0, input.cols, input.rows);
    if (roi.width > tempOutput.cols || roi.height > tempOutput.rows) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       "Invalid ROI dimensions for cropping"});
    }

    tempOutput(roi).copyTo(output);
    return {};
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in fftConvolve: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("FFT convolution failed: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in fftConvolve");
    return std::unexpected(ProcessError{ProcessError::Code::PROCESSING_FAILED,
                                        "Unknown exception in fftConvolve"});
  }
}

std::expected<void, ProcessError>
Convolve::blockProcessing(const cv::Mat &input, cv::Mat &output,
                          const std::function<std::expected<void, ProcessError>(
                              const cv::Mat &, cv::Mat &)> &processor,
                          int blockSize) {
  try {
    // Validate inputs
    if (!isValidMatrix(input) || !isValidMatrix(output) ||
        input.size() != output.size()) {
      return std::unexpected(
          ProcessError{ProcessError::Code::INVALID_INPUT,
                       "Invalid input/output matrices for block processing"});
    }

    // Ensure block size is reasonable
    blockSize = clamp(blockSize, 16, std::min(input.rows, input.cols) / 2);
    const int overlap =
        blockSize / 4; // Overlap area to avoid boundary artifacts

    // Create a view span of blocks to process in parallel
    std::vector<std::pair<int, int>> blocks;
    for (int y = 0; y < input.rows; y += blockSize - overlap) {
      for (int x = 0; x < input.cols; x += blockSize - overlap) {
        blocks.emplace_back(y, x);
      }
    }

    // Process blocks in parallel using C++20 parallel algorithms
    std::mutex error_mutex;
    std::optional<ProcessError> first_error;

    std::for_each(blocks.begin(), blocks.end(), [&](const auto &block) {
      // Skip if an error was already encountered
      {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error)
          return;
      }

      int y = block.first;
      int x = block.second;

      // Calculate current block size (handling edge cases)
      int currentBlockHeight = std::min(blockSize, input.rows - y);
      int currentBlockWidth = std::min(blockSize, input.cols - x);

      // Create views for this block
      cv::Rect blockRect(x, y, currentBlockWidth, currentBlockHeight);
      cv::Mat inBlock = input(blockRect);
      cv::Mat outBlock = output(blockRect);

      // Process this block and capture any errors
      auto result = processor(inBlock, outBlock);
      if (!result.has_value()) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!first_error) {
          first_error = result.error();
        }
      }
    });

    if (first_error) {
      return std::unexpected(*first_error);
    }

    return {};
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in blockProcessing: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("Block processing failed: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in blockProcessing");
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     "Unknown exception in blockProcessing"});
  }
}

std::expected<cv::Mat, ProcessError>
Convolve::deconvolve(const cv::Mat &input, const DeconvolutionConfig &cfg) {
  try {
    ScopedTimer timer("Deconvolution");
    convolveLogger->info("Starting deconvolution process");

    // Validate configuration
    auto validate_result = validateDeconvolutionConfig(input, cfg);
    if (!validate_result.has_value()) {
      return std::unexpected(validate_result.error());
    }

    // If multi-channel image and per-channel processing is requested
    if (input.channels() > 1 && cfg.per_channel) {
      convolveLogger->debug("Processing deconvolution per channel");
      return processMultiChannel(input, [&](const cv::Mat &channel) {
        return deconvolveSingleChannel(channel, cfg);
      });
    }

    // Get PSF with error handling
    auto psf_result = estimatePSF(input.size());
    if (!psf_result.has_value()) {
      return std::unexpected(psf_result.error());
    }

    cv::Mat psf = psf_result.value();
    cv::Mat output;

    // Use appropriate deconvolution method based on configuration
    std::expected<void, ProcessError> result;

    switch (cfg.method) {
    case DeconvMethod::RICHARDSON_LUCY: {
      convolveLogger->debug("Using Richardson-Lucy deconvolution method");
      output = cfg.use_memory_pool
                   ? MemoryPool::allocate(input.rows, input.cols, input.type())
                   : cv::Mat(input.rows, input.cols, input.type());

      result = richardsonLucyDeconv(input, psf, output, cfg);
      break;
    }
    case DeconvMethod::WIENER: {
      convolveLogger->debug("Using Wiener deconvolution method");
      output = cfg.use_memory_pool
                   ? MemoryPool::allocate(input.rows, input.cols, input.type())
                   : cv::Mat(input.rows, input.cols, input.type());

      result = wienerDeconv(input, psf, output, cfg);
      break;
    }
    case DeconvMethod::TIKHONOV: {
      convolveLogger->debug("Using Tikhonov deconvolution method");
      output = cfg.use_memory_pool
                   ? MemoryPool::allocate(input.rows, input.cols, input.type())
                   : cv::Mat(input.rows, input.cols, input.type());

      result = tikhonovDeconv(input, psf, output, cfg);
      break;
    }
    default:
      convolveLogger->error("Unsupported deconvolution method");
      return std::unexpected(ProcessError{ProcessError::Code::INVALID_CONFIG,
                                          "Unsupported deconvolution method"});
    }

    if (!result.has_value()) {
      return std::unexpected(result.error());
    }

    // Normalize output to valid range
    try {
      cv::normalize(output, output, 0, 255, cv::NORM_MINMAX, CV_8U);
    } catch (const cv::Exception &e) {
      return std::unexpected(ProcessError{
          ProcessError::Code::PROCESSING_FAILED,
          fmt::format("Failed to normalize output: {}", e.what())});
    }

    convolveLogger->info("Deconvolution completed successfully");
    return output;
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in deconvolve: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("Deconvolution failed: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in deconvolve");
    return std::unexpected(ProcessError{ProcessError::Code::PROCESSING_FAILED,
                                        "Unknown exception in deconvolve"});
  }
}

std::expected<cv::Mat, ProcessError>
Convolve::deconvolveSingleChannel(const cv::Mat &input,
                                  const DeconvolutionConfig &cfg) {
  try {
    convolveLogger->info("Starting single channel deconvolution");

    auto validate_result = validateDeconvolutionConfig(input, cfg);
    if (!validate_result.has_value()) {
      return std::unexpected(validate_result.error());
    }

    auto psf_result = estimatePSF(input.size());
    if (!psf_result.has_value()) {
      return std::unexpected(psf_result.error());
    }

    cv::Mat psf = psf_result.value();
    cv::Mat output =
        cfg.use_memory_pool
            ? MemoryPool::allocate(input.rows, input.cols, input.type())
            : cv::Mat(input.rows, input.cols, input.type());

    std::expected<void, ProcessError> result;

    switch (cfg.method) {
    case DeconvMethod::RICHARDSON_LUCY:
      convolveLogger->debug("Using Richardson-Lucy deconvolution method");
      result = richardsonLucyDeconv(input, psf, output, cfg);
      break;
    case DeconvMethod::WIENER:
      convolveLogger->debug("Using Wiener deconvolution method");
      result = wienerDeconv(input, psf, output, cfg);
      break;
    case DeconvMethod::TIKHONOV:
      convolveLogger->debug("Using Tikhonov deconvolution method");
      result = tikhonovDeconv(input, psf, output, cfg);
      break;
    default:
      convolveLogger->error("Unsupported deconvolution method");
      return std::unexpected(ProcessError{ProcessError::Code::INVALID_CONFIG,
                                          "Unsupported deconvolution method"});
    }

    if (!result.has_value()) {
      return std::unexpected(result.error());
    }

    // Normalize output
    cv::normalize(output, output, 0, 255, cv::NORM_MINMAX, CV_8U);
    convolveLogger->info("Single channel deconvolution completed");

    return output;
  } catch (const cv::Exception &e) {
    convolveLogger->error("OpenCV exception in deconvolveSingleChannel: {}",
                          e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("OpenCV exception: {}", e.what())});
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in deconvolveSingleChannel: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("Exception: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in deconvolveSingleChannel");
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     "Unknown exception in deconvolveSingleChannel"});
  }
}

std::expected<void, ProcessError>
Convolve::richardsonLucyDeconv(const cv::Mat &input, const cv::Mat &psf,
                               cv::Mat &output,
                               const DeconvolutionConfig &cfg) {
  try {
    convolveLogger->debug("Starting Richardson-Lucy deconvolution");

    // Validate inputs
    if (!isValidMatrix(input) || !isValidMatrix(psf) ||
        !isValidMatrix(output)) {
      return std::unexpected(ProcessError{
          ProcessError::Code::INVALID_INPUT,
          "Invalid input matrices for Richardson-Lucy deconvolution"});
    }

    cv::Mat imgEstimate;
    input.convertTo(imgEstimate, CV_32F);

    cv::Mat psfFlip;
    cv::flip(psf, psfFlip, -1);
    const int borderType = getOpenCVBorderType(cfg.border_mode);

    // Create optimized thread pool
    static DynamicThreadPool threadPool(
        cfg.thread_count <= 0 ? std::thread::hardware_concurrency()
                              : cfg.thread_count);

    const auto iterations = clamp(cfg.iterations, 1, 1000);
    for (int i = 0; i < iterations; ++i) {
      cv::Mat convResult, relativeBlur, errorEstimate;

      // Execute steps in parallel using futures
      auto convTask =
          threadPool.enqueue([&]() -> std::expected<void, ProcessError> {
            try {
              cv::filter2D(imgEstimate, convResult, -1, psf, cv::Point(-1, -1),
                           0, borderType);
              return {};
            } catch (const std::exception &e) {
              return std::unexpected(ProcessError{
                  ProcessError::Code::PROCESSING_FAILED,
                  fmt::format("Convolution failed: {}", e.what())});
            }
          });

      // Wait for convolution to complete
      auto convResult_status = convTask.get();
      if (!convResult_status.has_value()) {
        return std::unexpected(convResult_status.error());
      }

      // Continue with division in parallel
      auto divideTask =
          threadPool.enqueue([&]() -> std::expected<void, ProcessError> {
            try {
              // Avoid division by zero
              cv::Mat validDenominator;
              cv::max(convResult, 1e-10, validDenominator);
              cv::divide(input, validDenominator, relativeBlur);
              return {};
            } catch (const std::exception &e) {
              return std::unexpected(
                  ProcessError{ProcessError::Code::PROCESSING_FAILED,
                               fmt::format("Division failed: {}", e.what())});
            }
          });

      auto errorTask =
          threadPool.enqueue([&]() -> std::expected<void, ProcessError> {
            try {
              cv::filter2D(relativeBlur, errorEstimate, -1, psfFlip,
                           cv::Point(-1, -1), 0, borderType);
              return {};
            } catch (const std::exception &e) {
              return std::unexpected(ProcessError{
                  ProcessError::Code::PROCESSING_FAILED,
                  fmt::format("Error estimation failed: {}", e.what())});
            }
          });

      // Ensure division completed
      auto divideResult_status = divideTask.get();
      if (!divideResult_status.has_value()) {
        return std::unexpected(divideResult_status.error());
      }

      // Ensure error estimation completed
      auto errorResult_status = errorTask.get();
      if (!errorResult_status.has_value()) {
        return std::unexpected(errorResult_status.error());
      }

      // Update estimate
      auto updateTask =
          threadPool.enqueue([&]() -> std::expected<void, ProcessError> {
            try {
              cv::multiply(imgEstimate, errorEstimate, imgEstimate);

              // Apply non-negativity constraint
              cv::max(imgEstimate, 0, imgEstimate);
              return {};
            } catch (const std::exception &e) {
              return std::unexpected(
                  ProcessError{ProcessError::Code::PROCESSING_FAILED,
                               fmt::format("Update failed: {}", e.what())});
            }
          });

      auto updateResult_status = updateTask.get();
      if (!updateResult_status.has_value()) {
        return std::unexpected(updateResult_status.error());
      }

      convolveLogger->debug("Richardson-Lucy iteration {} completed", i);
    }

    imgEstimate.convertTo(output, CV_8U);
    convolveLogger->info("Richardson-Lucy deconvolution completed");
    return {};
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in richardsonLucyDeconv: {}", e.what());
    return std::unexpected(ProcessError{
        ProcessError::Code::PROCESSING_FAILED,
        fmt::format("Richardson-Lucy deconvolution failed: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in richardsonLucyDeconv");
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     "Unknown exception in richardsonLucyDeconv"});
  }
}

std::expected<void, ProcessError>
Convolve::wienerDeconv(const cv::Mat &input, const cv::Mat &psf,
                       cv::Mat &output, const DeconvolutionConfig &cfg) {
  try {
    ScopedTimer timer("Wiener Deconvolution");
    convolveLogger->debug("Starting Wiener deconvolution");

    // Input validation
    if (!isValidMatrix(input) || !isValidMatrix(psf) ||
        !isValidMatrix(output)) {
      return std::unexpected(
          ProcessError{ProcessError::Code::INVALID_INPUT,
                       "Invalid input matrices for Wiener deconvolution"});
    }

    // Prepare inputs
    cv::Mat inputF, psfF;
    input.convertTo(inputF, CV_32F);
    psf.convertTo(psfF, CV_32F);

    // Calculate optimal FFT size
    int m = cv::getOptimalDFTSize(std::max(input.rows, psf.rows) * 2 - 1);
    int n = cv::getOptimalDFTSize(std::max(input.cols, psf.cols) * 2 - 1);

    if (m <= 0 || n <= 0) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       "Failed to determine optimal DFT size"});
    }

    // Zero-pad PSF and input
    cv::Mat psfPadded, inputPadded;
    try {
      cv::copyMakeBorder(psfF, psfPadded, 0, m - psf.rows, 0, n - psf.cols,
                         cv::BORDER_CONSTANT, cv::Scalar::all(0));
      cv::copyMakeBorder(inputF, inputPadded, 0, m - input.rows, 0,
                         n - input.cols, cv::BORDER_CONSTANT,
                         cv::Scalar::all(0));
    } catch (const cv::Exception &e) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       fmt::format("Failed to pad matrices: {}", e.what())});
    }

    // Compute DFTs
    cv::Mat psfDFT, inputDFT;
    try {
      cv::dft(psfPadded, psfDFT, cv::DFT_COMPLEX_OUTPUT);
      cv::dft(inputPadded, inputDFT, cv::DFT_COMPLEX_OUTPUT);
    } catch (const cv::Exception &e) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       fmt::format("DFT computation failed: {}", e.what())});
    }

    // Compute Wiener filter with noise regularization
    cv::Mat complexH, wienerFilter;
    try {
      cv::mulSpectrums(psfDFT, psfDFT, complexH, 0, true);

      // Ensure noise power is non-negative
      double noise_power = std::max(0.0, cfg.noise_power);
      cv::divide(cv::abs(psfDFT), complexH + noise_power, wienerFilter);
    } catch (const cv::Exception &e) {
      return std::unexpected(ProcessError{
          ProcessError::Code::PROCESSING_FAILED,
          fmt::format("Wiener filter computation failed: {}", e.what())});
    }

    // Apply filter
    cv::Mat result;
    try {
      cv::mulSpectrums(inputDFT, wienerFilter, result, 0);
    } catch (const cv::Exception &e) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       fmt::format("Filter application failed: {}", e.what())});
    }

    // Invert transform
    cv::Mat tempOutput;
    try {
      cv::idft(result, tempOutput, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    } catch (const cv::Exception &e) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       fmt::format("Inverse DFT failed: {}", e.what())});
    }

    // Crop to original size and copy to output
    tempOutput(cv::Rect(0, 0, input.cols, input.rows)).copyTo(output);

    convolveLogger->info("Wiener deconvolution completed");
    return {};
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in wienerDeconv: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("Wiener deconvolution failed: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in wienerDeconv");
    return std::unexpected(ProcessError{ProcessError::Code::PROCESSING_FAILED,
                                        "Unknown exception in wienerDeconv"});
  }
}

std::expected<void, ProcessError>
Convolve::tikhonovDeconv(const cv::Mat &input, const cv::Mat &psf,
                         cv::Mat &output, const DeconvolutionConfig &cfg) {
  try {
    convolveLogger->debug("Starting Tikhonov deconvolution");

    // Input validation
    if (!isValidMatrix(input) || !isValidMatrix(psf) ||
        !isValidMatrix(output)) {
      return std::unexpected(
          ProcessError{ProcessError::Code::INVALID_INPUT,
                       "Invalid input matrices for Tikhonov deconvolution"});
    }

    cv::Mat inputF, psfF;
    input.convertTo(inputF, CV_32F);
    psf.convertTo(psfF, CV_32F);

    // Calculate optimal FFT size
    int m = cv::getOptimalDFTSize(std::max(input.rows, psf.rows) * 2 - 1);
    int n = cv::getOptimalDFTSize(std::max(input.cols, psf.cols) * 2 - 1);

    // Zero-pad PSF and input
    cv::Mat psfPadded, inputPadded;
    try {
      cv::copyMakeBorder(psfF, psfPadded, 0, m - psf.rows, 0, n - psf.cols,
                         cv::BORDER_CONSTANT, cv::Scalar::all(0));
      cv::copyMakeBorder(inputF, inputPadded, 0, m - input.rows, 0,
                         n - input.cols, cv::BORDER_CONSTANT,
                         cv::Scalar::all(0));
    } catch (const cv::Exception &e) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       fmt::format("Failed to pad matrices: {}", e.what())});
    }

    // Compute DFTs
    cv::Mat inputSpectrum, psfSpectrum;
    try {
      cv::dft(inputF, inputSpectrum, cv::DFT_COMPLEX_OUTPUT);
      cv::dft(psfF, psfSpectrum, cv::DFT_COMPLEX_OUTPUT);
    } catch (const cv::Exception &e) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       fmt::format("DFT computation failed: {}", e.what())});
    }

    // Construct regularization term L = λ∇²
    cv::Mat regTerm = cv::Mat::zeros(input.size(), CV_32FC2);

    // Parallel construction of regularization term
    const double reg_param = clamp(cfg.regularization, 1e-10, 1.0);

    std::vector<int> indices(input.rows);
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(indices.begin(), indices.end(), [&](int i) {
      for (int j = 0; j < input.cols; j++) {
        float freq = (i * i + j * j) * reg_param;
        regTerm.at<cv::Vec2f>(i, j) = cv::Vec2f(freq, freq);
      }
    });

    // Compute the denominator (HH* + λL)
    cv::Mat denom;
    try {
      cv::mulSpectrums(psfSpectrum, psfSpectrum, denom, 0, true);
      denom += regTerm;
    } catch (const cv::Exception &e) {
      return std::unexpected(ProcessError{
          ProcessError::Code::PROCESSING_FAILED,
          fmt::format("Denominator computation failed: {}", e.what())});
    }

    // Compute restoration filter H*(HH* + λL)^(-1)
    cv::Mat restoreFilter;
    try {
      // Safe division with epsilon to avoid division by zero
      cv::divide(psfSpectrum, denom + 1e-10, restoreFilter);
    } catch (const cv::Exception &e) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       fmt::format("Filter computation failed: {}", e.what())});
    }

    // Apply filter to get restored image
    cv::Mat result;
    try {
      cv::mulSpectrums(inputSpectrum, restoreFilter, result, 0);
    } catch (const cv::Exception &e) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       fmt::format("Filter application failed: {}", e.what())});
    }

    // Inverse transform
    cv::Mat tempOutput;
    try {
      cv::idft(result, tempOutput, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    } catch (const cv::Exception &e) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       fmt::format("Inverse DFT failed: {}", e.what())});
    }

    // Crop and copy to output
    tempOutput(cv::Rect(0, 0, input.cols, input.rows)).copyTo(output);

    convolveLogger->info("Tikhonov deconvolution completed");
    return {};
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in tikhonovDeconv: {}", e.what());
    return std::unexpected(ProcessError{
        ProcessError::Code::PROCESSING_FAILED,
        fmt::format("Tikhonov deconvolution failed: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in tikhonovDeconv");
    return std::unexpected(ProcessError{ProcessError::Code::PROCESSING_FAILED,
                                        "Unknown exception in tikhonovDeconv"});
  }
}

std::expected<void, ProcessError>
Convolve::validateConvolutionConfig(const cv::Mat &input,
                                    const ConvolutionConfig &cfg) {
  try {
    if (!isValidMatrix(input)) {
      convolveLogger->error("Input image is empty or invalid");
      return std::unexpected(ProcessError{ProcessError::Code::INVALID_INPUT,
                                          "Empty or invalid input image"});
    }

    if (!cfg.per_channel && input.channels() > 1) {
      convolveLogger->warn("Multi-channel image will be processed as a whole");
    }

    if (cfg.kernel_size % 2 == 0 || cfg.kernel_size < 3) {
      convolveLogger->error("Invalid kernel size: {}", cfg.kernel_size);
      return std::unexpected(
          ProcessError{ProcessError::Code::INVALID_CONFIG,
                       fmt::format("Kernel size must be odd and >=3, got {}",
                                   cfg.kernel_size)});
    }

    if (cfg.kernel.size() !=
        static_cast<size_t>(cfg.kernel_size * cfg.kernel_size)) {
      convolveLogger->error("Kernel size mismatch: expected {}, got {}",
                            cfg.kernel_size * cfg.kernel_size,
                            cfg.kernel.size());
      return std::unexpected(ProcessError{
          ProcessError::Code::INVALID_CONFIG,
          fmt::format("Kernel dimensions mismatch: expected {}, got {}",
                      cfg.kernel_size * cfg.kernel_size, cfg.kernel.size())});
    }

    // Validate thread count
    if (cfg.thread_count < 0) {
      convolveLogger->warn("Negative thread count specified ({}), using auto",
                           cfg.thread_count);
    }

    // Validate block size
    if (cfg.block_size <= 0) {
      convolveLogger->warn("Invalid block size: {}, using default",
                           cfg.block_size);
    }

    convolveLogger->debug("Convolution config validated");
    return {};
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in validateConvolutionConfig: {}",
                          e.what());
    return std::unexpected(ProcessError{
        ProcessError::Code::INVALID_CONFIG,
        fmt::format("Configuration validation failed: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in validateConvolutionConfig");
    return std::unexpected(
        ProcessError{ProcessError::Code::INVALID_CONFIG,
                     "Unknown exception in configuration validation"});
  }
}

std::expected<void, ProcessError>
Convolve::validateDeconvolutionConfig(const cv::Mat &input,
                                      const DeconvolutionConfig &cfg) {
  try {
    if (!isValidMatrix(input)) {
      convolveLogger->error("Input image is empty or invalid");
      return std::unexpected(ProcessError{ProcessError::Code::INVALID_INPUT,
                                          "Empty or invalid input image"});
    }

    if (cfg.iterations <= 0 || cfg.iterations > 1000) {
      convolveLogger->error("Invalid iterations count: {}", cfg.iterations);
      return std::unexpected(ProcessError{
          ProcessError::Code::INVALID_CONFIG,
          fmt::format(
              "Invalid iterations count: must be between 1-1000, got {}",
              cfg.iterations)});
    }

    if (cfg.noise_power < 0.0) {
      convolveLogger->error("Invalid noise power: {}", cfg.noise_power);
      return std::unexpected(
          ProcessError{ProcessError::Code::INVALID_CONFIG,
                       fmt::format("Noise power must be non-negative, got {}",
                                   cfg.noise_power)});
    }

    if (cfg.regularization <= 0.0) {
      convolveLogger->error("Invalid regularization parameter: {}",
                            cfg.regularization);
      return std::unexpected(ProcessError{
          ProcessError::Code::INVALID_CONFIG,
          fmt::format("Regularization parameter must be positive, got {}",
                      cfg.regularization)});
    }

    if (!cfg.per_channel && input.channels() > 1) {
      convolveLogger->warn("Multi-channel image will be processed as a whole");
    }

    // Validate thread count
    if (cfg.thread_count < 0) {
      convolveLogger->warn("Negative thread count specified ({}), using auto",
                           cfg.thread_count);
    }

    convolveLogger->debug("Deconvolution config validated");
    return {};
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in validateDeconvolutionConfig: {}",
                          e.what());
    return std::unexpected(ProcessError{
        ProcessError::Code::INVALID_CONFIG,
        fmt::format("Configuration validation failed: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in validateDeconvolutionConfig");
    return std::unexpected(
        ProcessError{ProcessError::Code::INVALID_CONFIG,
                     "Unknown exception in configuration validation"});
  }
}

std::expected<cv::Mat, ProcessError>
Convolve::prepareKernel(const ConvolutionConfig &cfg) {
  try {
    convolveLogger->debug("Preparing convolution kernel");

    // Validate kernel size
    if (cfg.kernel_size % 2 == 0 || cfg.kernel_size < 3) {
      return std::unexpected(
          ProcessError{ProcessError::Code::INVALID_CONFIG,
                       fmt::format("Kernel size must be odd and >=3, got {}",
                                   cfg.kernel_size)});
    }

    // Validate kernel data size
    if (cfg.kernel.size() !=
        static_cast<size_t>(cfg.kernel_size * cfg.kernel_size)) {
      return std::unexpected(ProcessError{
          ProcessError::Code::INVALID_CONFIG,
          fmt::format("Kernel dimensions mismatch: expected {}, got {}",
                      cfg.kernel_size * cfg.kernel_size, cfg.kernel.size())});
    }

    // Check for NaN values in kernel
    for (const auto &val : cfg.kernel) {
      if (std::isnan(val) || std::isinf(val)) {
        return std::unexpected(
            ProcessError{ProcessError::Code::INVALID_CONFIG,
                         "Kernel contains NaN or Inf values"});
      }
    }

    cv::Mat kernel(cfg.kernel_size, cfg.kernel_size, CV_32F);
    std::copy(cfg.kernel.begin(), cfg.kernel.end(), kernel.ptr<float>());

    if (cfg.normalize_kernel) {
      double sum = cv::sum(kernel)[0];
      if (std::abs(sum) < 1e-10) {
        convolveLogger->warn(
            "Kernel sum is nearly zero ({}), normalization skipped", sum);
      } else {
        kernel /= sum;
        convolveLogger->debug("Kernel normalized with sum {}", sum);
      }
    }

    convolveLogger->debug("Kernel prepared successfully");
    return kernel;
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in prepareKernel: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("Failed to prepare kernel: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in prepareKernel");
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     "Unknown exception in kernel preparation"});
  }
}

int Convolve::getOpenCVBorderType(BorderMode mode) noexcept {
  switch (mode) {
  case BorderMode::ZERO_PADDING:
    return cv::BORDER_CONSTANT;
  case BorderMode::MIRROR_REFLECT:
    return cv::BORDER_REFLECT101;
  case BorderMode::REPLICATE:
    return cv::BORDER_REPLICATE;
  case BorderMode::CIRCULAR:
    return cv::BORDER_WRAP;
  default:
    return cv::BORDER_DEFAULT;
  }
}

std::expected<cv::Mat, ProcessError> Convolve::estimatePSF(cv::Size imgSize) {
  try {
    ScopedTimer timer("PSF Estimation");
    convolveLogger->debug("Estimating PSF for image size: width={}, height={}",
                          imgSize.width, imgSize.height);

    // Validate image size
    if (imgSize.width <= 0 || imgSize.height <= 0) {
      return std::unexpected(
          ProcessError{ProcessError::Code::INVALID_INPUT,
                       fmt::format("Invalid image dimensions: {}x{}",
                                   imgSize.width, imgSize.height)});
    }

    // Compute optimal size for FFT efficiency
    const int optimal_size =
        cv::getOptimalDFTSize(std::max(imgSize.width, imgSize.height) * 2 - 1);

    if (optimal_size <= 0) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       "Failed to determine optimal size for PSF"});
    }

    // Create PSF as a Gaussian kernel
    cv::Mat psf = cv::Mat::zeros(optimal_size, optimal_size, CV_32F);

    // Calculate kernel parameters based on image size
    const int kernelSize = std::min(imgSize.width, imgSize.height) / 20;
    const double sigma = kernelSize / 6.0;
    const cv::Point center(optimal_size / 2, optimal_size / 2);

    // Use parallel computing to generate the PSF
    std::vector<int> rows(optimal_size);
    std::iota(rows.begin(), rows.end(), 0);

    std::for_each(rows.begin(), rows.end(), [&](int i) {
      for (int j = 0; j < optimal_size; j++) {
        const double dx = j - center.x;
        const double dy = i - center.y;
        const double r2 = dx * dx + dy * dy;
        psf.at<float>(i, j) =
            static_cast<float>(std::exp(-r2 / (2 * sigma * sigma)));
      }
    });

    // Normalize the PSF to ensure energy conservation
    cv::normalize(psf, psf, 1.0, 0.0, cv::NORM_L1);

    // Crop PSF to match the input image size
    if (imgSize.width > psf.cols || imgSize.height > psf.rows) {
      return std::unexpected(
          ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       "PSF dimensions smaller than requested size"});
    }

    cv::Mat croppedPsf =
        psf(cv::Rect(0, 0, imgSize.width, imgSize.height)).clone();

    convolveLogger->info("PSF estimated successfully");
    return croppedPsf;
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in estimatePSF: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("PSF estimation failed: {}", e.what())});
  } catch (...) {
    convolveLogger->error("Unknown exception in estimatePSF");
    return std::unexpected(ProcessError{ProcessError::Code::PROCESSING_FAILED,
                                        "Unknown exception in PSF estimation"});
  }
}

cv::Mat Convolve::MemoryPool::allocate(int rows, int cols, int type) {
  try {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    convolveLogger->debug(
        "Allocating memory from pool: rows={}, cols={}, type={}", rows, cols,
        type);

    for (auto it = pool_.begin(); it != pool_.end();) {
      auto &mat_ptr = *it;
      // Check if shared_ptr is unique (not in use elsewhere)
      if (mat_ptr.use_count() == 1 && mat_ptr->rows == rows &&
          mat_ptr->cols == cols && mat_ptr->type() == type) {

        cv::Mat mat = *mat_ptr;
        // Zero out the matrix to avoid data leaks
        mat.setTo(cv::Scalar::all(0));

        it = pool_.erase(it);
        convolveLogger->debug("Memory allocated from pool");
        return mat;
      } else {
        ++it;
      }
    }

    convolveLogger->debug("No suitable memory found in pool, allocating new");
    cv::Mat newMat(rows, cols, type);
    return newMat;
  } catch (const cv::Exception &e) {
    convolveLogger->error("OpenCV exception in memory allocation: {}",
                          e.what());
    throw; // Re-throw to let the caller handle it
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in memory allocation: {}", e.what());
    throw;
  } catch (...) {
    convolveLogger->error("Unknown exception in memory allocation");
    throw;
  }
}

void Convolve::MemoryPool::deallocate(cv::Mat &mat) {
  if (!mat.empty()) {
    try {
      std::lock_guard<std::mutex> lock(pool_mutex_);
      if (pool_.size() < max_pool_size_) {
        convolveLogger->debug("Deallocating matrix {}x{} type {} to pool",
                              mat.rows, mat.cols, mat.type());

        // Create a shared_ptr to the mat data
        auto mat_ptr = std::make_shared<cv::Mat>(mat);
        pool_.push_back(mat_ptr);
      } else {
        convolveLogger->warn("Memory pool is full (size={}), discarding matrix",
                             pool_.size());
      }
      // Set mat to empty to indicate it's been deallocated
      mat = cv::Mat();
    } catch (const std::exception &e) {
      convolveLogger->error("Exception in deallocate: {}", e.what());
    } catch (...) {
      convolveLogger->error("Unknown exception in deallocate");
    }
  }
}

void Convolve::MemoryPool::clear() {
  try {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    size_t count = pool_.size();
    pool_.clear();
    convolveLogger->info("Memory pool cleared, {} matrices released", count);
  } catch (const std::exception &e) {
    convolveLogger->error("Exception in clear: {}", e.what());
  } catch (...) {
    convolveLogger->error("Unknown exception in clear");
  }
}