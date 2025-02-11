#include "Convolve.hpp"
#include "../utils/ThreadPool.hpp"
#include "SIMDHelper.hpp"
#include <chrono>
#include <fmt/format.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <variant>

namespace {
std::shared_ptr<spdlog::logger> convolveLogger =
    spdlog::basic_logger_mt("ConvolveLogger", "logs/convolve.log");
} // namespace

cv::Mat Convolve::process(
    const cv::Mat &input,
    const std::variant<ConvolutionConfig, DeconvolutionConfig> &config) {
  return std::visit(
      [&](auto &&cfg) {
        using T = std::decay_t<decltype(cfg)>;

        if constexpr (std::is_same_v<T, ConvolutionConfig>) {
          return convolve(input, cfg);
        } else if constexpr (std::is_same_v<T, DeconvolutionConfig>) {
          return deconvolve(input, cfg);
        }
      },
      config);
}

Convolve::ScopedTimer::ScopedTimer(const std::string &op)
    : operation(op), start(std::chrono::steady_clock::now()) {
  convolveLogger->debug("Starting operation: {}", operation);
}

Convolve::ScopedTimer::~ScopedTimer() {
  auto duration = std::chrono::steady_clock::now() - start;
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
  convolveLogger->debug("{} took {} ms", operation, ms.count());
}

cv::Mat Convolve::processMultiChannel(
    const cv::Mat &input,
    const std::function<cv::Mat(const cv::Mat &)> &processor) {
  ScopedTimer timer("Multi-channel Processing");
  convolveLogger->info("Processing multi-channel image");

  std::vector<cv::Mat> channels;
  cv::split(input, channels);
  convolveLogger->debug("Image split into {} channels", channels.size());

#pragma omp parallel for if (channels.size() > 1)
  for (int i = 0; i < channels.size(); i++) {
    convolveLogger->debug("Processing channel {}", i);
    channels[i] = processor(channels[i]);
  }

  cv::Mat result;
  cv::merge(channels, result);
  convolveLogger->info("Channels merged back into a single image");
  return result;
}

// 优化的卷积实现
cv::Mat Convolve::convolve(const cv::Mat &input, const ConvolutionConfig &cfg) {
  convolveLogger->info("Starting convolution process");
  if (input.channels() > 1 && cfg.per_channel) {
    convolveLogger->debug("Processing convolution per channel");
    return processMultiChannel(input, [&](const cv::Mat &channel) {
      return convolveSingleChannel(channel, cfg);
    });
  }
  convolveLogger->debug("Processing convolution on a single channel");
  return convolveSingleChannel(input, cfg);
}

// 单通道卷积实现
cv::Mat Convolve::convolveSingleChannel(const cv::Mat &input,
                                        const ConvolutionConfig &cfg) {
  ScopedTimer timer("Optimized Convolution");

  cv::Mat kernel = prepareKernel(cfg);
  cv::Mat output =
      cfg.use_memory_pool
          ? MemoryPool::allocate(input.rows, input.cols, input.type())
          : cv::Mat(input.rows, input.cols, input.type());

  // 根据图像和核的大小选择最优算法
  if (cfg.use_fft && kernel.rows > 15) {
    // 大核使用FFT
    fftConvolve(input, output, kernel);
  } else if (cfg.use_avx) {
    // 使用AVX优化的直接卷积
    optimizedConvolveAVX(input, output, kernel, cfg);
  } else {
    // 分块处理
    blockProcessing(
        input, output,
        [&](const cv::Mat &inBlock, cv::Mat &outBlock) {
          cv::filter2D(inBlock, outBlock, -1, kernel);
        },
        cfg.block_size);
  }

  return output;
}

void Convolve::optimizedConvolveAVX(const cv::Mat &input, cv::Mat &output,
                                    const cv::Mat &kernel,
                                    const ConvolutionConfig &cfg) {
// 首先检查CPU是否支持AVX指令集
#if defined(__AVX__)
  const int kCacheLineSize = 64;
  const int kRows = input.rows;
  const int kCols = input.cols;
  const int kKernelSize = kernel.rows;
  const int kRadius = kKernelSize / 2;

#pragma omp parallel for num_threads(cfg.thread_count) schedule(dynamic)
  for (int i = kRadius; i < kRows - kRadius; i += 1) {
    for (int j = kRadius; j < kCols - kRadius; j += 8) {
      __m256 sum = _mm256_setzero_ps();

      // 使用AVX指令集优化的卷积核心计算
      for (int ki = -kRadius; ki <= kRadius; ki++) {
        for (int kj = -kRadius; kj <= kRadius; kj++) {
          __m256 k = _mm256_broadcast_ss(
              &kernel.at<float>(ki + kRadius, kj + kRadius));
          __m256 in = _mm256_loadu_ps(&input.at<float>(i + ki, j + kj));
          sum = _mm256_add_ps(sum, _mm256_mul_ps(k, in));
        }
      }

      _mm256_storeu_ps(&output.at<float>(i, j), sum);
    }
  }
#else
  // 如果不支持AVX,则回退到普通实现
  cv::filter2D(input, output, -1, kernel);
#endif
}

void Convolve::fftConvolve(const cv::Mat &input, cv::Mat &output,
                           const cv::Mat &kernel) {
  cv::Mat inputPadded, kernelPadded;

  // 优化的FFT填充
  int m = cv::getOptimalDFTSize(input.rows + kernel.rows - 1);
  int n = cv::getOptimalDFTSize(input.cols + kernel.cols - 1);

  cv::copyMakeBorder(input, inputPadded, 0, m - input.rows, 0, n - input.cols,
                     cv::BORDER_CONSTANT, 0);
  cv::copyMakeBorder(kernel, kernelPadded, 0, m - kernel.rows, 0,
                     n - kernel.cols, cv::BORDER_CONSTANT, 0);

  // 并行FFT变换
  cv::Mat inputDFT, kernelDFT;
  cv::dft(inputPadded, inputDFT, cv::DFT_COMPLEX_OUTPUT);
  cv::dft(kernelPadded, kernelDFT, cv::DFT_COMPLEX_OUTPUT);

  // 频域乘法
  cv::Mat productDFT;
  cv::mulSpectrums(inputDFT, kernelDFT, productDFT, 0);

  // 反变换
  cv::dft(productDFT, output,
          cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

  // 裁剪到原始大小
  output = output(cv::Rect(0, 0, input.cols, input.rows));
}

void Convolve::blockProcessing(
    const cv::Mat &input, cv::Mat &output,
    const std::function<void(const cv::Mat &, cv::Mat &)> &processor,
    int blockSize) {
  const int overlap = blockSize / 4; // 重叠区域大小

#pragma omp parallel for collapse(2)
  for (int y = 0; y < input.rows; y += blockSize - overlap) {
    for (int x = 0; x < input.cols; x += blockSize - overlap) {
      // 计算当前块的大小
      int currentBlockWidth = std::min(blockSize, input.cols - x);
      int currentBlockHeight = std::min(blockSize, input.rows - y);

      // 提取和处理块
      cv::Mat inBlock =
          input(cv::Rect(x, y, currentBlockWidth, currentBlockHeight));
      cv::Mat outBlock =
          output(cv::Rect(x, y, currentBlockWidth, currentBlockHeight));

      processor(inBlock, outBlock);
    }
  }
}

// 反卷积实现
cv::Mat Convolve::deconvolve(const cv::Mat &input,
                             const DeconvolutionConfig &cfg) {
  ScopedTimer timer("Deconvolution");
  convolveLogger->info("Starting deconvolution process");
  validateDeconvolutionConfig(input, cfg);

  // 如果是多通道图像且需要分通道处理
  if (input.channels() > 1 && cfg.per_channel) {
    convolveLogger->debug("Processing deconvolution per channel");
    return processMultiChannel(input, [&](const cv::Mat &channel) {
      return deconvolveSingleChannel(channel, cfg);
    });
  }

  // 获取PSF（点扩散函数）
  cv::Mat psf = estimatePSF(input.size());
  cv::Mat output;

  // 根据不同的反卷积方法选择对应的实现
  switch (cfg.method) {
  case DeconvMethod::RICHARDSON_LUCY: {
    convolveLogger->debug("Using Richardson-Lucy deconvolution method");
    // 使用内存池分配输出内存
    output = MemoryPool::allocate(input.rows, input.cols, input.type());
    richardsonLucyDeconv(input, psf, output, cfg);
    break;
  }
  case DeconvMethod::WIENER: {
    convolveLogger->debug("Using Wiener deconvolution method");
    output = MemoryPool::allocate(input.rows, input.cols, input.type());
    wienerDeconv(input, psf, output, cfg);
    break;
  }
  case DeconvMethod::TIKHONOV: {
    convolveLogger->debug("Using Tikhonov deconvolution method");
    output = MemoryPool::allocate(input.rows, input.cols, input.type());
    tikhonovDeconv(input, psf, output, cfg);
    break;
  }
  default:
    convolveLogger->error("Unsupported deconvolution method");
    throw std::invalid_argument("Unsupported deconvolution method");
  }

  // 处理完成后确保输出图像在有效范围内
  cv::normalize(output, output, 0, 255, cv::NORM_MINMAX, CV_8U);
  convolveLogger->info("Deconvolution completed");

  return output;
}

// 单通道反卷积实现
cv::Mat Convolve::deconvolveSingleChannel(const cv::Mat &input,
                                          const DeconvolutionConfig &cfg) {
  convolveLogger->info("Starting single channel deconvolution");
  validateDeconvolutionConfig(input, cfg);

  cv::Mat output;
  cv::Mat psf = estimatePSF(input.size()); // 实际应用中需要根据情况获取PSF

  switch (cfg.method) {
  case DeconvMethod::RICHARDSON_LUCY:
    convolveLogger->debug("Using Richardson-Lucy deconvolution method");
    richardsonLucyDeconv(input, psf, output, cfg);
    break;
  case DeconvMethod::WIENER:
    convolveLogger->debug("Using Wiener deconvolution method");
    wienerDeconv(input, psf, output, cfg);
    break;
  case DeconvMethod::TIKHONOV:
    convolveLogger->debug("Using Tikhonov deconvolution method");
    tikhonovDeconv(input, psf, output, cfg);
    break;
  default:
    convolveLogger->error("Unsupported deconvolution method");
    throw std::invalid_argument("Unsupported deconvolution method");
  }

  convolveLogger->info("Single channel deconvolution completed");
  return output;
}

// Richardson-Lucy 反卷积
void Convolve::richardsonLucyDeconv(const cv::Mat &input, const cv::Mat &psf,
                                    cv::Mat &output,
                                    const DeconvolutionConfig &cfg) {
  convolveLogger->debug("Starting Richardson-Lucy deconvolution");
  cv::Mat imgEstimate = input.clone();
  cv::Mat psfFlip;
  cv::flip(psf, psfFlip, -1);
  const int borderType = getOpenCVBorderType(cfg.border_mode);

  static DynamicThreadPool threadPool;

  for (int i = 0; i < cfg.iterations; ++i) {
    cv::Mat convResult, relativeBlur, errorEstimate;

    auto task1 = threadPool.enqueue([&]() {
      cv::filter2D(imgEstimate, convResult, -1, psf, cv::Point(-1, -1), 0,
                   borderType);
    });

    task1.wait(); // 等待第一步完成

    auto task2 = threadPool.enqueue(
        [&]() { cv::divide(input, convResult, relativeBlur); });

    auto task3 = threadPool.enqueue([&]() {
      cv::filter2D(relativeBlur, errorEstimate, -1, psfFlip, cv::Point(-1, -1),
                   0, borderType);
    });

    task2.wait();
    task3.wait();

    auto task4 = threadPool.enqueue(
        [&]() { cv::multiply(imgEstimate, errorEstimate, imgEstimate); });

    task4.wait();
    convolveLogger->debug("Richardson-Lucy iteration {} completed", i);
  }

  imgEstimate.convertTo(output, CV_8U);
  convolveLogger->info("Richardson-Lucy deconvolution completed");
}

// 完整的Wiener反卷积实现
void Convolve::wienerDeconv(const cv::Mat &input, const cv::Mat &psf,
                            cv::Mat &output, const DeconvolutionConfig &cfg) {
  ScopedTimer timer("Wiener Deconvolution");
  convolveLogger->debug("Starting Wiener deconvolution");

  // 准备输入
  cv::Mat inputF, psfF;
  input.convertTo(inputF, CV_32F);
  psf.convertTo(psfF, CV_32F);

  // 计算PSF的FFT
  cv::Mat psfPadded;
  int m = cv::getOptimalDFTSize(input.rows + psf.rows - 1);
  int n = cv::getOptimalDFTSize(input.cols + psf.cols - 1);
  cv::copyMakeBorder(psfF, psfPadded, 0, m - psf.rows, 0, n - psf.cols,
                     cv::BORDER_CONSTANT, cv::Scalar::all(0));

  cv::Mat psfDFT, inputDFT;
  cv::dft(psfPadded, psfDFT, cv::DFT_COMPLEX_OUTPUT);
  cv::dft(inputF, inputDFT, cv::DFT_COMPLEX_OUTPUT);

  // 计算维纳滤波器
  cv::Mat complexH;
  cv::mulSpectrums(psfDFT, psfDFT, complexH, 0, true);

  cv::Mat wienerFilter;
  cv::divide(cv::abs(psfDFT), complexH + cfg.noise_power, wienerFilter);

  // 应用滤波器
  cv::Mat result;
  cv::mulSpectrums(inputDFT, wienerFilter, result, 0);

  // 反变换
  cv::idft(result, output, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
  cv::normalize(output, output, 0, 255, cv::NORM_MINMAX, CV_8U);
  convolveLogger->info("Wiener deconvolution completed");
}

// Tikhonov 正则化反卷积
void Convolve::tikhonovDeconv(const cv::Mat &input, const cv::Mat &psf,
                              cv::Mat &output, const DeconvolutionConfig &cfg) {
  convolveLogger->debug("Starting Tikhonov deconvolution");
  cv::Mat inputF, psfF;
  input.convertTo(inputF, CV_32F);
  psf.convertTo(psfF, CV_32F);

  cv::Mat inputSpectrum, psfSpectrum;
  // 对输入图像和PSF做傅里叶变换
  cv::dft(inputF, inputSpectrum, cv::DFT_COMPLEX_OUTPUT);
  cv::dft(psfF, psfSpectrum, cv::DFT_COMPLEX_OUTPUT);

  // 构建正则化项 L = λ∇²
  cv::Mat regTerm = cv::Mat::zeros(input.size(), CV_32FC2);
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      float freq = (i * i + j * j) * cfg.regularization;
      regTerm.at<cv::Vec2f>(i, j) = cv::Vec2f(freq, freq);
    }
  }

  // 计算 H*(HH* + λL)^(-1)Y
  cv::Mat denom;
  cv::mulSpectrums(psfSpectrum, psfSpectrum, denom, 0, true);
  denom += regTerm;

  cv::Mat restoreFilter;
  cv::divide(psfSpectrum, denom, restoreFilter);

  cv::Mat result;
  cv::mulSpectrums(inputSpectrum, restoreFilter, result, 0);

  cv::idft(result, output, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
  cv::normalize(output, output, 0, 255, cv::NORM_MINMAX, CV_8U);
  convolveLogger->info("Tikhonov deconvolution completed");
}

// 参数校验
void Convolve::validateConvolutionConfig(const cv::Mat &input,
                                         const ConvolutionConfig &cfg) {
  if (input.empty()) {
    convolveLogger->error("Input image is empty");
    throw std::invalid_argument("Empty input image");
  }

  if (!cfg.per_channel && input.channels() > 1) {
    convolveLogger->warn("Multi-channel image will be processed as a whole");
  }

  if (cfg.kernel_size % 2 == 0 || cfg.kernel_size < 3) {
    convolveLogger->error("Invalid kernel size: {}", cfg.kernel_size);
    throw std::invalid_argument("Kernel size must be odd and >=3");
  }

  if (cfg.kernel.size() !=
      static_cast<size_t>(cfg.kernel_size * cfg.kernel_size)) {
    convolveLogger->error("Kernel size mismatch: expected {}, got {}",
                          cfg.kernel_size * cfg.kernel_size, cfg.kernel.size());
    throw std::invalid_argument("Kernel dimensions mismatch");
  }
  convolveLogger->debug("Convolution config validated");
}

// 改进的验证函数
void Convolve::validateDeconvolutionConfig(const cv::Mat &input,
                                           const DeconvolutionConfig &cfg) {
  if (input.empty()) {
    convolveLogger->error("Input image is empty");
    throw std::invalid_argument("Empty input image");
  }

  if (cfg.iterations <= 0 || cfg.iterations > 1000) {
    convolveLogger->error("Invalid iterations count: {}", cfg.iterations);
    throw std::invalid_argument(
        fmt::format("Invalid iterations count: {}", cfg.iterations));
  }

  if (cfg.noise_power < 0.0 || cfg.noise_power > 1.0) {
    convolveLogger->error("Invalid noise power: {}", cfg.noise_power);
    throw std::invalid_argument(
        fmt::format("Invalid noise power: {}", cfg.noise_power));
  }

  if (cfg.regularization <= 0.0 || cfg.regularization > 1.0) {
    convolveLogger->error("Invalid regularization parameter: {}",
                          cfg.regularization);
    throw std::invalid_argument(fmt::format(
        "Invalid regularization parameter: {}", cfg.regularization));
  }

  if (!cfg.per_channel && input.channels() > 1) {
    convolveLogger->warn("Multi-channel image will be processed as a whole");
  }
  convolveLogger->debug("Deconvolution config validated");
}

// 准备OpenCV兼容的核
cv::Mat Convolve::prepareKernel(const ConvolutionConfig &cfg) {
  convolveLogger->debug("Preparing convolution kernel");
  cv::Mat kernel(cfg.kernel_size, cfg.kernel_size, CV_32F);
  std::copy(cfg.kernel.begin(), cfg.kernel.end(), kernel.ptr<float>());

  if (cfg.normalize_kernel) {
    double sum = cv::sum(kernel)[0];
    if (sum != 0)
      kernel /= sum;
    convolveLogger->debug("Kernel normalized");
  }

  convolveLogger->debug("Kernel prepared successfully");
  return kernel;
}

// 转换边界模式到OpenCV常量
int Convolve::getOpenCVBorderType(BorderMode mode) {
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

// 优化的PSF估计
cv::Mat Convolve::estimatePSF(cv::Size imgSize) {
  ScopedTimer timer("PSF Estimation");
  convolveLogger->debug("Estimating PSF for image size: width={}, height={}",
                        imgSize.width, imgSize.height);

  const int optimal_size =
      cv::getOptimalDFTSize(std::max(imgSize.width, imgSize.height));
  cv::Mat psf = cv::Mat::zeros(optimal_size, optimal_size, CV_32F);

  const int kernelSize = std::min(imgSize.width, imgSize.height) / 20;
  const double sigma = kernelSize / 6.0;
  const cv::Point center(optimal_size / 2, optimal_size / 2);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < optimal_size; i++) {
    for (int j = 0; j < optimal_size; j++) {
      const double dx = j - center.x;
      const double dy = i - center.y;
      const double r2 = dx * dx + dy * dy;
      psf.at<float>(i, j) = std::exp(-r2 / (2 * sigma * sigma));
    }
  }

  cv::normalize(psf, psf, 1.0, 0.0, cv::NORM_L1);
  cv::Mat croppedPsf = psf(cv::Rect(0, 0, imgSize.width, imgSize.height));
  convolveLogger->info("PSF estimated successfully");
  return croppedPsf;
}

// 优化内存池管理
cv::Mat Convolve::MemoryPool::allocate(int rows, int cols, int type) {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  convolveLogger->debug(
      "Allocating memory from pool: rows={}, cols={}, type={}", rows, cols,
      type);

  for (auto it = pool_.begin(); it != pool_.end(); ++it) {
    if (it->rows == rows && it->cols == cols && it->type() == type) {
      cv::Mat mat = *it;
      pool_.erase(it);
      convolveLogger->debug("Memory allocated from pool");
      return mat;
    }
  }
  convolveLogger->debug("No suitable memory found in pool, allocating new");
  return cv::Mat(rows, cols, type);
}

void Convolve::MemoryPool::deallocate(cv::Mat &mat) {
  if (!mat.empty()) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (pool_.size() < max_pool_size_) {
      convolveLogger->debug("Deallocating memory to pool");
      pool_.push_back(mat);
    } else {
      convolveLogger->warn(
          "Memory pool is full, discarding deallocated memory");
    }
    mat = cv::Mat();
  }
}

void Convolve::MemoryPool::clear() {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  convolveLogger->warn("Clearing memory pool");
  pool_.clear();
}

std::vector<cv::Mat> Convolve::MemoryPool::pool_;
std::mutex Convolve::MemoryPool::pool_mutex_;