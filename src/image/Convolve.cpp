#include "Convolve.hpp"
#include "../utils/ThreadPool.hpp"
#include "SIMDHelper.hpp"
#include <chrono>
#include <fmt/format.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <variant>


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
    : operation(op), start(std::chrono::steady_clock::now()) {}

Convolve::ScopedTimer::~ScopedTimer() {
  auto duration = std::chrono::steady_clock::now() - start;
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
  spdlog::debug("{} took {} ms", operation, ms.count());
}

cv::Mat Convolve::processMultiChannel(
    const cv::Mat &input,
    const std::function<cv::Mat(const cv::Mat &)> &processor) {
  ScopedTimer timer("Multi-channel Processing");

  std::vector<cv::Mat> channels;
  cv::split(input, channels);

#pragma omp parallel for if (channels.size() > 1)
  for (int i = 0; i < channels.size(); i++) {
    channels[i] = processor(channels[i]);
  }

  cv::Mat result;
  cv::merge(channels, result);
  return result;
}

// 优化的卷积实现
cv::Mat Convolve::convolve(const cv::Mat &input, const ConvolutionConfig &cfg) {
  if (input.channels() > 1 && cfg.per_channel) {
    return processMultiChannel(input, [&](const cv::Mat &channel) {
      return convolveSingleChannel(channel, cfg);
    });
  }
  return convolveSingleChannel(input, cfg);
}

// 单通道卷积实现
cv::Mat Convolve::convolveSingleChannel(const cv::Mat &input,
                                        const ConvolutionConfig &cfg) {
  ScopedTimer timer("Optimized Convolution");

  const int tile_size = cfg.tile_size;
  cv::Mat kernel = prepareKernel(cfg);
  cv::Mat output =
      cfg.use_memory_pool
          ? MemoryPool::allocate(input.rows, input.cols, input.type())
          : cv::Mat(input.rows, input.cols, input.type());

  // 预热缓存
  if (cfg.use_memory_pool) {
    cv::Mat warmup = input(
        cv::Rect(0, 0, std::min(64, input.cols), std::min(64, input.rows)));
    cv::filter2D(warmup, warmup, -1, kernel);
  }

  // 使用线程池进行分块处理
  static DynamicThreadPool threadPool;
  std::vector<std::future<void>> futures;

  for (int y = 0; y < input.rows; y += tile_size) {
    for (int x = 0; x < input.cols; x += tile_size) {
      futures.push_back(threadPool.enqueue([&, x, y]() {
        cv::Rect roi(x, y, std::min(tile_size, input.cols - x),
                     std::min(tile_size, input.rows - y));
        cv::Mat inputTile = input(roi);
        cv::Mat outputTile = output(roi);

        if (cfg.use_simd) {
          SIMDHelper::processImageTile(inputTile, outputTile, kernel, roi);
        } else {
          cv::filter2D(inputTile, outputTile, -1, kernel);
        }
      }));
    }
  }

  // 等待所有任务完成
  for (auto &future : futures) {
    future.wait();
  }

  return output;
}

// 反卷积实现
cv::Mat Convolve::deconvolve(const cv::Mat &input,
                             const DeconvolutionConfig &cfg) {
  ScopedTimer timer("Deconvolution");
  validateDeconvolutionConfig(input, cfg);

  // 如果是多通道图像且需要分通道处理
  if (input.channels() > 1 && cfg.per_channel) {
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
    // 使用内存池分配输出内存
    output = MemoryPool::allocate(input.rows, input.cols, input.type());
    richardsonLucyDeconv(input, psf, output, cfg);
    break;
  }
  case DeconvMethod::WIENER: {
    output = MemoryPool::allocate(input.rows, input.cols, input.type());
    wienerDeconv(input, psf, output, cfg);
    break;
  }
  case DeconvMethod::TIKHONOV: {
    output = MemoryPool::allocate(input.rows, input.cols, input.type());
    tikhonovDeconv(input, psf, output, cfg);
    break;
  }
  default:
    throw std::invalid_argument("Unsupported deconvolution method");
  }

  // 处理完成后确保输出图像在有效范围内
  cv::normalize(output, output, 0, 255, cv::NORM_MINMAX, CV_8U);

  return output;
}

// 单通道反卷积实现
cv::Mat Convolve::deconvolveSingleChannel(const cv::Mat &input,
                                          const DeconvolutionConfig &cfg) {
  validateDeconvolutionConfig(input, cfg);

  cv::Mat output;
  cv::Mat psf = estimatePSF(input.size()); // 实际应用中需要根据情况获取PSF

  switch (cfg.method) {
  case DeconvMethod::RICHARDSON_LUCY:
    richardsonLucyDeconv(input, psf, output, cfg);
    break;
  case DeconvMethod::WIENER:
    wienerDeconv(input, psf, output, cfg);
    break;
  case DeconvMethod::TIKHONOV:
    tikhonovDeconv(input, psf, output, cfg);
    break;
  default:
    throw std::invalid_argument("Unsupported deconvolution method");
  }

  return output;
}

// Richardson-Lucy 反卷积
void Convolve::richardsonLucyDeconv(const cv::Mat &input, const cv::Mat &psf,
                                    cv::Mat &output,
                                    const DeconvolutionConfig &cfg) {
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
  }

  imgEstimate.convertTo(output, CV_8U);
}

// 完整的Wiener反卷积实现
void Convolve::wienerDeconv(const cv::Mat &input, const cv::Mat &psf,
                            cv::Mat &output, const DeconvolutionConfig &cfg) {
  ScopedTimer timer("Wiener Deconvolution");

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
}

// Tikhonov 正则化反卷积
void Convolve::tikhonovDeconv(const cv::Mat &input, const cv::Mat &psf,
                              cv::Mat &output, const DeconvolutionConfig &cfg) {
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
}

// 参数校验
void Convolve::validateConvolutionConfig(const cv::Mat &input,
                                         const ConvolutionConfig &cfg) {
  if (input.empty()) {
    spdlog::error("Input image is empty");
    throw std::invalid_argument("Empty input image");
  }

  if (!cfg.per_channel && input.channels() > 1) {
    spdlog::warn("Multi-channel image will be processed as a whole");
  }

  if (cfg.kernel_size % 2 == 0 || cfg.kernel_size < 3) {
    spdlog::error("Invalid kernel size: {}", cfg.kernel_size);
    throw std::invalid_argument("Kernel size must be odd and >=3");
  }

  if (cfg.kernel.size() !=
      static_cast<size_t>(cfg.kernel_size * cfg.kernel_size)) {
    spdlog::error("Kernel size mismatch: expected {}, got {}",
                  cfg.kernel_size * cfg.kernel_size, cfg.kernel.size());
    throw std::invalid_argument("Kernel dimensions mismatch");
  }
}

// 改进的验证函数
void Convolve::validateDeconvolutionConfig(const cv::Mat &input,
                                           const DeconvolutionConfig &cfg) {
  if (input.empty()) {
    throw std::invalid_argument("Empty input image");
  }

  if (cfg.iterations <= 0 || cfg.iterations > 1000) {
    throw std::invalid_argument(
        fmt::format("Invalid iterations count: {}", cfg.iterations));
  }

  if (cfg.noise_power < 0.0 || cfg.noise_power > 1.0) {
    throw std::invalid_argument(
        fmt::format("Invalid noise power: {}", cfg.noise_power));
  }

  if (cfg.regularization <= 0.0 || cfg.regularization > 1.0) {
    throw std::invalid_argument(fmt::format(
        "Invalid regularization parameter: {}", cfg.regularization));
  }

  if (!cfg.per_channel && input.channels() > 1) {
    spdlog::warn("Multi-channel image will be processed as a whole");
  }
}

// 准备OpenCV兼容的核
cv::Mat Convolve::prepareKernel(const ConvolutionConfig &cfg) {
  cv::Mat kernel(cfg.kernel_size, cfg.kernel_size, CV_32F);
  std::copy(cfg.kernel.begin(), cfg.kernel.end(), kernel.ptr<float>());

  if (cfg.normalize_kernel) {
    double sum = cv::sum(kernel)[0];
    if (sum != 0)
      kernel /= sum;
  }

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
  return psf(cv::Rect(0, 0, imgSize.width, imgSize.height));
}

// 优化内存池管理
cv::Mat Convolve::MemoryPool::allocate(int rows, int cols, int type) {
  std::lock_guard<std::mutex> lock(pool_mutex_);

  for (auto it = pool_.begin(); it != pool_.end(); ++it) {
    if (it->rows == rows && it->cols == cols && it->type() == type) {
      cv::Mat mat = *it;
      pool_.erase(it);
      return mat;
    }
  }
  return cv::Mat(rows, cols, type);
}

void Convolve::MemoryPool::deallocate(cv::Mat &mat) {
  if (!mat.empty()) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (pool_.size() < max_pool_size_) {
      pool_.push_back(mat);
    }
    mat = cv::Mat();
  }
}

void Convolve::MemoryPool::clear() {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  pool_.clear();
}

std::vector<cv::Mat> Convolve::MemoryPool::pool_;
std::mutex Convolve::MemoryPool::pool_mutex_;