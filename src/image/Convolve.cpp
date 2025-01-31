#include "Convolve.hpp"
#include "../utils/ThreadPool.hpp"
#include <chrono>
#include <fmt/format.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <variant>


namespace ImageProcessing {

enum class BorderMode { ZERO_PADDING, MIRROR_REFLECT, REPLICATE, CIRCULAR };

enum class DeconvMethod { RICHARDSON_LUCY, WIENER, TIKHONOV };

// 扩展配置结构
struct ConvolutionConfig {
  std::vector<float> kernel;
  int kernel_size;
  BorderMode border_mode = BorderMode::REPLICATE;
  bool normalize_kernel = true;
  bool parallel_execution = true;
  bool per_channel = false;    // 是否对每个通道单独处理
  int tile_size = 64;          // 分块大小
  bool use_memory_pool = true; // 是否使用内存池
  bool use_simd = true;        // 是否使用SIMD优化
};

struct DeconvolutionConfig {
  DeconvMethod method = DeconvMethod::RICHARDSON_LUCY;
  int iterations = 30;
  double noise_power = 0.0;
  double regularization = 1e-6;
  BorderMode border_mode = BorderMode::REPLICATE;
  bool per_channel = false; // 是否对每个通道单独处理
  int tile_size = 64;       // 分块大小
};

class ImageProcessor {
public:
  // 通用处理接口
  static cv::Mat
  process(const cv::Mat &input,
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

private:
  // 添加性能监控
  struct ScopedTimer {
    std::string operation;
    std::chrono::steady_clock::time_point start;

    ScopedTimer(const std::string &op)
        : operation(op), start(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
      auto duration = std::chrono::steady_clock::now() - start;
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
      spdlog::debug("{} took {} ms", operation, ms.count());
    }
  };

  // 添加通道处理函数
  static cv::Mat processMultiChannel(
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
  static cv::Mat convolve(const cv::Mat &input, const ConvolutionConfig &cfg) {
    if (input.channels() > 1 && cfg.per_channel) {
      return processMultiChannel(input, [&](const cv::Mat &channel) {
        return convolveSingleChannel(channel, cfg);
      });
    }
    return convolveSingleChannel(input, cfg);
  }

  // 单通道卷积实现
  static cv::Mat convolveSingleChannel(const cv::Mat &input,
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
  static cv::Mat deconvolve(const cv::Mat &input,
                            const DeconvolutionConfig &cfg) {
    if (input.channels() > 1 && cfg.per_channel) {
      return processMultiChannel(input, [&](const cv::Mat &channel) {
        return deconvolveSingleChannel(channel, cfg);
      });
    }
    return deconvolveSingleChannel(input, cfg);
  }

  // 单通道反卷积实现
  static cv::Mat deconvolveSingleChannel(const cv::Mat &input,
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
  static void richardsonLucyDeconv(const cv::Mat &input, const cv::Mat &psf,
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
        cv::filter2D(relativeBlur, errorEstimate, -1, psfFlip,
                     cv::Point(-1, -1), 0, borderType);
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
  static void wienerDeconv(const cv::Mat &input, const cv::Mat &psf,
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
  static void tikhonovDeconv(const cv::Mat &input, const cv::Mat &psf,
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
  static void validateConvolutionConfig(const cv::Mat &input,
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
  static void validateDeconvolutionConfig(const cv::Mat &input,
                                          const DeconvolutionConfig &cfg) {
    if (input.empty()) {
      throw std::invalid_argument("Empty input image");
    }

    if (input.type() != CV_8UC1 && input.type() != CV_32FC1) {
      throw std::invalid_argument("Unsupported image type");
    }

    if (!cfg.per_channel && input.channels() > 1) {
      spdlog::warn("Multi-channel image will be processed as a whole");
    }

    if (cfg.iterations <= 0 || cfg.iterations > 1000) {
      throw std::invalid_argument("Invalid iteration count");
    }

    if (cfg.noise_power < 0 || cfg.noise_power > 1.0) {
      throw std::invalid_argument("Invalid noise power");
    }

    if (cfg.regularization <= 0 || cfg.regularization > 1.0) {
      throw std::invalid_argument("Invalid regularization parameter");
    }
  }

  // 准备OpenCV兼容的核
  static cv::Mat prepareKernel(const ConvolutionConfig &cfg) {
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
  static int getOpenCVBorderType(BorderMode mode) {
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
  static cv::Mat estimatePSF(cv::Size imgSize) {
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
  struct MemoryPool {
    static std::mutex pool_mutex_;
    static std::vector<cv::Mat> pool_;
    static size_t max_pool_size_;

    static cv::Mat allocate(int rows, int cols, int type) {
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

    static void deallocate(cv::Mat &mat) {
      if (!mat.empty()) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        if (pool_.size() < max_pool_size_) {
          pool_.push_back(mat);
        }
        mat = cv::Mat();
      }
    }

    static void clear() {
      std::lock_guard<std::mutex> lock(pool_mutex_);
      pool_.clear();
    }

    static void setMaxPoolSize(size_t size) { max_pool_size_ = size; }
  };

  // SIMD优化实现
  struct SIMDHelper {
    static void processImageTile(const cv::Mat &src, cv::Mat &dst,
                                 const cv::Mat &kernel, const cv::Rect &roi) {
#ifdef __AVX2__
      if (src.type() == CV_32F && kernel.type() == CV_32F) {
        convolve2D_AVX2(src, dst, kernel, roi);
        return;
      }
#endif
      cv::filter2D(src, dst, -1, kernel);
    }

  private:
#ifdef __AVX2__
    static void convolve2D_AVX2(const cv::Mat &src, cv::Mat &dst,
                                const cv::Mat &kernel, const cv::Rect &roi) {
      // AVX2优化的实现
      const int ksize = kernel.rows;
      const int radius = ksize / 2;

      for (int y = roi.y; y < roi.y + roi.height; y++) {
        for (int x = roi.x; x < roi.x + roi.width; x += 8) {
          __m256 sum = _mm256_setzero_ps();

          for (int ky = 0; ky < ksize; ky++) {
            for (int kx = 0; kx < ksize; kx++) {
              int sy = y + ky - radius;
              int sx = x + kx - radius;

              if (sy >= 0 && sy < src.rows && sx >= 0 && sx < src.cols) {
                __m256 src_val = _mm256_loadu_ps(&src.at<float>(sy, sx));
                __m256 kernel_val = _mm256_set1_ps(kernel.at<float>(ky, kx));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(src_val, kernel_val));
              }
            }
          }

          _mm256_storeu_ps(&dst.at<float>(y, x), sum);
        }
      }
    }
#endif
  };

  // 添加缓存友好的数据结构处理
  class CacheAlignedBuffer {
    static constexpr size_t CACHE_LINE = 64;
    std::unique_ptr<float[]> data_;
    size_t size_;

  public:
    explicit CacheAlignedBuffer(size_t size) : size_(size) {
      size_t aligned_size = (size + CACHE_LINE - 1) & ~(CACHE_LINE - 1);
      data_ = std::make_unique<float[]>(aligned_size);
    }

    float *data() { return data_.get(); }
    const float *data() const { return data_.get(); }
    size_t size() const { return size_; }

    void prefetch(size_t index) const { __builtin_prefetch(&data_[index]); }

    float &operator[](size_t index) { return data_[index]; }
    const float &operator[](size_t index) const { return data_[index]; }
  };
};

std::vector<cv::Mat> ImageProcessor::MemoryPool::pool_;
std::mutex ImageProcessor::MemoryPool::pool_mutex_;
size_t ImageProcessor::MemoryPool::max_pool_size_ = 100;

} // namespace ImageProcessing