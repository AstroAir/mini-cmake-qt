#include <chrono>
#include <fmt/format.h>
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
  bool per_channel = false; // 是否对每个通道单独处理
};

struct DeconvolutionConfig {
  DeconvMethod method = DeconvMethod::RICHARDSON_LUCY;
  int iterations = 30;
  double noise_power = 0.0;
  double regularization = 1e-6;
  BorderMode border_mode = BorderMode::REPLICATE;
  bool per_channel = false; // 是否对每个通道单独处理
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
    ScopedTimer timer("Convolution");

    try {
      validateConvolutionConfig(input, cfg);

      cv::Mat kernel = prepareKernel(cfg);
      cv::Mat output;

      const int borderType = getOpenCVBorderType(cfg.border_mode);
      const cv::Point anchor(cfg.kernel_size / 2, cfg.kernel_size / 2);

      // 使用ROI优化边界处理
      cv::Mat padded;
      const int pad = cfg.kernel_size / 2;
      cv::copyMakeBorder(input, padded, pad, pad, pad, pad, borderType);

      if (cfg.parallel_execution) {
        const int threads = std::thread::hardware_concurrency();
        cv::setNumThreads(threads);
        cv::filter2D(padded, output, -1, kernel, anchor, 0,
                     cv::BORDER_ISOLATED);
        cv::setNumThreads(1);
      } else {
        cv::filter2D(padded, output, -1, kernel, anchor, 0,
                     cv::BORDER_ISOLATED);
      }

      return output(cv::Rect(pad, pad, input.cols, input.rows));
    } catch (const cv::Exception &e) {
      spdlog::error("OpenCV error during convolution: {}", e.what());
      throw;
    } catch (const std::exception &e) {
      spdlog::error("Error during convolution: {}", e.what());
      throw;
    }
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

    for (int i = 0; i < cfg.iterations; ++i) {
      cv::Mat convResult;
      cv::filter2D(imgEstimate, convResult, -1, psf, cv::Point(-1, -1), 0,
                   borderType);

      cv::Mat relativeBlur;
      cv::divide(input, convResult, relativeBlur);

      cv::Mat errorEstimate;
      cv::filter2D(relativeBlur, errorEstimate, -1, psfFlip, cv::Point(-1, -1),
                   0, borderType);

      cv::multiply(imgEstimate, errorEstimate, imgEstimate);
    }

    imgEstimate.convertTo(output, CV_8U);
  }

  // 优化的Wiener反卷积
  static void wienerDeconv(const cv::Mat &input, const cv::Mat &psf,
                           cv::Mat &output, const DeconvolutionConfig &cfg) {
    ScopedTimer timer("Wiener Deconvolution");

    const int optimal_rows = cv::getOptimalDFTSize(input.rows);
    const int optimal_cols = cv::getOptimalDFTSize(input.cols);

    cv::Mat padded_input, padded_psf;
    cv::copyMakeBorder(input, padded_input, 0, optimal_rows - input.rows, 0,
                       optimal_cols - input.cols, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(psf, padded_psf, 0, optimal_rows - psf.rows, 0,
                       optimal_cols - psf.cols, cv::BORDER_CONSTANT, 0);

    cv::Mat inputSpectrum, psfSpectrum;
    cv::dft(padded_input, inputSpectrum, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(padded_psf, psfSpectrum, cv::DFT_COMPLEX_OUTPUT);

    // 添加噪声功率谱估计
    const double nsr = std::max(cfg.noise_power, 1e-6);
    cv::Mat psfPower;
    cv::mulSpectrums(psfSpectrum, psfSpectrum, psfPower, 0, true);

    cv::Mat wienerFilter;
    cv::divide(cv::abs(psfSpectrum), psfPower + nsr, wienerFilter);

    cv::Mat result;
    cv::mulSpectrums(inputSpectrum, wienerFilter, result, 0);

    cv::Mat restored;
    cv::idft(result, restored, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    restored(cv::Rect(0, 0, input.cols, input.rows)).convertTo(output, CV_8U);
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
};

} // namespace ImageProcessing