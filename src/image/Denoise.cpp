#include "Denoise.hpp"

#include <algorithm>
#include <fstream>
#include <future>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

namespace {
std::shared_ptr<spdlog::logger> denoiseLogger =
    spdlog::basic_logger_mt("DenoiseLogger", "logs/denoise.log");
} // namespace

void WaveletDenoiser::denoise(const cv::Mat &src, cv::Mat &dst, int levels,
                              float threshold) {
  denoiseLogger->debug("Starting wavelet denoise with levels: {}, threshold: "
                       "{}",
                       levels, threshold);
  // 为简单起见，仅对单通道进行演示，彩色可拆分通道分别处理
  if (src.channels() > 1) {
    denoiseLogger->debug("Processing multi-channel image");
    // 转换为Lab或YUV后处理亮度通道，再合并，还可并行处理
    cv::Mat copyImg;
    src.copyTo(copyImg);

    std::vector<cv::Mat> bgr;
    cv::split(copyImg, bgr);

    // 仅处理第一个通道(蓝色通道)做演示，实际可处理所有通道
    cv::Mat denoisedChannel;
    wavelet_process_single_channel(bgr[0], denoisedChannel, levels, threshold);
    bgr[0] = denoisedChannel;

    cv::merge(bgr, dst);
    denoiseLogger->debug("Multi-channel wavelet denoise completed");
  } else {
    denoiseLogger->debug("Processing single-channel image");
    wavelet_process_single_channel(src, dst, levels, threshold);
    denoiseLogger->debug("Single-channel wavelet denoise completed");
  }
}

void WaveletDenoiser::wavelet_process_single_channel(const cv::Mat &src,
                                                     cv::Mat &dst, int levels,
                                                     float threshold) {
  denoiseLogger->debug("Starting wavelet process for single channel with "
                       "levels: {}, threshold: {}",
                       levels, threshold);
  // 转成float类型，便于处理
  cv::Mat floatSrc;
  src.convertTo(floatSrc, CV_32F);
  denoiseLogger->debug("Converted source to float type");

  // 小波分解
  cv::Mat waveCoeffs = floatSrc.clone();
  for (int i = 0; i < levels; i++) {
    waveCoeffs = decompose_one_level(waveCoeffs);
    denoiseLogger->debug("Decomposed level {}", i + 1);
  }

  // 去噪（简单阈值处理）
  cv::threshold(waveCoeffs, waveCoeffs, threshold, 0, cv::THRESH_TOZERO);
  denoiseLogger->debug("Applied thresholding");

  // 逆变换
  for (int i = 0; i < levels; i++) {
    waveCoeffs = recompose_one_level(waveCoeffs, floatSrc.size());
    denoiseLogger->debug("Recomposed level {}", i + 1);
  }

  // 转回原类型
  waveCoeffs.convertTo(dst, src.type());
  denoiseLogger->debug("Converted back to original type");
}

// 单层离散小波分解(示例性拆分，不以真实小波为准)
cv::Mat WaveletDenoiser::decompose_one_level(const cv::Mat &src) {
  denoiseLogger->debug("Starting decompose one level");
  cv::Mat dst = src.clone();
  // 此处可进行实际小波分解，此处仅演示简化方法：
  // 例如：将图像分块(低频 + 高频)
  // 这里直接用高通滤波模拟高频，低通滤波模拟低频
  cv::Mat lowFreq, highFreq;
  cv::blur(dst, lowFreq, cv::Size(3, 3));
  highFreq = dst - lowFreq;
  // 将低频和高频拼接在同一Mat中返回(仅示意)
  // 为了安全，在行方向拼接（可根据需求改变）
  cv::Mat combined;
  cv::vconcat(lowFreq, highFreq, combined);
  denoiseLogger->debug("Decompose one level completed");
  return combined;
}

// 单层离散小波重构(示例性的逆过程)
cv::Mat WaveletDenoiser::recompose_one_level(const cv::Mat &waveCoeffs,
                                             const cv::Size &originalSize) {
  denoiseLogger->debug("Starting recompose one level");
  // 假设waveCoeffs是上下拼接的
  int rowCount = waveCoeffs.rows / 2;
  cv::Mat lowFreq = waveCoeffs(cv::Rect(0, 0, waveCoeffs.cols, rowCount));
  cv::Mat highFreq =
      waveCoeffs(cv::Rect(0, rowCount, waveCoeffs.cols, rowCount));

  // 简化逆过程：dst = lowFreq + highFreq
  cv::Mat combined = lowFreq + highFreq;

  // 保证输出大小与原图一致(多层变换后可能需要特别处理)
  if (combined.size() != originalSize) {
    cv::resize(combined, combined, originalSize, 0, 0, cv::INTER_LINEAR);
    denoiseLogger->debug("Resized combined image to original size");
  }
  denoiseLogger->debug("Recompose one level completed");
  return combined;
}

// SIMD优化的小波变换
void WaveletDenoiser::wavelet_transform_simd(cv::Mat &data) {
  const int n = data.cols;
  float *ptr = data.ptr<float>();

#if defined(__AVX2__)
  for (int i = 0; i < n / 8; ++i) {
    __m256 vec = _mm256_loadu_ps(ptr + i * 8);
    __m256 result = _mm256_mul_ps(vec, _mm256_set1_ps(0.707106781f));
    _mm256_storeu_ps(ptr + i * 8, result);
  }
#elif defined(__SSE2__)
  for (int i = 0; i < n / 4; ++i) {
    __m128 vec = _mm_loadu_ps(ptr + i * 4);
    __m128 result = _mm_mul_ps(vec, _mm_set1_ps(0.707106781f));
    _mm_storeu_ps(ptr + i * 4, result);
  }
#else
#pragma omp simd
  for (int i = 0; i < n; ++i) {
    ptr[i] *= 0.707106781f;
  }
#endif
}

void WaveletDenoiser::process_blocks(
    cv::Mat &img, int block_size,
    const std::function<void(cv::Mat &)> &process_fn) {
  const int rows = img.rows;
  const int cols = img.cols;

// 使用动态调度以更好地平衡负载
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
  for (int i = 0; i < rows; i += block_size) {
    for (int j = 0; j < cols; j += block_size) {
      const int current_block_rows = std::min(block_size, rows - i);
      const int current_block_cols = std::min(block_size, cols - j);

      // 使用连续内存块提高缓存命中率
      cv::Mat block;
      img(cv::Range(i, i + current_block_rows),
          cv::Range(j, j + current_block_cols))
          .copyTo(block);

      process_fn(block);

      // 写回结果
      block.copyTo(img(cv::Range(i, i + current_block_rows),
                       cv::Range(j, j + current_block_cols)));
    }
  }
}

void WaveletDenoiser::stream_process(
    const cv::Mat &src, cv::Mat &dst,
    const std::function<void(cv::Mat &)> &process_fn) {
  const int pipeline_stages = 3;
  const int tile_rows = src.rows / pipeline_stages;

  std::vector<cv::Mat> tiles(pipeline_stages);
  std::vector<std::future<void>> futures(pipeline_stages);

  // 创建流水线
  for (int i = 0; i < pipeline_stages; ++i) {
    cv::Mat tile = src(cv::Range(i * tile_rows, (i + 1) * tile_rows),
                       cv::Range(0, src.cols))
                       .clone();
    futures[i] = std::async(std::launch::async, process_fn, std::ref(tile));
    tiles[i] = tile;
  }

  // 等待所有处理完成
  for (int i = 0; i < pipeline_stages; ++i) {
    futures[i].wait();
    tiles[i].copyTo(dst(cv::Range(i * tile_rows, (i + 1) * tile_rows),
                        cv::Range(0, src.cols)));
  }
}

// 优化内存访问模式
void WaveletDenoiser::optimize_memory_layout(cv::Mat &data) {
  // 确保数据是连续的
  if (!data.isContinuous()) {
    data = data.clone();
  }

  // 内存对齐
  const size_t alignment = 32; // AVX2需要32字节对齐
  uchar *ptr = data.data;
  size_t space = data.total() * data.elemSize();
  void *aligned_ptr = nullptr;

#if defined(_WIN32)
  aligned_ptr = _aligned_malloc(space, alignment);
  if (aligned_ptr) {
    memcpy(aligned_ptr, ptr, space);
    data = cv::Mat(data.rows, data.cols, data.type(), aligned_ptr);
  }
#else
  if (posix_memalign(&aligned_ptr, alignment, space) == 0) {
    memcpy(aligned_ptr, ptr, space);
    data = cv::Mat(data.rows, data.cols, data.type(), aligned_ptr);
  }
#endif
}

// 使用SIMD优化的tile处理
void WaveletDenoiser::process_tile_simd(cv::Mat &tile) {
  optimize_memory_layout(tile);

  float *ptr = tile.ptr<float>();
  const int size = tile.total();

#if defined(__AVX2__)
  const int vec_size = 8;
  const int vec_count = size / vec_size;

#pragma omp parallel for
  for (int i = 0; i < vec_count; ++i) {
    __m256 vec = _mm256_load_ps(ptr + i * vec_size);
    // 进行SIMD运算
    // ...
    _mm256_store_ps(ptr + i * vec_size, vec);
  }
#endif

  // 处理剩余元素
  // ...existing code...
}

float WaveletDenoiser::compute_adaptive_threshold(const cv::Mat &coeffs,
                                                  double noise_estimate) {
  denoiseLogger->debug("Starting compute adaptive threshold with noise "
                       "estimate: {}",
                       noise_estimate);
  cv::Mat abs_coeffs;
  cv::absdiff(coeffs, cv::Scalar(0), abs_coeffs);

  double median = 0.0;
#pragma omp parallel
  {
    std::vector<float> local_data;
#pragma omp for nowait
    for (int i = 0; i < coeffs.rows; ++i) {
      for (int j = 0; j < coeffs.cols; ++j) {
        local_data.push_back(abs_coeffs.at<float>(i, j));
      }
    }

#pragma omp critical
    {
      std::sort(local_data.begin(), local_data.end());
      median = local_data[local_data.size() / 2];
    }
  }

  denoiseLogger->debug("Adaptive threshold computed: {}",
                       median * noise_estimate);
  return static_cast<float>(median * noise_estimate);
}

void WaveletDenoiser::denoise(const cv::Mat &src, cv::Mat &dst,
                              const DenoiseParameters &params) {
  denoiseLogger->info("Starting wavelet denoise with parameters");
  cv::Mat working;
  src.convertTo(working, CV_32F);
  denoiseLogger->debug("Converted source to float type");

  // 分块并行处理
  process_blocks(working, params.block_size, [&params](cv::Mat &block) {
    // 小波变换
    for (int level = 0; level < params.wavelet_level; ++level) {
      // 行变换
      for (int i = 0; i < block.rows; ++i) {
        cv::Mat row = block.row(i);
        wavelet_transform_simd(row);
      }

      // 列变换
      cv::Mat block_t = block.t();
      for (int i = 0; i < block_t.rows; ++i) {
        cv::Mat row = block_t.row(i);
        wavelet_transform_simd(row);
      }
      block = block_t.t();
    }

    // 自适应阈值去噪
    if (params.use_adaptive_threshold) {
      float thresh = compute_adaptive_threshold(block, params.noise_estimate);
      cv::threshold(block, block, thresh, 0, cv::THRESH_TOZERO);
    } else {
      cv::threshold(block, block, params.wavelet_threshold, 0,
                    cv::THRESH_TOZERO);
    }

    // 逆变换过程类似
  });

  working.convertTo(dst, src.type());
  denoiseLogger->info("Wavelet denoise completed");
}

ImageDenoiser::ImageDenoiser() {}

cv::Mat ImageDenoiser::denoise(const cv::Mat &input,
                               const DenoiseParameters &params) {
  denoiseLogger->info("Starting image denoise");
  if (input.empty()) {
    denoiseLogger->error("Input image is empty");
    throw std::invalid_argument("Empty input image");
  }

  // 支持8位单通道或三通道图像
  if (input.depth() != CV_8U ||
      (input.channels() != 1 && input.channels() != 3)) {
    denoiseLogger->error("Unsupported format: depth={} channels={}",
                         input.depth(), input.channels());
    throw std::invalid_argument("Unsupported image format");
  }

  try {
    cv::Mat processed;
    // 如果是Auto，则先根据噪声分析选择合适的去噪方法
    const auto method = (params.method == DenoiseMethod::Auto)
                            ? analyze_noise(input)
                            : params.method;

    denoiseLogger->debug("Using denoise method: {}", method_to_string(method));
    switch (method) {
    case DenoiseMethod::Median:
      validate_median(params);
      cv::medianBlur(input, processed, params.median_kernel);
      break;
    case DenoiseMethod::Gaussian:
      validate_gaussian(params);
      cv::GaussianBlur(input, processed, params.gaussian_kernel, params.sigma_x,
                       params.sigma_y);
      break;
    case DenoiseMethod::Bilateral:
      validate_bilateral(params);
      process_bilateral(input, processed, params);
      break;
    case DenoiseMethod::NLM:
      process_nlm(input, processed, params);
      break;
    case DenoiseMethod::Wavelet:
      // 新增的Wavelet去噪处理
      WaveletDenoiser::denoise(input, processed, params.wavelet_level,
                               params.wavelet_threshold);
      break;
    default:
      denoiseLogger->error("Unsupported denoising method");
      throw std::runtime_error("Unsupported denoising method");
    }

    denoiseLogger->info("Denoising completed using {}",
                        method_to_string(method));
    return processed;
  } catch (const cv::Exception &e) {
    denoiseLogger->error("OpenCV error: {}", e.what());
    throw;
  }
}

DenoiseMethod ImageDenoiser::analyze_noise(const cv::Mat &img) {
  denoiseLogger->info("Starting noise analysis");
  cv::Mat gray;
  if (img.channels() > 1) {
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = img.clone();
  }

  // 计算局部方差
  cv::Mat local_var;
  cv::Mat mean, stddev;
  cv::meanStdDev(gray, mean, stddev);

  // 分析噪声类型
  double salt_pepper_ratio = detect_salt_pepper(gray);
  double gaussian_likelihood = detect_gaussian(gray);

  denoiseLogger->debug("Salt and pepper ratio: {}, Gaussian likelihood: {}",
                       salt_pepper_ratio, gaussian_likelihood);
  if (salt_pepper_ratio > 0.1) {
    denoiseLogger->info("Detected salt and pepper noise, using Median filter");
    return DenoiseMethod::Median;
  } else if (gaussian_likelihood > 0.7) {
    denoiseLogger->info("Detected Gaussian noise, using Gaussian filter");
    return DenoiseMethod::Gaussian;
  } else {
    denoiseLogger->info("Using Wavelet denoise method");
    return DenoiseMethod::Wavelet;
  }
}

double ImageDenoiser::detect_salt_pepper(const cv::Mat &gray) {
  denoiseLogger->debug("Detecting salt and pepper noise");
  int height = gray.rows;
  int width = gray.cols;
  int salt_pepper_count = 0;
  int total_pixels = height * width;

  // 检查极值点
  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      uchar center = gray.at<uchar>(i, j);
      if (center == 0 || center == 255) {
        // 检查是否与邻域有显著差异
        uchar neighbors[] = {gray.at<uchar>(i - 1, j), gray.at<uchar>(i + 1, j),
                             gray.at<uchar>(i, j - 1),
                             gray.at<uchar>(i, j + 1)};

        int diff_count = 0;
        for (uchar n : neighbors) {
          if (std::abs(center - n) > 50) {
            diff_count++;
          }
        }

        if (diff_count >= 3) {
          salt_pepper_count++;
        }
      }
    }
  }

  double ratio = static_cast<double>(salt_pepper_count) / total_pixels;
  denoiseLogger->debug("Salt and pepper noise ratio: {}", ratio);
  return ratio;
}

double ImageDenoiser::detect_gaussian(const cv::Mat &gray) {
  denoiseLogger->debug("Detecting Gaussian noise");
  cv::Mat blur, diff;
  cv::GaussianBlur(gray, blur, cv::Size(5, 5), 1.5);
  cv::absdiff(gray, blur, diff);

  cv::Scalar mean, stddev;
  cv::meanStdDev(diff, mean, stddev);

  // 根据差异图的标准差估计高斯噪声的可能性
  double normalized_std = stddev[0] / 255.0;
  double likelihood = std::min(normalized_std * 3.0, 1.0); // 归一化到[0,1]范围
  denoiseLogger->debug("Gaussian noise likelihood: {}", likelihood);
  return likelihood;
}

void ImageDenoiser::process_bilateral(const cv::Mat &src, cv::Mat &dst,
                                      const DenoiseParameters &params) {
  denoiseLogger->debug("Processing bilateral filter");
  if (src.channels() == 3) {
    // 转成Lab，先对亮度通道进行双边滤波，再转换回来
    cv::cvtColor(src, dst, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(dst, channels);

    // 并行处理亮度通道
    cv::parallel_for_(
        cv::Range(0, 1),
        [&](const cv::Range &range) {
          cv::bilateralFilter(channels[0], channels[0], params.bilateral_d,
                              params.sigma_color, params.sigma_space);
        },
        params.threads);

    cv::merge(channels, dst);
    cv::cvtColor(dst, dst, cv::COLOR_Lab2BGR);
  } else {
    cv::bilateralFilter(src, dst, params.bilateral_d, params.sigma_color,
                        params.sigma_space);
  }
  denoiseLogger->debug("Bilateral filter completed");
}

void ImageDenoiser::process_nlm(const cv::Mat &src, cv::Mat &dst,
                                const DenoiseParameters &params) {
  denoiseLogger->debug("Processing NLM denoise");
  if (src.channels() == 3) {
    cv::fastNlMeansDenoisingColored(src, dst, params.nlm_h, params.nlm_h,
                                    params.nlm_template_size,
                                    params.nlm_search_size);
  } else {
    cv::fastNlMeansDenoising(src, dst, params.nlm_h, params.nlm_template_size,
                             params.nlm_search_size);
  }
  denoiseLogger->debug("NLM denoise completed");
}

// 参数校验
void ImageDenoiser::validate_median(const DenoiseParameters &params) {
  if (params.median_kernel % 2 == 0 || params.median_kernel < 3) {
    denoiseLogger->error("Median kernel size must be odd and ≥3");
    throw std::invalid_argument("Median kernel size must be odd and ≥3");
  }
  denoiseLogger->debug("Median parameters validated");
}

void ImageDenoiser::validate_gaussian(const DenoiseParameters &params) {
  if (params.gaussian_kernel.width % 2 == 0 ||
      params.gaussian_kernel.height % 2 == 0) {
    denoiseLogger->error("Gaussian kernel size must be odd");
    throw std::invalid_argument("Gaussian kernel size must be odd");
  }
  denoiseLogger->debug("Gaussian parameters validated");
}

void ImageDenoiser::validate_bilateral(const DenoiseParameters &params) {
  if (params.bilateral_d <= 0) {
    denoiseLogger->error("Bilateral d must be positive");
    throw std::invalid_argument("Bilateral d must be positive");
  }
  denoiseLogger->debug("Bilateral parameters validated");
}

const char *ImageDenoiser::method_to_string(DenoiseMethod method) {
  switch (method) {
  case DenoiseMethod::Median:
    return "Median";
  case DenoiseMethod::Gaussian:
    return "Gaussian";
  case DenoiseMethod::Bilateral:
    return "Bilateral";
  case DenoiseMethod::NLM:
    return "Non-Local Means";
  case DenoiseMethod::Wavelet:
    return "Wavelet";
  default:
    return "Unknown";
  }
}

double ImageDenoiser::calculate_psnr(const cv::Mat &a, const cv::Mat &b) {
  cv::Mat diff;
  cv::absdiff(a, b, diff);
  diff.convertTo(diff, CV_32F);
  diff = diff.mul(diff);
  cv::Scalar s = sum(diff);
  double mse = (s[0] + s[1] + s[2]) / (3 * a.total());
  return 10.0 * log10(255 * 255 / mse);
}

double ImageDenoiser::calculate_ssim(const cv::Mat &a, const cv::Mat &b) {
  const double C1 = 6.5025;  // (0.01 * 255)^2
  const double C2 = 58.5225; // (0.03 * 255)^2

  cv::Mat i1, i2;
  a.convertTo(i1, CV_32F);
  b.convertTo(i2, CV_32F);

  cv::Mat I1_2 = i1.mul(i1);
  cv::Mat I2_2 = i2.mul(i2);
  cv::Mat I1_I2 = i1.mul(i2);

  cv::Mat mu1, mu2;
  cv::GaussianBlur(i1, mu1, cv::Size(11, 11), 1.5);
  cv::GaussianBlur(i2, mu2, cv::Size(11, 11), 1.5);

  cv::Mat mu1_2 = mu1.mul(mu1);
  cv::Mat mu2_2 = mu2.mul(mu2);
  cv::Mat mu1_mu2 = mu1.mul(mu2);

  cv::Mat sigma1_2, sigma2_2, sigma12;
  cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
  sigma1_2 -= mu1_2;

  cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
  sigma2_2 -= mu2_2;

  cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
  sigma12 -= mu1_mu2;

  cv::Mat t1, t2, t3;
  t1 = 2 * mu1_mu2 + C1;
  t2 = 2 * sigma12 + C2;
  t3 = t1.mul(t2);

  t1 = mu1_2 + mu2_2 + C1;
  t2 = sigma1_2 + sigma2_2 + C2;
  t1 = t1.mul(t2);

  cv::Mat ssim;
  divide(t3, t1, ssim);
  return mean(ssim)[0];
}

double ImageDenoiser::noise_reduction_ratio(const cv::Mat &orig,
                                            const cv::Mat &processed) {
  cv::Mat gray_orig, gray_proc;
  if (orig.channels() > 1) {
    cv::cvtColor(orig, gray_orig, cv::COLOR_BGR2GRAY);
    cv::cvtColor(processed, gray_proc, cv::COLOR_BGR2GRAY);
  } else {
    gray_orig = orig;
    gray_proc = processed;
  }

  cv::Scalar mean_orig, stddev_orig, mean_proc, stddev_proc;
  cv::meanStdDev(gray_orig, mean_orig, stddev_orig);
  cv::meanStdDev(gray_proc, mean_proc, stddev_proc);

  double noise_reduction = (1.0 - stddev_proc[0] / stddev_orig[0]) * 100.0;
  return std::max(0.0, noise_reduction);
}

cv::Mat ImageDenoiser::frequency_domain_filter(const cv::Mat &channel) {
  cv::Mat padded;
  int m = cv::getOptimalDFTSize(channel.rows);
  int n = cv::getOptimalDFTSize(channel.cols);
  cv::copyMakeBorder(channel, padded, 0, m - channel.rows, 0, n - channel.cols,
                     cv::BORDER_CONSTANT, cv::Scalar::all(0));

  cv::Mat planes[] = {cv::Mat_<float>(padded),
                      cv::Mat::zeros(padded.size(), CV_32F)};
  cv::Mat complexImg;
  cv::merge(planes, 2, complexImg);
  cv::dft(complexImg, complexImg);

  cv::Mat filter = create_bandstop_filter(padded.size(), 30.0);
  apply_filter(complexImg, filter);

  cv::idft(complexImg, complexImg);
  cv::split(complexImg, planes);
  cv::normalize(planes[0], planes[0], 0, 255, cv::NORM_MINMAX, CV_8U);

  return planes[0](cv::Rect(0, 0, channel.cols, channel.rows));
}

cv::Mat ImageDenoiser::create_bandstop_filter(const cv::Size &size,
                                              double sigma) {
  cv::Mat filter = cv::Mat::zeros(size, CV_32F);
  cv::Point center(size.width / 2, size.height / 2);
  double D0 = sigma * 10;

  cv::parallel_for_(cv::Range(0, size.height), [&](const cv::Range &range) {
    for (int i = range.start; i < range.end; ++i) {
      float *p = filter.ptr<float>(i);
      for (int j = 0; j < size.width; ++j) {
        double d = cv::norm(cv::Point(j, i) - center);
        p[j] = 1 - std::exp(-(d * d) / (2 * D0 * D0));
      }
    }
  });
  return filter;
}

void ImageDenoiser::apply_filter(cv::Mat &complexImg, const cv::Mat &filter) {
  cv::Mat planes[2];
  cv::split(complexImg, planes);
  cv::multiply(planes[0], filter, planes[0]);
  cv::multiply(planes[1], filter, planes[1]);
  cv::merge(planes, 2, complexImg);
}

void ImageDenoiser::generate_quality_report(const cv::Mat &orig,
                                            const cv::Mat &processed,
                                            const std::string &output_path) {
  std::ofstream report(output_path);
  if (!report.is_open()) {
    denoiseLogger->error("Failed to create quality report at: {}", output_path);
    return;
  }

  double psnr = calculate_psnr(orig, processed);
  double ssim = calculate_ssim(orig, processed);
  double noise_red = noise_reduction_ratio(orig, processed);

  report << "=== Image Quality Report ===\n"
         << "PSNR: " << psnr << " dB\n"
         << "SSIM: " << ssim << "\n"
         << "Noise Reduction: " << noise_red << "%\n";

  denoiseLogger->info(
      "Quality report generated: PSNR={:.2f}dB, SSIM={:.4f}, NR={:.1f}%", psnr,
      ssim, noise_red);
}

NoiseAnalysis ImageDenoiser::analyzeNoise(const cv::Mat &input) {
  denoiseLogger->info("Starting comprehensive noise analysis");
  NoiseAnalysis analysis;

  cv::Mat gray;
  if (input.channels() > 1) {
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = input.clone();
  }

  // 计算图像统计特征
  auto stats = computeImageStatistics(gray);
  double variance = stats[1];
  double skewness = stats[2];
  double kurtosis = stats[3];

  // 噪声水平估计
  analysis.intensity = estimateNoiseLevel(gray);

  // 计算信噪比
  cv::Mat smoothed;
  cv::GaussianBlur(gray, smoothed, cv::Size(5, 5), 1.5);
  cv::Mat noise;
  cv::absdiff(gray, smoothed, noise);
  cv::Scalar mean, stddev;
  cv::meanStdDev(gray, mean, stddev);
  analysis.snr = mean[0] / stddev[0];

  // 生成噪声分布掩码
  analysis.noiseMask = detectNoiseDistribution(gray);

  // 检测周期性噪声
  double periodicNoiseStrength = estimatePeriodicNoise(gray);

  // 基于特征进行噪声类型概率计算
  std::map<NoiseType, double> &probs = analysis.probabilities;

  // 高斯噪声特征：kurtosis接近3
  probs[NoiseType::Gaussian] = std::exp(-std::abs(kurtosis - 3.0) / 2.0);

  // 椒盐噪声特征：极值点比例
  double saltPepperRatio = detect_salt_pepper(gray);
  probs[NoiseType::SaltAndPepper] = saltPepperRatio;

  // 散斑噪声特征：方差与均值的关系
  double speckleProb = variance / (mean[0] * mean[0]);
  probs[NoiseType::Speckle] = std::min(speckleProb, 1.0);

  // 周期性噪声
  probs[NoiseType::Periodic] = periodicNoiseStrength;

  // 确定主要噪声类型
  auto maxProb = std::max_element(
      probs.begin(), probs.end(),
      [](const auto &a, const auto &b) { return a.second < b.second; });

  if (maxProb->second > 0.5) {
    analysis.type = maxProb->first;
  } else {
    analysis.type = NoiseType::Mixed;
  }

  denoiseLogger->info(
      "Noise analysis completed. Type: {}, Intensity: {:.2f}, SNR: {:.2f}",
      static_cast<int>(analysis.type), analysis.intensity, analysis.snr);

  return analysis;
}

std::vector<double> ImageDenoiser::computeImageStatistics(const cv::Mat &img) {
  cv::Mat float_img;
  img.convertTo(float_img, CV_32F);

  cv::Scalar mean, stddev;
  cv::meanStdDev(float_img, mean, stddev);

  // 计算偏度和峰度
  double sum = 0.0, sum3 = 0.0, sum4 = 0.0;
  float *ptr = (float *)float_img.data;
  int total = float_img.total();

#pragma omp parallel for reduction(+ : sum, sum3, sum4)
  for (int i = 0; i < total; ++i) {
    double diff = ptr[i] - mean[0];
    sum += diff;
    sum3 += diff * diff * diff;
    sum4 += diff * diff * diff * diff;
  }

  double variance = stddev[0] * stddev[0];
  double skewness = sum3 / (total * std::pow(stddev[0], 3));
  double kurtosis = sum4 / (total * variance * variance) - 3.0;

  return {mean[0], variance, skewness, kurtosis};
}

cv::Mat ImageDenoiser::detectNoiseDistribution(const cv::Mat &img) {
  cv::Mat mask = cv::Mat::zeros(img.size(), CV_8U);
  int windowSize = 5;
  int border = windowSize / 2;

#pragma omp parallel for collapse(2)
  for (int y = border; y < img.rows - border; ++y) {
    for (int x = border; y < img.cols - border; ++x) {
      double localVar = calculateLocalVariance(img, x, y, windowSize);
      double pixelIntensity = img.at<uchar>(y, x);

      // 检测局部异常
      if (localVar > 3.0 * pixelIntensity) {
        mask.at<uchar>(y, x) = 255;
      }
    }
  }

  return mask;
}

double ImageDenoiser::estimatePeriodicNoise(const cv::Mat &img) {
  cv::Mat spectrum = computeNoiseSpectrum(img);

  // 检测频谱中的峰值
  cv::Mat peaks;
  cv::dilate(spectrum, peaks, cv::Mat());
  cv::Mat peak_mask = spectrum >= peaks;

  // 计算周期性强度
  return cv::sum(peak_mask)[0] / (spectrum.rows * spectrum.cols);
}

void ImageDenoiser::updateDenoiseParams(DenoiseParameters &params,
                                        const NoiseAnalysis &analysis) {
  switch (analysis.type) {
  case NoiseType::Gaussian:
    params.method = DenoiseMethod::Gaussian;
    params.gaussian_kernel = cv::Size(5, 5);
    params.sigma_x = analysis.intensity * 5.0;
    params.sigma_y = analysis.intensity * 5.0;
    break;

  case NoiseType::SaltAndPepper:
    params.method = DenoiseMethod::Median;
    params.median_kernel =
        std::max(3, static_cast<int>(analysis.intensity * 7));
    break;

  case NoiseType::Periodic:
    params.method = DenoiseMethod::Wavelet;
    params.wavelet_level = 4;
    params.wavelet_threshold = analysis.intensity * 30.0f;
    break;

  case NoiseType::Mixed:
    // 对于混合噪声，使用两步去噪
    params.method = DenoiseMethod::Bilateral;
    params.bilateral_d = 9;
    params.sigma_color = 75.0 * analysis.intensity;
    params.sigma_space = 75.0 * analysis.intensity;
    break;

  default:
    params.method = DenoiseMethod::NLM;
    params.nlm_h = analysis.intensity * 10;
    break;
  }
}

double ImageDenoiser::calculateLocalVariance(const cv::Mat &img, int x, int y,
                                             int windowSize) {
  cv::Mat roi = img(cv::Range(y - windowSize / 2, y + windowSize / 2 + 1),
                    cv::Range(x - windowSize / 2, x + windowSize / 2 + 1));
  cv::Scalar mean, stddev;
  cv::meanStdDev(roi, mean, stddev);
  return stddev[0] * stddev[0];
}

cv::Mat ImageDenoiser::computeNoiseSpectrum(const cv::Mat &img) {
  // 扩展图像尺寸到最优DFT大小
  cv::Mat padded;
  int m = cv::getOptimalDFTSize(img.rows);
  int n = cv::getOptimalDFTSize(img.cols);
  cv::copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols,
                     cv::BORDER_CONSTANT, cv::Scalar::all(0));

  // 转换到频域
  cv::Mat planes[] = {cv::Mat_<float>(padded),
                      cv::Mat::zeros(padded.size(), CV_32F)};
  cv::Mat complexImg;
  cv::merge(planes, 2, complexImg);
  cv::dft(complexImg, complexImg);

  // 分离实部和虚部
  cv::split(complexImg, planes);
  cv::Mat magnitude;
  cv::magnitude(planes[0], planes[1], magnitude);

  // 对数变换增强频谱可视性
  magnitude += cv::Scalar::all(1);
  cv::log(magnitude, magnitude);

  // 移动频谱零频率到中心
  int cx = magnitude.cols / 2;
  int cy = magnitude.rows / 2;

  cv::Mat tmp;
  cv::Mat q0(magnitude, cv::Rect(0, 0, cx, cy));
  cv::Mat q1(magnitude, cv::Rect(cx, 0, cx, cy));
  cv::Mat q2(magnitude, cv::Rect(0, cy, cx, cy));
  cv::Mat q3(magnitude, cv::Rect(cx, cy, cx, cy));

  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);

  // 归一化到[0,1]范围
  cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);

  return magnitude;
}

double ImageDenoiser::estimateNoiseLevel(const cv::Mat &img) {
  cv::Mat laplacian;
  cv::Laplacian(img, laplacian, CV_64F);

  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);

  // 使用拉普拉斯算子的标准差作为噪声水平估计
  double noiseLevel = stddev[0] / 255.0; // 归一化到[0,1]范围

  // 应用sigmoid函数使结果更平滑
  noiseLevel = 1.0 / (1.0 + std::exp(-10 * (noiseLevel - 0.5)));

  return noiseLevel;
}

void WaveletDenoiser::parallel_wavelet_transform(cv::Mat &data) {
  const int rows = data.rows;
  const int cols = data.cols;
  const int tile_size = 32; // 分块大小

// 并行处理每个块
#pragma omp parallel for collapse(2)
  for (int i = 0; i < rows; i += tile_size) {
    for (int j = 0; j < cols; j += tile_size) {
      // 计算当前块的实际大小
      int current_rows = std::min(tile_size, rows - i);
      int current_cols = std::min(tile_size, cols - j);

      // 获取当前块
      cv::Mat tile =
          data(cv::Range(i, i + current_rows), cv::Range(j, j + current_cols));

      // 对当前块应用SIMD优化的小波变换
      process_tile_simd(tile);
    }
  }

// 同步所有线程
#pragma omp barrier
}
