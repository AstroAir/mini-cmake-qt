#include "Denoise.hpp"

#include <algorithm>
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