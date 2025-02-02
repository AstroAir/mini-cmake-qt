#include "Denoise.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

void WaveletDenoiser::denoise(const cv::Mat &src, cv::Mat &dst, int levels,
                              float threshold) {
  // 为简单起见，仅对单通道进行演示，彩色可拆分通道分别处理
  if (src.channels() > 1) {
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
  } else {
    wavelet_process_single_channel(src, dst, levels, threshold);
  }
}

void WaveletDenoiser::wavelet_process_single_channel(const cv::Mat &src,
                                                     cv::Mat &dst, int levels,
                                                     float threshold) {
  // 转成float类型，便于处理
  cv::Mat floatSrc;
  src.convertTo(floatSrc, CV_32F);

  // 小波分解
  cv::Mat waveCoeffs = floatSrc.clone();
  for (int i = 0; i < levels; i++) {
    waveCoeffs = decompose_one_level(waveCoeffs);
  }

  // 去噪（简单阈值处理）
  cv::threshold(waveCoeffs, waveCoeffs, threshold, 0, cv::THRESH_TOZERO);

  // 逆变换
  for (int i = 0; i < levels; i++) {
    waveCoeffs = recompose_one_level(waveCoeffs, floatSrc.size());
  }

  // 转回原类型
  waveCoeffs.convertTo(dst, src.type());
}

// 单层离散小波分解(示例性拆分，不以真实小波为准)
cv::Mat WaveletDenoiser::decompose_one_level(const cv::Mat &src) {
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
  return combined;
}

// 单层离散小波重构(示例性的逆过程)
cv::Mat WaveletDenoiser::recompose_one_level(const cv::Mat &waveCoeffs,
                                             const cv::Size &originalSize) {
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
  }
  return combined;
}

void WaveletDenoiser::process_blocks(
    cv::Mat &img, int block_size,
    const std::function<void(cv::Mat &)> &process_fn) {
  const int rows = img.rows;
  const int cols = img.cols;

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < rows; i += block_size) {
    for (int j = 0; j < cols; j += block_size) {
      cv::Rect block(j, i, std::min(block_size, cols - j),
                     std::min(block_size, rows - i));
      cv::Mat block_roi = img(block);
      process_fn(block_roi);
    }
  }
}

// SIMD优化的小波变换
void WaveletDenoiser::wavelet_transform_simd(cv::Mat &data) {
  const int n = data.cols;
  auto *ptr = data.ptr<float>();

#pragma omp simd
  for (int i = 0; i < n / 2; ++i) {
    const float a = ptr[2 * i];
    const float b = ptr[2 * i + 1];
    ptr[i] = (a + b) * 0.707106781f; // 1/sqrt(2)
    ptr[i + n / 2] = (a - b) * 0.707106781f;
  }
}

// 自适应阈值计算
float WaveletDenoiser::compute_adaptive_threshold(const cv::Mat &coeffs,
                                                  double noise_estimate) {
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

  return static_cast<float>(median * noise_estimate);
}

void WaveletDenoiser::denoise(const cv::Mat &src, cv::Mat &dst,
                              const DenoiseParameters &params) {
  cv::Mat working;
  src.convertTo(working, CV_32F);

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
}

ImageDenoiser::ImageDenoiser() {}

cv::Mat ImageDenoiser::denoise(const cv::Mat &input,
                               const DenoiseParameters &params) {
  if (input.empty()) {
    spdlog::error("Input image is empty");
    throw std::invalid_argument("Empty input image");
  }

  // 支持8位单通道或三通道图像
  if (input.depth() != CV_8U ||
      (input.channels() != 1 && input.channels() != 3)) {
    spdlog::error("Unsupported format: depth={} channels={}", input.depth(),
              input.channels());
    throw std::invalid_argument("Unsupported image format");
  }

  try {
    cv::Mat processed;
    // 如果是Auto，则先根据噪声分析选择合适的去噪方法
    const auto method = (params.method == DenoiseMethod::Auto)
                            ? analyze_noise(input)
                            : params.method;

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
      throw std::runtime_error("Unsupported denoising method");
    }

    spdlog::info("Denoising completed using {}", method_to_string(method));
    return processed;
  } catch (const cv::Exception &e) {
    spdlog::error("OpenCV error: {}", e.what());
    throw;
  }
}

DenoiseMethod ImageDenoiser::analyze_noise(const cv::Mat &img) {
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

  if (salt_pepper_ratio > 0.1) {
    return DenoiseMethod::Median;
  } else if (gaussian_likelihood > 0.7) {
    return DenoiseMethod::Gaussian;
  } else {
    return DenoiseMethod::Wavelet;
  }
}

double ImageDenoiser::detect_salt_pepper(const cv::Mat &gray) {
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

  return static_cast<double>(salt_pepper_count) / total_pixels;
}

double ImageDenoiser::detect_gaussian(const cv::Mat &gray) {
  cv::Mat blur, diff;
  cv::GaussianBlur(gray, blur, cv::Size(5, 5), 1.5);
  cv::absdiff(gray, blur, diff);

  cv::Scalar mean, stddev;
  cv::meanStdDev(diff, mean, stddev);

  // 根据差异图的标准差估计高斯噪声的可能性
  double normalized_std = stddev[0] / 255.0;
  return std::min(normalized_std * 3.0, 1.0); // 归一化到[0,1]范围
}

void ImageDenoiser::process_bilateral(const cv::Mat &src, cv::Mat &dst,
                                      const DenoiseParameters &params) {
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
}

void ImageDenoiser::process_nlm(const cv::Mat &src, cv::Mat &dst,
                                const DenoiseParameters &params) {
  if (src.channels() == 3) {
    cv::fastNlMeansDenoisingColored(src, dst, params.nlm_h, params.nlm_h,
                                    params.nlm_template_size,
                                    params.nlm_search_size);
  } else {
    cv::fastNlMeansDenoising(src, dst, params.nlm_h, params.nlm_template_size,
                             params.nlm_search_size);
  }
}

// 参数校验
void ImageDenoiser::validate_median(const DenoiseParameters &params) {
  if (params.median_kernel % 2 == 0 || params.median_kernel < 3) {
    throw std::invalid_argument("Median kernel size must be odd and ≥3");
  }
}

void ImageDenoiser::validate_gaussian(const DenoiseParameters &params) {
  if (params.gaussian_kernel.width % 2 == 0 ||
      params.gaussian_kernel.height % 2 == 0) {
    throw std::invalid_argument("Gaussian kernel size must be odd");
  }
}

void ImageDenoiser::validate_bilateral(const DenoiseParameters &params) {
  if (params.bilateral_d <= 0) {
    throw std::invalid_argument("Bilateral d must be positive");
  }
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