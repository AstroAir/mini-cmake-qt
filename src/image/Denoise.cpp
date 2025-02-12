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

namespace fs = std::filesystem;
using namespace cv;
using namespace std::chrono;

enum class DenoiseMethod { MEDIAN, GAUSSIAN, NLM, FREQUENCY, AUTO };
enum class NoiseType { GAUSSIAN, SPECKLE, IMPULSE, UNKNOWN };

struct ProcessingConfig {
    DenoiseMethod method = DenoiseMethod::AUTO;
    NoiseType noise_type = NoiseType::UNKNOWN;
    int kernel_size = 3;
    double sigma = 1.0;
    bool benchmark = false;
    bool save_report = false;
    bool use_gpu = false;
    fs::path output_dir = "results";
};

NoiseType detect_noise_type(const Mat& channel) {
    Mat hist;
    const int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    
    calcHist(&channel, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    normalize(hist, hist, 0, 1, NORM_MINMAX);

    double entropy = 0;
    for(int i = 0; i < histSize; ++i) {
        float prob = hist.at<float>(i);
        if(prob > 0)
            entropy -= prob * log2(prob);
    }

    if(entropy > 7.0) return NoiseType::GAUSSIAN;
    
    Mat diff;
    Laplacian(channel, diff, CV_16S);
    Scalar mean, stddev;
    meanStdDev(diff, mean, stddev);
    
    return (stddev[0] > 25) ? NoiseType::IMPULSE : NoiseType::SPECKLE;
}

class DenoiseProcessor {
public:
    explicit DenoiseProcessor(const ProcessingConfig& config) : config(config) {
        if(config.use_gpu && cuda::getCudaEnabledDeviceCount() == 0) {
            throw std::runtime_error("CUDA acceleration requested but no GPU available");
        }
    }

    Mat process(const Mat& input) {
        vector<Mat> channels;
        split(input, channels);

        if(config.noise_type == NoiseType::UNKNOWN) {
            config.noise_type = detect_noise_type(channels[1]);
        }

        if(config.method == DenoiseMethod::AUTO) {
            config.method = select_method_auto();
        }

        process_green_channel(channels[1]);

        Mat result;
        merge(channels, result);
        return result;
    }

private:
    ProcessingConfig config;

    DenoiseMethod select_method_auto() {
        switch(config.noise_type) {
            case NoiseType::IMPULSE: return DenoiseMethod::MEDIAN;
            case NoiseType::GAUSSIAN: return DenoiseMethod::GAUSSIAN;
            case NoiseType::SPECKLE: return DenoiseMethod::NLM;
            default: return DenoiseMethod::MEDIAN;
        }
    }

    void process_green_channel(Mat& green_channel) {
        switch(config.method) {
            case DenoiseMethod::MEDIAN:
                medianBlur(green_channel, green_channel, config.kernel_size);
                break;
            case DenoiseMethod::GAUSSIAN: {
                Size ksize(config.kernel_size, config.kernel_size);
                GaussianBlur(green_channel, green_channel, ksize, config.sigma);
                break;
            }
            case DenoiseMethod::NLM:
                fastNlMeansDenoising(green_channel, green_channel, 
                                   config.kernel_size, 7, 21);
                break;
            case DenoiseMethod::FREQUENCY:
                green_channel = frequency_domain_filter(green_channel);
                break;
            default:
                throw std::invalid_argument("Unsupported denoising method");
        }
    }

    Mat frequency_domain_filter(const Mat& channel) {
        Mat padded;
        int m = getOptimalDFTSize(channel.rows);
        int n = getOptimalDFTSize(channel.cols);
        copyMakeBorder(channel, padded, 0, m - channel.rows, 0, n - channel.cols, 
                     BORDER_CONSTANT, Scalar::all(0));

        Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
        Mat complexImg;
        merge(planes, 2, complexImg);
        dft(complexImg, complexImg);

        Mat filter = create_bandstop_filter(padded.size());
        apply_filter(complexImg, filter);

        idft(complexImg, complexImg);
        split(complexImg, planes);
        normalize(planes[0], planes[0], 0, 255, NORM_MINMAX, CV_8U);
        
        return planes[0](Rect(0, 0, channel.cols, channel.rows));
    }

    Mat create_bandstop_filter(Size size) {
        Mat filter = Mat::zeros(size, CV_32F);
        Point center = Point(size.width/2, size.height/2);
        double D0 = config.sigma * 10;
        
        parallel_for_(Range(0, size.height), [&](const Range& range) {
            for(int i = range.start; i < range.end; ++i) {
                float* p = filter.ptr<float>(i);
                for(int j = 0; j < size.width; ++j) {
                    double d = norm(Point(j, i) - center);
                    p[j] = 1 - exp(-(d*d)/(2*D0*D0));
                }
            }
        });
        return filter;
    }

    void apply_filter(Mat& complexImg, const Mat& filter) {
        Mat planes[2];
        split(complexImg, planes);
        multiply(planes[0], filter, planes[0]);
        multiply(planes[1], filter, planes[1]);
        merge(planes, 2, complexImg);
    }
};

int main(int argc, char** argv) {
    ProcessingConfig config;
    CLI::App app{"Advanced Green Noise Removal"};

    // 命令行参数配置
    app.add_option("-i,--input", config.input_path, "Input image path")->required();
    app.add_option("-o,--output", config.output_dir, "Output directory")
       ->default_val("results");
    app.add_option("-m,--method", config.method,
                  "Denoising method: 0=Median,1=Gaussian,2=NLM,3=Frequency,4=Auto")
       ->default_val(DenoiseMethod::AUTO)
       ->transform(CLI::CheckedTransformer(std::map<std::string, DenoiseMethod>{
           {"median", DenoiseMethod::MEDIAN},
           {"gaussian", DenoiseMethod::GAUSSIAN},
           {"nlm", DenoiseMethod::NLM},
           {"frequency", DenoiseMethod::FREQUENCY},
           {"auto", DenoiseMethod::AUTO}}));
    app.add_option("-k,--kernel", config.kernel_size, "Filter kernel size")
       ->check(CLI::Range(3, 15).description("Odd numbers only"));
    app.add_option("-s,--sigma", config.sigma, "Filter sigma value")
       ->check(CLI::Range(0.1, 10.0));
    app.add_flag("-b,--benchmark", config.benchmark, "Enable performance benchmarking");
    app.add_flag("-g,--gpu", config.use_gpu, "Enable GPU acceleration");
    app.add_flag("-r,--report", config.save_report, "Generate quality report");

    CLI11_PARSE(app, argc, argv);

    try {
        // 初始化处理
        DenoiseProcessor processor(config);
        Mat input = imread(config.input_path.string(), IMREAD_COLOR);
        
        if(config.use_gpu) {
            cuda::GpuMat gpu_img;
            gpu_img.upload(input);
            processor.process(gpu_img);
            gpu_img.download(input);
        } else {
            TickMeter tm;
            if(config.benchmark) tm.start();
            
            Mat result = processor.process(input);
            
            if(config.benchmark) {
                tm.stop();
                cout << "Processing time: " << tm.getTimeMilli() << " ms\n";
            }

            // 保存结果和报告
            fs::create_directories(config.output_dir);
            fs::path output_path = config.output_dir / config.input_path.filename();
            imwrite(output_path.string(), result);

            if(config.save_report) {
                generate_quality_report(input, result, output_path);
            }
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

// 质量报告生成函数示例
void generate_quality_report(const Mat& orig, const Mat& processed, 
                           const fs::path& output_path) {
    ofstream report(output_path.replace_extension(".txt"));
    
    auto calculate_psnr = [](const Mat& a, const Mat& b) {
        Mat diff;
        absdiff(a, b, diff);
        diff.convertTo(diff, CV_32F);
        diff = diff.mul(diff);
        Scalar s = sum(diff);
        double mse = (s[0] + s[1] + s[2]) / (3 * a.total());
        return 10.0 * log10(255*255 / mse);
    };

    report << "=== Image Quality Report ===\n"
           << "PSNR: " << calculate_psnr(orig, processed) << " dB\n"
           << "SSIM: " << calculate_ssim(orig, processed) << "\n"
           << "Noise Reduction: " << noise_reduction_ratio(orig, processed) << "%\n";
}

// 编译依赖：需要链接OpenCV和CLI11库