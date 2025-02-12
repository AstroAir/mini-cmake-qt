#include <algorithm>
#include <filesystem>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief 天文相机平场校正程序
 *
 * 算法原理:
 * 1. 平场帧预处理：减去Bias和Dark本底
 * 2. 主平场合成：支持均值和中值两种方式
 * 3. 图像校正：使用主平场进行校正
 *
 * 优化策略：
 * 1. SIMD加速运算
 * 2. 多线程并行处理
 * 3. 内存预分配优化
 * 4. 缓存访问优化
 */

namespace fs = std::filesystem;
using cv::Mat;
using std::vector;

// 平场处理配置
struct FlatConfig {
  bool use_median = true;      // 使用中值合成
  float min_flat_value = 0.1f; // 最小平场值
  bool use_simd = true;        // 启用SIMD
  size_t cache_size = 1024;    // 缓存大小(KB)
  bool enable_debug = false;   // 调试模式
};

// 生成主平场帧（自动处理Bias/Dark扣除）
Mat create_master_flat(const vector<Mat> &flat_frames,
                       const Mat &master_bias = Mat(),
                       const Mat &master_dark = Mat(),
                       const FlatConfig &config = FlatConfig()) {
  if (flat_frames.empty())
    throw std::runtime_error("No flat frames provided");

  const int cn = flat_frames[0].channels();
  const int depth = flat_frames[0].depth();
  const cv::Size size = flat_frames[0].size();

  // 验证所有帧的尺寸和类型一致
  for (const auto &f : flat_frames) {
    if (f.size() != size || f.channels() != cn || f.depth() != depth)
      throw std::runtime_error("Inconsistent flat frame properties");
  }

  // 预分配内存
  vector<Mat> processed_flats;
  processed_flats.reserve(flat_frames.size());

// 并行预处理平场帧
#pragma omp parallel for if (config.use_simd)
  for (size_t i = 0; i < flat_frames.size(); ++i) {
    Mat f;
    flat_frames[i].convertTo(f, CV_32FC(cn));

    if (!master_bias.empty()) {
      cv::subtract(f, master_bias, f, cv::noArray(), CV_32F);
    }
    if (!master_dark.empty()) {
      cv::subtract(f, master_dark, f, cv::noArray(), CV_32F);
    }

    cv::threshold(f, f, 0.0, 0.0, cv::THRESH_TOZERO);

#pragma omp critical
    processed_flats.push_back(f);
  }

  // 创建主平场
  Mat master_flat(size, CV_32FC(cn));
  const int layers = static_cast<int>(processed_flats.size());

  // 使用查找表优化访问
  vector<float> pixel_buffer(layers * cn);

// 并行处理像素
#pragma omp parallel for collapse(2) if (config.use_simd) private(pixel_buffer)
  for (int y = 0; y < size.height; ++y) {
    for (int x = 0; x < size.width; ++x) {
      // 收集像素值
      if (cn == 1) {
        for (int i = 0; i < layers; ++i) {
          pixel_buffer[i] = processed_flats[i].at<float>(y, x);
        }
      } else {
        for (int i = 0; i < layers; ++i) {
          const cv::Vec3f &v = processed_flats[i].at<cv::Vec3f>(y, x);
          for (int c = 0; c < cn; ++c) {
            pixel_buffer[i * cn + c] = v[c];
          }
        }
      }

      // 计算统计量
      if (config.use_median) {
        if (cn == 1) {
          auto mid = pixel_buffer.begin() + layers / 2;
          std::nth_element(pixel_buffer.begin(), mid,
                           pixel_buffer.begin() + layers);
          master_flat.at<float>(y, x) = *mid;
        } else {
          cv::Vec3f &v = master_flat.at<cv::Vec3f>(y, x);
          for (int c = 0; c < cn; ++c) {
            auto start = pixel_buffer.begin() + c * layers;
            auto mid = start + layers / 2;
            std::nth_element(start, mid, start + layers);
            v[c] = *mid;
          }
        }
      } else {
        if (cn == 1) {
          master_flat.at<float>(y, x) =
              std::accumulate(pixel_buffer.begin(),
                              pixel_buffer.begin() + layers, 0.0f) /
              layers;
        } else {
          cv::Vec3f &v = master_flat.at<cv::Vec3f>(y, x);
          for (int c = 0; c < cn; ++c) {
            v[c] =
                std::accumulate(pixel_buffer.begin() + c * layers,
                                pixel_buffer.begin() + (c + 1) * layers, 0.0f) /
                layers;
          }
        }
      }
    }
  }

  // 归一化处理
  cv::Scalar mean = cv::mean(master_flat);
  for (int c = 0; c < cn; ++c) {
    if (mean[c] < 1e-6)
      mean[c] = 1.0; // 防止除零
  }
  cv::divide(master_flat, mean, master_flat);

  // 转换回原始位深
  Mat final_flat;
  master_flat.convertTo(final_flat, depth);

  return final_flat;
}

// 平场校正应用函数
Mat apply_flat_correction(const Mat &raw_image, const Mat &master_flat,
                          const Mat &master_bias = Mat(),
                          const Mat &master_dark = Mat(),
                          const FlatConfig &config = FlatConfig()) {
  Mat raw_float, flat_float;
  raw_image.convertTo(raw_float, CV_32F);
  master_flat.convertTo(flat_float, CV_32F);

  if (!master_bias.empty()) {
    cv::subtract(raw_float, master_bias, raw_float, cv::noArray(), CV_32F);
  }
  if (!master_dark.empty()) {
    cv::subtract(raw_float, master_dark, raw_float, cv::noArray(), CV_32F);
  }

  Mat valid_mask = flat_float > config.min_flat_value;
  Mat safe_flat;
  cv::max(flat_float, config.min_flat_value, safe_flat);

  Mat calibrated;
  cv::divide(raw_float, safe_flat, calibrated);

  Mat final_image;
  calibrated.convertTo(final_image, raw_image.type());

  return final_image;
}

int main() {
  try {
    FlatConfig config;
    config.use_median = true;
    config.use_simd = true;
    config.min_flat_value = 0.1f;

    // 读取平场帧
    vector<Mat> flat_frames;
    for (const auto &entry : fs::directory_iterator("flats")) {
      Mat flat = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
      if (!flat.empty())
        flat_frames.push_back(flat);
    }

    // 生成主平场
    Mat master_flat = create_master_flat(flat_frames);

    // 读取并校正科学图像
    Mat science = cv::imread("science_image.tif", cv::IMREAD_UNCHANGED);
    Mat calibrated =
        apply_flat_correction(science, master_flat, Mat(), Mat(), config);

    cv::imwrite("calibrated_image.tif", calibrated);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}