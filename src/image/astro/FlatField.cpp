#include "FlatField.hpp"
#include <algorithm>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

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
using namespace cv;

std::string FlatQualityMetrics::to_string() const {
  return fmt::format("平场质量评估:\n"
                     "均匀度: {:.2f}%\n"
                     "信噪比: {:.2f}\n"
                     "暗角程度: {:.2f}%\n"
                     "热像素数: {}\n"
                     "总体评分: {:.2f}",
                     uniformity * 100, signal_noise, vignetting * 100,
                     hot_pixels, overall_score);
}

std::string FlatConfig::to_string() const {
  return fmt::format("FlatField配置:\n"
                     "使用中值合成: {}\n"
                     "最小平场值: {:.2f}\n"
                     "使用SIMD: {}\n"
                     "缓存大小: {} KB\n"
                     "调试模式: {}\n"
                     "热像素阈值: {:.2f}\n"
                     "分块大小: {}\n"
                     "保存调试信息: {}\n"
                     "调试输出目录: {}\n"
                     "自动校准: {}",
                     use_median, min_flat_value, use_simd, cache_size,
                     enable_debug, hot_pixel_threshold, block_size,
                     save_debug_info, output_dir, enable_auto_calibration);
}

// 添加blend函数实现
void medianBlend(const vector<Mat> &images, Mat &result) {
  if (images.empty())
    return;

  vector<Mat> channels(images[0].channels());
  vector<float> medians;

  for (int i = 0; i < images[0].rows; ++i) {
    for (int j = 0; j < images[0].cols; ++j) {
      medians.clear();
      for (const auto &img : images) {
        medians.push_back(img.at<float>(i, j));
      }
      std::sort(medians.begin(), medians.end());
      result.at<float>(i, j) = medians[medians.size() / 2];
    }
  }
}

void meanBlend(const vector<Mat> &images, Mat &result) {
  if (images.empty())
    return;
  result = Mat::zeros(images[0].size(), images[0].type());

  for (const auto &img : images) {
    result += img;
  }
  result /= static_cast<float>(images.size());
}

FlatFieldProcessor::FlatFieldProcessor(const FlatConfig &config)
    : config_(config) {
  spdlog::info("FlatFieldProcessor initialized with config: {}",
               config_.to_string());
}

// 主要处理函数
Mat FlatFieldProcessor::process(const vector<Mat> &flat_frames,
                                const Mat &master_bias,
                                const Mat &master_dark) {
  spdlog::info("Starting flat field processing");
  validateInputs(flat_frames);

  master_flat_ = createMasterFlat(flat_frames, master_bias, master_dark);
  quality_metrics_ = evaluateQuality(master_flat_);

  if (config_.save_debug_info) {
    saveDebugInfo();
  }

  spdlog::info("Flat field processing completed");
  return master_flat_;
}

// 获取质量评估结果
FlatQualityMetrics FlatFieldProcessor::getQualityMetrics() const {
  return quality_metrics_;
}

// 生成可视化结果
Mat FlatFieldProcessor::visualizeQuality() const {
  if (master_flat_.empty())
    return Mat();

  // 创建可视化图像
  Mat visualization;
  int width = master_flat_.cols * 2;
  int height = master_flat_.rows;
  visualization.create(height, width, CV_8UC3);

  // 绘制均匀度热图
  Mat uniformity_map;
  createUniformityMap(uniformity_map);

  // 绘制暗角分析
  Mat vignetting_map;
  createVignettingMap(vignetting_map);

  // 水平拼接
  Mat roi1 = visualization(Rect(0, 0, master_flat_.cols, height));
  Mat roi2 =
      visualization(Rect(master_flat_.cols, 0, master_flat_.cols, height));
  uniformity_map.copyTo(roi1);
  vignetting_map.copyTo(roi2);

  // 添加说明文字
  putText(visualization, "Uniformity Map", Point(10, 30), FONT_HERSHEY_SIMPLEX,
          1.0, Scalar(255, 255, 255), 2);
  putText(visualization, "Vignetting Map", Point(master_flat_.cols + 10, 30),
          FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);

  return visualization;
}

// 输入验证
void FlatFieldProcessor::validateInputs(const vector<Mat> &frames) {
  if (frames.empty()) {
    throw std::runtime_error("No flat frames provided");
  }

  const Size first_size = frames[0].size();
  const int first_type = frames[0].type();

  for (const auto &frame : frames) {
    if (frame.size() != first_size || frame.type() != first_type) {
      throw std::runtime_error("Inconsistent flat frame properties");
    }
  }
}

// 创建主平场
Mat FlatFieldProcessor::createMasterFlat(const vector<Mat> &frames,
                                         const Mat &bias, const Mat &dark) {
  spdlog::info("Creating master flat");
  vector<Mat> processed;
  processed.reserve(frames.size());

#ifdef USE_OMP
#pragma omp parallel for if (config_.use_simd)
#endif
  for (size_t i = 0; i < frames.size(); ++i) {
    Mat frame;
    frames[i].convertTo(frame, CV_32F);

    if (!bias.empty()) {
      subtract(frame, bias, frame);
    }
    if (!dark.empty()) {
      subtract(frame, dark, frame);
    }

#ifdef USE_OMP
#pragma omp critical
#endif
    processed.push_back(frame);
  }

  // 创建主平场
  Mat master;
  if (config_.use_median) {
    medianBlend(processed, master);
  } else {
    meanBlend(processed, master);
  }

  // 归一化
  normalize(master, master, 1.0, 0.0, NORM_MINMAX);

  spdlog::info("Master flat created");
  return master;
}

// 评估质量
FlatQualityMetrics FlatFieldProcessor::evaluateQuality(const Mat &flat) {
  FlatQualityMetrics metrics;

  // 计算均匀度
  cv::Scalar mean, stddev;
  meanStdDev(flat, mean, stddev);
  metrics.uniformity = 1.0 - (stddev[0] / mean[0]);

  // 计算信噪比
  metrics.signal_noise = mean[0] / stddev[0];

  // 计算暗角
  metrics.vignetting = calculateVignetting(flat);

  // 检测热像素
  metrics.hot_pixels = detectHotPixels(flat).total();

  // 计算总体评分
  metrics.overall_score = calculateOverallScore(metrics);

  return metrics;
}

// 计算暗角
double FlatFieldProcessor::calculateVignetting(const Mat &flat) {
  Mat center_region = flat(
      cv::Rect(flat.cols / 4, flat.rows / 4, flat.cols / 2, flat.rows / 2));

  cv::Scalar center_mean = mean(center_region);
  cv::Scalar total_mean = mean(flat);

  return center_mean[0] / total_mean[0];
}

// 检测热像素
Mat FlatFieldProcessor::detectHotPixels(const Mat &flat) {
  Mat mean, stddev;
  meanStdDev(flat, mean, stddev);

  Mat hot_pixels;
  double thresh =
      mean.at<double>(0) + config_.hot_pixel_threshold * stddev.at<double>(0);
  threshold(flat, hot_pixels, thresh, 255, THRESH_BINARY);

  return hot_pixels;
}

// 计算总体评分
double
FlatFieldProcessor::calculateOverallScore(const FlatQualityMetrics &metrics) {
  return 0.4 * metrics.uniformity +
         0.3 * std::min(1.0, metrics.signal_noise / 100.0) +
         0.3 * (1.0 - std::abs(1.0 - metrics.vignetting));
}

// 创建均匀度热图
void FlatFieldProcessor::createUniformityMap(Mat &map) const {
  Mat deviation;
  absdiff(master_flat_, mean(master_flat_), deviation);
  normalize(deviation, deviation, 0, 255, NORM_MINMAX);
  applyColorMap(deviation, map, COLORMAP_JET);
}

// 创建暗角分析图
void FlatFieldProcessor::createVignettingMap(Mat &map) const {
  Mat radial_profile;
  computeRadialProfile(master_flat_, radial_profile);
  normalize(radial_profile, radial_profile, 0, 255, NORM_MINMAX);
  applyColorMap(radial_profile, map, COLORMAP_VIRIDIS);
}

// 计算径向剖面
void FlatFieldProcessor::computeRadialProfile(const Mat &flat,
                                              Mat &profile) const {
  Point2f center(flat.cols / 2.0f, flat.rows / 2.0f);
  profile = Mat::zeros(flat.size(), CV_32F);

  for (int y = 0; y < flat.rows; ++y) {
    for (int x = 0; x < flat.cols; ++x) {
      float dx = x - center.x;
      float dy = y - center.y;
      float distance = std::sqrt(dx * dx + dy * dy);
      profile.at<float>(y, x) = distance;
    }
  }
}

// 保存调试信息
void FlatFieldProcessor::saveDebugInfo() {
  spdlog::info("Saving debug information");
  fs::create_directories(config_.output_dir);

  // 保存质量报告
  std::ofstream report(fs::path(config_.output_dir) / "quality_report.txt");
  report << quality_metrics_.to_string() << std::endl;

  // 保存可视化结果
  imwrite((fs::path(config_.output_dir) / "quality_visualization.png").string(),
          visualizeQuality());

  // 保存热像素图
  Mat hot_pixels = detectHotPixels(master_flat_);
  imwrite((fs::path(config_.output_dir) / "hot_pixels.png").string(),
          hot_pixels);

  spdlog::info("Debug information saved");
}

// 平场校正应用函数
Mat apply_flat_correction(const Mat &raw_image, const Mat &master_flat,
                          const Mat &master_bias, const Mat &master_dark,
                          const FlatConfig &config) {
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
