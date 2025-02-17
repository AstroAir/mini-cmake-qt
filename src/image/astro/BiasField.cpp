#include "BiasField.hpp"

#include <algorithm>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <vector>

namespace fs = std::filesystem;

// 宏定义控制
#define USE_OPENMP
#define USE_CUDA

std::string QualityMetrics::to_string() const {
  return fmt::format("质量评估结果:\n"
                     "噪声均匀性: {:.2f}\n"
                     "时间稳定性: {:.2f}\n"
                     "空间均匀性: {:.2f}\n"
                     "异常值数量: {}\n"
                     "总体评分: {:.2f}",
                     noise_uniformity, temporal_stability, spatial_uniformity,
                     outlier_count, overall_score);
}

BiasProcessor::BiasProcessor(const BiasConfig &config) : config_(config) {
  spdlog::info("BiasProcessor 初始化完成");
}

cv::Mat BiasProcessor::create_master_bias(const std::vector<cv::Mat> &frames) {
  spdlog::info("开始创建主Bias帧");
  validate_input(frames);

  frames_ = frames; // 保存输入帧

  const cv::Size frame_size = frames[0].size();
  const int type = frames[0].type();

  // 预分配内存
  cv::Mat master(frame_size, CV_32FC(CV_MAT_CN(type)));

  // 并行分块处理
  process_blocks(frames, master);

  master_ = master.clone(); // 保存主偏置帧

  // 质量评估
  quality_metrics_ = evaluate_quality(frames, master);

  if (quality_metrics_.overall_score < config_.quality_threshold) {
    spdlog::warn("主Bias质量低于阈值: {:.2f}", quality_metrics_.overall_score);
  }

  spdlog::info("主Bias帧创建完成");
  return convert_output(master, type);
}

QualityMetrics
BiasProcessor::analyze_noise(const cv::Mat &master,
                             const std::vector<cv::Mat> &frames) {
  spdlog::info("开始噪声分析");
  cv::Mat variance, temporal_std;
  compute_statistics(frames, master, variance, temporal_std);

  // 生成噪声分析报告
  generate_noise_report(variance, temporal_std);

  spdlog::info("噪声分析完成");
  return evaluate_quality(frames, master);
}

void BiasProcessor::save_results(const cv::Mat &master,
                                 const fs::path &output_dir) {
  spdlog::info("开始保存结果");
  fs::create_directories(output_dir);

  // 保存主Bias帧
  cv::imwrite((output_dir / "master_bias.tiff").string(), master);

  // 保存质量报告
  std::ofstream report(output_dir / "quality_report.txt");
  report << quality_metrics_.to_string() << std::endl;

  // 保存噪声分析可视化
  if (config_.save_debug_info) {
    save_debug_visualizations(output_dir);
  }

  spdlog::info("结果保存完成");
}

void BiasProcessor::validate_input(const std::vector<cv::Mat> &frames) {
  spdlog::info("验证输入帧");
  if (frames.empty()) {
    throw std::runtime_error("未提供Bias帧");
  }

  const cv::Size first_size = frames[0].size();
  const int first_type = frames[0].type();

  for (const auto &frame : frames) {
    if (frame.size() != first_size || frame.type() != first_type) {
      throw std::runtime_error("Bias帧属性不一致");
    }
  }
  spdlog::info("输入帧验证通过");
}

void BiasProcessor::process_blocks(const std::vector<cv::Mat> &frames,
                                   cv::Mat &master) {
  spdlog::info("开始并行分块处理");
  const int block_size = config_.block_size;
  std::vector<float> pixel_buffer;

#ifdef USE_OPENMP
#pragma omp parallel for collapse(2) private(pixel_buffer)
#endif
  for (int y = 0; y < master.rows; y += block_size) {
    for (int x = 0; x < master.cols; x += block_size) {
      process_block(frames, master, x, y, block_size, pixel_buffer);
    }
  }
  spdlog::info("并行分块处理完成");
}

void BiasProcessor::process_block(const std::vector<cv::Mat> &frames,
                                  cv::Mat &master, int x, int y, int block_size,
                                  std::vector<float> &buffer) {
  const int height = std::min(block_size, master.rows - y);
  const int width = std::min(block_size, master.cols - x);

  for (int by = 0; by < height; ++by) {
    for (int bx = 0; bx < width; ++bx) {
      compute_median_pixel(frames, master, x + bx, y + by, buffer);
    }
  }
}

void BiasProcessor::compute_median_pixel(const std::vector<cv::Mat> &frames,
                                         cv::Mat &master, int x, int y,
                                         std::vector<float> &buffer) {
  buffer.clear();
  const int channels = master.channels();

  for (const auto &frame : frames) {
    if (channels == 1) {
      buffer.push_back(frame.at<float>(y, x));
    } else {
      const auto &vec = frame.at<cv::Vec3f>(y, x);
      buffer.insert(buffer.end(), vec.val, vec.val + channels);
    }
  }

  std::nth_element(buffer.begin(), buffer.begin() + buffer.size() / 2,
                   buffer.end());

  if (channels == 1) {
    master.at<float>(y, x) = buffer[buffer.size() / 2];
  } else {
    auto &vec = master.at<cv::Vec3f>(y, x);
    for (int c = 0; c < channels; ++c) {
      vec[c] = buffer[buffer.size() / 2];
    }
  }
}

QualityMetrics
BiasProcessor::evaluate_quality(const std::vector<cv::Mat> &frames,
                                const cv::Mat &master) {
  spdlog::info("开始质量评估");
  QualityMetrics metrics;

  cv::Mat variance, temporal_std;
  compute_statistics(frames, master, variance, temporal_std);

  // 计算噪声均匀性
  cv::Scalar mean, stddev;
  cv::meanStdDev(variance, mean, stddev);
  metrics.noise_uniformity = 1.0 - (stddev[0] / mean[0]);

  // 计算时间稳定性
  metrics.temporal_stability = compute_temporal_stability(frames);

  // 计算空间均匀性
  metrics.spatial_uniformity = compute_spatial_uniformity(master);

  // 检测异常值
  cv::Mat mask = variance > mean[0] + config_.outlier_threshold * stddev[0];
  metrics.outlier_count = cv::countNonZero(mask);

  // 计算总体评分
  metrics.overall_score =
      (metrics.noise_uniformity * 0.3 + metrics.temporal_stability * 0.3 +
       metrics.spatial_uniformity * 0.4);

  spdlog::info("质量评估完成");
  return metrics;
}

double
BiasProcessor::compute_temporal_stability(const std::vector<cv::Mat> &frames) {
  spdlog::info("计算时间稳定性");
  cv::Mat mean_levels;
  for (const auto &frame : frames) {
    cv::Scalar mean = cv::mean(frame);
    mean_levels.push_back(mean[0]);
  }

  cv::Scalar mean, stddev;
  cv::meanStdDev(mean_levels, mean, stddev);
  return 1.0 - (stddev[0] / mean[0]);
}

double BiasProcessor::compute_spatial_uniformity(const cv::Mat &master) {
  spdlog::info("计算空间均匀性");
  cv::Mat smooth;
  cv::GaussianBlur(master, smooth, cv::Size(21, 21), 3);

  cv::Mat diff;
  cv::absdiff(master, smooth, diff);

  cv::Scalar mean = cv::mean(diff);
  return 1.0 - (mean[0] / cv::mean(master)[0]);
}

void BiasProcessor::compute_statistics(const std::vector<cv::Mat> &frames,
                                       const cv::Mat &master, cv::Mat &variance,
                                       cv::Mat &temporal_std) {
  spdlog::info("计算统计数据");
  variance = cv::Mat::zeros(master.size(), CV_32F);
  temporal_std = cv::Mat::zeros(master.size(), CV_32F);

  temporal_std_ = temporal_std.clone(); // 保存时间标准差

  cv::Mat sum_sq_diff = cv::Mat::zeros(master.size(), CV_32F);

#ifdef USE_OPENMP
#pragma omp parallel for if (config_.use_simd)
#endif
  for (size_t i = 0; i < frames.size(); ++i) {
    cv::Mat diff;
    cv::subtract(frames[i], master, diff);
    cv::multiply(diff, diff, diff);

#ifdef USE_OPENMP
#pragma omp critical
#endif
    {
      cv::add(sum_sq_diff, diff, sum_sq_diff);
    }
  }

  cv::divide(sum_sq_diff, frames.size() - 1, variance);
  cv::sqrt(variance, temporal_std);
  spdlog::info("统计数据计算完成");
}

void BiasProcessor::generate_noise_report(const cv::Mat &variance,
                                          const cv::Mat &temporal_std) {
  spdlog::info("生成噪声报告");
  // 计算噪声直方图
  double max_std = 0;
  cv::minMaxLoc(temporal_std, nullptr, &max_std, nullptr, nullptr);

  cv::Mat hist;
  float range[] = {0, static_cast<float>(max_std)};
  const float *ranges[] = {range};
  int bins[] = {config_.noise_analysis_bins};

  cv::calcHist(&temporal_std, 1, nullptr, cv::Mat(), hist, 1, bins, ranges);

  noise_histogram_ = hist;

  if (config_.enable_correlation) {
    compute_correlation_matrix();
  }
  spdlog::info("噪声报告生成完成");
}

void BiasProcessor::compute_correlation_matrix() {
  spdlog::info("计算相关性矩阵开始");
  const size_t frame_count = frames_.size();
  correlation_matrix_ = cv::Mat::zeros(frame_count, frame_count, CV_32F);

#ifdef USE_OPENMP
#pragma omp parallel for collapse(2) if (config_.use_simd)
#endif
  for (int i = 0; i < frame_count; ++i) {
    for (int j = i; j < frame_count; ++j) {
      cv::Mat diff_i, diff_j;
      cv::subtract(frames_[i], master_, diff_i);
      cv::subtract(frames_[j], master_, diff_j);

      double correlation = 0.0;
      cv::multiply(diff_i, diff_j, diff_i);
      correlation = cv::sum(diff_i)[0] / (diff_i.rows * diff_i.cols);

      correlation_matrix_.at<float>(i, j) = correlation;
      correlation_matrix_.at<float>(j, i) = correlation;
    }
  }

  // 归一化相关性矩阵
  double min_val, max_val;
  cv::minMaxLoc(correlation_matrix_, &min_val, &max_val);
  correlation_matrix_ = (correlation_matrix_ - min_val) / (max_val - min_val);

  spdlog::info("相关性矩阵计算完成");
}

void BiasProcessor::save_debug_visualizations(const fs::path &output_dir) {
  spdlog::info("保存调试可视化信息开始");

  // 保存噪声分布热图
  cv::Mat noise_heatmap;
  cv::applyColorMap(temporal_std_, noise_heatmap, cv::COLORMAP_JET);
  cv::imwrite((output_dir / "noise_heatmap.png").string(), noise_heatmap);

  // 保存噪声直方图
  if (!noise_histogram_.empty()) {
    spdlog::info("生成噪声直方图");
    cv::Mat histogram_image(400, 600, CV_8UC3, cv::Scalar(255, 255, 255));

    // 归一化直方图数据
    cv::Mat normalized_hist;
    cv::normalize(noise_histogram_, normalized_hist, 0, histogram_image.rows,
                  cv::NORM_MINMAX);

    // 绘制直方图
    const int bin_width =
        cvRound(histogram_image.cols / (float)noise_histogram_.rows);
    for (int i = 0; i < noise_histogram_.rows - 1; ++i) {
      cv::line(
          histogram_image,
          cv::Point(bin_width * i, histogram_image.rows -
                                       cvRound(normalized_hist.at<float>(i))),
          cv::Point(bin_width * (i + 1),
                    histogram_image.rows -
                        cvRound(normalized_hist.at<float>(i + 1))),
          cv::Scalar(0, 0, 255), 2);
    }

    // 添加标题和坐标轴
    cv::putText(histogram_image, "噪声分布直方图", cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);

    cv::imwrite((output_dir / "noise_histogram.png").string(), histogram_image);
  }

  // 保存相关性矩阵可视化
  if (!correlation_matrix_.empty()) {
    spdlog::info("生成相关性矩阵可视化");
    cv::Mat correlation_vis;
    cv::applyColorMap(correlation_matrix_ * 255, correlation_vis,
                      cv::COLORMAP_JET);

    // 添加标题
    cv::Mat header(50, correlation_vis.cols, CV_8UC3,
                   cv::Scalar(255, 255, 255));
    cv::putText(header, "帧间相关性矩阵", cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);

    // 合并标题和矩阵图像
    cv::Mat final_vis;
    cv::vconcat(header, correlation_vis, final_vis);

    cv::imwrite((output_dir / "correlation_matrix.png").string(), final_vis);
  }

  spdlog::info("调试可视化信息保存完成");
}

cv::Mat BiasProcessor::convert_output(const cv::Mat &master,
                                      int original_type) {
  spdlog::info("转换输出格式");
  cv::Mat output;
  master.convertTo(output, original_type);
  spdlog::info("输出格式转换完成");
  return output;
}
