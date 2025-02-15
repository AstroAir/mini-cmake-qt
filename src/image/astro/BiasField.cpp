#include <algorithm>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <vector>

namespace fs = std::filesystem;

// 扩展配置结构体
struct BiasConfig {
  int block_size = 128;           // 分块大小
  bool use_simd = true;           // 启用SIMD
  float outlier_threshold = 3.0;  // 异常值阈值(sigma)
  size_t cache_size = 1024;       // 缓存大小(KB)
  bool enable_correlation = true; // 启用相关性分析
  bool save_debug_info = false;   // 保存调试信息
  int noise_analysis_bins = 100;  // 噪声分析直方图bins
  double quality_threshold = 0.9; // 质量评分阈值
};

// 质量评估结果结构体
struct QualityMetrics {
  double noise_uniformity;   // 噪声均匀性
  double temporal_stability; // 时间稳定性
  double spatial_uniformity; // 空间均匀性
  int outlier_count;         // 异常值数量
  double overall_score;      // 总体评分

  std::string to_string() const {
    return fmt::format("质量评估结果:\n"
                       "噪声均匀性: {:.2f}\n"
                       "时间稳定性: {:.2f}\n"
                       "空间均匀性: {:.2f}\n"
                       "异常值数量: {}\n"
                       "总体评分: {:.2f}",
                       noise_uniformity, temporal_stability, spatial_uniformity,
                       outlier_count, overall_score);
  }
};

class BiasProcessor {
public:
  explicit BiasProcessor(const BiasConfig &config) : config_(config) {}

  cv::Mat create_master_bias(const std::vector<cv::Mat> &frames) {
    validate_input(frames);

    const cv::Size frame_size = frames[0].size();
    const int type = frames[0].type();

    // 预分配内存
    cv::Mat master(frame_size, CV_32FC(CV_MAT_CN(type)));

    // 并行分块处理
    process_blocks(frames, master);

    // 质量评估
    quality_metrics_ = evaluate_quality(frames, master);

    if (quality_metrics_.overall_score < config_.quality_threshold) {
      spdlog::warn("主Bias质量低于阈值: {:.2f}",
                   quality_metrics_.overall_score);
    }

    return convert_output(master, type);
  }

  QualityMetrics analyze_noise(const cv::Mat &master,
                               const std::vector<cv::Mat> &frames) {
    cv::Mat variance, temporal_std;
    compute_statistics(frames, master, variance, temporal_std);

    // 生成噪声分析报告
    generate_noise_report(variance, temporal_std);

    return evaluate_quality(frames, master);
  }

  void save_results(const cv::Mat &master, const fs::path &output_dir) {
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
  }

private:
  BiasConfig config_;
  QualityMetrics quality_metrics_;
  cv::Mat noise_histogram_;
  cv::Mat correlation_matrix_;

  void validate_input(const std::vector<cv::Mat> &frames) {
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
  }

  void process_blocks(const std::vector<cv::Mat> &frames, cv::Mat &master) {
    const int block_size = config_.block_size;
    std::vector<float> pixel_buffer;

#pragma omp parallel for collapse(2) private(pixel_buffer)
    for (int y = 0; y < master.rows; y += block_size) {
      for (int x = 0; x < master.cols; x += block_size) {
        process_block(frames, master, x, y, block_size, pixel_buffer);
      }
    }
  }

  void process_block(const std::vector<cv::Mat> &frames, cv::Mat &master, int x,
                     int y, int block_size, std::vector<float> &buffer) {
    const int height = std::min(block_size, master.rows - y);
    const int width = std::min(block_size, master.cols - x);

    for (int by = 0; by < height; ++by) {
      for (int bx = 0; bx < width; ++bx) {
        compute_median_pixel(frames, master, x + bx, y + by, buffer);
      }
    }
  }

  void compute_median_pixel(const std::vector<cv::Mat> &frames, cv::Mat &master,
                            int x, int y, std::vector<float> &buffer) {
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

  QualityMetrics evaluate_quality(const std::vector<cv::Mat> &frames,
                                  const cv::Mat &master) {
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

    return metrics;
  }

  double compute_temporal_stability(const std::vector<cv::Mat> &frames) {
    cv::Mat mean_levels;
    for (const auto &frame : frames) {
      cv::Scalar mean = cv::mean(frame);
      mean_levels.push_back(mean[0]);
    }

    cv::Scalar mean, stddev;
    cv::meanStdDev(mean_levels, mean, stddev);
    return 1.0 - (stddev[0] / mean[0]);
  }

  double compute_spatial_uniformity(const cv::Mat &master) {
    cv::Mat smooth;
    cv::GaussianBlur(master, smooth, cv::Size(21, 21), 3);

    cv::Mat diff;
    cv::absdiff(master, smooth, diff);

    cv::Scalar mean = cv::mean(diff);
    return 1.0 - (mean[0] / cv::mean(master)[0]);
  }

  void compute_statistics(const std::vector<cv::Mat> &frames,
                          const cv::Mat &master, cv::Mat &variance,
                          cv::Mat &temporal_std) {
    variance = cv::Mat::zeros(master.size(), CV_32F);
    temporal_std = cv::Mat::zeros(master.size(), CV_32F);

    cv::Mat sum_sq_diff = cv::Mat::zeros(master.size(), CV_32F);

#pragma omp parallel for if (config_.use_simd)
    for (size_t i = 0; i < frames.size(); ++i) {
      cv::Mat diff;
      cv::subtract(frames[i], master, diff);
      cv::multiply(diff, diff, diff);

#pragma omp critical
      {
        cv::add(sum_sq_diff, diff, sum_sq_diff);
      }
    }

    cv::divide(sum_sq_diff, frames.size() - 1, variance);
    cv::sqrt(variance, temporal_std);
  }

  void generate_noise_report(const cv::Mat &variance,
                             const cv::Mat &temporal_std) {
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
  }

  void compute_correlation_matrix() {
    // 实现帧间相关性分析
    // ...实现代码...
  }

  void save_debug_visualizations(const fs::path &output_dir) {
    // 保存噪声分布热图
    cv::Mat noise_heatmap;
    cv::applyColorMap(temporal_std_, noise_heatmap, cv::COLORMAP_JET);
    cv::imwrite((output_dir / "noise_heatmap.png").string(), noise_heatmap);

    // 保存噪声直方图
    if (!noise_histogram_.empty()) {
      // 绘制直方图
      // ...实现代码...
    }

    // 保存相关性矩阵
    if (!correlation_matrix_.empty()) {
      // 保存相关性可视化
      // ...实现代码...
    }
  }

  cv::Mat convert_output(const cv::Mat &master, int original_type) {
    cv::Mat output;
    master.convertTo(output, original_type);
    return output;
  }

  cv::Mat temporal_std_; // 用于存储时间标准差
};

// 使用示例
int main() {
  try {
    BiasConfig config;
    config.save_debug_info = true;
    config.enable_correlation = true;

    BiasProcessor processor(config);
    std::vector<cv::Mat> bias_frames;

    // 读取Bias帧
    const fs::path bias_dir = "calibration/bias";
    for (const auto &entry : fs::directory_iterator(bias_dir)) {
      cv::Mat frame = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
      if (!frame.empty()) {
        bias_frames.push_back(frame);
      }
    }

    // 处理Bias帧
    cv::Mat master_bias = processor.create_master_bias(bias_frames);

    // 分析噪声
    QualityMetrics metrics = processor.analyze_noise(master_bias, bias_frames);

    // 保存结果
    processor.save_results(master_bias, "result");

    spdlog::info("处理完成\n{}", metrics.to_string());
    return 0;

  } catch (const std::exception &e) {
    spdlog::error("错误: {}", e.what());
    return 1;
  }
}