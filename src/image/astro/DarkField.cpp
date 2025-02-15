#include <algorithm>
#include <atomic>
#include <cmath>
#include <execution>
#include <filesystem>
#include <fmt/format.h>
#include <fstream> // 添加 fstream
#include <memory>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <vector>

// 添加 erfinv 实现
namespace {
double erfinv(double x) {
  // 使用数值近似实现 erfinv
  double tt1, tt2, lnx, sgn;
  sgn = (x < 0) ? -1.0 : 1.0;
  x = (1 - x) * (1 + x);
  lnx = log(x);
  tt1 = 2 / (M_PI * 0.147) + 0.5 * lnx;
  tt2 = 1 / (0.147) * lnx;
  return sgn * sqrt(-tt1 + sqrt(tt1 * tt1 - tt2));
}

struct TemperatureData {
  double temperature;   // 温度值(摄氏度)
  double dark_current;  // 暗电流值
  double exposure_time; // 曝光时间(秒)
};

struct TemperatureCompensation {
  double baseline_temp;         // 基准温度
  double temp_coefficient;      // 温度系数
  double dark_current_baseline; // 基准暗电流
};
} // namespace

/**
 * @brief 天文相机暗场校正程序
 *
 * 算法原理：
 * 1. 统计分析：计算暗场帧的均值和标准差
 * 2. 缺陷检测：使用阈值法检测热像素和死像素
 * 3. 像素校正：采用中值滤波和双线性插值进行修复
 *
 * 优化策略：
 * 1. SIMD加速计算
 * 2. 多级缓存设计
 * 3. 并行计算优化
 * 4. 内存访问优化
 */

namespace fs = std::filesystem;

struct DefectDetectionConfig {
  enum class Algorithm {
    THRESHOLD,   // 阈值法
    STATISTICAL, // 统计法
    PATTERN,     // 模式识别
    HYBRID       // 混合方法
  };

  Algorithm method = Algorithm::HYBRID;
  int window_size = 5;         // 检测窗口大小
  float confidence = 0.95f;    // 置信度
  int min_cluster = 3;         // 最小聚类大小
  bool detect_clusters = true; // 是否检测坏点簇
};

struct QualityMetrics {
  float snr;                       // 信噪比
  float uniformity;                // 均匀性
  float defect_density;            // 缺陷密度
  std::vector<cv::Point> clusters; // 缺陷簇位置

  std::string to_string() const {
    return fmt::format(
        "SNR: {:.2f}, Uniformity: {:.2f}%, Defect Density: {:.2f}%", snr,
        uniformity * 100, defect_density * 100);
  }
};

class DefectPixelMapper {
public:
  struct Config {
    int warm_pixel_threshold = 5;  // 热像素阈值
    float dead_pixel_value = 0.0f; // 死像素阈值
    int cache_size = 1024;         // 缓存大小(KB)
    bool use_simd = true;          // 启用SIMD
    bool enable_debug = false;     // 调试模式
    DefectDetectionConfig detection;
    std::string log_file = "darkfield.log";
    int batch_size = 100;
    bool save_intermediates = false;
    bool enable_temp_compensation = false; // 是否启用温度补偿
    double temp_coefficient = 0.1;         // 默认温度系数(每度变化率)
    double baseline_temp = 20.0;           // 基准温度(摄氏度)
  };

  explicit DefectPixelMapper(const Config &config)
      : config_(config),
        cache_(new float[config.cache_size * 1024 / sizeof(float)]) {}

  void build_defect_map(const std::vector<cv::Mat> &dark_frames,
                        std::function<void(float)> progress_cb = nullptr) {
    CV_Assert(!dark_frames.empty());
    validate_input(dark_frames);

    const auto [mean, stddev] = compute_statistics(dark_frames, progress_cb);
    defect_map_.create(mean.size(), CV_8UC1);
    defect_map_.setTo(0);

    // 并行像素处理
    std::atomic<int> progress(0);
    const int total_pixels = mean.rows * mean.cols;

#pragma omp parallel for schedule(dynamic) if (config_.use_simd)
    for (int i = 0; i < total_pixels; ++i) {
      const int y = i / mean.cols;
      const int x = i % mean.cols;

      detect_defect_pixel(x, y, mean, stddev);

      if (progress_cb && (++progress % 10000 == 0)) {
        progress_cb(progress / static_cast<float>(total_pixels));
      }
    }

    optimize_defect_map();

    if (config_.enable_debug) {
      save_debug_info(mean, stddev);
    }
  }

  cv::Mat correct_image(const cv::Mat &raw_image, double current_temp = 20.0) {
    CV_Assert(raw_image.size() == defect_map_.size());

    // 首先进行温度补偿
    cv::Mat corrected =
        config_.enable_temp_compensation
            ? apply_temperature_compensation(raw_image, current_temp)
            : raw_image.clone();

// 继续进行坏点校正
#pragma omp parallel for collapse(2) if (config_.use_simd)
    for (int y = 0; y < corrected.rows; ++y) {
      for (int x = 0; x < corrected.cols; ++x) {
        if (defect_map_.at<uint8_t>(y, x) != 0) {
          corrected.at<float>(y, x) = interpolate_pixel(x, y, corrected);
        }
      }
    }

    return corrected;
  }

  void save_map(const fs::path &path) const {
    cv::FileStorage fs(path.string(), cv::FileStorage::WRITE);
    fs << "defect_map" << defect_map_ << "config" << "{"
       << "warm_threshold" << config_.warm_pixel_threshold << "dead_value"
       << config_.dead_pixel_value << "}";
  }

  void load_map(const fs::path &path) {
    cv::FileStorage fs(path.string(), cv::FileStorage::READ);
    fs["defect_map"] >> defect_map_;

    cv::FileNode config = fs["config"];
    if (!config.empty()) {
      config_.warm_pixel_threshold = (int)config["warm_threshold"];
      config_.dead_pixel_value = (float)config["dead_value"];
    }
  }

  QualityMetrics analyze_quality(const cv::Mat &image) const {
    QualityMetrics metrics;

    // 计算信噪比
    cv::Mat mean, stddev;
    cv::meanStdDev(image, mean, stddev);
    metrics.snr = mean.at<double>(0) / stddev.at<double>(0);

    // 计算均匀性
    cv::Mat normalized;
    cv::normalize(image, normalized, 0, 1, cv::NORM_MINMAX);
    cv::meanStdDev(normalized, mean, stddev);
    metrics.uniformity = 1.0f - stddev.at<double>(0);

    // 计算缺陷密度
    int defect_count = cv::countNonZero(defect_map_);
    metrics.defect_density = static_cast<float>(defect_count) /
                             (defect_map_.rows * defect_map_.cols);

    // 检测缺陷簇
    if (config_.detection.detect_clusters) {
      find_defect_clusters(metrics.clusters);
    }

    return metrics;
  }

  void batch_process(const std::vector<std::string> &input_files,
                     const std::string &output_dir) {
    fs::create_directories(output_dir);

    spdlog::info("Starting batch processing of {} files", input_files.size());
    std::atomic<int> processed{0};

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < input_files.size(); ++i) {
      try {
        const auto &input_file = input_files[i];
        cv::Mat img = cv::imread(input_file, cv::IMREAD_UNCHANGED);
        if (img.empty())
          continue;

        cv::Mat corrected = correct_image(img);

        std::string basename = fs::path(input_file).stem().string();
        std::string output_path =
            (fs::path(output_dir) / (basename + "_corrected.tiff")).string();
        cv::imwrite(output_path, corrected);

        if (config_.save_intermediates) {
          save_intermediate_results(img, corrected, basename, output_dir);
        }

        ++processed;
        spdlog::info("Processed {}/{}: {}", processed.load(),
                     input_files.size(), basename);

      } catch (const std::exception &e) {
        spdlog::error("Error processing file {}: {}", input_files[i], e.what());
      }
    }
  }

  // 添加温度补偿相关的公共接口
  void add_temperature_data(double temp, double dark_current,
                            double exposure_time) {
    temp_history_.push_back({temp, dark_current, exposure_time});
  }

  void enable_temperature_compensation(bool enable = true) {
    config_.enable_temp_compensation = enable;
  }

  TemperatureCompensation get_temperature_compensation() const {
    return temp_comp_;
  }

private:
  Config config_;
  cv::Mat defect_map_;
  std::unique_ptr<float[]> cache_;
  std::vector<TemperatureData> temp_history_;
  TemperatureCompensation temp_comp_;

  void validate_input(const std::vector<cv::Mat> &frames) {
    if (frames.empty()) {
      throw std::runtime_error("No dark frames provided");
    }

    const cv::Size first_size = frames[0].size();
    const int first_type = frames[0].type();

    for (const auto &frame : frames) {
      if (frame.size() != first_size || frame.type() != first_type) {
        throw std::runtime_error("Inconsistent frame dimensions or type");
      }
    }
  }

  std::pair<cv::Mat, cv::Mat>
  compute_statistics(const std::vector<cv::Mat> &frames,
                     std::function<void(float)> progress_cb = nullptr) {
    cv::Mat mean, stddev;
    cv::Mat accu(frames[0].size(), CV_32FC1, 0.0f);
    cv::Mat sq_accu(frames[0].size(), CV_32FC1, 0.0f);

    std::atomic<int> progress(0);
    const int total_frames = frames.size();

    // 并行累加
    std::for_each(std::execution::par, frames.begin(), frames.end(),
                  [&](const cv::Mat &frame) {
                    cv::Mat f32_frame;
                    frame.convertTo(f32_frame, CV_32F);

                    cv::add(accu, f32_frame, accu);
                    cv::accumulateSquare(f32_frame, sq_accu);

                    if (progress_cb) {
                      progress_cb((++progress) /
                                  static_cast<float>(total_frames));
                    }
                  });

    const float N = static_cast<float>(frames.size());
    mean = accu / N;
    cv::sqrt((sq_accu / N) - (mean.mul(mean)), stddev);

    return {mean, stddev};
  }

  void detect_defect_pixel(int x, int y, const cv::Mat &mean,
                           const cv::Mat &stddev) {
    const float pixel_mean = mean.at<float>(y, x);
    const float pixel_stddev = stddev.at<float>(y, x);

    // 热像素检测
    if (pixel_mean > config_.warm_pixel_threshold * pixel_stddev) {
      defect_map_.at<uint8_t>(y, x) |= 0x01;
    }

    // 死像素检测
    if (pixel_mean <= config_.dead_pixel_value) {
      defect_map_.at<uint8_t>(y, x) |= 0x02;
    }
  }

  float interpolate_pixel(int x, int y, const cv::Mat &img) const {
    const int radius = 1;
    std::vector<float> &values =
        *reinterpret_cast<std::vector<float> *>(cache_.get());
    values.clear();

    // 8邻域采样
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        if (dx == 0 && dy == 0)
          continue;

        const int nx = x + dx;
        const int ny = y + dy;

        if (nx >= 0 && nx < img.cols && ny >= 0 && ny < img.rows) {
          if (defect_map_.at<uint8_t>(ny, nx) == 0) {
            values.push_back(img.at<float>(ny, nx));
          }
        }
      }
    }

    // 使用中值或双线性插值
    if (values.size() >= 5) {
      std::nth_element(values.begin(), values.begin() + values.size() / 2,
                       values.end());
      return values[values.size() / 2];
    } else {
      return bilinear_interpolate(x, y, img);
    }
  }

  float bilinear_interpolate(int x, int y, const cv::Mat &img) const {
    const int x1 = std::clamp(x - 1, 0, img.cols - 2);
    const int y1 = std::clamp(y - 1, 0, img.rows - 2);
    const float a = static_cast<float>(x - x1);
    const float b = static_cast<float>(y - y1);

    return img.at<float>(y1, x1) * (1 - a) * (1 - b) +
           img.at<float>(y1, x1 + 1) * a * (1 - b) +
           img.at<float>(y1 + 1, x1) * (1 - a) * b +
           img.at<float>(y1 + 1, x1 + 1) * a * b;
  }

  void optimize_defect_map() {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
    cv::morphologyEx(defect_map_, defect_map_, cv::MORPH_CLOSE, kernel);
  }

  void save_debug_info(const cv::Mat &mean, const cv::Mat &stddev) {
    cv::imwrite("debug_mean.tiff", mean);
    cv::imwrite("debug_stddev.tiff", stddev);
    cv::imwrite("debug_defect_map.png", defect_map_);
  }

  void find_defect_clusters(std::vector<cv::Point> &clusters) const {
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(
        defect_map_, labels, stats, centroids, 8, CV_32S);

    for (int i = 1; i < num_labels; ++i) {
      if (stats.at<int>(i, cv::CC_STAT_AREA) >= config_.detection.min_cluster) {
        clusters.emplace_back(centroids.at<double>(i, 0),
                              centroids.at<double>(i, 1));
      }
    }
  }

  void detect_defects_statistical(const cv::Mat &mean, const cv::Mat &stddev) {
    cv::Mat z_score;
    cv::subtract(mean, cv::mean(mean)[0], z_score);
    cv::divide(z_score, stddev, z_score);

    double conf_level = std::sqrt(2.0) * erfinv(config_.detection.confidence);
    cv::Mat outliers = cv::abs(z_score) > conf_level;
    outliers.copyTo(defect_map_);
  }

  void detect_defects_pattern() {
    cv::Mat gradient_x, gradient_y;
    cv::Sobel(defect_map_, gradient_x, CV_32F, 1, 0);
    cv::Sobel(defect_map_, gradient_y, CV_32F, 0, 1);

    cv::Mat gradient_mag;
    cv::magnitude(gradient_x, gradient_y, gradient_mag);

    double threshold = cv::mean(gradient_mag)[0] * 2.0;
    cv::threshold(gradient_mag, defect_map_, threshold, 255, cv::THRESH_BINARY);
  }

  void save_intermediate_results(const cv::Mat &original,
                                 const cv::Mat &corrected,
                                 const std::string &basename,
                                 const std::string &output_dir) {
    fs::path base_path = fs::path(output_dir) / basename;

    // 保存差异图
    cv::Mat diff;
    cv::absdiff(original, corrected, diff);
    cv::imwrite((base_path.string() + "_diff.tiff"), diff);

    // 保存热力图
    cv::Mat heatmap;
    cv::normalize(diff, heatmap, 0, 255, cv::NORM_MINMAX);
    cv::applyColorMap(heatmap, heatmap, cv::COLORMAP_JET);
    cv::imwrite((base_path.string() + "_heatmap.png"), heatmap);

    // 保存质量指标
    QualityMetrics metrics = analyze_quality(corrected);
    std::ofstream report(base_path.string() + "_report.txt");
    report << metrics.to_string() << std::endl;
  }

  // 添加温度补偿相关方法
  void calibrate_temperature_compensation(
      const std::vector<cv::Mat> &dark_frames,
      const std::vector<TemperatureData> &temp_data) {
    if (temp_data.empty())
      return;

    // 计算温度系数
    std::vector<double> coefficients;
    for (size_t i = 1; i < temp_data.size(); ++i) {
      const auto &prev = temp_data[i - 1];
      const auto &curr = temp_data[i];

      double temp_diff = curr.temperature - prev.temperature;
      double current_diff = curr.dark_current - prev.dark_current;

      if (std::abs(temp_diff) > 0.1) { // 忽略温度变化太小的样本
        coefficients.push_back(current_diff / temp_diff);
      }
    }

    // 使用中位数作为最终温度系数
    if (!coefficients.empty()) {
      std::nth_element(coefficients.begin(),
                       coefficients.begin() + coefficients.size() / 2,
                       coefficients.end());
      temp_comp_.temp_coefficient = coefficients[coefficients.size() / 2];
    }

    // 设置基准值
    temp_comp_.baseline_temp = temp_data[0].temperature;
    temp_comp_.dark_current_baseline = temp_data[0].dark_current;

    spdlog::info("Temperature compensation calibrated: coefficient={:.3f}, "
                 "baseline_temp={:.1f}°C",
                 temp_comp_.temp_coefficient, temp_comp_.baseline_temp);
  }

  cv::Mat apply_temperature_compensation(const cv::Mat &image,
                                         double current_temp) {
    if (!config_.enable_temp_compensation)
      return image;

    cv::Mat compensated = image.clone();

    // 计算温度补偿系数
    double temp_diff = current_temp - temp_comp_.baseline_temp;
    double compensation_factor = 1.0 + temp_comp_.temp_coefficient * temp_diff;

    // 应用补偿
    compensated *= compensation_factor;

    spdlog::debug(
        "Applied temperature compensation: temp={:.1f}°C, factor={:.3f}",
        current_temp, compensation_factor);

    return compensated;
  }
};

// 示例用法
int main() {
  try {
    // 配置日志
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::debug);

    // 设置配置
    DefectPixelMapper::Config config;
    config.detection.method = DefectDetectionConfig::Algorithm::HYBRID;
    config.save_intermediates = true;

    // 创建映射器
    DefectPixelMapper mapper(config);

    // 批处理模式
    std::vector<std::string> input_files;
    for (const auto &entry : fs::directory_iterator("input/")) {
      if (entry.path().extension() == ".tiff") {
        input_files.push_back(entry.path().string());
      }
    }

    mapper.batch_process(input_files, "output/");

    spdlog::info("Processing completed successfully");
    return 0;

  } catch (const std::exception &e) {
    spdlog::critical("Fatal error: {}", e.what());
    return 1;
  }
}