#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>


namespace fs = std::filesystem;

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

  std::string to_string() const;
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

  explicit DefectPixelMapper(const Config &config);

  // 主要功能接口
  void build_defect_map(const std::vector<cv::Mat> &dark_frames,
                        std::function<void(float)> progress_cb = nullptr);
  cv::Mat correct_image(const cv::Mat &raw_image, double current_temp = 20.0);
  void save_map(const fs::path &path) const;
  void load_map(const fs::path &path);
  QualityMetrics analyze_quality(const cv::Mat &image) const;
  void batch_process(const std::vector<std::string> &input_files,
                     const std::string &output_dir);

  // 温度补偿相关接口
  void add_temperature_data(double temp, double dark_current,
                            double exposure_time);
  void enable_temperature_compensation(bool enable = true);
  TemperatureCompensation get_temperature_compensation() const;

private:
  Config config_;
  cv::Mat defect_map_;
  std::unique_ptr<float[]> cache_;
  std::vector<TemperatureData> temp_history_;
  TemperatureCompensation temp_comp_;

  // 私有辅助方法
  void validate_input(const std::vector<cv::Mat> &frames);
  std::pair<cv::Mat, cv::Mat>
  compute_statistics(const std::vector<cv::Mat> &frames,
                     std::function<void(float)> progress_cb = nullptr);
  void detect_defect_pixel(int x, int y, const cv::Mat &mean,
                           const cv::Mat &stddev);
  float interpolate_pixel(int x, int y, const cv::Mat &img) const;
  float bilinear_interpolate(int x, int y, const cv::Mat &img) const;
  void optimize_defect_map();
  void save_debug_info(const cv::Mat &mean, const cv::Mat &stddev);
  void find_defect_clusters(std::vector<cv::Point> &clusters) const;
  void detect_defects_statistical(const cv::Mat &mean, const cv::Mat &stddev);
  void detect_defects_pattern();
  void save_intermediate_results(const cv::Mat &original,
                                 const cv::Mat &corrected,
                                 const std::string &basename,
                                 const std::string &output_dir);
  void calibrate_temperature_compensation(
      const std::vector<cv::Mat> &dark_frames,
      const std::vector<TemperatureData> &temp_data);
  cv::Mat apply_temperature_compensation(const cv::Mat &image,
                                         double current_temp);
};