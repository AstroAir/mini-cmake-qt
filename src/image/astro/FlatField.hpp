#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// 添加质量评估结构体
struct FlatQualityMetrics {
  double uniformity;    // 均匀度
  double signal_noise;  // 信噪比
  double vignetting;    // 暗角程度
  int hot_pixels;       // 热像素数量
  double overall_score; // 总体评分

  std::string to_string() const;
};

// 平场处理配置
struct FlatConfig {
  bool use_median = true;      // 使用中值合成
  float min_flat_value = 0.1f; // 最小平场值
  bool use_simd = true;        // 启用SIMD
  size_t cache_size = 1024;    // 缓存大小(KB)
  bool enable_debug = false;   // 调试模式

  // 添加新配置选项
  double hot_pixel_threshold = 3.0;    // 热像素检测阈值(sigma)
  int block_size = 32;                 // 分块处理大小
  bool save_debug_info = false;        // 保存调试信息
  std::string output_dir = "debug";    // 调试输出目录
  bool enable_auto_calibration = true; // 启用自动校准

  std::string to_string() const;
};

// 添加blend函数实现
void medianBlend(const std::vector<cv::Mat> &images, cv::Mat &result);
void meanBlend(const std::vector<cv::Mat> &images, cv::Mat &result);

class FlatFieldProcessor {
public:
  FlatFieldProcessor(const FlatConfig &config);

  // 主要处理函数
  cv::Mat process(const std::vector<cv::Mat> &flat_frames,
                  const cv::Mat &master_bias = cv::Mat(),
                  const cv::Mat &master_dark = cv::Mat());

  // 获取质量评估结果
  FlatQualityMetrics getQualityMetrics() const;

  // 生成可视化结果
  cv::Mat visualizeQuality() const;

private:
  FlatConfig config_;
  cv::Mat master_flat_;
  FlatQualityMetrics quality_metrics_;

  // 输入验证
  void validateInputs(const std::vector<cv::Mat> &frames);

  // 创建主平场
  cv::Mat createMasterFlat(const std::vector<cv::Mat> &frames,
                           const cv::Mat &bias, const cv::Mat &dark);

  // 评估质量
  FlatQualityMetrics evaluateQuality(const cv::Mat &flat);

  // 计算暗角
  double calculateVignetting(const cv::Mat &flat);

  // 检测热像素
  cv::Mat detectHotPixels(const cv::Mat &flat);

  // 计算总体评分
  double calculateOverallScore(const FlatQualityMetrics &metrics);

  // 创建均匀度热图
  void createUniformityMap(cv::Mat &map) const;

  // 创建暗角分析图
  void createVignettingMap(cv::Mat &map) const;

  // 计算径向剖面
  void computeRadialProfile(const cv::Mat &flat, cv::Mat &profile) const;

  // 保存调试信息
  void saveDebugInfo();
};

// 平场校正应用函数
cv::Mat apply_flat_correction(const cv::Mat &raw_image,
                              const cv::Mat &master_flat,
                              const cv::Mat &master_bias = cv::Mat(),
                              const cv::Mat &master_dark = cv::Mat(),
                              const FlatConfig &config = FlatConfig());

// 主函数示例
int main();