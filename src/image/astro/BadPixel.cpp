#include <algorithm>
#include <fmt/format.h>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Function declarations
void collectNeighbors(const Mat &roi, vector<uchar> &neighbors);
Mat guidedFilter(const Mat &src, const Mat &guide, int radius, double eps);

// 添加统计信息结构体
struct BadPixelStats {
  int total_pixels;        // 总像素数
  int bad_pixels;          // 坏点数量
  float bad_ratio;         // 坏点比例
  double avg_correction;   // 平均校正量
  vector<Point> hot_spots; // 热点区域

  string to_string() const {
    return fmt::format( // 使用完整的命名空间
        "坏点统计:\n"
        "总像素数: {}\n"
        "坏点数量: {}\n"
        "坏点比例: {:.2f}%\n"
        "平均校正量: {:.2f}\n"
        "热点区域数: {}",
        total_pixels, bad_pixels, bad_ratio * 100, avg_correction,
        hot_spots.size());
  }
};

/**
 * @brief 坏点校正算法配置结构
 *
 * 支持多种检测和校正方法:
 * 1. 检测方法：中值检测、均值检测、高斯检测
 * 2. 校正方法：中值替换、双线性插值、边缘感知
 * 3. 优化选项：多线程、通道关联、动态阈值
 */
struct BadPixelConfig {
  // 添加新的检测方法
  enum DetectionMethod {
    MEDIAN,
    MEAN,
    GAUSSIAN,
    ADAPTIVE,     // 自适应阈值
    TEMPORAL,     // 时序检测
    PATTERN_BASED // 模式检测
  };

  // 添加新的校正方法
  enum CorrectionMethod {
    MEDIAN_REPLACE,
    BILINEAR,
    EDGE_AWARE,
    INPAINTING,    // OpenCV修复
    GUIDED_FILTER, // 导向滤波
    DEEP_PRIOR     // 深度先验
  };

  DetectionMethod detect_method = MEDIAN;
  CorrectionMethod correct_method = MEDIAN_REPLACE;

  int detect_window = 5;                // 检测窗口大小
  int correct_window = 5;               // 修复窗口大小
  float detect_threshold = 35.0f;       // 检测阈值
  bool use_dynamic_threshold = false;   // 动态阈值
  bool use_channel_correlation = false; // 通道关联
  bool multi_thread = true;             // 多线程处理
  bool save_debug = false;              // 调试信息
  string preset_map;                    // 预设坏点图

  // 添加新的配置选项
  bool enable_gpu = false;            // GPU加速
  int temporal_window = 3;            // 时序窗口大小
  float confidence_threshold = 0.95f; // 置信度阈值
  string pattern_file;                // 模式文件
  bool auto_threshold = true;         // 自动阈值
  vector<float> channel_weights;      // 通道权重

  // 缓存设置
  static constexpr size_t CACHE_SIZE = 1024; // 缓存大小
  static constexpr int BLOCK_SIZE = 32;      // 处理块大小
};

class BadPixelCorrector {
public:
  explicit BadPixelCorrector(const BadPixelConfig &cfg = {})
      : config_(cfg), cache_(new float[BadPixelConfig::CACHE_SIZE]) {
    init();
  }

  ~BadPixelCorrector() = default;

  Mat process(const Mat &input) {
    if (input.empty())
      return Mat();

    Mat result = input.clone();
    if (result.channels() > 1 && config_.use_channel_correlation) {
      return process_color(result);
    }
    return process_single_channel(result);
  }

  // 添加统计信息获取
  BadPixelStats getStats() const { return stats_; }

  // 添加质量评估
  float evaluateQuality(const Mat &original, const Mat &corrected) {
    Mat diff;
    absdiff(original, corrected, diff);

    Scalar mean_diff = mean(diff);
    double quality_score = 1.0 - (mean_diff[0] / 255.0);
    return static_cast<float>(quality_score);
  }

  // 添加可视化功能
  Mat visualizeCorrection() const {
    if (last_input_.empty() || last_mask_.empty())
      return Mat();

    Mat visualization;
    int width = last_input_.cols * 2;
    int height = last_input_.rows;

    visualization.create(height, width, CV_8UC3);

    // 绘制原始图像
    Mat roi1 = visualization(Rect(0, 0, last_input_.cols, height));
    cvtColor(last_input_, roi1, COLOR_GRAY2BGR);

    // 绘制校正结果
    Mat roi2 =
        visualization(Rect(last_input_.cols, 0, last_input_.cols, height));
    Mat corrected_color;
    cvtColor(last_output_, corrected_color, COLOR_GRAY2BGR);
    corrected_color.copyTo(roi2);

    // 标记坏点位置
    for (int y = 0; y < last_mask_.rows; ++y) {
      for (int x = 0; x < last_mask_.cols; ++x) {
        if (last_mask_.at<uchar>(y, x)) {
          circle(roi2, Point(x, y), 2, Scalar(0, 0, 255), -1);
        }
      }
    }

    // 添加统计信息
    putText(visualization,
            cv::format("Bad Pixels: %d (%.2f%%)", stats_.bad_pixels,
                       stats_.bad_ratio * 100),
            Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

    return visualization;
  }

private:
  BadPixelConfig config_;
  unique_ptr<float[]> cache_;      // 计算缓存
  Mat preset_mask_;                // 预设坏点掩码
  vector<float> gaussian_weights_; // 高斯权重
  BadPixelStats stats_;
  Mat last_input_;
  Mat last_output_;
  Mat last_mask_;
  Mat pattern_template_; // 添加模板成员变量

  // 添加深度学习模型相关成员
  struct DeepPriorModel {
    int patch_size = 64;
    float learning_rate = 0.001f;
    int max_iterations = 1000;
    vector<Mat> weights;

    void init() {
      // 初始化网络权重
      weights.resize(3);
      for (auto &w : weights) {
        w = Mat::ones(patch_size, patch_size, CV_32F) /
            (patch_size * patch_size);
      }
    }

    Mat forward(const Mat &input) {
      Mat result = input.clone();
      for (const auto &w : weights) {
        filter2D(result, result, -1, w);
        result = max(result, 0); // ReLU
      }
      return result;
    }
  };

  DeepPriorModel deep_prior_model_;
  vector<Mat> temporal_buffer_; // 用于时序检测的帧缓存

  // 初始化
  void init() {
    load_preset_map();
    precompute_gaussian_weights();
  }

  // 加载预设坏点图
  void load_preset_map() {
    if (config_.preset_map.empty())
      return;

    ifstream fin(config_.preset_map, ios::binary);
    if (!fin)
      return;

    int rows, cols;
    fin.read(reinterpret_cast<char *>(&rows), sizeof(int));
    fin.read(reinterpret_cast<char *>(&cols), sizeof(int));

    preset_mask_ = Mat(rows, cols, CV_8UC1);
    vector<char> buffer(rows * cols);
    fin.read(buffer.data(), rows * cols);
    memcpy(preset_mask_.data, buffer.data(), rows * cols);
  }

  // 预计算高斯权重
  void precompute_gaussian_weights() {
    const int size = config_.detect_window;
    const float sigma = size / 3.0f;
    gaussian_weights_.resize(size * size);

    float sum = 0;
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        const float dx = x - size / 2.0f; // 修复浮点除法
        const float dy = y - size / 2.0f; // 修复浮点除法
        const float weight = exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
        gaussian_weights_[y * size + x] = weight;
        sum += weight;
      }
    }

    // 归一化
    for (float &w : gaussian_weights_) {
      w /= sum;
    }
  }

  // 处理单通道图像
  Mat process_single_channel(Mat &image) {
    Mat mask = detect_bad_pixels(image);
    if (!preset_mask_.empty()) {
      bitwise_or(mask, preset_mask_, mask);
    }
    correct_bad_pixels(image, mask);
    return image;
  }

  // 处理彩色图像
  Mat process_color(Mat &image) {
    vector<Mat> channels(3);
    split(image, channels);

    Mat global_mask = Mat::zeros(image.size(), CV_8UC1);

#pragma omp parallel for if (config_.multi_thread)
    for (int i = 0; i < 3; ++i) {
      Mat mask = detect_bad_pixels(channels[i]);
#pragma omp critical
      bitwise_or(global_mask, mask, global_mask);
    }

    if (!preset_mask_.empty()) {
      bitwise_or(global_mask, preset_mask_, global_mask);
    }

#pragma omp parallel for if (config_.multi_thread)
    for (int i = 0; i < 3; ++i) {
      correct_bad_pixels(channels[i], global_mask);
    }

    merge(channels, image);
    return image;
  }

  // 坏点检测（已优化）
  Mat detect_bad_pixels(const Mat &image) {
    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    const int radius = config_.detect_window / 2;

    // 使用OpenCV的integral函数
    Mat integral_img;
    cv::integral(image, integral_img, CV_32S);

    parallel_for_(Range(radius, image.rows - radius), [&](const Range &range) {
      for (int y = range.start; y < range.end; ++y) {
        for (int x = radius; x < image.cols - radius; ++x) {
          float ref_value = 0;
          const float center = image.at<uchar>(y, x);

          switch (config_.detect_method) {
          case BadPixelConfig::MEDIAN: {
            // 实现中值检测
            vector<uchar> neighbors;
            Rect roi(x - radius, y - radius, 2 * radius + 1, 2 * radius + 1);
            collectNeighbors(image(roi), neighbors);
            nth_element(neighbors.begin(),
                        neighbors.begin() + neighbors.size() / 2,
                        neighbors.end());
            ref_value = neighbors[neighbors.size() / 2];
            break;
          }
          case BadPixelConfig::MEAN: {
            // 实现均值检测
            Rect roi(x - radius, y - radius, 2 * radius + 1, 2 * radius + 1);
            Scalar mean_val = mean(image(roi));
            ref_value = mean_val[0];
            break;
          }
          case BadPixelConfig::GAUSSIAN: {
            // 实现高斯检测
            ref_value = calculateGaussianReference(image, x, y, radius);
            break;
          }
          case BadPixelConfig::ADAPTIVE: {
            ref_value =
                calculate_adaptive_threshold(integral_img, x, y, radius);
            break;
          }
          case BadPixelConfig::TEMPORAL: {
            // 时序检测 - 需要多帧数据支持
            ref_value = center; // 临时实现
            break;
          }
          case BadPixelConfig::PATTERN_BASED: {
            ref_value = detect_pattern_deviation(image, x, y, radius);
            break;
          }
          }

          if (abs(center - ref_value) > get_threshold(image, x, y)) {
            mask.at<uchar>(y, x) = 255;
          }
        }
      }
    });

    return mask;
  }

  // 计算动态阈值
  float calculate_dynamic_threshold(const Mat &region) {
    Scalar mean, stddev;
    meanStdDev(region, mean, stddev);
    return mean[0] + 2.5f * stddev[0];
  }

  // 添加获取阈值函数
  float get_threshold(const Mat &image, int x, int y) {
    if (config_.use_dynamic_threshold) {
      Rect roi(max(0, x - 2), max(0, y - 2), min(5, image.cols - x + 2),
               min(5, image.rows - y + 2));
      return calculate_dynamic_threshold(image(roi));
    }
    return config_.detect_threshold;
  }

  // 坏点校正（已优化）
  void correct_bad_pixels(Mat &image, const Mat &mask) {
    Mat expanded_mask;
    const int radius = config_.correct_window / 2;
    dilate(mask, expanded_mask, Mat(), Point(-1, -1), radius);

    if (config_.correct_method == BadPixelConfig::INPAINTING) {
      inpaint(image, expanded_mask, image, 3, INPAINT_TELEA);
      return;
    }

    auto correct_block = [&](const Range &range) {
      vector<uchar> neighbors;
      neighbors.reserve(config_.correct_window * config_.correct_window);

      for (int y = range.start; y < range.end; ++y) {
        const uchar *mask_row = expanded_mask.ptr<uchar>(y);
        uchar *img_row = image.ptr<uchar>(y);

        for (int x = 0; x < image.cols; ++x) {
          if (!mask_row[x])
            continue;

          neighbors.clear();
          float weights_sum = 0;
          float value_sum = 0;

          // 收集邻域像素
          for (int dy = -radius; dy <= radius; ++dy) {
            const int ny = y + dy;
            if (ny < 0 || ny >= image.rows)
              continue;

            for (int dx = -radius; dx <= radius; ++dx) {
              const int nx = x + dx;
              if (nx < 0 || nx >= image.cols)
                continue;

              if (!expanded_mask.at<uchar>(ny, nx))
                neighbors.push_back(image.at<uchar>(ny, nx));
            }
          }

          if (neighbors.empty())
            continue;

          switch (config_.correct_method) {
          case BadPixelConfig::MEDIAN_REPLACE: {
            nth_element(neighbors.begin(),
                        neighbors.begin() + neighbors.size() / 2,
                        neighbors.end());
            img_row[x] = neighbors[neighbors.size() / 2];
            break;
          }
          case BadPixelConfig::BILINEAR: {
            float sum = 0;
            float weight_sum = 0;
            for (int dy = -1; dy <= 1; ++dy) {
              for (int dx = -1; dx <= 1; ++dx) {
                const int ny = y + dy;
                const int nx = x + dx;
                if (ny < 0 || ny >= image.rows || nx < 0 || nx >= image.cols)
                  continue;
                if (!expanded_mask.at<uchar>(ny, nx)) {
                  const float w = 1.0f / (abs(dy) + abs(dx) + 1);
                  sum += w * image.at<uchar>(ny, nx);
                  weight_sum += w;
                }
              }
            }
            if (weight_sum > 0) {
              img_row[x] = sum / weight_sum;
            }
            break;
          }
          case BadPixelConfig::EDGE_AWARE: {
            const int gy =
                abs(image.at<uchar>(y + 1, x) - image.at<uchar>(y - 1, x));
            const int gx =
                abs(image.at<uchar>(y, x + 1) - image.at<uchar>(y, x - 1));
            if (gx > gy) {
              const uchar left = image.at<uchar>(y, x - 1);
              const uchar right = image.at<uchar>(y, x + 1);
              img_row[x] = (left + right) / 2;
            } else {
              const uchar top = image.at<uchar>(y - 1, x);
              const uchar bottom = image.at<uchar>(y + 1, x);
              img_row[x] = (top + bottom) / 2;
            }
            break;
          }
          case BadPixelConfig::GUIDED_FILTER: {
            image = guidedFilter(image, image, 3, 0.1);
            break;
          }
          case BadPixelConfig::INPAINTING: {
            // Already handled before the switch
            break;
          }
          case BadPixelConfig::DEEP_PRIOR:
            correct_with_deep_prior(image, mask);
            break;
          }
        }
      }
    };

    if (config_.multi_thread) {
      const int strip_height = 32;

#pragma omp parallel for
      for (int y = 0; y < image.rows; y += strip_height) {
        Range range(y, std::min(y + strip_height, image.rows));
        correct_block(range);
      }
    } else {
      correct_block(Range(0, image.rows));
    }

    // 更新统计信息
    stats_.bad_pixels = countNonZero(mask);
    stats_.total_pixels = image.total();
    stats_.bad_ratio =
        static_cast<float>(stats_.bad_pixels) / stats_.total_pixels;
  }

  // 添加辅助函数
  float calculateGaussianReference(const Mat &image, int x, int y, int radius) {
    float sum = 0;
    float weight_sum = 0;

    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) { // 修复for循环
        if (dx == 0 && dy == 0)
          continue;

        int nx = x + dx;
        int ny = y + dy;
        if (nx < 0 || nx >= image.cols || ny < 0 || ny >= image.rows)
          continue;

        float weight =
            gaussian_weights_[(dy + radius) * (2 * radius + 1) + (dx + radius)];
        float value = image.at<uchar>(ny, nx);

        sum += weight * value;
        weight_sum += weight;
      }
    }

    return sum / weight_sum;
  }

  // 实现时序检测
  Mat detect_temporal_anomalies(const Mat &current_frame) {
    if (temporal_buffer_.empty()) {
      temporal_buffer_.push_back(current_frame.clone());
      return Mat::zeros(current_frame.size(), CV_8UC1);
    }

    Mat temporal_mask = Mat::zeros(current_frame.size(), CV_8UC1);
    Mat mean_frame = Mat::zeros(current_frame.size(), CV_32F);
    Mat std_frame = Mat::zeros(current_frame.size(), CV_32F);

    // 计算时序统计
    for (const auto &frame : temporal_buffer_) {
      Mat float_frame;
      frame.convertTo(float_frame, CV_32F);
      accumulate(float_frame, mean_frame);
    }

    mean_frame /= temporal_buffer_.size();

    // 计算标准差
    for (const auto &frame : temporal_buffer_) {
      Mat float_frame, diff;
      frame.convertTo(float_frame, CV_32F);
      subtract(float_frame, mean_frame, diff);
      multiply(diff, diff, diff);
      accumulate(diff, std_frame);
    }

    sqrt(std_frame / temporal_buffer_.size(), std_frame);

    // 检测异常
    Mat current_float;
    current_frame.convertTo(current_float, CV_32F);
    Mat diff = abs(current_float - mean_frame);

    // 应用3σ准则
    Mat threshold = 3 * std_frame;
    compare(diff, threshold, temporal_mask, CMP_GT);

    // 更新缓存
    temporal_buffer_.push_back(current_frame.clone());
    if (temporal_buffer_.size() > config_.temporal_window) {
      temporal_buffer_.erase(temporal_buffer_.begin());
    }

    return temporal_mask;
  }

  // 实现基于深度先验的校正
  void correct_with_deep_prior(Mat &image, const Mat &mask) {
    if (!deep_prior_model_.weights[0].data) {
      deep_prior_model_.init();
    }

    const int patch_size = deep_prior_model_.patch_size;
    const int padding = patch_size / 2;

    Mat padded_image;
    copyMakeBorder(image, padded_image, padding, padding, padding, padding,
                   BORDER_REFLECT);

    // 分块处理
#pragma omp parallel for if (config_.multi_thread)
    for (int y = 0; y < image.rows; y += patch_size / 2) {
      for (int x = 0; x < image.cols; x += patch_size / 2) {
        // 提取patch
        Rect patch_rect(x, y, min(patch_size, image.cols - x),
                        min(patch_size, image.rows - y));
        Mat patch = padded_image(Rect(x + padding, y + padding,
                                      patch_rect.width, patch_rect.height));

        // 检查是否包含坏点
        Mat patch_mask = mask(patch_rect);
        if (countNonZero(patch_mask) == 0)
          continue;

        // 应用深度先验
        Mat restored = deep_prior_model_.forward(patch);
        restored.copyTo(image(patch_rect), patch_mask);
      }
    }
  }

  // 优化自适应阈值计算
  float calculate_adaptive_threshold(const Mat &integral, int x, int y,
                                     int radius) {
    const int x1 = max(0, x - radius);
    const int y1 = max(0, y - radius);
    const int x2 = min(integral.cols - 1, x + radius);
    const int y2 = min(integral.rows - 1, y + radius);

    const int area = (x2 - x1) * (y2 - y1);
    if (area <= 0)
      return config_.detect_threshold;

    // 使用积分图快速计算区域和与方差
    double sum = integral.at<double>(y2, x2) - integral.at<double>(y2, x1) -
                 integral.at<double>(y1, x2) + integral.at<double>(y1, x1);

    double mean = sum / area;
    double variance = 0;

    // 计算局部方差
    Mat roi = integral(Rect(x1, y1, x2 - x1, y2 - y1));
    Mat local_variance;
    multiply(roi, roi, local_variance);
    double sq_sum = cv::sum(local_variance)[0];
    variance = (sq_sum / area) - (mean * mean);

    // 根据局部统计特征调整阈值
    float k = config_.confidence_threshold;
    return mean + k * sqrt(max(variance, 0.0));
  }

  // 实现模式检测
  float detect_pattern_deviation(const Mat &image, int x, int y, int radius) {
    if (pattern_template_.empty()) {
      // 如果没有预定义模式，使用局部模式
      Rect roi(max(0, x - radius), max(0, y - radius),
               min(2 * radius + 1, image.cols - x + radius),
               min(2 * radius + 1, image.rows - y + radius));
      pattern_template_ = image(roi).clone();
    }

    Mat patch = image(Rect(max(0, x - radius), max(0, y - radius),
                           pattern_template_.cols, pattern_template_.rows));

    Mat result;
    matchTemplate(patch, pattern_template_, result, TM_SQDIFF_NORMED);

    double min_val, max_val;
    minMaxLoc(result, &min_val, &max_val);

    return min_val;
  }
};

// 添加collectNeighbors函数实现
void collectNeighbors(const Mat &roi, vector<uchar> &neighbors) {
  neighbors.clear();
  neighbors.reserve(roi.rows * roi.cols);
  for (int y = 0; y < roi.rows; ++y) {
    const uchar *row = roi.ptr<uchar>(y);
    for (int x = 0; x < roi.cols; ++x) {
      neighbors.push_back(row[x]);
    }
  }
}

// 添加guidedFilter函数实现
Mat guidedFilter(const Mat &src, const Mat &guide, int radius, double eps) {
  Mat dst;
  cv::ximgproc::guidedFilter(guide, src, dst, radius, eps, -1);
  return dst;
}

// 使用示例
int main() {
  BadPixelConfig config;
  config.detect_method = BadPixelConfig::GAUSSIAN;
  config.correct_method = BadPixelConfig::EDGE_AWARE;
  config.detect_window = 5;
  config.correct_window = 3;
  config.use_channel_correlation = true;
  config.preset_map = "bad_pixels.bin";

  Mat image = imread("input.jpg");
  BadPixelCorrector corrector(config);
  Mat result = corrector.process(image);

  imshow("Original", image);
  imshow("Corrected", result);
  waitKey(0);

  return 0;
}