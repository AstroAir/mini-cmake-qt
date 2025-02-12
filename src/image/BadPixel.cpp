#include <algorithm>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

/**
 * @brief 坏点校正算法配置结构
 *
 * 支持多种检测和校正方法:
 * 1. 检测方法：中值检测、均值检测、高斯检测
 * 2. 校正方法：中值替换、双线性插值、边缘感知
 * 3. 优化选项：多线程、通道关联、动态阈值
 */
struct BadPixelConfig {
  enum DetectionMethod { MEDIAN, MEAN, GAUSSIAN };
  enum CorrectionMethod { MEDIAN_REPLACE, BILINEAR, EDGE_AWARE };

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

private:
  BadPixelConfig config_;
  unique_ptr<float[]> cache_;      // 计算缓存
  Mat preset_mask_;                // 预设坏点掩码
  vector<float> gaussian_weights_; // 高斯权重

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

    auto process_block = [&](const Range &range) {
      for (int y = range.start; y < range.end; ++y) {
        for (int x = radius; x < image.cols - radius; ++x) {
          if (y < radius || y >= image.rows - radius)
            continue;

          const float center = image.at<uchar>(y, x);
          float ref_value = 0;

          // 根据检测方法计算参考值
          switch (config_.detect_method) {
          case BadPixelConfig::MEDIAN: {
            vector<uchar> &block = reinterpret_cast<vector<uchar> &>(cache_);
            block.clear();
            for (int dy = -radius; dy <= radius; ++dy) {
              for (int dx = -radius; dx <= radius; ++dx) {
                block.push_back(image.at<uchar>(y + dy, x + dx));
              }
            }
            nth_element(block.begin(), block.begin() + block.size() / 2,
                        block.end());
            ref_value = block[block.size() / 2];
            break;
          }
          case BadPixelConfig::MEAN: {
            float sum = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
              for (int dx = -radius; dx <= radius; ++dx) {
                sum += image.at<uchar>(y + dy, x + dx);
              }
            }
            ref_value = sum / (config_.detect_window * config_.detect_window);
            break;
          }
          case BadPixelConfig::GAUSSIAN: {
            float sum = 0;
            int idx = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
              for (int dx = -radius; dx <= radius; ++dx) {
                sum +=
                    image.at<uchar>(y + dy, x + dx) * gaussian_weights_[idx++];
              }
            }
            ref_value = sum;
            break;
          }
          }

          float threshold = config_.detect_threshold;
          if (config_.use_dynamic_threshold) {
            threshold = calculate_dynamic_threshold(
                image(Rect(x - radius, y - radius, config_.detect_window,
                           config_.detect_window)));
          }

          if (abs(center - ref_value) > threshold) {
            mask.at<uchar>(y, x) = 255;
          }
        }
      }
    };

    if (config_.multi_thread) {
      const int strip_height = 32;

#pragma omp parallel for
      for (int y = 0; y < image.rows; y += strip_height) {
        Range range(y, std::min(y + strip_height, image.rows));
        process_block(range);
      }
    } else {
      process_block(Range(0, image.rows));
    }

    return mask;
  }

  // 计算动态阈值
  float calculate_dynamic_threshold(const Mat &region) {
    Scalar mean, stddev;
    meanStdDev(region, mean, stddev);
    return mean[0] + 2.5f * stddev[0];
  }

  // 坏点校正（已优化）
  void correct_bad_pixels(Mat &image, const Mat &mask) {
    Mat expanded_mask;
    const int radius = config_.correct_window / 2;
    dilate(mask, expanded_mask, Mat(), Point(-1, -1), radius);

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
  }
};

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