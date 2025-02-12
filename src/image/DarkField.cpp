#include <algorithm>
#include <atomic>
#include <execution>
#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>


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

class DefectPixelMapper {
public:
  struct Config {
    int warm_pixel_threshold = 5;  // 热像素阈值
    float dead_pixel_value = 0.0f; // 死像素阈值
    int cache_size = 1024;         // 缓存大小(KB)
    bool use_simd = true;          // 启用SIMD
    bool enable_debug = false;     // 调试模式
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

  cv::Mat correct_image(const cv::Mat &raw_image) const {
    CV_Assert(raw_image.size() == defect_map_.size());

    cv::Mat corrected = raw_image.clone();

#pragma omp parallel for collapse(2) if (config_.use_simd)
    for (int y = 0; y < corrected.rows; ++y) {
      for (int x = 0; x < corrected.cols; ++x) {
        if (defect_map_.at<uint8_t>(y, x) != 0) {
          corrected.at<float>(y, x) = interpolate_pixel(x, y, raw_image);
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

private:
  Config config_;
  cv::Mat defect_map_;
  std::unique_ptr<float[]> cache_;

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
};

// 示例用法
int main() {
  // 加载暗场序列
  std::vector<cv::Mat> dark_frames;
  for (const auto &entry : fs::directory_iterator("dark_frames/")) {
    cv::Mat frame = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
    if (!frame.empty()) {
      dark_frames.push_back(frame);
    }
  }

  // 构建缺陷映射
  DefectPixelMapper::Config config;
  DefectPixelMapper mapper(config);
  mapper.build_defect_map(dark_frames);
  mapper.save_map("defect_map.yml");

  // 校正单张图像
  cv::Mat raw_image = cv::imread("astronomy_image.tiff", cv::IMREAD_UNCHANGED);
  cv::Mat corrected = mapper.correct_image(raw_image);

  // 保存结果
  cv::imwrite("corrected_image.tiff", corrected);

  return 0;
}