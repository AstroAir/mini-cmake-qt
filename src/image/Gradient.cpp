#include <functional>
#include <numbers>
#include <opencv2/opencv.hpp>
#include <vector>

namespace rg = std::ranges;
using namespace cv;

/**
 * @brief 高级渐变生成器配置结构体
 *
 * 支持以下特性：
 * 1. 多种渐变模式：极坐标、仿射、混合
 * 2. 多重色彩插值：RGB、HSV、LAB
 * 3. 动态参数调整：大小、中心点、角度、速度
 * 4. 高级混合效果：径向、环形、螺旋等
 */
struct GradientConfig {
  int size = 512;
  Point2f center{0.5f, 0.5f};
  float angle_deg = 0.0f;
  float speed = 1.0f;
  std::vector<std::pair<float, Scalar>> color_stops = {
      {0.0f, {255, 0, 0}}, {0.5f, {0, 255, 0}}, {1.0f, {0, 0, 255}}};

  enum class InterpolationMode {
    RGB,
    HSV,
    LAB
  } interpolation_mode = InterpolationMode::HSV;

  enum class GradientType {
    POLAR,  // 极坐标渐变
    AFFINE, // 仿射渐变
    HYBRID, // 混合渐变
    SPIRAL, // 螺旋渐变
    RADIAL  // 径向渐变
  } type = GradientType::HYBRID;

  struct Effects {
    float noise_amount = 0.0f;   // 噪声强度
    float wave_frequency = 0.0f; // 波纹频率
    float wave_amplitude = 0.0f; // 波纹振幅
    bool mirror_mode = false;    // 镜像模式
  } effects;
};

class RotationalGradientGenerator {
public:
  explicit RotationalGradientGenerator(GradientConfig config)
      : cfg_(std::move(config)), cache_valid_(false) {
    validate_config();
    init_lookup_tables();
  }

  Mat generate_frame(float timestamp = 0.0f) {
    const float current_angle = cfg_.angle_deg + timestamp * cfg_.speed;

    switch (cfg_.type) {
    case GradientConfig::GradientType::POLAR:
      return generate_polar_gradient(current_angle);
    case GradientConfig::GradientType::AFFINE:
      return generate_affine_gradient(current_angle);
    case GradientConfig::GradientType::HYBRID:
      return generate_hybrid_gradient(current_angle);
    case GradientConfig::GradientType::SPIRAL:
      return generate_spiral_gradient(current_angle);
    case GradientConfig::GradientType::RADIAL:
      return generate_spiral_gradient(current_angle); // 使用spiral作为替代
    default:
      throw std::invalid_argument("Unknown gradient type");
    }
  }

  // 动态更新配置
  void update_config(std::function<void(GradientConfig &)> updater) {
    updater(cfg_);
    validate_config();
  }

private:
  GradientConfig cfg_;
  bool cache_valid_;
  std::vector<Vec3f> color_lookup_;
  std::vector<float> angle_lookup_;

  // 初始化查找表以提高性能
  void init_lookup_tables() {
    const int table_size = 1024;
    color_lookup_.resize(table_size);
    angle_lookup_.resize(table_size);

#pragma omp parallel for
    for (int i = 0; i < table_size; ++i) {
      float t = static_cast<float>(i) / (table_size - 1);
      color_lookup_[i] = interpolate_color(t);
      angle_lookup_[i] = t * 2 * std::numbers::pi_v<float>;
    }
  }

  Mat generate_polar_gradient(float angle) const {
    Mat gradient(cfg_.size, cfg_.size, CV_32FC3);
    const float cx = cfg_.center.x * cfg_.size;
    const float cy = cfg_.center.y * cfg_.size;
    const float rad_angle = angle * std::numbers::pi_v<float> / 180.0f;

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < cfg_.size; ++y) {
      auto *row = gradient.ptr<Vec3f>(y);
      for (int x = 0; x < cfg_.size; ++x) {
        float dx = x - cx;
        float dy = y - cy;
        float theta = std::atan2(dy, dx) + rad_angle;

        // 添加波纹效果
        if (cfg_.effects.wave_frequency > 0) {
          float r = std::sqrt(dx * dx + dy * dy);
          theta += std::sin(r * cfg_.effects.wave_frequency) *
                   cfg_.effects.wave_amplitude;
        }

        float t = std::fmod(theta, 2 * std::numbers::pi_v<float>) /
                  (2 * std::numbers::pi_v<float>);

        // 添加噪声
        if (cfg_.effects.noise_amount > 0) {
          t += (rand() / static_cast<float>(RAND_MAX) - 0.5f) *
               cfg_.effects.noise_amount;
        }

        row[x] = lookup_color(t);
      }
    }

    return apply_effects(gradient);
  }

  Mat generate_affine_gradient(float angle) const {
    // 创建带有多色渐变的基图像
    Mat base(cfg_.size, cfg_.size, CV_8UC3);
    for (int y = 0; y < cfg_.size; ++y) {
      const float t = static_cast<float>(y) / cfg_.size;
      base.row(y).setTo(interpolate_color(t, true));
    }

    Mat rotated;
    const auto center = Point2f(cfg_.center * cfg_.size);
    const auto M = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(base, rotated, M, base.size(), INTER_LANCZOS4, BORDER_WRAP);
    return rotated;
  }

  Mat generate_hybrid_gradient(float angle) const {
    Mat polar = generate_polar_gradient(angle);
    Mat affine = generate_affine_gradient(angle);

    // 混合两种渐变模式
    Mat result;
    addWeighted(polar, 0.5, affine, 0.5, 0, result);
    return result;
  }

  // 新增：生成螺旋渐变
  Mat generate_spiral_gradient(float angle) const {
    Mat gradient(cfg_.size, cfg_.size, CV_32FC3);
    const float cx = cfg_.center.x * cfg_.size;
    const float cy = cfg_.center.y * cfg_.size;
    const float rad_angle = angle * std::numbers::pi_v<float> / 180.0f;

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < cfg_.size; ++y) {
      auto *row = gradient.ptr<Vec3f>(y);
      for (int x = 0; x < cfg_.size; ++x) {
        float dx = x - cx;
        float dy = y - cy;
        float r = std::sqrt(dx * dx + dy * dy) / cfg_.size;
        float theta = std::atan2(dy, dx) + rad_angle; // 修复语法错误

        float t = std::fmod(theta + r * 10, 2 * std::numbers::pi_v<float>) /
                  (2 * std::numbers::pi_v<float>);

        row[x] = lookup_color(t);
      }
    }

    return apply_effects(gradient);
  }

  // 优化的颜色查找
  Vec3f lookup_color(float t) const {
    t = std::clamp(t, 0.0f, 1.0f);
    int index = static_cast<int>(t * (color_lookup_.size() - 1));
    return color_lookup_[index];
  }

  // 应用后期效果
  Mat apply_effects(const Mat &input) const {
    Mat result = input.clone();

    if (cfg_.effects.mirror_mode) {
      flip(result, result, 1);
      Mat left(result, Rect(0, 0, result.cols / 2, result.rows));
      Mat right(result, Rect(result.cols / 2, 0, result.cols / 2, result.rows));
      left.copyTo(right);
    }

    return result;
  }

  Vec3f interpolate_color(float t, bool force_rgb = false) const {
    t = std::clamp(t, 0.0f, 1.0f);

    // 修复 upper_bound 的比较函数
    auto it = std::upper_bound(
        cfg_.color_stops.begin(), cfg_.color_stops.end(), t,
        [](float val, const auto &pair) { return val < pair.first; });

    if (it == cfg_.color_stops.begin())
      return Vec3f(it->second[0], it->second[1], it->second[2]);
    if (it == cfg_.color_stops.end())
      return Vec3f(cfg_.color_stops.back().second[0],
                   cfg_.color_stops.back().second[1],
                   cfg_.color_stops.back().second[2]);

    auto [t1, c1] = *(it - 1);
    auto [t2, c2] = *it;
    const float ratio = (t - t1) / (t2 - t1);

    if (cfg_.interpolation_mode == GradientConfig::InterpolationMode::HSV &&
        !force_rgb) {
      return lerp_hsv(c1, c2, ratio);
    } else {
      return lerp_rgb(c1, c2, ratio);
    }
  }

  static Vec3f lerp_rgb(const Scalar &c1, const Scalar &c2, float t) {
    return Vec3f(c1[0] * (1 - t) + c2[0] * t, c1[1] * (1 - t) + c2[1] * t,
                 c1[2] * (1 - t) + c2[2] * t);
  }

  static Vec3f lerp_hsv(const Scalar &c1, const Scalar &c2, float t) {
    Vec3f hsv1, hsv2;
    Mat temp1(1, 1, CV_32FC3, Scalar(c1[0], c1[1], c1[2]));
    Mat temp2(1, 1, CV_32FC3, Scalar(c2[0], c2[1], c2[2]));

    cvtColor(temp1, temp1, COLOR_RGB2HSV);
    cvtColor(temp2, temp2, COLOR_RGB2HSV);

    hsv1 = temp1.at<Vec3f>(0, 0);
    hsv2 = temp2.at<Vec3f>(0, 0);

    // 处理色相环绕
    float hue_diff = hsv2[0] - hsv1[0];
    if (hue_diff > 180)
      hue_diff -= 360;
    else if (hue_diff < -180)
      hue_diff += 360;

    Vec3f result;
    float h = std::fmod(hsv1[0] + hue_diff * t, 180.0f);
    float s = hsv1[1] * (1 - t) + hsv2[1] * t;
    float v = hsv1[2] * (1 - t) + hsv2[2] * t;
    result = Vec3f(h, s, v);

    Mat rgb;
    Mat hsv(1, 1, CV_32FC3, result);
    cvtColor(hsv, rgb, COLOR_HSV2RGB);
    return rgb.at<Vec3f>(0, 0);
  }

  void validate_config() {
    cfg_.center.x = std::clamp(cfg_.center.x, 0.0f, 1.0f);
    cfg_.center.y = std::clamp(cfg_.center.y, 0.0f, 1.0f);
    rg::sort(cfg_.color_stops, {}, &std::pair<float, Scalar>::first);
  }
};

// 交互式GUI控制器
class GradientController {
public:
  GradientController() {
    namedWindow("Control");
    createTrackbar(
        "Size", "Control", &config_.size, 2048,
        [](int value, void *userdata) {
          auto *self = static_cast<GradientController *>(userdata);
          self->update_size(value);
        },
        this);
    int angle = static_cast<int>(config_.angle_deg);
    createTrackbar(
        "Angle", "Control", &angle, 360,
        [](int value, void *userdata) {
          auto *self = static_cast<GradientController *>(userdata);
          self->config_.angle_deg = static_cast<float>(value);
        },
        this);
    createTrackbar(
        "Speed", "Control", &speed_, 100, [](int, void *) {}, nullptr);
    createButton(
        "HSV Mode",
        [](int state, void *userdata) {
          auto *self = static_cast<GradientController *>(userdata);
          self->config_.interpolation_mode =
              state ? GradientConfig::InterpolationMode::HSV
                    : GradientConfig::InterpolationMode::RGB;
        },
        this, QT_CHECKBOX);
  }

  void run() {
    RotationalGradientGenerator gen(config_);
    int frame_count = 0;
    while (true) {
      config_.speed = speed_ / 10.0f; // 速度映射
      Mat frame = gen.generate_frame(frame_count++ * 0.1f);

      // 显示中心标记
      circle(frame, Point(config_.center * config_.size), 5,
             Scalar(255, 255, 255), -1);

      imshow("Output", frame);
      if (waitKey(30) == 27)
        break;
    }
  }

private:
  GradientConfig config_;
  int speed_ = 10; // 控制速度的整数值

  void update_size(int new_size) { config_.size = std::max(64, new_size); }
};

int main() {
  GradientController controller;
  controller.run();
  return 0;
}