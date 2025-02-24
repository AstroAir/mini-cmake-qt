#include "StarMarker.hpp"
#include <chrono>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>
#include <sstream>

namespace {
std::shared_ptr<spdlog::logger> marker_logger;

void ensure_logger_initialized() {
  if (!marker_logger) {
    marker_logger = spdlog::rotating_logger_mt(
        "star_marker", "logs/star_marker.log", 1024 * 1024 * 5, 3);
    marker_logger->set_level(spdlog::level::debug);
    marker_logger->flush_on(spdlog::level::info);
  }
}
} // namespace

StarMarker::StarMarker(StarMarkConfig config) : config_(std::move(config)) {
  ensure_logger_initialized();
  marker_logger->info("StarMarker initialized with {} thread(s)",
                      config_.thread_count);

#ifdef USE_CUDA
  if (config_.use_gpu_acceleration && check_cuda_device()) {
    marker_logger->info("CUDA acceleration enabled");
  }
#endif
}

cv::Mat StarMarker::mark_stars(const cv::Mat &image,
                               const std::vector<StarInfo> &stars) const {
  auto start_time = std::chrono::high_resolution_clock::now();
  marker_logger->info("Starting star marking process for {} stars",
                      stars.size());

  cv::Mat marked_image;
  try {
#ifdef USE_CUDA
    if (config_.use_gpu_acceleration && check_cuda_device()) {
      marked_image = mark_stars_cuda(image, stars);
      if (marked_image.empty() && parallel_config::ENABLE_GPU_FALLBACK) {
        marker_logger->warn("GPU processing failed, falling back to CPU");
        goto cpu_processing;
      }
      return marked_image;
    }
#endif

  cpu_processing:
    if (image.channels() == 1) {
      cv::cvtColor(image, marked_image, cv::COLOR_GRAY2BGR);
    } else {
      marked_image = image.clone();
    }

    std::vector<StarInfo> filtered_stars;
    filtered_stars.reserve(stars.size());
    marker_logger->debug("Filtering stars...");
    for (const auto &star : stars) {
      if (should_mark_star(star)) {
        filtered_stars.push_back(star);
      }
    }
    marker_logger->debug("Filtered {} stars", filtered_stars.size());

    if (config_.prevent_label_overlap) {
      optimize_label_positions(marked_image, filtered_stars);
    }

#ifdef USE_OPENMP
    if (config_.use_parallel_processing &&
        filtered_stars.size() >= parallel_config::MIN_PARALLEL_SIZE) {
      marker_logger->debug("Using OpenMP parallel processing");
      int num_threads =
          std::min(config_.thread_count, parallel_config::DEFAULT_THREAD_COUNT);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
      for (int i = 0; i < filtered_stars.size(); ++i) {
        const auto &star = filtered_stars[i];
        cv::Mat layer = marked_image.clone();
        mark_single_star(layer, star);
#pragma omp critical
        {
          marked_image = layer;
        }
      }
    } else
#endif
    {
      marker_logger->debug("Using sequential processing");
      for (const auto &star : filtered_stars) {
        mark_single_star(marked_image, star);
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

    log_performance_metrics("mark_stars", duration);

    if (config_.export_metadata) {
      export_metadata(stars);
    }

  } catch (const std::exception &e) {
    marker_logger->error("Error in mark_stars: {}", e.what());
    throw;
  }

  return marked_image;
}

#ifdef USE_CUDA
cv::Mat StarMarker::mark_stars_cuda(const cv::Mat &image,
                                    const std::vector<StarInfo> &stars) const {
  marker_logger->debug("Starting CUDA-accelerated star marking");

  try {
    cv::Mat marked_image;
    if (image.channels() == 1) {
      cv::cvtColor(image, marked_image, cv::COLOR_GRAY2BGR);
    } else {
      marked_image = image.clone();
    }

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 异步处理每个星点
    for (const auto &star : stars) {
      if (should_mark_star(star)) {
        process_star_cuda(marked_image, star, stream);
      }
    }

    // 同步并清理
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return marked_image;
  } catch (const std::exception &e) {
    marker_logger->error("CUDA processing failed: {}", e.what());
    if (parallel_config::ENABLE_GPU_FALLBACK) {
      marker_logger->warn("Falling back to CPU processing");
      return mark_stars(image, stars);
    }
    throw;
  }
}

bool StarMarker::check_cuda_device() const {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);

  if (error != cudaSuccess || device_count == 0) {
    marker_logger->warn("No CUDA devices available");
    return false;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  marker_logger->info("Using CUDA device: {} with {} MPs", prop.name,
                      prop.multiProcessorCount);
  return true;
}
#endif

void StarMarker::log_performance_metrics(const std::string &operation,
                                         double duration) const {
  marker_logger->info("{} completed in {:.2f}ms", operation, duration);

#ifdef USE_OPENMP
  if (config_.use_parallel_processing) {
    marker_logger->info("OpenMP threads used: {}", config_.thread_count);
  }
#endif

#ifdef USE_CUDA
  if (config_.use_gpu_acceleration) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    marker_logger->info("GPU utilization: {:.2f}%",
                        // 这里可以添加GPU利用率计算
                        0.0);
  }
#endif
}

void StarMarker::set_config(const StarMarkConfig &config) { config_ = config; }

const StarMarkConfig &StarMarker::get_config() const { return config_; }

bool StarMarker::should_mark_star(const StarInfo &star) const {
  return star.fwhm >= config_.min_fwhm && star.fwhm <= config_.max_fwhm;
}

void StarMarker::mark_single_star(cv::Mat &image, const StarInfo &star) const {
  try {
    if (!check_bounds(star.position, image.size())) {
      return;
    }

    // 绘制发光效果
    if (config_.draw_glow) {
      draw_glow_effect(image, star);
    }

    // 绘制标注
    draw_marker(image, star);

    // 生成文本
    std::string mark_text = generate_mark_text(star);
    if (!mark_text.empty()) {
      // 计算文本尺寸和位置
      int baseline = 0;
      cv::Size text_size = cv::getTextSize(mark_text, cv::FONT_HERSHEY_SIMPLEX,
                                           config_.font_scale,
                                           config_.text_thickness, &baseline);

      cv::Point text_pos =
          calculate_optimal_text_position(image, star, text_size);

      // 绘制引导线
      if (config_.use_leader_lines) {
        draw_leader_line(image, star.position, text_pos);
      }

      // 绘制文本背景
      if (config_.draw_shadow) {
        cv::Rect text_rect(text_pos.x - 2, text_pos.y - text_size.height - 2,
                           text_size.width + 4,
                           text_size.height + baseline + 4);

        // 创建阴影效果
        cv::Mat shadow = cv::Mat::zeros(text_rect.size(), CV_8UC3);
        cv::rectangle(shadow, cv::Rect(0, 0, text_rect.width, text_rect.height),
                      cv::Scalar(0, 0, 0), -1);
        cv::GaussianBlur(shadow, shadow, cv::Size(3, 3), 0);

        // 混合阴影
        cv::Mat roi = image(text_rect);
        cv::addWeighted(roi, 1.0 - config_.shadow_opacity, shadow,
                        config_.shadow_opacity, 0, roi);
      }

      // 绘制文本
      int line_type = config_.antialiased ? cv::LINE_AA : cv::LINE_8;
      cv::putText(image, mark_text, text_pos, cv::FONT_HERSHEY_SIMPLEX,
                  config_.font_scale, config_.text_color,
                  config_.text_thickness, line_type);
    }
  } catch (const std::exception &e) {
    spdlog::error("Error marking star: {}", e.what());
  }
}

std::string StarMarker::generate_mark_text(const StarInfo &star) const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1);

  if (config_.show_numbers && star.index >= 0) {
    oss << "#" << star.index;
  }

  if (config_.show_metrics) {
    if (config_.show_numbers && star.index >= 0) {
      oss << " ";
    }
    if (star.fwhm > 0) {
      oss << "F:" << star.fwhm;
    }
    if (star.hfr > 0) {
      oss << " H:" << star.hfr;
    }
    if (star.snr > 0) {
      oss << " S:" << star.snr;
    }
  }

  return oss.str();
}

void StarMarker::save_marked_image(const cv::Mat &image) const {
  if (config_.output_path.empty()) {
    spdlog::warn("Output path not specified for marked image");
    return;
  }

  try {
    // 创建输出目录
    fs::create_directories(config_.output_path.parent_path());

    // 构建完整的输出路径
    fs::path output_file = config_.output_path;
    if (output_file.extension().empty()) {
      output_file.replace_extension(config_.output_format);
    }

    // 保存图像
    cv::imwrite(output_file.string(), image);
    spdlog::info("Saved marked image to: {}", output_file.string());
  } catch (const std::exception &e) {
    spdlog::error("Failed to save marked image: {}", e.what());
  }
}

// 新增方法实现...
cv::Scalar StarMarker::get_color_from_mapping(const StarInfo &star) const {
  if (!config_.use_color_mapping) {
    return config_.circle_color;
  }

  switch (config_.color_map) {
  case StarMarkConfig::ColorMapType::Quality:
    return color_from_quality(star.quality_score);
  case StarMarkConfig::ColorMapType::Magnitude:
    return color_from_magnitude(star.magnitude);
  case StarMarkConfig::ColorMapType::Temperature:
    return color_from_temperature(star.temperature);
  default:
    return config_.circle_color;
  }
}

void StarMarker::draw_marker(cv::Mat &image, const StarInfo &star) const {
  cv::Scalar color = get_color_from_mapping(star);
  int line_type = config_.antialiased ? cv::LINE_AA : cv::LINE_8;

  switch (config_.marker_style) {
  case StarMarkConfig::MarkerStyle::Circle:
    cv::circle(image, star.position, config_.circle_radius, color,
               config_.circle_thickness, line_type);
    break;
  case StarMarkConfig::MarkerStyle::Cross:
    // 绘制十字标记
    cv::line(
        image,
        cv::Point(star.position.x - config_.circle_radius, star.position.y),
        cv::Point(star.position.x + config_.circle_radius, star.position.y),
        color, config_.circle_thickness, line_type);
    cv::line(
        image,
        cv::Point(star.position.x, star.position.y - config_.circle_radius),
        cv::Point(star.position.x, star.position.y + config_.circle_radius),
        color, config_.circle_thickness, line_type);
    break;
  case StarMarkConfig::MarkerStyle::Square:
    // 绘制方形标记
    cv::rectangle(image,
                  cv::Point(star.position.x - config_.circle_radius,
                            star.position.y - config_.circle_radius),
                  cv::Point(star.position.x + config_.circle_radius,
                            star.position.y + config_.circle_radius),
                  color, config_.circle_thickness, line_type);
    break;
  case StarMarkConfig::MarkerStyle::Diamond: {
    // 绘制菱形标记
    std::vector<cv::Point> diamond_points = {
        cv::Point(star.position.x, star.position.y - config_.circle_radius),
        cv::Point(star.position.x + config_.circle_radius, star.position.y),
        cv::Point(star.position.x, star.position.y + config_.circle_radius),
        cv::Point(star.position.x - config_.circle_radius, star.position.y)};
    cv::polylines(image, diamond_points, true, color, config_.circle_thickness,
                  line_type);
    break;
  }
  case StarMarkConfig::MarkerStyle::Combo:
    // 绘制组合标记
    cv::circle(image, star.position, config_.circle_radius, color,
               config_.circle_thickness, line_type);
    cv::line(
        image,
        cv::Point(star.position.x - config_.circle_radius / 2, star.position.y),
        cv::Point(star.position.x + config_.circle_radius / 2, star.position.y),
        color, config_.circle_thickness, line_type);
    cv::line(
        image,
        cv::Point(star.position.x, star.position.y - config_.circle_radius / 2),
        cv::Point(star.position.x, star.position.y + config_.circle_radius / 2),
        color, config_.circle_thickness, line_type);
    break;
  }
}

// ...更多新方法的实现...

namespace {
// 辅助函数：将值规范化到 0-1 范围
inline double normalize(double value, double min_val, double max_val) {
  return std::clamp((value - min_val) / (max_val - min_val), 0.0, 1.0);
}

// 辅助函数：从 HSV 转换到 BGR
cv::Scalar hsv_to_bgr(double h, double s, double v) {
  cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h * 179, s * 255, v * 255));
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  return cv::Scalar(bgr.at<cv::Vec3b>(0)[0], bgr.at<cv::Vec3b>(0)[1],
                    bgr.at<cv::Vec3b>(0)[2]);
}
} // namespace

cv::Scalar StarMarker::color_from_quality(double quality_score) const {
  // 使用HSV色彩空间，便于生成渐变色
  // 质量分数 0-1 映射到色相 0-120 (红色到绿色)
  double normalized = normalize(quality_score, 0.0, 1.0);
  double hue = normalized * 0.333; // 将色相限制在红色到绿色范围 (0-120度)
  return hsv_to_bgr(hue, 1.0, 1.0);
}

cv::Scalar StarMarker::color_from_magnitude(double magnitude) const {
  // 亮星(小星等)为蓝色，暗星(大星等)为红色
  // 典型星等范围：-1到6等
  constexpr double MIN_MAG = -1.0;
  constexpr double MAX_MAG = 6.0;

  double normalized = 1.0 - normalize(magnitude, MIN_MAG, MAX_MAG);
  // 使用色温类似的映射，从红色(低温)到蓝色(高温)
  double hue = 0.667 * normalized; // 0.667对应240度，蓝色
  double saturation = 1.0;
  double value = std::max(0.7, normalized); // 亮星也更亮

  return hsv_to_bgr(hue, saturation, value);
}

cv::Scalar StarMarker::color_from_temperature(double temperature) const {
  // 基于黑体辐射的近似色温映射
  // 典型恒星温度范围：2000K到30000K
  constexpr double MIN_TEMP = 2000.0;
  constexpr double MAX_TEMP = 30000.0;

  double normalized = normalize(temperature, MIN_TEMP, MAX_TEMP);

  // 使用分段函数模拟黑体辐射颜色
  double hue, saturation, value;

  if (normalized < 0.33) {    // 红色到黄色
    hue = normalized * 0.167; // 0-60度
    saturation = 1.0;
    value = 0.9 + normalized * 0.1;
  } else if (normalized < 0.67) { // 黄色到白色
    hue = 0.167;                  // 60度(黄色)
    saturation = 1.0 - (normalized - 0.33) * 2;
    value = 1.0;
  } else {                                    // 白色到蓝白色
    hue = 0.583 + (normalized - 0.67) * 0.25; // 210-300度
    saturation = 0.3 + (normalized - 0.67);
    value = 1.0;
  }

  return hsv_to_bgr(hue, saturation, value);
}

bool StarMarker::check_bounds(const cv::Point &point,
                              const cv::Size &size) const {
  static constexpr int MARGIN = 5; // 边界安全边距

  // 检查点是否在图像范围内（考虑边距）
  if (point.x < MARGIN || point.y < MARGIN || point.x >= size.width - MARGIN ||
      point.y >= size.height - MARGIN) {
    spdlog::warn("Point ({}, {}) out of bounds for image size {}x{}", point.x,
                 point.y, size.width, size.height);
    return false;
  }

  return true;
}
