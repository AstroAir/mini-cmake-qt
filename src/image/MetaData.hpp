#pragma once

#include <chrono>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

// 定义 OpenCV 图像类型概念
template <typename T>
concept OpenCVImageType = std::same_as<T, cv::Mat> || std::same_as<T, cv::UMat>;

// 图像元数据结构体
struct ImageMetadata {
  fs::path path;
  cv::Size size;
  int channels;
  int depth;
  std::string color_space;
  std::chrono::system_clock::time_point timestamp;
  json custom_data;

  // 添加标签
  template <typename T> void add_tag(const std::string &key, T &&value) {
    custom_data[key] = std::forward<T>(value);
  }

  // 移除标签
  void remove_tag(const std::string &key) { custom_data.erase(key); }

  // 检查是否存在标签
  bool has_tag(const std::string &key) const noexcept {
    return custom_data.contains(key);
  }

  // 获取标签值
  template <typename T>
  std::optional<T> get_tag(const std::string &key) const noexcept {
    try {
      return custom_data.at(key).get<T>();
    } catch (...) {
      return std::nullopt;
    }
  }

  // 合并标签
  void merge_tags(const json &other) { custom_data.update(other); }

  // 清空所有标签
  void clear_tags() noexcept { custom_data.clear(); }
};

// 图像处理器类
class ImageProcessor {
public:
  ImageProcessor();

  // 图像加载
  [[nodiscard]] std::optional<ImageMetadata>
  load_image(const fs::path &img_path,
             int flags = cv::IMREAD_ANYCOLOR) noexcept;

  // 图像保存
  template <OpenCVImageType ImageT>
  bool save_image(const fs::path &output_path, const ImageT &image,
                  const ImageMetadata &meta, int quality = 95) noexcept {
    try {
      validate_save_parameters(output_path, image);
      create_output_directory(output_path);

      if (!write_image_file(output_path, image, quality)) {
        throw std::runtime_error("图像写入操作失败");
      }

      save_associated_json(output_path, meta.custom_data);
      logger->info("保存图像成功: {}", output_path.string());
      return true;
    } catch (const std::exception &e) {
      log_error("save_image", e.what());
      return false;
    }
  }

  // 添加自定义标签
  template <typename T>
  bool add_custom_tag(ImageMetadata &meta, const std::string &key,
                      T &&value) noexcept {
    try {
      validate_tag_key(key);
      meta.add_tag(key, std::forward<T>(value));
      logger->info("添加标签成功: {} -> {}", key, value);
      return true;
    } catch (const std::exception &e) {
      log_error("add_custom_tag", e.what());
      return false;
    }
  }

  // 批量添加标签
  bool batch_add_tags(
      ImageMetadata &meta,
      const std::vector<std::pair<std::string, json>> &tags) noexcept;

  // 从 JSON 文件导入标签
  bool import_tags_from_json(ImageMetadata &meta,
                             const fs::path &json_file) noexcept;

private:
  std::shared_ptr<spdlog::logger> logger;

  // 校验工具方法
  void validate_file_exists(const fs::path &path) const;
  void validate_image_not_empty(const cv::Mat &img) const;
  void validate_save_parameters(const fs::path &path, const auto &image) const;
  void validate_tag_key(const std::string &key) const;

  // 元数据创建
  ImageMetadata create_metadata(const fs::path &path, const cv::Mat &img) const;

  // 图像分析
  std::string detect_color_space(const cv::Mat &img) const;

  // 文件操作
  void create_output_directory(const fs::path &path) const;
  bool write_image_file(const fs::path &path, const auto &image,
                        int quality) const;

  // FITS保存专用方法
  bool save_fits_image(const fs::path &path, const cv::Mat &image) const;

  // JSON处理
  void load_associated_json(ImageMetadata &meta) const;
  void save_associated_json(const fs::path &img_path, const json &data) const;
  json read_json_file(const fs::path &path) const;

  // 日志记录
  void log_error(const std::string &function, const std::string &msg) const;
};