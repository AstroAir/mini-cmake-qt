#include "MetaData.hpp"
#include "ImageIO.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <string>

#include <fitsio.h>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace std::chrono_literals;

ImageProcessor::ImageProcessor() {
  try {
    // 初始化spdlog
    logger = spdlog::basic_logger_mt("basic_logger", "logs/metadata.log");
    spdlog::set_default_logger(logger);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");
    spdlog::flush_on(spdlog::level::info);
  } catch (const spdlog::spdlog_ex &ex) {
    std::cerr << "日志初始化失败: " << ex.what() << std::endl;
  }
}

// 图像加载实现
std::optional<ImageMetadata>
ImageProcessor::load_image(const fs::path &img_path, int flags) noexcept {
  try {
    validate_file_exists(img_path);

    cv::Mat img = loadImage(img_path.string(), flags);
    validate_image_not_empty(img);

    ImageMetadata meta = create_metadata(img_path, img);
    load_associated_json(meta);

    logger->info("加载图像成功: {}", img_path.string());
    return meta;
  } catch (const std::exception &e) {
    log_error("load_image", e.what());
    return std::nullopt;
  }
}

// 校验文件是否存在
void ImageProcessor::validate_file_exists(const fs::path &path) const {
  if (!fs::exists(path)) {
    throw std::runtime_error("文件未找到: " + path.string());
  }
}

// 校验图像是否为空
void ImageProcessor::validate_image_not_empty(const cv::Mat &img) const {
  if (img.empty()) {
    throw std::runtime_error("解码后的图像为空");
  }
}

// 校验保存参数
void ImageProcessor::validate_save_parameters(const fs::path &path,
                                              const auto &image) const {
  if (image.empty()) {
    throw std::invalid_argument("无法保存空图像");
  }
  if (path.extension().empty()) {
    throw std::invalid_argument("缺少文件扩展名: " + path.string());
  }
}

// 校验标签键
void ImageProcessor::validate_tag_key(const std::string &key) const {
  if (key.empty()) {
    throw std::invalid_argument("标签键不能为空");
  }
  if (key.find_first_of(".:") != std::string::npos) {
    throw std::invalid_argument("标签键中包含无效字符: " + key);
  }
}

// 创建图像元数据
ImageMetadata ImageProcessor::create_metadata(const fs::path &path,
                                              const cv::Mat &img) const {
    ImageMetadata meta{
        .path = path,
        .size = img.size(),
        .channels = img.channels(),
        .depth = img.depth(),
        .color_space = detect_color_space(img),
        .timestamp = std::chrono::clock_cast<std::chrono::system_clock>(
            fs::last_write_time(path)),
        .custom_data = {}
    };

    // 对FITS文件加载额外元数据
    if (path.extension() == ".fits" || path.extension() == ".fit") {
        auto fitsMetadata = getFitsMetadata(path.string());
        for (const auto& [key, value] : fitsMetadata) {
            meta.add_tag(key, value);
        }
        logger->info("已加载FITS元数据: {} 项", fitsMetadata.size());
    }

    return meta;
}

// 检测颜色空间
std::string ImageProcessor::detect_color_space(const cv::Mat &img) const {
  switch (img.channels()) {
  case 1:
    return "GRAY";
  case 3:
    return (img.type() == CV_8UC3) ? "BGR" : "RGB";
  case 4:
    return "BGRA";
  default:
    return "unknown";
  }
}

// 创建输出目录
void ImageProcessor::create_output_directory(const fs::path &path) const {
  fs::create_directories(path.parent_path());
}

// 写入图像文件
bool ImageProcessor::write_image_file(const fs::path &path, const auto &image,
                                      int quality) const {
  if (path.extension() == ".jpg" || path.extension() == ".jpeg") {
    return saveMatTo8BitJpg(image, path.string());
  } else if (path.extension() == ".png") {
    return saveMatTo16BitPng(image, path.string());
  } else if (path.extension() == ".fits") {
    return saveMatToFits(image, path.string());
  } else {
    // 其他格式使用默认的保存方法
    return saveImage(path.string(), image);
  }
}

// 加载关联的 JSON 文件
void ImageProcessor::load_associated_json(ImageMetadata &meta) const {
  auto json_path = meta.path;
  json_path.replace_extension(".json");

  if (fs::exists(json_path)) {
    try {
      meta.custom_data = read_json_file(json_path);
      logger->info("加载关联 JSON 文件: {}", json_path.string());
    } catch (const std::exception &e) {
      log_error("load_associated_json", e.what());
    }
  }
}

// 保存关联的 JSON 文件
void ImageProcessor::save_associated_json(const fs::path &img_path,
                                          const json &data) const {
  if (!data.empty()) {
    auto json_path = img_path;
    json_path.replace_extension(".json");

    try {
      std::ofstream f(json_path);
      if (!f) {
        throw std::runtime_error("无法打开文件进行写入: " + json_path.string());
      }
      f << data.dump(4);
      logger->info("保存关联 JSON 文件: {}", json_path.string());
    } catch (const std::exception &e) {
      log_error("save_associated_json", e.what());
    }
  }
}

// 读取 JSON 文件
json ImageProcessor::read_json_file(const fs::path &path) const {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("无法打开 JSON 文件: " + path.string());
  }
  json j;
  f >> j;
  return j;
}

// 记录错误日志
void ImageProcessor::log_error(const std::string &function,
                               const std::string &msg) const {
  logger->error("[{}] {}", function, msg);
}

// 保存元数据
bool ImageProcessor::save_metadata(const ImageMetadata &meta,
                                 const std::optional<fs::path> &output_path) noexcept {
    try {
        fs::path json_path;
        if (output_path.has_value()) {
            json_path = *output_path;
        } else {
            json_path = meta.path;
            json_path.replace_extension(".json");
        }

        // 创建要保存的JSON对象
        json metadata_json = {
            {"path", meta.path.string()},
            {"size", {
                {"width", meta.size.width},
                {"height", meta.size.height}
            }},
            {"channels", meta.channels},
            {"depth", meta.depth},
            {"color_space", meta.color_space},
            {"timestamp", std::chrono::system_clock::to_time_t(meta.timestamp)},
            {"custom_data", meta.custom_data}
        };

        create_output_directory(json_path);
        
        std::ofstream f(json_path);
        if (!f) {
            throw std::runtime_error("无法打开文件进行写入: " + json_path.string());
        }
        f << metadata_json.dump(4);
        
        logger->info("元数据保存成功: {}", json_path.string());
        return true;
    } catch (const std::exception &e) {
        log_error("save_metadata", e.what());
        return false;
    }
}