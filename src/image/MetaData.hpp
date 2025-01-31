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

/**
 * @brief Concept to define OpenCV image types.
 */
template <typename T>
concept OpenCVImageType = std::same_as<T, cv::Mat> || std::same_as<T, cv::UMat>;

/**
 * @struct ImageMetadata
 * @brief Structure representing metadata for an image.
 */
struct ImageMetadata {
  fs::path path;           ///< Path to the image file.
  cv::Size size;           ///< Size of the image.
  int channels;            ///< Number of channels in the image.
  int depth;               ///< Bit depth of the image.
  std::string color_space; ///< Color space of the image.
  std::chrono::system_clock::time_point timestamp; ///< Timestamp of the image.
  json custom_data; ///< Custom metadata in JSON format.

  /**
   * @brief Adds a custom tag to the metadata.
   * @tparam T The type of the value.
   * @param key The key for the tag.
   * @param value The value for the tag.
   */
  template <typename T> void add_tag(const std::string &key, T &&value) {
    custom_data[key] = std::forward<T>(value);
  }

  /**
   * @brief Removes a custom tag from the metadata.
   * @param key The key for the tag to remove.
   */
  void remove_tag(const std::string &key) { custom_data.erase(key); }

  /**
   * @brief Checks if a custom tag exists in the metadata.
   * @param key The key for the tag to check.
   * @return True if the tag exists, false otherwise.
   */
  bool has_tag(const std::string &key) const noexcept {
    return custom_data.contains(key);
  }

  /**
   * @brief Gets the value of a custom tag.
   * @tparam T The type of the value.
   * @param key The key for the tag.
   * @return An optional containing the value if the tag exists, std::nullopt
   * otherwise.
   */
  template <typename T>
  std::optional<T> get_tag(const std::string &key) const noexcept {
    try {
      return custom_data.at(key).get<T>();
    } catch (...) {
      return std::nullopt;
    }
  }

  /**
   * @brief Merges custom tags from another JSON object.
   * @param other The JSON object containing the tags to merge.
   */
  void merge_tags(const json &other) { custom_data.update(other); }

  /**
   * @brief Clears all custom tags from the metadata.
   */
  void clear_tags() noexcept { custom_data.clear(); }
};

/**
 * @class ImageProcessor
 * @brief Class for processing images and managing their metadata.
 */
class ImageProcessor {
public:
  /**
   * @brief Constructor for ImageProcessor.
   */
  ImageProcessor();

  /**
   * @brief Loads an image and its metadata from a file.
   * @param img_path The path to the image file.
   * @param flags The flags for loading the image (default is
   * cv::IMREAD_ANYCOLOR).
   * @return An optional containing the image metadata if the image was loaded
   * successfully, std::nullopt otherwise.
   */
  [[nodiscard]] std::optional<ImageMetadata>
  load_image(const fs::path &img_path,
             int flags = cv::IMREAD_ANYCOLOR) noexcept;

  /**
   * @brief Saves an image and its metadata to a file.
   * @tparam ImageT The type of the image (cv::Mat or cv::UMat).
   * @param output_path The path to the output image file.
   * @param image The image to save.
   * @param meta The metadata to save with the image.
   * @param quality The quality of the saved image (default is 95).
   * @return True if the image was saved successfully, false otherwise.
   */
  template <OpenCVImageType ImageT>
  bool save_image(const fs::path &output_path, const ImageT &image,
                  const ImageMetadata &meta, int quality = 95) noexcept {
    try {
      validate_save_parameters(output_path, image);
      create_output_directory(output_path);

      if (!write_image_file(output_path, image, quality)) {
        throw std::runtime_error("Failed to write image file");
      }

      save_associated_json(output_path, meta.custom_data);
      logger->info("Image saved successfully: {}", output_path.string());
      return true;
    } catch (const std::exception &e) {
      log_error("save_image", e.what());
      return false;
    }
  }

  /**
   * @brief Adds a custom tag to the image metadata.
   * @tparam T The type of the value.
   * @param meta The image metadata.
   * @param key The key for the tag.
   * @param value The value for the tag.
   * @return True if the tag was added successfully, false otherwise.
   */
  template <typename T>
  bool add_custom_tag(ImageMetadata &meta, const std::string &key,
                      T &&value) noexcept {
    try {
      validate_tag_key(key);
      meta.add_tag(key, std::forward<T>(value));
      logger->info("Tag added successfully: {} -> {}", key, value);
      return true;
    } catch (const std::exception &e) {
      log_error("add_custom_tag", e.what());
      return false;
    }
  }

  /**
   * @brief Adds multiple custom tags to the image metadata.
   * @param meta The image metadata.
   * @param tags The vector of key-value pairs representing the tags to add.
   * @return True if the tags were added successfully, false otherwise.
   */
  bool batch_add_tags(
      ImageMetadata &meta,
      const std::vector<std::pair<std::string, json>> &tags) noexcept;

  /**
   * @brief Imports custom tags from a JSON file.
   * @param meta The image metadata.
   * @param json_file The path to the JSON file.
   * @return True if the tags were imported successfully, false otherwise.
   */
  bool import_tags_from_json(ImageMetadata &meta,
                             const fs::path &json_file) noexcept;

  /**
   * @brief 保存图像元数据到JSON文件
   * @param meta 要保存的元数据
   * @param output_path 输出路径（可选，默认使用meta.path对应的JSON文件）
   * @return 保存是否成功
   */
  bool save_metadata(const ImageMetadata &meta,
                      const std::optional<fs::path> &output_path = std::nullopt) noexcept;

private:
  std::shared_ptr<spdlog::logger> logger; ///< Logger for logging messages.

  /**
   * @brief Validates if a file exists.
   * @param path The path to the file.
   */
  void validate_file_exists(const fs::path &path) const;

  /**
   * @brief Validates if an image is not empty.
   * @param img The image to validate.
   */
  void validate_image_not_empty(const cv::Mat &img) const;

  /**
   * @brief Validates the parameters for saving an image.
   * @param path The path to the output image file.
   * @param image The image to save.
   */
  void validate_save_parameters(const fs::path &path, const auto &image) const;

  /**
   * @brief Validates a custom tag key.
   * @param key The key to validate.
   */
  void validate_tag_key(const std::string &key) const;

  /**
   * @brief Creates metadata for an image.
   * @param path The path to the image file.
   * @param img The image.
   * @return The created metadata.
   */
  ImageMetadata create_metadata(const fs::path &path, const cv::Mat &img) const;

  /**
   * @brief Detects the color space of an image.
   * @param img The image.
   * @return The detected color space.
   */
  std::string detect_color_space(const cv::Mat &img) const;

  /**
   * @brief Creates the output directory if it does not exist.
   * @param path The path to the output directory.
   */
  void create_output_directory(const fs::path &path) const;

  /**
   * @brief Writes an image file to disk.
   * @param path The path to the output image file.
   * @param image The image to save.
   * @param quality The quality of the saved image.
   * @return True if the image was saved successfully, false otherwise.
   */
  bool write_image_file(const fs::path &path, const auto &image,
                        int quality) const;

  /**
   * @brief Saves a FITS image file to disk.
   * @param path The path to the output FITS file.
   * @param image The image to save.
   * @return True if the image was saved successfully, false otherwise.
   */
  bool save_fits_image(const fs::path &path, const cv::Mat &image) const;

  /**
   * @brief Loads associated JSON metadata for an image.
   * @param meta The image metadata.
   */
  void load_associated_json(ImageMetadata &meta) const;

  /**
   * @brief Saves associated JSON metadata for an image.
   * @param img_path The path to the image file.
   * @param data The JSON data to save.
   */
  void save_associated_json(const fs::path &img_path, const json &data) const;

  /**
   * @brief Reads a JSON file from disk.
   * @param path The path to the JSON file.
   * @return The parsed JSON data.
   */
  json read_json_file(const fs::path &path) const;

  /**
   * @brief Logs an error message.
   * @param function The name of the function where the error occurred.
   * @param msg The error message.
   */
  void log_error(const std::string &function, const std::string &msg) const;
};