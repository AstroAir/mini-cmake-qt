#include "MetaData.hpp"
#include "ImageIO.hpp"

#include <algorithm>
#include <chrono>
#include <execution>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>

// Optional SIMD support
#ifdef __SSE4_2__
#include <immintrin.h>
#endif

#include <fitsio.h>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace std::chrono_literals;

namespace {
// Mutex for thread-safe logger access
std::mutex logger_mutex;

// Thread-safe logger wrapper
template <typename... Args>
void log_safe(const std::shared_ptr<spdlog::logger> &logger,
              spdlog::level::level_enum level, const std::string &format,
              Args &&...args) {
  std::lock_guard<std::mutex> lock(logger_mutex);
  logger->log(level, format, std::forward<Args>(args)...);
}
} // namespace

ImageProcessor::ImageProcessor() {
  try {
    // Initialize spdlog with a fallback mechanism
    fs::create_directories("logs");
    logger = spdlog::basic_logger_mt("basic_logger", "logs/metadata.log");

    spdlog::set_default_logger(logger);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] %v");
    spdlog::flush_every(3s);
    logger->info("ImageProcessor initialized successfully");
  } catch (const spdlog::spdlog_ex &ex) {
    // Fallback to stderr if file logger fails
    std::cerr << "Log initialization failed: " << ex.what() << std::endl;
    logger = spdlog::stderr_color_mt("stderr_logger");
  }
}

// Image loading implementation with improved error handling
std::optional<ImageMetadata>
ImageProcessor::load_image(const fs::path &img_path, int flags) noexcept {
  try {
    validate_file_exists(img_path);

    cv::Mat img = loadImage(img_path.string(), flags);
    validate_image_not_empty(img);

    // Apply SIMD optimizations if available
    process_metadata_simd(img);

    ImageMetadata meta = create_metadata(img_path, img);
    load_associated_json(meta);

    logger->info("Image loaded successfully: {}", img_path.string());
    return meta;
  } catch (const std::exception &e) {
    log_error("load_image", e.what());
    return std::nullopt;
  }
}

// New method to load multiple images in parallel
std::vector<std::optional<ImageMetadata>>
ImageProcessor::load_images_parallel(std::span<const fs::path> img_paths,
                                     int flags) noexcept {
  std::vector<std::optional<ImageMetadata>> results(img_paths.size());

  try {
    std::vector<std::future<std::optional<ImageMetadata>>> futures;
    futures.reserve(img_paths.size());

    // Launch tasks asynchronously
    for (const auto &path : img_paths) {
      futures.push_back(std::async(std::launch::async, [this, path, flags]() {
        return this->load_image(path, flags);
      }));
    }

    // Collect results
    for (size_t i = 0; i < futures.size(); ++i) {
      results[i] = futures[i].get();
    }

    logger->info("Parallel loading completed: {} images processed",
                 img_paths.size());

  } catch (const std::exception &e) {
    log_error("load_images_parallel", e.what());
  }

  return results;
}

void ImageProcessor::validate_file_exists(const fs::path &path) const {
  if (!fs::exists(path)) {
    throw std::runtime_error("File not found: " + path.string());
  }

  std::error_code ec;
  auto file_size = fs::file_size(path, ec);
  if (ec) {
    throw std::runtime_error("Cannot determine file size: " + ec.message());
  }

  if (file_size == 0) {
    throw std::runtime_error("File is empty: " + path.string());
  }
}

void ImageProcessor::validate_image_not_empty(const cv::Mat &img) const {
  if (img.empty()) {
    throw std::runtime_error("Decoded image is empty");
  }

  // Additional validation for image dimensions
  if (img.cols <= 0 || img.rows <= 0) {
    throw std::runtime_error(
        "Invalid image dimensions: " + std::to_string(img.cols) + "x" +
        std::to_string(img.rows));
  }
}

void ImageProcessor::validate_save_parameters(const fs::path &path,
                                              const auto &image) const {
  if (image.empty()) {
    throw std::invalid_argument("Cannot save empty image");
  }

  if (path.extension().empty()) {
    throw std::invalid_argument("Missing file extension: " + path.string());
  }

  // Check for valid extensions
  auto ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::array<std::string_view, 7> valid_extensions = {
      ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".fits", ".fit"};

  if (std::ranges::none_of(valid_extensions, [&ext](const auto &valid_ext) {
        return valid_ext == ext;
      })) {
    throw std::invalid_argument("Unsupported file extension: " + ext);
  }
}

void ImageProcessor::validate_tag_key(std::string_view key) const {
  if (key.empty()) {
    throw std::invalid_argument("Tag key cannot be empty");
  }

  // Check for invalid characters using ranges view
  constexpr std::string_view invalid_chars = ".:/\\*?\"<>|";
  if (std::ranges::any_of(key, [&invalid_chars](char c) {
        return invalid_chars.find(c) != std::string_view::npos;
      })) {
    throw std::invalid_argument("Tag key contains invalid characters: " +
                                std::string(key));
  }
}

// Process metadata using SIMD optimizations where applicable
void ImageProcessor::process_metadata_simd(cv::Mat &img) const {
#ifdef __SSE4_2__
  // Apply SIMD optimizations only for continuous matrices of appropriate sizes
  if (img.isContinuous() && img.depth() == CV_8U && img.channels() == 3) {
    const int width = img.cols;
    const int height = img.rows;
    const int step = img.step;

    // Example SIMD processing: calculare average color values
    __m128i sum = _mm_setzero_si128();
    int pixelCount = 0;

    for (int y = 0; y < height; y++) {
      const uint8_t *row = img.ptr<uint8_t>(y);
      for (int x = 0; x < width * 3; x += 16) {
        if (x + 16 <= width * 3) {
          // Load 16 bytes (multiple pixels) at once
          __m128i pixels =
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(&row[x]));
          sum = _mm_add_epi8(sum, pixels);
          pixelCount += 16;
        }
      }
    }

    // Extract results if needed for metadata
    // This is just an example of SIMD usage
  }
#endif
}

ImageMetadata ImageProcessor::create_metadata(const fs::path &path,
                                              const cv::Mat &img) const {
  // Try to safely get the file's last modification time, use current time on
  // failure
  std::chrono::system_clock::time_point timestamp;
  try {
    timestamp = std::chrono::clock_cast<std::chrono::system_clock>(
        fs::last_write_time(path));
  } catch (const std::exception &e) {
    logger->warn(
        "Failed to get file modification time: {}, using current time instead",
        e.what());
    timestamp = std::chrono::system_clock::now();
  }

  ImageMetadata meta{.path = path,
                     .size = img.size(),
                     .channels = img.channels(),
                     .depth = img.depth(),
                     .color_space = detect_color_space(img),
                     .timestamp = timestamp,
                     .custom_data = {}};

  std::string ext = path.extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  if (ext == ".fits" || ext == ".fit" || ext == ".fts") {
    try {
      auto fitsMetadata = getFitsMetadata(path.string());
      if (fitsMetadata) {
        for (const auto &[key, value] : fitsMetadata.value()) {
          meta.add_tag(key, value);
        }
        logger->info("Loaded FITS metadata: {} items",
                     fitsMetadata->size());
      }
    } catch (const std::exception &e) {
      logger->error("Failed to load FITS metadata: {}", e.what());
    }
  }

  return meta;
}

// Detection of color space with more precise identification
std::string ImageProcessor::detect_color_space(const cv::Mat &img) const {
  if (img.channels() == 1) {
    return "GRAY";
  } else if (img.channels() == 3) {
    // Advanced color space detection could be implemented here
    // For example, analyzing sample pixels to determine if it's RGB, HSV, etc.
    return (img.type() == CV_8UC3 || img.type() == CV_16UC3) ? "BGR" : "RGB";
  } else if (img.channels() == 4) {
    return (img.type() == CV_8UC4 || img.type() == CV_16UC4) ? "BGRA" : "RGBA";
  } else {
    return "unknown_" + std::to_string(img.channels());
  }
}

void ImageProcessor::create_output_directory(const fs::path &path) const {
  try {
    if (!path.parent_path().empty()) {
      std::error_code ec;
      fs::create_directories(path.parent_path(), ec);

      if (ec) {
        throw std::runtime_error("Failed to create directory: " + ec.message());
      }
    }
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("Directory creation error: ") +
                             e.what());
  }
}

// Improved write_image_file implementation with better error handling
bool ImageProcessor::write_image_file(const fs::path &path, const auto &image,
                                      int quality) const {
  try {
    if (path.extension() == ".jpg" || path.extension() == ".jpeg") {
      if (!saveMatTo8BitJpg(image, path.string(), quality)) {
        throw std::runtime_error("Failed to save JPEG image");
      }
    } else if (path.extension() == ".png") {
      if (!saveMatTo16BitPng(image, path.string())) {
        throw std::runtime_error("Failed to save PNG image");
      }
    } else if (path.extension() == ".fits" || path.extension() == ".fit" ||
               path.extension() == ".fts") {
      if (!saveMatToFits(image, path.string())) {
        throw std::runtime_error("Failed to save FITS image");
      }
    } else {
      // For other formats use default saving method
      std::vector<int> compression_params;
      if (path.extension() == ".jpg" || path.extension() == ".jpeg") {
        compression_params = {cv::IMWRITE_JPEG_QUALITY, quality};
      }
      if (!saveImage(path.string(), image, compression_params)) {
        throw std::runtime_error("Failed to save image with default method");
      }
    }
    return true;
  } catch (const cv::Exception &e) {
    logger->error("OpenCV error while saving image: {}", e.what());
    return false;
  } catch (const std::exception &e) {
    logger->error("Error saving image to {}: {}", path.string(), e.what());
    return false;
  }
}

// Improved JSON handling with exception safety
void ImageProcessor::load_associated_json(ImageMetadata &meta) const {
  auto json_path = meta.path;
  json_path.replace_extension(".json");

  if (fs::exists(json_path)) {
    try {
      meta.custom_data = read_json_file(json_path);
      logger->info("Loaded associated JSON file: {}", json_path.string());
    } catch (const std::exception &e) {
      log_error("load_associated_json", e.what());
      meta.custom_data = json(); // Ensure we have valid empty JSON
    }
  }
}

// Thread-safe JSON saving
void ImageProcessor::save_associated_json(const fs::path &img_path,
                                          const json &data) const {
  if (!data.empty()) {
    auto json_path = img_path;
    json_path.replace_extension(".json");

    try {
      create_output_directory(json_path);

      // Use atomic writing pattern with temporary file
      auto temp_path = json_path.string() + ".tmp";
      {
        std::ofstream f(temp_path);
        if (!f) {
          throw std::runtime_error("Cannot open file for writing: " +
                                   temp_path);
        }
        f << data.dump(4);
        f.flush();
        if (f.bad()) {
          throw std::runtime_error("Error writing JSON data");
        }
      }

      // Atomic rename
      std::error_code ec;
      fs::rename(temp_path, json_path, ec);

      if (ec) {
        // Fallback to copy and delete if rename fails (e.g., across
        // filesystems)
        fs::copy_file(temp_path, json_path,
                      fs::copy_options::overwrite_existing, ec);
        if (ec) {
          throw std::runtime_error("Failed to create final JSON file: " +
                                   ec.message());
        }
        fs::remove(temp_path, ec);
      }

      logger->info("Saved associated JSON file: {}", json_path.string());
    } catch (const std::exception &e) {
      log_error("save_associated_json", e.what());
    }
  }
}

json ImageProcessor::read_json_file(const fs::path &path) const {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("Cannot open JSON file: " + path.string());
  }

  try {
    json j;
    f >> j;
    return j;
  } catch (const json::exception &e) {
    throw std::runtime_error("JSON parsing error in " + path.string() + ": " +
                             e.what());
  }
}

// Thread-safe error logging
void ImageProcessor::log_error(const std::string &function,
                               const std::string &msg) const {
  log_safe(logger, spdlog::level::err, "[{}] {}", function, msg);
}

// Batch processing of tags with parallel execution for large batches
bool ImageProcessor::batch_add_tags(
    ImageMetadata &meta,
    std::span<const std::pair<std::string, json>> tags) noexcept {
  try {
    // Use parallel algorithm for large batches
    if (tags.size() > 100) {
      // Create a vector of results to track validation failures
      std::vector<bool> results(tags.size(), true);

      // Validate keys in parallel
      std::transform(std::execution::par_unseq, tags.begin(), tags.end(),
                     results.begin(), [this](const auto &pair) {
                       try {
                         validate_tag_key(pair.first);
                         return true;
                       } catch (...) {
                         return false;
                       }
                     });

      // Filter out invalid keys
      for (size_t i = 0; i < tags.size(); ++i) {
        if (results[i]) {
          meta.add_tag(tags[i].first, tags[i].second);
        } else {
          logger->warn("Skipped invalid tag key: {}", tags[i].first);
        }
      }
    } else {
      // For small batches, process sequentially
      for (const auto &[key, value] : tags) {
        try {
          validate_tag_key(key);
          meta.add_tag(key, value);
        } catch (const std::exception &e) {
          logger->warn("Skipped tag '{}': {}", key, e.what());
        }
      }
    }

    logger->info("Added {} tags to metadata", tags.size());
    return true;
  } catch (const std::exception &e) {
    log_error("batch_add_tags", e.what());
    return false;
  }
}

// Enhanced import_tags_from_json with improved error handling
bool ImageProcessor::import_tags_from_json(ImageMetadata &meta,
                                           const fs::path &json_file) noexcept {
  try {
    validate_file_exists(json_file);

    auto imported_json = read_json_file(json_file);

    // Validate JSON is an object
    if (!imported_json.is_object()) {
      throw std::runtime_error("JSON file does not contain a valid object");
    }

    // Use merge operation for efficiency
    meta.custom_data.update(imported_json);

    logger->info("Successfully imported tags from {}: {} items",
                 json_file.string(), imported_json.size());
    return true;
  } catch (const std::exception &e) {
    log_error("import_tags_from_json", e.what());
    return false;
  }
}

// Save metadata with atomic operations
bool ImageProcessor::save_metadata(
    const ImageMetadata &meta,
    const std::optional<fs::path> &output_path) noexcept {
  try {
    fs::path json_path;
    if (output_path.has_value()) {
      json_path = *output_path;
    } else {
      json_path = meta.path;
      json_path.replace_extension(".json");
    }

    // Create full metadata JSON with error handling
    json metadata_json;
    try {
      metadata_json = {
          {"path", meta.path.string()},
          {"size", {{"width", meta.size.width}, {"height", meta.size.height}}},
          {"channels", meta.channels},
          {"depth", meta.depth},
          {"color_space", meta.color_space},
          {"timestamp", std::chrono::system_clock::to_time_t(meta.timestamp)},
          {"custom_data", meta.custom_data}};
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("Failed to create metadata JSON: ") +
                               e.what());
    }

    create_output_directory(json_path);

    // Use atomic write with temporary file
    auto temp_path = json_path.string() + ".tmp";
    {
      std::ofstream f(temp_path, std::ios::out | std::ios::trunc);
      if (!f) {
        throw std::runtime_error("Cannot open file for writing: " + temp_path);
      }

      f << metadata_json.dump(4);
      f.flush();

      if (f.bad()) {
        throw std::runtime_error("Error writing metadata to file");
      }
    }

    // Atomic rename for data integrity
    std::error_code ec;
    fs::rename(temp_path, json_path, ec);

    if (ec) {
      // Fallback if rename fails
      fs::copy_file(temp_path, json_path, fs::copy_options::overwrite_existing,
                    ec);
      if (ec) {
        throw std::runtime_error("Failed to create final metadata file: " +
                                 ec.message());
      }
      fs::remove(temp_path, ec);
    }

    logger->info("Metadata successfully saved to: {}", json_path.string());
    return true;
  } catch (const std::exception &e) {
    log_error("save_metadata", e.what());
    return false;
  }
}