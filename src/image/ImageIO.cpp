#include "ImageIO.hpp"

#include <fitsio.h>

#ifdef _WIN32
#undef TBYTE
#endif

#include "spdlog/sinks/basic_file_sink.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <ranges>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace std::string_literals;

namespace {
// Initialize logger with modern pattern
std::shared_ptr<spdlog::logger> imageIOLogger =
    spdlog::basic_logger_mt("ImageIOLogger", "logs/image_io.log");

// Use RAII pattern for FITS file operations
class FitsFileGuard {
  fitsfile *fptr_ = nullptr;
  int status_ = 0;

public:
  explicit FitsFileGuard(fitsfile *fptr = nullptr) noexcept : fptr_(fptr) {}

  ~FitsFileGuard() noexcept {
    if (fptr_) {
      fits_close_file(fptr_, &status_);
      if (status_ != 0) {
        char error_msg[FLEN_ERRMSG];
        fits_get_errstatus(status_, error_msg);
        imageIOLogger->error("FITS cleanup error: {}", error_msg);
      }
    }
  }

  // Move operations
  FitsFileGuard(FitsFileGuard &&other) noexcept
      : fptr_(std::exchange(other.fptr_, nullptr)),
        status_(std::exchange(other.status_, 0)) {}

  FitsFileGuard &operator=(FitsFileGuard &&other) noexcept {
    if (this != &other) {
      if (fptr_)
        fits_close_file(fptr_, &status_);
      fptr_ = std::exchange(other.fptr_, nullptr);
      status_ = std::exchange(other.status_, 0);
    }
    return *this;
  }

  // No copy operations
  FitsFileGuard(const FitsFileGuard &) = delete;
  FitsFileGuard &operator=(const FitsFileGuard &) = delete;

  // Accessors
  [[nodiscard]] fitsfile *get() const noexcept { return fptr_; }
  [[nodiscard]] int &status() noexcept { return status_; }
  [[nodiscard]] const int &status() const noexcept { return status_; }

  // Set file pointer
  void reset(fitsfile *fptr = nullptr) noexcept {
    if (fptr_)
      fits_close_file(fptr_, &status_);
    fptr_ = fptr;
    status_ = 0;
  }

  // Check if operation succeeded
  [[nodiscard]] bool ok() const noexcept { return status_ == 0; }

  // Retrieve error message
  [[nodiscard]] std::string error_msg() const noexcept {
    if (status_ == 0)
      return "";
    char error_msg[FLEN_ERRMSG];
    fits_get_errstatus(status_, error_msg);
    return std::string(error_msg);
  }
};

// Utility to check if a file is a FITS file
[[nodiscard]] bool isFitsFile(const fs::path &path) noexcept {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return ext == ".fits" || ext == ".fit" || ext == ".fts";
}

// SIMD-accelerated min-max normalization when available
template <typename T>
void normalizeWithSIMD(const cv::Mat &src, cv::Mat &dst, double alpha = 0,
                       double beta = 255) {
  // OpenCV already uses SIMD internally when possible
  double minVal, maxVal;
  cv::minMaxLoc(src, &minVal, &maxVal);

  if (std::abs(maxVal - minVal) < 1e-6) {
    // Avoid division by zero
    src.convertTo(dst, CV_8U, 0, beta);
  } else {
    double scale = alpha < beta ? (beta - alpha) / (maxVal - minVal)
                                : (alpha - beta) / (maxVal - minVal);
    double shift = alpha - minVal * scale;
    src.convertTo(dst, CV_8U, scale, shift);
  }
}

// Task executor for parallel image processing
class TaskExecutor {
private:
  unsigned int max_threads_;
  std::atomic<unsigned> active_tasks_{0};

public:
  explicit TaskExecutor(unsigned max_threads = 0) noexcept
      : max_threads_(max_threads == 0 ? std::thread::hardware_concurrency()
                                      : max_threads) {}

  template <typename F> auto schedule(F &&task) {
    active_tasks_++;

    auto wrapped_task = [task = std::forward<F>(task), this]() mutable {
      auto result = task();
      active_tasks_--;
      return result;
    };

    // If we're at max capacity, run synchronously
    if (active_tasks_ > max_threads_) {
      auto result = wrapped_task();
      return std::async(std::launch::deferred,
                        [r = std::move(result)]() { return r; });
    }

    return std::async(std::launch::async, wrapped_task);
  }

  void wait_all(std::vector<std::future<void>> &futures) {
    for (auto &future : futures) {
      if (future.valid())
        future.wait();
    }
  }
};

} // namespace

// Implementation of error to string conversion
std::string_view errorToString(ImageIOError error) noexcept {
  switch (error) {
  case ImageIOError::FileNotFound:
    return "File not found";
  case ImageIOError::InvalidFormat:
    return "Invalid format";
  case ImageIOError::ReadError:
    return "Read error";
  case ImageIOError::WriteError:
    return "Write error";
  case ImageIOError::EmptyImage:
    return "Empty image";
  case ImageIOError::InvalidOperation:
    return "Invalid operation";
  case ImageIOError::UnsupportedFormat:
    return "Unsupported format";
  case ImageIOError::MetadataError:
    return "Metadata error";
  default:
    return "Unknown error";
  }
}

// Load FITS image implementation with RAII and improved error handling
static auto loadFitsImage(const fs::path &filename) noexcept
    -> std::expected<cv::Mat, ImageIOError> {

  FitsFileGuard fitsGuard;
  int &status = fitsGuard.status();
  fitsfile *fptr = nullptr;

  try {
    // Open FITS file
    if (fits_open_file(&fptr, filename.string().c_str(), READONLY, &status)) {
      imageIOLogger->error("Cannot open FITS file: {}", filename.string());
      return std::unexpected(ImageIOError::ReadError);
    }

    fitsGuard.reset(fptr);

    // Get image info
    int naxis = 0;
    long naxes[2] = {0, 0};
    int bitpix = 0;

    if (fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status)) {
      imageIOLogger->error("Cannot get FITS image parameters");
      return std::unexpected(ImageIOError::ReadError);
    }

    // Validate dimensions
    if (naxis != 2 || naxes[0] <= 0 || naxes[1] <= 0 || naxes[0] > 65535 ||
        naxes[1] > 65535) {
      imageIOLogger->error("Invalid FITS image dimensions: {}x{}", naxes[0],
                           naxes[1]);
      return std::unexpected(ImageIOError::InvalidFormat);
    }

    // Allocate memory and read data using RAII vector
    std::vector<float> contents(naxes[0] * naxes[1]);
    long fpixel[2] = {1, 1}; // FITS uses 1-based indexing

    if (fits_read_pix(fptr, TFLOAT, fpixel, naxes[0] * naxes[1], nullptr,
                      contents.data(), nullptr, &status)) {
      imageIOLogger->error("Failed to read FITS pixel data");
      return std::unexpected(ImageIOError::ReadError);
    }

    // Convert to OpenCV format with efficient memory handling
    cv::Mat result(naxes[1], naxes[0], CV_32F);
    std::memcpy(result.data, contents.data(), contents.size() * sizeof(float));

    // Normalize to 0-255 range efficiently
    cv::Mat normalized;
    normalizeWithSIMD<float>(result, normalized);

    return normalized;
  } catch (const std::exception &e) {
    imageIOLogger->error("FITS loading exception: {}", e.what());
    return std::unexpected(ImageIOError::ReadError);
  }
}

auto loadImage(PathLike auto &&filename, int flags) noexcept
    -> std::expected<cv::Mat, ImageIOError> {

  const fs::path filepath{std::forward<decltype(filename)>(filename)};

  try {
    imageIOLogger->info("Loading image '{}' with flags={}", filepath.string(),
                        flags);

    // Check if file exists
    std::error_code ec;
    if (!fs::exists(filepath, ec) || ec) {
      imageIOLogger->error("Image file does not exist: {}", filepath.string());
      return std::unexpected(ImageIOError::FileNotFound);
    }

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat image;

    // Load appropriate format
    if (isFitsFile(filepath)) {
      auto result = loadFitsImage(filepath);
      if (!result)
        return result; // Forward the error
      image = std::move(result.value());
    } else {
      image = cv::imread(filepath.string(), flags);
      if (image.empty()) {
        imageIOLogger->error("Failed to load image: {}", filepath.string());
        return std::unexpected(ImageIOError::ReadError);
      }
    }

    // Calculate duration
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    // Log success
    imageIOLogger->info("Successfully loaded image: {}", filepath.string());
    imageIOLogger->info(
        "Image properties: {}x{}, {} channels, type={}, depth={}", image.cols,
        image.rows, image.channels(), image.type(), image.depth());
    imageIOLogger->info("Loading time: {}ms", duration);

    return image;
  } catch (const cv::Exception &e) {
    imageIOLogger->error("OpenCV exception in loadImage: {}", e.what());
    return std::unexpected(ImageIOError::ReadError);
  } catch (const std::exception &e) {
    imageIOLogger->error("Exception in loadImage: {}", e.what());
    return std::unexpected(ImageIOError::ReadError);
  }
}

auto loadImages(PathLike auto &&folder, std::span<const std::string> filenames,
                int flags, unsigned max_threads) noexcept
    -> std::vector<
        std::pair<std::string, std::expected<cv::Mat, ImageIOError>>> {

  const fs::path folderPath{std::forward<decltype(folder)>(folder)};

  try {
    imageIOLogger->info("Starting batch image loading from folder: {}",
                        folderPath.string());

    // Check if folder exists
    std::error_code ec;
    if (!fs::exists(folderPath, ec) || !fs::is_directory(folderPath, ec) ||
        ec) {
      imageIOLogger->error("Folder does not exist or is not accessible: {}",
                           folderPath.string());
      return {};
    }

    // Create executor for parallel processing
    TaskExecutor executor(max_threads);

    std::vector<std::pair<std::string, std::expected<cv::Mat, ImageIOError>>>
        images;
    std::mutex images_mutex; // Protect concurrent writes to results vector
    auto startTotal = std::chrono::high_resolution_clock::now();
    std::atomic<int> successCount = 0;
    std::atomic<int> failCount = 0;

    std::vector<std::future<void>> futures;

    if (filenames.empty()) {
      imageIOLogger->info("Scanning directory for all image files...");

      // Use C++20 ranges to collect and process image files
      std::vector<fs::path> imageFiles;
      std::error_code ec;

      for (const auto &entry : fs::directory_iterator(folderPath, ec)) {
        if (entry.is_regular_file(ec)) {
          imageFiles.push_back(entry.path());
        }
      }

      // Start one task per image file
      for (const auto &filepath : imageFiles) {
        futures.push_back(executor.schedule([&filepath, flags, &successCount,
                                             &failCount, &images_mutex,
                                             &images]() {
          auto start = std::chrono::high_resolution_clock::now();
          auto result = loadImage(filepath, flags);
          auto end = std::chrono::high_resolution_clock::now();
          auto duration =
              std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();

          if (result) {
            std::lock_guard<std::mutex> lock(images_mutex);
            images.emplace_back(filepath.string(), std::move(result));
            successCount++;

            imageIOLogger->info("Loaded image {}: {}x{}, {} channels ({}ms)",
                                filepath.string(), result.value().cols,
                                result.value().rows, result.value().channels(),
                                duration);
          } else {
            std::lock_guard<std::mutex> lock(images_mutex);
            images.emplace_back(filepath.string(), result);
            failCount++;

            imageIOLogger->error("Failed to load image: {} ({}ms) - {}",
                                 filepath.string(), duration,
                                 errorToString(result.error()));
          }
        }));
      }
    } else {
      imageIOLogger->info("Loading {} specified image files...",
                          filenames.size());

      for (const auto &filename : filenames) {
        fs::path filepath = folderPath / filename;
        futures.push_back(executor.schedule([filepath, flags, &successCount,
                                             &failCount, &images_mutex,
                                             &images]() {
          auto start = std::chrono::high_resolution_clock::now();
          auto result = loadImage(filepath, flags);
          auto end = std::chrono::high_resolution_clock::now();
          auto duration =
              std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();

          if (result) {
            std::lock_guard<std::mutex> lock(images_mutex);
            images.emplace_back(filepath.string(), std::move(result));
            successCount++;

            imageIOLogger->info("Loaded image {}: {}x{}, {} channels ({}ms)",
                                filepath.string(), result.value().cols,
                                result.value().rows, result.value().channels(),
                                duration);
          } else {
            std::lock_guard<std::mutex> lock(images_mutex);
            images.emplace_back(filepath.string(), result);
            failCount++;

            imageIOLogger->error("Failed to load image: {} ({}ms) - {}",
                                 filepath.string(), duration,
                                 errorToString(result.error()));
          }
        }));
      }
    }

    // Wait for all tasks to complete
    executor.wait_all(futures);

    // Sort results by filename for consistent output
    std::ranges::sort(
        images, [](const auto &a, const auto &b) { return a.first < b.first; });

    auto endTotal = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                             endTotal - startTotal)
                             .count();

    imageIOLogger->info("Batch loading complete:");
    imageIOLogger->info("  Success: {} images", successCount.load());
    imageIOLogger->info("  Failed: {} images", failCount.load());
    imageIOLogger->info("  Total time: {}ms", totalDuration);
    imageIOLogger->info("  Average time per image: {}ms",
                        (successCount > 0) ? totalDuration / successCount.load()
                                           : 0);

    return images;
  } catch (const cv::Exception &e) {
    imageIOLogger->error("OpenCV exception in loadImages: {}", e.what());
    return {};
  } catch (const std::exception &e) {
    imageIOLogger->error("Exception in loadImages: {}", e.what());
    return {};
  }
}

auto saveImage(PathLike auto &&filename, const cv::Mat &image,
               int quality) noexcept -> std::expected<void, ImageIOError> {

  const fs::path filepath{std::forward<decltype(filename)>(filename)};

  try {
    imageIOLogger->info("Starting image save: {}", filepath.string());
    imageIOLogger->info("Image properties: {}x{}, {} channels, type={}",
                        image.cols, image.rows, image.channels(), image.type());

    // Check image
    if (image.empty()) {
      imageIOLogger->error("Cannot save empty image: {}", filepath.string());
      return std::unexpected(ImageIOError::EmptyImage);
    }

    // Ensure parent directory exists
    std::error_code ec;
    const auto parent_path = filepath.parent_path();
    if (!parent_path.empty() && !fs::exists(parent_path, ec)) {
      if (!fs::create_directories(parent_path, ec) || ec) {
        imageIOLogger->error("Failed to create directory: {} - {}",
                             parent_path.string(), ec.message());
        return std::unexpected(ImageIOError::WriteError);
      }
    }

    // Set format-specific parameters
    std::vector<int> params;
    std::string ext = filepath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (ext == ".jpg" || ext == ".jpeg") {
      params = {cv::IMWRITE_JPEG_QUALITY, std::clamp(quality, 0, 100)};
    } else if (ext == ".png") {
      params = {cv::IMWRITE_PNG_COMPRESSION, std::clamp(quality / 10, 0, 9)};
    } else if (ext == ".webp") {
      params = {cv::IMWRITE_WEBP_QUALITY, std::clamp(quality, 0, 100)};
    }

    auto start = std::chrono::high_resolution_clock::now();
    bool success = cv::imwrite(filepath.string(), image, params);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    if (!success) {
      imageIOLogger->error("Failed to save image: {} ({}ms)", filepath.string(),
                           duration);
      return std::unexpected(ImageIOError::WriteError);
    }

    imageIOLogger->info("Successfully saved image: {} ({}ms)",
                        filepath.string(), duration);

    try {
      auto fileSize = fs::file_size(filepath, ec);
      if (!ec) {
        imageIOLogger->info("File size: {} bytes", fileSize);
      }
    } catch (...) {
      // Ignore file size errors
    }

    return {};
  } catch (const cv::Exception &e) {
    imageIOLogger->error("OpenCV exception in saveImage: {}", e.what());
    return std::unexpected(ImageIOError::WriteError);
  } catch (const std::exception &e) {
    imageIOLogger->error("Exception in saveImage: {}", e.what());
    return std::unexpected(ImageIOError::WriteError);
  }
}

auto saveMatTo8BitJpg(const cv::Mat &image, PathLike auto &&output_path,
                      int quality) noexcept
    -> std::expected<void, ImageIOError> {

  const fs::path filepath{std::forward<decltype(output_path)>(output_path)};

  try {
    imageIOLogger->info("Converting image to 8-bit JPG: {}x{}", image.cols,
                        image.rows);

    if (image.empty()) {
      imageIOLogger->error("Input image is empty");
      return std::unexpected(ImageIOError::EmptyImage);
    }

    imageIOLogger->info("Input image: type={}, depth={}, channels={}",
                        image.type(), image.depth(), image.channels());

    cv::Mat outputImage;

    // Handle different input depths
    if (image.depth() == CV_8U) {
      // Already 8-bit, just normalize if needed
      if (image.channels() == 1) {
        normalizeWithSIMD<uint8_t>(image, outputImage);
      } else {
        outputImage = image.clone();
      }
    } else if (image.depth() == CV_16U) {
      // 16-bit to 8-bit with proper scaling
      normalizeWithSIMD<uint16_t>(image, outputImage);
    } else if (image.depth() == CV_32F || image.depth() == CV_64F) {
      // Float to 8-bit
      normalizeWithSIMD<float>(image, outputImage);
    } else {
      imageIOLogger->error("Unsupported image depth for JPEG conversion: {}",
                           image.depth());
      return std::unexpected(ImageIOError::UnsupportedFormat);
    }

    // Configure JPEG parameters
    std::vector<int> compressionParams = {cv::IMWRITE_JPEG_QUALITY,
                                          std::clamp(quality, 0, 100)};

    // Save the image with quality parameters
    return saveImage(filepath, outputImage, quality);
  } catch (const cv::Exception &e) {
    imageIOLogger->error("OpenCV exception in saveMatTo8BitJpg: {}", e.what());
    return std::unexpected(ImageIOError::WriteError);
  } catch (const std::exception &e) {
    imageIOLogger->error("Exception in saveMatTo8BitJpg: {}", e.what());
    return std::unexpected(ImageIOError::WriteError);
  }
}

auto saveMatTo16BitPng(const cv::Mat &image, PathLike auto &&output_path,
                       int compression_level) noexcept
    -> std::expected<void, ImageIOError> {

  const fs::path filepath{std::forward<decltype(output_path)>(output_path)};

  try {
    imageIOLogger->info("Converting image to 16-bit PNG: {}x{}", image.cols,
                        image.rows);

    if (image.empty()) {
      imageIOLogger->error("Input image is empty");
      return std::unexpected(ImageIOError::EmptyImage);
    }

    cv::Mat outputImage;

    // Optimal 16-bit conversion based on input depth
    if (image.depth() == CV_8U) {
      // Scale 8-bit to 16-bit preserving full dynamic range
      image.convertTo(outputImage, CV_16U, 256.0);
    } else if (image.depth() == CV_16U) {
      // Already 16-bit
      outputImage = image.clone();
    } else if (image.depth() == CV_32F || image.depth() == CV_64F) {
      // Find min/max for optimal scaling
      double minVal, maxVal;
      cv::minMaxLoc(image, &minVal, &maxVal);

      double scale = 65535.0 / (maxVal - minVal);
      double shift = -minVal * scale;
      image.convertTo(outputImage, CV_16U, scale, shift);
    } else {
      imageIOLogger->error("Unsupported image depth for 16-bit PNG: {}",
                           image.depth());
      return std::unexpected(ImageIOError::UnsupportedFormat);
    }

    // Configure PNG compression
    auto clamped_level = std::clamp(compression_level, 0, 9);
    std::vector<int> compressionParams = {cv::IMWRITE_PNG_COMPRESSION,
                                          clamped_level};

    return saveImage(filepath, outputImage, clamped_level * 10);
  } catch (const cv::Exception &e) {
    imageIOLogger->error("OpenCV exception in saveMatTo16BitPng: {}", e.what());
    return std::unexpected(ImageIOError::WriteError);
  } catch (const std::exception &e) {
    imageIOLogger->error("Exception in saveMatTo16BitPng: {}", e.what());
    return std::unexpected(ImageIOError::WriteError);
  }
}

auto saveMatToFits(const cv::Mat &image, PathLike auto &&output_path,
                   const std::map<std::string, std::string> &metadata) noexcept
    -> std::expected<void, ImageIOError> {

  const fs::path filepath{std::forward<decltype(output_path)>(output_path)};

  try {
    imageIOLogger->info("Converting image to FITS: {}x{}", image.cols,
                        image.rows);

    if (image.empty()) {
      imageIOLogger->error("Input image is empty");
      return std::unexpected(ImageIOError::EmptyImage);
    }

    // Ensure grayscale image
    cv::Mat grayImage;
    if (image.channels() == 3) {
      cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
      grayImage = image.clone();
    }

    // Convert to 16-bit if not already
    cv::Mat fitsMat;
    if (grayImage.depth() != CV_16U) {
      double minVal, maxVal;
      cv::minMaxLoc(grayImage, &minVal, &maxVal);

      // Scale to full 16-bit range
      double scale =
          65535.0 / (maxVal - minVal + 1e-9); // Prevent division by zero
      grayImage.convertTo(fitsMat, CV_16U, scale, -minVal * scale);
    } else {
      fitsMat = grayImage;
    }

    // FITS specific parameters
    const long NAXES[2] = {static_cast<long>(fitsMat.cols),
                           static_cast<long>(fitsMat.rows)};

    // Validate dimensions
    if (NAXES[0] <= 0 || NAXES[1] <= 0 ||
        NAXES[0] > std::numeric_limits<long>::max() / NAXES[1]) {
      imageIOLogger->error("Invalid image dimensions for FITS: {}x{}", NAXES[0],
                           NAXES[1]);
      return std::unexpected(ImageIOError::InvalidFormat);
    }

    // Use our RAII guard
    FitsFileGuard fitsGuard;
    int &status = fitsGuard.status();
    fitsfile *fptr = nullptr;

    // Create FITS file with overwrite (! prefix)
    std::string fits_path = "!" + filepath.string();
    if (fits_create_file(&fptr, fits_path.c_str(), &status)) {
      imageIOLogger->error("Cannot create FITS file: {} - {}",
                           filepath.string(), fitsGuard.error_msg());
      return std::unexpected(ImageIOError::WriteError);
    }

    fitsGuard.reset(fptr);

    // Create image structure
    if (fits_create_img(fptr, SHORT_IMG, 2, const_cast<long *>(NAXES),
                        &status)) {
      imageIOLogger->error("Cannot create FITS image structure - {}",
                           fitsGuard.error_msg());
      return std::unexpected(ImageIOError::WriteError);
    }

    // Write metadata if provided
    for (const auto &[key, value] : metadata) {
      if (key.length() <= FLEN_KEYWORD) {
        // Write string value
        if (fits_update_key(fptr, TSTRING, key.c_str(),
                            const_cast<char *>(value.c_str()), nullptr,
                            &status)) {
          imageIOLogger->warn("Failed to write metadata key '{}': {}", key,
                              fitsGuard.error_msg());
          status = 0; // Reset status and continue with next keys
        }
      } else {
        imageIOLogger->warn("Skipping metadata key '{}': name too long", key);
      }
    }

    // Add creation timestamp
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_time);

    char date_str[FLEN_VALUE];
    std::strftime(date_str, sizeof(date_str), "%Y-%m-%dT%H:%M:%S", &now_tm);

    if (fits_update_key(fptr, TSTRING, "DATE", date_str,
                        "File creation date (YYYY-MM-DDThh:mm:ss)", &status)) {
      imageIOLogger->warn("Failed to write DATE key: {}",
                          fitsGuard.error_msg());
      status = 0;
    }

    // Write pixel data
    if (fits_write_img(fptr, TSHORT, 1, fitsMat.total(), fitsMat.ptr<short>(),
                       &status)) {
      imageIOLogger->error("Cannot write FITS image data - {}",
                           fitsGuard.error_msg());
      return std::unexpected(ImageIOError::WriteError);
    }

    // The FitsFileGuard destructor will close the file safely

    if (!fitsGuard.ok()) {
      imageIOLogger->error("FITS error: {}", fitsGuard.error_msg());
      return std::unexpected(ImageIOError::WriteError);
    }

    imageIOLogger->info("Successfully saved FITS file: {}", filepath.string());
    return {};
  } catch (const cv::Exception &e) {
    imageIOLogger->error("OpenCV exception in saveMatToFits: {}", e.what());
    return std::unexpected(ImageIOError::WriteError);
  } catch (const std::exception &e) {
    imageIOLogger->error("Exception in saveMatToFits: {}", e.what());
    return std::unexpected(ImageIOError::WriteError);
  }
}

auto getFitsMetadata(PathLike auto &&filepath) noexcept
    -> std::expected<std::map<std::string, std::string>, ImageIOError> {

  const fs::path file_path{std::forward<decltype(filepath)>(filepath)};
  std::map<std::string, std::string> metadata;

  try {
    // Check file exists
    std::error_code ec;
    if (!fs::exists(file_path, ec) || ec) {
      imageIOLogger->error("FITS file does not exist: {}", file_path.string());
      return std::unexpected(ImageIOError::FileNotFound);
    }

    // Check extension
    if (!isFitsFile(file_path)) {
      imageIOLogger->error("Not a FITS file: {}", file_path.string());
      return std::unexpected(ImageIOError::InvalidFormat);
    }

    // Use RAII guard
    FitsFileGuard fitsGuard;
    int &status = fitsGuard.status();
    fitsfile *fptr = nullptr;

    // Open FITS file
    if (fits_open_file(&fptr, file_path.string().c_str(), READONLY, &status)) {
      imageIOLogger->error("Cannot open FITS file: {}", file_path.string());
      return std::unexpected(ImageIOError::ReadError);
    }

    fitsGuard.reset(fptr);

    // Get number of keys in header
    int nkeys;
    if (fits_get_hdrspace(fptr, &nkeys, nullptr, &status)) {
      imageIOLogger->error("Cannot get FITS header info count");
      return std::unexpected(ImageIOError::MetadataError);
    }

    // Extract header keys and values
    for (int i = 1; i <= nkeys; i++) {
      char card[FLEN_CARD];
      char keyname[FLEN_KEYWORD];
      char value[FLEN_VALUE];
      char comment[FLEN_COMMENT];

      // Read record
      if (fits_read_record(fptr, i, card, &status)) {
        imageIOLogger->warn("Cannot read FITS record {}", i);
        status = 0; // Reset status and continue
        continue;
      }

      int name_len = 0; // New parameter for fits_get_keyname
      if (fits_get_keyname(card, keyname, &status, &name_len)) {
        imageIOLogger->warn("Cannot get key name for record {}", i);
        status = 0; // Reset status and continue
        continue;
      }

      // Parse value
      if (fits_parse_value(card, value, comment, &status)) {
        imageIOLogger->warn("Cannot parse value for key {}", keyname);
        status = 0; // Reset status and continue
        continue;
      }

      std::string key(keyname);
      std::string val(value);

      // Filter out empty values and special keywords
      if (!key.empty() && key != "SIMPLE" && key != "END" && key != "EXTEND" &&
          !val.empty()) {

        // Remove quote marks from string values
        if (!val.empty() && val.front() == '\'' && val.back() == '\'') {
          val = val.substr(1, val.length() - 2);
        }

        metadata[key] = val;
      }
    }

    imageIOLogger->info("Successfully read FITS metadata: {} items",
                        metadata.size());
    return metadata;
  } catch (const std::exception &e) {
    imageIOLogger->error("Failed to read FITS metadata: {}", e.what());
    return std::unexpected(ImageIOError::MetadataError);
  }
}