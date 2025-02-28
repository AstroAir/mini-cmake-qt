#ifndef IMAGEIO_HPP
#define IMAGEIO_HPP

#include <concepts>
#include <expected>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cv {
class Mat;
}

namespace fs = std::filesystem;

// Concepts for file path validation
template <typename T>
concept PathLike = std::convertible_to<T, std::string_view> ||
                   std::convertible_to<T, fs::path>;

// Error types for better error handling
enum class ImageIOError {
  FileNotFound,
  InvalidFormat,
  ReadError,
  WriteError,
  EmptyImage,
  InvalidOperation,
  UnsupportedFormat,
  MetadataError
};

/**
 * @brief Loads a single image from a file with improved error handling.
 * @param filename The path to the image file.
 * @param flags The flags for loading the image (default is 1).
 * @return The loaded image as a cv::Mat object or an error.
 */
auto loadImage(PathLike auto &&filename, int flags = 1) noexcept
    -> std::expected<cv::Mat, ImageIOError>;

/**
 * @brief Loads all images from a folder with parallelization.
 * @param folder The path to the folder containing the images.
 * @param filenames The list of filenames to load (optional).
 * @param flags The flags for loading the images (default is 1).
 * @param max_threads Maximum number of threads to use (0 = hardware
 * concurrency)
 * @return A vector of pairs containing the filename and the loaded image or
 * error info.
 */
auto loadImages(PathLike auto &&folder,
                std::span<const std::string> filenames = {}, int flags = 1,
                unsigned max_threads = 0) noexcept
    -> std::vector<
        std::pair<std::string, std::expected<cv::Mat, ImageIOError>>>;

/**
 * @brief Saves an image to a file with improved error handling.
 * @param filename The path to the output image file.
 * @param image The image to save.
 * @param quality Compression quality for supported formats (0-100)
 * @return Success or specific error.
 */
auto saveImage(PathLike auto &&filename, const cv::Mat &image,
               int quality = 95) noexcept -> std::expected<void, ImageIOError>;

/**
 * @brief Saves a cv::Mat image as an 8-bit JPG file.
 * @param image The image to save.
 * @param output_path The path to the output JPG file.
 * @param quality JPEG quality (0-100).
 * @return Success or specific error.
 */
auto saveMatTo8BitJpg(const cv::Mat &image, PathLike auto &&output_path,
                      int quality = 95) noexcept
    -> std::expected<void, ImageIOError>;

/**
 * @brief Saves a cv::Mat image as a 16-bit PNG file.
 * @param image The image to save.
 * @param output_path The path to the output PNG file.
 * @param compression_level PNG compression level (0-9).
 * @return Success or specific error.
 */
auto saveMatTo16BitPng(const cv::Mat &image, PathLike auto &&output_path,
                       int compression_level = 9) noexcept
    -> std::expected<void, ImageIOError>;

/**
 * @brief Saves a cv::Mat image as a FITS file.
 * @param image The image to save.
 * @param output_path The path to the output FITS file.
 * @param metadata Optional metadata to include in the FITS header.
 * @return Success or specific error.
 */
auto saveMatToFits(
    const cv::Mat &image, PathLike auto &&output_path,
    const std::map<std::string, std::string> &metadata = {}) noexcept
    -> std::expected<void, ImageIOError>;

/**
 * @brief Retrieves metadata from a FITS image file.
 * @param filepath The path to the FITS file.
 * @return A map containing the metadata key-value pairs or an error.
 */
auto getFitsMetadata(PathLike auto &&filepath) noexcept
    -> std::expected<std::map<std::string, std::string>, ImageIOError>;

// String representation for ImageIOError
std::string_view errorToString(ImageIOError error) noexcept;

#endif // IMAGEIO_HPP