#pragma once

#include <filesystem>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openssl/sha.h>
#include <span>
#include <vector>
#include <zlib.h>
#include <unordered_map>
#include <chrono>
#include <functional>

namespace fs = std::filesystem;

/**
 * @class ImageVersionControl
 * @brief Class for managing version control of images.
 *
 * This class provides functionality to commit, checkout, and manage versions of
 * images.
 */
class ImageVersionControl {
public:
  /**
   * @brief Constructor for ImageVersionControl.
   * @param repo_path The path to the repository directory.
   */
  explicit ImageVersionControl(const fs::path &repo_path = ".ivc");

  /**
   * @brief Commits an image to the version control repository.
   * @param image The image to commit.
   * @param author The author of the commit.
   */
  void commit(const cv::Mat &image, std::string_view author);

  /**
   * @brief Checks out an image from the version control repository.
   * @param target_hash The hash of the target image to checkout.
   * @return The checked-out image.
   */
  cv::Mat checkout(std::string_view target_hash) const;

  /**
   * @brief Computes a shallow hash of an image.
   * @param image The image to compute the hash for.
   * @return The computed hash as a string.
   */
  [[nodiscard]] std::string compute_shallow_hash(const cv::Mat &image) const;

  // 设置缓存大小（以MB为单位）
  void set_cache_size(size_t size_mb);

  // 分支管理
  void create_branch(std::string_view branch_name);
  void switch_branch(std::string_view branch_name);
  std::vector<std::string> list_branches() const;
  void merge_branch(std::string_view source_branch);

  // 标签管理
  void create_tag(std::string_view tag_name, std::string_view commit_hash);
  void delete_tag(std::string_view tag_name);
  std::vector<std::string> list_tags() const;

  // 历史查询
  struct CommitInfo {
      std::string hash;
      std::string author;
      std::string date;
      std::string message;
      std::string parent_hash;
  };
  std::vector<CommitInfo> get_history() const;
  
  // 图像对比和合并
  struct DiffResult {
      cv::Mat visual_diff;     // 可视化的差异图
      double diff_percentage;   // 差异百分比
      std::vector<cv::Rect> diff_regions; // 差异区域
  };
  DiffResult compare_images(const cv::Mat& img1, const cv::Mat& img2) const;
  cv::Mat merge_images(const cv::Mat& base, const cv::Mat& img1, const cv::Mat& img2) const;

  // 元数据管理
  void add_metadata(std::string_view commit_hash, 
                   const std::unordered_map<std::string, std::string>& metadata);
  std::unordered_map<std::string, std::string> 
  get_metadata(std::string_view commit_hash) const;

private:
  /**
   * @brief Computes the optimized difference between two images.
   * @param base The base image.
   * @param current The current image.
   * @return The computed difference image.
   */
  cv::Mat compute_optimized_diff(const cv::Mat &base,
                                 const cv::Mat &current) const;

  /**
   * @brief Saves a compressed difference image to the repository.
   * @param diff The difference image.
   * @param hash The hash of the difference image.
   */
  void save_compressed_diff(const cv::Mat &diff, std::string_view hash) const;

  /**
   * @brief Loads an image from the repository.
   * @param hash The hash of the image to load.
   * @return The loaded image.
   */
  cv::Mat load_image(std::string_view hash) const;

  /**
   * @brief Applies a difference image to a base image.
   * @param base The base image.
   * @param diff The difference image.
   */
  void apply_diff(cv::Mat &base, const cv::Mat &diff) const;

  /**
   * @brief Gets the file path for an object in the repository.
   * @param hash The hash of the object.
   * @return The file path of the object.
   */
  fs::path get_object_path(std::string_view hash) const;

  /**
   * @brief Compresses data using zlib.
   * @param data The data to compress.
   * @return A pair containing the compressed data and its size.
   */
  static std::pair<std::vector<char>, uint32_t>
  zlib_compress(const cv::Mat &data);

  /**
   * @brief Converts bytes to a hexadecimal string.
   * @param bytes The bytes to convert.
   * @return The hexadecimal string.
   */
  static std::string bytes_to_hex(std::span<const unsigned char> bytes);

  /**
   * @brief Saves a full image to the repository.
   * @param image The image to save.
   * @param hash The hash of the image.
   */
  void save_full_image(const cv::Mat &image, std::string_view hash) const;

  /**
   * @brief Creates a commit object in the repository.
   * @param hash The hash of the commit.
   * @param author The author of the commit.
   */
  void create_commit_object(std::string_view hash,
                            std::string_view author) const;

  /**
   * @brief Builds the commit chain for a target hash.
   * @param target_hash The target hash.
   * @return A vector of strings representing the commit chain.
   */
  std::vector<std::string>
  build_commit_chain(std::string_view target_hash) const;

  /**
   * @brief Loads a compressed difference image from the repository.
   * @param hash The hash of the difference image.
   * @return The loaded difference image.
   */
  cv::Mat load_compressed_diff(std::string_view hash) const;

  /**
   * @brief Handles errors by logging the exception message.
   * @param e The exception to handle.
   */
  void handle_error(const std::exception &e) const;

  /**
   * @brief 使用分块并行处理计算图像差异
   */
  cv::Mat compute_parallel_diff(const cv::Mat &base, const cv::Mat &current) const;

  /**
   * @brief 对图像进行分块处理
   */
  void process_image_blocks(cv::Mat &image, 
                            const std::function<void(cv::Mat&)>& processor) const;

  /**
   * @brief LRU缓存最近使用的图像
   */
  struct CacheEntry {
      cv::Mat image;
      std::chrono::system_clock::time_point last_access;
  };

  mutable std::unordered_map<std::string, CacheEntry> image_cache_;
  mutable std::mutex cache_mutex_;
  size_t max_cache_size_ = 1024 * 1024 * 512; // 默认512MB

  /**
   * @brief 从缓存中获取图像
   */
  cv::Mat get_from_cache(const std::string& hash) const;

  /**
   * @brief 将图像添加到缓存
   */
  void add_to_cache(const std::string& hash, const cv::Mat& image) const;

  /**
   * @brief 清理过期缓存
   */
  void cleanup_cache() const;

  struct Branch {
      std::string name;
      std::string head_commit;
      std::chrono::system_clock::time_point created_at;
  };

  struct Tag {
      std::string name;
      std::string commit_hash;
      std::string message;
      std::chrono::system_clock::time_point created_at;
  };

  // 新增私有方法
  void save_branch_info(const Branch& branch) const;
  void save_tag_info(const Tag& tag) const;
  Branch load_branch_info(std::string_view branch_name) const;
  Tag load_tag_info(std::string_view tag_name) const;
  void update_branch_head(std::string_view branch_name, std::string_view commit_hash);
  cv::Mat find_conflict_regions(const cv::Mat& base, const cv::Mat& img1, const cv::Mat& img2) const;

  fs::path repo_root;               ///< Root path of the repository.
  fs::path objects_dir;             ///< Directory for storing objects.
  std::string parent_hash_;         ///< Hash of the parent commit.
  mutable std::mutex commit_mutex_; ///< Mutex for synchronizing commits.
  std::string current_branch_ = "main";
  fs::path branches_dir_;
  fs::path tags_dir_;
  fs::path metadata_dir_;
};