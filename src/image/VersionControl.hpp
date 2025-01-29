
#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openssl/sha.h>
#include <span>
#include <vector>
#include <zlib.h>


namespace fs = std::filesystem;

class ImageVersionControl {
public:
  explicit ImageVersionControl(const fs::path &repo_path = ".ivc");

  void commit(const cv::Mat &image, std::string_view author);
  cv::Mat checkout(std::string_view target_hash) const;
  [[nodiscard]] std::string compute_shallow_hash(const cv::Mat &image) const;

private:
  cv::Mat compute_optimized_diff(const cv::Mat &base,
                                 const cv::Mat &current) const;
  void save_compressed_diff(const cv::Mat &diff, std::string_view hash) const;
  cv::Mat load_image(std::string_view hash) const;
  void apply_diff(cv::Mat &base, const cv::Mat &diff) const;
  fs::path get_object_path(std::string_view hash) const;
  static std::pair<std::vector<char>, uint32_t>
  zlib_compress(const cv::Mat &data);
  static std::string bytes_to_hex(std::span<const unsigned char> bytes);
  void save_full_image(const cv::Mat &image, std::string_view hash) const;
  void create_commit_object(std::string_view hash,
                            std::string_view author) const;
  std::vector<std::string>
  build_commit_chain(std::string_view target_hash) const;
  cv::Mat load_compressed_diff(std::string_view hash) const;
  void handle_error(const std::exception &e) const;

  fs::path repo_root;
  fs::path objects_dir;
  std::string parent_hash_;
  mutable std::mutex commit_mutex_;
};