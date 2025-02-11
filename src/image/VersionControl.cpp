#include "VersionControl.hpp"

#include <fstream>

#include <future>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

namespace {
std::shared_ptr<spdlog::logger> versionControlLogger =
    spdlog::basic_logger_mt("VersionControlLogger", "logs/versioncontrol.log");
} // namespace

ImageVersionControl::ImageVersionControl(const fs::path &repo_path)
    : repo_root(repo_path), objects_dir(repo_path / "objects") {
  fs::create_directories(objects_dir);
  versionControlLogger->info(
      "Initialized ImageVersionControl with repo path: {}", repo_path.string());
}

void ImageVersionControl::commit(const cv::Mat &image,
                                 std::string_view author) {
  std::scoped_lock lock(commit_mutex_);
  versionControlLogger->info("Committing image by author: {}", author);

  const auto current_hash = compute_shallow_hash(image);
  versionControlLogger->debug("Computed hash for current image: {}",
                              current_hash);

  if (!parent_hash_.empty() && current_hash == parent_hash_) {
    versionControlLogger->info(
        "Current image is identical to the parent image. Skipping commit.");
    return;
  }

  try {
    if (parent_hash_.empty()) {
      save_full_image(image, current_hash);
      versionControlLogger->info("Saved full image with hash: {}",
                                 current_hash);
    } else {
      const auto diff = compute_optimized_diff(load_image(parent_hash_), image);
      save_compressed_diff(diff, current_hash);
      versionControlLogger->info("Saved compressed diff with hash: {}",
                                 current_hash);
    }
    create_commit_object(current_hash, author);
    versionControlLogger->info("Created commit object for hash: {}",
                               current_hash);
    parent_hash_ = current_hash;
  } catch (const std::exception &e) {
    handle_error(e);
    throw;
  }
}

cv::Mat ImageVersionControl::checkout(std::string_view target_hash) const {
  versionControlLogger->info("Checking out image with hash: {}", target_hash);
  auto commit_chain = build_commit_chain(target_hash);
  cv::Mat image;

  for (const auto &hash : commit_chain) {
    if (hash == commit_chain.front()) {
      image = load_image(hash);
      versionControlLogger->info("Loaded base image with hash: {}", hash);
    } else {
      apply_diff(image, load_compressed_diff(hash));
      versionControlLogger->info("Applied diff with hash: {}", hash);
    }
  }
  return image;
}

std::string
ImageVersionControl::compute_shallow_hash(const cv::Mat &image) const {
  std::vector<uchar> buffer;
  cv::imencode(".png", image, buffer);

  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(buffer.data(), buffer.size(), hash);

  return bytes_to_hex(std::span(hash));
}

cv::Mat
ImageVersionControl::compute_optimized_diff(const cv::Mat &base,
                                            const cv::Mat &current) const {
  return compute_parallel_diff(base, current);
}

cv::Mat
ImageVersionControl::compute_parallel_diff(const cv::Mat &base,
                                           const cv::Mat &current) const {
  CV_Assert(base.size() == current.size() && base.type() == current.type());

  cv::Mat diff(base.size(), CV_16SC3);
  const int block_rows = 32; // 分块大小
  const int block_cols = 32;

  std::vector<std::future<void>> futures;

  for (int i = 0; i < base.rows; i += block_rows) {
    for (int j = 0; j < base.cols; j += block_cols) {
      futures.push_back(std::async(std::launch::async, [&, i, j]() {
        const int rows = std::min(block_rows, base.rows - i);
        const int cols = std::min(block_cols, base.cols - j);

        cv::Rect roi(j, i, cols, rows);
        cv::subtract(current(roi), base(roi), diff(roi), cv::noArray(),
                     CV_16SC3);
      }));
    }
  }

  for (auto &future : futures) {
    future.wait();
  }

  return diff;
}

void ImageVersionControl::save_compressed_diff(const cv::Mat &diff,
                                               std::string_view hash) const {
  const auto path = get_object_path(hash);
  std::ofstream file(path, std::ios::binary);

  const auto [compressed, crc] = zlib_compress(diff);
  file.write(compressed.data(), compressed.size());
}

cv::Mat ImageVersionControl::load_image(std::string_view hash) const {
  const auto hash_str = std::string(hash);

  // 尝试从缓存获取
  if (auto cached = get_from_cache(hash_str); !cached.empty()) {
    return cached;
  }

  // 从文件加载
  const auto path = get_object_path(hash);
  if (!fs::exists(path))
    throw std::runtime_error("Object not found");

  cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);

  // 添加到缓存
  add_to_cache(hash_str, image);

  return image;
}

void ImageVersionControl::apply_diff(cv::Mat &base, const cv::Mat &diff) const {
  CV_Assert(base.size() == diff.size() && base.type() == diff.type());

  cv::add(base, diff, base, cv::noArray(), base.type());
}

fs::path ImageVersionControl::get_object_path(std::string_view hash) const {
  return objects_dir / std::string(hash);
}

std::pair<std::vector<char>, uint32_t>
ImageVersionControl::zlib_compress(const cv::Mat &data) {
  const uLong source_size = data.total() * data.elemSize();
  uLongf dest_size = compressBound(source_size);
  std::vector<char> compressed(dest_size);

  if (compress2(reinterpret_cast<Bytef *>(compressed.data()), &dest_size,
                reinterpret_cast<const Bytef *>(data.data), source_size,
                Z_BEST_SPEED) != Z_OK) {
    throw std::runtime_error("Compression failed");
  }
  compressed.resize(dest_size);

  const auto crc = crc32(0L, Z_NULL, 0);
  return {compressed,
          crc32(crc, reinterpret_cast<const Bytef *>(data.data), source_size)};
}

std::string
ImageVersionControl::bytes_to_hex(std::span<const unsigned char> bytes) {
  std::stringstream ss;
  ss << std::hex << std::setfill('0');
  for (auto b : bytes)
    ss << std::setw(2) << static_cast<int>(b);
  return ss.str();
}

void ImageVersionControl::save_full_image(const cv::Mat &image,
                                          std::string_view hash) const {
  const auto path = get_object_path(hash);
  if (!cv::imwrite(path.string(), image)) {
    throw std::runtime_error("Failed to save full image");
  }
}

void ImageVersionControl::create_commit_object(std::string_view hash,
                                               std::string_view author) const {
  const auto commit_path = objects_dir / (std::string(hash) + ".commit");
  std::ofstream commit_file(commit_path);
  if (!commit_file) {
    throw std::runtime_error("Failed to create commit object");
  }

  commit_file << "hash: " << hash << '\n'
              << "author: " << author << '\n'
              << "parent: " << parent_hash_ << '\n'
              << "date: " << std::time(nullptr) << '\n';
}

std::vector<std::string>
ImageVersionControl::build_commit_chain(std::string_view target_hash) const {
  std::vector<std::string> chain;
  std::string current = std::string(target_hash);

  while (!current.empty()) {
    chain.push_back(current);
    const auto commit_path = objects_dir / (current + ".commit");

    if (!fs::exists(commit_path)) {
      break;
    }

    std::ifstream commit_file(commit_path);
    std::string line;
    while (std::getline(commit_file, line)) {
      if (line.starts_with("parent: ")) {
        current = line.substr(8);
        break;
      }
    }
  }

  std::reverse(chain.begin(), chain.end());
  return chain;
}

cv::Mat ImageVersionControl::load_compressed_diff(std::string_view hash) const {
  const auto path = get_object_path(hash);
  if (!fs::exists(path)) {
    throw std::runtime_error("Diff object not found");
  }

  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const auto size = file.tellg();
  file.seekg(0);

  std::vector<char> compressed(size);
  file.read(compressed.data(), size);

  uLongf dest_size = size * 2; // 估计解压后大小
  std::vector<Bytef> decompressed(dest_size);

  while (uncompress(decompressed.data(), &dest_size,
                    reinterpret_cast<const Bytef *>(compressed.data()),
                    compressed.size()) == Z_BUF_ERROR) {
    dest_size *= 2;
    decompressed.resize(dest_size);
  }

  cv::Mat diff(1, dest_size / sizeof(short), CV_16SC3, decompressed.data());
  return diff.clone(); // 创建深拷贝以确保内存安全
}

void ImageVersionControl::handle_error(const std::exception &e) const {
  versionControlLogger->error("Error: {}", e.what());
  // 可以添加更多错误处理逻辑，如日志记录
}

void ImageVersionControl::set_cache_size(size_t size_mb) {
  max_cache_size_ = size_mb * 1024 * 1024;
  cleanup_cache();
}

cv::Mat ImageVersionControl::get_from_cache(const std::string &hash) const {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  if (auto it = image_cache_.find(hash); it != image_cache_.end()) {
    it->second.last_access = std::chrono::system_clock::now();
    return it->second.image;
  }

  return cv::Mat();
}

void ImageVersionControl::add_to_cache(const std::string &hash,
                                       const cv::Mat &image) const {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  // 检查缓存大小并清理
  cleanup_cache();

  // 添加新条目
  image_cache_[hash] = {image.clone(), std::chrono::system_clock::now()};
}

void ImageVersionControl::cleanup_cache() const {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  // 计算当前缓存大小
  size_t current_size = 0;
  for (const auto &[_, entry] : image_cache_) {
    current_size += entry.image.total() * entry.image.elemSize();
  }

  // 如果超过最大大小，删除最旧的条目
  while (current_size > max_cache_size_ && !image_cache_.empty()) {
    auto oldest =
        std::min_element(image_cache_.begin(), image_cache_.end(),
                         [](const auto &a, const auto &b) {
                           return a.second.last_access < b.second.last_access;
                         });

    current_size -=
        oldest->second.image.total() * oldest->second.image.elemSize();
    image_cache_.erase(oldest);
  }
}

void ImageVersionControl::create_branch(std::string_view branch_name) {
  const Branch branch{std::string(branch_name), parent_hash_,
                      std::chrono::system_clock::now()};
  save_branch_info(branch);
  versionControlLogger->info("Created branch: {}", branch_name);
}

void ImageVersionControl::switch_branch(std::string_view branch_name) {
  const auto branch = load_branch_info(branch_name);
  if (branch.head_commit.empty()) {
    throw std::runtime_error("Branch not found");
  }

  current_branch_ = branch.name;
  parent_hash_ = branch.head_commit;
  versionControlLogger->info("Switched to branch: {}", branch_name);
}

std::vector<std::string> ImageVersionControl::list_branches() const {
  std::vector<std::string> branches;
  for (const auto &entry : fs::directory_iterator(branches_dir_)) {
    if (entry.is_regular_file()) {
      branches.push_back(entry.path().stem().string());
    }
  }
  return branches;
}

void ImageVersionControl::create_tag(std::string_view tag_name,
                                     std::string_view commit_hash) {
  const Tag tag{std::string(tag_name), std::string(commit_hash), "",
                std::chrono::system_clock::now()};
  save_tag_info(tag);
  versionControlLogger->info("Created tag: {} pointing to commit: {}", tag_name,
                             commit_hash);
}

ImageVersionControl::DiffResult
ImageVersionControl::compare_images(const cv::Mat &img1,
                                    const cv::Mat &img2) const {
  DiffResult result;

  // 计算差异图
  cv::Mat diff;
  cv::absdiff(img1, img2, diff);

  // 转换为灰度图进行分析
  cv::Mat gray_diff;
  cv::cvtColor(diff, gray_diff, cv::COLOR_BGR2GRAY);

  // 计算差异百分比
  result.diff_percentage =
      (cv::countNonZero(gray_diff) * 100.0) / (gray_diff.rows * gray_diff.cols);

  // 找出差异区域
  cv::Mat binary;
  cv::threshold(gray_diff, binary, 30, 255, cv::THRESH_BINARY);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  for (const auto &contour : contours) {
    result.diff_regions.push_back(cv::boundingRect(contour));
  }

  // 创建可视化差异图
  result.visual_diff = img1.clone();
  cv::drawContours(result.visual_diff, contours, -1, cv::Scalar(0, 0, 255), 2);

  return result;
}

cv::Mat ImageVersionControl::merge_images(const cv::Mat &base,
                                          const cv::Mat &img1,
                                          const cv::Mat &img2) const {
  // 找出冲突区域
  cv::Mat conflicts = find_conflict_regions(base, img1, img2);

  // 创建合并结果
  cv::Mat merged = base.clone();

  // 对于非冲突区域，使用较新的更改
  cv::Mat mask1, mask2;
  cv::absdiff(base, img1, mask1);
  cv::absdiff(base, img2, mask2);

  cv::cvtColor(mask1, mask1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(mask2, mask2, cv::COLOR_BGR2GRAY);

  cv::threshold(mask1, mask1, 30, 255, cv::THRESH_BINARY);
  cv::threshold(mask2, mask2, 30, 255, cv::THRESH_BINARY);

  img1.copyTo(merged, mask1);
  img2.copyTo(merged, mask2);

  // 标记冲突区域
  conflicts.copyTo(merged, conflicts);

  return merged;
}

void ImageVersionControl::add_metadata(
    std::string_view commit_hash,
    const std::unordered_map<std::string, std::string> &metadata) {
  const auto metadata_path = metadata_dir_ / std::string(commit_hash);
  std::ofstream file(metadata_path);

  for (const auto &[key, value] : metadata) {
    file << key << ": " << value << '\n';
  }
}

std::unordered_map<std::string, std::string>
ImageVersionControl::get_metadata(std::string_view commit_hash) const {
  std::unordered_map<std::string, std::string> metadata;
  const auto metadata_path = metadata_dir_ / std::string(commit_hash);

  std::ifstream file(metadata_path);
  std::string line;

  while (std::getline(file, line)) {
    const auto pos = line.find(": ");
    if (pos != std::string::npos) {
      const auto key = line.substr(0, pos);
      const auto value = line.substr(pos + 2);
      metadata[key] = value;
    }
  }

  return metadata;
}

void ImageVersionControl::save_branch_info(const Branch &branch) const {
  const auto branch_path = branches_dir_ / (branch.name + ".branch");
  std::ofstream file(branch_path);

  file << "name: " << branch.name << '\n'
       << "head: " << branch.head_commit << '\n'
       << "created: " << std::chrono::system_clock::to_time_t(branch.created_at)
       << '\n';
}

ImageVersionControl::Branch
ImageVersionControl::load_branch_info(std::string_view branch_name) const {
  const auto branch_path =
      branches_dir_ / (std::string(branch_name) + ".branch");
  if (!fs::exists(branch_path)) {
    throw std::runtime_error("Branch not found");
  }

  Branch branch;
  std::ifstream file(branch_path);
  std::string line;

  while (std::getline(file, line)) {
    if (line.starts_with("name: ")) {
      branch.name = line.substr(6);
    } else if (line.starts_with("head: ")) {
      branch.head_commit = line.substr(6);
    } else if (line.starts_with("created: ")) {
      std::time_t time = std::stoll(line.substr(9));
      branch.created_at = std::chrono::system_clock::from_time_t(time);
    }
  }

  return branch;
}

void ImageVersionControl::save_tag_info(const Tag &tag) const {
  const auto tag_path = tags_dir_ / (tag.name + ".tag");
  std::ofstream file(tag_path);

  file << "name: " << tag.name << '\n'
       << "commit: " << tag.commit_hash << '\n'
       << "message: " << tag.message << '\n'
       << "created: " << std::chrono::system_clock::to_time_t(tag.created_at)
       << '\n';
}

cv::Mat ImageVersionControl::find_conflict_regions(const cv::Mat &base,
                                                   const cv::Mat &img1,
                                                   const cv::Mat &img2) const {
  cv::Mat diff1, diff2;
  cv::absdiff(base, img1, diff1);
  cv::absdiff(base, img2, diff2);

  // 转换为灰度图
  cv::Mat gray1, gray2;
  cv::cvtColor(diff1, gray1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(diff2, gray2, cv::COLOR_BGR2GRAY);

  // 二值化
  cv::Mat binary1, binary2;
  cv::threshold(gray1, binary1, 30, 255, cv::THRESH_BINARY);
  cv::threshold(gray2, binary2, 30, 255, cv::THRESH_BINARY);

  // 寻找重叠的冲突区域
  cv::Mat conflicts;
  cv::bitwise_and(binary1, binary2, conflicts);

  // 创建彩色冲突标记
  cv::Mat colored_conflicts = cv::Mat::zeros(base.size(), base.type());
  colored_conflicts.setTo(cv::Scalar(0, 0, 255), conflicts); // 红色标记冲突

  return colored_conflicts;
}