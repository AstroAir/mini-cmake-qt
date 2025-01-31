#include "VersionControl.hpp"

#include <fstream>

#include <spdlog/spdlog.h>

ImageVersionControl::ImageVersionControl(const fs::path &repo_path)
    : repo_root(repo_path), objects_dir(repo_path / "objects") {
  fs::create_directories(objects_dir);
  spdlog::info("Initialized ImageVersionControl with repo path: {}",
               repo_path.string());
}

void ImageVersionControl::commit(const cv::Mat &image,
                                 std::string_view author) {
  std::scoped_lock lock(commit_mutex_);
  spdlog::info("Committing image by author: {}", author);

  const auto current_hash = compute_shallow_hash(image);
  spdlog::debug("Computed hash for current image: {}", current_hash);

  if (!parent_hash_.empty() && current_hash == parent_hash_) {
    spdlog::info(
        "Current image is identical to the parent image. Skipping commit.");
    return;
  }

  try {
    if (parent_hash_.empty()) {
      save_full_image(image, current_hash);
      spdlog::info("Saved full image with hash: {}", current_hash);
    } else {
      const auto diff = compute_optimized_diff(load_image(parent_hash_), image);
      save_compressed_diff(diff, current_hash);
      spdlog::info("Saved compressed diff with hash: {}", current_hash);
    }
    create_commit_object(current_hash, author);
    spdlog::info("Created commit object for hash: {}", current_hash);
    parent_hash_ = current_hash;
  } catch (const std::exception &e) {
    handle_error(e);
    throw;
  }
}

cv::Mat ImageVersionControl::checkout(std::string_view target_hash) const {
  spdlog::info("Checking out image with hash: {}", target_hash);
  auto commit_chain = build_commit_chain(target_hash);
  cv::Mat image;

  for (const auto &hash : commit_chain) {
    if (hash == commit_chain.front()) {
      image = load_image(hash);
      spdlog::info("Loaded base image with hash: {}", hash);
    } else {
      apply_diff(image, load_compressed_diff(hash));
      spdlog::info("Applied diff with hash: {}", hash);
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
  CV_Assert(base.size() == current.size() && base.type() == current.type());

  cv::Mat diff;
  cv::subtract(current, base, diff, cv::noArray(), CV_16SC3);
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
  const auto path = get_object_path(hash);
  if (!fs::exists(path))
    throw std::runtime_error("Object not found");

  return cv::imread(path.string(), cv::IMREAD_UNCHANGED);
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
  spdlog::error("Error: {}", e.what());
  // 可以添加更多错误处理逻辑，如日志记录
}