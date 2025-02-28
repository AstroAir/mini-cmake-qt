#include "VersionControl.hpp"
#include <fstream>
#include <spdlog/spdlog.h>

void ImageVersionControl::add_metadata(
    std::string_view commit_hash,
    const std::unordered_map<std::string, std::string> &metadata) {
  const auto metadata_path = metadata_dir_ / std::string(commit_hash);
  std::ofstream file(metadata_path);

  for (const auto &[key, value] : metadata) {
    file << key << ": " << value << '\n';
  }

  spdlog::info("Added metadata for commit: {}", commit_hash);
}

std::unordered_map<std::string, std::string>
ImageVersionControl::get_metadata(std::string_view commit_hash) const {
  std::unordered_map<std::string, std::string> metadata;
  const auto metadata_path = metadata_dir_ / std::string(commit_hash);

  if (!fs::exists(metadata_path)) {
    return metadata; // 返回空映射
  }

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

void ImageVersionControl::update_metadata(
    std::string_view commit_hash,
    const std::unordered_map<std::string, std::string> &metadata) {
  const auto metadata_path = metadata_dir_ / std::string(commit_hash);

  // 如果已存在元数据，先读取现有数据
  std::unordered_map<std::string, std::string> existing_metadata;
  if (fs::exists(metadata_path)) {
    existing_metadata = get_metadata(commit_hash);
  }

  // 合并新的元数据
  for (const auto &[key, value] : metadata) {
    existing_metadata[key] = value;
  }

  // 写入更新后的元数据
  std::ofstream file(metadata_path);
  for (const auto &[key, value] : existing_metadata) {
    file << key << ": " << value << '\n';
  }

  spdlog::info("Updated metadata for commit: {}", commit_hash);
}

std::vector<CommitInfo> ImageVersionControl::get_history() const {
  return commits();
}

CommitInfo
ImageVersionControl::commit_info(std::string_view commit_hash) const {
  const auto commit_path = objects_dir / (std::string(commit_hash) + ".commit");
  if (!fs::exists(commit_path)) {
    throw std::runtime_error("Commit not found: " + std::string(commit_hash));
  }

  CommitInfo info;
  std::ifstream file(commit_path);
  std::string line;

  while (std::getline(file, line)) {
    if (line.starts_with("hash: ")) {
      info.hash = line.substr(6);
    } else if (line.starts_with("author: ")) {
      info.author = line.substr(8);
    } else if (line.starts_with("parent: ")) {
      info.parent_hash = line.substr(8);
    } else if (line.starts_with("date: ")) {
      info.date = line.substr(6);
    } else if (line.starts_with("message: ")) {
      info.message = line.substr(9);
    }
  }

  // 获取关联的元数据
  try {
    auto metadata = get_metadata(commit_hash);
    if (metadata.contains("message")) {
      info.message = metadata["message"];
    }
  } catch (const std::exception &e) {
    spdlog::warn("Failed to get metadata for commit {}: {}", commit_hash,
                 e.what());
  }

  return info;
}

std::vector<CommitInfo> ImageVersionControl::commits() const {
  std::vector<CommitInfo> result;

  // 遍历objects目录
  for (const auto &entry : fs::directory_iterator(objects_dir)) {
    if (entry.path().extension() == ".commit") {
      try {
        // 获取提交哈希（文件名去掉.commit后缀）
        std::string hash = entry.path().stem().string();
        result.push_back(commit_info(hash));
      } catch (const std::exception &e) {
        spdlog::error("Failed to read commit file {}: {}",
                      entry.path().string(), e.what());
      }
    }
  }

  // 按日期排序，最新的在前
  std::sort(result.begin(), result.end(),
            [](const CommitInfo &a, const CommitInfo &b) {
              return std::stoll(a.date) > std::stoll(b.date);
            });

  return result;
}