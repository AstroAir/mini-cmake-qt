#include "VersionControl.hpp"
#include <fstream>
#include <spdlog/spdlog.h>

void ImageVersionControl::create_tag(std::string_view tag_name,
                                     std::string_view commit_hash) {
  // 验证提交是否存在
  const auto commit_path = objects_dir / (std::string(commit_hash) + ".commit");
  if (!fs::exists(commit_path)) {
    throw std::runtime_error("Commit not found: " + std::string(commit_hash));
  }

  // 创建标签
  const Tag tag{std::string(tag_name), std::string(commit_hash), "",
                std::chrono::system_clock::now()};
  save_tag_info(tag);

  spdlog::info("Created tag: {} pointing to commit: {}", tag_name, commit_hash);
}

void ImageVersionControl::delete_tag(std::string_view tag_name) {
  const auto tag_path = tags_dir_ / (std::string(tag_name) + ".tag");

  // 检查标签是否存在
  if (!fs::exists(tag_path)) {
    throw std::runtime_error("Tag not found: " + std::string(tag_name));
  }

  // 删除标签文件
  fs::remove(tag_path);
  spdlog::info("Deleted tag: {}", tag_name);
}

std::vector<std::string> ImageVersionControl::list_tags() const {
  std::vector<std::string> tags;
  for (const auto &entry : fs::directory_iterator(tags_dir_)) {
    if (entry.is_regular_file() && entry.path().extension() == ".tag") {
      tags.push_back(entry.path().stem().string());
    }
  }
  return tags;
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

Tag ImageVersionControl::load_tag_info(std::string_view tag_name) const {
  const auto tag_path = tags_dir_ / (std::string(tag_name) + ".tag");
  if (!fs::exists(tag_path)) {
    throw std::runtime_error("Tag not found");
  }

  Tag tag;
  std::ifstream file(tag_path);
  std::string line;

  while (std::getline(file, line)) {
    if (line.starts_with("name: ")) {
      tag.name = line.substr(6);
    } else if (line.starts_with("commit: ")) {
      tag.commit_hash = line.substr(8);
    } else if (line.starts_with("message: ")) {
      tag.message = line.substr(9);
    } else if (line.starts_with("created: ")) {
      std::time_t time = std::stoll(line.substr(9));
      tag.created_at = std::chrono::system_clock::from_time_t(time);
    }
  }

  return tag;
}
