#include "VersionControl.hpp"
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

void ImageVersionControl::create_branch(std::string_view branch_name) {
  const Branch branch{std::string(branch_name), parent_hash_,
                      std::chrono::system_clock::now()};
  save_branch_info(branch);
  spdlog::info("Created branch: {}", branch_name);
}

void ImageVersionControl::switch_branch(std::string_view branch_name) {
  const auto branch = load_branch_info(branch_name);
  if (branch.head_commit.empty()) {
    throw std::runtime_error("Branch not found");
  }

  current_branch_ = branch.name;
  parent_hash_ = branch.head_commit;
  spdlog::info("Switched to branch: {}", branch_name);
}

std::vector<std::string> ImageVersionControl::list_branches() const {
  std::vector<std::string> branches;
  for (const auto &entry : fs::directory_iterator(branches_dir_)) {
    if (entry.is_regular_file() && entry.path().extension() == ".branch") {
      branches.push_back(entry.path().stem().string());
    }
  }
  return branches;
}

void ImageVersionControl::save_branch_info(const Branch &branch) const {
  const auto branch_path = branches_dir_ / (branch.name + ".branch");
  std::ofstream file(branch_path);

  file << "name: " << branch.name << '\n'
       << "head: " << branch.head_commit << '\n'
       << "created: " << std::chrono::system_clock::to_time_t(branch.created_at)
       << '\n';
}

Branch
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

void ImageVersionControl::update_branch_head(std::string_view branch_name,
                                             std::string_view commit_hash) {
  const auto branch_path =
      branches_dir_ / (std::string(branch_name) + ".branch");

  // 首先检查分支是否存在
  if (!fs::exists(branch_path)) {
    throw std::runtime_error("Branch not found: " + std::string(branch_name));
  }

  // 读取现有分支信息
  Branch branch = load_branch_info(branch_name);

  // 更新head commit
  branch.head_commit = std::string(commit_hash);

  // 保存更新后的分支信息
  save_branch_info(branch);

  spdlog::info("Updated branch {} head to commit {}", branch_name, commit_hash);
}

void ImageVersionControl::merge_branch(std::string_view source_branch) {
  // 获取源分支信息
  const auto source = load_branch_info(source_branch);
  if (source.head_commit.empty()) {
    throw std::runtime_error("Source branch not found: " +
                             std::string(source_branch));
  }

  // 获取当前分支信息
  const auto current = load_branch_info(current_branch_);
  if (current.head_commit.empty()) {
    throw std::runtime_error("Current branch not found: " + current_branch_);
  }

  // 找到共同祖先
  auto source_history = build_commit_chain(source.head_commit);
  auto current_history = build_commit_chain(current.head_commit);

  std::string common_ancestor;
  for (const auto &hash : source_history) {
    if (std::find(current_history.begin(), current_history.end(), hash) !=
        current_history.end()) {
      common_ancestor = hash;
      break;
    }
  }

  if (common_ancestor.empty()) {
    throw std::runtime_error("No common ancestor found between branches");
  }

  // 获取三个版本的图像
  cv::Mat base_image = checkout(common_ancestor);
  cv::Mat source_image = checkout(source.head_commit);
  cv::Mat current_image = checkout(current.head_commit);

  // 执行三路合并
  cv::Mat merged = merge_images(base_image, current_image, source_image);

  // 检查是否有冲突
  cv::Mat conflicts =
      find_conflict_regions(base_image, current_image, source_image);
  cv::Mat gray_conflicts;
  cv::cvtColor(conflicts, gray_conflicts, cv::COLOR_BGR2GRAY);

  if (cv::countNonZero(gray_conflicts) > 0) {
    // 如果有冲突，将合并结果和冲突信息保存为临时文件
    const auto temp_dir = repo_root / "temp";
    fs::create_directories(temp_dir);

    cv::imwrite((temp_dir / "merged.png").string(), merged);
    cv::imwrite((temp_dir / "conflicts.png").string(), conflicts);

    throw std::runtime_error(
        "Merge conflicts detected. Resolve conflicts in temp directory");
  }

  // 如果没有冲突，提交合并结果
  std::string merge_message = "Merge branch '" + std::string(source_branch) +
                              "' into " + current_branch_;
  commit(merged, "Merge Bot");

  // 更新元数据
  std::unordered_map<std::string, std::string> metadata;
  metadata["merge_source"] = std::string(source_branch);
  metadata["merge_target"] = current_branch_;
  metadata["merge_base"] = common_ancestor;
  metadata["message"] = merge_message;

  add_metadata(parent_hash_, metadata);

  spdlog::info("Successfully merged {} into {}", source_branch,
               current_branch_);
}
