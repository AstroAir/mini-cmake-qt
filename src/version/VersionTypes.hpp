#pragma once

#include <string>
#include <chrono>
#include <opencv2/core.hpp>
#include <vector>

/**
 * @struct CommitInfo
 * @brief 保存提交信息的结构体
 */
struct CommitInfo {
    std::string hash;
    std::string author;
    std::string date;
    std::string message;
    std::string parent_hash;
};

/**
 * @struct DiffResult
 * @brief 图像比较结果的结构体
 */
struct DiffResult {
    cv::Mat visual_diff;        // 可视化的差异图
    double diff_percentage;     // 差异百分比
    std::vector<cv::Rect> diff_regions; // 差异区域
};

/**
 * @struct Branch
 * @brief 分支信息的结构体
 */
struct Branch {
    std::string name;
    std::string head_commit;
    std::chrono::system_clock::time_point created_at;
};

/**
 * @struct Tag
 * @brief 标签信息的结构体
 */
struct Tag {
    std::string name;
    std::string commit_hash;
    std::string message;
    std::chrono::system_clock::time_point created_at;
};
