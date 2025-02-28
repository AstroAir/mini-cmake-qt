#pragma once

#include <opencv2/core.hpp>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <string>

/**
 * @class CacheManager
 * @brief 管理图像缓存的类
 *
 * 使用LRU策略管理图像缓存，提高频繁访问的图像的加载速度
 */
class CacheManager {
public:
    /**
     * @brief 构造函数
     * @param max_size_mb 最大缓存大小（MB）
     */
    explicit CacheManager(size_t max_size_mb = 512);

    /**
     * @brief 从缓存获取图像
     * @param hash 图像哈希
     * @return 缓存的图像，如果不存在则返回空Mat
     */
    cv::Mat get(const std::string& hash) const;

    /**
     * @brief 添加图像到缓存
     * @param hash 图像哈希
     * @param image 要缓存的图像
     */
    void add(const std::string& hash, const cv::Mat& image) const;

    /**
     * @brief 设置缓存大小限制
     * @param size_mb 缓存大小（MB）
     */
    void set_max_size(size_t size_mb);

    /**
     * @brief 清空缓存
     */
    void clear();

private:
    /**
     * @brief 缓存项结构
     */
    struct CacheEntry {
        cv::Mat image;
        std::chrono::system_clock::time_point last_access;
    };

    /**
     * @brief 清理过期缓存项
     */
    void cleanup() const;

    /**
     * @brief 计算当前缓存大小
     * @return 当前缓存大小（字节）
     */
    size_t calculate_current_size() const;

    mutable std::unordered_map<std::string, CacheEntry> cache_;
    mutable std::mutex mutex_;
    size_t max_size_; // 最大缓存大小（字节）
};
