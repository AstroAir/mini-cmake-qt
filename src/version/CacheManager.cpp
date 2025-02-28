#include "CacheManager.hpp"

CacheManager::CacheManager(size_t max_size_mb)
    : max_size_(max_size_mb * 1024 * 1024) {
}

cv::Mat CacheManager::get(const std::string& hash) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (auto it = cache_.find(hash); it != cache_.end()) {
        it->second.last_access = std::chrono::system_clock::now();
        return it->second.image;
    }

    return cv::Mat();
}

void CacheManager::add(const std::string& hash, const cv::Mat& image) const {
    std::lock_guard<std::mutex> lock(mutex_);

    // 检查缓存大小并清理
    cleanup();

    // 添加新条目
    cache_[hash] = {image.clone(), std::chrono::system_clock::now()};
}

void CacheManager::set_max_size(size_t size_mb) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_size_ = size_mb * 1024 * 1024;
    cleanup();
}

void CacheManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

void CacheManager::cleanup() const {
    // 计算当前缓存大小
    size_t current_size = calculate_current_size();

    // 如果超过最大大小，删除最旧的条目
    while (current_size > max_size_ && !cache_.empty()) {
        auto oldest = std::min_element(
            cache_.begin(), cache_.end(),
            [](const auto& a, const auto& b) {
                return a.second.last_access < b.second.last_access;
            }
        );

        current_size -= oldest->second.image.total() * oldest->second.image.elemSize();
        cache_.erase(oldest);
    }
}

size_t CacheManager::calculate_current_size() const {
    size_t total_size = 0;
    for (const auto& [_, entry] : cache_) {
        total_size += entry.image.total() * entry.image.elemSize();
    }
    return total_size;
}
