#include "VersionControl.hpp"
#include <future>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

DiffResult ImageVersionControl::compare_images(const cv::Mat &img1,
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

void ImageVersionControl::process_image_blocks(
    cv::Mat &image, const std::function<void(cv::Mat &)> &processor) const {
  const int block_size = 64; // 分块大小
  std::vector<std::future<void>> futures;

  for (int i = 0; i < image.rows; i += block_size) {
    for (int j = 0; j < image.cols; j += block_size) {
      futures.push_back(std::async(std::launch::async, [&, i, j]() {
        const int rows = std::min(block_size, image.rows - i);
        const int cols = std::min(block_size, image.cols - j);

        cv::Rect roi(j, i, cols, rows);
        cv::Mat block = image(roi);
        processor(block);
      }));
    }
  }

  for (auto &future : futures) {
    future.wait();
  }
}
