#include "../utils/MemoryPool.hpp" // 在文件开头添加内存池相关头文件
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <optional>
#include <string>
#include <vector>


namespace fs = std::filesystem; // 添加命名空间声明

// Forward declarations
class HDRProcessor;
class ImagePreprocessor;
class ToneMapper;

// Custom exceptions
class HDRException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Configuration struct for HDR processing
struct HDRConfig {
  enum class AlignmentMethod { MTB, HOMOGRAPHY, NONE };
  enum class ResponseMethod { DEBEVEC, ROBERTSON };
  enum class MergeMethod { DEBEVEC, MERTENS };

  AlignmentMethod alignment = AlignmentMethod::MTB;
  ResponseMethod response = ResponseMethod::DEBEVEC;
  MergeMethod merge = MergeMethod::DEBEVEC;
  bool use_gpu = false;
  int num_threads = 4;
  float gamma = 2.2f;
  bool auto_exposure = true;
};

// Image metadata
struct ImageMetadata {
  float exposure_time;
  float iso;
  float aperture;
  std::string camera_model;
  std::chrono::system_clock::time_point capture_time;
};

// Result statistics
struct ProcessingStats {
  std::chrono::milliseconds processing_time;
  size_t peak_memory_usage;
  std::string alignment_quality;
  float dynamic_range;
};

class ImagePreprocessor {
public:
  ImagePreprocessor(const cv::Mat &image) : original_image(image) {}

  cv::Mat denoise(float strength = 3.0f) {
    cv::Mat result;
    cv::fastNlMeansDenoisingColored(original_image, result, strength);
    return result;
  }

  cv::Mat adjustExposure(float ev) {
    cv::Mat result;
    original_image.convertTo(result, -1, std::pow(2.0, ev));
    return result;
  }

  cv::Mat correctLens(float k1 = 0.0f, float k2 = 0.0f) {
    // Implement lens correction
    // This is a placeholder for actual implementation
    return original_image.clone();
  }

private:
  cv::Mat original_image;
};

class ToneMapper {
public:
  enum class Method { DRAGO, REINHARD, MANTIUK, CUSTOM };

  ToneMapper(Method method = Method::DRAGO) : method_(method) {}

  cv::Mat process(const cv::Mat &hdr_image, float gamma = 2.2f) {
    void *memory = tone_mapping_pool_.allocate();
    cv::Mat *result = new (memory) cv::Mat();

    switch (method_) {
    case Method::DRAGO: {
      auto tonemap = cv::createTonemapDrago(gamma);
      tonemap->process(hdr_image, *result);
      break;
    }
    case Method::REINHARD: {
      auto tonemap = cv::createTonemapReinhard();
      tonemap->process(hdr_image, *result);
      break;
    }
    case Method::MANTIUK: {
      auto tonemap = cv::createTonemapMantiuk();
      tonemap->process(hdr_image, *result);
      break;
    }
    case Method::CUSTOM:
      *result = customToneMapping(hdr_image, gamma);
      break;
    }

    *result = *result * 255;
    result->convertTo(*result, CV_8UC3);
    tone_mapping_pool_.deallocate(result);
    return *result;
  }

private:
  Method method_;
  MemoryPool<1024 * 1024>
      tone_mapping_pool_; // 1MB blocks for intermediate results

  cv::Mat customToneMapping(const cv::Mat &hdr_image, float gamma) {
    cv::Mat result;
    cv::Mat lum;

    // 转换到对数域
    cv::cvtColor(hdr_image, lum, cv::COLOR_BGR2GRAY);
    cv::log(lum + 1.0f, lum);

    // 计算平均亮度
    cv::Scalar mean_lum = cv::mean(lum);
    float log_mean = static_cast<float>(mean_lum[0]);

    // 应用自适应色调映射
    float key = 0.18f; // 中灰度值
    cv::Mat mapped;
    cv::exp((key * lum - log_mean) / gamma, mapped);

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(hdr_image, channels);

    // 对每个通道应用色调映射
    for (auto &channel : channels) {
      channel = channel.mul(1.0f / (channel + 1.0f));
    }

    // 合并通道
    cv::merge(channels, result);

    return result;
  }
};

// 为HDR处理相关的数据结构添加内存池支持
class HDRProcessor {
private:
  static constexpr size_t IMAGE_BLOCK_SIZE = 1024 * 1024; // 1MB blocks
  static constexpr size_t METADATA_BLOCK_SIZE = sizeof(ImageMetadata);

  MemoryPool<IMAGE_BLOCK_SIZE> image_pool_;
  MemoryPool<METADATA_BLOCK_SIZE> metadata_pool_;

  // 使用标准容器存储图像和元数据
  std::vector<cv::Mat> images_;
  std::vector<ImageMetadata> metadata_;
  HDRConfig config_;
  cv::Mat hdr_result_;
  cv::Mat response_curve_;

public:
  HDRProcessor(const HDRConfig &config = HDRConfig{}) : config_(config) {
    images_.reserve(16);
    metadata_.reserve(16);
  }

  void addImage(const cv::Mat &image, const ImageMetadata &metadata) {
    try {
      // 直接使用标准容器
      images_.push_back(image.clone());
      metadata_.push_back(metadata);
    } catch (...) {
      throw;
    }
  }

  ~HDRProcessor() { clearImages(); }

  void clearImages() {
    for (auto &img : images_) {
      img.release();
    }
    images_.clear();
    metadata_.clear();
  }

  // 添加内存池统计信息方法
  struct PoolStats {
    MemoryPool<IMAGE_BLOCK_SIZE>::Stats image_stats;
    MemoryPool<METADATA_BLOCK_SIZE>::Stats metadata_stats;
  };

  PoolStats getPoolStats() const {
    return PoolStats{image_pool_.get_stats(), metadata_pool_.get_stats()};
  }

  std::optional<ProcessingStats> process() {
    if (images_.empty())
      return std::nullopt;

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
      preprocessImages();
      alignImages();
      calculateResponse();
      mergeImages();

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          end_time - start_time);

      return ProcessingStats{.processing_time = duration,
                             .peak_memory_usage = calculateMemoryUsage(),
                             .alignment_quality = evaluateAlignment(),
                             .dynamic_range = calculateDynamicRange()};
    } catch (const std::exception &e) {
      throw HDRException(std::format("HDR processing failed: {}", e.what()));
    }
  }

  cv::Mat getResult(
      ToneMapper::Method tone_mapping_method = ToneMapper::Method::DRAGO) {
    if (hdr_result_.empty()) {
      throw HDRException("No HDR result available");
    }

    ToneMapper tone_mapper(tone_mapping_method);
    return tone_mapper.process(hdr_result_, config_.gamma);
  }

  void saveHDR(const std::string &path) const {
    if (!cv::imwrite(path, hdr_result_)) {
      throw HDRException("Failed to save HDR image");
    }
  }

  // Add visualization methods
  cv::Mat visualizeExposures() const {
    if (images_.empty())
      return cv::Mat();

    const int padding = 10;     // 图像间距
    const int text_height = 30; // 文本区域高度

    // 计算总宽度和高度
    int total_width =
        images_[0].cols * images_.size() + padding * (images_.size() - 1);
    int total_height = images_[0].rows + text_height;

    // 创建结果图像
    cv::Mat result(total_height, total_width, CV_8UC3,
                   cv::Scalar(255, 255, 255));

    int x_offset = 0;
    for (size_t i = 0; i < images_.size(); ++i) {
      // 复制图像
      cv::Mat roi = result(
          cv::Rect(x_offset, text_height, images_[i].cols, images_[i].rows));
      images_[i].copyTo(roi);

      // 添加曝光信息
      std::string text =
          std::format("EV: {:.1f}", std::log2(metadata_[i].exposure_time));
      cv::putText(result, text, cv::Point(x_offset + 10, text_height - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

      x_offset += images_[i].cols + padding;
    }

    return result;
  }

  cv::Mat visualizeAlignment() const {
    if (images_.size() < 2)
      return cv::Mat();

    // 创建彩色叠加显示
    cv::Mat result;
    cv::Mat reference = images_[0];
    cv::Mat aligned = images_[1];

    // 将图像转换为不同的颜色通道
    std::vector<cv::Mat> channels(3);
    channels[0] = reference.clone(); // 红色通道使用参考图像
    channels[1] = aligned.clone();   // 绿色通道使用对齐后的图像
    channels[2] = cv::Mat::zeros(reference.size(), CV_8UC1); // 蓝色通道置零

    cv::merge(channels, result);

    // 添加对齐网格
    const int grid_size = 50;
    for (int x = 0; x < result.cols; x += grid_size) {
      cv::line(result, cv::Point(x, 0), cv::Point(x, result.rows),
               cv::Scalar(0, 255, 255), 1);
    }

    for (int y = 0; y < result.rows; y += grid_size) {
      cv::line(result, cv::Point(0, y), cv::Point(result.cols, y),
               cv::Scalar(0, 255, 255), 1);
    }

    // 添加对齐质量信息
    std::string quality = evaluateAlignment();
    cv::putText(result, "Alignment Quality: " + quality, cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

    // 添加误差热图
    cv::Mat diff_map;
    cv::absdiff(reference, aligned, diff_map);
    cv::applyColorMap(diff_map, diff_map, cv::COLORMAP_JET);

    // 水平拼接原始叠加图和误差热图
    cv::Mat combined;
    cv::hconcat(result, diff_map, combined);

    return combined;
  }

  // 添加辅助函数用于生成调试信息
  void saveDebugInfo(const fs::path &output_dir) const {
    if (!fs::exists(output_dir)) {
      fs::create_directories(output_dir); // 使用 fs 命名空间
    }

    // 保存响应曲线
    if (!response_curve_.empty()) {
      cv::Mat curve_vis;
      cv::normalize(response_curve_, curve_vis, 0, 255, cv::NORM_MINMAX);
      cv::imwrite((output_dir / "response_curve.png").string(), curve_vis);
    }

    // 保存每张输入图像的调试信息
    for (size_t i = 0; i < images_.size(); ++i) {
      const auto &img = images_[i];
      const auto &meta = metadata_[i];

      std::string basename = std::format("frame_{:02d}", i);

      // 保存直方图
      cv::Mat hist;
      int histSize = 256;
      float range[] = {0, 256};
      const float *histRange = {range};
      cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

      // 绘制直方图
      cv::Mat histImage(400, 512, CV_8UC3, cv::Scalar(0, 0, 0));
      cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX);

      for (int i = 1; i < histSize; i++) {
        cv::line(histImage,
                 cv::Point((i - 1) * 2, 400 - cvRound(hist.at<float>(i - 1))),
                 cv::Point(i * 2, 400 - cvRound(hist.at<float>(i))),
                 cv::Scalar(255, 255, 255), 2);
      }

      // 使用 fs::path 进行路径操作
      fs::path hist_path = output_dir / (basename + "_hist.png");
      cv::imwrite(hist_path.string(), histImage);

      // 使用 std::ofstream 写入元数据
      std::ofstream meta_file(output_dir / (basename + "_meta.txt"));
      if (meta_file.is_open()) {
        meta_file << std::format("Exposure: {:.3f}s\n"
                                 "ISO: {}\n"
                                 "Aperture: f/{:.1f}\n"
                                 "Camera: {}\n",
                                 meta.exposure_time, meta.iso, meta.aperture,
                                 meta.camera_model);
        meta_file.close();
      }
    }
  }

private:
  // 优化图像处理过程中的内存分配
  void preprocessImages() {
    std::vector<std::future<cv::Mat>> futures;
    futures.reserve(images_.size()); // 预分配空间避免重分配

    for (size_t i = 0; i < images_.size(); ++i) {
      futures.push_back(std::async(std::launch::async, [this, i]() {
        void *memory = image_pool_.allocate();
        cv::Mat *result = new (memory) cv::Mat();

        ImagePreprocessor preprocessor(images_[i]);
        *result = preprocessor.denoise();

        if (config_.auto_exposure) {
          *result = preprocessor.adjustExposure(calculateOptimalEV(images_[i]));
        }

        return *result;
      }));
    }

    for (size_t i = 0; i < futures.size(); ++i) {
      image_pool_.deallocate(&images_[i]);
      images_[i] = futures[i].get();
    }
  }

  void alignImages() {
    switch (config_.alignment) {
    case HDRConfig::AlignmentMethod::MTB: {
      auto aligner = cv::createAlignMTB();
      aligner->process(images_, images_);
      break;
    }
    case HDRConfig::AlignmentMethod::HOMOGRAPHY:
      alignWithHomography();
      break;
    case HDRConfig::AlignmentMethod::NONE:
      break;
    }
  }

  void calculateResponse() {
    std::vector<float> times;
    for (const auto &meta : metadata_) {
      times.push_back(meta.exposure_time);
    }

    switch (config_.response) {
    case HDRConfig::ResponseMethod::DEBEVEC: {
      auto calibrate = cv::createCalibrateDebevec();
      calibrate->process(images_, response_curve_, times);
      break;
    }
    case HDRConfig::ResponseMethod::ROBERTSON: {
      auto calibrate = cv::createCalibrateRobertson();
      calibrate->process(images_, response_curve_, times);
      break;
    }
    }
  }

  void mergeImages() {
    std::vector<float> times;
    for (const auto &meta : metadata_) {
      times.push_back(meta.exposure_time);
    }

    switch (config_.merge) {
    case HDRConfig::MergeMethod::DEBEVEC: {
      auto merge = cv::createMergeDebevec();
      merge->process(images_, hdr_result_, times);
      break;
    }
    case HDRConfig::MergeMethod::MERTENS: {
      auto merge = cv::createMergeMertens();
      merge->process(images_, hdr_result_);
      break;
    }
    }
  }

  void alignWithHomography() {
    if (images_.empty())
      return;

    cv::Mat reference = images_[0];
    cv::Ptr<cv::Feature2D> detector = cv::SIFT::create();

    for (size_t i = 1; i < images_.size(); i++) {
      // 检测特征点
      std::vector<cv::KeyPoint> keypoints1, keypoints2;
      cv::Mat descriptors1, descriptors2;

      detector->detectAndCompute(reference, cv::noArray(), keypoints1,
                                 descriptors1);
      detector->detectAndCompute(images_[i], cv::noArray(), keypoints2,
                                 descriptors2);

      // 特征匹配
      cv::Ptr<cv::DescriptorMatcher> matcher =
          cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
      std::vector<std::vector<cv::DMatch>> knn_matches;
      matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

      // 应用比率测试进行筛选
      std::vector<cv::DMatch> good_matches;
      for (const auto &match : knn_matches) {
        if (match[0].distance < 0.7f * match[1].distance) {
          good_matches.push_back(match[0]);
        }
      }

      // 提取匹配点对
      std::vector<cv::Point2f> points1, points2;
      for (const auto &match : good_matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
      }

      // 计算单应性矩阵
      if (points1.size() >= 4) {
        cv::Mat H = cv::findHomography(points2, points1, cv::RANSAC);
        cv::warpPerspective(images_[i], images_[i], H, images_[i].size());
      }
    }
  }

  float calculateOptimalEV(const cv::Mat &image) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // 计算图像亮度直方图
    cv::Mat hist;
    float range[] = {0, 256};
    const float *ranges[] = {range};
    int histSize[] = {256};
    int channels[] = {0};
    cv::calcHist(&gray, 1, channels, cv::Mat(), hist, 1, histSize, ranges);

    // 计算平均亮度和标准差
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);

    // 根据亮度分布计算最优曝光补偿
    float target_mean = 127.0f; // 目标平均亮度
    float ev = std::log2(target_mean / mean[0]);

    return std::clamp(ev, -2.0f, 2.0f);
  }

  size_t calculateMemoryUsage() const {
    size_t total = 0;

    // 计算图像内存使用
    for (const auto &img : images_) {
      total += img.total() * img.elemSize();
    }

    // 计算HDR结果内存
    if (!hdr_result_.empty()) {
      total += hdr_result_.total() * hdr_result_.elemSize();
    }

    // 计算响应曲线内存
    if (!response_curve_.empty()) {
      total += response_curve_.total() * response_curve_.elemSize();
    }

    return total;
  }

  std::string evaluateAlignment() const {
    if (images_.size() < 2)
      return "N/A";

    float mse = 0.0f;
    cv::Mat diff;

    // 计算相邻图像之间的均方误差
    for (size_t i = 1; i < images_.size(); i++) {
      cv::absdiff(images_[i], images_[i - 1], diff);
      cv::multiply(diff, diff, diff);
      mse += static_cast<float>(cv::mean(diff)[0]);
    }

    mse /= (images_.size() - 1);

    if (mse < 100)
      return "Excellent";
    else if (mse < 500)
      return "Good";
    else if (mse < 1000)
      return "Fair";
    else
      return "Poor";
  }

  float calculateDynamicRange() const {
    if (hdr_result_.empty())
      return 0.0f;

    cv::Mat gray;
    cv::cvtColor(hdr_result_, gray, cv::COLOR_BGR2GRAY);

    double min_val, max_val;
    cv::minMaxLoc(gray, &min_val, &max_val);

    // 避免除以0，添加小的偏移量
    const float epsilon = 1e-6f;
    min_val = std::max<float>(min_val, epsilon); // 修复 std::max 调用

    // 以EV(曝光值)为单位计算动态范围
    return std::log2(max_val / min_val);
  }

  // 添加RAII包装类用于自动释放内存池资源
  template <typename T> class PoolRAII {
    MemoryPool<sizeof(T)> &pool_;
    T *ptr_;

  public:
    PoolRAII(MemoryPool<sizeof(T)> &pool) : pool_(pool) {
      void *memory = pool_.allocate();
      ptr_ = new (memory) T();
    }

    ~PoolRAII() {
      if (ptr_) {
        ptr_->~T();
        pool_.deallocate(ptr_);
      }
    }

    T *get() { return ptr_; }
    T *release() {
      T *tmp = ptr_;
      ptr_ = nullptr;
      return tmp;
    }
  };

  // 优化内存管理的辅助方法
  template <typename T> T *allocateFromPool(MemoryPool<sizeof(T)> &pool) {
    void *memory = pool.allocate();
    return new (memory) T();
  }
};

// Usage example
void processHDRExample() {
  HDRConfig config;
  config.alignment = HDRConfig::AlignmentMethod::MTB;
  config.response = HDRConfig::ResponseMethod::DEBEVEC;
  config.merge = HDRConfig::MergeMethod::DEBEVEC;
  config.use_gpu = true;
  config.num_threads = 8;

  HDRProcessor processor(config);

  // Add images with metadata
  std::vector<std::string> image_paths = {"img1.jpg", "img2.jpg", "img3.jpg"};
  for (const auto &path : image_paths) {
    cv::Mat img = cv::imread(path);
    ImageMetadata metadata{.exposure_time = 1.0f / 30.0f,
                           .iso = 100.0f,
                           .aperture = 2.8f,
                           .camera_model = "Example Camera",
                           .capture_time = std::chrono::system_clock::now()};
    processor.addImage(img, metadata);
  }

  try {
    auto stats = processor.process();
    if (stats) {
      std::cout << "Processing time: " << stats->processing_time.count()
                << "ms\n";
    }

    // Get tone-mapped result
    cv::Mat result = processor.getResult(ToneMapper::Method::DRAGO);
    cv::imwrite("hdr_result.jpg", result);

    // Save HDR file
    processor.saveHDR("hdr_result.hdr");

    // Visualizations
    cv::imwrite("exposure_vis.jpg", processor.visualizeExposures());
    cv::imwrite("alignment_vis.jpg", processor.visualizeAlignment());
  } catch (const HDRException &e) {
    std::cerr << "HDR processing failed: " << e.what() << std::endl;
  }
}
