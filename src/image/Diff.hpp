#pragma once

#include <vector>

#include <QFuture>
#include <QImage>
#include <QPromise>
#include <QRect>


#include <spdlog/spdlog.h>

/**
 * @brief Struct to hold the result of an image comparison.
 */
struct ComparisonResult {
  QImage differenceImage; ///< The image showing the differences.
  double similarityPercent =
      0.0; ///< The percentage of similarity between the images.
  std::vector<QRect>
      differenceRegions; ///< The regions where differences were found.
  std::chrono::milliseconds
      duration; ///< The duration of the comparison process.
};

/**
 * @brief Concept to define a comparison strategy.
 */
template <typename T>
concept ComparisonStrategy = requires(T s, const QImage &a, const QImage &b,
                                      QPromise<ComparisonResult> &p) {
  { s.compare(a, b, p) } -> std::same_as<ComparisonResult>;
  { s.name() } -> std::same_as<QString>;
};

/**
 * @brief Unpacks a QRgb value into its individual components.
 *
 * @param rgb The QRgb value.
 * @return A tuple containing the red, green, and blue components.
 */
std::tuple<int, int, int> qUnpack(QRgb rgb);

namespace ColorSpace {

/**
 * @brief Struct to represent a color in the CIELAB color space.
 */
struct CIELAB {
  double L; ///< Lightness component.
  double a; ///< Green-Red component.
  double b; ///< Blue-Yellow component.
};

/**
 * @brief Converts an RGB color to the CIELAB color space.
 *
 * @param rgb The RGB color.
 * @return The corresponding CIELAB color.
 */
CIELAB RGB2LAB(QRgb rgb);

} // namespace ColorSpace

/**
 * @brief Class to perform image difference calculations.
 */
class ImageDiff {
public:
  /**
   * @brief Compares two images using a specified strategy.
   *
   * @tparam Strategy The comparison strategy to use.
   * @param img1 The first image.
   * @param img2 The second image.
   * @param strategy The comparison strategy.
   * @param promise The promise to report the comparison result.
   * @return The result of the comparison.
   */
  template <typename Strategy>
  ComparisonResult compare(const QImage &img1, const QImage &img2,
                           Strategy &&strategy,
                           QPromise<ComparisonResult> &promise) {
    if (!validateImages(img1, img2)) {
      promise.future().cancel();
      return {};
    }

    try {
      const auto start = std::chrono::high_resolution_clock::now();

      QImage converted1 = img1.convertToFormat(QImage::Format_ARGB32);
      QImage converted2 = img2.convertToFormat(QImage::Format_ARGB32);

      auto result = strategy.compare(converted1, converted2, promise);
      result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - start);

      postProcessResult(result);
      return result;
    } catch (const std::exception &e) {
      spdlog::error("Comparison failed: {}", e.what());
      promise.future().cancel();
      return {};
    }
  }

private:
  /**
   * @brief Validates the input images.
   *
   * @param img1 The first image.
   * @param img2 The second image.
   * @return True if the images are valid for comparison, false otherwise.
   */
  bool validateImages(const QImage &img1, const QImage &img2);

  /**
   * @brief Post-processes the comparison result.
   *
   * @param result The comparison result to post-process.
   */
  void postProcessResult(ComparisonResult &result) const;
};

/**
 * @brief Processes rows of an image.
 *
 * @param img The image to process.
 * @param height The height of the image.
 * @param fn The function to apply to each row.
 */
void processRows(const QImage &img, int height,
                 const std::function<void(int)> &fn);

/**
 * @brief 图像比较策略的基类
 */
class ComparisonStrategyBase {
protected:
  static constexpr int BLOCK_SIZE = 16;
  static constexpr int SUBSAMPLE_FACTOR = 2;

  QImage preprocessImage(const QImage &img) const;
  void compareBlockSIMD(const uchar *block1, const uchar *block2, uchar *dest,
                        size_t size) const;
  std::vector<QRect> findDifferenceRegions(const QImage &diffImg) const;

  class DisjointSet {
    std::vector<int> parent;
    std::vector<int> rank;

  public:
    DisjointSet(int size);
    int find(int x);
    void unite(int x, int y);
  };
};

/**
 * @brief Class to perform pixel difference comparison.
 */
class PixelDifferenceStrategy : public ComparisonStrategyBase {
public:
  /**
   * @brief Compares two images and finds the pixel differences.
   *
   * @param img1 The first image.
   * @param img2 The second image.
   * @param promise The promise to report the comparison result.
   * @return The result of the comparison.
   */
  ComparisonResult compare(const QImage &img1, const QImage &img2,
                           QPromise<ComparisonResult> &promise) const;

  /**
   * @brief Gets the name of the strategy.
   *
   * @return The name of the strategy.
   */
  QString name() const { return "像素差异比较"; }
};

/**
 * @brief 使用结构相似性(SSIM)的比较策略
 */
class SSIMStrategy : public ComparisonStrategyBase {
public:
  ComparisonResult compare(const QImage &img1, const QImage &img2,
                           QPromise<ComparisonResult> &promise) const;
  QString name() const { return "结构相似性比较"; }

private:
  static constexpr double K1 = 0.01;
  static constexpr double K2 = 0.03;
  static constexpr int WINDOW_SIZE = 8;

  double computeSSIM(const QImage &img1, const QImage &img2, int x,
                     int y) const;
};

/**
 * @brief 使用感知哈希(pHash)的比较策略
 */
class PerceptualHashStrategy : public ComparisonStrategyBase {
public:
  ComparisonResult compare(const QImage &img1, const QImage &img2,
                           QPromise<ComparisonResult> &promise) const;
  QString name() const { return "感知哈希比较"; }

private:
  static constexpr int HASH_SIZE = 64;
  uint64_t computeHash(const QImage &img) const;
  int hammingDistance(uint64_t hash1, uint64_t hash2) const;
};

/**
 * @brief 基于颜色直方图的比较策略
 */
class HistogramStrategy : public ComparisonStrategyBase {
public:
  ComparisonResult compare(const QImage &img1, const QImage &img2,
                           QPromise<ComparisonResult> &promise) const;
  QString name() const { return "颜色直方图比较"; }

private:
  static constexpr int HIST_BINS = 256;
  std::vector<int> computeHistogram(const QImage &img) const;
  double compareHistograms(const std::vector<int> &hist1,
                           const std::vector<int> &hist2) const;
};
