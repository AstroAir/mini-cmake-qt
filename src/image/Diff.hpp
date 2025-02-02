#pragma once

#include <vector>

// Qt头文件
#include <QApplication>
#include <QComboBox>
#include <QFileDialog>
#include <QFutureWatcher>
#include <QGridLayout>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QMessageBox>
#include <QPainter>
#include <QProgressBar>
#include <QPushButton>
#include <QWidget>
#include <QtConcurrent>

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
 * @brief Class to perform pixel difference comparison.
 */
class PixelDifferenceStrategy {
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
  QString name() const;

private:
  /**
   * @brief Finds the regions of difference in the image.
   *
   * @param diffImg The difference image.
   * @return A vector of rectangles representing the regions of difference.
   */
  std::vector<QRect> findDifferenceRegions(const QImage &diffImg) const;

  /**
   * @brief Performs a flood fill to find a region of difference.
   *
   * @param img The image to process.
   * @param visited The image to mark visited pixels.
   * @param x The x-coordinate to start the flood fill.
   * @param y The y-coordinate to start the flood fill.
   * @param threshold The threshold for difference.
   * @param region The region to update with the found difference.
   */
  void floodFill(const QImage &img, QImage &visited, int x, int y,
                 int threshold, QRect &region) const;
};
