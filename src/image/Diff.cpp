#include "Diff.hpp"

#include <algorithm>
#include <atomic>
#include <ranges>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>
#include <stack>
#include <vector>

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

std::tuple<int, int, int> qUnpack(QRgb rgb) {
  return {qRed(rgb), qGreen(rgb), qBlue(rgb)};
}

namespace ColorSpace {

CIELAB RGB2LAB(QRgb rgb) {
  const auto [r, g, b] = qUnpack(rgb);
  return {0.2126 * r + 0.7152 * g + 0.0722 * b, static_cast<double>(r - g),
          static_cast<double>(g - b)};
}
} // namespace ColorSpace

bool ImageDiff::validateImages(const QImage &img1, const QImage &img2) {
  if (img1.isNull() || img2.isNull()) {
    spdlog::error("Invalid input images");
    return false;
  }

  if (img1.size() != img2.size()) {
    spdlog::warn("Image size mismatch: {} vs {}", img1.size(), img2.size());
    return false;
  }

  return true;
}

void ImageDiff::postProcessResult(ComparisonResult &result) const {
  if (!result.differenceImage.isNull()) {
    auto bits = result.differenceImage.bits();
    const auto [min, max] =
        std::minmax_element(bits, bits + result.differenceImage.sizeInBytes());

    const float scale = 255.0f / (*max - *min + 1e-5f);
    std::transform(bits, bits + result.differenceImage.sizeInBytes(), bits,
                   [=](auto val) { return (val - *min) * scale; });
  }
}

void processRows(const QImage &img, int height,
                 const std::function<void(int)> &fn) {
  auto rows = std::views::iota(0, height);
  std::ranges::for_each(rows, fn);
}

ComparisonResult
PixelDifferenceStrategy::compare(const QImage &img1, const QImage &img2,
                                 QPromise<ComparisonResult> &promise) const {
  QImage diffImg(img1.size(), QImage::Format_ARGB32);
  std::atomic_uint64_t totalDiff{0};
  std::atomic_int progress{0};

  const int height = img1.height();
  const int bytesPerLine = img1.bytesPerLine();

#pragma omp parallel for reduction(+ : totalDiff) schedule(dynamic)
  for (int y = 0; y < height; ++y) {
    if (promise.isCanceled())
      continue;

    const uchar *line1 = img1.scanLine(y);
    const uchar *line2 = img2.scanLine(y);
    uchar *dest = diffImg.scanLine(y);

    for (int x = 0; x < bytesPerLine; ++x) {
      int diff =
          std::abs(static_cast<int>(line1[x]) - static_cast<int>(line2[x]));
      dest[x] = static_cast<uchar>(diff);
      totalDiff += diff;
    }

#pragma omp atomic
    ++progress;

    if (progress % 10 == 0) {
#pragma omp critical
      {
        promise.setProgressValue(static_cast<int>(progress * 100.0 / height));
      }
    }
  }
  double mse =
      static_cast<double>(totalDiff) / (img1.width() * img1.height() * 4);
  double rmse = std::sqrt(mse);
  double similarity = 100.0 * (1.0 - rmse / 255.0);

  return {diffImg, similarity, findDifferenceRegions(diffImg)};
}

QString PixelDifferenceStrategy::name() const { return "Pixel Difference"; }

std::vector<QRect>
PixelDifferenceStrategy::findDifferenceRegions(const QImage &diffImg) const {
  // 连通区域分析实现
  std::vector<QRect> regions;
  QImage visited(diffImg.size(), QImage::Format_Mono);
  visited.fill(0);

  const int threshold = 32;
  const QPoint directions[] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  for (int y = 0; y < diffImg.height(); ++y) {
    for (int x = 0; x < diffImg.width(); ++x) {
      if (qGray(diffImg.pixel(x, y)) > threshold && !visited.pixelIndex(x, y)) {
        QRect region;
        floodFill(diffImg, visited, x, y, threshold, region);
        regions.push_back(region);
      }
    }
  }

  return regions;
}

void PixelDifferenceStrategy::floodFill(const QImage &img, QImage &visited,
                                        int x, int y, int threshold,
                                        QRect &region) const {
  const std::array<QPoint, 4> directions = {QPoint{1, 0}, QPoint{-1, 0},
                                            QPoint{0, 1}, QPoint{0, -1}};
  std::stack<QPoint> stack;
  stack.emplace(x, y);
  visited.setPixel(x, y, 1);

  int minX = x, maxX = x;
  int minY = y, maxY = y;

  while (!stack.empty()) {
    auto [cx, cy] = stack.top();
    stack.pop();

    minX = std::min(minX, cx);
    maxX = std::max(maxX, cx);
    minY = std::min(minY, cy);
    maxY = std::max(maxY, cy);

    for (const auto &[dx, dy] : directions) {
      const int nx = cx + dx;
      const int ny = cy + dy;
      if (nx >= 0 && nx < img.width() && ny >= 0 && ny < img.height() &&
          !visited.pixelIndex(nx, ny) && qGray(img.pixel(nx, ny)) > threshold) {
        visited.setPixel(nx, ny, 1);
        stack.emplace(nx, ny);
      }
    }
  }

  region = QRect(QPoint(minX, minY), QPoint(maxX, maxY));
}
