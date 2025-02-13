#include "Diff.hpp"

#include <algorithm>
#include <atomic>
#include <immintrin.h> // SIMD指令
#include <map>
#include <numeric>
#include <ranges>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <vector>

#include <QPainter>

namespace {
std::shared_ptr<spdlog::logger> diffLogger =
    spdlog::basic_logger_mt("DiffLogger", "logs/diff.log");
} // namespace

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
    diffLogger->error("Invalid input images");
    return false;
  }
  if (img1.size() != img2.size()) {
    diffLogger->warn("Image size mismatch: {} vs {}", img1.size(), img2.size());
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

//////////////////////////////////////////////////////////////
// ComparisonStrategyBase method implementations
//////////////////////////////////////////////////////////////

QImage ComparisonStrategyBase::preprocessImage(const QImage &img) const {
  if (SUBSAMPLE_FACTOR > 1) {
    return img.scaled(img.width() / SUBSAMPLE_FACTOR,
                      img.height() / SUBSAMPLE_FACTOR, Qt::IgnoreAspectRatio,
                      Qt::FastTransformation);
  }
  return img;
}

void ComparisonStrategyBase::compareBlockSIMD(const uchar *block1,
                                              const uchar *block2, uchar *dest,
                                              size_t size) const {
  size_t i = 0;
#ifdef __AVX2__
  for (; i + 32 <= size; i += 32) {
    __m256i a =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block1 + i));
    __m256i b =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block2 + i));
    __m256i diff =
        _mm256_sub_epi8(_mm256_max_epu8(a, b), _mm256_min_epu8(a, b));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest + i), diff);
  }
#endif
  for (; i < size; ++i) {
    dest[i] = std::abs(block1[i] - block2[i]);
  }
}

std::vector<QRect>
ComparisonStrategyBase::findDifferenceRegions(const QImage &diffImg) const {
  const int width = diffImg.width();
  const int height = diffImg.height();
  const int threshold = 32;

  DisjointSet ds(width * height);
  std::map<int, QRect> regionMap;
  // 遍历每个像素，记录相似区域
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (qGray(diffImg.pixel(x, y)) > threshold) {
        const int idx = y * width + x;
        std::array<std::pair<int, int>, 4> directions = {
            std::make_pair(-1, 0), std::make_pair(0, -1), std::make_pair(1, 0),
            std::make_pair(0, 1)};
        for (const auto &[dx, dy] : directions) {
          int nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
              qGray(diffImg.pixel(nx, ny)) > threshold) {
            ds.unite(idx, ny * width + nx);
          }
        }
      }
    }
  }

  // 根据并查集构造矩形区域
  std::vector<QRect> result;
  // 遍历每个像素重新构造区域
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (qGray(diffImg.pixel(x, y)) > threshold) {
        int setId = ds.find(y * width + x);
        auto &rect = regionMap[setId];
        if (rect.isNull()) {
          rect = QRect(x, y, 1, 1);
        } else {
          rect = rect.united(QRect(x, y, 1, 1));
        }
      }
    }
  }
  for (const auto &[_, rect] : regionMap) {
    result.push_back(rect);
  }
  return result;
}

ComparisonStrategyBase::DisjointSet::DisjointSet(int size)
    : parent(size), rank(size, 0) {
  std::iota(parent.begin(), parent.end(), 0);
}

int ComparisonStrategyBase::DisjointSet::find(int x) {
  if (parent[x] != x) {
    parent[x] = find(parent[x]);
  }
  return parent[x];
}

void ComparisonStrategyBase::DisjointSet::unite(int x, int y) {
  x = find(x);
  y = find(y);
  if (x != y) {
    if (rank[x] < rank[y]) {
      std::swap(x, y);
    }
    parent[y] = x;
    if (rank[x] == rank[y]) {
      ++rank[x];
    }
  }
}

//////////////////////////////////////////////////////////////
// PixelDifferenceStrategy method implementations
//////////////////////////////////////////////////////////////

ComparisonResult
PixelDifferenceStrategy::compare(const QImage &img1, const QImage &img2,
                                 QPromise<ComparisonResult> &promise) const {
  QImage processed1 = preprocessImage(img1);
  QImage processed2 = preprocessImage(img2);

  const int height = processed1.height();
  const int width = processed1.width();
  QImage diffImg(processed1.size(), QImage::Format_ARGB32);
  std::atomic_uint64_t totalDiff{0};
  std::atomic_int progress{0};

#pragma omp parallel for collapse(2) reduction(+ : totalDiff) schedule(dynamic)
  for (int y = 0; y < height; y += BLOCK_SIZE) {
    for (int x = 0; x < width; x += BLOCK_SIZE) {
      if (promise.isCanceled())
        continue;
      const int blockHeight = std::min(BLOCK_SIZE, height - y);
      const int blockWidth = std::min(BLOCK_SIZE, width - x);
      for (int by = 0; by < blockHeight; ++by) {
        const uchar *line1 = processed1.scanLine(y + by) + x * 4;
        const uchar *line2 = processed2.scanLine(y + by) + x * 4;
        uchar *dest = diffImg.scanLine(y + by) + x * 4;
        compareBlockSIMD(line1, line2, dest, blockWidth * 4);
        for (int bx = 0; bx < blockWidth * 4; ++bx) {
          totalDiff += dest[bx];
        }
      }
#pragma omp atomic
      progress += blockHeight;
    }
    if (progress % (height / 10) == 0) {
      promise.setProgressValue(static_cast<int>(progress * 100.0 / height));
    }
  }
  double mse = static_cast<double>(totalDiff) / (width * height * 4);
  double similarity = 100.0 * (1.0 - std::sqrt(mse) / 255.0);
  return {diffImg, similarity, findDifferenceRegions(diffImg)};
}

//////////////////////////////////////////////////////////////
// SSIMStrategy method implementations
//////////////////////////////////////////////////////////////

ComparisonResult
SSIMStrategy::compare(const QImage &img1, const QImage &img2,
                      QPromise<ComparisonResult> &promise) const {
  QImage processed1 = preprocessImage(img1);
  QImage processed2 = preprocessImage(img2);

  const int width = processed1.width();
  const int height = processed1.height();
  QImage diffImg(width, height, QImage::Format_ARGB32);
  double totalSSIM = 0.0;

#pragma omp parallel for reduction(+ : totalSSIM)
  for (int y = 0; y < height - WINDOW_SIZE; y += WINDOW_SIZE) {
    if (promise.isCanceled())
      continue;
    for (int x = 0; x < width - WINDOW_SIZE; x += WINDOW_SIZE) {
      double ssim = computeSSIM(processed1, processed2, x, y);
      totalSSIM += ssim;
      int color = static_cast<int>((1.0 - ssim) * 255);
      QRgb value = qRgb(color, color, color);
      for (int wy = 0; wy < WINDOW_SIZE; ++wy) {
        for (int wx = 0; wx < WINDOW_SIZE; ++wx) {
          diffImg.setPixel(x + wx, y + wy, value);
        }
      }
    }
  }
  double numWindows = (static_cast<double>(width) / WINDOW_SIZE) *
                      (static_cast<double>(height) / WINDOW_SIZE);
  double similarity = (totalSSIM * 100.0) / numWindows;
  return {diffImg, similarity, findDifferenceRegions(diffImg)};
}

double SSIMStrategy::computeSSIM(const QImage &img1, const QImage &img2, int x,
                                 int y) const {
  double mean1 = 0, mean2 = 0, variance1 = 0, variance2 = 0, covariance = 0;
  for (int wy = 0; wy < WINDOW_SIZE; ++wy) {
    for (int wx = 0; wx < WINDOW_SIZE; ++wx) {
      double v1 = qGray(img1.pixel(x + wx, y + wy));
      double v2 = qGray(img2.pixel(x + wx, y + wy));
      mean1 += v1;
      mean2 += v2;
    }
  }
  mean1 /= (WINDOW_SIZE * WINDOW_SIZE);
  mean2 /= (WINDOW_SIZE * WINDOW_SIZE);
  for (int wy = 0; wy < WINDOW_SIZE; ++wy) {
    for (int wx = 0; wx < WINDOW_SIZE; ++wx) {
      double v1 = qGray(img1.pixel(x + wx, y + wy)) - mean1;
      double v2 = qGray(img2.pixel(x + wx, y + wy)) - mean2;
      variance1 += v1 * v1;
      variance2 += v2 * v2;
      covariance += v1 * v2;
    }
  }
  variance1 /= (WINDOW_SIZE * WINDOW_SIZE - 1);
  variance2 /= (WINDOW_SIZE * WINDOW_SIZE - 1);
  covariance /= (WINDOW_SIZE * WINDOW_SIZE - 1);
  const double C1 = (K1 * 255) * (K1 * 255);
  const double C2 = (K2 * 255) * (K2 * 255);
  return ((2 * mean1 * mean2 + C1) * (2 * covariance + C2)) /
         ((mean1 * mean1 + mean2 * mean2 + C1) * (variance1 + variance2 + C2));
}

//////////////////////////////////////////////////////////////
// PerceptualHashStrategy method implementations
//////////////////////////////////////////////////////////////

ComparisonResult
PerceptualHashStrategy::compare(const QImage &img1, const QImage &img2,
                                QPromise<ComparisonResult> &promise) const {
  uint64_t hash1 = computeHash(img1);
  uint64_t hash2 = computeHash(img2);
  int distance = hammingDistance(hash1, hash2);
  double similarity = 100.0 * (1.0 - distance / 64.0);
  QImage diffImg(img1.size(), QImage::Format_ARGB32);
  QPainter painter(&diffImg);
  painter.fillRect(diffImg.rect(), Qt::white);
  if (distance > 0) {
    for (int i = 0; i < 64; ++i) {
      if ((hash1 & (1ULL << i)) != (hash2 & (1ULL << i))) {
        int x = (i % 8) * (img1.width() / 8);
        int y = (i / 8) * (img1.height() / 8);
        painter.fillRect(x, y, img1.width() / 8, img1.height() / 8,
                         QColor(255, 0, 0, 127));
      }
    }
  }
  return {diffImg, similarity, findDifferenceRegions(diffImg)};
}

uint64_t PerceptualHashStrategy::computeHash(const QImage &img) const {
  QImage scaled =
      img.scaled(8, 8, Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
          .convertToFormat(QImage::Format_Grayscale8);
  int sum = 0;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      sum += qGray(scaled.pixel(x, y));
    }
  }
  int avg = sum / 64;
  uint64_t hash = 0;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      hash = (hash << 1) | (qGray(scaled.pixel(x, y)) > avg ? 1 : 0);
    }
  }
  return hash;
}

int PerceptualHashStrategy::hammingDistance(uint64_t hash1,
                                            uint64_t hash2) const {
  uint64_t diff = hash1 ^ hash2;
  return std::popcount(diff);
}

//////////////////////////////////////////////////////////////
// HistogramStrategy method implementations
//////////////////////////////////////////////////////////////

ComparisonResult
HistogramStrategy::compare(const QImage &img1, const QImage &img2,
                           QPromise<ComparisonResult> &promise) const {
  auto hist1 = computeHistogram(img1);
  auto hist2 = computeHistogram(img2);
  double similarity = compareHistograms(hist1, hist2);
  QImage diffImg(img1.size(), QImage::Format_ARGB32);
  QPainter painter(&diffImg);
  for (int i = 0; i < HIST_BINS; ++i) {
    int h1 = hist1[i] * diffImg.height() / (img1.width() * img1.height());
    int h2 = hist2[i] * diffImg.height() / (img2.width() * img2.height());
    painter.setPen(Qt::blue);
    painter.drawLine(i * diffImg.width() / HIST_BINS, diffImg.height(),
                     i * diffImg.width() / HIST_BINS, diffImg.height() - h1);
    painter.setPen(Qt::red);
    painter.drawLine(i * diffImg.width() / HIST_BINS + 1, diffImg.height(),
                     i * diffImg.width() / HIST_BINS + 1,
                     diffImg.height() - h2);
  }
  return {diffImg, similarity * 100.0, {}};
}

std::vector<int> HistogramStrategy::computeHistogram(const QImage &img) const {
  std::vector<int> histogram(HIST_BINS, 0);
  for (int y = 0; y < img.height(); ++y) {
    for (int x = 0; x < img.width(); ++x) {
      histogram[qGray(img.pixel(x, y))]++;
    }
  }
  return histogram;
}

double
HistogramStrategy::compareHistograms(const std::vector<int> &hist1,
                                     const std::vector<int> &hist2) const {
  double correlation = 0;
  double norm1 = 0, norm2 = 0;
  for (int i = 0; i < HIST_BINS; ++i) {
    correlation += hist1[i] * hist2[i];
    norm1 += hist1[i] * hist1[i];
    norm2 += hist2[i] * hist2[i];
  }
  return correlation / (std::sqrt(norm1) * std::sqrt(norm2));
}