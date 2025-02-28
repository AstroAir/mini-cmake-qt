#include "Diff.hpp"

#include <algorithm>
#include <atomic>
#include <bit>         // C++20 bit manipulation
#include <immintrin.h> // SIMD instructions
#include <iostream>
#include <numeric>
#include <ranges>
#include <span> // C++20 span
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <thread>
#include <vector>

#include <QPainter>

namespace {
std::shared_ptr<spdlog::logger> diffLogger;

void initLogger() noexcept {
  if (!diffLogger) {
    try {
      diffLogger = spdlog::basic_logger_mt("DiffLogger", "logs/diff.log");
      diffLogger->set_level(spdlog::level::debug);
      diffLogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
      diffLogger->flush_on(spdlog::level::warn);
    } catch (const spdlog::spdlog_ex &ex) {
      std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
    }
  }
}

struct LoggerInitializer {
  LoggerInitializer() noexcept { initLogger(); }
} loggerInit;

} // namespace

constexpr std::tuple<int, int, int> qUnpack(QRgb rgb) noexcept {
  return {qRed(rgb), qGreen(rgb), qBlue(rgb)};
}

namespace ColorSpace {

CIELAB RGB2LAB(QRgb rgb) noexcept {
  const auto [r, g, b] = qUnpack(rgb);
  return {0.2126 * r + 0.7152 * g + 0.0722 * b, static_cast<double>(r - g),
          static_cast<double>(g - b)};
}
} // namespace ColorSpace

bool ImageDiff::validateImages(const QImage &img1,
                               const QImage &img2) noexcept {
  if (img1.isNull() || img2.isNull()) {
    diffLogger->error("One or both images are null");
    return false;
  }

  if (img1.size() != img2.size()) {
    diffLogger->error("Image sizes don't match: {}x{} vs {}x{}", img1.width(),
                      img1.height(), img2.width(), img2.height());
    return false;
  }

  // Enhanced validation for pixel format
  const auto validFormat = [](const QImage &img) {
    return img.format() == QImage::Format_ARGB32 ||
           img.format() == QImage::Format_RGB32 ||
           img.format() == QImage::Format_RGB888 ||
           img.format() == QImage::Format_RGBA8888;
  };

  if (!validFormat(img1)) {
    diffLogger->warn("First image format is not optimal: {}",
                     static_cast<int>(img1.format()));
  }

  if (!validFormat(img2)) {
    diffLogger->warn("Second image format is not optimal: {}",
                     static_cast<int>(img2.format()));
  }

  return true;
}

void ImageDiff::postProcessResult(ComparisonResult &result) const noexcept {
  if (result.differenceImage.isNull()) {
    diffLogger->warn("Difference image is null, skipping post-processing");
    return;
  }

  // Normalize the difference image
  try {
    uchar *bits = result.differenceImage.bits();
    const size_t size = result.differenceImage.sizeInBytes();

    // Find min/max using parallel algorithm
    std::vector<uchar> pixelData(bits, bits + size);
    const auto [minIter, maxIter] =
        std::minmax_element(pixelData.begin(), pixelData.end());

    if (minIter == pixelData.end() || maxIter == pixelData.end()) {
      diffLogger->error("Failed to find min/max pixel values");
      return;
    }

    const uchar minVal = *minIter;
    const uchar maxVal = *maxIter;

    // Avoid division by zero
    if (maxVal == minVal) {
      diffLogger->warn("No variance in difference image");
      return;
    }

    // Apply contrast stretching with SIMD where possible
    const float scale = 255.0f / (maxVal - minVal);

#pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
      bits[i] = static_cast<uchar>((bits[i] - minVal) * scale);
    }

    // Log statistics
    diffLogger->debug("Post-process stats - Min: {}, Max: {}, Scale: {:.4f}",
                      static_cast<int>(minVal), static_cast<int>(maxVal),
                      scale);
  } catch (const std::exception &e) {
    diffLogger->error("Error in postProcessResult: {}", e.what());
  }
}

void processRows(const QImage &img, int height,
                 const std::function<void(int)> &fn) {
  if (img.isNull() || height <= 0 || !fn) {
    throw std::invalid_argument("Invalid parameters in processRows");
  }

  try {
    // Use C++20 ranges for cleaner iteration
    auto rows = std::views::iota(0, height);
    auto threadCount = std::thread::hardware_concurrency();

    // Determine optimal batch size for parallelization
    int batchSize = std::max(1, height / (static_cast<int>(threadCount) * 2));

    std::vector<std::thread> threads;
    threads.reserve(threadCount);

    for (unsigned t = 0; t < threadCount; ++t) {
      threads.emplace_back([&rows, &fn, t, threadCount, height]() {
        auto rowsSubset =
            rows | std::views::filter([t, threadCount, height](int row) {
              return row % threadCount == t;
            });
        for (int row : rowsSubset) {
          fn(row);
        }
      });
    }

    for (auto &thread : threads) {
      thread.join();
    }
  } catch (const std::exception &e) {
    diffLogger->error("Error processing rows: {}", e.what());
    throw;
  }
}

//////////////////////////////////////////////////////////////
// ComparisonStrategyBase method implementations
//////////////////////////////////////////////////////////////

QImage ComparisonStrategyBase::preprocessImage(const QImage &img) const {
  if (img.isNull()) {
    throw std::invalid_argument("Cannot preprocess null image");
  }

  try {
    if (SUBSAMPLE_FACTOR > 1) {
      return img.scaled(img.width() / SUBSAMPLE_FACTOR,
                        img.height() / SUBSAMPLE_FACTOR, Qt::IgnoreAspectRatio,
                        Qt::FastTransformation);
    }
    return img;
  } catch (const std::exception &e) {
    diffLogger->error("Image preprocessing failed: {}", e.what());
    throw std::runtime_error(std::string("Image preprocessing failed: ") +
                             e.what());
  }
}

void ComparisonStrategyBase::compareBlockSIMD(
    std::span<const uchar> block1, std::span<const uchar> block2,
    std::span<uchar> dest) const noexcept {
  if (block1.size() != block2.size() || block1.size() != dest.size() ||
      block1.empty()) {
    diffLogger->error("Invalid block sizes in SIMD comparison");
    return;
  }

  const size_t size = block1.size();
  size_t i = 0;

#ifdef __AVX2__
  // Use AVX2 instructions when available
  for (; i + 32 <= size; i += 32) {
    __m256i a = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(block1.data() + i));
    __m256i b = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(block2.data() + i));
    __m256i diff =
        _mm256_sub_epi8(_mm256_max_epu8(a, b), _mm256_min_epu8(a, b));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest.data() + i), diff);
  }
#elif defined(__SSE4_1__)
  // Fallback to SSE4.1 if AVX2 is not available
  for (; i + 16 <= size; i += 16) {
    __m128i a =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(block1.data() + i));
    __m128i b =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(block2.data() + i));
    __m128i diff = _mm_sub_epi8(_mm_max_epu8(a, b), _mm_min_epu8(a, b));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dest.data() + i), diff);
  }
#endif

  // Process remaining elements
  for (; i < size; ++i) {
    dest[i] = static_cast<uchar>(
        std::abs(static_cast<int>(block1[i]) - static_cast<int>(block2[i])));
  }
}

std::vector<QRect>
ComparisonStrategyBase::findDifferenceRegions(const QImage &diffImg) const {
  if (diffImg.isNull()) {
    diffLogger->error("Cannot find difference regions in null image");
    return {};
  }

  try {
    const int width = diffImg.width();
    const int height = diffImg.height();
    const int threshold = 32; // Minimum difference to consider

    // Use a more efficient union-find structure
    DisjointSet ds(width * height);
    std::unordered_map<int, QRect> regionMap;

    // Determine connected components of difference regions
    std::atomic<int> progress{0};

#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (qGray(diffImg.pixel(x, y)) > threshold) {
          const int idx = y * width + x;

          // Use array of directions for cleaner code
          constexpr std::array<std::pair<int, int>, 4> directions = {
              std::make_pair(-1, 0), std::make_pair(0, -1),
              std::make_pair(1, 0), std::make_pair(0, 1)};

          for (const auto &[dx, dy] : directions) {
            const int nx = x + dx;
            const int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                qGray(diffImg.pixel(nx, ny)) > threshold) {
#pragma omp critical
              {
                ds.unite(idx, ny * width + nx);
              }
            }
          }
        }
      }

      // Update progress
      ++progress;
      if (progress % (height / 10) == 0) {
        diffLogger->debug("Finding regions: {}% complete",
                          (progress * 100) / height);
      }
    }

    // Construct rectangle regions from connected components
    std::mutex regionMutex;

#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (qGray(diffImg.pixel(x, y)) > threshold) {
          int setId = ds.find(y * width + x);

#pragma omp critical
          {
            auto &rect = regionMap[setId];
            if (rect.isNull()) {
              rect = QRect(x, y, 1, 1);
            } else {
              rect = rect.united(QRect(x, y, 1, 1));
            }
          }
        }
      }
    }

    // Extract results
    std::vector<QRect> result;
    result.reserve(regionMap.size());
    for (const auto &[_, rect] : regionMap) {
      result.push_back(rect);
    }

    // Optimize by merging overlapping or nearby regions
    if (result.size() > 1) {
      bool merged = true;
      while (merged) {
        merged = false;
        for (size_t i = 0; i < result.size() && !merged; ++i) {
          for (size_t j = i + 1; j < result.size() && !merged; ++j) {
            const int distance =
                10; // Maximum distance between regions to merge
            QRect expandedRect =
                result[i].adjusted(-distance, -distance, distance, distance);
            if (expandedRect.intersects(result[j])) {
              result[i] = result[i].united(result[j]);
              result.erase(result.begin() + j);
              merged = true;
            }
          }
        }
      }
    }

    diffLogger->debug("Found {} difference regions", result.size());
    return result;
  } catch (const std::exception &e) {
    diffLogger->error("Error finding difference regions: {}", e.what());
    return {};
  }
}

ComparisonStrategyBase::DisjointSet::DisjointSet(int size)
    : parent(size), rank(size, 0) {
  if (size <= 0) {
    throw std::invalid_argument("DisjointSet size must be positive");
  }
  std::iota(parent.begin(), parent.end(), 0);
}

int ComparisonStrategyBase::DisjointSet::find(int x) noexcept {
  // Path compression for better performance
  if (x < 0 || x >= static_cast<int>(parent.size())) {
    return -1; // Invalid index
  }

  if (parent[x] != x) {
    parent[x] = find(parent[x]);
  }
  return parent[x];
}

void ComparisonStrategyBase::DisjointSet::unite(int x, int y) noexcept {
  if (x < 0 || x >= static_cast<int>(parent.size()) || y < 0 ||
      y >= static_cast<int>(parent.size())) {
    return; // Invalid indices
  }

  x = find(x);
  y = find(y);

  if (x != y) {
    // Union by rank for better performance
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
  try {
    QImage processed1 = preprocessImage(img1);
    QImage processed2 = preprocessImage(img2);

    const int height = processed1.height();
    const int width = processed1.width();

    if (height <= 0 || width <= 0) {
      throw std::runtime_error("Invalid image dimensions after preprocessing");
    }

    QImage diffImg(processed1.size(), QImage::Format_ARGB32);
    std::atomic_uint64_t totalDiff{0};
    std::atomic_int progress{0};

    // Determine optimal number of threads and block size
    const unsigned int threadCount = std::thread::hardware_concurrency();
    constexpr int BLOCK_SIZE =
        16; // Define BLOCK_SIZE with an appropriate value
    const int optimalBlockSize =
        std::max(BLOCK_SIZE, static_cast<int>(32 / sizeof(uchar)));

// Use OpenMP for parallelization
#pragma omp parallel for collapse(2) reduction(+ : totalDiff) schedule(dynamic)
    for (int y = 0; y < height; y += optimalBlockSize) {
      for (int x = 0; x < width; x += optimalBlockSize) {
        if (promise.isCanceled())
          continue;

        const int blockHeight = std::min(optimalBlockSize, height - y);
        const int blockWidth = std::min(optimalBlockSize, width - x);

        for (int by = 0; by < blockHeight; ++by) {
          const uchar *line1 = processed1.scanLine(y + by) + x * 4;
          const uchar *line2 = processed2.scanLine(y + by) + x * 4;
          uchar *dest = diffImg.scanLine(y + by) + x * 4;

          // Use span for safer memory access
          std::span<const uchar> block1(line1, blockWidth * 4);
          std::span<const uchar> block2(line2, blockWidth * 4);
          std::span<uchar> destSpan(dest, blockWidth * 4);

          compareBlockSIMD(block1, block2, destSpan);

          // Calculate total difference using SIMD where possible
          uint64_t localDiff = 0;
          for (int bx = 0; bx < blockWidth * 4; ++bx) {
            localDiff += dest[bx];
          }
          totalDiff += localDiff;
        }

#pragma omp atomic
        progress += blockHeight;
      }

      // Update progress every 10%
      if (progress % (height / 10) == 0) {
        promise.setProgressValue(static_cast<int>(progress * 100.0 / height));
      }
    }

    // Calculate similarity with proper normalization
    const double maxPossibleDiff = 255.0 * width * height * 4;
    const double mse = static_cast<double>(totalDiff) / (width * height * 4);
    const double similarity = 100.0 * (1.0 - std::sqrt(mse) / 255.0);

    diffLogger->debug(
        "Pixel difference comparison - MSE: {:.4f}, Similarity: {:.2f}%", mse,
        similarity);

    return {diffImg, similarity, findDifferenceRegions(diffImg)};
  } catch (const std::exception &e) {
    diffLogger->error("Error in pixel difference comparison: {}", e.what());
    promise.future().cancel();
    throw;
  }
}

//////////////////////////////////////////////////////////////
// SSIMStrategy method implementations
//////////////////////////////////////////////////////////////

ComparisonResult
SSIMStrategy::compare(const QImage &img1, const QImage &img2,
                      QPromise<ComparisonResult> &promise) const {
  try {
    QImage processed1 = preprocessImage(img1);
    QImage processed2 = preprocessImage(img2);

    const int width = processed1.width();
    const int height = processed1.height();

    if (width < WINDOW_SIZE || height < WINDOW_SIZE) {
      throw std::runtime_error("Images too small for SSIM comparison");
    }

    QImage diffImg(width, height, QImage::Format_ARGB32);
    diffImg.fill(Qt::white); // Initialize to white
    std::atomic<double> totalSSIM{0.0};
    std::atomic<int> processedWindows{0};
    const int totalWindows = ((width - WINDOW_SIZE) / WINDOW_SIZE) *
                             ((height - WINDOW_SIZE) / WINDOW_SIZE);

// Parallel processing of windows
#pragma omp parallel for collapse(2) reduction(+ : totalSSIM) schedule(dynamic)
    for (int y = 0; y <= height - WINDOW_SIZE; y += WINDOW_SIZE) {
      for (int x = 0; x <= width - WINDOW_SIZE; x += WINDOW_SIZE) {
        if (promise.isCanceled())
          continue;

        double ssim = computeSSIM(processed1, processed2, x, y);
        totalSSIM += ssim;

        // Convert SSIM to color (1.0 = white, 0.0 = black)
        int color = static_cast<int>((1.0 - ssim) * 255);
        color = std::clamp(color, 0, 255);
        QRgb value = qRgb(color, color, color);

        // Fill the window in the difference image
        for (int wy = 0; wy < WINDOW_SIZE && y + wy < height; ++wy) {
          for (int wx = 0; wx < WINDOW_SIZE && x + wx < width; ++wx) {
            diffImg.setPixel(x + wx, y + wy, value);
          }
        }

#pragma omp atomic
        ++processedWindows;

        // Update progress
        if (processedWindows % std::max(1, totalWindows / 20) == 0) {
          double progress =
              static_cast<double>(processedWindows) / totalWindows;
          promise.setProgressValue(static_cast<int>(progress * 100));
          diffLogger->debug("SSIM progress: {:.1f}%", progress * 100);
        }
      }
    }

    // Calculate final similarity percentage
    double numWindows = totalWindows > 0 ? totalWindows : 1.0;
    double similarity = (totalSSIM * 100.0) / numWindows;
    similarity = std::clamp(similarity, 0.0, 100.0);

    diffLogger->debug(
        "SSIM comparison - Average SSIM: {:.4f}, Similarity: {:.2f}%",
        totalSSIM / numWindows, similarity);

    return {diffImg, similarity, findDifferenceRegions(diffImg)};
  } catch (const std::exception &e) {
    diffLogger->error("Error in SSIM comparison: {}", e.what());
    promise.future().cancel();
    throw;
  }
}

double SSIMStrategy::computeSSIM(const QImage &img1, const QImage &img2, int x,
                                 int y) const noexcept {
  if (img1.isNull() || img2.isNull() || x < 0 || y < 0 ||
      x + WINDOW_SIZE > img1.width() || y + WINDOW_SIZE > img1.height() ||
      x + WINDOW_SIZE > img2.width() || y + WINDOW_SIZE > img2.height()) {
    return 0.0; // Invalid inputs
  }

  double mean1 = 0, mean2 = 0, variance1 = 0, variance2 = 0, covariance = 0;
  std::array<double, WINDOW_SIZE * WINDOW_SIZE> values1;
  std::array<double, WINDOW_SIZE * WINDOW_SIZE> values2;

  // First pass: calculate means
  int idx = 0;
  for (int wy = 0; wy < WINDOW_SIZE; ++wy) {
    for (int wx = 0; wx < WINDOW_SIZE; ++wx) {
      values1[idx] = qGray(img1.pixel(x + wx, y + wy));
      values2[idx] = qGray(img2.pixel(x + wx, y + wy));
      mean1 += values1[idx];
      mean2 += values2[idx];
      ++idx;
    }
  }

  const double windowSize = WINDOW_SIZE * WINDOW_SIZE;
  mean1 /= windowSize;
  mean2 /= windowSize;

  // Second pass: calculate variances and covariance
  for (size_t i = 0; i < values1.size(); ++i) {
    double diff1 = values1[i] - mean1;
    double diff2 = values2[i] - mean2;
    variance1 += diff1 * diff1;
    variance2 += diff2 * diff2;
    covariance += diff1 * diff2;
  }

  variance1 /= (windowSize - 1);
  variance2 /= (windowSize - 1);
  covariance /= (windowSize - 1);

  // Constants to stabilize division
  const double C1 = (K1 * 255) * (K1 * 255);
  const double C2 = (K2 * 255) * (K2 * 255);

  // SSIM formula
  double numerator = (2 * mean1 * mean2 + C1) * (2 * covariance + C2);
  double denominator =
      (mean1 * mean1 + mean2 * mean2 + C1) * (variance1 + variance2 + C2);

  if (denominator < 1e-10) {
    return 0.0; // Avoid division by zero
  }

  double ssim = numerator / denominator;
  return std::clamp(ssim, 0.0, 1.0); // Ensure value is in valid range
}

//////////////////////////////////////////////////////////////
// PerceptualHashStrategy method implementations
//////////////////////////////////////////////////////////////

ComparisonResult
PerceptualHashStrategy::compare(const QImage &img1, const QImage &img2,
                                QPromise<ComparisonResult> &promise) const {
  try {
    promise.setProgressValue(10);
    uint64_t hash1 = computeHash(img1);

    promise.setProgressValue(50);
    uint64_t hash2 = computeHash(img2);

    promise.setProgressValue(70);

    int distance = hammingDistance(hash1, hash2);
    double similarity =
        100.0 * (1.0 - static_cast<double>(distance) / HASH_SIZE);

    // Create difference visualization
    QImage diffImg(std::max(img1.width(), img2.width()),
                   std::max(img1.height(), img2.height()),
                   QImage::Format_ARGB32);
    QPainter painter(&diffImg);
    painter.fillRect(diffImg.rect(), Qt::white);

    if (distance > 0) {
      // Visualize bits that differ
      const int blockWidth = diffImg.width() / 8;
      const int blockHeight = diffImg.height() / 8;

      for (int i = 0; i < 64; ++i) {
        bool bit1 = (hash1 & (1ULL << i)) != 0;
        bool bit2 = (hash2 & (1ULL << i)) != 0;

        if (bit1 != bit2) {
          int x = (i % 8) * blockWidth;
          int y = (i / 8) * blockHeight;
          painter.fillRect(x, y, blockWidth, blockHeight,
                           QColor(255, 0, 0, 127));
        }
      }
    }

    promise.setProgressValue(90);

    diffLogger->debug(
        "Perceptual hash comparison - Distance: {}/64, Similarity: {:.2f}%",
        distance, similarity);

    // Create generic difference regions since exact pixel differences aren't
    // captured by pHash
    std::vector<QRect> regions;
    if (distance > 0) {
      const int regionSize = 50; // Approximate region size
      const int numRegions =
          std::min(5, 1 + distance / 10); // Limit number of regions

      for (int i = 0; i < numRegions; ++i) {
        int x = (diffImg.width() - regionSize) * (i + 1) / (numRegions + 1);
        int y = (diffImg.height() - regionSize) / 2;
        regions.push_back(QRect(x, y, regionSize, regionSize));
      }
    }

    promise.setProgressValue(100);
    return {diffImg, similarity, regions};
  } catch (const std::exception &e) {
    diffLogger->error("Error in perceptual hash comparison: {}", e.what());
    promise.future().cancel();
    throw;
  }
}

uint64_t PerceptualHashStrategy::computeHash(const QImage &img) const {
  if (img.isNull()) {
    throw std::invalid_argument("Cannot compute hash of null image");
  }

  try {
    // Resize to 8x8 grayscale image
    QImage scaled =
        img.scaled(8, 8, Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
            .convertToFormat(QImage::Format_Grayscale8);

    // Calculate average pixel value
    int sum = 0;
    std::array<int, 64> pixels;
    int idx = 0;

    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        pixels[idx] = qGray(scaled.pixel(x, y));
        sum += pixels[idx];
        ++idx;
      }
    }

    int avg = sum / 64;

    // Compute the hash: each bit is set if pixel >= average
    uint64_t hash = 0;
    for (int i = 0; i < 64; ++i) {
      hash = (hash << 1) | (pixels[i] >= avg ? 1 : 0);
    }

    return hash;
  } catch (const std::exception &e) {
    diffLogger->error("Error computing perceptual hash: {}", e.what());
    throw std::runtime_error(std::string("Error computing perceptual hash: ") +
                             e.what());
  }
}

int PerceptualHashStrategy::hammingDistance(uint64_t hash1,
                                            uint64_t hash2) const noexcept {
  // Use C++20's std::popcount for efficient bit counting
  return std::popcount(hash1 ^ hash2);
}

//////////////////////////////////////////////////////////////
// HistogramStrategy method implementations
//////////////////////////////////////////////////////////////

ComparisonResult
HistogramStrategy::compare(const QImage &img1, const QImage &img2,
                           QPromise<ComparisonResult> &promise) const {
  try {
    promise.setProgressValue(10);
    auto hist1 = computeHistogram(img1);

    promise.setProgressValue(40);
    auto hist2 = computeHistogram(img2);

    promise.setProgressValue(70);
    double similarity = compareHistograms(hist1, hist2);

    // Create difference visualization as a histogram chart
    QImage diffImg(std::max(img1.width(), img2.width()),
                   std::max(img1.height(), img2.height()),
                   QImage::Format_ARGB32);

    // Draw histograms using smart pointer to ensure proper cleanup
    {
      std::unique_ptr<QPainter> painter = std::make_unique<QPainter>(&diffImg);
      painter->fillRect(diffImg.rect(), Qt::white);

      // Draw histogram background grid
      painter->setPen(QPen(QColor(200, 200, 200), 1, Qt::DotLine));
      for (int i = 0; i < 4; ++i) {
        int y = diffImg.height() * i / 4;
        painter->drawLine(0, y, diffImg.width(), y);
      }

      // Calculate normalization factor for display
      int maxHistValue1 = *std::max_element(hist1.begin(), hist1.end());
      int maxHistValue2 = *std::max_element(hist2.begin(), hist2.end());
      double normFactor = static_cast<double>(diffImg.height()) /
                          (std::max(maxHistValue1, maxHistValue2) * 1.1);

      // Draw first histogram (blue)
      painter->setPen(QPen(QColor(0, 0, 255, 200), 2));
      for (int i = 0; i < HIST_BINS - 1; ++i) {
        int x1 = i * diffImg.width() / HIST_BINS;
        int x2 = (i + 1) * diffImg.width() / HIST_BINS;
        int y1 = diffImg.height() - static_cast<int>(hist1[i] * normFactor);
        int y2 = diffImg.height() - static_cast<int>(hist1[i + 1] * normFactor);
        painter->drawLine(x1, y1, x2, y2);
      }

      // Draw second histogram (red)
      painter->setPen(QPen(QColor(255, 0, 0, 200), 2));
      for (int i = 0; i < HIST_BINS - 1; ++i) {
        int x1 = i * diffImg.width() / HIST_BINS;
        int x2 = (i + 1) * diffImg.width() / HIST_BINS;
        int y1 = diffImg.height() - static_cast<int>(hist2[i] * normFactor);
        int y2 = diffImg.height() - static_cast<int>(hist2[i + 1] * normFactor);
        painter->drawLine(x1, y1, x2, y2);
      }

      // Draw legend
      QRect legendRect(10, 10, 200, 40);
      painter->fillRect(legendRect, QColor(255, 255, 255, 200));
      painter->setPen(Qt::black);
      painter->drawRect(legendRect);

      painter->setPen(Qt::blue);
      painter->drawText(20, 30, "Image 1");
      painter->fillRect(100, 22, 20, 10, QColor(0, 0, 255, 200));

      painter->setPen(Qt::red);
      painter->drawText(130, 30, "Image 2");
      painter->fillRect(190, 22, 20, 10, QColor(255, 0, 0, 200));
    }

    promise.setProgressValue(100);

    // Convert similarity to percentage
    double similarityPercent = similarity * 100.0;

    diffLogger->debug(
        "Histogram comparison - Correlation: {:.4f}, Similarity: {:.2f}%",
        similarity, similarityPercent);

    return {diffImg, similarityPercent, {}};
  } catch (const std::exception &e) {
    diffLogger->error("Error in histogram comparison: {}", e.what());
    promise.future().cancel();
    throw;
  }
}

std::vector<int> HistogramStrategy::computeHistogram(const QImage &img) const {
  if (img.isNull()) {
    throw std::invalid_argument("Cannot compute histogram of null image");
  }

  try {
    std::vector<int> histogram(HIST_BINS, 0);
    std::vector<std::thread> threads;
    const unsigned int threadCount = std::thread::hardware_concurrency();
    std::vector<std::vector<int>> threadHistograms(
        threadCount, std::vector<int>(HIST_BINS, 0));

    // Split work among threads
    for (unsigned int t = 0; t < threadCount; ++t) {
      threads.emplace_back([&img, t, threadCount, &threadHistograms]() {
        for (int y = t; y < img.height(); y += threadCount) {
          for (int x = 0; x < img.width(); ++x) {
            int gray = qGray(img.pixel(x, y));
            if (gray >= 0 && gray < HIST_BINS) {
              ++threadHistograms[t][gray];
            }
          }
        }
      });
    }

    // Join threads and combine results
    for (auto &thread : threads) {
      thread.join();
    }

    // Merge thread-local histograms
    for (const auto &threadHist : threadHistograms) {
      for (int i = 0; i < HIST_BINS; ++i) {
        histogram[i] += threadHist[i];
      }
    }

    return histogram;
  } catch (const std::exception &e) {
    diffLogger->error("Error computing histogram: {}", e.what());
    throw std::runtime_error(std::string("Error computing histogram: ") +
                             e.what());
  }
}

double HistogramStrategy::compareHistograms(
    const std::vector<int> &hist1,
    const std::vector<int> &hist2) const noexcept {
  if (hist1.size() != hist2.size() || hist1.empty()) {
    return 0.0;
  }

  try {
    // Calculate correlation using optimized algorithm
    double correlation = 0;
    double norm1 = 0, norm2 = 0;

#pragma omp parallel for simd reduction(+ : correlation, norm1, norm2)
    for (size_t i = 0; i < hist1.size(); ++i) {
      correlation += static_cast<double>(hist1[i]) * hist2[i];
      norm1 += static_cast<double>(hist1[i]) * hist1[i];
      norm2 += static_cast<double>(hist2[i]) * hist2[i];
    }

    if (norm1 < 1e-10 || norm2 < 1e-10) {
      return 0.0; // Avoid division by zero
    }

    return correlation / (std::sqrt(norm1) * std::sqrt(norm2));
  } catch (...) {
    return 0.0; // Safely handle any unexpected errors
  }
}