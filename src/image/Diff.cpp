#include "Diff.hpp"

#include <algorithm>
#include <atomic>
#include <map>
#include <ranges>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <vector>
#include <immintrin.h>  // SIMD指令

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

ComparisonResult
PixelDifferenceStrategy::compare(const QImage &img1, const QImage &img2,
                                QPromise<ComparisonResult> &promise) const {
    // 预处理图像
    QImage processed1 = preprocessImage(img1);
    QImage processed2 = preprocessImage(img2);
    
    QImage diffImg(processed1.size(), QImage::Format_ARGB32);
    std::atomic_uint64_t totalDiff{0};
    std::atomic_int progress{0};

    const int height = processed1.height();
    const int width = processed1.width();
    const int bytesPerLine = processed1.bytesPerLine();

    // OpenMP并行处理
    #pragma omp parallel for collapse(2) reduction(+:totalDiff) schedule(dynamic)
    for (int y = 0; y < height; y += BLOCK_SIZE) {
        for (int x = 0; x < width; x += BLOCK_SIZE) {
            if (promise.isCanceled()) continue;

            const int blockHeight = std::min(BLOCK_SIZE, height - y);
            const int blockWidth = std::min(BLOCK_SIZE, width - x);

            for (int by = 0; by < blockHeight; ++by) {
                const uchar* line1 = processed1.scanLine(y + by) + x * 4;
                const uchar* line2 = processed2.scanLine(y + by) + x * 4;
                uchar* dest = diffImg.scanLine(y + by) + x * 4;
                
                compareBlockSIMD(line1, line2, dest, blockWidth * 4);
            }
            
            #pragma omp atomic
            progress += blockHeight;
        }
        
        if (progress % (height/10) == 0) {
            promise.setProgressValue(static_cast<int>(progress * 100.0 / height));
        }
    }

    // 找出差异区域
    auto regions = findDifferenceRegions(diffImg);
    
    // 计算相似度
    double mse = static_cast<double>(totalDiff) / (width * height * 4);
    double similarity = 100.0 * (1.0 - std::sqrt(mse) / 255.0);

    return {diffImg, similarity, regions};
}

void PixelDifferenceStrategy::compareBlockSIMD(const uchar* block1, 
                                              const uchar* block2,
                                              uchar* dest, 
                                              size_t size) const {
    size_t i = 0;
    // 使用AVX2指令集进行SIMD优化
    #ifdef __AVX2__
    for (; i + 32 <= size; i += 32) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block1 + i));
        __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block2 + i));
        __m256i diff = _mm256_sub_epi8(_mm256_max_epu8(a, b), 
                                      _mm256_min_epu8(a, b));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest + i), diff);
    }
    #endif

    // 处理剩余的像素
    for (; i < size; ++i) {
        dest[i] = std::abs(block1[i] - block2[i]);
    }
}

QImage PixelDifferenceStrategy::preprocessImage(const QImage& img) const {
    // 降采样以减少计算量
    if (SUBSAMPLE_FACTOR > 1) {
        return img.scaled(img.width() / SUBSAMPLE_FACTOR, 
                         img.height() / SUBSAMPLE_FACTOR,
                         Qt::IgnoreAspectRatio, 
                         Qt::FastTransformation);
    }
    return img;
}

std::vector<QRect>
PixelDifferenceStrategy::findDifferenceRegions(const QImage &diffImg) const {
    const int width = diffImg.width();
    const int height = diffImg.height();
    const int threshold = 32;
    
    // 使用并查集优化连通区域分析
    DisjointSet ds(width * height);
    std::vector<std::pair<int, int>> diffPoints;
    std::map<int, QRect> regionMap;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (qGray(diffImg.pixel(x, y)) > threshold) {
                diffPoints.emplace_back(x, y);
                const int idx = y * width + x;
                
                // 检查4个方向，使用std::pair正确构造
                std::array<std::pair<int,int>, 4> directions = {
                    std::make_pair(-1, 0),
                    std::make_pair(0, -1),
                    std::make_pair(1, 0),
                    std::make_pair(0, 1)
                };
                
                for (const auto& [dx, dy] : directions) {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                        qGray(diffImg.pixel(nx, ny)) > threshold) {
                        ds.unite(idx, ny * width + nx);
                    }
                }
            }
        }
    }
    
    // 使用迭代器构建返回值
    std::vector<QRect> result;
    for (const auto& [setId, rect] : regionMap) {
        result.push_back(rect);
    }
    
    return result;
}

PixelDifferenceStrategy::DisjointSet::DisjointSet(int size) 
    : parent(size), rank(size, 0) {
    std::iota(parent.begin(), parent.end(), 0);
}

int PixelDifferenceStrategy::DisjointSet::find(int x) {
    if (parent[x] != x) {
        parent[x] = find(parent[x]); // 路径压缩
    }
    return parent[x];
}

void PixelDifferenceStrategy::DisjointSet::unite(int x, int y) {
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