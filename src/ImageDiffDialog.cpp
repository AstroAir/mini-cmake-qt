#include <algorithm>
#include <atomic>
#include <execution>
#include <ranges>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>
#include <stack>
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

struct ComparisonResult {
  QImage differenceImage;
  double similarityPercent = 0.0;
  std::vector<QRect> differenceRegions;
  std::chrono::milliseconds duration;
};

// 在命名空间外部定义concept
template <typename T>
concept ComparisonStrategy = requires(T s, const QImage &a, const QImage &b,
                                      QPromise<ComparisonResult> &p) {
  { s.compare(a, b, p) } -> std::same_as<ComparisonResult>;
  { s.name() } -> std::same_as<QString>;
};

namespace ImageComparison {
// 颜色解包辅助函数
inline std::tuple<int, int, int> qUnpack(QRgb rgb) {
  return {qRed(rgb), qGreen(rgb), qBlue(rgb)};
}

namespace ColorSpace {
struct CIELAB {
  double L, a, b;
};

CIELAB RGB2LAB(QRgb rgb) {
  const auto [r, g, b] = qUnpack(rgb);
  return {0.2126 * r + 0.7152 * g + 0.0722 * b, static_cast<double>(r - g),
          static_cast<double>(g - b)};
}
} // namespace ColorSpace

class ImageComparator {
public:
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
  bool validateImages(const QImage &img1, const QImage &img2) {
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

  void postProcessResult(ComparisonResult &result) const {
    if (!result.differenceImage.isNull()) {
      auto bits = result.differenceImage.bits();
      const auto [min, max] = std::minmax_element(
          bits, bits + result.differenceImage.sizeInBytes());

      const float scale = 255.0f / (*max - *min + 1e-5f);
      std::transform(std::execution::par_unseq, bits,
                     bits + result.differenceImage.sizeInBytes(), bits,
                     [=](auto val) { return (val - *min) * scale; });
    }
  }
};

// 基本像素差异策略
void processRows(const QImage &img, int height,
                 const std::function<void(int)> &fn) {
  auto rows = std::views::iota(0, height);
  std::ranges::for_each(rows, fn);
}

class PixelDifferenceStrategy {
public:
  ComparisonResult compare(const QImage &img1, const QImage &img2,
                           QPromise<ComparisonResult> &promise) const {
    QImage diffImg(img1.size(), QImage::Format_ARGB32);
    std::atomic_uint64_t totalDiff = 0;
    std::atomic_int progress = 0;

    const int height = img1.height();
    const int bytesPerLine = img1.bytesPerLine();

    processRows(img1, height, [&](int y) {
      if (promise.isCanceled())
        return;

      const uchar *line1 = img1.scanLine(y);
      const uchar *line2 = img2.scanLine(y);
      uchar *dest = diffImg.scanLine(y);

      for (int x = 0; x < bytesPerLine; ++x) {
        dest[x] = std::abs(line1[x] - line2[x]);
        totalDiff += dest[x];
      }

      if (++progress % 10 == 0) {
        promise.setProgressValue(static_cast<int>(progress * 100.0 / height));
      }
    });

    const double similarity =
        1.0 - static_cast<double>(totalDiff) /
                  (img1.width() * img1.height() * 255 * 4);

    return {diffImg, similarity * 100, findDifferenceRegions(diffImg)};
  }

  QString name() const { return "Pixel Difference"; }

private:
  std::vector<QRect> findDifferenceRegions(const QImage &diffImg) const {
    // 连通区域分析实现
    std::vector<QRect> regions;
    QImage visited(diffImg.size(), QImage::Format_Mono);
    visited.fill(0);

    const int threshold = 32;
    const QPoint directions[] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (int y = 0; y < diffImg.height(); ++y) {
      for (int x = 0; x < diffImg.width(); ++x) {
        if (qGray(diffImg.pixel(x, y)) > threshold &&
            !visited.pixelIndex(x, y)) {
          QRect region;
          floodFill(diffImg, visited, x, y, threshold, region);
          regions.push_back(region);
        }
      }
    }

    return regions;
  }

  void floodFill(const QImage &img, QImage &visited, int x, int y,
                 int threshold, QRect &region) const {
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
            !visited.pixelIndex(nx, ny) &&
            qGray(img.pixel(nx, ny)) > threshold) {
          visited.setPixel(nx, ny, 1);
          stack.emplace(nx, ny);
        }
      }
    }

    region = QRect(QPoint(minX, minY), QPoint(maxX, maxY));
  }
};

// 感知差异策略
class PerceptualDifferenceStrategy {
public:
  ComparisonResult compare(const QImage &img1, const QImage &img2,
                           QPromise<ComparisonResult> &promise) const {
    QImage diffImg(img1.size(), QImage::Format_ARGB32);
    std::atomic<double> totalError = 0.0;
    std::atomic_int progress = 0;

    const int height = img1.height();
    const int width = img1.width();

    std::for_each(std::execution::par, std::views::iota(0, height).begin(),
                  std::views::iota(0, height).end(), [&](int y) {
                    if (promise.isCanceled())
                      return;

                    for (int x = 0; x < width; ++x) {
                      const auto lab1 = ColorSpace::RGB2LAB(img1.pixel(x, y));
                      const auto lab2 = ColorSpace::RGB2LAB(img2.pixel(x, y));

                      const double deltaL = lab1.L - lab2.L;
                      const double deltaA = lab1.a - lab2.a;
                      const double deltaB = lab1.b - lab2.b;

                      const double error = std::sqrt(
                          deltaL * deltaL + deltaA * deltaA + deltaB * deltaB);

                      totalError += error;
                      const uint8_t value = static_cast<uint8_t>(
                          std::clamp(error * 10.0, 0.0, 255.0));
                      diffImg.setPixel(x, y, qRgb(value, value, value));
                    }

                    if (++progress % 5 == 0) {
                      promise.setProgressValue(
                          static_cast<int>(progress * 100.0 / height));
                    }
                  });

    const double maxError = width * height * 100.0; // 最大可能误差
    const double similarity = 1.0 - (totalError / maxError);

    return {diffImg, similarity * 100, findDifferenceRegions(diffImg)};
  }

  QString name() const { return "Perceptual Difference"; }

private:
  std::vector<QRect> findDifferenceRegions(const QImage &diffImg) const {
    // Same region-detection logic as in PixelDifferenceStrategy
    std::vector<QRect> regions;
    QImage visited(diffImg.size(), QImage::Format_Mono);
    visited.fill(0);

    const int threshold = 32;

    for (int y = 0; y < diffImg.height(); ++y) {
      for (int x = 0; x < diffImg.width(); ++x) {
        if (qGray(diffImg.pixel(x, y)) > threshold &&
            !visited.pixelIndex(x, y)) {
          QRect region;
          floodFill(diffImg, visited, x, y, threshold, region);
          regions.push_back(region);
        }
      }
    }
    return regions;
  }

  void floodFill(const QImage &img, QImage &visited, int x, int y,
                 int threshold, QRect &region) const {
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
            !visited.pixelIndex(nx, ny) &&
            qGray(img.pixel(nx, ny)) > threshold) {
          visited.setPixel(nx, ny, 1);
          stack.emplace(nx, ny);
        }
      }
    }

    region = QRect(QPoint(minX, minY), QPoint(maxX, maxY));
  }
};
} // namespace ImageComparison

template <typename Strategy>
ComparisonResult runComparison(const QImage &image1, const QImage &image2,
                               ImageComparison::ImageComparator comparator,
                               Strategy strategy) {
  QPromise<ComparisonResult> promise;
  auto future = promise.future();
  return comparator.compare(image1, image2, strategy, promise);
}

template <typename Strategy> struct ComparisonRunner {
  ComparisonResult operator()(const QImage &image1, const QImage &image2,
                              ImageComparison::ImageComparator comparator,
                              Strategy strategy) {
    return runComparison(image1, image2, comparator, strategy);
  }
};

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  MainWindow(QWidget *parent = nullptr) : QMainWindow(parent) {
    setupUI();
    setupConnections();
    setupLogger();
  }

private:
  void setupUI() {
    centralWidget = new QWidget();
    setCentralWidget(centralWidget);

    layout = new QGridLayout(centralWidget);

    btnLoad1 = new QPushButton(tr("Load Image 1"), centralWidget);
    btnLoad2 = new QPushButton(tr("Load Image 2"), centralWidget);
    btnCompare = new QPushButton(tr("Compare"), centralWidget);
    btnSave = new QPushButton(tr("Save Result"), centralWidget);

    lblImage1 = new QLabel(tr("Image 1"), centralWidget);
    lblImage2 = new QLabel(tr("Image 2"), centralWidget);
    lblDiff = new QLabel(tr("Difference"), centralWidget);
    lblResult = new QLabel(tr("Similarity: N/A"), centralWidget);

    progressBar = new QProgressBar(centralWidget);
    strategyCombo = new QComboBox(centralWidget);
    strategyCombo->addItems({"Pixel Difference", "Perceptual Difference"});

    // 布局设置
    layout->addWidget(btnLoad1, 0, 0);
    layout->addWidget(btnLoad2, 0, 1);
    layout->addWidget(strategyCombo, 0, 2);
    layout->addWidget(btnCompare, 0, 3);
    layout->addWidget(btnSave, 0, 4);

    layout->addWidget(lblImage1, 1, 0);
    layout->addWidget(lblImage2, 1, 1);
    layout->addWidget(lblDiff, 1, 2);
    layout->addWidget(lblResult, 2, 0, 1, 3);
    layout->addWidget(progressBar, 3, 0, 1, 5);

    setCentralWidget(centralWidget);
    resize(1200, 800);
  }

  void setupConnections() {
    connect(btnLoad1, &QPushButton::clicked, [this] { loadImage(1); });
    connect(btnLoad2, &QPushButton::clicked, [this] { loadImage(2); });
    connect(btnCompare, &QPushButton::clicked, this,
            &MainWindow::startComparison);
    connect(btnSave, &QPushButton::clicked, this, &MainWindow::saveResult);
  }

  void setupLogger() {
    auto logger = spdlog::rotating_logger_mt("app_logger", "comparison.log",
                                             1024 * 1024 * 5, 3);
    spdlog::set_default_logger(logger);
  }

  void loadImage(int index) {
    QString path = QFileDialog::getOpenFileName(this, "Open Image", "",
                                                "Images (*.png *.jpg *.bmp)");
    if (path.isEmpty())
      return;

    QImage image(path);
    if (image.isNull()) {
      QMessageBox::critical(this, "Error", "Failed to load image");
      return;
    }

    (index == 1 ? image1 : image2) =
        image.scaled(400, 400, Qt::KeepAspectRatio);
    QLabel *target = index == 1 ? lblImage1 : lblImage2;
    target->setPixmap(QPixmap::fromImage(image));
  }

  void startComparison() {
    if (image1.isNull() || image2.isNull()) {
      QMessageBox::warning(this, "Warning", "Please load both images first");
      return;
    }

    ImageComparison::ImageComparator comparator;
    std::function<ComparisonResult()> comparisonFunction;

    if (strategyCombo->currentText() == "Pixel Difference") {
      ImageComparison::PixelDifferenceStrategy strategy;
      comparisonFunction = [=]() {
        return ComparisonRunner<ImageComparison::PixelDifferenceStrategy>()(
            image1, image2, comparator, strategy);
      };
    } else {
      ImageComparison::PerceptualDifferenceStrategy strategy;
      comparisonFunction = [=]() {
        return ComparisonRunner<
            ImageComparison::PerceptualDifferenceStrategy>()(
            image1, image2, comparator, strategy);
      };
    }

    future = QtConcurrent::run(comparisonFunction);

    auto *watcher = new QFutureWatcher<ComparisonResult>(this);
    connect(watcher, &QFutureWatcherBase::progressValueChanged, progressBar,
            &QProgressBar::setValue);
    connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher] {
      btnCompare->setEnabled(true);
      auto result = watcher->result();
      showComparisonResult(result);
      watcher->deleteLater();
    });
    watcher->setFuture(future);
  }

  void showComparisonResult(const ComparisonResult &result) {
    lblDiff->setPixmap(QPixmap::fromImage(result.differenceImage));
    lblResult->setText(QString("Similarity: %1% | Time: %2ms")
                           .arg(result.similarityPercent, 0, 'f', 2)
                           .arg(result.duration.count()));

    // 绘制差异区域
    QImage markedImage = image1.copy();
    QPainter painter(&markedImage);
    painter.setPen(Qt::red);
    for (const auto &rect : result.differenceRegions) {
      painter.drawRect(rect);
    }
    lblImage1->setPixmap(QPixmap::fromImage(markedImage));
  }

  void saveResult() {
    QString path = QFileDialog::getSaveFileName(this, "Save Result", "",
                                                "PNG Image (*.png)");
    if (!path.isEmpty()) {
      lblDiff->pixmap().save(path);
    }
  }

  // UI成员变量
  QWidget *centralWidget;
  QGridLayout *layout;
  QPushButton *btnLoad1, *btnLoad2, *btnCompare, *btnSave;
  QLabel *lblImage1, *lblImage2, *lblDiff, *lblResult;
  QProgressBar *progressBar;
  QComboBox *strategyCombo;
  QImage image1, image2;
  QFuture<ComparisonResult> future;
};
