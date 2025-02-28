#pragma once

#include <concepts>  // C++20 concepts
#include <coroutine> // C++20 coroutines
#include <span>      // C++20 span
#include <stdexcept> // Standard exceptions
#include <vector>

#include <QFuture>
#include <QImage>
#include <QPromise>
#include <QRect>
#include <QString>

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

  // Add validity check
  [[nodiscard]] bool isValid() const noexcept {
    return !differenceImage.isNull() && similarityPercent >= 0.0 &&
           similarityPercent <= 100.0;
  }
};

/**
 * @brief Concept to define a comparison strategy.
 */
template <typename T>
concept ComparisonStrategy = requires(T s, const QImage &a, const QImage &b,
                                      QPromise<ComparisonResult> &p) {
  { s.compare(a, b, p) } -> std::same_as<ComparisonResult>;
  { s.name() } -> std::convertible_to<QString>;
  {
    s.name()
  } noexcept -> std::convertible_to<QString>; // Ensuring name() is noexcept
};

/**
 * @brief Unpacks a QRgb value into its individual components.
 *
 * @param rgb The QRgb value.
 * @return A tuple containing the red, green, and blue components.
 */
[[nodiscard]] constexpr std::tuple<int, int, int> qUnpack(QRgb rgb) noexcept;

namespace ColorSpace {

/**
 * @brief Struct to represent a color in the CIELAB color space.
 */
struct CIELAB {
  double L; ///< Lightness component.
  double a; ///< Green-Red component.
  double b; ///< Blue-Yellow component.

  // Add equality comparison
  bool operator==(const CIELAB &other) const noexcept {
    constexpr double epsilon = 1e-6;
    return std::abs(L - other.L) < epsilon && std::abs(a - other.a) < epsilon &&
           std::abs(b - other.b) < epsilon;
  }

  // Add inequality comparison
  bool operator!=(const CIELAB &other) const noexcept {
    return !(*this == other);
  }
};

/**
 * @brief Converts an RGB color to the CIELAB color space.
 *
 * @param rgb The RGB color.
 * @return The corresponding CIELAB color.
 */
[[nodiscard]] CIELAB RGB2LAB(QRgb rgb) noexcept;

} // namespace ColorSpace

template <typename T> struct Task {
  struct promise_type {
    T result;
    std::exception_ptr exception;

    Task get_return_object() noexcept {
      return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }

    void unhandled_exception() noexcept {
      exception = std::current_exception();
    }

    template <std::convertible_to<T> U> void return_value(U &&value) noexcept {
      result = std::forward<U>(value);
    }
  };

  std::coroutine_handle<promise_type> handle;

  Task(std::coroutine_handle<promise_type> h) : handle(h) {}
  Task(Task &&t) noexcept : handle(t.handle) { t.handle = nullptr; }
  ~Task() {
    if (handle)
      handle.destroy();
  }

  T result() const {
    if (handle.promise().exception)
      std::rethrow_exception(handle.promise().exception);
    return handle.promise().result;
  }
};

// ComparisonResult特化紧跟在通用模板之后
template <> struct Task<ComparisonResult> {
  struct promise_type {
    ComparisonResult result;
    std::exception_ptr exception;

    Task<ComparisonResult> get_return_object() noexcept {
      return Task<ComparisonResult>{
          std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }

    void unhandled_exception() noexcept {
      exception = std::current_exception();
    }

    void return_value(ComparisonResult &&value) noexcept {
      result = std::move(value);
    }

    void return_value(const ComparisonResult &value) noexcept {
      result = value;
    }
  };

  std::coroutine_handle<promise_type> handle;

  Task(std::coroutine_handle<promise_type> h) : handle(h) {}
  Task(Task &&t) noexcept : handle(t.handle) { t.handle = nullptr; }
  ~Task() {
    if (handle)
      handle.destroy();
  }

  Task(const Task &) = delete;
  Task &operator=(const Task &) = delete;

  [[nodiscard]] ComparisonResult result() const {
    if (handle.promise().exception) {
      std::rethrow_exception(handle.promise().exception);
    }
    return handle.promise().result;
  }
};

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
   * @throws std::invalid_argument If images are invalid
   * @throws std::runtime_error If comparison fails
   */
  template <ComparisonStrategy Strategy>
  ComparisonResult compare(const QImage &img1, const QImage &img2,
                           Strategy &&strategy,
                           QPromise<ComparisonResult> &promise) {
    if (!validateImages(img1, img2)) {
      promise.future().cancel();
      throw std::invalid_argument("Invalid images for comparison");
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
      throw std::runtime_error(std::string("Comparison failed: ") + e.what());
    }
  }

  // C++20 coroutine task generator for async comparison
  template <ComparisonStrategy Strategy>
  [[nodiscard]] Task<ComparisonResult>
  compareAsync(const QImage &img1, const QImage &img2, Strategy &&strategy,
               QPromise<ComparisonResult> &promise) {
    co_return compare(img1, img2, std::forward<Strategy>(strategy), promise);
  }

private:
  /**
   * @brief Validates the input images.
   *
   * @param img1 The first image.
   * @param img2 The second image.
   * @return True if the images are valid for comparison, false otherwise.
   */
  [[nodiscard]] bool validateImages(const QImage &img1,
                                    const QImage &img2) noexcept;

  /**
   * @brief Post-processes the comparison result.
   *
   * @param result The comparison result to post-process.
   */
  void postProcessResult(ComparisonResult &result) const noexcept;
};

/**
 * @brief Processes rows of an image.
 *
 * @param img The image to process.
 * @param height The height of the image.
 * @param fn The function to apply to each row.
 * @throws std::invalid_argument If image is invalid
 */
void processRows(const QImage &img, int height,
                 const std::function<void(int)> &fn);

/**
 * @brief Base class for image comparison strategies
 */
class ComparisonStrategyBase {
protected:
  static constexpr int BLOCK_SIZE = 16;
  static constexpr int SUBSAMPLE_FACTOR = 2;

  [[nodiscard]] QImage preprocessImage(const QImage &img) const;
  void compareBlockSIMD(std::span<const uchar> block1,
                        std::span<const uchar> block2,
                        std::span<uchar> dest) const noexcept;
  [[nodiscard]] std::vector<QRect>
  findDifferenceRegions(const QImage &diffImg) const;

  class DisjointSet {
    std::vector<int> parent;
    std::vector<int> rank;

  public:
    explicit DisjointSet(int size);
    [[nodiscard]] int find(int x) noexcept;
    void unite(int x, int y) noexcept;
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
   * @throws std::runtime_error If comparison fails
   */
  [[nodiscard]] ComparisonResult
  compare(const QImage &img1, const QImage &img2,
          QPromise<ComparisonResult> &promise) const;

  /**
   * @brief Gets the name of the strategy.
   *
   * @return The name of the strategy.
   */
  [[nodiscard]] QString name() const noexcept {
    return "Pixel Difference Comparison";
  }
};

/**
 * @brief Structural Similarity Index (SSIM) comparison strategy
 */
class SSIMStrategy : public ComparisonStrategyBase {
public:
  [[nodiscard]] ComparisonResult
  compare(const QImage &img1, const QImage &img2,
          QPromise<ComparisonResult> &promise) const;
  [[nodiscard]] QString name() const noexcept {
    return "Structural Similarity Comparison";
  }

private:
  static constexpr double K1 = 0.01;
  static constexpr double K2 = 0.03;
  static constexpr int WINDOW_SIZE = 8;

  [[nodiscard]] double computeSSIM(const QImage &img1, const QImage &img2,
                                   int x, int y) const noexcept;
};

/**
 * @brief Perceptual Hash (pHash) comparison strategy
 */
class PerceptualHashStrategy : public ComparisonStrategyBase {
public:
  [[nodiscard]] ComparisonResult
  compare(const QImage &img1, const QImage &img2,
          QPromise<ComparisonResult> &promise) const;
  [[nodiscard]] QString name() const noexcept {
    return "Perceptual Hash Comparison";
  }

private:
  static constexpr int HASH_SIZE = 64;
  [[nodiscard]] uint64_t computeHash(const QImage &img) const;
  [[nodiscard]] int hammingDistance(uint64_t hash1,
                                    uint64_t hash2) const noexcept;
};

/**
 * @brief Color histogram-based comparison strategy
 */
class HistogramStrategy : public ComparisonStrategyBase {
public:
  [[nodiscard]] ComparisonResult
  compare(const QImage &img1, const QImage &img2,
          QPromise<ComparisonResult> &promise) const;
  [[nodiscard]] QString name() const noexcept {
    return "Color Histogram Comparison";
  }

private:
  static constexpr int HIST_BINS = 256;
  [[nodiscard]] std::vector<int> computeHistogram(const QImage &img) const;
  [[nodiscard]] double
  compareHistograms(const std::vector<int> &hist1,
                    const std::vector<int> &hist2) const noexcept;
};
