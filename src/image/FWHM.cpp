#include "FWHM.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <spdlog/spdlog.h>
#include <vector>

namespace GaussianFit {
namespace detail {
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

constexpr double EPSILON = 1e-10;

inline auto safe_division(auto numerator, auto denominator) {
  return denominator < EPSILON ? numerator / EPSILON : numerator / denominator;
}

cv::Mat create_optimization_matrix(std::size_t rows, std::size_t cols) {
  cv::Mat mat(static_cast<int>(rows), static_cast<int>(cols), CV_64F);
  if (mat.empty()) {
    throw std::bad_alloc();
  }
  return mat;
}

template <typename Container> class enumerate_iterator {
  using iterator = std::ranges::iterator_t<Container>;
  iterator it_;
  std::size_t index_;

public:
  enumerate_iterator(iterator it, std::size_t index) : it_(it), index_(index) {}

  auto operator*() const { return std::make_pair(index_, *it_); }
  enumerate_iterator &operator++() {
    ++it_;
    ++index_;
    return *this;
  }
  bool operator!=(const enumerate_iterator &other) const {
    return it_ != other.it_;
  }
};

template <typename Container> class enumerate_view {
  Container container_;

public:
  explicit enumerate_view(Container &&cont)
      : container_(std::forward<Container>(cont)) {}

  auto begin() {
    return enumerate_iterator<Container>(std::begin(container_), 0);
  }

  auto end() {
    return enumerate_iterator<Container>(std::end(container_),
                                         std::size(container_));
  }
};

template <typename Container> auto enumerate(Container &&cont) {
  return enumerate_view<Container>(std::forward<Container>(cont));
}

struct Statistics {
  double min;
  double max;
  double mean;
  double stddev;
};

// 修改 calculate_statistics 使用 OpenMP 并行归约计算 mean 和 variance
Statistics calculate_statistics(std::span<const DataPoint> points) {
  if (points.empty())
    return {};

  const std::size_t n = points.size();
  double sum_x = 0.0;
#pragma omp parallel for reduction(+ : sum_x)
  for (std::size_t i = 0; i < n; ++i) {
    sum_x += points[i].x;
  }
  const double mean_x = sum_x / n;

  double variance = 0.0;
#pragma omp parallel for reduction(+ : variance)
  for (std::size_t i = 0; i < n; ++i) {
    const double diff = points[i].x - mean_x;
    variance += diff * diff;
  }
  variance /= (n - 1);

  auto [min_it, max_it] = std::minmax_element(
      points.begin(), points.end(),
      [](const DataPoint &a, const DataPoint &b) { return a.y < b.y; });

  return Statistics{min_it->y, max_it->y, mean_x, std::sqrt(variance)};
}

#ifdef __AVX2__
inline __m256d gaussian_avx(__m256d x, __m256d center, __m256d width,
                            __m256d base, __m256d peak) {
  const __m256d t = _mm256_div_pd(_mm256_sub_pd(x, center), width);
  const __m256d neg_half = _mm256_set1_pd(-0.5);
  const __m256d exp_term =
      _mm256_exp_pd(_mm256_mul_pd(neg_half, _mm256_mul_pd(t, t)));
  return _mm256_add_pd(base, _mm256_mul_pd(peak, exp_term));
}
#endif

} // namespace detail

std::optional<GaussianParams>
GaussianFitter::fit(std::span<const DataPoint> points, double epsilon,
                    int max_iterations) {
  SPDLOG_INFO("Initializing Gaussian fit with {} points", points.size());

  if (points.empty()) {
    SPDLOG_ERROR("Empty input data points");
    return std::nullopt;
  }

  try {
    const auto stats = detail::calculate_statistics(points);
    SPDLOG_DEBUG("Calculated statistics: min={}, max={}, mean={}, stddev={}",
                 stats.min, stats.max, stats.mean, stats.stddev);

    cv::Mat params = (cv::Mat_<double>(4, 1) << stats.min,
                      stats.max - stats.min, stats.mean, stats.stddev * 2.0);

    cv::Mat residuals = detail::create_optimization_matrix(points.size(), 1);
    double prev_error = std::numeric_limits<double>::max();

    const double tau = 1e-3;
    double lambda = 1e-3;
    cv::Mat prev_params;

    for (int iter = 0; iter < max_iterations; ++iter) {
      compute_residuals(params, points, residuals);
      double current_error = cv::norm(residuals);

      SPDLOG_DEBUG("Iteration {:03d} - Error: {:.4e}", iter, current_error);

      if (std::abs(current_error - prev_error) < epsilon) {
        SPDLOG_INFO("Convergence achieved at iteration {}", iter);
        break;
      }

      if (current_error > prev_error) {
        SPDLOG_WARN("Error increasing at iteration {}", iter);
        break;
      }

      cv::Mat jacobian = detail::create_optimization_matrix(points.size(), 4);
      compute_jacobian(params, points, jacobian);

      cv::Mat JtJ = jacobian.t() * jacobian;
      cv::Mat JtR = jacobian.t() * residuals;

      cv::Mat diag = JtJ.diag();
      diag *= (1.0 + lambda);

      cv::Mat delta;
      if (cv::solve(JtJ, -JtR, delta, cv::DECOMP_CHOLESKY)) {
        prev_params = params.clone();
        params += delta;

        if (current_error < prev_error) {
          lambda = std::max(lambda / tau, 1e-7);
        } else {
          params = prev_params;
          lambda = std::min(lambda * tau, 1e7);
        }
      }

      prev_error = current_error;

      if (!validate_parameters(params)) {
        SPDLOG_WARN("Invalid parameters detected at iteration {}", iter);
        return std::nullopt;
      }
    }

    GaussianParams result{params.at<double>(0), params.at<double>(1),
                          params.at<double>(2), std::abs(params.at<double>(3))};

    if (!result.valid()) {
      SPDLOG_ERROR("Final parameters validation failed");
      return std::nullopt;
    }

    SPDLOG_INFO(
        "Fit successful: base={:.3f}, peak={:.3f}, center={:.3f}, width={:.3f}",
        result.base, result.peak, result.center, result.width);
    return result;

  } catch (const cv::Exception &e) {
    SPDLOG_CRITICAL("OpenCV exception: {}", e.what());
    return std::nullopt;
  } catch (const std::exception &e) {
    SPDLOG_CRITICAL("Standard exception: {}", e.what());
    return std::nullopt;
  }
}

double GaussianFitter::evaluate(const GaussianParams &params,
                                double x) noexcept {
#ifdef __AVX2__
  if (std::abs(params.width) > detail::EPSILON) {
    __m256d x_vec = _mm256_set1_pd(x);
    __m256d center_vec = _mm256_set1_pd(params.center);
    __m256d width_vec = _mm256_set1_pd(params.width);
    __m256d base_vec = _mm256_set1_pd(params.base);
    __m256d peak_vec = _mm256_set1_pd(params.peak);

    __m256d result =
        detail::gaussian_avx(x_vec, center_vec, width_vec, base_vec, peak_vec);
    double results[4];
    _mm256_store_pd(results, result);
    return results[0];
  }
#endif

  const double t = detail::safe_division(x - params.center, params.width);
  return params.base + params.peak * std::exp(-0.5 * t * t);
}

// ----------------------------------------------------------------
// 修改: 优化 visualize 中计算拟合曲线点的过程
void GaussianFitter::visualize(std::span<const DataPoint> points,
                               const GaussianParams &params,
                               std::string_view window_name) {
  const int plot_height = 600;
  const int plot_width = 800;
  const cv::Scalar background_color{245, 245, 245};
  const cv::Scalar data_point_color{25, 25, 25};
  const cv::Scalar curve_color{200, 50, 50};

  cv::Mat plot(plot_height, plot_width, CV_8UC3, background_color);

  // 计算 x 轴范围
  auto [x_min_it, x_max_it] = std::minmax_element(
      points.begin(), points.end(),
      [](const DataPoint &a, const DataPoint &b) { return a.x < b.x; });
  const double x_min_val = x_min_it->x;
  const double x_max_val = x_max_it->x;

  const double y_scale = detail::safe_division(plot_height * 0.9, params.peak);

  // 并行计算曲线点，预分配 vector 大小
  std::vector<cv::Point> curve_points(plot_width);
#pragma omp parallel for
  for (int px = 0; px < plot_width; ++px) {
    double x = detail::safe_division(px * (x_max_val - x_min_val), plot_width) +
               x_min_val;
    double y = evaluate(params, x);
    int py = static_cast<int>((params.base + params.peak - y) * y_scale);
    curve_points[px] = cv::Point(px, std::clamp(py, 0, plot_height - 1));
  }
  cv::polylines(plot, curve_points, false, curve_color, 2, cv::LINE_AA);

  // 绘制数据点 (顺序执行)
  for (const auto &p : points) {
    int px = static_cast<int>(detail::safe_division(
        (p.x - x_min_val) * plot_width, (x_max_val - x_min_val)));
    int py = static_cast<int>((params.base + params.peak - p.y) * y_scale);
    cv::circle(plot, {px, py}, 4, data_point_color, -1);
  }

  cv::namedWindow(window_name.data(), cv::WINDOW_AUTOSIZE);
  cv::imshow(window_name.data(), plot);
  cv::waitKey(1);
}

// ----------------------------------------------------------------
// 修改: 将 compute_residuals 并行化
void GaussianFitter::compute_residuals(const cv::Mat &params,
                                       std::span<const DataPoint> points,
                                       cv::Mat &residuals) {
  const GaussianParams p{params.at<double>(0), params.at<double>(1),
                         params.at<double>(2), params.at<double>(3)};
  if (!p.valid()) {
    throw std::invalid_argument("Invalid parameters in residual calculation");
  }
  const std::size_t n = points.size();
#pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    residuals.at<double>(i) = points[i].y - evaluate(p, points[i].x);
  }
}

// ----------------------------------------------------------------
// 修改: 将 compute_jacobian 并行化
void GaussianFitter::compute_jacobian(const cv::Mat &params,
                                      std::span<const DataPoint> points,
                                      cv::Mat &jacobian) {
  const double base = params.at<double>(0);
  const double peak = params.at<double>(1);
  const double center = params.at<double>(2);
  const double width = params.at<double>(3);

  if (width < detail::EPSILON) {
    throw std::invalid_argument("Invalid width in Jacobian calculation");
  }

  const std::size_t n = points.size();
#pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    const double x = points[i].x;
    const double delta = x - center;
    const double scaled_delta = detail::safe_division(delta, width);
    const double exp_term = std::exp(-0.5 * scaled_delta * scaled_delta);

    jacobian.at<double>(i, 0) = -1.0;
    jacobian.at<double>(i, 1) = -exp_term;
    jacobian.at<double>(i, 2) = peak * exp_term * scaled_delta / width;
    jacobian.at<double>(i, 3) =
        peak * exp_term * scaled_delta * scaled_delta / width;
  }
}

bool GaussianFitter::validate_parameters(const cv::Mat &params) {
  const double width = params.at<double>(3);
  if (width < detail::EPSILON) {
    SPDLOG_WARN("Invalid width parameter: {:.4e}", width);
    return false;
  }
  return true;
}

} // namespace GaussianFit