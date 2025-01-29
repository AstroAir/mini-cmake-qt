#pragma once

#include <optional>
#include <span>

#include <opencv2/core.hpp>

namespace GaussianFit {

struct DataPoint {
  double x;
  double y;
};

struct GaussianParams {
  double base;   // Baseline level
  double peak;   // Peak height
  double center; // Center position
  double width;  // Gaussian width (sigma)

  constexpr bool valid() const noexcept {
    return width > 0.0 && peak > 0.0 && std::isfinite(base);
  }
};

class GaussianFitter {
public:
  static std::optional<GaussianParams> fit(std::span<const DataPoint> points,
                                           double epsilon = 1e-6,
                                           int max_iterations = 100);

  static double evaluate(const GaussianParams &params, double x) noexcept;

  static void visualize(std::span<const DataPoint> points,
                        const GaussianParams &params,
                        std::string_view window_name = "Gaussian Fit");

private:
  static void compute_residuals(const cv::Mat &params,
                                std::span<const DataPoint> points,
                                cv::Mat &residuals);

  static void compute_jacobian(const cv::Mat &params,
                               std::span<const DataPoint> points,
                               cv::Mat &jacobian);

  static bool validate_parameters(const cv::Mat &params);
};

} // namespace GaussianFit