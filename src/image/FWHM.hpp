#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <span>


namespace GaussianFit {

/**
 * @struct DataPoint
 * @brief Structure representing a data point with x and y coordinates.
 */
struct DataPoint {
  double x; ///< The x-coordinate of the data point.
  double y; ///< The y-coordinate of the data point.
};

/**
 * @struct GaussianParams
 * @brief Structure representing the parameters of a Gaussian function.
 */
struct GaussianParams {
  double base;   ///< Baseline level.
  double peak;   ///< Peak height.
  double center; ///< Center position.
  double width;  ///< Gaussian width (sigma).

  /**
   * @brief Checks if the Gaussian parameters are valid.
   * @return True if the parameters are valid, false otherwise.
   */
  constexpr bool valid() const noexcept {
    return width > 0.0 && peak > 0.0 && std::isfinite(base);
  }
};

/**
 * @class GaussianFitter
 * @brief Class for fitting a Gaussian function to data points.
 */
class GaussianFitter {
public:
  /**
   * @brief Fits a Gaussian function to the given data points.
   * @param points The data points to fit the Gaussian function to.
   * @param epsilon The convergence threshold for the fitting algorithm.
   * @param max_iterations The maximum number of iterations for the fitting
   * algorithm.
   * @return An optional GaussianParams structure representing the fitted
   * parameters.
   */
  static std::optional<GaussianParams> fit(std::span<const DataPoint> points,
                                           double epsilon = 1e-6,
                                           int max_iterations = 100);

  /**
   * @brief Evaluates the Gaussian function at a given x-coordinate.
   * @param params The parameters of the Gaussian function.
   * @param x The x-coordinate to evaluate the function at.
   * @return The value of the Gaussian function at the given x-coordinate.
   */
  static double evaluate(const GaussianParams &params, double x) noexcept;

  /**
   * @brief Visualizes the Gaussian fit along with the data points.
   * @param points The data points.
   * @param params The parameters of the Gaussian function.
   * @param window_name The name of the window for visualization.
   */
  static void visualize(std::span<const DataPoint> points,
                        const GaussianParams &params,
                        std::string_view window_name = "Gaussian Fit");

private:
  /**
   * @brief Computes the residuals between the data points and the Gaussian
   * function.
   * @param params The parameters of the Gaussian function.
   * @param points The data points.
   * @param residuals The output residuals.
   */
  static void compute_residuals(const cv::Mat &params,
                                std::span<const DataPoint> points,
                                cv::Mat &residuals);

  /**
   * @brief Computes the Jacobian matrix for the Gaussian function.
   * @param params The parameters of the Gaussian function.
   * @param points The data points.
   * @param jacobian The output Jacobian matrix.
   */
  static void compute_jacobian(const cv::Mat &params,
                               std::span<const DataPoint> points,
                               cv::Mat &jacobian);

  /**
   * @brief Validates the Gaussian parameters.
   * @param params The parameters of the Gaussian function.
   * @return True if the parameters are valid, false otherwise.
   */
  static bool validate_parameters(const cv::Mat &params);
};

} // namespace GaussianFit