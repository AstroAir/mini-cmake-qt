#ifndef IMAGESHARPENER_HPP
#define IMAGESHARPENER_HPP

#include <QtGui/QImage>
#include <opencv2/opencv.hpp>
#include <type_traits>
#include <variant>


// The ImageSharpener class provides several image sharpening algorithms.
// In addition to the original Laplace, Unsharp Mask, and Custom Kernel methods,
// we have added a High-Boost filtering method to further enhance details.
class ImageSharpener {
public:
  enum class Method {
    Laplace,
    UnsharpMask,
    HighBoost, // New: High-Boost filtering method
    CustomKernel
  };

  // Parameters for Laplace filtering.
  struct LaplaceParams {
    double scale = 1.0;
    double delta = 0.0;
    int ddepth = CV_16S;
  };

  // Parameters for Unsharp Mask filtering.
  struct UnsharpMaskParams {
    double sigma = 1.0;
    double amount = 1.0;
    int radius = 5; // Must be odd
  };

  // Parameters for High-Boost filtering.
  // High-Boost filtering is similar to Unsharp Mask filtering but applies a
  // boost factor greater than 1.0.
  struct HighBoostParams {
    double sigma = 1.0;
    double boostFactor =
        1.5;        // Must be >= 1.0. Higher value gives stronger sharpening.
    int radius = 5; // Must be odd
  };

  // Parameters for a custom kernel based filtering.
  struct CustomKernelParams {
    cv::Mat kernel;
    double delta = 0.0;
  };

  // Constructor sets the sharpening method.
  explicit ImageSharpener(Method method = Method::UnsharpMask);

  // Main processing function; applies the sharpening algorithm.
  cv::Mat sharpen(const cv::Mat &input);

  // Template function to set parameters (only supports the defined parameter
  // types).
  template <typename T> void setParameters(T params) {
    static_assert(std::is_same_v<std::decay_t<T>, LaplaceParams> ||
                      std::is_same_v<std::decay_t<T>, UnsharpMaskParams> ||
                      std::is_same_v<std::decay_t<T>, HighBoostParams> ||
                      std::is_same_v<std::decay_t<T>, CustomKernelParams>,
                  "Unsupported parameter type");
    m_params = std::forward<T>(params);
  }

  // Converts a cv::Mat to a QImage (useful for Qt applications).
  static QImage matToQImage(const cv::Mat &mat);

private:
  Method m_method;
  std::variant<LaplaceParams, UnsharpMaskParams, HighBoostParams,
               CustomKernelParams>
      m_params;

  void validateInput(const cv::Mat &input);

  // Sharpening implementations
  cv::Mat applyLaplace(const cv::Mat &input, const LaplaceParams &params);
  cv::Mat applyUnsharpMask(const cv::Mat &input,
                           const UnsharpMaskParams &params);
  cv::Mat applyHighBoost(const cv::Mat &input, const HighBoostParams &params);
  cv::Mat applyCustomKernel(const cv::Mat &input,
                            const CustomKernelParams &params);

  // Parameter validation functions
  void validateUnsharpParams(const UnsharpMaskParams &params);
  void validateHighBoostParams(const HighBoostParams &params);
  void validateKernel(const cv::Mat &kernel);

  template <class T> struct always_false : std::false_type {};
};

#endif // IMAGESHARPENER_HPP