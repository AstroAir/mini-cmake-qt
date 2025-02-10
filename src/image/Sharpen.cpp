#include "Sharpen.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

ImageSharpener::ImageSharpener(Method method)
    : m_method(method), m_params(UnsharpMaskParams{}) {
  // Set default parameters according to the selected method.
  if (method == Method::HighBoost) {
    m_params = HighBoostParams{};
  } else if (method == Method::Laplace) {
    m_params = LaplaceParams{};
  } else if (method == Method::CustomKernel) {
    m_params = CustomKernelParams{};
  } else { // default to UnsharpMask
    m_params = UnsharpMaskParams{};
  }
}

void ImageSharpener::validateInput(const cv::Mat &input) {
  if (input.empty()) {
    throw std::invalid_argument("Input image is empty");
  }
  if (input.channels() > 4) {
    throw std::invalid_argument("Unsupported number of channels");
  }
}

QImage ImageSharpener::matToQImage(const cv::Mat &mat) {
  switch (mat.type()) {
  case CV_8UC3: {
    QImage img(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
               QImage::Format_BGR888);
    return img.copy();
  }
  case CV_8UC1: {
    QImage img(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
               QImage::Format_Grayscale8);
    return img.copy();
  }
  default:
    throw std::invalid_argument("Unsupported image format for Qt conversion");
  }
}

cv::Mat ImageSharpener::sharpen(const cv::Mat &input) {
  validateInput(input);
  try {
    return std::visit(
        [&](auto &&params) -> cv::Mat {
          using T = std::decay_t<decltype(params)>;
          if constexpr (std::is_same_v<T, LaplaceParams>) {
            return applyLaplace(input, params);
          } else if constexpr (std::is_same_v<T, UnsharpMaskParams>) {
            return applyUnsharpMask(input, params);
          } else if constexpr (std::is_same_v<T, HighBoostParams>) {
            return applyHighBoost(input, params);
          } else if constexpr (std::is_same_v<T, CustomKernelParams>) {
            return applyCustomKernel(input, params);
          } else {
            static_assert(always_false<T>::value, "Non-exhaustive visitor!");
          }
        },
        m_params);
  } catch (const cv::Exception &e) {
    throw std::runtime_error("OpenCV error: " + std::string(e.what()));
  }
}

cv::Mat ImageSharpener::applyLaplace(const cv::Mat &input,
                                     const LaplaceParams &params) {
  cv::Mat laplace;
  Laplacian(input, laplace, params.ddepth, 3, params.scale, params.delta,
            BORDER_DEFAULT);
  cv::Mat result;
  laplace.convertTo(result, input.type());
  addWeighted(input, 1.0, result, 1.0, 0.0, result);
  return result;
}

cv::Mat ImageSharpener::applyUnsharpMask(const cv::Mat &input,
                                         const UnsharpMaskParams &params) {
  validateUnsharpParams(params);
  cv::Mat blurred;
  GaussianBlur(input, blurred, Size(params.radius, params.radius),
               params.sigma);
  cv::Mat mask;
  subtract(input, blurred, mask);
  cv::Mat result;
  addWeighted(input, 1.0 + params.amount, mask, params.amount, 0, result);
  return result;
}

cv::Mat ImageSharpener::applyHighBoost(const cv::Mat &input,
                                       const HighBoostParams &params) {
  validateHighBoostParams(params);
  cv::Mat blurred;
  GaussianBlur(input, blurred, Size(params.radius, params.radius),
               params.sigma);
  cv::Mat mask;
  subtract(input, blurred, mask);
  cv::Mat result;
  // High-Boost filtering: result = input + (boostFactor - 1) * (input -
  // blurred)
  addWeighted(input, params.boostFactor, blurred, 1.0 - params.boostFactor, 0.0,
              result);
  return result;
}

cv::Mat ImageSharpener::applyCustomKernel(const cv::Mat &input,
                                          const CustomKernelParams &params) {
  validateKernel(params.kernel);
  cv::Mat result;
  filter2D(input, result, input.depth(), params.kernel, Point(-1, -1),
           params.delta, BORDER_DEFAULT);
  return result;
}

void ImageSharpener::validateUnsharpParams(const UnsharpMaskParams &params) {
  if (params.radius % 2 == 0) {
    throw std::invalid_argument("Radius must be an odd number for UnsharpMask");
  }
  if (params.sigma <= 0) {
    throw std::invalid_argument("Sigma must be positive for UnsharpMask");
  }
}

void ImageSharpener::validateHighBoostParams(const HighBoostParams &params) {
  if (params.radius % 2 == 0) {
    throw std::invalid_argument("Radius must be an odd number for HighBoost");
  }
  if (params.sigma <= 0) {
    throw std::invalid_argument("Sigma must be positive for HighBoost");
  }
  if (params.boostFactor < 1.0) {
    throw std::invalid_argument(
        "Boost factor must be at least 1.0 for HighBoost");
  }
}

void ImageSharpener::validateKernel(const cv::Mat &kernel) {
  if (kernel.empty()) {
    throw std::invalid_argument("Kernel is empty");
  }
  if (kernel.cols % 2 == 0 || kernel.rows % 2 == 0) {
    throw std::invalid_argument("Kernel dimensions must be odd");
  }
}