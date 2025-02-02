#include <functional>
#include <opencv2/opencv.hpp>
#include <random>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <variant>


enum class NoiseType {
  GAUSSIAN,
  SALT_AND_PEPPER,
  POISSON,
  SPECKLE,
  PERIODIC,
  LAPLACIAN
};

// 扩展参数结构体
struct GaussianParams {
  double mean = 0.0;
  double stddev = 25.0;
};

struct SaltAndPepperParams {
  double density = 0.05;
  double salt_prob = 0.5;
};

struct SpeckleParams {
  double intensity = 0.1;
};

struct PeriodicParams {
  double amplitude = 10.0;
  double frequency = 0.1;
};

struct LaplacianParams {
  double scale = 1.0;
};

using NoiseParams =
    std::variant<GaussianParams, SaltAndPepperParams, SpeckleParams,
                 PeriodicParams, LaplacianParams>;

// 进度回调类型
using ProgressCallback = std::function<void(float)>;

namespace detail {
void add_gaussian_noise(const cv::Mat &input, cv::Mat &output,
                        const GaussianParams &params) {
  cv::Mat noise(input.size(), input.type());
  cv::randn(noise, params.mean, params.stddev);
  cv::add(input, noise, output);
  output.convertTo(output, input.type());
}

void add_salt_and_pepper(const cv::Mat &input, cv::Mat &output,
                         const SaltAndPepperParams &params) {
  input.copyTo(output);

  const int num_pixels = input.rows * input.cols * input.channels();
  const int num_salt =
      static_cast<int>(num_pixels * params.density * params.salt_prob);
  const int num_pepper =
      static_cast<int>(num_pixels * params.density * (1 - params.salt_prob));

  static thread_local std::mt19937 gen{std::random_device{}()};
  std::uniform_int_distribution<> dis(0, num_pixels - 1);

  // Salt noise
  for (int i = 0; i < num_salt; ++i) {
    const int idx = dis(gen);
    output.data[idx] = 255;
  }

  // Pepper noise
  for (int i = 0; i < num_pepper; ++i) {
    const int idx = dis(gen);
    output.data[idx] = 0;
  }
}

void add_poisson_noise(const cv::Mat &input, cv::Mat &output) {
  cv::Mat noise(input.size(), input.type());
  cv::randn(noise, 0, 1);
  noise = noise * 0.5 + 0.5; // 转换到[0,1)范围
  cv::exp(-input, noise);
  cv::threshold(noise, noise, cv::randu<double>(), 255, cv::THRESH_BINARY);
  cv::multiply(input, noise, output);
  output.convertTo(output, input.type());
}

void add_speckle_noise(const cv::Mat &input, cv::Mat &output,
                       const SpeckleParams &params) {
  cv::Mat noise(input.size(), CV_32F);
  cv::randn(noise, 0, params.intensity);

  input.convertTo(output, CV_32F);
  output = output.mul(1.0f + noise);
  output.convertTo(output, input.type());
}

void add_periodic_noise(const cv::Mat &input, cv::Mat &output,
                        const PeriodicParams &params) {
  output = input.clone();
#pragma omp parallel for
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      double noise = params.amplitude *
                     std::sin(2 * CV_PI * params.frequency * i) *
                     std::cos(2 * CV_PI * params.frequency * j);
      for (int c = 0; c < input.channels(); c++) {
        int value = cv::saturate_cast<uchar>(
            input.at<uchar>(i, j * input.channels() + c) + noise);
        output.at<uchar>(i, j * input.channels() + c) = value;
      }
    }
  }
}

void add_laplacian_noise(const cv::Mat &input, cv::Mat &output,
                         const LaplacianParams &params) {
  cv::Mat noise(input.size(), input.type());
  static thread_local std::mt19937 gen{std::random_device{}()};
  std::exponential_distribution<> d(1.0 / params.scale);

#pragma omp parallel for
  for (int i = 0; i < noise.total(); i++) {
    noise.data[i] = cv::saturate_cast<uchar>(d(gen));
  }

  cv::add(input, noise, output);
}

} // namespace detail

void add_image_noise(const cv::Mat &input, cv::Mat &output, NoiseType type,
                     const NoiseParams &params,
                     ProgressCallback progress = nullptr) {
  // 输入验证
  if (input.empty()) {
    spdlog::error("Input image is empty");
    throw std::invalid_argument("Input image is empty");
  }

  if (input.depth() != CV_8U && input.depth() != CV_32F) {
    spdlog::error("Unsupported image depth: {}", input.depth());
    throw std::invalid_argument("Unsupported image depth");
  }

  // 处理不同噪声类型
  try {
    float progress_value = 0.0f;
    switch (type) {
    case NoiseType::GAUSSIAN: {
      const auto &p = std::get<GaussianParams>(params);
      if (p.stddev < 0) {
        spdlog::error("Invalid standard deviation: {}", p.stddev);
        throw std::invalid_argument("Negative standard deviation");
      }
      detail::add_gaussian_noise(input, output, p);
      break;
    }
    case NoiseType::SALT_AND_PEPPER: {
      const auto &p = std::get<SaltAndPepperParams>(params);
      if (p.density < 0 || p.density > 1) {
        spdlog::error("Invalid density: {}", p.density);
        throw std::invalid_argument("Density out of [0,1] range");
      }
      detail::add_salt_and_pepper(input, output, p);
      break;
    }
    case NoiseType::POISSON:
      detail::add_poisson_noise(input, output);
      break;
    case NoiseType::SPECKLE: {
      const auto &p = std::get<SpeckleParams>(params);
      detail::add_speckle_noise(input, output, p);
      break;
    }
    case NoiseType::PERIODIC: {
      const auto &p = std::get<PeriodicParams>(params);
      detail::add_periodic_noise(input, output, p);
      break;
    }
    case NoiseType::LAPLACIAN: {
      const auto &p = std::get<LaplacianParams>(params);
      detail::add_laplacian_noise(input, output, p);
      break;
    }
    default:
      spdlog::error("Unknown noise type: {}", static_cast<int>(type));
      throw std::invalid_argument("Unknown noise type");
    }

    if (progress) {
      progress(1.0f);
    }

  } catch (const cv::Exception &e) {
    spdlog::error("OpenCV exception: {}", e.what());
    throw;
  } catch (const std::bad_variant_access &) {
    spdlog::error("Parameter type mismatch for noise type");
    throw;
  }

  // 后处理
  if (input.data != output.data) {
    cv::normalize(output, output, 0, 255, cv::NORM_MINMAX, input.type());
  }
}

// 新增：噪声估计函数
double estimate_noise_level(const cv::Mat &image) {
  cv::Mat gray;
  if (image.channels() > 1) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = image;
  }

  cv::Mat laplacian;
  cv::Laplacian(gray, laplacian, CV_64F);

  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);

  return stddev[0];
}