#include "Calibration.hpp"

#include <cmath>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <tuple>

cv::Mat instrument_response_correction(cv::InputArray &image,
                                       cv::InputArray &response_function) {
  spdlog::debug("Applying instrument response correction.");
  if (image.getMat().size() != response_function.getMat().size()) {
    spdlog::error("Image and response function shapes do not match.");
    throw std::invalid_argument(
        "Image and response function must have the same size.");
  }
  cv::Mat corrected;
  cv::multiply(image, response_function, corrected);
  spdlog::info("Instrument response correction applied.");
  return corrected;
}

cv::Mat background_noise_correction(cv::InputArray &image) {
  spdlog::debug("Applying background noise correction.");
  double medianValue = cv::mean(image)[0]; // 单通道图像
  cv::Mat imgMat = image.getMat();
  cv::Mat corrected = imgMat - medianValue;
  spdlog::info("Background noise correction applied.");
  return corrected;
}

cv::Mat apply_flat_field_correction(cv::InputArray &image,
                                    cv::InputArray &flat_field) {
  spdlog::debug("Applying flat-field correction.");
  if (image.getMat().size() != flat_field.getMat().size()) {
    spdlog::error("Image and flat-field image shapes do not match.");
    throw std::invalid_argument(
        "Image and flat-field image must have the same size.");
  }
  cv::Mat corrected;
  cv::divide(image, flat_field, corrected);
  spdlog::info("Flat-field correction applied.");
  return corrected;
}

cv::Mat apply_dark_frame_subtraction(cv::InputArray &image,
                                     cv::InputArray &dark_frame) {
  spdlog::debug("Applying dark frame subtraction.");
  if (image.getMat().size() != dark_frame.getMat().size()) {
    spdlog::error("Image and dark frame image shapes do not match.");
    throw std::invalid_argument(
        "Image and dark frame image must have the same size.");
  }
  cv::Mat corrected = image.getMat() - dark_frame.getMat();
  spdlog::info("Dark frame subtraction applied.");
  return corrected;
}

double compute_flx2dn(const CalibrationParams &params) {
  spdlog::debug("Starting FLX2DN computation.");
  try {
    const double c = 3.0e8;                         // 光速，单位 m/s
    const double h = 6.626e-34;                     // 普朗克常数，单位 J·s
    double wavelength_m = params.wavelength * 1e-9; // 纳米转米

    double aperture_area = M_PI * ((params.aperture * params.aperture -
                                    params.obstruction * params.obstruction) /
                                   4.0);
    double FLX2DN = params.exposure_time * aperture_area * params.filter_width *
                    params.transmissivity * params.gain *
                    params.quantum_efficiency * (1 - params.extinction) *
                    (wavelength_m / (c * h));
    spdlog::info("Computed FLX2DN: {}", FLX2DN);
    return FLX2DN;
  } catch (const std::exception &e) {
    spdlog::error("Error computing FLX2DN: {}", e.what());
    throw std::runtime_error("Failed to compute FLX2DN.");
  }
}

std::tuple<cv::Mat, double, double, double>
flux_calibration_ex(const cv::Mat &image, const CalibrationParams &params,
                    const cv::Mat *response_function, const cv::Mat *flat_field,
                    const cv::Mat *dark_frame, bool enable_optimization) {
  spdlog::debug("Starting extended flux calibration process.");
  try {
    cv::Mat img;
    // 如果启用优化，则利用 UMat（若 OpenCL/OpenVX 后端支持则可能加速）
    if (enable_optimization) {
      cv::UMat uimg;
      image.copyTo(uimg);
      img = uimg.getMat(cv::ACCESS_READ);
      spdlog::debug("Performance optimization enabled: using UMat.");
    } else {
      img = image.clone();
    }

    // 仪器响应校正
    if (response_function != nullptr) {
      spdlog::debug("Applying instrument response correction.");
      img = instrument_response_correction(img, *response_function);
    }
    // 平场校正
    if (flat_field != nullptr) {
      spdlog::debug("Applying flat-field correction.");
      img = apply_flat_field_correction(img, *flat_field);
    }
    // 暗场扣除
    if (dark_frame != nullptr) {
      spdlog::debug("Applying dark frame subtraction.");
      img = apply_dark_frame_subtraction(img, *dark_frame);
    }

    // 计算 FLX2DN
    double FLX2DN = compute_flx2dn(params);
    // 流量校准：像素值除以 FLX2DN
    cv::Mat calibrated;
    cv::divide(img, FLX2DN, calibrated);
    spdlog::debug("Applied FLX2DN conversion.");

    // 背景噪声校正
    calibrated = background_noise_correction(calibrated);
    spdlog::debug("Applied background noise correction.");

    // 归一化校准图像到 [0,1] 范围
    double minVal, maxVal;
    cv::minMaxLoc(calibrated, &minVal, &maxVal);
    double FLXMIN = minVal;
    double FLXRANGE = maxVal - minVal;
    cv::Mat rescaled;
    if (FLXRANGE > 0)
      rescaled = (calibrated - FLXMIN) / FLXRANGE;
    else
      rescaled = calibrated.clone();
    spdlog::info("Rescaled calibrated image to [0, 1] range.");

    return {rescaled, FLXMIN, FLXRANGE, FLX2DN};
  } catch (const std::exception &e) {
    spdlog::error("Extended flux calibration failed: {}", e.what());
    throw std::runtime_error("Extended flux calibration process failed.");
  }
}

// 示例使用
int main() {
  try {
    // 加载测试图像（假设为灰度图），请根据需要调整路径及读取方式
    cv::Mat image = cv::imread("input_image.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
      spdlog::error("Failed to read input image.");
      return -1;
    }

    // 设置校准参数
    CalibrationParams params;
    params.wavelength = 550;         // 550 nm
    params.aperture = 100;           // 100 mm
    params.obstruction = 20;         // 20 mm
    params.filter_width = 10;        // 10 nm
    params.transmissivity = 0.85;    // 85%
    params.gain = 1.0;               // 1.0
    params.quantum_efficiency = 0.9; // 90%
    params.extinction = 0.05;        // 5%
    params.exposure_time = 1.0;      // 1 s

    // 可选校准项：仪器响应、平场校正、暗场扣除
    cv::Mat response_function; // 可加载响应图，如：cv::imread("response.png",
                               // cv::IMREAD_GRAYSCALE);
    cv::Mat flat_field;        // 平场图
    cv::Mat dark_frame;        // 暗场图

    // 选择是否开启性能优化（默认为 false）
    bool enable_optimization = true;

    auto [calibrated_image, FLXMIN, FLXRANGE, FLX2DN] = flux_calibration_ex(
        image, params, response_function.empty() ? nullptr : &response_function,
        flat_field.empty() ? nullptr : &flat_field,
        dark_frame.empty() ? nullptr : &dark_frame, enable_optimization);
    spdlog::info("Extended flux calibration completed successfully.");

    // 显示结果图像
    cv::imshow("Calibrated Image", calibrated_image);
    cv::waitKey(0);
  } catch (const std::exception &e) {
    spdlog::error("Error in calibration process: {}", e.what());
    return -1;
  }
  return 0;
}