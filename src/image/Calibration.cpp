#include "Calibration.hpp"

#include <cmath>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <tuple>

namespace {
std::shared_ptr<spdlog::logger> calibrationLogger =
    spdlog::basic_logger_mt("CalibrationLogger", "logs/calibration.log");
} // namespace

cv::Mat instrument_response_correction(cv::InputArray &image,
                                       cv::InputArray &response_function) {
  calibrationLogger->debug("Applying instrument response correction.");
  if (image.getMat().size() != response_function.getMat().size()) {
    calibrationLogger->error(
        "Image and response function shapes do not match.");
    throw std::invalid_argument(
        "Image and response function must have the same size.");
  }
  cv::Mat corrected;
  cv::multiply(image, response_function, corrected);
  calibrationLogger->info("Instrument response correction applied.");
  return corrected;
}

cv::Mat background_noise_correction(cv::InputArray &image) {
  calibrationLogger->debug("Applying background noise correction.");
  double medianValue = cv::mean(image)[0]; // 单通道图像
  cv::Mat imgMat = image.getMat();
  cv::Mat corrected = imgMat - medianValue;
  calibrationLogger->info("Background noise correction applied.");
  return corrected;
}

cv::Mat apply_flat_field_correction(cv::InputArray &image,
                                    cv::InputArray &flat_field) {
  calibrationLogger->debug("Applying flat-field correction.");
  if (image.getMat().size() != flat_field.getMat().size()) {
    calibrationLogger->error("Image and flat-field image shapes do not match.");
    throw std::invalid_argument(
        "Image and flat-field image must have the same size.");
  }
  cv::Mat corrected;
  cv::divide(image, flat_field, corrected);
  calibrationLogger->info("Flat-field correction applied.");
  return corrected;
}

cv::Mat apply_dark_frame_subtraction(cv::InputArray &image,
                                     cv::InputArray &dark_frame) {
  calibrationLogger->debug("Applying dark frame subtraction.");
  if (image.getMat().size() != dark_frame.getMat().size()) {
    calibrationLogger->error("Image and dark frame image shapes do not match.");
    throw std::invalid_argument(
        "Image and dark frame image must have the same size.");
  }
  cv::Mat corrected = image.getMat() - dark_frame.getMat();
  calibrationLogger->info("Dark frame subtraction applied.");
  return corrected;
}

double compute_flx2dn(const CalibrationParams &params) {
  calibrationLogger->debug("Starting FLX2DN computation.");
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
    calibrationLogger->info("Computed FLX2DN: {}", FLX2DN);
    return FLX2DN;
  } catch (const std::exception &e) {
    calibrationLogger->error("Error computing FLX2DN: {}", e.what());
    throw std::runtime_error("Failed to compute FLX2DN.");
  }
}

std::tuple<cv::Mat, double, double, double>
flux_calibration_ex(const cv::Mat &image, const CalibrationParams &params,
                    const cv::Mat *response_function, const cv::Mat *flat_field,
                    const cv::Mat *dark_frame, bool enable_optimization) {
  calibrationLogger->debug("Starting extended flux calibration process.");
  try {
    cv::Mat img;
    // 如果启用优化，则利用 UMat（若 OpenCL/OpenVX 后端支持则可能加速）
    if (enable_optimization) {
      cv::UMat uimg;
      image.copyTo(uimg);
      img = uimg.getMat(cv::ACCESS_READ);
      calibrationLogger->debug("Performance optimization enabled: using UMat.");
    } else {
      img = image.clone();
    }

    // 仪器响应校正
    if (response_function != nullptr) {
      calibrationLogger->debug("Applying instrument response correction.");
      img = instrument_response_correction(img, *response_function);
    }
    // 平场校正
    if (flat_field != nullptr) {
      calibrationLogger->debug("Applying flat-field correction.");
      img = apply_flat_field_correction(img, *flat_field);
    }
    // 暗场扣除
    if (dark_frame != nullptr) {
      calibrationLogger->debug("Applying dark frame subtraction.");
      img = apply_dark_frame_subtraction(img, *dark_frame);
    }

    // 计算 FLX2DN
    double FLX2DN = compute_flx2dn(params);
    // 流量校准：像素值除以 FLX2DN
    cv::Mat calibrated;
    cv::divide(img, FLX2DN, calibrated);
    calibrationLogger->debug("Applied FLX2DN conversion.");

    // 背景噪声校正
    calibrated = background_noise_correction(calibrated);
    calibrationLogger->debug("Applied background noise correction.");

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
    calibrationLogger->info("Rescaled calibrated image to [0, 1] range.");

    return {rescaled, FLXMIN, FLXRANGE, FLX2DN};
  } catch (const std::exception &e) {
    calibrationLogger->error("Extended flux calibration failed: {}", e.what());
    throw std::runtime_error("Extended flux calibration process failed.");
  }
}
