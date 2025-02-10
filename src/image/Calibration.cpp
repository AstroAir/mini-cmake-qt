#include "Calibration.hpp"
#include <cmath>
#include <opencv2/core/ocl.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
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

cv::Mat
instrument_response_correction_optimized(cv::InputArray &image,
                                         cv::InputArray &response_function,
                                         const OptimizationParams &params) {
  if (params.use_gpu && cv::ocl::haveOpenCL()) {
    cv::UMat uImage = image.getUMat();
    cv::UMat uResponse = response_function.getUMat();
    cv::UMat uResult;
    cv::multiply(uImage, uResponse, uResult);
    return uResult.getMat(cv::ACCESS_READ);
  }

  cv::Mat result;
  if (params.use_parallel) {
    cv::Mat img = image.getMat();
    cv::Mat resp = response_function.getMat();
    result = cv::Mat::zeros(img.size(), img.type());

    tbb::parallel_for(tbb::blocked_range<int>(0, img.rows),
                      [&](const tbb::blocked_range<int> &range) {
                        for (int i = range.begin(); i < range.end(); ++i) {
                          auto *img_ptr = img.ptr<float>(i);
                          auto *resp_ptr = resp.ptr<float>(i);
                          auto *result_ptr = result.ptr<float>(i);
                          for (int j = 0; j < img.cols; ++j) {
                            result_ptr[j] = img_ptr[j] * resp_ptr[j];
                          }
                        }
                      });
    return result;
  }

  return instrument_response_correction(image, response_function);
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
  calibrationLogger->debug("Starting optimized flux calibration process.");

  OptimizationParams optParams;
  optParams.use_gpu = enable_optimization;
  optParams.use_parallel = enable_optimization;

  try {
    cv::Mat img;
    if (optParams.use_gpu && cv::ocl::haveOpenCL()) {
      cv::UMat uimg = image.getUMat(cv::ACCESS_READ);
      img = uimg.getMat(cv::ACCESS_READ);
    } else {
      img = image.clone();
    }

    if (response_function != nullptr) {
      img = instrument_response_correction_optimized(img, *response_function,
                                                     optParams);
    }

#pragma omp parallel sections if (optParams.use_parallel)
    {
#pragma omp section
      if (flat_field != nullptr) {
        img = apply_flat_field_correction(img, *flat_field);
      }

#pragma omp section
      if (dark_frame != nullptr) {
        img = apply_dark_frame_subtraction(img, *dark_frame);
      }
    }

    double FLX2DN = compute_flx2dn(params);

    cv::Mat calibrated;
    if (optParams.use_gpu && cv::ocl::haveOpenCL()) {
      cv::UMat uimg = img.getUMat(cv::ACCESS_READ);
      cv::UMat ucalibrated;
      cv::divide(uimg, FLX2DN, ucalibrated);
      calibrated = ucalibrated.getMat(cv::ACCESS_READ);
    } else {
      cv::divide(img, FLX2DN, calibrated);
    }

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
    calibrationLogger->error("Optimized flux calibration failed: {}", e.what());
    throw std::runtime_error("Optimized flux calibration process failed.");
  }
}
