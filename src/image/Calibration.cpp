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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>


/**
 * Enhanced Camera Calibration Settings
 */
struct CalibrationSettings {
  cv::Size patternSize;     // Checkerboard pattern size
  float squareSize;         // Physical square size in mm
  int minImages;            // Minimum images required
  double maxRMS;            // Maximum acceptable RMS error
  bool useFixedAspectRatio; // Fix aspect ratio during calibration
  bool assumeZeroTangentialDistortion;
  bool fixPrincipalPoint;
  std::string outputDir; // Directory for output files

  CalibrationSettings() {
    patternSize = cv::Size(9, 6);
    squareSize = 25.0f;
    minImages = 10;
    maxRMS = 1.0;
    useFixedAspectRatio = false;
    assumeZeroTangentialDistortion = false;
    fixPrincipalPoint = false;
    outputDir = "calibration_output/";
  }
};

/**
 * Enhanced Camera Calibration Results
 */
struct CalibrationResults {
  cv::Mat cameraMatrix;
  cv::Mat distCoeffs;
  double totalRMS;
  std::vector<double> perViewErrors;
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  double avgReprError;
  double maxReprError;
  double fovX, fovY;
  cv::Point2d principalPoint;
  cv::Point2d focalLength;
  double aspectRatio;
};

class EnhancedCameraCalibrator {
private:
  CalibrationSettings settings;
  CalibrationResults results;
  cv::Size imageSize;
  std::vector<std::vector<cv::Point3f>> objectPoints;
  std::vector<std::vector<cv::Point2f>> imagePoints;
  std::mutex calibMutex;
  bool isCalibrated;

  // New members for enhanced features
  std::vector<cv::Mat> originalImages;
  std::vector<bool> validImages;
  std::vector<std::string> imageFilenames;

public:
  EnhancedCameraCalibrator(const CalibrationSettings &settings)
      : settings(settings), isCalibrated(false) {
    std::filesystem::create_directories(settings.outputDir);
  }

  /**
   * Enhanced pattern detection with multiple pattern types
   */
  bool detectPattern(const cv::Mat &image, std::vector<cv::Point2f> &corners,
                     bool drawCorners = false) {
    bool found = false;
    cv::Mat gray;
    cv::cvtColor(image.clone(), gray, cv::COLOR_BGR2GRAY);

    // Try different detection methods
    try {
      // Method 1: Standard checkerboard
      found = cv::findChessboardCorners(gray, settings.patternSize, corners,
                                        cv::CALIB_CB_ADAPTIVE_THRESH +
                                            cv::CALIB_CB_NORMALIZE_IMAGE +
                                            cv::CALIB_CB_FAST_CHECK);

      if (!found) {
        // Method 2: Asymmetric circles grid
        found = cv::findCirclesGrid(gray, settings.patternSize, corners,
                                    cv::CALIB_CB_ASYMMETRIC_GRID);
      }

      if (found) {
        cv::cornerSubPix(
            gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                             30, 0.1));

        if (drawCorners) {
          cv::Mat display = image.clone();
          cv::drawChessboardCorners(display, settings.patternSize, corners,
                                    found);
          cv::imshow("Pattern Detection", display);
          cv::waitKey(100);
        }
      }
    } catch (const cv::Exception &e) {
      std::cerr << "OpenCV error: " << e.what() << std::endl;
      return false;
    }

    return found;
  }

  /**
   * Parallel image processing for calibration
   */
  bool processImagesParallel(const std::vector<std::string> &imageFiles) {
    originalImages.clear();
    imagePoints.clear();
    objectPoints.clear();
    validImages.clear();
    imageFilenames = imageFiles;

    std::vector<std::future<bool>> futures;

    for (const auto &file : imageFiles) {
      futures.push_back(std::async(std::launch::async, [this, file]() {
        cv::Mat image = cv::imread(file);
        if (image.empty())
          return false;

        std::vector<cv::Point2f> corners;
        bool found = detectPattern(image, corners, true);

        if (found) {
          std::lock_guard<std::mutex> lock(calibMutex);
          originalImages.push_back(image);
          imagePoints.push_back(corners);
          objectPoints.push_back(createObjectPoints());
          validImages.push_back(true);
          return true;
        }
        return false;
      }));
    }

    int validCount = 0;
    for (auto &future : futures) {
      if (future.get())
        validCount++;
    }

    return validCount >= settings.minImages;
  }

  /**
   * Enhanced calibration with multiple options
   */
  bool calibrate() {
    if (imagePoints.empty() || imagePoints.size() < settings.minImages) {
      std::cerr << "Insufficient valid images for calibration" << std::endl;
      return false;
    }

    try {
      int flags = cv::CALIB_RATIONAL_MODEL;
      if (settings.useFixedAspectRatio)
        flags |= cv::CALIB_FIX_ASPECT_RATIO;
      if (settings.assumeZeroTangentialDistortion)
        flags |= cv::CALIB_ZERO_TANGENT_DIST;
      if (settings.fixPrincipalPoint)
        flags |= cv::CALIB_FIX_PRINCIPAL_POINT;

      results.totalRMS = cv::calibrateCamera(
          objectPoints, imagePoints, imageSize, results.cameraMatrix,
          results.distCoeffs, results.rvecs, results.tvecs, flags);

      isCalibrated = true;
      calculateCalibrationResults();
      return results.totalRMS < settings.maxRMS;
    } catch (const cv::Exception &e) {
      std::cerr << "Calibration failed: " << e.what() << std::endl;
      return false;
    }
  }

  /**
   * Calculate detailed calibration results
   */
  void calculateCalibrationResults() {
    if (!isCalibrated)
      return;

    // Calculate reprojection errors
    results.perViewErrors.clear();
    results.avgReprError = 0;
    results.maxReprError = 0;

    for (size_t i = 0; i < objectPoints.size(); i++) {
      std::vector<cv::Point2f> projectedPoints;
      cv::projectPoints(objectPoints[i], results.rvecs[i], results.tvecs[i],
                        results.cameraMatrix, results.distCoeffs,
                        projectedPoints);

      double err = cv::norm(imagePoints[i], projectedPoints, cv::NORM_L2);
      err /= projectedPoints.size();
      results.perViewErrors.push_back(err);

      results.avgReprError += err;
      results.maxReprError = std::max(results.maxReprError, err);
    }
    results.avgReprError /= objectPoints.size();

    // Calculate FOV
    cv::calibrationMatrixValues(
        results.cameraMatrix, imageSize, 0.0, 0.0, // Assume sensor size unknown
        results.fovX, results.fovY, results.focalLength.x,
        results.principalPoint, results.aspectRatio);
  }

  /**
   * Generate calibration report
   */
  void generateReport(const std::string &filename) {
    std::ofstream report(settings.outputDir + filename);
    if (!report.is_open())
      return;

    report << "Camera Calibration Report\n";
    report << "========================\n\n";
    report << "Calibration Date: " << getCurrentDateTime() << "\n\n";

    report << "Settings:\n";
    report << "- Pattern Size: " << settings.patternSize.width << "x"
           << settings.patternSize.height << "\n";
    report << "- Square Size: " << settings.squareSize << "mm\n";
    report << "- Number of images: " << originalImages.size() << "\n\n";

    report << "Results:\n";
    report << "- RMS Error: " << results.totalRMS << "\n";
    report << "- Average Reprojection Error: " << results.avgReprError << "\n";
    report << "- Maximum Reprojection Error: " << results.maxReprError << "\n";
    report << "- FOV: " << results.fovX << "x" << results.fovY << " degrees\n";
    report << "- Principal Point: (" << results.principalPoint.x << ", "
           << results.principalPoint.y << ")\n";
    report << "- Focal Length: (" << results.focalLength.x << ", "
           << results.focalLength.y << ")\n";
    report << "- Aspect Ratio: " << results.aspectRatio << "\n";

    report.close();
  }

  /**
   * Save calibration visualization
   */
  void saveCalibrationVisualization() {
    if (!isCalibrated)
      return;

    for (size_t i = 0; i < originalImages.size(); i++) {
      cv::Mat undistorted;
      undistortImage(originalImages[i], undistorted);

      // Create side-by-side comparison
      cv::Mat comparison;
      cv::hconcat(originalImages[i], undistorted, comparison);

      // Add text with error information
      std::string text =
          "RMS Error: " + std::to_string(results.perViewErrors[i]);
      cv::putText(comparison, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                  1.0, cv::Scalar(0, 255, 0), 2);

      // Save comparison image
      std::string filename =
          settings.outputDir + "comparison_" + std::to_string(i) + ".jpg";
      cv::imwrite(filename, comparison);
    }
  }

  /**
   * Real-time calibration preview
   */
  void startLivePreview(cv::VideoCapture &cap) {
    if (!cap.isOpened())
      return;

    cv::Mat frame;
    while (true) {
      cap >> frame;
      if (frame.empty())
        break;

      std::vector<cv::Point2f> corners;
      bool found = detectPattern(frame, corners, true);

      if (isCalibrated) {
        cv::Mat undistorted;
        undistortImage(frame, undistorted);
        cv::imshow("Live Preview - Undistorted", undistorted);
      }

      char key = cv::waitKey(1);
      if (key == 27)
        break; // ESC to exit
    }
  }

  // Additional utility functions
  void undistortImage(const cv::Mat &input, cv::Mat &output) {
    if (!isCalibrated) {
      output = input.clone();
      return;
    }
    cv::undistort(input, output, results.cameraMatrix, results.distCoeffs);
  }

  std::vector<cv::Point3f> createObjectPoints() {
    std::vector<cv::Point3f> points;
    for (int i = 0; i < settings.patternSize.height; i++) {
      for (int j = 0; j < settings.patternSize.width; j++) {
        points.push_back(
            cv::Point3f(j * settings.squareSize, i * settings.squareSize, 0));
      }
    }
    return points;
  }

  std::string getCurrentDateTime() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::string datetime = std::ctime(&time);
    datetime.pop_back(); // Remove newline
    return datetime;
  }

  // Getters
  const CalibrationResults &getResults() const { return results; }
  bool isCalibrationValid() const { return isCalibrated; }
};

// Example usage
int main() {
  CalibrationSettings settings;
  settings.patternSize = cv::Size(9, 6);
  settings.squareSize = 25.0f;
  settings.outputDir = "calibration_output/";

  EnhancedCameraCalibrator calibrator(settings);

  // Process images from directory
  std::vector<std::string> imageFiles;
  for (const auto &entry :
       std::filesystem::directory_iterator("calibration_images/")) {
    imageFiles.push_back(entry.path().string());
  }

  if (calibrator.processImagesParallel(imageFiles)) {
    if (calibrator.calibrate()) {
      calibrator.generateReport("calibration_report.txt");
      calibrator.saveCalibrationVisualization();

      // Optional: Live preview with webcam
      cv::VideoCapture cap(0);
      if (cap.isOpened()) {
        calibrator.startLivePreview(cap);
      }
    }
  }

  return 0;
}
