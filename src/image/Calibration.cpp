#include "Calibration.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <execution>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <opencv2/core/ocl.hpp>
#include <shared_mutex>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

// SIMD support
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace {
// Thread-safe singleton logger pattern
class Logger {
public:
  static std::shared_ptr<spdlog::logger> getInstance() {
    static std::once_flag flag;
    static std::shared_ptr<spdlog::logger> instance;

    std::call_once(flag, []() {
      try {
        std::filesystem::create_directories("logs");
        instance = spdlog::basic_logger_mt("CalibrationLogger",
                                           "logs/calibration.log");
        instance->set_level(spdlog::level::debug);
        instance->flush_on(spdlog::level::warn);
      } catch (const spdlog::spdlog_ex &ex) {
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        instance = spdlog::null_logger_mt("null_calibration_logger");
      }
    });

    return instance;
  }
};

// Thread-local cache for optimization
class ImageCache {
private:
  std::unordered_map<size_t, cv::Mat> cache;
  std::shared_mutex mutex;
  size_t max_size;
  std::atomic<size_t> current_size{0};

  // Hash function for cv::Mat
  static size_t hashMat(const cv::Mat &mat) {
    size_t hash = 0;
    auto dataPtr = mat.data;
    auto dataSize = mat.total() * mat.elemSize();

    for (size_t i = 0; i < dataSize; i += sizeof(size_t)) {
      size_t value = 0;
      std::memcpy(&value, dataPtr + i, std::min(sizeof(size_t), dataSize - i));
      hash ^= value + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }

public:
  explicit ImageCache(size_t max_size_mb)
      : max_size(max_size_mb * 1024 * 1024) {}

  std::optional<cv::Mat> get(const cv::Mat &key) {
    size_t hash = hashMat(key);
    std::shared_lock lock(mutex);
    auto it = cache.find(hash);
    if (it != cache.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  void put(const cv::Mat &key, const cv::Mat &value) {
    size_t newSize = value.total() * value.elemSize();
    if (newSize > max_size)
      return;

    size_t hash = hashMat(key);
    std::unique_lock lock(mutex);

    // Make space if needed
    while (current_size + newSize > max_size && !cache.empty()) {
      auto it = cache.begin();
      current_size -= it->second.total() * it->second.elemSize();
      cache.erase(it);
    }

    cache[hash] = value.clone();
    current_size += newSize;
  }

  void clear() {
    std::unique_lock lock(mutex);
    cache.clear();
    current_size = 0;
  }
};

// Singleton image cache
std::unique_ptr<ImageCache> getGlobalCache(size_t size_mb) {
  static std::mutex mutex;
  static std::weak_ptr<ImageCache> weakCache;

  std::lock_guard lock(mutex);
  auto cache = weakCache.lock();

  if (!cache) {
    cache = std::make_shared<ImageCache>(size_mb);
    weakCache = cache;
  }

  return std::make_unique<ImageCache>(size_mb);
}

// SIMD optimization detection
bool hasSIMDSupport() {
#ifdef __AVX2__
  return true;
#else
  return false;
#endif
}

} // namespace

cv::Mat instrument_response_correction(cv::InputArray &image,
                                       cv::InputArray &response_function) {
  const auto &logger = Logger::getInstance();
  logger->debug("Applying instrument response correction.");

  try {
    cv::Mat img = image.getMat();
    cv::Mat resp = response_function.getMat();

    if (img.size() != resp.size()) {
      logger->error("Image and response function shapes do not match: {} vs {}",
                    img.size(), resp.size());
      throw InvalidParameterError(
          "Image and response function must have the same size.");
    }

    if (img.type() != resp.type()) {
      logger->warn(
          "Image and response function types do not match. Converting...");
      cv::Mat converted;
      resp.convertTo(converted, img.type());
      resp = converted;
    }

    cv::Mat corrected;
    cv::multiply(img, resp, corrected);
    logger->info("Instrument response correction applied successfully.");
    return corrected;
  } catch (const cv::Exception &e) {
    logger->error("OpenCV error during response correction: {}", e.what());
    throw ProcessingError(
        std::string("OpenCV error during response correction: ") + e.what());
  } catch (const std::exception &e) {
    logger->error("Error during instrument response correction: {}", e.what());
    throw;
  }
}

cv::Mat
instrument_response_correction_optimized(cv::InputArray &image,
                                         cv::InputArray &response_function,
                                         const OptimizationParams &params) {
  const auto &logger = Logger::getInstance();
  logger->debug("Applying optimized instrument response correction.");

  if (!params.isValid()) {
    logger->error("Invalid optimization parameters");
    throw InvalidParameterError("Invalid optimization parameters");
  }

  try {
    // Check cache first if enabled
    if (params.use_cache) {
      auto cache = getGlobalCache(params.cache_size);
      cv::Mat img = image.getMat();
      cv::Mat resp = response_function.getMat();

      auto cachedResult = cache->get(img);
      if (cachedResult) {
        logger->debug("Cache hit for image");
        return cachedResult.value();
      }
    }

    // GPU optimization path
    if (params.use_gpu && cv::ocl::haveOpenCL()) {
      logger->debug("Using GPU acceleration");
      cv::UMat uImage = image.getUMat();
      cv::UMat uResponse = response_function.getUMat();
      cv::UMat uResult;
      cv::multiply(uImage, uResponse, uResult);
      cv::Mat result = uResult.getMat(cv::ACCESS_READ);

      // Store in cache if enabled
      if (params.use_cache) {
        auto cache = getGlobalCache(params.cache_size);
        cache->put(image.getMat(), result);
      }

      return result;
    }

    // SIMD optimization path
    if (params.use_simd && hasSIMDSupport()) {
      logger->debug("Using SIMD acceleration");
      cv::Mat img = image.getMat();
      cv::Mat resp = response_function.getMat();
      cv::Mat result = cv::Mat::zeros(img.size(), img.type());

      // Ensure single-channel float type for SIMD processing
      if (img.channels() == 1 && img.depth() == CV_32F) {
#ifdef __AVX2__
        // Process 8 floats at a time with AVX2
        for (int i = 0; i < img.rows; i++) {
          float *imgPtr = img.ptr<float>(i);
          float *respPtr = resp.ptr<float>(i);
          float *resultPtr = result.ptr<float>(i);

          int j = 0;
          for (; j <= img.cols - 8; j += 8) {
            __m256 imgVec = _mm256_loadu_ps(imgPtr + j);
            __m256 respVec = _mm256_loadu_ps(respPtr + j);
            __m256 resultVec = _mm256_mul_ps(imgVec, respVec);
            _mm256_storeu_ps(resultPtr + j, resultVec);
          }

          // Process remaining elements
          for (; j < img.cols; j++) {
            resultPtr[j] = imgPtr[j] * respPtr[j];
          }
        }

        // Store in cache if enabled
        if (params.use_cache) {
          auto cache = getGlobalCache(params.cache_size);
          cache->put(img, result);
        }

        return result;
#endif
      }
    }

    // Parallel optimization path
    if (params.use_parallel) {
      logger->debug("Using parallel processing with {} threads",
                    params.num_threads);
      cv::Mat img = image.getMat();
      cv::Mat resp = response_function.getMat();
      cv::Mat result = cv::Mat::zeros(img.size(), img.type());

      // Set number of threads if specified
      int oldThreads = 0;
      if (params.num_threads > 0) {
        oldThreads = tbb::global_control::active_value(
            tbb::global_control::max_allowed_parallelism);
        tbb::global_control control(
            tbb::global_control::max_allowed_parallelism, params.num_threads);
      }

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

      // Store in cache if enabled
      if (params.use_cache) {
        auto cache = getGlobalCache(params.cache_size);
        cache->put(img, result);
      }

      return result;
    }

    // Default path
    return instrument_response_correction(image, response_function);

  } catch (const cv::Exception &e) {
    logger->error("OpenCV error during optimized response correction: {}",
                  e.what());
    throw ProcessingError(std::string("OpenCV error: ") + e.what());
  } catch (const std::exception &e) {
    logger->error("Error during optimized instrument response correction: {}",
                  e.what());
    throw;
  }
}

cv::Mat background_noise_correction(cv::InputArray &image) noexcept {
  const auto &logger = Logger::getInstance();
  logger->debug("Applying background noise correction.");

  try {
    double medianValue = cv::mean(image)[0];
    cv::Mat imgMat = image.getMat();
    cv::Mat corrected = imgMat - medianValue;
    logger->info("Background noise correction applied with median value: {}",
                 medianValue);
    return corrected;
  } catch (const std::exception &e) {
    logger->error("Error in background noise correction: {}", e.what());
    return image.getMat().clone(); // Return original image in case of failure
  }
}

cv::Mat background_noise_correction_optimized(
    cv::InputArray &image, const OptimizationParams &params) noexcept {
  const auto &logger = Logger::getInstance();
  logger->debug("Applying optimized background noise correction.");

  try {
    if (params.use_gpu && cv::ocl::haveOpenCL()) {
      cv::UMat uImage = image.getUMat();
      double medianValue = cv::mean(uImage)[0];
      cv::UMat uCorrected;
      cv::subtract(uImage, medianValue, uCorrected);
      return uCorrected.getMat(cv::ACCESS_READ);
    } else if (params.use_simd && hasSIMDSupport()) {
      cv::Mat imgMat = image.getMat();
      double medianValue = cv::mean(imgMat)[0];
      cv::Mat corrected = cv::Mat::zeros(imgMat.size(), imgMat.type());

      if (imgMat.depth() == CV_32F && imgMat.channels() == 1) {
#ifdef __AVX2__
        __m256 medianVec = _mm256_set1_ps(static_cast<float>(medianValue));

        for (int i = 0; i < imgMat.rows; i++) {
          float *src = imgMat.ptr<float>(i);
          float *dst = corrected.ptr<float>(i);

          int j = 0;
          for (; j <= imgMat.cols - 8; j += 8) {
            __m256 srcVec = _mm256_loadu_ps(src + j);
            __m256 result = _mm256_sub_ps(srcVec, medianVec);
            _mm256_storeu_ps(dst + j, result);
          }

          // Process remaining elements
          for (; j < imgMat.cols; j++) {
            dst[j] = src[j] - static_cast<float>(medianValue);
          }
        }
        return corrected;
#endif
      }
    }

    // Fall back to standard implementation
    return background_noise_correction(image);
  } catch (const std::exception &e) {
    logger->error("Error in optimized background noise correction: {}",
                  e.what());
    return image.getMat().clone(); // Return original image in case of failure
  }
}

cv::Mat apply_flat_field_correction(cv::InputArray &image,
                                    cv::InputArray &flat_field) {
  const auto &logger = Logger::getInstance();
  logger->debug("Applying flat-field correction.");

  try {
    cv::Mat img = image.getMat();
    cv::Mat flat = flat_field.getMat();

    if (img.size() != flat.size()) {
      logger->error("Image and flat-field image shapes do not match: {} vs {}",
                    img.size(), flat.size());
      throw InvalidParameterError(
          "Image and flat-field image must have the same size.");
    }

    // Check for division by zero
    double minVal;
    cv::minMaxLoc(flat, &minVal);
    if (std::abs(minVal) < 1e-10) {
      logger->warn(
          "Very small values detected in flat field. Applying threshold.");
      cv::threshold(flat, flat, 1e-10, 1e-10, cv::THRESH_TOZERO_INV);
      flat += 1e-10;
    }

    cv::Mat corrected;
    cv::divide(img, flat, corrected);
    logger->info("Flat-field correction applied successfully.");
    return corrected;
  } catch (const cv::Exception &e) {
    logger->error("OpenCV error during flat field correction: {}", e.what());
    throw ProcessingError(std::string("OpenCV error: ") + e.what());
  } catch (const std::exception &e) {
    logger->error("Error during flat field correction: {}", e.what());
    throw;
  }
}

cv::Mat apply_dark_frame_subtraction(cv::InputArray &image,
                                     cv::InputArray &dark_frame) {
  const auto &logger = Logger::getInstance();
  logger->debug("Applying dark frame subtraction.");

  try {
    cv::Mat img = image.getMat();
    cv::Mat dark = dark_frame.getMat();

    if (img.size() != dark.size()) {
      logger->error("Image and dark frame image shapes do not match: {} vs {}",
                    img.size(), dark.size());
      throw InvalidParameterError(
          "Image and dark frame image must have the same size.");
    }

    if (img.type() != dark.type()) {
      logger->warn("Image and dark frame types do not match. Converting...");
      cv::Mat converted;
      dark.convertTo(converted, img.type());
      dark = converted;
    }

    cv::Mat corrected = img - dark;
    logger->info("Dark frame subtraction applied successfully.");
    return corrected;
  } catch (const cv::Exception &e) {
    logger->error("OpenCV error during dark frame subtraction: {}", e.what());
    throw ProcessingError(std::string("OpenCV error: ") + e.what());
  } catch (const std::exception &e) {
    logger->error("Error during dark frame subtraction: {}", e.what());
    throw;
  }
}

double compute_flx2dn(const CalibrationParams &params) {
  const auto &logger = Logger::getInstance();
  logger->debug("Starting FLX2DN computation.");

  if (!params.isValid()) {
    logger->error("Invalid calibration parameters");
    throw InvalidParameterError(
        "Invalid calibration parameters for FLX2DN computation");
  }

  try {
    const double c = 3.0e8;     // Speed of light, unit: m/s
    const double h = 6.626e-34; // Planck constant, unit: J·s
    double wavelength_m = params.wavelength * 1e-9; // Convert nm to m

    // Check for division by zero or negative values
    if (wavelength_m <= 0) {
      logger->error("Wavelength must be positive");
      throw InvalidParameterError("Wavelength must be positive");
    }

    if (c <= 0 || h <= 0) {
      logger->error("Physical constants must be positive");
      throw InvalidParameterError("Physical constants must be positive");
    }

    double aperture_area = M_PI * ((params.aperture * params.aperture -
                                    params.obstruction * params.obstruction) /
                                   4.0);

    if (aperture_area <= 0) {
      logger->warn("Calculated aperture area is not positive. Check aperture "
                   "and obstruction values.");
      throw InvalidParameterError("Calculated aperture area must be positive");
    }

    double FLX2DN = params.exposure_time * aperture_area * params.filter_width *
                    params.transmissivity * params.gain *
                    params.quantum_efficiency * (1 - params.extinction) *
                    (wavelength_m / (c * h));

    logger->info("Computed FLX2DN: {}", FLX2DN);
    return FLX2DN;
  } catch (const std::exception &e) {
    logger->error("Error computing FLX2DN: {}", e.what());
    throw;
  }
}

std::optional<FluxCalibrationResult>
flux_calibration_ex(const cv::Mat &image, const CalibrationParams &params,
                    const cv::Mat *response_function, const cv::Mat *flat_field,
                    const cv::Mat *dark_frame, bool enable_optimization) {
  const auto &logger = Logger::getInstance();
  logger->debug("Starting flux calibration process.");

  if (image.empty()) {
    logger->error("Input image is empty");
    return std::nullopt;
  }

  if (!params.isValid()) {
    logger->error("Invalid calibration parameters");
    return std::nullopt;
  }

  OptimizationParams optParams;
  optParams.use_gpu = enable_optimization;
  optParams.use_parallel = enable_optimization;
  optParams.use_simd = enable_optimization && hasSIMDSupport();
  optParams.use_cache = enable_optimization;

  try {
    // Start performance measurement
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat img;
    if (optParams.use_gpu && cv::ocl::haveOpenCL()) {
      cv::UMat uimg = image.getUMat(cv::ACCESS_READ);
      img = uimg.getMat(cv::ACCESS_READ);
    } else {
      img = image.clone();
    }

    // Validate input images
    if (response_function != nullptr && response_function->empty()) {
      logger->warn("Empty response function provided");
      response_function = nullptr;
    }

    if (flat_field != nullptr && flat_field->empty()) {
      logger->warn("Empty flat field provided");
      flat_field = nullptr;
    }

    if (dark_frame != nullptr && dark_frame->empty()) {
      logger->warn("Empty dark frame provided");
      dark_frame = nullptr;
    }

    // Apply response function correction if available
    if (response_function != nullptr) {
      cv::InputArray imgArray = img;
      cv::InputArray respArray = *response_function;
      img = instrument_response_correction_optimized(imgArray, respArray,
                                                     optParams);
    }

// Use C++20 barrier for concurrent operations
#if __cpp_lib_barrier >= 201907L
    if (optParams.use_parallel && flat_field != nullptr &&
        dark_frame != nullptr) {
      std::barrier sync_point(2, [&]() noexcept {
        logger->debug("All parallel corrections completed.");
      });

      std::vector<std::jthread> threads;
      cv::Mat flat_result, dark_result;

      threads.emplace_back([&]() {
        try {
          cv::InputArray imgArray = img;
          cv::InputArray flatArray = *flat_field;
          flat_result = apply_flat_field_correction(imgArray, flatArray);
          sync_point.arrive_and_wait();
        } catch (...) {
          logger->error("Error in flat field thread");
        }
      });

      threads.emplace_back([&]() {
        try {
          cv::InputArray imgArray = img;
          cv::InputArray darkArray = *dark_frame;
          dark_result = apply_dark_frame_subtraction(imgArray, darkArray);
          sync_point.arrive_and_wait();
        } catch (...) {
          logger->error("Error in dark frame thread");
        }
      });

      // Wait for all threads to complete
      for (auto &t : threads) {
        if (t.joinable())
          t.join();
      }

      // Combine results
      img = flat_result.mul(dark_result / img);
    } else
#endif
    // Standard sequential processing
    {
      if (flat_field != nullptr) {
        cv::InputArray imgArray = img;
        cv::InputArray flatArray = *flat_field;
        img = apply_flat_field_correction(imgArray, flatArray);
      }

      if (dark_frame != nullptr) {
        cv::InputArray imgArray = img;
        cv::InputArray darkArray = *dark_frame;
        img = apply_dark_frame_subtraction(imgArray, darkArray);
      }
    }

    // Calculate flux-to-DN conversion factor
    double FLX2DN = compute_flx2dn(params);
    if (FLX2DN <= 0) {
      logger->error("Invalid FLX2DN value: {}", FLX2DN);
      return std::nullopt;
    }

    // Apply flux calibration
    cv::Mat calibrated;
    if (optParams.use_gpu && cv::ocl::haveOpenCL()) {
      cv::UMat uimg = img.getUMat(cv::ACCESS_READ);
      cv::UMat ucalibrated;
      cv::divide(uimg, FLX2DN, ucalibrated);
      calibrated = ucalibrated.getMat(cv::ACCESS_READ);
    } else {
      cv::divide(img, FLX2DN, calibrated);
    }

    // Apply background noise correction
    cv::InputArray calibratedArray = calibrated;
    calibrated =
        background_noise_correction_optimized(calibratedArray, optParams);
    logger->debug("Applied background noise correction.");

    // Normalize calibrated image to [0,1] range
    double minVal, maxVal;
    cv::minMaxLoc(calibrated, &minVal, &maxVal);
    double FLXMIN = minVal;
    double FLXRANGE = maxVal - minVal;
    cv::Mat rescaled;

    if (FLXRANGE > 0) {
      if (optParams.use_gpu && cv::ocl::haveOpenCL()) {
        cv::UMat ucalibrated = calibrated.getUMat(cv::ACCESS_READ);
        cv::UMat urescaled;

        // Using OpenCV's normalize function which is optimized for GPU
        cv::normalize(ucalibrated, urescaled, 0, 1, cv::NORM_MINMAX, CV_32F);
        rescaled = urescaled.getMat(cv::ACCESS_READ);
      } else if (optParams.use_simd && hasSIMDSupport()) {
        rescaled = cv::Mat::zeros(calibrated.size(), CV_32F);

#ifdef __AVX2__
        float fFLXMIN = static_cast<float>(FLXMIN);
        float fFLXRANGE = static_cast<float>(FLXRANGE);
        __m256 vFLXMIN = _mm256_set1_ps(fFLXMIN);
        __m256 vFLXRANGE = _mm256_set1_ps(fFLXRANGE);

        for (int i = 0; i < calibrated.rows; ++i) {
          float *src = calibrated.ptr<float>(i);
          float *dst = rescaled.ptr<float>(i);
          int j = 0;

          // Process 8 elements at a time
          for (; j <= calibrated.cols - 8; j += 8) {
            __m256 vals = _mm256_loadu_ps(src + j);
            __m256 normalized =
                _mm256_div_ps(_mm256_sub_ps(vals, vFLXMIN), vFLXRANGE);
            _mm256_storeu_ps(dst + j, normalized);
          }

          // Process remaining elements
          for (; j < calibrated.cols; ++j) {
            dst[j] = (src[j] - fFLXMIN) / fFLXRANGE;
          }
        }
#endif
      } else if (optParams.use_parallel) {
        rescaled = cv::Mat::zeros(calibrated.size(), CV_32F);

        tbb::parallel_for(tbb::blocked_range<int>(0, calibrated.rows),
                          [&](const tbb::blocked_range<int> &range) {
                            for (int i = range.begin(); i < range.end(); ++i) {
                              float *src = calibrated.ptr<float>(i);
                              float *dst = rescaled.ptr<float>(i);

                              for (int j = 0; j < calibrated.cols; ++j) {
                                dst[j] = static_cast<float>((src[j] - FLXMIN) /
                                                            FLXRANGE);
                              }
                            }
                          });
      } else {
        // Standard normalization
        rescaled = (calibrated - FLXMIN) / FLXRANGE;
      }
      logger->info("Rescaled calibrated image to [0, 1] range.");
    } else {
      logger->warn(
          "Zero range detected in calibrated image. Skipping rescaling.");
      rescaled = calibrated.clone();
    }

    // Measure performance
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    logger->info("Flux calibration completed in {} seconds", elapsed.count());

    // Return the result
    FluxCalibrationResult result;
    result.image = rescaled;
    result.min_value = FLXMIN;
    result.range_value = FLXRANGE;
    result.flx2dn_factor = FLX2DN;

    return result;

  } catch (const cv::Exception &e) {
    logger->error("OpenCV error during flux calibration: {}", e.what());
    return std::nullopt;
  } catch (const std::exception &e) {
    logger->error("Error during flux calibration: {}", e.what());
    return std::nullopt;
  } catch (...) {
    logger->error("Unknown error during flux calibration");
    return std::nullopt;
  }
}

// Implement batch processing using C++20 ranges and parallelism
std::vector<cv::Mat>
batch_process_images(const std::vector<cv::Mat> &images,
                     const std::function<cv::Mat(const cv::Mat &)> &processor,
                     const OptimizationParams &params) {

  const auto &logger = Logger::getInstance();
  logger->debug("Starting batch processing of {} images", images.size());

  std::vector<cv::Mat> results(images.size());

  try {
    if (images.empty()) {
      return results;
    }

    if (params.use_parallel) {
#if __cplusplus >= 202002L
      // Using C++20 ranges and views
      std::vector<size_t> indices(images.size());
      std::iota(indices.begin(), indices.end(), 0);

      if (params.use_gpu && cv::ocl::haveOpenCL()) {
        std::for_each(
            std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
              try {
                cv::UMat uimage = images[i].getUMat(cv::ACCESS_READ);
                cv::Mat result = processor(uimage.getMat(cv::ACCESS_READ));
                results[i] = result;
              } catch (const std::exception &e) {
                logger->error("Error processing image {}: {}", i, e.what());
                results[i] = images[i].clone(); // Return original on error
              }
            });
      } else {
        std::for_each(
            std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
              try {
                results[i] = processor(images[i]);
              } catch (const std::exception &e) {
                logger->error("Error processing image {}: {}", i, e.what());
                results[i] = images[i].clone(); // Return original on error
              }
            });
      }
#else
      // Fallback for older C++ standards
      tbb::parallel_for(tbb::blocked_range<size_t>(0, images.size()),
                        [&](const tbb::blocked_range<size_t> &range) {
                          for (size_t i = range.begin(); i < range.end(); ++i) {
                            try {
                              results[i] = processor(images[i]);
                            } catch (const std::exception &e) {
                              logger->error("Error processing image {}: {}", i,
                                            e.what());
                              results[i] = images[i].clone();
                            }
                          }
                        });
#endif
    } else {
      // Sequential processing
      for (size_t i = 0; i < images.size(); ++i) {
        try {
          results[i] = processor(images[i]);
        } catch (const std::exception &e) {
          logger->error("Error processing image {}: {}", i, e.what());
          results[i] = images[i].clone();
        }
      }
    }

    logger->info("Batch processing completed for {} images", images.size());
    return results;
  } catch (const std::exception &e) {
    logger->error("Error during batch processing: {}", e.what());
    return images; // Return original images on error
  }
}

// Modern C++20 implementation of CameraCalibrator class
class CameraCalibrator {
public:
  struct Settings {
    cv::Size patternSize{9, 6};          // Checkerboard pattern size
    float squareSize{25.0f};             // Physical square size in mm
    int minImages{10};                   // Minimum images required
    double maxRMS{1.0};                  // Maximum acceptable RMS error
    int flags{cv::CALIB_RATIONAL_MODEL}; // Calibration flags
    std::string outputDir{"calibration_output/"};

    // C++20 designated initializers support
    static Settings createDefault() {
      return Settings{.patternSize = {9, 6},
                      .squareSize = 25.0f,
                      .minImages = 10,
                      .maxRMS = 1.0,
                      .flags = cv::CALIB_RATIONAL_MODEL,
                      .outputDir = "calibration_output/"};
    }

    // Flag helpers
    Settings &withFixedAspectRatio(bool enable = true) {
      if (enable)
        flags |= cv::CALIB_FIX_ASPECT_RATIO;
      else
        flags &= ~cv::CALIB_FIX_ASPECT_RATIO;
      return *this;
    }

    Settings &withZeroTangentialDistortion(bool enable = true) {
      if (enable)
        flags |= cv::CALIB_ZERO_TANGENT_DIST;
      else
        flags &= ~cv::CALIB_ZERO_TANGENT_DIST;
      return *this;
    }

    Settings &withFixedPrincipalPoint(bool enable = true) {
      if (enable)
        flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
      else
        flags &= ~cv::CALIB_FIX_PRINCIPAL_POINT;
      return *this;
    }
  };

  struct Results {
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    double totalRMS{0.0};
    std::vector<double> perViewErrors;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    double avgReprError{0.0};
    double maxReprError{0.0};
    double fovX{0.0}, fovY{0.0};
    cv::Point2d principalPoint;
    cv::Point2d focalLength;
    double aspectRatio{0.0};

    bool isValid() const noexcept {
      return !cameraMatrix.empty() && !distCoeffs.empty() && totalRMS >= 0;
    }

    // Export/import functions
    bool saveToFile(const std::string &filename) const {
      try {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (!fs.isOpened())
          return false;

        fs << "camera_matrix" << cameraMatrix;
        fs << "dist_coeffs" << distCoeffs;
        fs << "rms" << totalRMS;
        fs << "avg_error" << avgReprError;
        fs << "max_error" << maxReprError;
        fs << "fov_x" << fovX;
        fs << "fov_y" << fovY;

        fs.release();
        return true;
      } catch (...) {
        return false;
      }
    }

    bool loadFromFile(const std::string &filename) {
      try {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened())
          return false;

        fs["camera_matrix"] >> cameraMatrix;
        fs["dist_coeffs"] >> distCoeffs;
        fs["rms"] >> totalRMS;
        fs["avg_error"] >> avgReprError;
        fs["max_error"] >> maxReprError;
        fs["fov_x"] >> fovX;
        fs["fov_y"] >> fovY;

        fs.release();
        return true;
      } catch (...) {
        return false;
      }
    }
  };

private:
  Settings settings;
  Results results;
  cv::Size imageSize;
  std::vector<std::vector<cv::Point3f>> objectPoints;
  std::vector<std::vector<cv::Point2f>> imagePoints;
  std::vector<cv::Mat> calibrationImages;
  std::atomic<bool> isCalibrated{false};
  std::shared_mutex mutex;

  // Create a grid of 3D points for the calibration pattern
  [[nodiscard]] std::vector<cv::Point3f> createObjectPoints() const {
    std::vector<cv::Point3f> points;
    points.reserve(settings.patternSize.width * settings.patternSize.height);

    for (int i = 0; i < settings.patternSize.height; i++) {
      for (int j = 0; j < settings.patternSize.width; j++) {
        points.emplace_back(j * settings.squareSize, i * settings.squareSize,
                            0);
      }
    }
    return points;
  }

public:
  explicit CameraCalibrator(
      const Settings &settings = Settings::createDefault())
      : settings(settings) {
    std::filesystem::create_directories(settings.outputDir);
  }

  // Enhanced pattern detection with modern C++ exception handling
  [[nodiscard]] std::optional<std::vector<cv::Point2f>>
  detectPattern(const cv::Mat &image, bool drawCorners = false) noexcept {
    if (image.empty())
      return std::nullopt;

    try {
      std::vector<cv::Point2f> corners;
      cv::Mat gray;

      // Convert to grayscale if needed
      if (image.channels() == 3 || image.channels() == 4)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
      else
        gray = image.clone();

      // Try different detection methods
      bool found = cv::findChessboardCorners(
          gray, settings.patternSize, corners,
          cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE +
              cv::CALIB_CB_FAST_CHECK);

      if (!found) {
        found = cv::findCirclesGrid(gray, settings.patternSize, corners,
                                    cv::CALIB_CB_ASYMMETRIC_GRID);
      }

      if (found) {
        // Refine corner locations
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

        return corners;
      }

      return std::nullopt;
    } catch (const cv::Exception &e) {
      const auto &logger = Logger::getInstance();
      logger->error("OpenCV error during pattern detection: {}", e.what());
      return std::nullopt;
    } catch (const std::exception &e) {
      const auto &logger = Logger::getInstance();
      logger->error("Error during pattern detection: {}", e.what());
      return std::nullopt;
    }
  }

  // Process a batch of images using C++20 coroutines (if available) or async
  bool processImages(const std::vector<std::string> &imageFiles) {
    const auto &logger = Logger::getInstance();
    logger->info("Processing {} calibration images", imageFiles.size());

    calibrationImages.clear();
    imagePoints.clear();
    objectPoints.clear();

    std::vector<std::future<
        std::optional<std::pair<cv::Mat, std::vector<cv::Point2f>>>>>
        futures;

    for (const auto &file : imageFiles) {
      futures.push_back(std::async(
          std::launch::async,
          [file, this]()
              -> std::optional<std::pair<cv::Mat, std::vector<cv::Point2f>>> {
            try {
              cv::Mat image = cv::imread(file);
              if (image.empty()) {
                const auto &logger = Logger::getInstance();
                logger->warn("Failed to load image: {}", file);
                return std::nullopt;
              }

              auto corners = detectPattern(image, true);
              if (!corners)
                return std::nullopt;

              // Set image size from first valid image
              if (imageSize.empty()) {
                std::unique_lock lock(mutex);
                if (imageSize.empty()) {
                  imageSize = image.size();
                }
              }

              return std::make_pair(image, *corners);
            } catch (...) {
              return std::nullopt;
            }
          }));
    }

    int validCount = 0;
    for (auto &future : futures) {
      auto result = future.get();
      if (result) {
        std::unique_lock lock(mutex);
        calibrationImages.push_back(result->first);
        imagePoints.push_back(result->second);
        objectPoints.push_back(createObjectPoints());
        validCount++;
      }
    }

    logger->info("Found valid patterns in {}/{} images", validCount,
                 imageFiles.size());
    return validCount >= settings.minImages;
  }

  // Modern calibration method with C++20 features
  [[nodiscard]] std::optional<Results> calibrate() {
    const auto &logger = Logger::getInstance();

    std::shared_lock readLock(mutex);
    if (imagePoints.empty() || imagePoints.size() < settings.minImages) {
      logger->error(
          "Insufficient valid images for calibration. Found {}, need {}",
          imagePoints.size(), settings.minImages);
      return std::nullopt;
    }

    if (imageSize.empty()) {
      logger->error("Image size not determined");
      return std::nullopt;
    }
    readLock.unlock();

    try {
      // Need exclusive lock for calibration
      std::unique_lock writeLock(mutex);

      // Initialize output matrices
      results.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
      results.distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

      // Perform calibration
      results.totalRMS = cv::calibrateCamera(
          objectPoints, imagePoints, imageSize, results.cameraMatrix,
          results.distCoeffs, results.rvecs, results.tvecs, settings.flags);

      isCalibrated = true;

      // Calculate detailed results
      calculateCalibrationResults();

      logger->info("Calibration completed with RMS error: {}",
                   results.totalRMS);

      // Save calibration parameters
      saveCalibrationData();

      return results;
    } catch (const cv::Exception &e) {
      logger->error("OpenCV error during calibration: {}", e.what());
      return std::nullopt;
    } catch (const std::exception &e) {
      logger->error("Error during calibration: {}", e.what());
      return std::nullopt;
    }
  }

  void calculateCalibrationResults() {
    const auto &logger = Logger::getInstance();

    if (!isCalibrated) {
      logger->warn("Cannot calculate results: not calibrated yet");
      return;
    }

    try {
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

      // Calculate FOV and other parameters
      cv::calibrationMatrixValues(results.cameraMatrix, imageSize, 0.0,
                                  0.0, // Assume sensor size unknown
                                  results.fovX, results.fovY,
                                  results.focalLength.x, results.principalPoint,
                                  results.aspectRatio);

      logger->info("Average reprojection error: {}", results.avgReprError);
    } catch (const std::exception &e) {
      logger->error("Error calculating calibration results: {}", e.what());
    }
  }

  void generateReport(const std::string &filename = "calibration_report.txt") {
    const auto &logger = Logger::getInstance();

    if (!isCalibrated) {
      logger->warn("Cannot generate report: not calibrated yet");
      return;
    }

    try {
      std::ofstream report(settings.outputDir + filename);
      if (!report.is_open()) {
        logger->error("Failed to open report file: {}",
                      settings.outputDir + filename);
        return;
      }

      report << "Camera Calibration Report\n";
      report << "========================\n\n";

      // Get current date and time
      auto now = std::chrono::system_clock::now();
      auto time_t = std::chrono::system_clock::to_time_t(now);
      std::string datetime = std::ctime(&time_t);
      datetime.pop_back(); // Remove trailing newline

      report << "Calibration Date: " << datetime << "\n\n";

      report << "Settings:\n";
      report << "- Pattern Size: " << settings.patternSize.width << "x"
             << settings.patternSize.height << "\n";
      report << "- Square Size: " << settings.squareSize << "mm\n";
      report << "- Number of images: " << calibrationImages.size() << "\n\n";

      report << "Results:\n";
      report << "- RMS Error: " << results.totalRMS << "\n";
      report << "- Average Reprojection Error: " << results.avgReprError
             << "\n";
      report << "- Maximum Reprojection Error: " << results.maxReprError
             << "\n";
      report << "- FOV: " << results.fovX << "x" << results.fovY
             << " degrees\n";
      report << "- Principal Point: (" << results.principalPoint.x << ", "
             << results.principalPoint.y << ")\n";
      report << "- Focal Length: (" << results.focalLength.x << ", "
             << results.focalLength.y << ")\n";
      report << "- Aspect Ratio: " << results.aspectRatio << "\n\n";

      report << "Camera Matrix:\n" << results.cameraMatrix << "\n\n";
      report << "Distortion Coefficients:\n" << results.distCoeffs << "\n";

      report.close();
      logger->info("Calibration report generated: {}",
                   settings.outputDir + filename);
    } catch (const std::exception &e) {
      logger->error("Error generating report: {}", e.what());
    }
  }

  void saveCalibrationData(const std::string &filename = "calibration.yml") {
    if (!isCalibrated)
      return;

    try {
      results.saveToFile(settings.outputDir + filename);
    } catch (...) {
      const auto &logger = Logger::getInstance();
      logger->error("Failed to save calibration data");
    }
  }

  cv::Mat undistortImage(const cv::Mat &input) const {
    if (!isCalibrated || input.empty()) {
      return input.clone();
    }

    cv::Mat output;
    cv::undistort(input, output, results.cameraMatrix, results.distCoeffs);
    return output;
  }

  // Generate visualization of calibration results
  void saveCalibrationVisualization() {
    const auto &logger = Logger::getInstance();

    if (!isCalibrated) {
      logger->warn("Cannot visualize: not calibrated yet");
      return;
    }

    std::shared_lock lock(mutex);
    auto images = calibrationImages; // Make a copy to safely unlock
    lock.unlock();

    try {
      for (size_t i = 0; i < images.size(); i++) {
        // Create undistorted image
        cv::Mat undistorted = undistortImage(images[i]);

        // Create side-by-side comparison
        cv::Mat comparison;
        cv::hconcat(images[i], undistorted, comparison);

        // Add error information
        std::string text =
            "RMS Error: " + (i < results.perViewErrors.size()
                                 ? std::to_string(results.perViewErrors[i])
                                 : "N/A");

        cv::putText(comparison, text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // Save comparison image
        std::string filename =
            settings.outputDir + "comparison_" + std::to_string(i) + ".jpg";
        cv::imwrite(filename, comparison);
      }
      logger->info("Saved {} calibration visualizations", images.size());
    } catch (const std::exception &e) {
      logger->error("Error saving visualizations: {}", e.what());
    }
  }

  // Getters
  [[nodiscard]] const Results &getResults() const noexcept { return results; }

  [[nodiscard]] bool isCalibrationValid() const noexcept {
    return isCalibrated && results.isValid();
  }

  [[nodiscard]] const cv::Mat &getCameraMatrix() const noexcept {
    return results.cameraMatrix;
  }

  [[nodiscard]] const cv::Mat &getDistCoeffs() const noexcept {
    return results.distCoeffs;
  }
};
