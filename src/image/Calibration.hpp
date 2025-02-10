#pragma once

#include <opencv2/opencv.hpp>

struct CalibrationParams {
  double wavelength;         // 波长，单位：纳米
  double aperture;           // 光圈直径，单位：毫米
  double obstruction;        // 遮挡直径，单位：毫米
  double filter_width;       // 滤光器带宽，单位：纳米
  double transmissivity;     // 透射率
  double gain;               // 增益
  double quantum_efficiency; // 量子效率
  double extinction;         // 消光系数
  double exposure_time;      // 曝光时间，单位：秒
};

struct OptimizationParams {
    bool use_gpu{false};         // 是否使用GPU加速
    bool use_parallel{false};    // 是否使用并行处理
    int num_threads{4};          // 并行处理线程数
    bool use_cache{false};       // 是否使用缓存
    size_t cache_size{1024};     // 缓存大小（MB）
};

cv::Mat instrument_response_correction(cv::InputArray &image,
                                       cv::InputArray &response_function);

cv::Mat background_noise_correction(cv::InputArray &image);

cv::Mat apply_flat_field_correction(cv::InputArray &image,
                                    cv::InputArray &flat_field);

cv::Mat apply_dark_frame_subtraction(cv::InputArray &image,
                                     cv::InputArray &dark_frame);

double compute_flx2dn(const CalibrationParams &params);

std::tuple<cv::Mat, double, double, double>
flux_calibration_ex(const cv::Mat &image, const CalibrationParams &params,
                    const cv::Mat *response_function = nullptr,
                    const cv::Mat *flat_field = nullptr,
                    const cv::Mat *dark_frame = nullptr,
                    bool enable_optimization = false);

cv::Mat instrument_response_correction_optimized(cv::InputArray &image,
                                              cv::InputArray &response_function,
                                              const OptimizationParams &params);

cv::Mat background_noise_correction_optimized(cv::InputArray &image,
                                           const OptimizationParams &params);