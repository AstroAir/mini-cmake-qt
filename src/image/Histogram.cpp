#include "Histogram.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <immintrin.h>
#include <opencv2/imgproc.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>
#include <execution>
#include <span>
#include <ranges>
#include <future>

namespace {
const cv::Scalar DEFAULT_HIST_COLOR(255, 0, 0);
constexpr int DEFAULT_LINE_TYPE = cv::LINE_AA;
constexpr float DEFAULT_THRESHOLD = 4.0f;

// Use std::shared_ptr for thread-safe logger initialization
std::shared_ptr<spdlog::logger> getHistogramLogger() {
    static std::shared_ptr<spdlog::logger> logger = []() {
        try {
            return spdlog::basic_logger_mt("HistogramLogger", "logs/histogram.log");
        } catch (const spdlog::spdlog_ex& ex) {
            std::cerr << "Log initialization failed: " << ex.what() << std::endl;
            return spdlog::stdout_color_mt("HistogramLogger");
        }
    }();
    return logger;
}

// Validate input image and configuration
template<ImageType T>
bool validateImage(const T& img, int expectedChannels = -1) noexcept {
    if (img.empty()) {
        return false;
    }
    if (expectedChannels > 0 && img.channels() != expectedChannels) {
        return false;
    }
    return true;
}

// Helper for clipping values using std::views
template<std::floating_point T>
auto clip(T value, T min, T max) noexcept {
    return std::clamp(value, min, max);
}

// SIMD-optimized histogram calculation for 256-bin grayscale histograms
void calculateHistogramSIMD(const cv::Mat& img, float* histData) noexcept {
    alignas(32) std::array<int, 256> counts{};
    const uchar* data = img.data;
    const int size = img.rows * img.cols;
    
#ifdef __AVX2__
    // Process 32 pixels at a time with AVX2
    alignas(32) std::array<int, 32> temp{};
    const int simdSize = size - (size % 32);
    
    for (int i = 0; i < simdSize; i += 32) {
        __m256i pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        
        // Extract and count each pixel value
        for (int j = 0; j < 32; ++j) {
            ++counts[_mm256_extract_epi8(pixels, j)];
        }
    }
    
    // Handle remaining pixels
    for (int i = simdSize; i < size; ++i) {
        ++counts[data[i]];
    }
#else
    // Fallback to plain loop
    for (int i = 0; i < size; ++i) {
        ++counts[data[i]];
    }
#endif

    // Convert to float
    std::transform(std::execution::par_unseq, counts.begin(), counts.end(), 
                   histData, [](int count) { return static_cast<float>(count); });
}

} // namespace

auto calculateHist(const cv::Mat &img, const HistogramConfig &config)
    -> HistogramResult<std::vector<cv::Mat>> {
    
    auto logger = getHistogramLogger();
    logger->info("Calculating BGR histograms with histSize: {}", config.histSize);
    
    HistogramResult<std::vector<cv::Mat>> result;
    
    try {
        // Validate input parameters
        if (!validateImage(img, 3)) {
            result.status = HistogramStatus::EmptyImage;
            result.message = "Input image is empty or doesn't have 3 channels";
            logger->error(result.message);
            return result;
        }
        
        if (config.histSize <= 0) {
            result.status = HistogramStatus::InvalidSize;
            result.message = "Histogram size must be positive";
            logger->error(result.message);
            return result;
        }
        
        // Prepare channels
        std::array<cv::Mat, 3> bgrPlanes;
        cv::split(img, bgrPlanes);
        
        const int dims = 1;
        const std::array<int, 1> channels{0};
        const float *ranges[] = {config.range.data()};
        
        std::vector<cv::Mat> histograms(3);
        
        const int numThreads = (config.numThreads <= 0) ? 
                              std::thread::hardware_concurrency() : 
                              config.numThreads;
        
        // Process each channel in parallel using std::async C++20 style
        std::array<std::future<void>, 3> futures;
        for (int i = 0; i < 3; ++i) {
            futures[i] = std::async(std::launch::async, [&, i]() {
                cv::calcHist(&bgrPlanes[i], 1, channels.data(), cv::Mat(),
                            histograms[i], dims, &config.histSize, ranges);
                
                if (config.useLog) {
                    cv::log(histograms[i] + 1, histograms[i]);
                }
                
                if (config.gamma != 1.0) {
                    cv::pow(histograms[i], config.gamma, histograms[i]);
                }
                
                if (config.threshold > 0) {
                    cv::threshold(histograms[i], histograms[i], config.threshold, 0,
                                cv::THRESH_TOZERO);
                }
                
                if (config.normalize) {
                    cv::normalize(histograms[i], histograms[i], 0, 1, cv::NORM_MINMAX);
                }
            });
        }
        
        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.get();
        }
        
        // Set successful result
        result.value = std::move(histograms);
        result.status = HistogramStatus::Success;
        
    } catch (const cv::Exception& e) {
        result.status = HistogramStatus::ProcessError;
        result.message = std::string("OpenCV error: ") + e.what();
        logger->error(result.message);
    } catch (const std::exception& e) {
        result.status = HistogramStatus::ProcessError;
        result.message = std::string("Error: ") + e.what();
        logger->error(result.message);
    }
    
    return result;
}

auto calculateGrayHist(const cv::Mat &img, const HistogramConfig &config)
    -> HistogramResult<cv::Mat> {
    
    auto logger = getHistogramLogger();
    logger->info("Calculating grayscale histogram with histSize: {}", config.histSize);
    
    HistogramResult<cv::Mat> result;
    
    try {
        // Validate input
        if (!validateImage(img, 1)) {
            result.status = HistogramStatus::EmptyImage;
            result.message = "Input image is empty or not grayscale";
            logger->error(result.message);
            return result;
        }
        
        if (config.histSize <= 0) {
            result.status = HistogramStatus::InvalidSize;
            result.message = "Histogram size must be positive";
            logger->error(result.message);
            return result;
        }
        
        if (config.channel < 0 || config.channel >= img.channels()) {
            result.status = HistogramStatus::InvalidChannel;
            result.message = "Invalid channel specified";
            logger->error(result.message);
            return result;
        }
        
        // Standard histogram calculation
        const std::array<int, 1> channels{config.channel};
        const float *ranges[] = {config.range.data()};
        
        cv::Mat grayHist;
        
        // Use optimized SIMD implementation for standard 256-bin case
        if (config.histSize == 256 && config.useSIMD) {
            grayHist = cv::Mat::zeros(256, 1, CV_32F);
            calculateHistogramSIMD(img, grayHist.ptr<float>());
        } else {
            cv::calcHist(&img, 1, channels.data(), cv::Mat(), grayHist, 1,
                        &config.histSize, ranges);
        }
        
        // Post-processing
        if (config.useLog) {
            cv::log(grayHist + 1, grayHist);
        }
        
        if (config.gamma != 1.0) {
            cv::pow(grayHist, config.gamma, grayHist);
        }
        
        if (config.threshold > 0) {
            cv::threshold(grayHist, grayHist, config.threshold, 0, cv::THRESH_TOZERO);
        }
        
        if (config.normalize) {
            cv::normalize(grayHist, grayHist, 0, 1, cv::NORM_MINMAX);
        }
        
        result.value = std::move(grayHist);
        result.status = HistogramStatus::Success;
        logger->info("Completed grayscale histogram calculation");
        
    } catch (const cv::Exception& e) {
        result.status = HistogramStatus::ProcessError;
        result.message = std::string("OpenCV error: ") + e.what();
        logger->error(result.message);
    } catch (const std::exception& e) {
        result.status = HistogramStatus::ProcessError;
        result.message = std::string("Error: ") + e.what();
        logger->error(result.message);
    }
    
    return result;
}

auto calculateCDF(const cv::Mat &hist) -> HistogramResult<cv::Mat> {
    auto logger = getHistogramLogger();
    logger->info("Calculating CDF");
    
    HistogramResult<cv::Mat> result;
    
    try {
        if (hist.empty()) {
            result.status = HistogramStatus::EmptyImage;
            result.message = "Input histogram is empty";
            logger->error(result.message);
            return result;
        }
        
        cv::Mat cdf = hist.clone();
        float *data = cdf.ptr<float>();
        const int size = hist.rows;
        
        // Optimized parallel scan algorithm using C++20 features
        const int blockSize = std::min(256, size);
        std::vector<float> blockSums((size + blockSize - 1) / blockSize);
        
        // Compute local sums
        std::for_each(std::execution::par, 
                     blockSums.begin(), blockSums.end(),
                     [&](float& blockSum) {
            const int blockIdx = &blockSum - &blockSums[0];
            const int start = blockIdx * blockSize;
            const int end = std::min(start + blockSize, size);
            float sum = 0;
            for (int i = start; i < end; ++i) {
                sum += data[i];
                data[i] = sum;
            }
            blockSum = sum;
        });
        
        // Perform prefix sum on block sums
        std::exclusive_scan(std::execution::par_unseq, 
                           blockSums.begin(), blockSums.end(), 
                           blockSums.begin(), 0.0f);
        
        // Add block sums to each element except for the first block
        std::for_each(std::execution::par,
                     std::views::iota(1, static_cast<int>(blockSums.size())),
                     [&](int blockIdx) {
            const int start = blockIdx * blockSize;
            const int end = std::min(start + blockSize, size);
            const float blockSum = blockSums[blockIdx];
            for (int i = start; i < end; ++i) {
                data[i] += blockSum;
            }
        });
        
        result.value = std::move(cdf);
        result.status = HistogramStatus::Success;
        
    } catch (const std::exception& e) {
        result.status = HistogramStatus::ProcessError;
        result.message = std::string("Error calculating CDF: ") + e.what();
        logger->error(result.message);
    }
    
    return result;
}

auto equalizeHistogram(const cv::Mat &img, const EqualizeConfig &config)
    -> HistogramResult<cv::Mat> {
    
    auto logger = getHistogramLogger();
    logger->info("Starting histogram equalization with clip limit: {}", 
                config.clipLimit ? "enabled" : "disabled");
    
    HistogramResult<cv::Mat> result;
    
    try {
        if (!validateImage(img)) {
            result.status = HistogramStatus::EmptyImage;
            result.message = "Input image is empty";
            logger->error(result.message);
            return result;
        }
        
        cv::Mat equalized;
        
        if (img.channels() == 1) {
            // Single channel image equalization
            if (config.clipLimit) {
                auto clahe = cv::createCLAHE(clip(config.clipValue, 0.0, 100.0));
                clahe->apply(img, equalized);
            } else {
                cv::equalizeHist(img, equalized);
            }
        } else {
            // Multi-channel processing
            if (config.preserveColor) {
                cv::Mat ycrcb;
                cv::cvtColor(img, ycrcb, cv::COLOR_BGR2YCrCb);
                std::vector<cv::Mat> channels(3);
                cv::split(ycrcb, channels);
                
                // Only equalize luminance channel
                if (config.clipLimit) {
                    auto clahe = cv::createCLAHE(clip(config.clipValue, 0.0, 100.0));
                    clahe->apply(channels[0], channels[0]);
                } else {
                    cv::equalizeHist(channels[0], channels[0]);
                }
                
                cv::merge(channels, ycrcb);
                cv::cvtColor(ycrcb, equalized, cv::COLOR_YCrCb2BGR);
            } else {
                // Apply to all channels independently
                std::vector<cv::Mat> channels;
                cv::split(img, channels);
                
                std::for_each(std::execution::par, 
                             channels.begin(), channels.end(),
                             [&](cv::Mat& channel) {
                    if (config.clipLimit) {
                        auto clahe = cv::createCLAHE(clip(config.clipValue, 0.0, 100.0));
                        clahe->apply(channel, channel);
                    } else {
                        cv::equalizeHist(channel, channel);
                    }
                });
                
                cv::merge(channels, equalized);
            }
        }
        
        result.value = std::move(equalized);
        result.status = HistogramStatus::Success;
        
    } catch (const cv::Exception& e) {
        result.status = HistogramStatus::ProcessError;
        result.message = std::string("OpenCV error: ") + e.what();
        logger->error(result.message);
    } catch (const std::exception& e) {
        result.status = HistogramStatus::ProcessError;
        result.message = std::string("Error: ") + e.what();
        logger->error(result.message);
    }
    
    return result;
}

auto drawHistogram(const cv::Mat &hist, int width, int height, cv::Scalar color,
                  bool cumulative) -> HistogramResult<cv::Mat> {
    
    auto logger = getHistogramLogger();
    HistogramResult<cv::Mat> result;
    
    try {
        // Input validation
        if (hist.empty()) {
            result.status = HistogramStatus::EmptyImage;
            result.message = "Input histogram is empty";
            logger->error(result.message);
            return result;
        }
        
        if (width <= 0 || height <= 0) {
            result.status = HistogramStatus::InvalidSize;
            result.message = "Invalid width or height";
            logger->error(result.message);
            return result;
        }
        
        cv::Mat histImage(height, width, CV_8UC3, cv::Scalar::all(0));
        
        cv::Mat displayHist;
        if (cumulative) {
            auto cdfResult = calculateCDF(hist);
            if (!cdfResult) {
                return {std::nullopt, cdfResult.status, cdfResult.message};
            }
            displayHist = cdfResult.value;
        } else {
            cv::normalize(hist, displayHist, 0, height, cv::NORM_MINMAX);
        }
        
        // Safely calculate bin width
        const int binWidth = std::max(1, cvRound(static_cast<double>(width) / 
                                               std::max(1, hist.rows)));
        
        // Reserve capacity for vector
        std::vector<cv::Point> points;
        points.reserve(hist.rows + 2);
        
        // Create contour points using range-based iteration
        points.emplace_back(0, height);
        
        for (int i = 0; i < hist.rows; ++i) {
            int binHeight = cvRound(displayHist.at<float>(i) * height);
            // Clamp values to prevent overflowing the image
            binHeight = std::clamp(binHeight, 0, height);
            points.emplace_back(binWidth * i, height - binHeight);
        }
        
        points.emplace_back(width, height);
        
        // Draw the histogram
        const std::vector<std::vector<cv::Point>> contours{points};
        if (cumulative) {
            cv::fillPoly(histImage, contours, color * 0.5);
        }
        cv::polylines(histImage, points, false, color, 2, DEFAULT_LINE_TYPE);
        
        result.value = std::move(histImage);
        result.status = HistogramStatus::Success;
        
    } catch (const std::exception& e) {
        result.status = HistogramStatus::ProcessError;
        result.message = std::string("Error drawing histogram: ") + e.what();
        logger->error(result.message);
    }
    
    return result;
}

auto compareHistograms(const cv::Mat &hist1, const cv::Mat &hist2, int method)
    -> HistogramResult<double> {
    
    auto logger = getHistogramLogger();
    HistogramResult<double> result;
    
    try {
        // Validate inputs
        if (hist1.empty() || hist2.empty()) {
            result.status = HistogramStatus::EmptyImage;
            result.message = "One or both input histograms are empty";
            logger->error(result.message);
            return result;
        }
        
        // Validate method
        if (method < 0 || method > 4) {  // CV_COMP_* values are 0-4
            result.status = HistogramStatus::InvalidSize;
            result.message = "Invalid comparison method";
            logger->error(result.message);
            return result;
        }
        
        // Compare histograms
        double comparison = cv::compareHist(hist1, hist2, method);
        
        result.value = comparison;
        result.status = HistogramStatus::Success;
        
    } catch (const cv::Exception& e) {
        result.status = HistogramStatus::ProcessError;
        result.message = std::string("OpenCV error: ") + e.what();
        logger->error(result.message);
    } catch (const std::exception& e) {
        result.status = HistogramStatus::ProcessError;
        result.message = std::string("Error: ") + e.what();
        logger->error(result.message);
    }
    
    return result;
}

auto calculateHistogramStats(const cv::Mat &hist) noexcept
    -> HistogramResult<HistogramStats> {
    
    auto logger = getHistogramLogger();
    HistogramResult<HistogramStats> result;
    
    try {
        if (hist.empty()) {
            result.status = HistogramStatus::EmptyImage;
            result.message = "Input histogram is empty";
            return result;
        }
        
        HistogramStats stats;
        
        // Normalize histogram if needed
        cv::Mat normHist;
        if (std::abs(cv::sum(hist)[0] - 1.0) > 1e-6) {
            cv::normalize(hist, normHist, 0, 1, cv::NORM_L1);
        } else {
            normHist = hist;
        }
        
        // Use C++20 parallelism for reduction operations
        // Calculate mean
        const int size = normHist.rows;
        std::span<const float> histData(normHist.ptr<float>(), size);
        
        // Calculate mean (first moment)
        stats.mean = std::transform_reduce(
            std::execution::par_unseq,
            std::views::iota(0, size).begin(),
            std::views::iota(0, size).end(),
            0.0,
            std::plus<>(),
            [&histData](int i) { return i * histData[i]; }
        );
        
        // Calculate higher moments in parallel
        auto [m2, m3, m4] = std::transform_reduce(
            std::execution::par_unseq,
            std::views::iota(0, size).begin(),
            std::views::iota(0, size).end(),
            std::tuple(0.0, 0.0, 0.0),
            [](const auto& a, const auto& b) {
                return std::tuple(
                    std::get<0>(a) + std::get<0>(b),
                    std::get<1>(a) + std::get<1>(b),
                    std::get<2>(a) + std::get<2>(b)
                );
            },
            [&](int i) {
                const double diff = i - stats.mean;
                const double diff2 = diff * diff;
                const double w = histData[i];
                return std::tuple(
                    diff2 * w,              // m2
                    diff * diff2 * w,       // m3
                    diff2 * diff2 * w       // m4
                );
            }
        );
        
        // Calculate statistics from moments
        stats.stdDev = std::sqrt(m2);
        stats.skewness = (stats.stdDev > 1e-10) ? m3 / std::pow(stats.stdDev, 3) : 0;
        stats.kurtosis = (m2 > 1e-10) ? (m4 / (m2 * m2)) - 3.0 : 0;
        
        // Calculate entropy and uniformity
        auto entropyResult = calculateEntropy(normHist);
        auto uniformityResult = calculateUniformity(normHist);
        
        if (entropyResult) stats.entropy = *entropyResult.value;
        if (uniformityResult) stats.uniformity = *uniformityResult.value;
        
        result.value = stats;
        result.status = HistogramStatus::Success;
        
    } catch (...) {
        result.status = HistogramStatus::ProcessError;
        result.message = "Error calculating histogram statistics";
    }
    
    return result;
}

auto calculateEntropy(const cv::Mat &hist) noexcept -> HistogramResult<double> {
    HistogramResult<double> result;
    
    try {
        if (hist.empty()) {
            result.status = HistogramStatus::EmptyImage;
            result.message = "Input histogram is empty";
            return result;
        }
        
        // Use C++20 parallelism for calculation
        const int size = hist.rows;
        std::span<const float> histData(hist.ptr<float>(), size);
        
        double entropy = std::transform_reduce(
            std::execution::par_unseq,
            histData.begin(), histData.end(),
            0.0,
            std::plus<>(),
            [](float p) {
                return (p > 0) ? -p * std::log2(p) : 0.0;
            }
        );
        
        result.value = entropy;
        result.status = HistogramStatus::Success;
        
    } catch (...) {
        result.status = HistogramStatus::ProcessError;
        result.message = "Error calculating entropy";
    }
    
    return result;
}

auto calculateUniformity(const cv::Mat &hist) noexcept -> HistogramResult<double> {
    HistogramResult<double> result;
    
    try {
        if (hist.empty()) {
            result.status = HistogramStatus::EmptyImage;
            result.message = "Input histogram is empty";
            return result;
        }
        
        // Use C++20 parallelism for calculation
        const int size = hist.rows;
        std::span<const float> histData(hist.ptr<float>(), size);
        
        double uniformity = std::transform_reduce(
            std::execution::par_unseq,
            histData.begin(), histData.end(),
            0.0,
            std::plus<>(),
            [](float p) {
                return p * p;
            }
        );
        
        result.value = uniformity;
        result.status = HistogramStatus::Success;
        
    } catch (...) {
        result.status = HistogramStatus::ProcessError;
        result.message = "Error calculating uniformity";
    }
    
    return result;
}

auto matchHistograms(const cv::Mat &source, const cv::Mat &reference,
                     bool preserveColor) -> HistogramResult<cv::Mat> {
    auto logger = getHistogramLogger();
    HistogramResult<cv::Mat> result;
    
    try {
        if (source.empty() || reference.empty()) {
            result.status = HistogramStatus::EmptyImage;
            result.message = "Empty input image";
            logger->error(result.message);
            return result;
        }
        
        cv::Mat matched;
        
        if (source.channels() == 1 || !preserveColor) {
            auto srcHistResult = calculateGrayHist(source, {256, {0, 256}});
            auto refHistResult = calculateGrayHist(reference, {256, {0, 256}});
            
            if (!srcHistResult || !refHistResult) {
                return {std::nullopt, HistogramStatus::ProcessError, "Error calculating histograms"};
            }
            
            auto srcCdfResult = calculateCDF(srcHistResult.value);
            auto refCdfResult = calculateCDF(refHistResult.value);
            
            if (!srcCdfResult || !refCdfResult) {
                return {std::nullopt, HistogramStatus::ProcessError, "Error calculating CDFs"};
            }
            
            cv::Mat lut(1, 256, CV_8U);
            for (int i = 0; i < 256; ++i) {
                int j = 0;
                while (j < 256 && refCdfResult.value.at<float>(j) <= srcCdfResult.value.at<float>(i)) {
                    ++j;
                }
                lut.at<uchar>(i) = static_cast<uchar>(j);
            }
            
            cv::LUT(source, lut, matched);
        } else {
            cv::Mat ycrcb, refYcrcb;
            cv::cvtColor(source, ycrcb, cv::COLOR_BGR2YCrCb);
            cv::cvtColor(reference, refYcrcb, cv::COLOR_BGR2YCrCb);
            
            std::vector<cv::Mat> channels, refChannels;
            cv::split(ycrcb, channels);
            cv::split(refYcrcb, refChannels);
            
            // Only match luminance channel
            auto srcHistResult = calculateGrayHist(channels[0], {256, {0, 256}});
            auto refHistResult = calculateGrayHist(refChannels[0], {256, {0, 256}});
            
            if (!srcHistResult || !refHistResult) {
                return {std::nullopt, HistogramStatus::ProcessError, "Error calculating histograms"};
            }
            
            auto srcCdfResult = calculateCDF(srcHistResult.value);
            auto refCdfResult = calculateCDF(refHistResult.value);
            
            if (!srcCdfResult || !refCdfResult) {
                return {std::nullopt, HistogramStatus::ProcessError, "Error calculating CDFs"};
            }
            
            cv::Mat lut(1, 256, CV_8U);
            for (int i = 0; i < 256; ++i) {
                int j = 0;
                while (j < 256 && refCdfResult.value.at<float>(j) <= srcCdfResult.value.at<float>(i)) {
                    ++j;
                }
                lut.at<uchar>(i) = static_cast<uchar>(j);
            }
            
            cv::LUT(channels[0], lut, channels[0]);
            cv::merge(channels, ycrcb);
            cv::cvtColor(ycrcb, matched, cv::COLOR_YCrCb2BGR);
        }
        
        result.value = std::move(matched);
        result.status = HistogramStatus::Success;
        
    } catch (const cv::Exception& e) {
        result.status = HistogramStatus::ProcessError;
        result.message = std::string("OpenCV error: ") + e.what();
        logger->error(result.message);
    } catch (const std::exception& e) {
        result.status = HistogramStatus::ProcessError;
        result.message = std::string("Error: ") + e.what();
        logger->error(result.message);
    }
    
    return result;
}

auto backProjectHistogram(const cv::Mat &image, const cv::Mat &hist,
                          const HistogramConfig &config) -> HistogramResult<cv::Mat> {
    auto logger = getHistogramLogger();
    HistogramResult<cv::Mat> result;
    
    try {
        if (image.empty() || hist.empty()) {
            result.status = HistogramStatus::EmptyImage;
            result.message = "Empty input image or histogram";
            logger->error(result.message);
            return result;
        }
        
        cv::Mat backProj;
        const float *ranges[] = {config.range.data()};
        cv::calcBackProject(&image, 1, &config.channel, hist, backProj, ranges, 1, true);
        
        if (config.normalize) {
            cv::normalize(backProj, backProj, 0, 255, cv::NORM_MINMAX);
        }
        
        result.value = std::move(backProj);
        result.status = HistogramStatus::Success;
        
    } catch (const cv::Exception& e) {
        result.status = HistogramStatus::ProcessError;
        result.message = std::string("OpenCV error: ") + e.what();
        logger->error(result.message);
    } catch (const std::exception& e) {
        result.status = HistogramStatus::ProcessError;
        result.message = std::string("Error: ") + e.what();
        logger->error(result.message);
    }
    
    return result;
}