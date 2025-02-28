#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <concepts>
#include <span>
#include <array>
#include <optional>
#include <type_traits>
#include <expected>
#include <source_location>
#include <string_view>

// Advanced histogram calculation configuration
struct HistogramConfig {
    int histSize{256};
    bool normalize{true};
    float threshold{4.0f};
    int numThreads{-1};
    bool useLog{false};      // Use logarithmic scale
    double gamma{1.0};       // Gamma correction value
    std::array<float, 2> range{0.0f, 256.0f};  // Value range
    int channel{0};          // Channel to process
    bool useGPU{false};      // Use GPU acceleration
    bool useSIMD{true};      // Use SIMD instructions
    size_t blockSize{256};   // Block size for processing
};

struct HistogramStats {
    double mean{0.0};
    double stdDev{0.0};
    double skewness{0.0};
    double kurtosis{0.0};
    double entropy{0.0};
    double uniformity{0.0};
    
    // C++20 operator overloading with defaulted spaceship operator
    auto operator<=>(const HistogramStats&) const = default;
};

// Exception hierarchy with source location
class HistogramException : public std::runtime_error {
    std::source_location location_;
public:
    explicit HistogramException(const std::string& msg, 
                              const std::source_location& location = std::source_location::current())
        : std::runtime_error(msg), location_(location) {}
    
    auto source_location() const noexcept -> const std::source_location& {
        return location_;
    }
};

// More specific exceptions
class EmptyImageException : public HistogramException {
public:
    using HistogramException::HistogramException;
};

class InvalidChannelException : public HistogramException {
public:
    using HistogramException::HistogramException;
};

// Error status enum
enum class HistogramStatus {
    Success,
    EmptyImage,
    InvalidChannel,
    InvalidSize,
    ProcessError
};

// Result wrapper with C++23 std::expected (could use a backport for C++20)
template<typename T>
struct HistogramResult {
    std::optional<T> value;
    HistogramStatus status{HistogramStatus::Success};
    std::string message;
    
    explicit operator bool() const noexcept {
        return status == HistogramStatus::Success && value.has_value();
    }
    
    // C++20 implicit conversion to value type when successful
    operator const T&() const {
        if (!value.has_value()) {
            throw HistogramException(message);
        }
        return *value;
    }
};

// Concept for image types
template<typename T>
concept ImageType = requires(T img) {
    { img.empty() } -> std::convertible_to<bool>;
    { img.channels() } -> std::convertible_to<int>;
    { img.rows } -> std::convertible_to<int>;
    { img.cols } -> std::convertible_to<int>;
};

// Function declarations with noexcept specifications and concepts
auto calculateHist(const cv::Mat &img, const HistogramConfig &config = {})
    -> HistogramResult<std::vector<cv::Mat>>;

auto calculateGrayHist(const cv::Mat &img, const HistogramConfig &config = {})
    -> HistogramResult<cv::Mat>;

auto calculateCDF(const cv::Mat &hist) 
    -> HistogramResult<cv::Mat>;

struct EqualizeConfig {
    bool preserveColor{true};
    bool clipLimit{true};
    double clipValue{40.0};
};

auto equalizeHistogram(const cv::Mat &img, const EqualizeConfig &config = {})
    -> HistogramResult<cv::Mat>;

auto drawHistogram(const cv::Mat &hist, int width, int height,
                  cv::Scalar color = cv::Scalar(255, 0, 0),
                  bool cumulative = false) -> HistogramResult<cv::Mat>;

// Additional functionality with improved signatures
auto calculateHistogramStats(const cv::Mat& hist) noexcept
    -> HistogramResult<HistogramStats>;

auto calculateEntropy(const cv::Mat& hist) noexcept
    -> HistogramResult<double>;

auto calculateUniformity(const cv::Mat& hist) noexcept
    -> HistogramResult<double>;

// Histogram matching with better error handling
auto matchHistograms(const cv::Mat& source, const cv::Mat& reference,
                    bool preserveColor = true) -> HistogramResult<cv::Mat>;

// Histogram back projection with boundary checks
auto backProjectHistogram(const cv::Mat& image, const cv::Mat& hist,
                        const HistogramConfig& config = {}) -> HistogramResult<cv::Mat>;

// Histogram comparison with method validation
auto compareHistograms(const cv::Mat &hist1, const cv::Mat &hist2,
                      int method = cv::HISTCMP_CORREL) -> HistogramResult<double>;

#endif // HISTOGRAM_H