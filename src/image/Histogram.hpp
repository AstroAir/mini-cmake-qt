#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <opencv2/opencv.hpp>
#include <vector>

// 扩展直方图计算的配置结构体
struct HistogramConfig {
    int histSize{256};
    bool normalize{true};
    float threshold{4.0f};
    int numThreads{-1};
    bool useLog{false};      // 是否使用对数尺度
    double gamma{1.0};       // gamma校正值
    std::array<float, 2> range{0.0f, 256.0f};  // 添加范围成员
    int channel{0};                             // 添加通道成员
};

struct HistogramStats {
    double mean{0.0};
    double stdDev{0.0};
    double skewness{0.0};
    double kurtosis{0.0};
    double entropy{0.0};
    double uniformity{0.0};
};

auto calculateHist(const cv::Mat &img, const HistogramConfig &config = {})
    -> std::vector<cv::Mat>;

auto calculateGrayHist(const cv::Mat &img, const HistogramConfig &config = {})
    -> cv::Mat;

auto calculateCDF(const cv::Mat &hist) -> cv::Mat;

struct EqualizeConfig {
    bool preserveColor{true};
    bool clipLimit{true};
    double clipValue{40.0};
};

auto equalizeHistogram(const cv::Mat &img, const EqualizeConfig &config = {})
    -> cv::Mat;

auto drawHistogram(const cv::Mat &hist, int width, int height,
                  cv::Scalar color = cv::Scalar(255, 0, 0),
                  bool cumulative = false) -> cv::Mat;

// 添加新的功能接口
auto calculateHistogramStats(const cv::Mat& hist) -> HistogramStats;

auto calculateEntropy(const cv::Mat& hist) -> double;

auto calculateUniformity(const cv::Mat& hist) -> double;

// 添加直方图匹配功能
auto matchHistograms(const cv::Mat& source, const cv::Mat& reference,
                    bool preserveColor = true) -> cv::Mat;

// 添加直方图反投影
auto backProjectHistogram(const cv::Mat& image, const cv::Mat& hist,
                        const HistogramConfig& config = {}) -> cv::Mat;

// 添加新的实用函数
auto compareHistograms(const cv::Mat &hist1, const cv::Mat &hist2,
                      int method = cv::HISTCMP_CORREL) -> double;

#endif // HISTOGRAM_H