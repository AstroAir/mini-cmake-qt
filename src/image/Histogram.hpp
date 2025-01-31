#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <opencv2/opencv.hpp>
#include <vector>

// 添加直方图计算的配置结构体
struct HistogramConfig {
    int histSize{256};
    bool normalize{true};
    float threshold{4.0f};
    int numThreads{-1}; // -1表示使用系统默认线程数
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

// 添加新的实用函数
auto compareHistograms(const cv::Mat &hist1, const cv::Mat &hist2,
                      int method = cv::HISTCMP_CORREL) -> double;

#endif // HISTOGRAM_H