#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <opencv2/opencv.hpp>
#include <vector>

auto calculateHist(const cv::Mat &img, int histSize, bool normalize = true,
                   float threshold = 4.0f) -> std::vector<cv::Mat>;

auto calculateGrayHist(const cv::Mat &img, int histSize, bool normalize = true,
                       float threshold = 4.0f) -> cv::Mat;

auto calculateCDF(const cv::Mat &hist) -> cv::Mat;

auto equalizeHistogram(const cv::Mat &img, bool preserveColor = true)
    -> cv::Mat;

auto drawHistogram(const cv::Mat &hist, int histSize, int width, int height,
                   cv::Scalar color = cv::Scalar(255, 0, 0)) -> cv::Mat;

#endif // HISTOGRAM_H