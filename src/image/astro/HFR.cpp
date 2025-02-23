/**
 * @file HFR.cpp
 * @brief Implements functions for star detection and half-flux radius (HFR)
 * calculation.
 *
 * This file contains implementations for image processing functions that detect
 * stars in an astronomical image and compute the HFR value associated with each
 * detected star. The processing pipeline includes image preprocessing, noise
 * removal, contour detection, and several shape validations based on the
 * geometric and brightness properties of star-like objects.
 *
 * Functions:
 *
 * - checkElongated(int width, int height)
 *   @brief Checks if a given rectangular region is elongated.
 *   @param width The width of the bounding rectangle.
 *   @param height The height of the bounding rectangle.
 *   @return true if the ratio between width and height exceeds a predefined
 * threshold.
 *
 * - defineNarrowRadius(int minArea, double maxArea, double area, double scale)
 *   @brief Determines the number of narrow radius checks and corresponding
 * parameters based on the area.
 *   @param minArea The minimum area threshold.
 *   @param maxArea The maximum area threshold.
 *   @param area The current contour area.
 *   @param scale The scaling factor based on image size.
 *   @return A tuple containing the number of checks, a list of radii (as
 * integers), and a list of corresponding threshold values.
 *
 * - checkWhitePixel(const cv::Mat &rect_contour, int xCoord, int yCoord)
 *   @brief Determines if a pixel located at given coordinates in the image
 * matrix is white.
 *   @param rect_contour The image matrix to be checked.
 *   @param xCoord The x-coordinate of the pixel.
 *   @param yCoord The y-coordinate of the pixel.
 *   @return An integer representing a boolean value (non-zero if white, zero
 * otherwise).
 *
 * - eightSymmetryCircleCheck(const cv::Mat &rect_contour, const cv::Point
 * &center, int xCoord, int yCoord)
 *   @brief Performs an eight-way symmetry check for circle pixels around a
 * center point.
 *   @param rect_contour The image region containing the contour.
 *   @param center The center point of the circle.
 *   @param xCoord The x offset from center for symmetry check.
 *   @param yCoord The y offset from center for symmetry check.
 *   @return The count of white pixels detected in symmetric positions.
 *
 * - fourSymmetryCircleCheck(const cv::Mat &rect_contour, const cv::Point
 * &center, float radius)
 *   @brief Checks four cardinal symmetry positions on a circle boundary.
 *   @param rect_contour The image region containing the contour.
 *   @param center The center point of the circle.
 *   @param radius The radius at which to perform the check.
 *   @return The count of white pixels at the four symmetric positions.
 *
 * - checkBresenhamCircle(const cv::Mat &rect_contour, float radius, float
 * pixelRatio, bool ifDebug)
 *   @brief Uses Bresenham's circle algorithm to evaluate if the pixel ratio
 * within the circle meets a threshold.
 *   @param rect_contour The image region that represents the contour.
 *   @param radius The radius used for the circle check.
 *   @param pixelRatio The threshold ratio of white pixels.
 *   @param ifDebug Flag to indicate if debugging output should be generated.
 *   @return true if the ratio of white pixels exceeds pixelRatio, false
 * otherwise.
 *
 * - calcHfr(const cv::Mat &inImage, float radius)
 *   @brief Calculates the half-flux radius (HFR) by evaluating pixel brightness
 * weighted by their distance.
 *   @param inImage The input image (or its ROI) converted to 32F.
 *   @param radius The estimated radius of the star.
 *   @return The calculated HFR value.
 *
 * - caldim(const cv::Mat &img)
 *   @brief Checks if a given image region is dim by thresholding the grayscale
 * brightness.
 *   @param img The input image for which to evaluate brightness.
 *   @return true if the non-zero pixel ratio is below a predefined threshold,
 * false otherwise.
 *
 * - preprocessImage(const cv::Mat &img, cv::Mat &grayimg, cv::Mat &rgbImg,
 * cv::Mat &mark_img)
 *   @brief Preprocesses the input image by converting it to grayscale and RGB
 * formats as required.
 *   @param img The original input image.
 *   @param grayimg Output grayscale image.
 *   @param rgbImg Output RGB image.
 *   @param mark_img Image used for marking detected stars.
 *
 * - removeNoise(cv::Mat &map, bool if_removehotpixel, bool if_noiseremoval)
 *   @brief Performs noise removal on the image using median filtering and
 * Gaussian smoothing.
 *   @param map The image to be processed for noise reduction.
 *   @param if_removehotpixel Flag indicating whether hot pixels should be
 * removed.
 *   @param if_noiseremoval Flag indicating whether additional noise removal
 * should be applied.
 *
 * - calculateMeanAndStd(const cv::Mat &map, bool down_sample_mean_std, double
 * &medianVal, double &stdDev)
 *   @brief Computes the mean and standard deviation of image brightness with an
 * option for downsampling.
 *   @param map The input image (or noise-reduced map).
 *   @param down_sample_mean_std Flag indicating whether to downsample for
 * faster computation.
 *   @param medianVal Output value representing the mean (or median) brightness.
 *   @param stdDev Output standard deviation of the brightness.
 *
 * - processContours(const cv::Mat &grayimg, const cv::Mat &rgbImg, cv::Mat
 * &mark_img, const std::vector<std::vector<cv::Point>> &contours, double
 * threshold, bool do_star_mark)
 *   @brief Processes detected contours to filter potential stars and compute
 * HFR for valid regions. This includes checks for area, elongation, brightness
 * consistency (caldim), and Bresenham circle criteria.
 *   @param grayimg Grayscale image used for contour evaluation.
 *   @param rgbImg RGB image used for star marking.
 *   @param mark_img Image on which the detected stars and HFR are annotated.
 *   @param contours Vector of contours (each a vector of points) detected in
 * the binary image.
 *   @param threshold The threshold value used for binarization.
 *   @param do_star_mark Flag to indicate whether to annotate the stars on
 * mark_img.
 *   @return A tuple containing:
 *           - The number of valid star contours detected.
 *           - The average HFR over all detected stars.
 *           - A vector of HFR values for each star.
 *           - A vector of area measurements for each star.
 *
 * - starDetectAndHfr(const cv::Mat &img, bool if_removehotpixel, bool
 * if_noiseremoval, bool do_star_mark, bool down_sample_mean_std, cv::Mat
 * mark_img)
 *   @brief Main function to perform star detection and HFR calculation.
 *   The image is preprocessed, noise is removed, and contours are extracted
 * from a thresholded map. Each contour is validated, its HFR computed, and
 * optionally annotated.
 *   @param img The input astronomical image.
 *   @param if_removehotpixel Flag indicating whether hot pixel removal should
 * be applied.
 *   @param if_noiseremoval Flag indicating whether Gaussian noise removal
 * should be applied.
 *   @param do_star_mark Flag to annotate detected stars on the output image.
 *   @param down_sample_mean_std Flag to control downsampling during mean and
 * std calculation.
 *   @param mark_img Image used for marking the detected stars (updated within
 * the function).
 *   @return A tuple containing:
 *           - The image with marked stars (if do_star_mark is true).
 *           - The number of stars detected.
 *           - The average HFR computed.
 *           - A JSON object with additional area statistics (max, min,
 * average).
 */
#include "HFR.hpp"

#include "spdlog/sinks/basic_file_sink.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/imgproc.hpp>

#include <vector>

using json = nlohmann::json;
using namespace std;
using namespace cv;

namespace {
std::shared_ptr<spdlog::logger> hfrLogger =
    spdlog::basic_logger_mt("HFRLogger", "logs/hfr.log");
} // namespace

constexpr double MIN_LONG_RATIO = 1.5;

auto checkElongated(int width, int height) -> bool {
  hfrLogger->info("Checking elongation for width: {}, height: {}", width,
                  height);
  double ratio = width > height ? static_cast<double>(width) / height
                                : static_cast<double>(height) / width;
  bool elongated = ratio > MIN_LONG_RATIO;
  hfrLogger->info("Elongated: {}", elongated);
  return elongated;
}

auto defineNarrowRadius(int minArea, double maxArea, double area, double scale)
    -> std::tuple<int, std::vector<int>, std::vector<double>> {
  hfrLogger->info("Defining narrow radius with minArea: {}, maxArea: {}, area: "
                  "{}, scale: {}",
                  minArea, maxArea, area, scale);
  std::vector<int> checklist;
  std::vector<double> thresholdList;
  int checkNum = 0;

  constexpr int AREA_THRESHOLD_1 = 500;
  constexpr int AREA_THRESHOLD_2 = 1000;
  constexpr double THRESHOLD_1 = 0.5;
  constexpr double THRESHOLD_2 = 0.65;
  constexpr double THRESHOLD_3 = 0.75;

  if (minArea <= area && area <= AREA_THRESHOLD_1 * scale) {
    checkNum = 2;
    checklist = {1, 2};
    thresholdList = {THRESHOLD_1, THRESHOLD_2};
  } else if (AREA_THRESHOLD_1 * scale < area &&
             area <= AREA_THRESHOLD_2 * scale) {
    checkNum = 3;
    checklist = {2, 3, 4};
    thresholdList = {THRESHOLD_1, THRESHOLD_2, THRESHOLD_3};
  } else if (AREA_THRESHOLD_2 * scale < area && area <= maxArea) {
    checkNum = 3;
    checklist = {2, 3, 4};
    thresholdList = {THRESHOLD_1, THRESHOLD_2, THRESHOLD_3};
  } else {
    checkNum = 0;
    checklist = {};
    thresholdList = {};
    hfrLogger->warn("Area {} is out of defined thresholds.", area);
  }
  hfrLogger->info("defineNarrowRadius result - checkNum: {}, checklist size: "
                  "{}, thresholdList size: {}",
                  checkNum, checklist.size(), thresholdList.size());
  return {checkNum, checklist, thresholdList};
}

// 优化 checkWhitePixel 函数，添加边界检查优化
inline auto checkWhitePixel(const cv::Mat &rect_contour, int xCoord, int yCoord)
    -> int {
  if (xCoord >= 0 && xCoord < rect_contour.cols && yCoord >= 0 &&
      yCoord < rect_contour.rows) {
    return rect_contour.at<uint16_t>(yCoord, xCoord, 0) > 0;
  }
  return 0;
}

auto eightSymmetryCircleCheck(const cv::Mat &rect_contour,
                              const cv::Point &center, int xCoord, int yCoord)
    -> int {
  hfrLogger->info(
      "Performing EightSymmetryCircleCheck with xCoord: {}, yCoord: {}", xCoord,
      yCoord);
  int whitePixelCount = 0;
  whitePixelCount +=
      checkWhitePixel(rect_contour, center.x + xCoord, center.y + yCoord);
  whitePixelCount +=
      checkWhitePixel(rect_contour, center.x - xCoord, center.y + yCoord);
  whitePixelCount +=
      checkWhitePixel(rect_contour, center.x + xCoord, center.y - yCoord);
  whitePixelCount +=
      checkWhitePixel(rect_contour, center.x - xCoord, center.y - yCoord);
  whitePixelCount +=
      checkWhitePixel(rect_contour, center.x + yCoord, center.y + xCoord);
  whitePixelCount +=
      checkWhitePixel(rect_contour, center.x + yCoord, center.y - xCoord);
  whitePixelCount +=
      checkWhitePixel(rect_contour, center.x - yCoord, center.y + xCoord);
  whitePixelCount +=
      checkWhitePixel(rect_contour, center.x - yCoord, center.y - xCoord);
  hfrLogger->info("White pixel count after symmetry check: {}",
                  whitePixelCount);
  return whitePixelCount;
}

auto fourSymmetryCircleCheck(const cv::Mat &rect_contour,
                             const cv::Point &center, float radius) -> int {
  hfrLogger->info("Performing FourSymmetryCircleCheck with radius: {}", radius);
  int whitePixelCount = 0;
  whitePixelCount += checkWhitePixel(rect_contour, center.x,
                                     center.y + static_cast<int>(radius));
  whitePixelCount += checkWhitePixel(rect_contour, center.x,
                                     center.y - static_cast<int>(radius));
  whitePixelCount += checkWhitePixel(
      rect_contour, center.x - static_cast<int>(radius), center.y);
  whitePixelCount += checkWhitePixel(
      rect_contour, center.x + static_cast<int>(radius), center.y);
  hfrLogger->info("White pixel count after four symmetry check: {}",
                  whitePixelCount);
  return whitePixelCount;
}

auto checkBresenhamCircle(const cv::Mat &rect_contour, float radius,
                          float pixelRatio, bool ifDebug) -> bool {
  hfrLogger->info("Starting BresenhamCircleCheck with radius: {}, pixelRatio: "
                  "{}, ifDebug: {}",
                  radius, pixelRatio, ifDebug);
  cv::Mat rectContourRgb;
  if (ifDebug) {
    cv::cvtColor(rect_contour, rectContourRgb, cv::COLOR_GRAY2BGR);
    hfrLogger->info("Converted rect_contour to RGB for debugging.");
  }

  int totalPixelCount = 0;
  int whitePixelCount = 0;

  cv::Size shape = rect_contour.size();
  cv::Point center(shape.width / 2, shape.height / 2);

  int p = 1 - static_cast<int>(radius);
  int xCoord = 0;
  int yCoord = static_cast<int>(radius);
  whitePixelCount += fourSymmetryCircleCheck(rect_contour, center, radius);
  totalPixelCount += 4;

  while (xCoord <= yCoord) {
    xCoord += 1;
    if (p < 0) {
      p += 2 * xCoord + 1;
    } else {
      yCoord -= 1;
      p += 2 * (xCoord - yCoord) + 1;
    }

    if (ifDebug) {
      // Future implementation for debugging can be added here
      hfrLogger->info("Debug mode: xCoord = {}, yCoord = {}", xCoord, yCoord);
    } else {
      whitePixelCount +=
          eightSymmetryCircleCheck(rect_contour, center, xCoord, yCoord);
    }

    totalPixelCount += 8;
  }

  float ratio = static_cast<float>(whitePixelCount) / totalPixelCount;
  hfrLogger->info("BresenhamCircleCheck ratio: {}", ratio);

  bool result = ratio > pixelRatio;
  hfrLogger->info("BresenhamCircleCheck result: {}", result);
  return result;
}

// 优化 calcHfr 函数，使用 SIMD 和多线程
auto calcHfr(const cv::Mat &inImage, float radius) -> double {
  try {
    cv::Mat img;
    inImage.convertTo(img, CV_32F);

    // 使用 OpenCV 优化的函数
    cv::Scalar meanVal = cv::mean(img);
    img -= meanVal[0];
    cv::max(img, 0.0F, img);

    const int centerX = std::ceil(img.cols / 2.0);
    const int centerY = std::ceil(img.rows / 2.0);
    constexpr double K_MAGIC_NUMBER = 1.2;
    const float max_radius = radius * K_MAGIC_NUMBER;

    // 预计算半径查找表以避免重复计算
    std::vector<float> radius_lut(img.rows * img.cols);
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < img.rows; ++i) {
      for (int j = 0; j < img.cols; ++j) {
        radius_lut[i * img.cols + j] = std::sqrt((i - centerY) * (i - centerY) +
                                                 (j - centerX) * (j - centerX));
      }
    }

    double sum = 0.0;
    double sumDist = 0.0;

    // 使用 OpenCV 优化的访问方式
    const float *img_data = img.ptr<float>();
#pragma omp parallel for reduction(+ : sum, sumDist) schedule(static)
    for (int idx = 0; idx < img.rows * img.cols; ++idx) {
      const float dist = radius_lut[idx];
      if (dist <= max_radius) {
        const float val = img_data[idx];
        sum += val;
        sumDist += val * dist;
      }
    }

    return sum <= 0 ? std::sqrt(2.0) * max_radius : sumDist / sum;
  } catch (const std::exception &e) {
    hfrLogger->error("Exception in calcHfr: {}", e.what());
    throw;
  }
}

auto caldim(const cv::Mat &img) -> bool {
  try {
    hfrLogger->info("Performing caldim check.");
    cv::Mat gray;
    if (img.channels() == 3) {
      cvtColor(img, gray, COLOR_BGR2GRAY);
    } else {
      gray = img;
    }

    double minVal, maxVal;
    minMaxLoc(gray, &minVal, &maxVal);

    double threshold = minVal + (maxVal - minVal) * 0.5;
    cv::Mat binary;
    cv::threshold(gray, binary, threshold, 255, THRESH_BINARY);

    int nonZeroCount = countNonZero(binary);
    double nonZeroRatio =
        static_cast<double>(nonZeroCount) / (binary.rows * binary.cols);

    hfrLogger->info("caldim check: non-zero ratio = {}", nonZeroRatio);

    constexpr double K_NON_ZERO_RATIO_THRESHOLD = 0.1;
    return nonZeroRatio < K_NON_ZERO_RATIO_THRESHOLD;
  } catch (const std::exception &e) {
    hfrLogger->error("Exception in caldim: {}", e.what());
    throw;
  }
}

auto preprocessImage(const Mat &img, Mat &grayimg, Mat &rgbImg, Mat &mark_img)
    -> void {
  try {
    if (img.channels() == 3) {
      cvtColor(img, grayimg, COLOR_BGR2GRAY);
      rgbImg = img;
      hfrLogger->info("Converted BGR to grayscale.");
    } else {
      grayimg = img;
      cvtColor(grayimg, rgbImg, COLOR_GRAY2BGR);
      hfrLogger->info("Converted grayscale to RGB.");
    }

    if (mark_img.empty()) {
      mark_img = rgbImg.clone();
      hfrLogger->info("Initialized mark_img with cloned RGB image.");
    } else if (mark_img.channels() == 1) {
      cvtColor(mark_img, mark_img, COLOR_GRAY2BGR);
      hfrLogger->info("Converted single-channel mark_img to BGR.");
    }
  } catch (const std::exception &e) {
    hfrLogger->error("Exception in preprocessImage: {}", e.what());
    throw;
  }
}

// 优化 removeNoise 函数，添加并行处理
auto removeNoise(Mat &map, bool if_removehotpixel, bool if_noiseremoval)
    -> void {
  try {
    if (if_removehotpixel) {
      cv::parallel_for_(cv::Range(0, map.rows), [&](const cv::Range &range) {
        for (int i = range.start; i < range.end; i++) {
          // 实现并行中值滤波
          medianBlur(map.row(i), map.row(i), 3);
        }
      });
    }

    if (if_noiseremoval) {
      // 使用分离的高斯核进行优化
      cv::Mat kernel_x = cv::getGaussianKernel(3, 1.0);
      cv::Mat kernel_y = cv::getGaussianKernel(3, 1.0);
      cv::sepFilter2D(map, map, -1, kernel_x, kernel_y);
    }
  } catch (const std::exception &e) {
    hfrLogger->error("Exception in removeNoise: {}", e.what());
    throw;
  }
}

auto calculateMeanAndStd(const Mat &map, bool down_sample_mean_std,
                         double &medianVal, double &stdDev) -> void {
  try {
    if (!down_sample_mean_std) {
      medianVal = mean(map)[0];
      Scalar meanVal, stddev;
      meanStdDev(map, meanVal, stddev);
      stdDev = stddev[0];
      hfrLogger->info("Calculated mean and std without downsampling.");
    } else {
      hfrLogger->info("Calculating mean and std with downsampling.");
      vector<uchar> bufferValue;
      if (map.isContinuous()) {
        bufferValue.assign(map.datastart, map.dataend);
      } else {
        for (int i = 0; i < map.rows; ++i) {
          bufferValue.insert(bufferValue.end(), map.ptr<uchar>(i),
                             map.ptr<uchar>(i) + map.cols);
        }
      }
      constexpr int K_MAX_SAMPLES = 500000;
      int sampleBy = 1;
      if (map.rows * map.cols > K_MAX_SAMPLES) {
        sampleBy = map.rows * map.cols / K_MAX_SAMPLES;
        hfrLogger->info("Downsampling with step: {}", sampleBy);
      }

      vector<uchar> sampleValue;
      sampleValue.reserve(bufferValue.size() / sampleBy + 1);
      for (size_t i = 0; i < bufferValue.size(); i += sampleBy) {
        sampleValue.push_back(bufferValue[i]);
      }
      medianVal = std::accumulate(sampleValue.begin(), sampleValue.end(), 0.0) /
                  sampleValue.size();
      double sum = 0;
#pragma omp parallel for reduction(+ : sum)
      for (size_t i = 0; i < sampleValue.size(); ++i) {
        double diff = sampleValue[i] - medianVal;
        sum += diff * diff;
      }
      stdDev = sqrt(sum / sampleValue.size());
      hfrLogger->info("Calculated downsampled mean: {} and std: {}", medianVal,
                      stdDev);
    }
  } catch (const std::exception &e) {
    hfrLogger->error("Exception in calculateMeanAndStd: {}", e.what());
    throw;
  }
}

auto processContours(const Mat &grayimg, const Mat &rgbImg, Mat &mark_img,
                     const vector<vector<Point>> &contours, double threshold,
                     bool do_star_mark)
    -> tuple<int, double, vector<double>, vector<double>> {
  try {
    constexpr double K_STAND_SIZE = 1552;
    Size imgShps = grayimg.size();
    double sclsize = max(imgShps.width, imgShps.height);
    double maximumArea = 1500 * (sclsize / K_STAND_SIZE);
    double minimumArea = max(1.0, ceil(sclsize / K_STAND_SIZE));
    double bshScale = sclsize / 2048;
    vector<double> hfrList;
    vector<double> arelist;
    int starnum = 0;

    for (size_t i = 0; i < contours.size(); i++) {
      double area = contourArea(contours[i]);
      if (area >= minimumArea && area < maximumArea) {
        Point2f center;
        float radius;
        minEnclosingCircle(contours[i], center, radius);

        Rect boundingBox = boundingRect(contours[i]);
        Point rectCenter(boundingBox.x + boundingBox.width / 2,
                         boundingBox.y + boundingBox.height / 2);

        if (checkElongated(boundingBox.width, boundingBox.height)) {
          hfrLogger->info("Contour {} is elongated. Skipping.", i);
          continue;
        }

        int bshNum;
        vector<int> bshList;
        vector<double> bshThresList;
        tie(bshNum, bshList, bshThresList) = defineNarrowRadius(
            static_cast<int>(minimumArea), maximumArea, area, bshScale);

        bool bshCheck = false;
        for (int bshIndex = 0; bshIndex < bshNum; bshIndex++) {
          int narrowRadius = bshList[bshIndex];
          double pixelthresh = bshThresList[bshIndex];
          Rect expandedRect(boundingBox.x - 5, boundingBox.y - 5,
                            boundingBox.width + 10, boundingBox.height + 10);

          if (expandedRect.x < 0 || expandedRect.y < 0 ||
              expandedRect.x + expandedRect.width >= grayimg.cols ||
              expandedRect.y + expandedRect.height >= grayimg.rows) {
            hfrLogger->warn(
                "Expanded rectangle out of bounds. Skipping contour {}", i);
            continue;
          }

          Mat rectThresExpand = grayimg(expandedRect);

          if (checkBresenhamCircle(rectThresExpand, radius - narrowRadius,
                                   pixelthresh, false)) {
            bshCheck = true;
            break;
          }
        }
        if (!bshCheck) {
          hfrLogger->info("Contour {} failed BresenHam check. Skipping.", i);
          continue;
        }

        Rect starRegion(static_cast<int>(center.x - radius),
                        static_cast<int>(center.y - radius),
                        static_cast<int>(2 * radius),
                        static_cast<int>(2 * radius));
        if (starRegion.x < 0 || starRegion.y < 0 ||
            starRegion.x + starRegion.width >= grayimg.cols ||
            starRegion.y + starRegion.height >= grayimg.rows) {
          hfrLogger->warn("Star region out of bounds for contour {}. "
                          "Skipping.",
                          i);
          continue;
        }

        Mat rectExpand = rgbImg(starRegion);

        if (caldim(rectExpand)) {
          hfrLogger->info("Contour {} failed caldim check. Skipping.", i);
          continue;
        }

        double hfr = calcHfr(grayimg(starRegion), radius);
        constexpr double K_HFR_THRESHOLD = 0.05;
        if (hfr < K_HFR_THRESHOLD) {
          hfrLogger->info("HFR below threshold for contour {}. Skipping.", i);
          continue;
        }
        hfrList.push_back(hfr);
        starnum++;
        arelist.push_back(area);

        if (do_star_mark) {
          circle(mark_img, rectCenter, static_cast<int>(radius) + 5,
                 Scalar(0, 255, 0), 1);
          putText(mark_img, to_string(hfr), rectCenter, FONT_HERSHEY_SIMPLEX,
                  1.0, Scalar(0, 255, 0), 1, LINE_AA);
          hfrLogger->info("Marked star at contour {} with HFR: {}", i, hfr);
        }
      }
    }

    double avgHfr =
        hfrList.empty()
            ? 0.0
            : accumulate(hfrList.begin(), hfrList.end(), 0.0) / hfrList.size();
    return make_tuple(starnum, avgHfr, hfrList, arelist);
  } catch (const std::exception &e) {
    hfrLogger->error("Exception in processContours: {}", e.what());
    throw;
  }
}

auto starDetectAndHfr(const Mat &img, bool if_removehotpixel,
                      bool if_noiseremoval, bool do_star_mark,
                      bool down_sample_mean_std, Mat mark_img)
    -> tuple<Mat, int, double, json> {
  try {
    hfrLogger->info("Starting StarDetectAndHfr processing.");
    Mat grayimg, rgbImg;
    preprocessImage(img, grayimg, rgbImg, mark_img);

    Mat map = grayimg.clone();
    removeNoise(map, if_removehotpixel, if_noiseremoval);

    double medianVal, stdDev;
    calculateMeanAndStd(map, down_sample_mean_std, medianVal, stdDev);

    double threshold = medianVal + 3 * stdDev;
    hfrLogger->info("Applying threshold: {}", threshold);
    Mat thresMap;
    cv::threshold(map, thresMap, threshold, 255, THRESH_BINARY);

    Mat closekernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(thresMap, thresMap, MORPH_OPEN, closekernel);
    hfrLogger->info("Performed morphological opening.");

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(thresMap, contours, hierarchy, RETR_EXTERNAL,
                 CHAIN_APPROX_NONE);
    hfrLogger->info("Found {} contours.", contours.size());

    int starnum;
    double avghfr;
    vector<double> hfrList;
    vector<double> arelist;
    tie(starnum, avghfr, hfrList, arelist) = processContours(
        grayimg, rgbImg, mark_img, contours, threshold, do_star_mark);

    double maxarea =
        arelist.empty() ? -1 : *max_element(arelist.begin(), arelist.end());
    double minarea =
        arelist.empty() ? -1 : *min_element(arelist.begin(), arelist.end());
    double avgarea =
        arelist.empty()
            ? -1
            : accumulate(arelist.begin(), arelist.end(), 0.0) / arelist.size();

    hfrLogger->info("Processed {} stars.", starnum);
    hfrLogger->info("Average HFR: {}, Max Area: {}, Min Area: {}, Avg Area: "
                    "{}",
                    avghfr, maxarea, minarea, avgarea);

    json result = {{"max", maxarea}, {"min", minarea}, {"average", avgarea}};

    return make_tuple(mark_img, starnum, avghfr, result);
  } catch (const std::exception &e) {
    hfrLogger->error("Exception in StarDetectAndHfr: {}", e.what());
    throw;
  }
}