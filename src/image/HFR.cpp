#include "HFR.hpp"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/imgproc.hpp>

#include <vector>

using json = nlohmann::json;
using namespace std;
using namespace cv;

constexpr double MIN_LONG_RATIO = 1.5;

auto checkElongated(int width, int height) -> bool {
  spdlog::info("Checking elongation for width: {}, height: {}", width, height);
  double ratio = width > height ? static_cast<double>(width) / height
                                : static_cast<double>(height) / width;
  bool elongated = ratio > MIN_LONG_RATIO;
  spdlog::info("Elongated: {}", elongated);
  return elongated;
}

auto defineNarrowRadius(int minArea, double maxArea, double area, double scale)
    -> std::tuple<int, std::vector<int>, std::vector<double>> {
  spdlog::info("Defining narrow radius with minArea: {}, maxArea: {}, area: "
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
    spdlog::warn("Area {} is out of defined thresholds.", area);
  }
  spdlog::info("defineNarrowRadius result - checkNum: {}, checklist size: {}, "
               "thresholdList size: {}",
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
  spdlog::info(
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
  spdlog::info("White pixel count after symmetry check: {}", whitePixelCount);
  return whitePixelCount;
}

auto fourSymmetryCircleCheck(const cv::Mat &rect_contour,
                             const cv::Point &center, float radius) -> int {
  spdlog::info("Performing FourSymmetryCircleCheck with radius: {}", radius);
  int whitePixelCount = 0;
  whitePixelCount += checkWhitePixel(rect_contour, center.x,
                                     center.y + static_cast<int>(radius));
  whitePixelCount += checkWhitePixel(rect_contour, center.x,
                                     center.y - static_cast<int>(radius));
  whitePixelCount += checkWhitePixel(
      rect_contour, center.x - static_cast<int>(radius), center.y);
  whitePixelCount += checkWhitePixel(
      rect_contour, center.x + static_cast<int>(radius), center.y);
  spdlog::info("White pixel count after four symmetry check: {}",
               whitePixelCount);
  return whitePixelCount;
}

auto checkBresenhamCircle(const cv::Mat &rect_contour, float radius,
                          float pixelRatio, bool ifDebug) -> bool {
  spdlog::info("Starting BresenhamCircleCheck with radius: {}, pixelRatio: {}, "
               "ifDebug: {}",
               radius, pixelRatio, ifDebug);
  cv::Mat rectContourRgb;
  if (ifDebug) {
    cv::cvtColor(rect_contour, rectContourRgb, cv::COLOR_GRAY2BGR);
    spdlog::info("Converted rect_contour to RGB for debugging.");
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
      spdlog::info("Debug mode: xCoord = {}, yCoord = {}", xCoord, yCoord);
    } else {
      whitePixelCount +=
          eightSymmetryCircleCheck(rect_contour, center, xCoord, yCoord);
    }

    totalPixelCount += 8;
  }

  float ratio = static_cast<float>(whitePixelCount) / totalPixelCount;
  spdlog::info("BresenhamCircleCheck ratio: {}", ratio);

  bool result = ratio > pixelRatio;
  spdlog::info("BresenhamCircleCheck result: {}", result);
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
    spdlog::error("Exception in calcHfr: {}", e.what());
    throw;
  }
}

auto caldim(const cv::Mat &img) -> bool {
  try {
    spdlog::info("Performing caldim check.");
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

    spdlog::info("caldim check: non-zero ratio = {}", nonZeroRatio);

    constexpr double K_NON_ZERO_RATIO_THRESHOLD = 0.1;
    return nonZeroRatio < K_NON_ZERO_RATIO_THRESHOLD;
  } catch (const std::exception &e) {
    spdlog::error("Exception in caldim: {}", e.what());
    throw;
  }
}

auto preprocessImage(const Mat &img, Mat &grayimg, Mat &rgbImg, Mat &mark_img)
    -> void {
  try {
    if (img.channels() == 3) {
      cvtColor(img, grayimg, COLOR_BGR2GRAY);
      rgbImg = img;
      spdlog::info("Converted BGR to grayscale.");
    } else {
      grayimg = img;
      cvtColor(grayimg, rgbImg, COLOR_GRAY2BGR);
      spdlog::info("Converted grayscale to RGB.");
    }

    if (mark_img.empty()) {
      mark_img = rgbImg.clone();
      spdlog::info("Initialized mark_img with cloned RGB image.");
    } else if (mark_img.channels() == 1) {
      cvtColor(mark_img, mark_img, COLOR_GRAY2BGR);
      spdlog::info("Converted single-channel mark_img to BGR.");
    }
  } catch (const std::exception &e) {
    spdlog::error("Exception in preprocessImage: {}", e.what());
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
    spdlog::error("Exception in removeNoise: {}", e.what());
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
      spdlog::info("Calculated mean and std without downsampling.");
    } else {
      spdlog::info("Calculating mean and std with downsampling.");
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
        spdlog::info("Downsampling with step: {}", sampleBy);
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
      spdlog::info("Calculated downsampled mean: {} and std: {}", medianVal,
                   stdDev);
    }
  } catch (const std::exception &e) {
    spdlog::error("Exception in calculateMeanAndStd: {}", e.what());
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
          spdlog::info("Contour {} is elongated. Skipping.", i);
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
            spdlog::warn(
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
          spdlog::info("Contour {} failed BresenHam check. Skipping.", i);
          continue;
        }

        Rect starRegion(static_cast<int>(center.x - radius),
                        static_cast<int>(center.y - radius),
                        static_cast<int>(2 * radius),
                        static_cast<int>(2 * radius));
        if (starRegion.x < 0 || starRegion.y < 0 ||
            starRegion.x + starRegion.width >= grayimg.cols ||
            starRegion.y + starRegion.height >= grayimg.rows) {
          spdlog::warn("Star region out of bounds for contour {}. Skipping.",
                       i);
          continue;
        }

        Mat rectExpand = rgbImg(starRegion);

        if (caldim(rectExpand)) {
          spdlog::info("Contour {} failed caldim check. Skipping.", i);
          continue;
        }

        double hfr = calcHfr(grayimg(starRegion), radius);
        constexpr double K_HFR_THRESHOLD = 0.05;
        if (hfr < K_HFR_THRESHOLD) {
          spdlog::info("HFR below threshold for contour {}. Skipping.", i);
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
          spdlog::info("Marked star at contour {} with HFR: {}", i, hfr);
        }
      }
    }

    double avgHfr =
        hfrList.empty()
            ? 0.0
            : accumulate(hfrList.begin(), hfrList.end(), 0.0) / hfrList.size();
    return make_tuple(starnum, avgHfr, hfrList, arelist);
  } catch (const std::exception &e) {
    spdlog::error("Exception in processContours: {}", e.what());
    throw;
  }
}

auto starDetectAndHfr(const Mat &img, bool if_removehotpixel,
                      bool if_noiseremoval, bool do_star_mark,
                      bool down_sample_mean_std, Mat mark_img)
    -> tuple<Mat, int, double, json> {
  try {
    spdlog::info("Starting StarDetectAndHfr processing.");
    Mat grayimg, rgbImg;
    preprocessImage(img, grayimg, rgbImg, mark_img);

    Mat map = grayimg.clone();
    removeNoise(map, if_removehotpixel, if_noiseremoval);

    double medianVal, stdDev;
    calculateMeanAndStd(map, down_sample_mean_std, medianVal, stdDev);

    double threshold = medianVal + 3 * stdDev;
    spdlog::info("Applying threshold: {}", threshold);
    Mat thresMap;
    cv::threshold(map, thresMap, threshold, 255, THRESH_BINARY);

    Mat closekernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(thresMap, thresMap, MORPH_OPEN, closekernel);
    spdlog::info("Performed morphological opening.");

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(thresMap, contours, hierarchy, RETR_EXTERNAL,
                 CHAIN_APPROX_NONE);
    spdlog::info("Found {} contours.", contours.size());

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

    spdlog::info("Processed {} stars.", starnum);
    spdlog::info("Average HFR: {}, Max Area: {}, Min Area: {}, Avg Area: {}",
                 avghfr, maxarea, minarea, avgarea);

    json result = {{"max", maxarea}, {"min", minarea}, {"average", avgarea}};

    return make_tuple(mark_img, starnum, avghfr, result);
  } catch (const std::exception &e) {
    spdlog::error("Exception in StarDetectAndHfr: {}", e.what());
    throw;
  }
}