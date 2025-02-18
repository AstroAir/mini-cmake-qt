#pragma once
#include <opencv2/core.hpp>

class VideoFilters {
public:
  // 色彩调整滤镜
  static void adjustHSV(cv::Mat &frame, float hue = 0.0f,
                        float saturation = 1.0f, float value = 1.0f);

  static void adjustColorBalance(cv::Mat &frame, float red = 1.0f,
                                 float green = 1.0f, float blue = 1.0f);

  // 艺术效果滤镜
  static void applySketch(cv::Mat &frame);
  static void applyCartoon(cv::Mat &frame);
  static void applyOilPainting(cv::Mat &frame, int size = 5);
  static void applyDream(cv::Mat &frame, float strength = 0.5f);

  // LUT滤镜
  static void applyLUT(cv::Mat &frame, const cv::Mat &lut);
  static cv::Mat createLUT(const std::string &style);

  // 新增高级滤镜
  static void applyNoise(cv::Mat &frame, float amount = 0.1f);
  static void applyTiltShift(cv::Mat &frame, int focusPosition = -1);
  static void applyGlitch(cv::Mat &frame, float intensity = 0.5f);
  static void applyColorGrading(cv::Mat &frame, const cv::Mat &grading);

  // 优化选项
  struct ProcessOptions {
    bool useGPU = false;
    int threads = 1;
    bool preprocessFrame = true;
  };

  static void setProcessOptions(const ProcessOptions &options);

private:
  static ProcessOptions processOptions;
  static void preprocessFrame(cv::Mat &frame);
  static void bilateralStyleFilter(cv::Mat &frame, int iterations = 2);
};
