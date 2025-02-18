#pragma once
#include <opencv2/core.hpp>
#include <vector>

class VideoEffects {
public:
  struct TransitionEffect {
    enum Type { FADE, DISSOLVE, WIPE, SLIDE };
    Type type;
    int durationFrames;
  };

  // 视频转场效果
  static bool applyTransition(const cv::Mat &frame1, const cv::Mat &frame2,
                              cv::Mat &output, const TransitionEffect &effect,
                              float progress);

  // 特效处理
  static void applySepia(cv::Mat &frame);
  static void applyVignette(cv::Mat &frame, float strength = 0.5f);
  static void applyBloom(cv::Mat &frame, float intensity = 1.0f);
  static void applySlowMotion(const std::vector<cv::Mat> &input,
                              std::vector<cv::Mat> &output,
                              float factor = 2.0f);

  // 画面分割效果
  static void splitScreen(const std::vector<cv::Mat> &inputs, cv::Mat &output,
                          int rows, int cols);

private:
  static void interpolateFrames(const cv::Mat &frame1, const cv::Mat &frame2,
                                std::vector<cv::Mat> &output, int numFrames);
};
