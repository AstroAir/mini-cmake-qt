#pragma once
#include <opencv2/core.hpp>
#include <vector>

class VideoAnalyzer {
public:
  struct SceneInfo {
    int startFrame;
    int endFrame;
    double confidence;
    std::string description;
  };

  struct MotionInfo {
    cv::Point2f direction;
    float magnitude;
    cv::Rect region;
  };

  // 场景分析
  static std::vector<SceneInfo> detectScenes(const std::string &videoPath,
                                             double threshold = 0.3);

  // 运动分析
  static std::vector<MotionInfo> analyzeMotion(const cv::Mat &frame1,
                                               const cv::Mat &frame2);

  // 内容分析
  static bool detectBlur(const cv::Mat &frame, double &bluriness);
  static bool analyzeExposure(const cv::Mat &frame, double &exposure);
  static cv::Rect detectMainSubject(const cv::Mat &frame);

  // 新增分析功能
  static double computeVideoQuality(const cv::Mat &frame);
  static bool detectFaces(const cv::Mat &frame, std::vector<cv::Rect> &faces);
  static void trackObjects(const cv::Mat &prevFrame, const cv::Mat &currFrame,
                          std::vector<cv::Point2f> &prevPoints,
                          std::vector<cv::Point2f> &currPoints);
                          
  // 性能优化选项
  struct AnalysisOptions {
    bool useGPU = false;
    int threads = 1;
    bool cacheResults = true;
  };
  
  static void setAnalysisOptions(const AnalysisOptions &options);

private:
  static double computeFrameDifference(const cv::Mat &frame1,
                                       const cv::Mat &frame2);
  static AnalysisOptions analysisOptions;
};
