#include "VideoAnalyzer.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>

std::vector<VideoAnalyzer::SceneInfo>
VideoAnalyzer::detectScenes(const std::string &videoPath, double threshold) {
  std::vector<SceneInfo> scenes;
  cv::VideoCapture cap(videoPath);

  if (!cap.isOpened()) {
    spdlog::error("无法打开视频文件: {}", videoPath);
    return scenes;
  }

  cv::Mat prevFrame, currFrame;
  SceneInfo currentScene = {0, 0, 0.0, ""};
  int frameIndex = 0;

  while (cap.read(currFrame)) {
    if (!prevFrame.empty()) {
      double diff = computeFrameDifference(prevFrame, currFrame);

      if (diff > threshold) {
        // 检测到场景变化
        currentScene.endFrame = frameIndex - 1;
        currentScene.confidence = diff;
        scenes.push_back(currentScene);

        // 开始新场景
        currentScene.startFrame = frameIndex;
        currentScene.description = "Scene " + std::to_string(scenes.size() + 1);
      }
    } else {
      // 第一帧
      currentScene.startFrame = 0;
      currentScene.description = "Scene 1";
    }

    currFrame.copyTo(prevFrame);
    frameIndex++;
  }

  // 处理最后一个场景
  if (frameIndex > 0) {
    currentScene.endFrame = frameIndex - 1;
    scenes.push_back(currentScene);
  }

  return scenes;
}

std::vector<VideoAnalyzer::MotionInfo>
VideoAnalyzer::analyzeMotion(const cv::Mat &frame1, const cv::Mat &frame2) {
  std::vector<MotionInfo> motionInfo;

  // 转换为灰度图
  cv::Mat gray1, gray2;
  cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

  // 计算光流
  cv::Mat flow;
  cv::calcOpticalFlowFarneback(gray1, gray2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

  // 将图像分成网格进行分析
  const int gridSize = 32;
  for (int y = 0; y < flow.rows; y += gridSize) {
    for (int x = 0; x < flow.cols; x += gridSize) {
      MotionInfo info;
      info.region = cv::Rect(x, y, std::min(gridSize, flow.cols - x),
                             std::min(gridSize, flow.rows - y));

      // 计算区域内的平均运动
      cv::Mat flowROI = flow(info.region);
      cv::Point2f avgFlow(0, 0);
      float totalMagnitude = 0;
      int count = 0;

      for (int i = 0; i < flowROI.rows; i++) {
        for (int j = 0; j < flowROI.cols; j++) {
          const cv::Point2f &fxy = flowROI.at<cv::Point2f>(i, j);
          avgFlow += fxy;
          totalMagnitude += std::sqrt(fxy.x * fxy.x + fxy.y * fxy.y);
          count++;
        }
      }

      if (count > 0) {
        avgFlow *= 1.0f / count;
        info.direction = avgFlow;
        info.magnitude = totalMagnitude / count;

        // 只记录显著的运动
        if (info.magnitude > 0.5) {
          motionInfo.push_back(info);
        }
      }
    }
  }

  return motionInfo;
}

bool VideoAnalyzer::detectBlur(const cv::Mat &frame, double &bluriness) {
  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  cv::Mat laplacian;
  cv::Laplacian(gray, laplacian, CV_64F);

  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);

  // 使用拉普拉斯算子的方差作为模糊度度量
  bluriness = stddev[0] * stddev[0];

  // 返回是否模糊的判断结果
  return bluriness < 100.0;
}

bool VideoAnalyzer::analyzeExposure(const cv::Mat &frame, double &exposure) {
  cv::Mat hsv;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

  // 分离通道，获取V通道
  std::vector<cv::Mat> channels;
  cv::split(hsv, channels);

  // 计算平均亮度
  cv::Scalar meanVal = cv::mean(channels[2]);
  exposure = meanVal[0];

  // 判断是否曝光正常 (0-255范围内的合理值)
  return exposure >= 85 && exposure <= 170;
}

cv::Rect VideoAnalyzer::detectMainSubject(const cv::Mat &frame) {
  // 使用Saliency检测主要对象
  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  // 使用FAST特征点检测
  std::vector<cv::KeyPoint> keypoints;
  cv::FAST(gray, keypoints, 20, true);

  if (keypoints.empty()) {
    return cv::Rect();
  }

  // 计算特征点的边界框
  float minX = frame.cols, minY = frame.rows;
  float maxX = 0, maxY = 0;

  for (const auto &kp : keypoints) {
    minX = std::min(minX, kp.pt.x);
    minY = std::min(minY, kp.pt.y);
    maxX = std::max(maxX, kp.pt.x);
    maxY = std::max(maxY, kp.pt.y);
  }

  // 扩大边界框
  int padding = 20;
  minX = std::max(0.0f, minX - padding);
  minY = std::max(0.0f, minY - padding);
  maxX = std::min(static_cast<float>(frame.cols), maxX + padding);
  maxY = std::min(static_cast<float>(frame.rows), maxY + padding);

  return cv::Rect(minX, minY, maxX - minX, maxY - minY);
}

double VideoAnalyzer::computeFrameDifference(const cv::Mat &frame1,
                                             const cv::Mat &frame2) {
  cv::Mat diff;
  cv::absdiff(frame1, frame2, diff);

  cv::Mat grayDiff;
  cv::cvtColor(diff, grayDiff, cv::COLOR_BGR2GRAY);

  cv::Scalar meanDiff = cv::mean(grayDiff);
  return meanDiff[0] / 255.0;
}
