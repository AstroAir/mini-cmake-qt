#pragma once

#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

class VideoIO {
public:
  struct VideoInfo {
    int width;
    int height;
    double fps;
    int fourcc;
    int frameCount;
    bool isColor;
    std::string codecName;
  };

  // 视频读取相关
  static cv::VideoCapture openVideo(const std::string &filename);
  static cv::VideoCapture openCamera(int deviceId = 0);
  static VideoInfo getVideoInfo(const cv::VideoCapture &cap);
  static cv::Mat readFrame(cv::VideoCapture &cap);
  static std::vector<cv::Mat> extractFrames(const std::string &filename,
                                            int maxFrames = -1);

  // 视频写入相关
  static cv::VideoWriter createVideo(const std::string &filename, int fourcc,
                                     double fps, const cv::Size &frameSize,
                                     bool isColor = true);
  static bool writeFrame(cv::VideoWriter &writer, const cv::Mat &frame);
  static bool saveFramesToVideo(const std::string &filename,
                                const std::vector<cv::Mat> &frames,
                                double fps = 30.0);

  // 编解码器相关
  static int getFourCC(const std::string &codec);
  static std::string getFourCCString(int fourcc);
  static std::vector<std::string> getAvailableCodecs();

  // 实用工具
  static bool isVideoFile(const std::string &filename);
  static std::string getDefaultCodec();
  static double getOptimalFPS(const std::vector<cv::Mat> &frames);

  // 高级视频处理功能
  static bool resizeVideo(const std::string &input, const std::string &output,
                          const cv::Size &newSize);
  static bool changeVideoSpeed(const std::string &input,
                               const std::string &output, double speedFactor);
  static bool extractKeyFrames(const std::string &filename,
                               std::vector<cv::Mat> &keyFrames,
                               double threshold = 0.1);

  // 性能优化相关
  static void enableParallelProcessing(bool enable = true);
  static void setBufferSize(size_t size);
  static void enableGPUAcceleration(bool enable = true);

  // 批处理支持
  static bool batchProcessFrames(
      const std::vector<cv::Mat> &input, std::vector<cv::Mat> &output,
      const std::function<void(cv::Mat &)> &processor, bool useParallel = true);

private:
  static bool isGPUAvailable();
  static void initializeGPU();
  static size_t m_bufferSize;
  static bool m_useGPU;
  static bool m_enableParallel;
};
