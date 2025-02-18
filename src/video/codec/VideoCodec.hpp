#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>


class VideoCodec {
public:
  struct CodecInfo {
    std::string name;
    std::string description;
    bool hasHWAccel;
    std::vector<std::string> supportedExtensions;
  };

  struct EncodingParams {
    int bitrate;
    int gopSize;
    int threads;
    bool useHWAccel;
  };

  // 编解码器管理
  static std::vector<CodecInfo> getAvailableCodecs();
  static bool isCodecSupported(const std::string &codec);
  static bool hasHardwareAcceleration(const std::string &codec);

  // 编码参数优化
  static EncodingParams getOptimalParams(const std::string &codec,
                                         const cv::Size &frameSize, double fps);

  // 转码功能
  static bool transcodeVideo(const std::string &input,
                             const std::string &output,
                             const std::string &targetCodec,
                             const EncodingParams &params = EncodingParams());

private:
  static void initHWAccel();
  static bool checkHWSupport();
};
