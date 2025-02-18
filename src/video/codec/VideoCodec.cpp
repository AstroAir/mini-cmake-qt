#include "VideoCodec.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/videoio.hpp>


std::vector<VideoCodec::CodecInfo> VideoCodec::getAvailableCodecs() {
  std::vector<CodecInfo> codecs;

  CodecInfo h264 = {"H264", "H.264/AVC codec", true, {".mp4", ".mkv"}};

  CodecInfo xvid = {"XVID", "MPEG-4 Part 2 codec", false, {".avi"}};

  CodecInfo vp9 = {"VP9", "Google VP9 codec", true, {".webm"}};

  codecs.push_back(h264);
  codecs.push_back(xvid);
  codecs.push_back(vp9);

  return codecs;
}

bool VideoCodec::isCodecSupported(const std::string &codec) {
  auto codecs = getAvailableCodecs();
  return std::any_of(codecs.begin(), codecs.end(),
                     [&](const CodecInfo &info) { return info.name == codec; });
}

bool VideoCodec::hasHardwareAcceleration(const std::string &codec) {
  if (!checkHWSupport())
    return false;

  // 检查特定编解码器的硬件加速支持
  if (codec == "H264" || codec == "HEVC") {
    return true;
  }
  return false;
}

VideoCodec::EncodingParams
VideoCodec::getOptimalParams(const std::string &codec,
                             const cv::Size &frameSize, double fps) {

  EncodingParams params;

  // 根据分辨率和帧率计算合适的比特率
  int pixelCount = frameSize.width * frameSize.height;

  if (codec == "H264") {
    params.bitrate = static_cast<int>(pixelCount * fps * 0.1);
    params.gopSize = static_cast<int>(fps * 2);
    params.threads = cv::getNumberOfCPUs();
    params.useHWAccel = hasHardwareAcceleration(codec);
  } else if (codec == "XVID") {
    params.bitrate = static_cast<int>(pixelCount * fps * 0.2);
    params.gopSize = static_cast<int>(fps);
    params.threads = cv::getNumberOfCPUs();
    params.useHWAccel = false;
  }

  return params;
}

// ... 继续实现其他方法 ...
