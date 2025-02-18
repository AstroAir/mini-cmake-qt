#include "VideoIO.hpp"
#include <chrono>
#include <filesystem>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

namespace {
std::shared_ptr<spdlog::logger> videoLogger =
    spdlog::basic_logger_mt("VideoIOLogger", "logs/video_io.log");

const std::vector<std::string> VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv",
                                                   ".mov", ".wmv", ".flv"};

bool handleException(const std::exception &e, const std::string &context) {
  videoLogger->error("{}: {}", context, e.what());
  return false;
}
} // namespace

// 静态成员初始化
size_t VideoIO::m_bufferSize = 1024 * 1024; // 1MB默认缓冲区
bool VideoIO::m_useGPU = false;
bool VideoIO::m_enableParallel = true;

cv::VideoCapture VideoIO::openVideo(const std::string &filename) {
  try {
    videoLogger->info("正在打开视频文件: {}", filename);

    if (!std::filesystem::exists(filename)) {
      videoLogger->error("视频文件不存在: {}", filename);
      return cv::VideoCapture();
    }

    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
      videoLogger->error("无法打开视频文件: {}", filename);
      return cv::VideoCapture();
    }

    VideoInfo info = getVideoInfo(cap);
    videoLogger->info("成功打开视频: {}x{} @{}fps, 总帧数: {}", info.width,
                      info.height, info.fps, info.frameCount);
    return cap;
  } catch (const std::exception &e) {
    handleException(e, "openVideo");
    return cv::VideoCapture();
  }
}

cv::VideoCapture VideoIO::openCamera(int deviceId) {
  try {
    videoLogger->info("正在打开摄像头设备: {}", deviceId);

    cv::VideoCapture cap(deviceId);
    if (!cap.isOpened()) {
      videoLogger->error("无法打开摄像头设备: {}", deviceId);
      return cv::VideoCapture();
    }

    // 设置常用的摄像头属性
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);

    VideoInfo info = getVideoInfo(cap);
    videoLogger->info("成功打开摄像头: {}x{} @{}fps", info.width, info.height,
                      info.fps);
    return cap;
  } catch (const std::exception &e) {
    handleException(e, "openCamera");
    return cv::VideoCapture();
  }
}

VideoIO::VideoInfo VideoIO::getVideoInfo(const cv::VideoCapture &cap) {
  VideoInfo info;
  info.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  info.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  info.fps = cap.get(cv::CAP_PROP_FPS);
  info.fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
  info.frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  info.isColor = true; // 默认为彩色
  info.codecName = getFourCCString(info.fourcc);
  return info;
}

cv::Mat VideoIO::readFrame(cv::VideoCapture &cap) {
  try {
    cv::Mat frame;
    if (!cap.read(frame)) {
      videoLogger->warn("读取帧失败或到达视频结尾");
      return cv::Mat();
    }
    return frame;
  } catch (const std::exception &e) {
    handleException(e, "readFrame");
    return cv::Mat();
  }
}

std::vector<cv::Mat> VideoIO::extractFrames(const std::string &filename,
                                            int maxFrames) {
  try {
    std::vector<cv::Mat> frames;
    auto cap = openVideo(filename);
    if (!cap.isOpened()) {
      return frames;
    }

    videoLogger->info("开始提取帧, 最大帧数: {}",
                      maxFrames < 0 ? "全部" : std::to_string(maxFrames));

    int frameCount = 0;
    auto startTime = std::chrono::steady_clock::now();

    while (true) {
      if (maxFrames > 0 && frameCount >= maxFrames) {
        break;
      }

      cv::Mat frame = readFrame(cap);
      if (frame.empty()) {
        break;
      }

      frames.push_back(frame.clone());
      frameCount++;

      if (frameCount % 100 == 0) {
        videoLogger->info("已提取 {} 帧", frameCount);
      }
    }

    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        endTime - startTime)
                        .count();

    videoLogger->info("帧提取完成: {} 帧, 耗时: {}ms", frames.size(), duration);
    return frames;
  } catch (const std::exception &e) {
    handleException(e, "extractFrames");
    return std::vector<cv::Mat>();
  }
}

cv::VideoWriter VideoIO::createVideo(const std::string &filename, int fourcc,
                                     double fps, const cv::Size &frameSize,
                                     bool isColor) {
  try {
    videoLogger->info("创建视频文件: {}", filename);
    videoLogger->info("参数: {}x{} @{}fps, codec: {}", frameSize.width,
                      frameSize.height, fps, getFourCCString(fourcc));

    cv::VideoWriter writer(filename, fourcc, fps, frameSize, isColor);

    if (!writer.isOpened()) {
      videoLogger->error("无法创建视频写入器");
      return cv::VideoWriter();
    }

    return writer;
  } catch (const std::exception &e) {
    handleException(e, "createVideo");
    return cv::VideoWriter();
  }
}

bool VideoIO::writeFrame(cv::VideoWriter &writer, const cv::Mat &frame) {
  try {
    if (!writer.isOpened()) {
      videoLogger->error("视频写入器未打开");
      return false;
    }

    writer.write(frame);
    return true;
  } catch (const std::exception &e) {
    return handleException(e, "writeFrame");
  }
}

bool VideoIO::saveFramesToVideo(const std::string &filename,
                                const std::vector<cv::Mat> &frames,
                                double fps) {
  try {
    if (frames.empty()) {
      videoLogger->error("没有帧可以保存");
      return false;
    }

    const cv::Size frameSize(frames[0].cols, frames[0].rows);
    int fourcc = getFourCC(getDefaultCodec());

    auto writer = createVideo(filename, fourcc, fps, frameSize);
    if (!writer.isOpened()) {
      return false;
    }

    videoLogger->info("开始保存 {} 帧到视频文件", frames.size());
    auto startTime = std::chrono::steady_clock::now();

    for (size_t i = 0; i < frames.size(); ++i) {
      if (!writeFrame(writer, frames[i])) {
        videoLogger->error("写入第 {} 帧时失败", i);
        return false;
      }

      if (i % 100 == 0) {
        videoLogger->info("已写入 {} 帧", i);
      }
    }

    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        endTime - startTime)
                        .count();

    videoLogger->info("视频保存完成, 耗时: {}ms", duration);
    return true;
  } catch (const std::exception &e) {
    return handleException(e, "saveFramesToVideo");
  }
}

int VideoIO::getFourCC(const std::string &codec) {
  return cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]);
}

std::string VideoIO::getFourCCString(int fourcc) {
  char code[5] = {0};
  for (int i = 0; i < 4; ++i) {
    code[i] = static_cast<char>(fourcc & 0xFF);
    fourcc >>= 8;
  }
  return std::string(code);
}

std::vector<std::string> VideoIO::getAvailableCodecs() {
  return {
      "XVID", // Xvid MPEG-4
      "MJPG", // Motion JPEG
      "H264", // H.264/AVC
      "MP4V", // MPEG-4
      "AVC1", // H.264/AVC
      "DIVX", // DivX
  };
}

bool VideoIO::isVideoFile(const std::string &filename) {
  std::string ext = std::filesystem::path(filename).extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

  return std::find(VIDEO_EXTENSIONS.begin(), VIDEO_EXTENSIONS.end(), ext) !=
         VIDEO_EXTENSIONS.end();
}

std::string VideoIO::getDefaultCodec() {
#ifdef _WIN32
  return "XVID";
#else
  return "MJPG";
#endif
}

double VideoIO::getOptimalFPS(const std::vector<cv::Mat> &frames) {
  return frames.empty() ? 30.0 : 30.0; // 默认返回30fps
}

bool VideoIO::resizeVideo(const std::string &input, const std::string &output,
                          const cv::Size &newSize) {
  try {
    auto inputCap = openVideo(input);
    if (!inputCap.isOpened())
      return false;

    VideoInfo info = getVideoInfo(inputCap);
    auto writer =
        createVideo(output, getFourCC(getDefaultCodec()), info.fps, newSize);

    cv::Mat frame, resized;
    while (true) {
      if (!inputCap.read(frame))
        break;
      cv::resize(frame, resized, newSize);
      writer.write(resized);
    }

    return true;
  } catch (const std::exception &e) {
    return handleException(e, "resizeVideo");
  }
}

bool VideoIO::changeVideoSpeed(const std::string &input,
                               const std::string &output, double speedFactor) {
  try {
    if (speedFactor <= 0) {
      videoLogger->error("无效的速度因子: {}", speedFactor);
      return false;
    }

    auto inputCap = openVideo(input);
    if (!inputCap.isOpened())
      return false;

    VideoInfo info = getVideoInfo(inputCap);
    auto writer =
        createVideo(output, getFourCC(getDefaultCodec()),
                    info.fps * speedFactor, cv::Size(info.width, info.height));

    cv::Mat frame;
    while (true) {
      if (!inputCap.read(frame))
        break;
      writer.write(frame);
    }

    return true;
  } catch (const std::exception &e) {
    return handleException(e, "changeVideoSpeed");
  }
}

bool VideoIO::extractKeyFrames(const std::string &filename,
                               std::vector<cv::Mat> &keyFrames,
                               double threshold) {
  try {
    auto cap = openVideo(filename);
    if (!cap.isOpened())
      return false;

    cv::Mat currentFrame, previousFrame;
    cv::Mat diff;

    while (true) {
      if (!cap.read(currentFrame))
        break;

      if (!previousFrame.empty()) {
        cv::absdiff(currentFrame, previousFrame, diff);
        cv::Scalar meanDiff = cv::mean(diff);

        if (meanDiff[0] + meanDiff[1] + meanDiff[2] > threshold * 255 * 3) {
          keyFrames.push_back(currentFrame.clone());
        }
      } else {
        keyFrames.push_back(currentFrame.clone());
      }

      currentFrame.copyTo(previousFrame);
    }

    videoLogger->info("提取了 {} 个关键帧", keyFrames.size());
    return true;
  } catch (const std::exception &e) {
    return handleException(e, "extractKeyFrames");
  }
}

bool VideoIO::batchProcessFrames(
    const std::vector<cv::Mat> &input, std::vector<cv::Mat> &output,
    const std::function<void(cv::Mat &)> &processor, bool useParallel) {
  try {
    output.resize(input.size());

    if (useParallel && m_enableParallel) {
#pragma omp parallel for if (m_enableParallel)
      for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        output[i] = input[i].clone();
        processor(output[i]);
      }
    } else {
      for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i].clone();
        processor(output[i]);
      }
    }

    return true;
  } catch (const std::exception &e) {
    return handleException(e, "batchProcessFrames");
  }
}

void VideoIO::enableParallelProcessing(bool enable) {
  m_enableParallel = enable;
  if (enable) {
    cv::setNumThreads(cv::getNumberOfCPUs());
  } else {
    cv::setNumThreads(1);
  }
}

void VideoIO::setBufferSize(size_t size) { m_bufferSize = size; }

void VideoIO::enableGPUAcceleration(bool enable) {
  if (enable && !isGPUAvailable()) {
    videoLogger->warn("GPU加速不可用");
    return;
  }
  m_useGPU = enable;
  if (enable) {
    initializeGPU();
  }
}

bool VideoIO::isGPUAvailable() {
  return cv::cuda::getCudaEnabledDeviceCount() > 0;
}

void VideoIO::initializeGPU() {
  if (isGPUAvailable()) {
    cv::cuda::setDevice(0);
    videoLogger->info("GPU初始化成功");
  }
}

bool VideoIO::stabilizeVideo(const std::string &input,
                             const std::string &output,
                             double smoothingRadius) {
  try {
    auto cap = openVideo(input);
    if (!cap.isOpened())
      return false;

    VideoInfo info = getVideoInfo(cap);
    auto writer = createVideo(output, getFourCC(getDefaultCodec()), info.fps,
                              cv::Size(info.width, info.height));

    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> transforms;
    cv::Mat prevFrame, currFrame;

    // 计算帧间变换
    while (cap.read(currFrame)) {
      if (!prevFrame.empty()) {
        transforms.push_back(getMotionTransform(prevFrame, currFrame));
      }
      frames.push_back(currFrame.clone());
      currFrame.copyTo(prevFrame);
    }

    // 平滑变换
    for (size_t i = 0; i < transforms.size(); ++i) {
      int start = std::max(0, static_cast<int>(i - smoothingRadius));
      int end = std::min(static_cast<int>(transforms.size()),
                         static_cast<int>(i + smoothingRadius + 1));
      cv::Mat smoothed =
          cv::Mat::zeros(transforms[i].size(), transforms[i].type());

      for (int j = start; j < end; ++j) {
        smoothed += transforms[j];
      }
      smoothed /= (end - start);

      cv::warpAffine(frames[i + 1], frames[i + 1], smoothed,
                     frames[i + 1].size());
    }

    // 写入稳定后的帧
    for (const auto &frame : frames) {
      writer.write(frame);
    }

    return true;
  } catch (const std::exception &e) {
    return handleException(e, "stabilizeVideo");
  }
}

bool VideoIO::addWatermark(const std::string &input, const std::string &output,
                           const cv::Mat &watermark,
                           const cv::Point &position) {
  try {
    auto cap = openVideo(input);
    if (!cap.isOpened())
      return false;

    VideoInfo info = getVideoInfo(cap);
    auto writer = createVideo(output, getFourCC(getDefaultCodec()), info.fps,
                              cv::Size(info.width, info.height));

    cv::Mat frame;
    cv::Mat watermarkResized;
    cv::resize(watermark, watermarkResized,
               cv::Size(info.width / 4, info.height / 4));

    while (cap.read(frame)) {
      cv::Mat roi =
          frame(cv::Rect(position.x, position.y, watermarkResized.cols,
                         watermarkResized.rows));
      cv::addWeighted(roi, 1.0, watermarkResized, 0.3, 0.0, roi);
      writer.write(frame);
    }

    return true;
  } catch (const std::exception &e) {
    return handleException(e, "addWatermark");
  }
}

cv::Mat VideoIO::getMotionTransform(const cv::Mat &frame1,
                                    const cv::Mat &frame2) {
  cv::Mat gray1, gray2;
  cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

  std::vector<cv::Point2f> points1, points2;
  cv::goodFeaturesToTrack(gray1, points1, 100, 0.01, 10);

  std::vector<uchar> status;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(gray1, gray2, points1, points2, status, err);

  std::vector<cv::Point2f> goodPoints1, goodPoints2;
  for (size_t i = 0; i < status.size(); i++) {
    if (status[i]) {
      goodPoints1.push_back(points1[i]);
      goodPoints2.push_back(points2[i]);
    }
  }

  return cv::estimateAffinePartial2D(goodPoints1, goodPoints2);
}
