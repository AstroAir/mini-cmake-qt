#include "VideoFilters.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <random>
#include <thread>

VideoFilters::ProcessOptions VideoFilters::processOptions;

void VideoFilters::setProcessOptions(const ProcessOptions &options) {
  processOptions = options;
}

void VideoFilters::preprocessFrame(cv::Mat &frame) {
  if (!processOptions.preprocessFrame) return;

  // 基本图像预处理
  cv::Mat temp;
  cv::medianBlur(frame, temp, 3);
  cv::normalize(temp, frame, 0, 255, cv::NORM_MINMAX);
}

void VideoFilters::applyNoise(cv::Mat &frame, float amount) {
  cv::Mat noise = cv::Mat::zeros(frame.size(), frame.type());
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0, amount * 255);

  if (processOptions.useGPU && cv::cuda::getCudaEnabledDeviceCount() > 0) {
    cv::cuda::GpuMat gpuFrame, gpuNoise;
    gpuFrame.upload(frame);
    // GPU噪声生成和添加
    // ...
    gpuFrame.download(frame);
  } else {
    // 多线程CPU实现
    int rows_per_thread = frame.rows / processOptions.threads;
    std::vector<std::thread> threads;
    
    for (int t = 0; t < processOptions.threads; ++t) {
      threads.emplace_back([&, t]() {
        int start_row = t * rows_per_thread;
        int end_row = (t == processOptions.threads - 1) ? frame.rows : (t + 1) * rows_per_thread;
        
        for (int i = start_row; i < end_row; ++i) {
          for (int j = 0; j < frame.cols; ++j) {
            for (int c = 0; c < frame.channels(); ++c) {
              float noise_value = distribution(generator);
              frame.at<cv::Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(
                frame.at<cv::Vec3b>(i, j)[c] + noise_value);
            }
          }
        }
      });
    }
    
    for (auto &thread : threads) {
      thread.join();
    }
  }
}

void VideoFilters::applyTiltShift(cv::Mat &frame, int focusPosition) {
  if (focusPosition < 0) focusPosition = frame.rows / 2;
  
  cv::Mat mask = cv::Mat::zeros(frame.size(), CV_32F);
  float sigma = frame.rows * 0.15f;
  
  for (int i = 0; i < frame.rows; ++i) {
    float dist = std::abs(i - focusPosition);
    float blur_amount = 1.0f - std::exp(-dist * dist / (2 * sigma * sigma));
    mask.row(i).setTo(blur_amount);
  }
  
  cv::Mat blurred;
  cv::GaussianBlur(frame, blurred, cv::Size(31, 31), 0);
  
  // 混合原始图像和模糊图像
  frame = frame.mul(1.0f - mask) + blurred.mul(mask);
}

void VideoFilters::applyGlitch(cv::Mat &frame, float intensity) {
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
  
  // 随机通道偏移
  std::vector<cv::Mat> channels;
  cv::split(frame, channels);
  
  for (auto &channel : channels) {
    if (distribution(generator) < intensity) {
      int offset_x = distribution(generator) * 20 - 10;
      cv::Mat shifted;
      cv::warpAffine(channel, shifted, 
                     cv::getRotationMatrix2D(cv::Point2f(0,0), 0, 1.0),
                     channel.size(),
                     cv::INTER_LINEAR,
                     cv::BORDER_REPLICATE);
      channel = shifted;
    }
  }
  
  cv::merge(channels, frame);
}

void VideoFilters::applyColorGrading(cv::Mat &frame, const cv::Mat &grading) {
  // 应用色彩分级查找表(LUT)
  if (processOptions.useGPU && cv::cuda::getCudaEnabledDeviceCount() > 0) {
    cv::cuda::GpuMat gpuFrame;
    gpuFrame.upload(frame);
    // 使用GPU LUT
    cv::LUT(gpuFrame, grading, gpuFrame);
    gpuFrame.download(frame);
  } else {
    cv::LUT(frame, grading, frame);
  }
}

void VideoFilters::adjustHSV(cv::Mat &frame, float hue, float saturation,
                             float value) {
  cv::Mat hsv;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

  std::vector<cv::Mat> channels;
  cv::split(hsv, channels);

  // 调整Hue
  channels[0] = channels[0] * hue;
  // 调整Saturation
  channels[1] = channels[1] * saturation;
  // 调整Value
  channels[2] = channels[2] * value;

  cv::merge(channels, hsv);
  cv::cvtColor(hsv, frame, cv::COLOR_HSV2BGR);
}

void VideoFilters::adjustColorBalance(cv::Mat &frame, float red, float green,
                                      float blue) {
  std::vector<cv::Mat> channels;
  cv::split(frame, channels);

  channels[0] = channels[0] * blue;
  channels[1] = channels[1] * green;
  channels[2] = channels[2] * red;

  cv::merge(channels, frame);
}

void VideoFilters::applySketch(cv::Mat &frame) {
  cv::Mat gray, inverted;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
  cv::bitwise_not(gray, inverted);

  cv::Mat result;
  cv::divide(gray, inverted, result, 256.0);
  cv::cvtColor(result, frame, cv::COLOR_GRAY2BGR);
}

void VideoFilters::applyCartoon(cv::Mat &frame) {
  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  // 边缘检测
  cv::Mat edges;
  cv::medianBlur(gray, gray, 5);
  cv::adaptiveThreshold(gray, edges, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                        cv::THRESH_BINARY, 9, 9);

  // 色彩简化
  cv::Mat color;
  bilateralStyleFilter(frame, 2);

  // 合并边缘和色彩
  cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
  cv::bitwise_and(frame, edges, frame);
}

void VideoFilters::applyOilPainting(cv::Mat &frame, int size) {
  cv::Mat result = frame.clone();
  int halfSize = size / 2;

  for (int i = halfSize; i < frame.rows - halfSize; i++) {
    for (int j = halfSize; j < frame.cols - halfSize; j++) {
      std::vector<int> intensityCounts(256, 0);
      std::vector<cv::Vec3f> averageColors(256, cv::Vec3f(0, 0, 0));

      // 统计邻域像素
      for (int y = -halfSize; y <= halfSize; y++) {
        for (int x = -halfSize; x <= halfSize; x++) {
          cv::Vec3b color = frame.at<cv::Vec3b>(i + y, j + x);
          int intensity = (color[0] + color[1] + color[2]) / 3;
          intensityCounts[intensity]++;
          averageColors[intensity] += cv::Vec3f(color[0], color[1], color[2]);
        }
      }

      // 找出最常见的强度值
      int maxCount = 0;
      int maxIntensity = 0;
      for (int k = 0; k < 256; k++) {
        if (intensityCounts[k] > maxCount) {
          maxCount = intensityCounts[k];
          maxIntensity = k;
        }
      }

      // 计算平均颜色
      cv::Vec3f avgColor = averageColors[maxIntensity] / maxCount;
      result.at<cv::Vec3b>(i, j) =
          cv::Vec3b(avgColor[0], avgColor[1], avgColor[2]);
    }
  }

  frame = result;
}

void VideoFilters::applyDream(cv::Mat &frame, float strength) {
  cv::Mat blurred;
  cv::GaussianBlur(frame, blurred, cv::Size(21, 21), 0);

  cv::addWeighted(frame, 1.0f, blurred, strength, 0, frame);

  // 增加亮度和对比度
  frame.convertTo(frame, -1, 1.1, 10);
}

void VideoFilters::applyLUT(cv::Mat &frame, const cv::Mat &lut) {
  CV_Assert(lut.total() == 256 && lut.type() == CV_8UC3);

  cv::Mat result;
  cv::LUT(frame, lut, result);
  frame = result;
}

cv::Mat VideoFilters::createLUT(const std::string &style) {
  cv::Mat lut(1, 256, CV_8UC3);

  if (style == "warm") {
    for (int i = 0; i < 256; i++) {
      lut.at<cv::Vec3b>(0, i) =
          cv::Vec3b(cv::saturate_cast<uchar>(i * 1.1), // B
                    cv::saturate_cast<uchar>(i),       // G
                    cv::saturate_cast<uchar>(i * 1.2)  // R
          );
    }
  } else if (style == "cool") {
    for (int i = 0; i < 256; i++) {
      lut.at<cv::Vec3b>(0, i) =
          cv::Vec3b(cv::saturate_cast<uchar>(i * 1.2), // B
                    cv::saturate_cast<uchar>(i),       // G
                    cv::saturate_cast<uchar>(i * 0.8)  // R
          );
    }
  }
  // 可以添加更多预设样式

  return lut;
}

void VideoFilters::bilateralStyleFilter(cv::Mat &frame, int iterations) {
  for (int i = 0; i < iterations; i++) {
    cv::bilateralFilter(frame, frame, 9, 75, 75);
  }
}
