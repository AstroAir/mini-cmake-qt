#include "Channels.hpp"

#include <bitset>
#include <numeric>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <stdexcept>
#include <vector>

using namespace cv;
using namespace std;

vector<bool> BinaryConverter::stringToBits(const string &message,
                                           bool addTerminator) {
  vector<bool> bits;
  bits.reserve(message.length() * 8 + (addTerminator ? 8 : 0));

  for (char c : message) {
    bitset<8> bs(c);
    for (int i = 7; i >= 0; --i) {
      bits.push_back(bs[i]);
    }
  }

  if (addTerminator) {
    bits.insert(bits.end(), 8, false); // 添加终止符
  }
  return bits;
}

string BinaryConverter::bitsToString(const vector<bool> &bits) {
  string result;
  result.reserve(bits.size() / 8);

  for (size_t i = 0; i < bits.size(); i += 8) {
    if (i + 8 > bits.size())
      break;

    char c = 0;
    for (int j = 0; j < 8; ++j) {
      c |= bits[i + j] << (7 - j);
    }

    if (c == '\0')
      break;
    result += c;
  }
  return result;
}

// 优化后的Alpha通道隐写嵌入函数
void alpha_channel_hide(Mat &image, const string &message) {
  if (image.channels() != 4) {
    throw runtime_error("Image does not have alpha channel");
  }

  // 将消息转换为二进制流
  vector<bool> bits = BinaryConverter::stringToBits(message);

  // 检查容量
  const int max_bits = image.rows * image.cols;
  if (bits.size() > max_bits) {
    throw runtime_error("Message too large for image capacity");
  }

  // 并行处理参数
  const int chunk_size = 1024;
  const int num_chunks = (bits.size() + chunk_size - 1) / chunk_size;

  parallel_for_(Range(0, num_chunks), [&](const Range &range) {
    for (int chunk = range.start; chunk < range.end; chunk++) {
      const int start_idx = chunk * chunk_size;
      const int end_idx =
          min(start_idx + chunk_size, static_cast<int>(bits.size()));

      for (int idx = start_idx; idx < end_idx; ++idx) {
        const int y = idx / image.cols;
        const int x = idx % image.cols;
        Vec4b &pixel = image.at<Vec4b>(y, x);
        pixel[3] = (pixel[3] & 0xFE) | bits[idx];
      }
    }
  });
}

// 优化后的Alpha通道隐写提取函数
string alpha_channel_extract(const Mat &image) {
  if (image.channels() != 4) {
    throw runtime_error("Image does not have alpha channel");
  }

  const int total_pixels = image.rows * image.cols;
  vector<bool> bits;
  bits.reserve(total_pixels);

  // 使用多线程提取位信息
  const int chunk_size = 1024;
  const int num_chunks = (total_pixels + chunk_size - 1) / chunk_size;
  vector<vector<bool>> chunk_bits(num_chunks);

  parallel_for_(Range(0, num_chunks), [&](const Range &range) {
    for (int chunk = range.start; chunk < range.end; chunk++) {
      const int start_idx = chunk * chunk_size;
      const int end_idx = min(start_idx + chunk_size, total_pixels);
      vector<bool> &local_bits = chunk_bits[chunk];
      local_bits.reserve(end_idx - start_idx);

      for (int idx = start_idx; idx < end_idx; ++idx) {
        const int y = idx / image.cols;
        const int x = idx % image.cols;
        const Vec4b &pixel = image.at<Vec4b>(y, x);
        local_bits.push_back(pixel[3] & 1);
      }
    }
  });

  // 合并结果
  bits.reserve(total_pixels);
  for (const auto &chunk : chunk_bits) {
    bits.insert(bits.end(), chunk.begin(), chunk.end());
  }

  return BinaryConverter::bitsToString(bits);
}

// 优化后的通道分析函数
void analyze_channels(const Mat &image) {
  vector<Mat> channels;
  split(image, channels);

  const string names[] = {"Blue", "Green", "Red", "Alpha"};
  parallel_for_(Range(0, image.channels()), [&](const Range &range) {
    for (int i = range.start; i < range.end; ++i) {
      Mat display;
      normalize(channels[i], display, 0, 255, NORM_MINMAX);
      imshow(names[i] + " Channel", display);
    }
  });
  waitKey(0);
}

// 计算图像可隐写容量
size_t calculate_capacity(const Mat &image, const ChannelConfig &config) {
  int usedChannels = 0;
  usedChannels +=
      config.useBlue + config.useGreen + config.useRed + config.useAlpha;

  return (image.rows * image.cols * usedChannels * config.bitsPerChannel) / 8;
}

// 多通道隐写嵌入
void multi_channel_hide(Mat &image, const string &message,
                        const ChannelConfig &config) {
  if (message.length() > calculate_capacity(image, config)) {
    throw runtime_error("Message too large for selected channels");
  }

  vector<bool> bits = BinaryConverter::stringToBits(message);
  vector<int> activeChannels;

  if (config.useBlue)
    activeChannels.push_back(0);
  if (config.useGreen)
    activeChannels.push_back(1);
  if (config.useRed)
    activeChannels.push_back(2);
  if (config.useAlpha && image.channels() == 4)
    activeChannels.push_back(3);

  // 使用密钥生成随机序列
  vector<int> pixelOrder(image.rows * image.cols);
  iota(pixelOrder.begin(), pixelOrder.end(), 0);
  if (config.scrambleKey != 0) {
    mt19937 gen(static_cast<unsigned int>(config.scrambleKey * 1000000));
    shuffle(pixelOrder.begin(), pixelOrder.end(), gen);
  }

  parallel_for_(Range(0, bits.size()), [&](const Range &range) {
    for (int i = range.start; i < range.end; i++) {
      int pixelIdx = pixelOrder[i / activeChannels.size()];
      int channelIdx = activeChannels[i % activeChannels.size()];

      int y = pixelIdx / image.cols;
      int x = pixelIdx % image.cols;

      Vec4b &pixel = image.at<Vec4b>(y, x);
      int bitPos = (i / activeChannels.size()) % config.bitsPerChannel;

      // 修改指定位
      uchar mask = ~(1 << bitPos);
      pixel[channelIdx] =
          (pixel[channelIdx] & mask) | (bits[i] ? (1 << bitPos) : 0);
    }
  });
}

// 多通道隐写提取
string multi_channel_extract(const Mat &image, size_t messageLength,
                             const ChannelConfig &config) {
  vector<int> activeChannels;
  if (config.useBlue)
    activeChannels.push_back(0);
  if (config.useGreen)
    activeChannels.push_back(1);
  if (config.useRed)
    activeChannels.push_back(2);
  if (config.useAlpha && image.channels() == 4)
    activeChannels.push_back(3);

  // 使用相同的密钥重建随机序列
  vector<int> pixelOrder(image.rows * image.cols);
  iota(pixelOrder.begin(), pixelOrder.end(), 0);
  if (config.scrambleKey != 0) {
    mt19937 gen(static_cast<unsigned int>(config.scrambleKey * 1000000));
    shuffle(pixelOrder.begin(), pixelOrder.end(), gen);
  }

  // 预分配结果向量，避免并发写入
  const size_t totalBits = messageLength * 8;
  vector<vector<bool>> threadBits(omp_get_max_threads());
  vector<bool> bits(totalBits);

  parallel_for_(Range(0, totalBits), [&](const Range &range) {
    const int threadId = omp_get_thread_num();
    auto &localBits = threadBits[threadId];
    localBits.reserve(range.end - range.start);

    for (int i = range.start; i < range.end; i++) {
      int pixelIdx = pixelOrder[i / activeChannels.size()];
      int channelIdx = activeChannels[i % activeChannels.size()];

      int y = pixelIdx / image.cols;
      int x = pixelIdx % image.cols;

      const Vec4b &pixel = image.at<Vec4b>(y, x);
      int bitPos = (i / activeChannels.size()) % config.bitsPerChannel;

      localBits.push_back((pixel[channelIdx] >> bitPos) & 1);
    }
  });

  // 合并线程局部结果
  size_t offset = 0;
  for (const auto &threadResult : threadBits) {
    copy(threadResult.begin(), threadResult.end(), bits.begin() + offset);
    offset += threadResult.size();
  }

  return BinaryConverter::bitsToString(bits);
}

ChannelQuality analyze_channel_quality(const Mat &channel) {
  ChannelQuality quality;

  // 计算信息熵
  vector<int> histogram(256, 0);
  for (int i = 0; i < channel.rows; i++) {
    for (int j = 0; j < channel.cols; j++) {
      histogram[channel.at<uchar>(i, j)]++;
    }
  }

  quality.entropy = 0;
  double total = channel.total();
  for (int i = 0; i < 256; i++) {
    if (histogram[i] > 0) {
      double p = histogram[i] / total;
      quality.entropy -= p * log2(p);
    }
  }

  // 计算方差
  Scalar channelMean, stddev;
  meanStdDev(channel, channelMean, stddev);
  quality.variance = stddev[0] * stddev[0];

  // 计算相邻像素相关性
  Mat shifted;
  copyMakeBorder(channel, shifted, 0, 0, 1, 0, BORDER_REPLICATE);
  shifted = shifted(Rect(1, 0, channel.cols, channel.rows));

  Mat diff;
  absdiff(channel, shifted, diff);
  Scalar meanDiff = cv::mean(diff); // 使用 cv::mean 避免与变量名冲突
  quality.correlation = 1.0 - meanDiff[0] / 255.0;

  return quality;
}

// 分析所有通道的质量
vector<ChannelQuality> analyze_all_channels_quality(const Mat &image) {
  vector<Mat> channels;
  split(image, channels);

  vector<ChannelQuality> qualities;
  qualities.reserve(channels.size());

  for (const auto &channel : channels) {
    qualities.push_back(analyze_channel_quality(channel));
  }

  return qualities;
}
