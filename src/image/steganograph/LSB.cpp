#include "LSB.hpp"

#include <bitset>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <span>
#include <vector>

using namespace cv;
using namespace std;

// 将字符串预处理为位流缓存
BitStreamBuffer::BitStreamBuffer(const string &message, bool add_terminator) {
  bits.reserve(message.length() * 8 + (add_terminator ? 8 : 0));
  for (char c : message) {
    bitset<8> charBits(c);
    for (int i = 7; i >= 0; --i) {
      bits.push_back(charBits[i]);
    }
  }
  if (add_terminator) {
    bits.push_back(false); // 终止符
  }
}

const vector<bool> &BitStreamBuffer::getBits() const { return bits; }
size_t BitStreamBuffer::size() const { return bits.size(); }

// 优化后的LSB嵌入函数
void embedLSB(Mat &image, const string &message) {
  BitStreamBuffer bit_buffer(message);
  const auto &bits = bit_buffer.getBits();

  // 容量检查
  const size_t maxBits =
      static_cast<size_t>(image.rows) * image.cols * image.channels();
  if (bits.size() > maxBits) {
    throw runtime_error("Message too large for image capacity");
  }

  // 并行处理参数
  const int chunk_size = 1024;
  const int num_chunks =
      (image.total() * image.channels() + chunk_size - 1) / chunk_size;

#ifdef USE_PARALLEL
#pragma omp parallel for
#endif
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    const int start = chunk * chunk_size;
    const int end = min(start + chunk_size,
                        static_cast<int>(image.total() * image.channels()));

    uchar *data = image.data + start;
    for (int i = start; i < end && (i / 3) < bits.size(); ++i) {
      const int bit_idx = i / 3;
      if (bit_idx < bits.size()) {
        data[i - start] = (data[i - start] & 0xFE) | bits[bit_idx];
      }
    }
  }
}

// 优化后的LSB提取函数
string extractLSB(const Mat &image) {
  vector<bool> bits;
  bits.reserve(image.total() * image.channels());

  // 使用 std::execution::par_unseq 进行并行处理
  const int total_pixels = image.total();
  vector<uchar> extracted_bits(total_pixels * 3);

  // 使用 ranges 和 views 替代 counting_iterator
  vector<int> indices(total_pixels);
  iota(indices.begin(), indices.end(), 0);

#ifdef USE_PARALLEL
  for_each(execution::par_unseq, indices.begin(), indices.end(), [&](int i) {
#else
  for_each(indices.begin(), indices.end(), [&](int i) {
#endif
    const Vec3b &pixel = image.at<Vec3b>(i / image.cols, i % image.cols);
    for (int c = 0; c < 3; ++c) {
      extracted_bits[i * 3 + c] = pixel[c] & 1;
    }
  });

  // 转换为字符串
  string message;
  message.reserve(extracted_bits.size() / 8);

  for (size_t i = 0; i < extracted_bits.size(); i += 8) {
    if (i + 8 > extracted_bits.size())
      break;

    bitset<8> charBits;
    for (int j = 0; j < 8; ++j) {
      charBits[7 - j] = extracted_bits[i + j];
    }

    char c = static_cast<char>(charBits.to_ulong());
    if (c == '\0')
      break;
    message += c;
  }

  return message;
}

// 优化后的位平面分解函数
Mat getBitPlane(const Mat &image, int bitPosition) {
  CV_Assert(image.depth() == CV_8U);
  Mat plane = Mat::zeros(image.size(), CV_8UC1);

  const uchar mask = 1 << bitPosition;
  const int total_pixels = image.total();

#ifdef USE_CUDA
  // 使用CUDA加速
  cuda::GpuMat d_image(image);
  cuda::GpuMat d_plane(plane);
  cuda::bitwise_and(d_image, Scalar(mask), d_plane);
  cuda::divide(d_plane, Scalar(mask), d_plane);
  d_plane.convertTo(plane, CV_8UC1, 255);
#else
  // 使用SIMD优化的并行处理
  parallel_for_(Range(0, total_pixels), [&](const Range &range) {
    for (int i = range.start; i < range.end; ++i) {
      plane.data[i] = ((image.data[i] & mask) >> bitPosition) * 255;
    }
  });
#endif

  return plane;
}

// 流处理版本的LSB嵌入函数（适用于大文件）
StreamingLSBEncoder::StreamingLSBEncoder(Mat &image)
    : image_(image), current_pos_(0) {}

bool StreamingLSBEncoder::embedChunk(span<const char> data) {
  BitStreamBuffer bit_buffer(string(data.begin(), data.end()), false);
  const auto &bits = bit_buffer.getBits();

  const size_t available_bits =
      (image_.total() * image_.channels()) - current_pos_;
  if (bits.size() > available_bits)
    return false;

  const int chunk_size = min(static_cast<int>(bits.size()), 1024);
  auto *img_data = image_.ptr<uchar>();

  for (size_t i = 0; i < bits.size(); ++i) {
    img_data[current_pos_ + i] = (img_data[current_pos_ + i] & 0xFE) | bits[i];
  }
  current_pos_ += bits.size();
  return true;
}