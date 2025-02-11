#include <bitset>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <ranges>


using namespace cv;
using namespace std;

// 辅助函数：字符串转二进制位流
vector<bool> str_to_bits(const string &message) {
  vector<bool> bits;
  for (char c : message) {
    bitset<8> bs(c);
    for (int i = 7; i >= 0; --i) {
      bits.push_back(bs[i]);
    }
  }
  return bits;
}

// 辅助函数：二进制位流转字符串
string bits_to_str(const vector<bool> &bits) {
  string str;
  for (size_t i = 0; i < bits.size(); i += 8) {
    bitset<8> bs;
    for (int j = 0; j < 8 && (i + j) < bits.size(); ++j) {
      bs[7 - j] = bits[i + j];
    }
    str += static_cast<char>(bs.to_ulong());
  }
  return str;
}

// 傅里叶隐写嵌入函数
Mat embed_message(Mat carrier, const string &message, double alpha = 0.1) {
  // 转换为灰度图
  Mat gray;
  cvtColor(carrier, gray, COLOR_BGR2GRAY);
  gray.convertTo(gray, CV_32F);

  // 执行DFT
  Mat planes[] = {gray, Mat::zeros(gray.size(), CV_32F)};
  Mat complex;
  merge(planes, 2, complex);
  dft(complex, complex);

  // 频域移位
  int cx = complex.cols / 2;
  int cy = complex.rows / 2;
  Mat q0(complex, Rect(0, 0, cx, cy));   // 左上
  Mat q1(complex, Rect(cx, 0, cx, cy));  // 右上
  Mat q2(complex, Rect(0, cy, cx, cy));  // 左下
  Mat q3(complex, Rect(cx, cy, cx, cy)); // 右下
  Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);

  // 转换消息为二进制位流
  vector<bool> msg_bits = str_to_bits(message);
  const int required_bits = complex.rows * complex.cols / 16;
  if (msg_bits.size() > required_bits) {
    throw runtime_error("Message too long for carrier");
  }

  // 在中高频区域嵌入信息（示例：环形区域）
  int min_radius = complex.rows / 8;
  int max_radius = complex.rows / 4;
  auto coord_view = views::iota(0, complex.rows * complex.cols) |
                    views::transform([&](int idx) {
                      int y = idx / complex.cols;
                      int x = idx % complex.cols;
                      double dx = x - cx;
                      double dy = y - cy;
                      return make_tuple(y, x, sqrt(dx * dx + dy * dy));
                    }) |
                    views::filter([&](const auto &t) {
                      auto [y, x, r] = t;
                      return r > min_radius && r < max_radius;
                    });

  // 嵌入数据
  auto bit_it = msg_bits.begin();
  for (const auto &[y, x, r] : coord_view) {
    if (bit_it == msg_bits.end())
      break;

    Vec2f &pixel = complex.at<Vec2f>(y, x);
    float magnitude = norm(pixel);
    float new_mag = magnitude + alpha * (*bit_it ? 1.0f : -1.0f);
    float scale = new_mag / magnitude;

    pixel[0] *= scale; // 实部
    pixel[1] *= scale; // 虚部

    ++bit_it;
  }

  // 逆频域移位
  Mat shifted_back;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);

  // 逆DFT
  Mat inverse;
  idft(complex, inverse, DFT_SCALE | DFT_REAL_OUTPUT);

  // 转换为8UC1
  Mat result;
  inverse.convertTo(result, CV_8U);
  return result;
}

// 傅里叶隐写提取函数
string extract_message(Mat stego, int msg_length, double alpha = 0.1) {
  Mat gray;
  cvtColor(stego, gray, COLOR_BGR2GRAY);
  gray.convertTo(gray, CV_32F);

  // 执行DFT
  Mat planes[] = {gray, Mat::zeros(gray.size(), CV_32F)};
  Mat complex;
  merge(planes, 2, complex);
  dft(complex, complex);

  // 频域移位优化
  const int cx = complex.cols / 2;
  const int cy = complex.rows / 2;
  const int block_size = 32; // 缓存友好的块大小

  // 预先创建四个ROI Mat对象
  Mat q0(complex, Rect(0, 0, cx, cy));   // 左上
  Mat q1(complex, Rect(cx, 0, cx, cy));  // 右上
  Mat q2(complex, Rect(0, cy, cx, cy));  // 左下
  Mat q3(complex, Rect(cx, cy, cx, cy)); // 右下

  parallel_for_(Range(0, 4), [&](const Range &range) {
    for (int k = range.start; k < range.end; k++) {
      Mat *q[4] = {&q0, &q1, &q2, &q3};
      Mat tmp;
      q[k]->copyTo(tmp);
      q[3 - k]->copyTo(*q[k]);
      tmp.copyTo(*q[3 - k]);
    }
  });

  // 优化坐标生成和过滤
  const int min_radius = complex.rows / 8;
  const int max_radius = complex.rows / 4;
  const int min_radius_sq = min_radius * min_radius;
  const int max_radius_sq = max_radius * max_radius;

  vector<bool> extracted_bits;
  extracted_bits.reserve(msg_length * 8);

  // 并行处理数据提取
  const int total_pixels = complex.rows * complex.cols;
  mutex mtx;

  parallel_for_(Range(0, total_pixels), [&](const Range &range) {
    vector<bool> local_bits;
    local_bits.reserve((range.end - range.start) / 64); // 预估大小

    for (int idx = range.start; idx < range.end; idx++) {
      if (extracted_bits.size() >= msg_length * 8)
        break;

      const int y = idx / complex.cols;
      const int x = idx % complex.cols;
      const int dx = x - cx;
      const int dy = y - cy;
      const int r_sq = dx * dx + dy * dy; // 避免开方计算

      if (r_sq > min_radius_sq && r_sq < max_radius_sq) {
        Vec2f pixel = complex.at<Vec2f>(y, x);
        const float magnitude = norm(pixel);
        // 改进的阈值判断，使用浮点比较
        const bool bit = std::abs(magnitude - alpha) >
                             std::numeric_limits<float>::epsilon() &&
                         magnitude > alpha / 2;
        local_bits.push_back(bit);
      }
    }

    // 合并局部结果
    if (!local_bits.empty()) {
      lock_guard<mutex> lock(mtx);
      extracted_bits.insert(extracted_bits.end(), local_bits.begin(),
                            local_bits.end());
    }
  });

  // 截取所需长度
  if (extracted_bits.size() > msg_length * 8) {
    extracted_bits.resize(msg_length * 8);
  }

  return bits_to_str(extracted_bits);
}