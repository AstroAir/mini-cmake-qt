#include "Stego.hpp"
#include "LSB.hpp"

#include <bitset>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <ranges>

#ifdef USE_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#endif

using namespace cv;
using namespace std;

namespace {
// 新增：图像加密工具类
class ImageEncryptor {
  // ...implementation...
};

// 新增：DCT实现
cv::Mat dct_embed(cv::Mat &carrier, const std::vector<bool> &bits,
                  const StegoConfig &config) {
  Mat gray;
  cvtColor(carrier, gray, COLOR_BGR2GRAY);
  gray.convertTo(gray, CV_32F);

  const int blockSize = 8;
  auto bit_it = bits.begin();

  for (int i = 0; i < gray.rows; i += blockSize) {
    for (int j = 0; j < gray.cols; j += blockSize) {
      if (bit_it == bits.end())
        break;

      Rect roi(j, i, min(blockSize, gray.cols - j),
               min(blockSize, gray.rows - i));
      Mat block = gray(roi);
      dct(block, block);

      // 修改中频系数
      if (*bit_it) {
        block.at<float>(3, 3) += config.alpha * block.at<float>(0, 0);
        block.at<float>(4, 4) += config.alpha * block.at<float>(0, 0);
      }

      idct(block, block);
      block.copyTo(gray(roi));
      ++bit_it;
    }
  }

  Mat result;
  gray.convertTo(result, CV_8U);
  return result;
}

// 新增：DWT实现
cv::Mat dwt_embed(cv::Mat &carrier, const std::vector<bool> &bits,
                  const StegoConfig &config) {
  Mat gray;
  cvtColor(carrier, gray, COLOR_BGR2GRAY);
  gray.convertTo(gray, CV_32F);

  // 执行3层小波变换
  vector<Mat> wavelets;
  Mat current = gray.clone();

  for (int level = 0; level < 3; level++) {
    Mat LL, LH, HL, HH;
    int rows = current.rows / 2;
    int cols = current.cols / 2;

    // 小波分解
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        float sum = current.at<float>(i * 2, j * 2) +
                    current.at<float>(i * 2 + 1, j * 2) +
                    current.at<float>(i * 2, j * 2 + 1) +
                    current.at<float>(i * 2 + 1, j * 2 + 1);
        LL.at<float>(i, j) = sum / 4.0f;
      }
    }

    wavelets.push_back(current);
    current = LL;
  }

  // 在LL子带嵌入信息
  auto bit_it = bits.begin();
  for (int i = 0; i < current.rows && bit_it != bits.end(); i++) {
    for (int j = 0; j < current.cols && bit_it != bits.end(); j++) {
      if (*bit_it) {
        current.at<float>(i, j) += config.alpha;
      }
      ++bit_it;
    }
  }

  // 逆变换
  Mat result = current.clone();
  for (int level = wavelets.size() - 1; level >= 0; level--) {
    pyrUp(result, result, wavelets[level].size());
    result = result.mul(wavelets[level]);
  }

  result.convertTo(result, CV_8U);
  return result;
}

// 新增：LSB实现
cv::Mat lsb_embed(cv::Mat &carrier, const std::vector<bool> &bits,
                  const StegoConfig &config) {
  Mat result = carrier.clone();
  string bitString;
  for (bool bit : bits) {
    bitString += (bit ? '1' : '0');
  }

  // 使用现有的LSB功能
  embedLSB(result, bitString);
  return result;
}

#ifdef USE_CUDA
void process_image_cuda(cv::cuda::GpuMat &d_image, const StegoConfig &config) {
  cv::cuda::GpuMat d_gray;
  cv::cuda::cvtColor(d_image, d_gray, COLOR_BGR2GRAY);

  // 创建滤波器
  Ptr<cuda::Filter> gaussian =
      cuda::createGaussianFilter(CV_32F, CV_32F, Size(3, 3), config.alpha);

  // 应用滤波
  gaussian->apply(d_gray, d_gray);

  // 频域处理
  cv::cuda::dft(d_gray, d_gray, d_gray.size());

  // 返回空域
  cv::cuda::idft(d_gray, d_gray);

  cv::cuda::cvtColor(d_gray, d_image, COLOR_GRAY2BGR);
}
#endif

} // namespace

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
Mat embed_dft(Mat carrier, const vector<bool> &msg_bits,
              const StegoConfig &config) {
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
    float new_mag = magnitude + config.alpha * (*bit_it ? 1.0f : -1.0f);
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

Mat embed_message(Mat carrier, const string &message,
                  const StegoConfig &config) {
  // 输入验证
  if (carrier.empty()) {
    throw runtime_error("Empty carrier image");
  }

  // 根据选择的方法调用相应的实现
  vector<bool> msg_bits = str_to_bits(message);

  switch (config.method) {
  case StegoMethod::DFT:
    // 使用现有的DFT实现，但传入config参数
    return embed_dft(carrier, msg_bits, config);
  case StegoMethod::DCT:
    return dct_embed(carrier, msg_bits, config);
  case StegoMethod::DWT:
    return dwt_embed(carrier, msg_bits, config);
  case StegoMethod::LSB:
    return lsb_embed(carrier, msg_bits, config);
  default:
    throw runtime_error("Unsupported steganography method");
  }
}

// 傅里叶隐写提取函数
vector<bool> extract_dft(Mat stego, int msg_length, const StegoConfig &config) {
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

#ifdef USE_PARALLEL
  parallel_for_(Range(0, 4), [&](const Range &range) {
    for (int k = range.start; k < range.end; k++) {
      Mat *q[4] = {&q0, &q1, &q2, &q3};
      Mat tmp;
      q[k]->copyTo(tmp);
      q[3 - k]->copyTo(*q[k]);
      tmp.copyTo(*q[3 - k]);
    }
  });
#else
  Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);
#endif

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

#ifdef USE_PARALLEL
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
        const bool bit = std::abs(magnitude - config.alpha) >
                             std::numeric_limits<float>::epsilon() &&
                         magnitude > config.alpha / 2;
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
#else
  for (int idx = 0; idx < total_pixels; idx++) {
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
      const bool bit = std::abs(magnitude - config.alpha) >
                           std::numeric_limits<float>::epsilon() &&
                       magnitude > config.alpha / 2;
      extracted_bits.push_back(bit);
    }
  }
#endif

  // 截取所需长度
  if (extracted_bits.size() > msg_length * 8) {
    extracted_bits.resize(msg_length * 8);
  }

  return extracted_bits;
}

// 基于DCT的提取实现
vector<bool> extract_dct(Mat stego, int msg_length, const StegoConfig &config) {
  Mat gray;
  cvtColor(stego, gray, COLOR_BGR2GRAY);
  gray.convertTo(gray, CV_32F);

  vector<bool> extracted_bits;
  extracted_bits.reserve(msg_length * 8);

  const int blockSize = 8;
  for (int i = 0; i < gray.rows; i += blockSize) {
    for (int j = 0; j < gray.cols; j += blockSize) {
      if (extracted_bits.size() >= msg_length * 8)
        break;

      Rect roi(j, i, min(blockSize, gray.cols - j),
               min(blockSize, gray.rows - i));
      Mat block = gray(roi).clone();
      dct(block, block);

      // 从中频系数提取信息
      float magnitude = block.at<float>(3, 3);
      extracted_bits.push_back(magnitude > config.alpha / 2);
    }
  }

  return extracted_bits;
}

// 基于DWT的提取实现
vector<bool> extract_dwt(Mat stego, int msg_length, const StegoConfig &config) {
  Mat gray;
  cvtColor(stego, gray, COLOR_BGR2GRAY);
  gray.convertTo(gray, CV_32F);

  vector<bool> extracted_bits;
  extracted_bits.reserve(msg_length * 8);

  // 执行3层小波变换
  Mat wavelet = gray.clone();
  for (int level = 0; level < 3; level++) {
    int rows = wavelet.rows / 2;
    int cols = wavelet.cols / 2;
    Mat LL(rows, cols, CV_32F);

    // 提取LL子带系数
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (extracted_bits.size() >= msg_length * 8)
          break;
        float coef = wavelet.at<float>(i, j);
        extracted_bits.push_back(coef > config.alpha / 2);
      }
    }

    // 更新为下一层
    wavelet = LL;
  }

  return extracted_bits;
}

// 基于LSB的提取实现
vector<bool> extract_lsb(Mat stego, int msg_length, const StegoConfig &config) {
  // 直接使用已有的LSB提取功能
  string extracted = extractLSB(stego);

  // 转换为位流
  BitStreamBuffer buffer(extracted);
  auto bits = buffer.getBits();

  // 截取所需长度
  if (bits.size() > msg_length * 8) {
    bits.resize(msg_length * 8);
  }

  return bits;
}

string extract_message(Mat stego, int msg_length, const StegoConfig &config) {
  // 输入验证
  if (stego.empty()) {
    throw runtime_error("Empty stego image");
  }

  vector<bool> extracted_bits;

  switch (config.method) {
  case StegoMethod::DFT:
    extracted_bits = extract_dft(stego, msg_length, config);
    break;
  case StegoMethod::DCT:
    extracted_bits = extract_dct(stego, msg_length, config);
    break;
  case StegoMethod::DWT:
    extracted_bits = extract_dwt(stego, msg_length, config);
    break;
  case StegoMethod::LSB:
    extracted_bits = extract_lsb(stego, msg_length, config);
    break;
  default:
    throw runtime_error("Unsupported steganography method");
  }

  return bits_to_str(extracted_bits);
}

#ifdef USE_CUDA
// CUDA优化实现
namespace {
void process_image_cuda(cv::cuda::GpuMat &d_image, const StegoConfig &config) {
  cv::cuda::GpuMat d_gray;
  cv::cuda::cvtColor(d_image, d_gray, COLOR_BGR2GRAY);

  // 创建滤波器
  Ptr<cuda::Filter> gaussian =
      cuda::createGaussianFilter(CV_32F, CV_32F, Size(3, 3), config.alpha);

  // 应用滤波
  gaussian->apply(d_gray, d_gray);

  // 频域处理
  cv::cuda::dft(d_gray, d_gray, d_gray.size());

  // 返回空域
  cv::cuda::idft(d_gray, d_gray);

  cv::cuda::cvtColor(d_gray, d_image, COLOR_GRAY2BGR);
}
} // namespace
#endif

cv::Scalar MSSIM(const cv::Mat &i1, const cv::Mat &i2) {
  const double C1 = 6.5025;  // (0.01 * 255)^2
  const double C2 = 58.5225; // (0.03 * 255)^2

  cv::Mat I1, I2;
  i1.convertTo(I1, CV_32F);
  i2.convertTo(I2, CV_32F);

  cv::Mat I1_2 = I1.mul(I1);  // I1^2
  cv::Mat I2_2 = I2.mul(I2);  // I2^2
  cv::Mat I1_I2 = I1.mul(I2); // I1*I2

  cv::Mat mu1, mu2;
  cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
  cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

  cv::Mat mu1_2 = mu1.mul(mu1);
  cv::Mat mu2_2 = mu2.mul(mu2);
  cv::Mat mu1_mu2 = mu1.mul(mu2);

  cv::Mat sigma1_2, sigma2_2, sigma12;

  cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
  sigma1_2 -= mu1_2;

  cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
  sigma2_2 -= mu2_2;

  cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
  sigma12 -= mu1_mu2;

  cv::Mat ssim_map;
  cv::Mat numerator = (2 * mu1_mu2 + C1).mul(2 * sigma12 + C2);
  cv::Mat denominator = (mu1_2 + mu2_2 + C1).mul(sigma1_2 + sigma2_2 + C2);
  cv::divide(numerator, denominator, ssim_map);

  return cv::mean(ssim_map);
}

double evaluate_image_quality(const Mat &original, const Mat &stego) {
  if (original.size() != stego.size()) {
    throw runtime_error("Images must have same size");
  }

  // 计算PSNR
  double psnr = PSNR(original, stego);

  // 计算SSIM
  Mat gray1, gray2;
  cvtColor(original, gray1, COLOR_BGR2GRAY);
  cvtColor(stego, gray2, COLOR_BGR2GRAY);

  Scalar mssim = MSSIM(gray1, gray2);
  double ssim = mssim[0];

  // 返回综合评分
  return (psnr * 0.6 + ssim * 0.4);
}

// 新增：鲁棒性测试实现
bool test_robustness(const Mat &stego, const string &message,
                     const StegoConfig &config) {
  vector<Mat> attacked_images;

  // 添加高斯噪声
  Mat noisy = stego.clone();
  randn(noisy, 0, 25);
  attacked_images.push_back(noisy);

  // JPEG压缩
  vector<uchar> buffer;
  vector<int> params = {IMWRITE_JPEG_QUALITY, 75};
  imencode(".jpg", stego, buffer, params);
  attacked_images.push_back(imdecode(buffer, IMREAD_COLOR));

  // 旋转
  Mat rotated;
  Point2f center(stego.cols / 2.0f, stego.rows / 2.0f);
  Mat rotation = getRotationMatrix2D(center, 1.0, 1.0);
  warpAffine(stego, rotated, rotation, stego.size());
  attacked_images.push_back(rotated);

  // 测试每个攻击后的提取
  for (const auto &img : attacked_images) {
    string extracted = extract_message(img, message.length(), config);
    if (extracted != message) {
      return false;
    }
  }

  return true;
}

// 新增：压缩率估算实现
double estimate_compression_ratio(const Mat &image, const StegoConfig &config) {
  // 原始大小
  size_t original_size = image.total() * image.elemSize();

  // 压缩后大小估算
  vector<uchar> buffer;
  vector<int> params = {IMWRITE_JPEG_QUALITY, 95};
  imencode(".jpg", image, buffer, params);

  return static_cast<double>(buffer.size()) / original_size;
}

double calculateSimilarity(const cv::Mat &img1, const cv::Mat &img2, 
                         std::vector<double> *metrics) {
    if (img1.empty() || img2.empty() || img1.size() != img2.size()) {
        throw std::runtime_error("Invalid images for similarity calculation");
    }

    // 计算MSE (Mean Square Error)
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff = diff.mul(diff);
    cv::Scalar mse = cv::mean(diff);
    double mseScore = 1.0 - std::min(1.0, (mse[0] + mse[1] + mse[2]) / (3 * 255 * 255));

    // 计算PSNR (Peak Signal-to-Noise Ratio)
    double psnr = cv::PSNR(img1, img2);
    double psnrScore = std::min(1.0, psnr / 50.0); // 归一化到0-1范围

    // 计算SSIM (Structural Similarity Index)
    cv::Scalar ssimScalar = MSSIM(img1, img2);
    double ssimScore = ssimScalar[0];

    // 计算直方图相似度
    std::vector<cv::Mat> hist1, hist2;
    cv::split(img1, hist1);
    cv::split(img2, hist2);
    double histScore = 0.0;
    for (int i = 0; i < 3; i++) {
        cv::Mat h1, h2;
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        cv::calcHist(&hist1[i], 1, 0, cv::Mat(), h1, 1, &histSize, &histRange);
        cv::calcHist(&hist2[i], 1, 0, cv::Mat(), h2, 1, &histSize, &histRange);
        cv::normalize(h1, h1, 0, 1, cv::NORM_MINMAX);
        cv::normalize(h2, h2, 0, 1, cv::NORM_MINMAX);
        histScore += cv::compareHist(h1, h2, cv::HISTCMP_CORREL);
    }
    histScore /= 3.0;

    // 如果需要返回各指标分数
    if (metrics) {
        metrics->clear();
        metrics->push_back(mseScore);
        metrics->push_back(psnrScore);
        metrics->push_back(ssimScore);
        metrics->push_back(histScore);
    }

    // 计算加权平均得分
    constexpr double w1 = 0.25; // MSE权重
    constexpr double w2 = 0.25; // PSNR权重
    constexpr double w3 = 0.30; // SSIM权重
    constexpr double w4 = 0.20; // Histogram权重

    return w1 * mseScore + w2 * psnrScore + w3 * ssimScore + w4 * histScore;
}