#include "WaterMark.h"
#include <atomic>
#include <execution>
#include <filesystem>
#include <functional>
#include <future>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <ranges>
#include <thread>
#include <variant>

using namespace cv;
namespace fs = std::filesystem;

// 水印类型定义
enum class WatermarkType {
  Magnitude, // 幅度调制（默认）
  Phase,     // 相位调制
  Complex,   // 复数域嵌入
  QRCode,    // QR码水印
  Diffused   // 扩散水印
};

// 水印参数结构体
struct WatermarkParams {
  float alpha = 0.15f;        // 强度系数
  int qr_version = 5;         // QR码版本
  uint32_t seed = 0xDEADBEEF; // 随机种子
  bool anti_aliasing = true;  // 抗锯齿
};

// 前向声明辅助函数
void optimized_fft_shift(Mat &mag) noexcept {
  const int cx = mag.cols / 2;
  const int cy = mag.rows / 2;

  auto swap_block = [](float *a, float *b, int block_size) {
    for (int i = 0; i < block_size; ++i) {
      std::swap(a[i], b[i]);
    }
  };

  float *data = mag.ptr<float>();
  const int row_step = static_cast<int>(mag.step1());

  // 并行交换四个象限
#pragma omp parallel sections
  {
#pragma omp section
    {
      swap_block(data, data + cy * row_step + cx, cx * cy);
    }
#pragma omp section
    {
      swap_block(data + cx, data + cy * row_step, cx * cy);
    }
  }
}

// 并行嵌入核心实现
Mat parallel_embed(Mat &magnitude, const Mat &watermark, float alpha,
                   Rect center) {
  Mat modified_mag = magnitude.clone();
  Mat roi = modified_mag(center);

  // 使用 C++17 并行算法处理
  std::for_each(std::execution::par_unseq, roi.begin<float>(), roi.end<float>(),
                [start = watermark.begin<float>(), alpha](
                    float &mag_val) mutable { mag_val += alpha * (*start++); });

  optimized_fft_shift(modified_mag);
  return modified_mag;
}

// 傅里叶水印嵌入类
class FourierWatermarker {
public:
  explicit FourierWatermarker(WatermarkType type = WatermarkType::Magnitude,
                              WatermarkParams params = {})
      : type_(type), params_(params) {
    init_strategies();
  }

  Mat embed(const Mat &src, const Mat &watermark) {
    return strategy_(src, watermark);
  }

  // QR码专用接口
  Mat embedQR(const Mat &src, const std::string &text) {
    Mat qr = generateQRCode(text);
    return qrEmbed(src, qr);
  }

private:
  // 根据水印类型初始化对应策略
  void init_strategies() {
    switch (type_) {
    case WatermarkType::Phase:
      strategy_ = [this](const Mat &s, const Mat &w) {
        return phaseEmbed(s, w);
      };
      break;
    case WatermarkType::Complex:
      strategy_ = [this](const Mat &s, const Mat &w) {
        return complexEmbed(s, w);
      };
      break;
    case WatermarkType::QRCode:
      strategy_ = [this](const Mat &s, const Mat &w) { return qrEmbed(s, w); };
      break;
    case WatermarkType::Diffused:
      strategy_ = [this](const Mat &s, const Mat &w) {
        return diffuseEmbed(s, w);
      };
      break;
    default:
      strategy_ = [this](const Mat &s, const Mat &w) {
        return magnitudeEmbed(s, w);
      };
    }
  }

  // 基础幅度嵌入实现
  Mat magnitudeEmbed(const Mat &src, const Mat &watermark) {
    // ... 基础幅度嵌入实现 ...
    return src;
  }

  // 相位嵌入实现（示例代码，具体实现请补充）
  Mat phaseEmbed(const Mat &src, const Mat &watermark) {
    // 示例：分离通道、修改相位后合成
    // ... 相位嵌入实现 ...
    return src;
  }

  // 复数域联合嵌入实现（示例代码，具体实现请补充）
  Mat complexEmbed(const Mat &src, const Mat &watermark) {
    // ... 复数域联合嵌入实现 ...
    return src;
  }

  // QR码嵌入实现（示例代码，具体实现请补充）
  Mat qrEmbed(const Mat &src, const Mat &qr) {
    // QR码预处理：缩放到目标嵌入区域
    Mat qr_resized;
    Rect roi(src.cols / 2 - qr.cols / 2, src.rows / 2 - qr.rows / 2, qr.cols,
             qr.rows);
    resize(qr, qr_resized, roi.size(), 0, 0,
           params_.anti_aliasing ? INTER_AREA : INTER_NEAREST);
    // ... 高频区域嵌入实现 ...
    return src;
  }

  // 扩散水印生成实现（示例代码，具体实现请补充）
  Mat diffuseEmbed(const Mat &src, const Mat &watermark) {
    // 生成扩散水印
    Mat spread_wm = spreadSpectrum(watermark);
    // 随机频点嵌入
    RNG rng(params_.seed);
    // ... 扩散水印嵌入实现 ...
    return src;
  }

  // 生成QR码（示例代码，依赖OpenCV内置QR生成器）
  Mat generateQRCode(const std::string &text) {
    // 示例使用伪代码，具体实现需依靠实际QR生成接口
    // QRCodeEncoder::Params qrParams;
    // qrParams.version = params_.qr_version;
    // Ptr<QRCodeEncoder> encoder = QRCodeEncoder::create(qrParams);
    // Mat qr;
    // encoder->encode(text, qr);
    // return qr;
    return Mat();
  }

  // 扩频调制实现
  Mat spreadSpectrum(const Mat &watermark) {
    Mat spread;
    dft(watermark, spread, DFT_COMPLEX_OUTPUT);
    cv::RNG rng(params_.seed);
    cv::randShuffle(spread, 1.0, &rng);
    return spread;
  }

  WatermarkType type_;
  WatermarkParams params_;
  std::function<Mat(const Mat &, const Mat &)> strategy_;
};

// 高频优化版傅里叶水印嵌入函数
Mat optimized_embed(const Mat &src, const Mat &watermark, float alpha = 0.15f) {
  CV_Assert(src.type() == CV_8UC3 && !src.empty());

  // 异步流水线处理
  Mat gray_src, padded;
  std::mutex mtx;
  std::thread([&] {
    Mat tmp;
    cvtColor(src, tmp, COLOR_BGR2GRAY);
    Size opt_size(getOptimalDFTSize(src.cols), getOptimalDFTSize(src.rows));
    copyMakeBorder(tmp, padded, 0, opt_size.height - src.rows, 0,
                   opt_size.width - src.cols, BORDER_CONSTANT);
  }).join();

  // 原地DFT优化
  Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
  Mat complex_img;
  merge(planes, 2, complex_img);
  dft(complex_img, complex_img, DFT_COMPLEX_OUTPUT);

  // 并行频谱处理
  Mat magnitude, phase;
  {
    std::vector<Mat> planess(2);
    split(complex_img, planess);
    cartToPolar(planess[0], planess[1], magnitude, phase);
    optimized_fft_shift(magnitude);
  }

  // 水印预处理：并行缩放
  Mat scaled_watermark;
  std::atomic<bool> resize_done{false};
  std::thread([&] {
    resize(watermark, scaled_watermark, magnitude.size(), 0, 0, INTER_AREA);
    scaled_watermark.convertTo(scaled_watermark, CV_32F, 1.0 / 255);
    resize_done.store(true, std::memory_order_release);
  }).detach();

  // 准备嵌入区域（居中区域）
  Rect center(magnitude.cols / 2 - watermark.cols / 2,
              magnitude.rows / 2 - watermark.rows / 2, watermark.cols,
              watermark.rows);

  // 等待水印预处理完成
  while (!resize_done.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  // 并行嵌入核心
  Mat modified_mag = parallel_embed(magnitude, scaled_watermark, alpha, center);

  // 逆变换优化
  Mat inverse_transform;
  {
    Mat planes[2];
    polarToCart(modified_mag, phase, planes[0], planes[1]);
    merge(planes, 2, complex_img);
    idft(complex_img, inverse_transform, DFT_REAL_OUTPUT | DFT_SCALE);
  }

  // 异步结果裁剪
  Mat output;
  std::async(std::launch::async, [&] {
    inverse_transform(Rect(0, 0, src.cols, src.rows)).convertTo(output, CV_8U);
  }).wait();

  return output;
}