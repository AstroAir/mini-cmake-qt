#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <qrencode.h>
#include <thread>
#include <zbar.h>


using namespace cv;
using namespace std;
using namespace zbar;

// QR码配置结构体
struct QRConfig {
  int size = 256;                 // QR码大小
  int version = 0;                // QR码版本(0为自动)
  QRecLevel level = QR_ECLEVEL_L; // 纠错级别
  QRencodeMode mode = QR_MODE_8;  // 编码模式
  bool caseSensitive = 1;         // 大小写敏感
  double blendRatio = 0.3;        // 混合比例
  int adaptiveBlockSize = 11;     // 自适应阈值块大小
  double adaptiveC = 2.0;         // 自适应阈值常数
};

// 智能指针包装QRcode释放
struct QRCodeDeleter {
  void operator()(QRcode *qr) { QRcode_free(qr); }
};
using QRCodePtr = unique_ptr<QRcode, QRCodeDeleter>;

// 优化的QR码生成函数
Mat generate_qrcode(const string &data, const QRConfig &config) {
  // 输入验证
  if (data.empty()) {
    throw invalid_argument("Empty input data");
  }

  // 生成QR码
  QRCodePtr qr(QRcode_encodeString(data.c_str(), config.version, config.level,
                                   config.mode, config.caseSensitive));

  if (!qr) {
    throw runtime_error("QR code generation failed");
  }

  // 使用连续内存创建Mat
  Mat qrcode_mat(qr->width, qr->width, CV_8UC1, Scalar(255));

  // 优化像素访问
  uchar *ptr = qrcode_mat.ptr<uchar>(0);
  const int total_pixels = qr->width * qr->width;

#pragma omp parallel for
  for (int i = 0; i < total_pixels; ++i) {
    ptr[i] = (qr->data[i] & 1) ? 0 : 255;
  }

  // 使用INTER_NEAREST进行快速缩放
  Mat resized;
  resize(qrcode_mat, resized, Size(config.size, config.size), 0, 0,
         INTER_NEAREST);
  return resized;
}

// 优化的QR码嵌入函数
void embed_qrcode(Mat &host_image, const Mat &qrcode, Point position,
                  const QRConfig &config) {
  // 参数验证
  CV_Assert(!host_image.empty() && !qrcode.empty());
  CV_Assert(host_image.type() == CV_8UC3 || host_image.type() == CV_8UC1);

  // ROI边界检查
  Rect roi(position.x, position.y, qrcode.cols, qrcode.rows);
  if (!Rect(0, 0, host_image.cols, host_image.rows).contains(roi)) {
    throw out_of_range("QR code position out of bounds");
  }

  Mat host_gray;
  if (host_image.channels() == 3) {
    cvtColor(host_image, host_gray, COLOR_BGR2GRAY);
  } else {
    host_gray = host_image;
  }

  // 优化自适应阈值处理
  Mat qr_adjusted;
  adaptiveThreshold(qrcode, qr_adjusted, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                    THRESH_BINARY,
                    config.adaptiveBlockSize | 1, // 确保为奇数
                    config.adaptiveC);

  // 优化混合计算
  Mat background = host_gray(roi);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < background.rows; i++) {
    for (int j = 0; j < background.cols; j++) {
      background.at<uchar>(i, j) = saturate_cast<uchar>(
          background.at<uchar>(i, j) * (1 - config.blendRatio) +
          qr_adjusted.at<uchar>(i, j) * config.blendRatio);
    }
  }

  if (host_image.channels() == 3) {
    cvtColor(host_gray, host_image, COLOR_GRAY2BGR);
  } else {
    host_gray.copyTo(host_image);
  }
}

// 优化的QR码检测函数
string detect_qrcode(const Mat &image) {
  static mutex scanner_mutex; // 保护scanner共享访问
  static ImageScanner scanner;

  {
    lock_guard<mutex> lock(scanner_mutex);
    scanner.set_config(ZBAR_QRCODE, ZBAR_CFG_ENABLE, 1);
    scanner.set_config(ZBAR_QRCODE, ZBAR_CFG_X_DENSITY, 2);
    scanner.set_config(ZBAR_QRCODE, ZBAR_CFG_Y_DENSITY, 2);
  }

  // 图像预处理
  Mat gray;
  if (image.channels() == 3) {
    cvtColor(image, gray, COLOR_BGR2GRAY);
  } else {
    gray = image;
  }

  // 增强对比度
  Mat enhanced;
  equalizeHist(gray, enhanced);

  // 创建ZBar图像
  Image zbar_image(enhanced.cols, enhanced.rows, "Y800", enhanced.data,
                   enhanced.cols * enhanced.rows);

  string result;
  {
    lock_guard<mutex> lock(scanner_mutex);
    int n = scanner.scan(zbar_image);
    if (n > 0) {
      for (Image::SymbolIterator symbol = zbar_image.symbol_begin();
           symbol != zbar_image.symbol_end(); ++symbol) {
        result = symbol->get_data();
        break;
      }
    }
  }

  return result;
}

// 添加结果验证函数
bool verify_qrcode(const string &original, const string &detected) {
  if (detected.empty()) {
    return false;
  }
  return original == detected;
}