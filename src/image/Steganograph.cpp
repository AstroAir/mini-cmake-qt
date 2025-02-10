#include <bitset>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

// ================== LSB隐写术 ==================
// 将字符串嵌入到图像LSB中
void embedLSB(Mat &image, const string &message) {
  // 转换为二进制数据
  vector<bool> bits;
  for (char c : message) {
    bitset<8> charBits(c);
    for (int i = 7; i >= 0; --i) {
      bits.push_back(charBits[i]);
    }
  }
  bits.push_back(false); // 终止符

  // 检查容量
  const int maxBits = image.rows * image.cols * image.channels();
  if (bits.size() > maxBits) {
    throw runtime_error("Message too large for image capacity");
  }

  // 嵌入数据
  auto it = bits.begin();
  image.forEach<Vec3b>([&](Vec3b &pixel, const int *) {
    for (int c = 0; c < 3; ++c) { // 处理BGR三个通道
      if (it != bits.end()) {
        pixel[c] = (pixel[c] & 0xFE) | *it++;
      }
    }
  });
}

// 从图像LSB中提取信息
string extractLSB(const Mat &image) {
  vector<bool> bits;
  image.forEach<Vec3b>([&](const Vec3b &pixel, const int *) {
    for (int c = 0; c < 3; ++c) {
      bits.push_back(pixel[c] & 1);
    }
  });

  // 转换为字符
  string message;
  for (size_t i = 0; i < bits.size(); i += 8) {
    if (i + 8 > bits.size())
      break;

    bitset<8> charBits;
    for (int j = 0; j < 8; ++j) {
      charBits[7 - j] = bits[i + j];
    }

    char c = static_cast<char>(charBits.to_ulong());
    if (c == '\0')
      break; // 终止符
    message += c;
  }
  return message;
}

// ================== 位平面分解 ==================
Mat getBitPlane(const Mat &image, int bitPosition) {
  CV_Assert(image.depth() == CV_8U);
  Mat plane(image.size(), CV_8UC1);

  const uchar mask = 1 << bitPosition;
  image.forEach<uchar>([&](const uchar pixel, const int *) {
    plane.ptr<uchar>(&pixel - image.data)[0] =
        ((pixel & mask) >> bitPosition) * 255;
  });

  return plane;
}

// ================== 示例使用 ==================
int main() {
  // 1. 读取图像
  Mat image = imread("input.png");
  if (image.empty()) {
    cerr << "Error loading image" << endl;
    return -1;
  }

  // 2. LSB隐写示例
  string secretMsg = "DeepSeek-R1";
  Mat embeddedImage = image.clone();
  embedLSB(embeddedImage, secretMsg);
  imwrite("embedded.png", embeddedImage);

  // 3. 提取信息
  string extracted = extractLSB(embeddedImage);
  cout << "Extracted message: " << extracted << endl;

  // 4. 显示位平面
  Mat msbPlane = getBitPlane(image, 7); // MSB平面
  Mat lsbPlane = getBitPlane(image, 0); // LSB平面

  imshow("Original", image);
  imshow("MSB Plane", msbPlane);
  imshow("LSB Plane", lsbPlane);
  waitKey(0);

  return 0;
}

#include <atomic>
#include <bit>
#include <execution>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>
#include <zlib.h>


// IHDR块结构（13字节）
struct IHDRData {
  uint32_t width;
  uint32_t height;
  uint8_t bit_depth;
  uint8_t color_type;
  uint8_t compression;
  uint8_t filter;
  uint8_t interlace;
};

// 预计算CRC表（加速计算）
constexpr auto generate_crc_table() {
  std::array<uint32_t, 256> table{};
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t crc = i;
    for (int j = 0; j < 8; ++j) {
      crc = (crc >> 1) ^ ((crc & 1) * 0xEDB88320);
    }
    table[i] = crc;
  }
  return table;
}

static constexpr auto crc_table = generate_crc_table();

// 快速CRC32计算（使用预计算表）
inline uint32_t fast_crc32(const uint8_t *data, size_t length,
                           uint32_t crc = 0) {
  crc = ~crc;
  for (size_t i = 0; i < length; ++i) {
    crc = crc_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  }
  return ~crc;
}

// 并行爆破核心
std::pair<int, int>
brute_force_crc(const IHDRData &original_ihdr, uint32_t target_crc,
                int max_dim = 5000,
                int num_threads = std::thread::hardware_concurrency()) {
  std::atomic<bool> found(false);
  std::pair<int, int> result(-1, -1);

  // 并行任务划分
  auto task = [&](int thread_id) {
    for (int w = thread_id; w <= max_dim && !found; w += num_threads) {
      for (int h = 1; h <= max_dim; ++h) {
        if (found)
          break;

        // 构建IHDR数据块
        IHDRData modified = original_ihdr;
        modified.width = std::byteswap(static_cast<uint32_t>(w));
        modified.height = std::byteswap(static_cast<uint32_t>(h));

        // 计算CRC
        const auto *bytes = reinterpret_cast<const uint8_t *>(&modified);
        uint32_t crc = fast_crc32(
            bytes, sizeof(IHDRData),
            fast_crc32(reinterpret_cast<const uint8_t *>("IHDR"), 4));

        if (crc == target_crc) {
          result = {w, h};
          found.store(true);
          return;
        }
      }
    }
  };

  // 启动线程池
  std::vector<std::jthread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(task, i);
  }

  for (auto &t : threads)
    t.join();
  return result;
}

// 主流程
int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input.png>\n";
    return 1;
  }

  // 读取PNG文件头
  std::ifstream file(argv[1], std::ios::binary);
  if (!file) {
    std::cerr << "Error opening file\n";
    return 1;
  }

  // 定位IHDR块（跳过PNG签名和块长度）
  file.seekg(8 + 4, std::ios::beg);

  IHDRData ihdr;
  file.read(reinterpret_cast<char *>(&ihdr), sizeof(IHDRData));

  // 读取原始CRC
  uint32_t original_crc;
  file.read(reinterpret_cast<char *>(&original_crc), 4);
  original_crc = std::byteswap(original_crc);

  // 执行爆破
  auto [width, height] = brute_force_crc(ihdr, original_crc);

  if (width != -1) {
    std::cout << "Found valid dimensions: " << width << "x" << height << "\n";

    // 使用OpenCV验证
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    if (!img.empty()) {
      cv::imshow("Recovered Image", img);
      cv::waitKey(0);
    }
  } else {
    std::cout << "No valid dimensions found\n";
  }

  return 0;
}

#include <bitset>
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

  // 频域移位
  int cx = complex.cols / 2;
  int cy = complex.rows / 2;
  Mat q0(complex, Rect(0, 0, cx, cy));
  Mat q1(complex, Rect(cx, 0, cx, cy));
  Mat q2(complex, Rect(0, cy, cx, cy));
  Mat q3(complex, Rect(cx, cy, cx, cy));
  Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);

  // 生成相同坐标序列
  vector<bool> extracted_bits;
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

  // 提取数据
  for (const auto &[y, x, r] : coord_view) {
    if (extracted_bits.size() >= msg_length * 8)
      break;

    Vec2f pixel = complex.at<Vec2f>(y, x);
    float magnitude = norm(pixel);
    bool bit = (magnuth > alpha / 2); // 简单阈值解码
    extracted_bits.push_back(bit);
  }

  return bits_to_str(extracted_bits);
}

int main() {
  try {
    // 嵌入示例
    Mat carrier = imread("carrier.jpg");
    if (carrier.empty())
      throw runtime_error("Failed to load carrier image");

    const string secret_msg = "DeepSeek-R1";
    Mat stego = embed_message(carrier, secret_msg, 0.15);
    imwrite("stego_image.png", stego);

    // 提取示例
    Mat loaded_stego = imread("stego_image.png");
    string extracted = extract_message(loaded_stego, secret_msg.length());
    cout << "Extracted message: " << extracted << endl;
  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }
  return 0;
}

#include <opencv2/opencv.hpp>
#include <vector>
#include <bitset>
#include <stdexcept>

using namespace cv;
using namespace std;

// Alpha通道隐写嵌入函数
void alpha_channel_hide(Mat& image, const string& message) {
    if (image.channels() != 4) {
        throw runtime_error("Image does not have alpha channel");
    }

    // 将消息转换为二进制流
    vector<bool> bits;
    for (char c : message) {
        bitset<8> bs(c);
        for (int i=7; i>=0; --i) {
            bits.push_back(bs[i]);
        }
    }

    // 添加终止符
    for (int i=0; i<8; ++i) {
        bits.push_back(false);
    }

    // 检查容量
    int max_bits = image.rows * image.cols;
    if (bits.size() > max_bits) {
        throw runtime_error("Message too large for image capacity");
    }

    // 修改Alpha通道LSB
    int bit_counter = 0;
    for (int y=0; y<image.rows; ++y) {
        for (int x=0; x<image.cols; ++x) {
            Vec4b& pixel = image.at<Vec4b>(y, x);
            if (bit_counter < bits.size()) {
                pixel[3] = (pixel[3] & 0xFE) | bits[bit_counter];
                ++bit_counter;
            } else {
                return;
            }
        }
    }
}

// Alpha通道隐写提取函数
string alpha_channel_extract(const Mat& image) {
    if (image.channels() != 4) {
        throw runtime_error("Image does not have alpha channel");
    }

    vector<bool> bits;
    for (int y=0; y<image.rows; ++y) {
        for (int x=0; x<image.cols; ++x) {
            const Vec4b& pixel = image.at<Vec4b>(y, x);
            bits.push_back(pixel[3] & 1);
        }
    }

    // 转换为字节
    string result;
    for (size_t i=0; i<bits.size(); i+=8) {
        if (i+8 > bits.size()) break;
        
        char c = 0;
        for (int j=0; j<8; ++j) {
            c |= bits[i+j] << (7-j);
        }
        
        // 检查终止符
        if (c == '\0') break;
        result += c;
    }
    return result;
}

// 通道分离与异常检测
void analyze_channels(const Mat& image) {
    vector<Mat> channels;
    split(image, channels);

    // 显示各通道
    const string names[] = {"Blue", "Green", "Red", "Alpha"};
    for (int i=0; i<image.channels(); ++i) {
        Mat display;
        normalize(channels[i], display, 0, 255, NORM_MINMAX);
        imshow(names[i] + " Channel", display);
    }
    waitKey(0);
}

int main() {
    try {
        // 示例使用
        Mat img = imread("input.png", IMREAD_UNCHANGED);
        if (img.empty()) throw runtime_error("Failed to load image");

        // 添加Alpha通道（如果不存在）
        if (img.channels() == 3) {
            cvtColor(img, img, COLOR_BGR2BGRA);
        }

        // 隐写操作
        string secret = "CTF{Steg@Alpha}";
        alpha_channel_hide(img, secret);
        imwrite("hidden.png", img);

        // 提取测试
        Mat hidden_img = imread("hidden.png", IMREAD_UNCHANGED);
        string extracted = alpha_channel_extract(hidden_img);
        cout << "Extracted message: " << extracted << endl;

        // 通道分析
        analyze_channels(hidden_img);

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// 水印嵌入函数
Mat embedWatermark(const Mat& host, const Mat& watermark, double alpha = 0.1) {
    Mat hostYUV;
    cvtColor(host, hostYUV, COLOR_BGR2YUV);
    vector<Mat> channels;
    split(hostYUV, channels);
    Mat Y = channels[0];

    // 调整水印大小并二值化
    Mat wm;
    resize(watermark, wm, Y.size());
    cvtColor(wm, wm, COLOR_BGR2GRAY);
    threshold(wm, wm, 128, 1, THRESH_BINARY);

    Mat blended = Y.clone();
    const int blockSize = 8;
    
    for(int i=0; i<Y.rows; i+=blockSize) {
        for(int j=0; j<Y.cols; j+=blockSize) {
            Rect roi(j, i, blockSize, blockSize);
            if(roi.br().x > Y.cols || roi.br().y > Y.rows) continue;
            
            Mat block = Y(roi).clone();
            block.convertTo(block, CV_32F);
            
            // DCT变换
            dct(block, block);
            
            // 嵌入水印信息（修改中频系数）
            if(wm.at<uchar>(i,j) > 0) {
                block.at<float>(3,3) += alpha * block.at<float>(0,0);
                block.at<float>(4,4) += alpha * block.at<float>(0,0);
            }
            
            // 逆DCT
            idct(block, block);
            block.convertTo(block, CV_8U);
            
            block.copyTo(blended(roi));
        }
    }

    channels[0] = blended;
    merge(channels, hostYUV);
    Mat result;
    cvtColor(hostYUV, result, COLOR_YUV2BGR);
    return result;
}

// 水印提取函数
Mat extractWatermark(const Mat& watermarked, double alpha = 0.1, int wmSize = 64) {
    Mat yuv;
    cvtColor(watermarked, yuv, COLOR_BGR2YUV);
    vector<Mat> channels;
    split(yuv, channels);
    Mat Y = channels[0];

    Mat wm = Mat::zeros(Y.size(), CV_32F);
    const int blockSize = 8;
    
    for(int i=0; i<Y.rows; i+=blockSize) {
        for(int j=0; j<Y.cols; j+=blockSize) {
            Rect roi(j, i, blockSize, blockSize);
            if(roi.br().x > Y.cols || roi.br().y > Y.rows) continue;
            
            Mat block = Y(roi).clone();
            block.convertTo(block, CV_32F);
            dct(block, block);
            
            // 提取水印信息
            float diff = (block.at<float>(3,3) + block.at<float>(4,4)) / 2;
            wm.at<float>(i,j) = diff / (alpha * block.at<float>(0,0));
        }
    }
    
    // 后处理
    Mat resized;
    resize(wm, resized, Size(wmSize, wmSize));
    normalize(resized, resized, 0, 255, NORM_MINMAX);
    resized.convertTo(resized, CV_8U);
    
    return resized;
}

int main() {
    // 示例使用
    Mat host = imread("host_image.jpg");
    Mat watermark = imread("watermark.png");
    
    if(host.empty() || watermark.empty()) {
        cerr << "Error loading images!" << endl;
        return -1;
    }

    // 嵌入水印
    Mat watermarked = embedWatermark(host, watermark, 0.15);
    imwrite("watermarked.jpg", watermarked);

    // 提取水印
    Mat extracted = extractWatermark(watermarked, 0.15, watermark.rows);
    imwrite("extracted_watermark.png", extracted);

    return 0;
}

#include <opencv2/opencv.hpp>
#include <qrencode.h>
#include <zbar.h>

using namespace cv;
using namespace std;
using namespace zbar;

// 生成二维码图像
Mat generate_qrcode(const string& data, int size) {
    QRcode* qr = QRcode_encodeString(data.c_str(), 0, QR_ECLEVEL_L, QR_MODE_8, 1);
    if (!qr) {
        throw runtime_error("QR code generation failed");
    }

    // 创建OpenCV矩阵
    Mat qrcode_mat(qr->width, qr->width, CV_8UC1, Scalar(255));
    for (int y = 0; y < qr->width; ++y) {
        for (int x = 0; x < qr->width; ++x) {
            if (qr->data[y * qr->width + x] & 1) {
                qrcode_mat.at<uchar>(y, x) = 0; // 黑色像素
            }
        }
    }

    QRcode_free(qr);
    
    // 调整大小并保持清晰
    Mat resized;
    resize(qrcode_mat, resized, Size(size, size), 0, 0, INTER_NEAREST);
    return resized;
}

// 将二维码嵌入到宿主图像
void embed_qrcode(Mat& host_image, const Mat& qrcode, Point position) {
    // 转换为灰度图处理
    Mat host_gray;
    cvtColor(host_image, host_gray, COLOR_BGR2GRAY);
    
    // 创建ROI区域
    Rect roi(position.x, position.y, qrcode.cols, qrcode.rows);
    if (roi.x + roi.width > host_gray.cols || roi.y + roi.height > host_gray.rows) {
        throw runtime_error("QR code position out of bounds");
    }

    // 使用自适应阈值增强嵌入效果
    Mat qr_adjusted;
    adaptiveThreshold(qrcode, qr_adjusted, 255, ADAPTIVE_THRESH_GAUSSIAN_C, 
                     THRESH_BINARY, 11, 2);

    // 混合嵌入（保留宿主图像高频信息）
    Mat background = host_gray(roi);
    addWeighted(background, 0.7, qr_adjusted, 0.3, 0, background);
    
    // 合并回原图
    cvtColor(host_gray, host_image, COLOR_GRAY2BGR);
}

// 使用ZBar检测二维码
string detect_qrcode(const Mat& image) {
    ImageScanner scanner;
    scanner.set_config(ZBAR_QRCODE, ZBAR_CFG_ENABLE, 1);

    // 转换为ZBar兼容格式
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Image zbar_image(gray.cols, gray.rows, "Y800", 
                    gray.data, gray.cols * gray.rows);

    // 扫描图像
    if (scanner.scan(zbar_image) < 0) {
        return "";
    }

    // 获取结果
    for (auto symbol = zbar_image.symbol_begin(); 
         symbol != zbar_image.symbol_end(); ++symbol) {
        return symbol->get_data();
    }
    return "";
}

int main() {
    try {
        // 1. 生成宿主图像
        Mat host = imread("background.jpg");
        if (host.empty()) {
            host = Mat(800, 600, CV_8UC3, Scalar(255, 255, 255));
        }

        // 2. 生成二维码
        Mat qrcode = generate_qrcode("flag{hidden_qrcode_123}", 50);

        // 3. 嵌入到右下角
        Point position(host.cols - qrcode.cols - 10, 
                      host.rows - qrcode.rows - 10);
        embed_qrcode(host, qrcode, position);

        // 4. 保存结果
        imwrite("output.jpg", host);
        cout << "QR code embedded successfully\n";

        // 5. 检测验证
        Mat test_image = imread("output.jpg");
        string result = detect_qrcode(test_image);
        if (!result.empty()) {
            cout << "Detected QR code: " << result << endl;
        } else {
            cout << "No QR code detected" << endl;
        }
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}