#include <immintrin.h> // for AVX2
#include <iostream>
#include <opencv2/opencv.hpp>


// 针对 CV_16U 类型数据的解拜尔实现（使用 OpenCV 内置函数）
// 参数 bayerCode 可设为 cv::COLOR_BayerBG2BGR、cv::COLOR_BayerRG2BGR
// 等，根据实际 Bayer 排列调整
cv::Mat debayer_ushort(const cv::Mat &raw, int bayerCode) {
  cv::Mat color;
  cv::demosaicing(raw, color, bayerCode);
  return color;
}

// 针对 CV_32F 类型数据的解拜尔实现（使用双线性插值，假定 Bayer 格式为 BGGR）
// 经过优化：使用 parallel_for_ 并利用行指针直接访问数据以提高性能
cv::Mat debayer_float(const cv::Mat &raw) {
  cv::Mat color(raw.rows, raw.cols, CV_32FC3, cv::Scalar(0, 0, 0));

  cv::parallel_for_(cv::Range(0, raw.rows), [&](const cv::Range &r) {
    for (int i = r.start; i < r.end; i++) {
      const float *inRow = raw.ptr<float>(i);
      cv::Vec3f *outRow = color.ptr<cv::Vec3f>(i);
      for (int j = 0; j < raw.cols; j++) {
        float R = 0.f, G = 0.f, B = 0.f;
        bool row_even = ((i & 1) == 0);
        bool col_even = ((j & 1) == 0);

        if (row_even && col_even) {
          // 蓝色像素
          B = inRow[j];
          float sumG = 0.f;
          int countG = 0;
          if (i - 1 >= 0) {
            sumG += raw.ptr<float>(i - 1)[j];
            countG++;
          }
          if (i + 1 < raw.rows) {
            sumG += raw.ptr<float>(i + 1)[j];
            countG++;
          }
          if (j - 1 >= 0) {
            sumG += inRow[j - 1];
            countG++;
          }
          if (j + 1 < raw.cols) {
            sumG += inRow[j + 1];
            countG++;
          }
          G = (countG > 0) ? (sumG / countG) : 0.f;

          float sumR = 0.f;
          int countR = 0;
          if (i - 1 >= 0 && j - 1 >= 0) {
            sumR += raw.ptr<float>(i - 1)[j - 1];
            countR++;
          }
          if (i - 1 >= 0 && j + 1 < raw.cols) {
            sumR += raw.ptr<float>(i - 1)[j + 1];
            countR++;
          }
          if (i + 1 < raw.rows && j - 1 >= 0) {
            sumR += raw.ptr<float>(i + 1)[j - 1];
            countR++;
          }
          if (i + 1 < raw.rows && j + 1 < raw.cols) {
            sumR += raw.ptr<float>(i + 1)[j + 1];
            countR++;
          }
          R = (countR > 0) ? (sumR / countR) : 0.f;
        } else if (!row_even && !col_even) {
          // 红色像素
          R = inRow[j];
          float sumG = 0.f;
          int countG = 0;
          if (i - 1 >= 0) {
            sumG += raw.ptr<float>(i - 1)[j];
            countG++;
          }
          if (i + 1 < raw.rows) {
            sumG += raw.ptr<float>(i + 1)[j];
            countG++;
          }
          if (j - 1 >= 0) {
            sumG += inRow[j - 1];
            countG++;
          }
          if (j + 1 < raw.cols) {
            sumG += inRow[j + 1];
            countG++;
          }
          G = (countG > 0) ? (sumG / countG) : 0.f;

          float sumB = 0.f;
          int countB = 0;
          if (i - 1 >= 0 && j - 1 >= 0) {
            sumB += raw.ptr<float>(i - 1)[j - 1];
            countB++;
          }
          if (i - 1 >= 0 && j + 1 < raw.cols) {
            sumB += raw.ptr<float>(i - 1)[j + 1];
            countB++;
          }
          if (i + 1 < raw.rows && j - 1 >= 0) {
            sumB += raw.ptr<float>(i + 1)[j - 1];
            countB++;
          }
          if (i + 1 < raw.rows && j + 1 < raw.cols) {
            sumB += raw.ptr<float>(i + 1)[j + 1];
            countB++;
          }
          B = (countB > 0) ? (sumB / countB) : 0.f;
        } else if (row_even && !col_even) {
          // 绿色像素（蓝色行）
          G = inRow[j];
          float sumR = 0.f;
          int countR = 0;
          if (j - 1 >= 0) {
            sumR += inRow[j - 1];
            countR++;
          }
          if (j + 1 < raw.cols) {
            sumR += inRow[j + 1];
            countR++;
          }
          R = (countR > 0) ? (sumR / countR) : 0.f;

          float sumB = 0.f;
          int countB = 0;
          if (i - 1 >= 0) {
            sumB += raw.ptr<float>(i - 1)[j];
            countB++;
          }
          if (i + 1 < raw.rows) {
            sumB += raw.ptr<float>(i + 1)[j];
            countB++;
          }
          B = (countB > 0) ? (sumB / countB) : 0.f;
        } else { // (!row_even && col_even)
          // 绿色像素（红色行）
          G = inRow[j];
          float sumR = 0.f;
          int countR = 0;
          if (i - 1 >= 0) {
            sumR += raw.ptr<float>(i - 1)[j];
            countR++;
          }
          if (i + 1 < raw.rows) {
            sumR += raw.ptr<float>(i + 1)[j];
            countR++;
          }
          R = (countR > 0) ? (sumR / countR) : 0.f;

          float sumB = 0.f;
          int countB = 0;
          if (j - 1 >= 0) {
            sumB += inRow[j - 1];
            countB++;
          }
          if (j + 1 < raw.cols) {
            sumB += inRow[j + 1];
            countB++;
          }
          B = (countB > 0) ? (sumB / countB) : 0.f;
        }
        // OpenCV默认通道顺序为 BGR
        outRow[j] = cv::Vec3f(B, G, R);
      }
    }
  });

  return color;
}

// 定义支持的 Bayer 模式
enum class BayerPattern { RGGB, BGGR, GRBG, GBRG };

// 高质量的边缘感知解拜尔实现
cv::Mat debayer_edge_aware(const cv::Mat &raw, BayerPattern pattern) {
  cv::Mat color(raw.rows, raw.cols, CV_32FC3);

  cv::parallel_for_(cv::Range(2, raw.rows - 2), [&](const cv::Range &range) {
    for (int y = range.start; y < range.end; y++) {
      const float *in_m2 = raw.ptr<float>(y - 2);
      const float *in_m1 = raw.ptr<float>(y - 1);
      const float *in = raw.ptr<float>(y);
      const float *in_p1 = raw.ptr<float>(y + 1);
      const float *in_p2 = raw.ptr<float>(y + 2);
      cv::Vec3f *out = color.ptr<cv::Vec3f>(y);

      for (int x = 2; x < raw.cols - 2; x++) {
        // 计算水平和垂直梯度
        float h_grad =
            std::abs(in[x + 1] - in[x - 1]) + std::abs(in_m1[x] - in_p1[x]);
        float v_grad =
            std::abs(in[x + 1] - in[x - 1]) + std::abs(in_m1[x] - in_p1[x]);

        // 根据梯度选择插值方向
        if (h_grad < v_grad * 0.5f) {
          // 水平插值
          out[x][1] = (in[x - 1] + in[x + 1]) * 0.5f;
        } else if (v_grad < h_grad * 0.5f) {
          // 垂直插值
          out[x][1] = (in_m1[x] + in_p1[x]) * 0.5f;
        } else {
          // 双向插值
          out[x][1] = (in[x - 1] + in[x + 1] + in_m1[x] + in_p1[x]) * 0.25f;
        }

        // 根据 Bayer 模式计算其他颜色分量
        switch (pattern) {
        case BayerPattern::RGGB:
          if ((y & 1) == 0) {    // R行
            if ((x & 1) == 0) {  // R位置
              out[x][2] = in[x]; // R
              out[x][0] =
                  (in_m1[x - 1] + in_m1[x + 1] + in_p1[x - 1] + in_p1[x + 1]) *
                  0.25f;                                  // B
            } else {                                      // G位置
              out[x][2] = (in[x - 1] + in[x + 1]) * 0.5f; // R
              out[x][0] = (in_m1[x] + in_p1[x]) * 0.5f;   // B
            }
          } else {                                        // G行
            if ((x & 1) == 0) {                           // G位置
              out[x][2] = (in_m1[x] + in_p1[x]) * 0.5f;   // R
              out[x][0] = (in[x - 1] + in[x + 1]) * 0.5f; // B
            } else {                                      // B位置
              out[x][2] =
                  (in_m1[x - 1] + in_m1[x + 1] + in_p1[x - 1] + in_p1[x + 1]) *
                  0.25f;         // R
              out[x][0] = in[x]; // B
            }
          }
          break;
          // ... 其他 Bayer 模式类似
        }
      }
    }
  });

  return color;
}

// AVX2 优化的解拜尔实现
#ifdef __AVX2__
cv::Mat debayer_avx2(const cv::Mat &raw, BayerPattern pattern) {
  cv::Mat color(raw.rows, raw.cols, CV_32FC3);

  // 每次处理 8 个像素
  const int simd_width = 8;

  cv::parallel_for_(cv::Range(2, raw.rows - 2), [&](const cv::Range &range) {
    for (int y = range.start; y < range.end; y++) {
      const float *in = raw.ptr<float>(y);
      cv::Vec3f *out = color.ptr<cv::Vec3f>(y);

      // SIMD 主循环
      for (int x = 2; x < raw.cols - 2 - simd_width; x += simd_width) {
        __m256 row = _mm256_loadu_ps(&in[x]);
        __m256 prev = _mm256_loadu_ps(&in[x - 1]);
        __m256 next = _mm256_loadu_ps(&in[x + 1]);

        // 计算绿色分量
        __m256 green =
            _mm256_mul_ps(_mm256_add_ps(prev, next), _mm256_set1_ps(0.5f));

        // 存储结果
        float g_buffer[8];
        _mm256_storeu_ps(g_buffer, green);

        for (int i = 0; i < simd_width; i++) {
          out[x + i][1] = g_buffer[i];
        }
      }

      // 处理剩余像素
      for (int x = raw.cols - 2 - simd_width; x < raw.cols - 2; x++) {
        // ...使用标准实现处理边缘像素
      }
    }
  });

  return color;
}
#endif

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "用法：debayer <image_path> <bayer_pattern>" << std::endl;
    std::cout << "bayer_pattern: rggb, bggr, grbg, gbrg" << std::endl;
    return -1;
  }

  // 以未改变数据格式读取图像（确保加载原始 Bayer 数据）
  cv::Mat raw = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
  if (raw.empty()) {
    std::cerr << "读取图像失败：" << argv[1] << std::endl;
    return -1;
  }

  // 解析 Bayer 模式
  std::string pattern_str = argv[2];
  BayerPattern pattern;
  if (pattern_str == "rggb")
    pattern = BayerPattern::RGGB;
  else if (pattern_str == "bggr")
    pattern = BayerPattern::BGGR;
  else if (pattern_str == "grbg")
    pattern = BayerPattern::GRBG;
  else if (pattern_str == "gbrg")
    pattern = BayerPattern::GBRG;
  else {
    std::cerr << "不支持的 Bayer 模式" << std::endl;
    return -1;
  }

  cv::Mat debayered;
  // 判断数据类型并调用相应的解拜尔函数
  if (raw.type() == CV_16U) {
    // 假定 Bayer 格式为 BGGR，根据实际情况可以选择 cv::COLOR_BayerRG2BGR 等
    debayered = debayer_ushort(raw, cv::COLOR_BayerBG2BGR);
  } else if (raw.type() == CV_32F) {
#ifdef __AVX2__
    // 使用 AVX2 优化版本
    debayered = debayer_avx2(raw, pattern);
#else
    // 使用边缘感知版本
    debayered = debayer_edge_aware(raw, pattern);
#endif
  } else {
    std::cerr << "不支持的数据类型，仅支持 CV_16U 和 CV_32F" << std::endl;
    return -1;
  }

  // 显示结果
  cv::namedWindow("Debayered Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Debayered Image", debayered);
  std::cout << "按任意键退出..." << std::endl;
  cv::waitKey(0);

  // 保存处理后图像
  cv::imwrite("debayered_output.png", debayered);

  return 0;
}