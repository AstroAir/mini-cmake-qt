#include "MSB.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// 定义 SIMD 向量长度常量
#if CV_SIMD128
constexpr int SIMD_VECTOR_LENGTH = v_uint8x16::nlanes;
#else
constexpr int SIMD_VECTOR_LENGTH = 16; // 降级方案
#endif

// 优化后的MSB平面提取
Mat extractMSBPlane(const Mat &image) {
  Mat gray;
  // 异步转换灰度图
  if (image.channels() > 1) {
    gray.create(image.size(), CV_8UC1);
    parallel_for_(Range(0, image.rows), [&](const Range &range) {
      for (int r = range.start; r < range.end; r++) {
        const uchar *src = image.ptr<uchar>(r);
        uchar *dst = gray.ptr<uchar>(r);
        for (int c = 0; c < image.cols; c++) {
          dst[c] =
              static_cast<uchar>(src[3 * c] * 0.114 + src[3 * c + 1] * 0.587 +
                                 src[3 * c + 2] * 0.299);
        }
      }
    });
  } else {
    gray = image;
  }

  // 使用SIMD优化的MSB提取
  Mat msbPlane(image.size(), CV_8UC1);
  const int width = static_cast<int>(gray.step);
  const uchar msbMask = 0x80;

  parallel_for_(Range(0, gray.rows), [&](const Range &range) {
    for (int r = range.start; r < range.end; r++) {
      const uchar *src = gray.ptr<uchar>(r);
      uchar *dst = msbPlane.ptr<uchar>(r);
      int c = 0;

      // SIMD向量化处理
#if CV_SIMD128
      v_uint8x16 vmask = v_setall_u8(msbMask);
      for (; c <= width - SIMD_VECTOR_LENGTH; c += SIMD_VECTOR_LENGTH) {
        v_uint8x16 vsrc = v_load(src + c);
        v_uint8x16 vmsb = v_and(vsrc, vmask);
        v_uint8x16 vres = v_shl<1>(vmsb);
        v_store(dst + c, vres);
      }
#endif

      // 处理剩余像素
      for (; c < width; c++) {
        dst[c] = (src[c] & msbMask) ? 255 : 0;
      }
    }
  });

  return msbPlane;
}

// 优化后的MSB修改函数
void modifyMSB(Mat &image, bool setToOne) {
  const uchar mask = setToOne ? 0x80 : 0x7F;
  const int total = static_cast<int>(image.total() * image.channels());

  parallel_for_(Range(0, total), [&](const Range &range) {
    uchar *data = image.ptr<uchar>();
    if (setToOne) {
      for (int i = range.start; i < range.end; i++) {
        data[i] |= mask;
      }
    } else {
      for (int i = range.start; i < range.end; i++) {
        data[i] &= mask;
      }
    }
  });
}
Mat MSBCompressor::compress(const Mat &image, int keepBits) {
  CV_Assert(keepBits >= 1 && keepBits <= 8);

  Mat compressed;
  image.copyTo(compressed);

  const uchar mask = 0xFF << (8 - keepBits);
  const int total =
      static_cast<int>(compressed.total() * compressed.channels());
  const int chunk_size = 4096;

  parallel_for_(Range(0, (total + chunk_size - 1) / chunk_size),
                [&](const Range &range) {
                  for (int chunk = range.start; chunk < range.end; chunk++) {
                    const int start = chunk * chunk_size;
                    const int end = min(start + chunk_size, total);

                    uchar *data = compressed.ptr<uchar>() + start;
#if CV_SIMD128
                    v_uint8x16 vmask = v_setall_u8(mask);

                    for (int i = 0; i < (end - start - SIMD_VECTOR_LENGTH);
                         i += SIMD_VECTOR_LENGTH) {
                      v_uint8x16 v = v_load(data + i);
                      v_store(data + i, v_and(v, vmask));
                    }
#endif

                    // 处理剩余像素
                    for (int i = ((end - start) / SIMD_VECTOR_LENGTH) *
                                 SIMD_VECTOR_LENGTH;
                         i < end - start; i++) {
                      data[i] &= mask;
                    }
                  }
                });

  return compressed;
}

// 批处理接口
vector<Mat> MSBCompressor::compressBatch(const vector<Mat> &images,
                                         int keepBits) {
  vector<Mat> results(images.size());

  parallel_for_(Range(0, static_cast<int>(images.size())),
                [&](const Range &range) {
                  for (int i = range.start; i < range.end; i++) {
                    results[i] = compress(images[i], keepBits);
                  }
                });

  return results;
}
