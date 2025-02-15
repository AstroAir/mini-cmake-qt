#include "WaterMark.hpp"

#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <opencv2/cudaarithm.hpp>
#endif

using namespace cv;
using namespace std;

Mat embedWatermark(const Mat &host, const Mat &watermark, double alpha) {
  Mat hostYUV;
  cvtColor(host, hostYUV, COLOR_BGR2YUV);
  vector<Mat> channels;
  split(hostYUV, channels);
  Mat Y = channels[0];

  Mat wm;
  resize(watermark, wm, Y.size());
  cvtColor(wm, wm, COLOR_BGR2GRAY);
  threshold(wm, wm, 128, 1, THRESH_BINARY);

  Mat blended = Y.clone();
  const int blockSize = 8;
  int rows = Y.rows;
  int cols = Y.cols;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
  for (int i = 0; i < rows; i += blockSize) {
    for (int j = 0; j < cols; j += blockSize) {
      if (i + blockSize > rows || j + blockSize > cols)
        continue;
      Rect roi(j, i, blockSize, blockSize);
      Mat block = Y(roi);
      Mat blockF;
      block.convertTo(blockF, CV_32F);

      // DCT变换
      dct(blockF, blockF);

      // 嵌入水印信息（修改中频系数）
      if (wm.at<uchar>(i, j) > 0) {
        blockF.at<float>(3, 3) += alpha * blockF.at<float>(0, 0);
        blockF.at<float>(4, 4) += alpha * blockF.at<float>(0, 0);
      }

      // 逆DCT
      idct(blockF, blockF);
      blockF.convertTo(block, CV_8U);
      block.copyTo(blended(roi));
    }
  }

  channels[0] = blended;
  merge(channels, hostYUV);
  Mat result;
  cvtColor(hostYUV, result, COLOR_YUV2BGR);
  return result;
}

Mat embedWatermarkMultiChannel(const Mat &host, const Mat &watermark,
                               const vector<int> &channelsToEmbed,
                               double alpha) {
  Mat hostYUV;
  cvtColor(host, hostYUV, COLOR_BGR2YUV);
  vector<Mat> yuvChannels;
  split(hostYUV, yuvChannels);

  // 调整水印大小并二值化
  Mat wm;
  resize(watermark, wm, yuvChannels[0].size());
  cvtColor(wm, wm, COLOR_BGR2GRAY);
  threshold(wm, wm, 128, 1, THRESH_BINARY);

  const int blockSize = 8;
  int rows = yuvChannels[0].rows;
  int cols = yuvChannels[0].cols;

  for (int ch : channelsToEmbed) {
    if (ch >= yuvChannels.size())
      continue;

    Mat channelEmbed = yuvChannels[ch].clone();
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (int i = 0; i < rows; i += blockSize) {
      for (int j = 0; j < cols; j += blockSize) {
        if (i + blockSize > rows || j + blockSize > cols)
          continue;
        Rect roi(j, i, blockSize, blockSize);
        Mat block = channelEmbed(roi);
        Mat blockF;
        block.convertTo(blockF, CV_32F);
        dct(blockF, blockF);
        if (wm.at<uchar>(i, j) > 0) {
          blockF.at<float>(3, 3) += alpha * blockF.at<float>(0, 0);
        }
        idct(blockF, blockF);
        blockF.convertTo(block, CV_8U);
        block.copyTo(channelEmbed(roi));
      }
    }
    yuvChannels[ch] = channelEmbed;
  }
  merge(yuvChannels, hostYUV);
  Mat result;
  cvtColor(hostYUV, result, COLOR_YUV2BGR);
  return result;
}

Mat extractWatermark(const Mat &watermarked, double alpha, int wmSize) {
  Mat yuv;
  cvtColor(watermarked, yuv, COLOR_BGR2YUV);
  vector<Mat> channels;
  split(yuv, channels);
  Mat Y = channels[0];

  Mat wm = Mat::zeros(Y.size(), CV_32F);
  const int blockSize = 8;
  int rows = Y.rows;
  int cols = Y.cols;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
  for (int i = 0; i < rows; i += blockSize) {
    for (int j = 0; j < cols; j += blockSize) {
      if (i + blockSize > rows || j + blockSize > cols)
        continue;
      Rect roi(j, i, blockSize, blockSize);
      Mat block = Y(roi);
      Mat blockF;
      block.convertTo(blockF, CV_32F);
      dct(blockF, blockF);

      // 提取水印信息
      float diff = (blockF.at<float>(3, 3) + blockF.at<float>(4, 4)) / 2;
      wm.at<float>(i, j) = diff / (alpha * blockF.at<float>(0, 0));
    }
  }

  Mat resized;
  resize(wm, resized, Size(wmSize, wmSize));
  normalize(resized, resized, 0, 255, NORM_MINMAX);
  resized.convertTo(resized, CV_8U);
  return resized;
}

size_t estimateWatermarkCapacity(const Mat &host, int blockSize) {
  size_t blocks = (host.rows / blockSize) * (host.cols / blockSize);
  return blocks;
}

double compareWatermarks(const Mat &wm1, const Mat &wm2) {
  if (wm1.size() != wm2.size() || wm1.type() != wm2.type()) {
    throw std::runtime_error("Watermark images must have same size and type");
  }
  Mat wm1F, wm2F;
  wm1.convertTo(wm1F, CV_32F);
  wm2.convertTo(wm2F, CV_32F);
  Scalar mean1, stddev1, mean2, stddev2;
  meanStdDev(wm1F, mean1, stddev1);
  meanStdDev(wm2F, mean2, stddev2);
  Mat norm1 = wm1F - mean1[0];
  Mat norm2 = wm2F - mean2[0];
  double numerator = norm1.dot(norm2);
  double denominator = norm(norm1, NORM_L2) * norm(norm2, NORM_L2);
  return (denominator == 0) ? 0.0 : numerator / denominator;
}