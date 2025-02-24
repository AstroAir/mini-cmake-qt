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

void embedDCTadaptive(Mat &block, const Mat &wm, double alpha);
void embedDWT(Mat &block, const Mat &wm, double alpha);
void embedDCTstandard(Mat &block, const Mat &wm, double alpha);
void embedHybrid(Mat &block, const Mat &wm, double alpha);

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

Mat embedWatermarkAdvanced(const Mat &host, const Mat &watermark,
                           const WatermarkConfig &config) {
  if (config.useCuda && cv::cuda::getCudaEnabledDeviceCount() > 0) {
#ifdef USE_CUDA
    return embedWatermarkCuda(host, watermark, config);
#endif
  }

  Mat hostYUV;
  cvtColor(host, hostYUV, COLOR_BGR2YUV);
  vector<Mat> channels;
  split(hostYUV, channels);

  Mat wm;
  resize(watermark, wm, channels[0].size());
  cvtColor(wm, wm, COLOR_BGR2GRAY);
  threshold(wm, wm, 128, 1, THRESH_BINARY);

  if (config.useAdaptiveStrength) {
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ch : config.channelsToEmbed) {
      if (ch >= channels.size())
        continue;

      for (int i = 0; i < channels[ch].rows; i += config.blockSize) {
        for (int j = 0; j < channels[ch].cols; j += config.blockSize) {
          Rect roi(j, i, config.blockSize, config.blockSize);
          if (roi.x + roi.width > channels[ch].cols ||
              roi.y + roi.height > channels[ch].rows)
            continue;

          Mat block = channels[ch](roi);
          double adaptiveAlpha = calculateAdaptiveStrength(block, config);

          Mat blockF;
          block.convertTo(blockF, CV_32F);

          switch (config.mode) {
          case WatermarkMode::DCT_STANDARD:
            embedDCTstandard(blockF, wm(roi), adaptiveAlpha);
            break;
          case WatermarkMode::DWT_BASED:
            embedDWT(blockF, wm(roi), adaptiveAlpha);
            break;
          case WatermarkMode::HYBRID:
            embedHybrid(blockF, wm(roi), adaptiveAlpha);
            break;
          default:
            embedDCTadaptive(blockF, wm(roi), adaptiveAlpha);
          }

          blockF.convertTo(block, CV_8U);
          block.copyTo(channels[ch](roi));
        }
      }
    }
  }

  merge(channels, hostYUV);
  Mat result;
  cvtColor(hostYUV, result, COLOR_YUV2BGR);
  return result;
}

double calculateAdaptiveStrength(const Mat &block,
                                 const WatermarkConfig &config) {
  Mat grad_x, grad_y;
  Sobel(block, grad_x, CV_32F, 1, 0);
  Sobel(block, grad_y, CV_32F, 0, 1);

  Mat magnitude;
  magnitude = abs(grad_x) + abs(grad_y);

  Scalar mean_val = mean(magnitude);
  double textureStrength = mean_val[0] / 255.0;

  return config.alpha * (1.0 + textureStrength * config.robustnessFactor);
}

WatermarkMetrics evaluateWatermark(const Mat &original, const Mat &watermarked,
                                   const Mat &extractedWatermark) {
  WatermarkMetrics metrics;

  // 计算PSNR
  metrics.psnr = PSNR(original, watermarked);

  // 计算SSIM
  metrics.ssim = SSIM(original, watermarked);

  // 计算鲁棒性
  metrics.robustness = compareWatermarks(watermarked, extractedWatermark);

  // 计算容量
  metrics.capacity = estimateWatermarkCapacity(original);

  return metrics;
}

Mat applyAttack(const Mat &image, const string &attackType, double intensity) {
  Mat attacked = image.clone();

  if (attackType == "noise") {
    Mat noise(image.size(), image.type());
    randn(noise, 0, intensity);
    attacked += noise;
  } else if (attackType == "blur") {
    GaussianBlur(image, attacked, Size(3, 3), intensity);
  } else if (attackType == "compression") {
    vector<uchar> buffer;
    vector<int> params = {IMWRITE_JPEG_QUALITY, static_cast<int>(intensity)};
    imencode(".jpg", image, buffer, params);
    attacked = imdecode(buffer, IMREAD_COLOR);
  } else if (attackType == "rotation") {
    Point2f center(image.cols / 2.0f, image.rows / 2.0f);
    Mat rotation = getRotationMatrix2D(center, intensity, 1.0);
    warpAffine(image, attacked, rotation, image.size());
  }

  return attacked;
}

#ifdef USE_CUDA
Mat embedWatermarkCuda(const Mat &host, const Mat &watermark,
                       const WatermarkConfig &config) {
  cuda::GpuMat d_host, d_watermark, d_result;
  d_host.upload(host);
  d_watermark.upload(watermark);

  // CUDA实现的水印嵌入
  // ...实现代码...

  Mat result;
  d_result.download(result);
  return result;
}
#endif

void embedDCTstandard(Mat &block, const Mat &wm, double alpha) {
  // 标准DCT系数修改
  dct(block, block);
  if (wm.at<uchar>(0, 0) > 0) {
    block.at<float>(3, 3) += alpha * block.at<float>(0, 0);
    block.at<float>(4, 4) += alpha * block.at<float>(0, 0);
  }
  idct(block, block);
}

void embedDWT(Mat &block, const Mat &wm, double alpha) {
  vector<Mat> waveletBands;
  // 3级DWT分解
  for (int i = 0; i < 3; i++) {
    Mat tmp;
    wavelet(block, tmp);
    waveletBands.push_back(tmp);
  }

  // 在LL子带嵌入水印
  if (wm.at<uchar>(0, 0) > 0) {
    waveletBands[2].at<float>(0, 0) += alpha * waveletBands[0].at<float>(0, 0);
  }

  // 逆变换
  for (int i = 2; i >= 0; i--) {
    inversewavelet(waveletBands[i], block);
  }
}

void embedHybrid(Mat &block, const Mat &wm, double alpha) {
  // 结合DCT和DWT的混合方法
  Mat dwtCoef;
  wavelet(block, dwtCoef);

  Mat dctCoef = dwtCoef.clone();
  dct(dctCoef, dctCoef);

  if (wm.at<uchar>(0, 0) > 0) {
    dctCoef.at<float>(3, 3) += alpha * dctCoef.at<float>(0, 0);
    dwtCoef.at<float>(0, 0) += alpha * dwtCoef.at<float>(0, 0);
  }

  idct(dctCoef, dctCoef);
  inversewavelet(dwtCoef, block);

  // 融合两种结果
  block = 0.5 * dctCoef + 0.5 * dwtCoef;
}

void embedDCTadaptive(Mat &block, const Mat &wm, double alpha) {
  dct(block, block);

  // 分析DCT系数选择最佳嵌入位置
  Mat dctMap = abs(block);
  double minVal, maxVal;
  Point minLoc, maxLoc;
  minMaxLoc(dctMap, &minVal, &maxVal, &minLoc, &maxLoc);

  if (wm.at<uchar>(0, 0) > 0) {
    // 在中频位置嵌入
    int midX = block.rows / 2;
    int midY = block.cols / 2;
    block.at<float>(midX, midY) += alpha * maxVal;
  }

  idct(block, block);
}

double SSIM(const Mat &img1, const Mat &img2) {
  const double C1 = 6.5025;  // (0.01 * 255)^2
  const double C2 = 58.5225; // (0.03 * 255)^2

  Mat I1, I2;
  img1.convertTo(I1, CV_32F);
  img2.convertTo(I2, CV_32F);

  Mat I1_2 = I1.mul(I1);
  Mat I2_2 = I2.mul(I2);
  Mat I1_I2 = I1.mul(I2);

  Mat mu1, mu2;
  GaussianBlur(I1, mu1, Size(11, 11), 1.5);
  GaussianBlur(I2, mu2, Size(11, 11), 1.5);

  Mat mu1_2 = mu1.mul(mu1);
  Mat mu2_2 = mu2.mul(mu2);
  Mat mu1_mu2 = mu1.mul(mu2);

  Mat sigma1_2, sigma2_2, sigma12;
  GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
  sigma1_2 -= mu1_2;

  GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
  sigma2_2 -= mu2_2;

  GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
  sigma12 -= mu1_mu2;

  Mat ssim;
  divide((2 * mu1_mu2 + C1).mul(2 * sigma12 + C2),
         (mu1_2 + mu2_2 + C1).mul(sigma1_2 + sigma2_2 + C2), ssim);

  Scalar mssim = mean(ssim);
  return mssim[0];
}

WatermarkMetrics evaluateWatermark(const Mat &original, const Mat &watermarked,
                                   const Mat &watermark,
                                   const Mat &extractedWatermark) {
  WatermarkMetrics metrics;
  metrics.psnr = PSNR(original, watermarked);
  metrics.ssim = SSIM(original, watermarked);
  metrics.robustness = compareWatermarks(watermark, extractedWatermark);
  metrics.capacity = estimateWatermarkCapacity(original);
  return metrics;
}

void wavelet(const Mat &input, Mat &output) {
  Mat temp = input.clone();
  int rows = temp.rows;
  int cols = temp.cols;

  output = Mat::zeros(rows, cols, CV_32F);

  // 水平方向变换
  for (int i = 0; i < rows; i++) {
    float *row = temp.ptr<float>(i);
    for (int j = 0; j < cols / 2; j++) {
      output.at<float>(i, j) = (row[2 * j] + row[2 * j + 1]) / 2.0f;
      output.at<float>(i, j + cols / 2) = (row[2 * j] - row[2 * j + 1]) / 2.0f;
    }
  }

  temp = output.clone();

  // 垂直方向变换
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows / 2; i++) {
      output.at<float>(i, j) =
          (temp.at<float>(2 * i, j) + temp.at<float>(2 * i + 1, j)) / 2.0f;
      output.at<float>(i + rows / 2, j) =
          (temp.at<float>(2 * i, j) - temp.at<float>(2 * i + 1, j)) / 2.0f;
    }
  }
}

void inversewavelet(const Mat &input, Mat &output) {
  Mat temp = input.clone();
  int rows = temp.rows;
  int cols = temp.cols;

  output = Mat::zeros(rows, cols, CV_32F);

  // 垂直方向逆变换
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows / 2; i++) {
      output.at<float>(2 * i, j) =
          temp.at<float>(i, j) + temp.at<float>(i + rows / 2, j);
      output.at<float>(2 * i + 1, j) =
          temp.at<float>(i, j) - temp.at<float>(i + rows / 2, j);
    }
  }

  temp = output.clone();

  // 水平方向逆变换
  for (int i = 0; i < rows; i++) {
    float *row = output.ptr<float>(i);
    for (int j = 0; j < cols / 2; j++) {
      row[2 * j] = temp.at<float>(i, j) + temp.at<float>(i, j + cols / 2);
      row[2 * j + 1] = temp.at<float>(i, j) - temp.at<float>(i, j + cols / 2);
    }
  }
}

double PSNR(const Mat &img1, const Mat &img2) {
  Mat diff;
  absdiff(img1, img2, diff);
  diff.convertTo(diff, CV_32F);
  diff = diff.mul(diff);

  Scalar mse = mean(diff);
  double MSE = mse[0] + mse[1] + mse[2];
  MSE /= 3.0;

  if (MSE <= 1e-10) {
    return 100.0;
  }

  double PSNR = 10.0 * log10((255 * 255) / MSE);
  return PSNR;
}

Mat compressWatermark(const Mat &watermark, int quality) {
  vector<uchar> buffer;
  vector<int> params = {IMWRITE_JPEG_QUALITY, quality};
  imencode(".jpg", watermark, buffer, params);
  return imdecode(buffer, IMREAD_UNCHANGED);
}

Mat encryptWatermark(const Mat &watermark, const string &key) {
  Mat encrypted = watermark.clone();

  // 简单的XOR加密
  vector<uchar> keyBytes(key.begin(), key.end());
  int keyLength = keyBytes.size();

  for (int i = 0; i < watermark.rows; i++) {
    for (int j = 0; j < watermark.cols; j++) {
      uchar keyByte = keyBytes[(i * watermark.cols + j) % keyLength];
      encrypted.at<uchar>(i, j) ^= keyByte;
    }
  }

  return encrypted;
}