#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <QRect>

/**
 * @brief QR码配置结构体
 */
struct QRConfig {
  int size = 256;                 ///< QR码大小
  int version = 0;                ///< QR码版本(0为自动)
  QRecLevel level = QR_ECLEVEL_L; ///< 纠错级别
  QRencodeMode mode = QR_MODE_8;  ///< 编码模式
  bool caseSensitive = 1;         ///< 大小写敏感
  double blendRatio = 0.3;        ///< 混合比例
  int adaptiveBlockSize = 11;     ///< 自适应阈值块大小
  double adaptiveC = 2.0;         ///< 自适应阈值常数
};

/**
 * @brief 生成QR码
 * @param data 输入数据
 * @param config QR码配置
 * @return 生成的QR码图像
 */
cv::Mat generate_qrcode(const std::string &data, const QRConfig &config);

/**
 * @brief 嵌入QR码到宿主图像
 * @param host_image 宿主图像
 * @param qrcode QR码图像
 * @param position 嵌入位置
 * @param config QR码配置
 */
void embed_qrcode(cv::Mat &host_image, const cv::Mat &qrcode,
                  cv::Point position, const QRConfig &config);

/**
 * @brief 检测QR码
 * @param image 输入图像
 * @return 检测到的QR码数据
 */
std::string detect_qrcode(const cv::Mat &image);

/**
 * @brief 验证QR码
 * @param original 原始数据
 * @param detected 检测到的数据
 * @return 验证结果
 */
bool verify_qrcode(const std::string &original, const std::string &detected);