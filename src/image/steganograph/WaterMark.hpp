#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 水印嵌入模式
enum class WatermarkMode {
    DCT_STANDARD,    // 标准DCT
    DCT_ADAPTIVE,    // 自适应DCT
    DWT_BASED,       // 小波变换
    HYBRID          // 混合模式
};

// 水印配置参数
struct WatermarkConfig {
    double alpha = 0.1;              // 嵌入强度
    int blockSize = 8;               // 分块大小
    WatermarkMode mode = WatermarkMode::DCT_STANDARD;
    std::vector<int> channelsToEmbed = {0}; // 要嵌入的通道
    bool useAdaptiveStrength = false; // 是否使用自适应强度
    double robustnessFactor = 1.0;    // 鲁棒性因子
    bool useCuda = false;             // 是否使用CUDA加速
};

// 水印评估结果
struct WatermarkMetrics {
    double psnr;           // 峰值信噪比
    double ssim;          // 结构相似度
    double robustness;    // 鲁棒性评分
    double capacity;      // 容量评估
};

/**
 * @brief Embeds a watermark into a host image using DCT.
 * @param host The host image.
 * @param watermark The watermark image.
 * @param alpha The embedding strength (default is 0.1).
 * @return The image with the embedded watermark.
 */
cv::Mat embedWatermark(const cv::Mat &host, const cv::Mat &watermark, double alpha = 0.1);

/**
 * @brief Embeds a watermark into multiple channels of a host image using DCT.
 * @param host The host image.
 * @param watermark The watermark image.
 * @param channelsToEmbed The channels to embed the watermark into.
 * @param alpha The embedding strength (default is 0.1).
 * @return The image with the embedded watermark.
 */
cv::Mat embedWatermarkMultiChannel(const cv::Mat &host, const cv::Mat &watermark,
                                   const std::vector<int> &channelsToEmbed,
                                   double alpha = 0.1);

/**
 * @brief Extracts a watermark from a watermarked image using DCT.
 * @param watermarked The watermarked image.
 * @param alpha The embedding strength used during embedding (default is 0.1).
 * @param wmSize The size of the extracted watermark (default is 64).
 * @return The extracted watermark image.
 */
cv::Mat extractWatermark(const cv::Mat &watermarked, double alpha = 0.1, int wmSize = 64);

/**
 * @brief Estimates the capacity of a host image for watermark embedding.
 * @param host The host image.
 * @param blockSize The size of the blocks used for embedding (default is 8).
 * @return The estimated capacity in number of blocks.
 */
size_t estimateWatermarkCapacity(const cv::Mat &host, int blockSize = 8);

/**
 * @brief Compares two watermarks using normalized cross-correlation.
 * @param wm1 The first watermark image.
 * @param wm2 The second watermark image.
 * @return The similarity score between the two watermarks.
 */
double compareWatermarks(const cv::Mat &wm1, const cv::Mat &wm2);

// 新增函数
cv::Mat embedWatermarkAdvanced(const cv::Mat &host, const cv::Mat &watermark, 
                              const WatermarkConfig &config);

WatermarkMetrics evaluateWatermark(const cv::Mat &original, const cv::Mat &watermarked, 
                                  const cv::Mat &extractedWatermark);

cv::Mat applyAttack(const cv::Mat &image, const std::string &attackType, 
                    double intensity = 1.0);

// CUDA加速版本
#ifdef USE_CUDA
cv::Mat embedWatermarkCuda(const cv::Mat &host, const cv::Mat &watermark,
                          const WatermarkConfig &config);
#endif

// 自适应强度计算
double calculateAdaptiveStrength(const cv::Mat &block, const WatermarkConfig &config);

// 水印压缩和加密
cv::Mat compressWatermark(const cv::Mat &watermark, int quality);
cv::Mat encryptWatermark(const cv::Mat &watermark, const std::string &key);

// DWT相关函数声明
void wavelet(const cv::Mat &input, cv::Mat &output);
void inversewavelet(const cv::Mat &input, cv::Mat &output);

// PSNR计算
double PSNR(const cv::Mat &img1, const cv::Mat &img2);

// SSIM计算（已存在）
double SSIM(const cv::Mat &img1, const cv::Mat &img2);