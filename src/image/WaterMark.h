#pragma once

#include <opencv2/opencv.hpp>
#include <functional>

using namespace cv;

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

// 主要类和接口声明
class FourierWatermarker {
public:
    explicit FourierWatermarker(WatermarkType type = WatermarkType::Magnitude,
                                WatermarkParams params = {});
    Mat embed(const Mat &src, const Mat &watermark);
    Mat embedQR(const Mat &src, const std::string &text);

private:
    void init_strategies();
    Mat magnitudeEmbed(const Mat &src, const Mat &watermark);
    Mat phaseEmbed(const Mat &src, const Mat &watermark);
    Mat complexEmbed(const Mat &src, const Mat &watermark);
    Mat qrEmbed(const Mat &src, const Mat &qr);
    Mat diffuseEmbed(const Mat &src, const Mat &watermark);
    Mat generateQRCode(const std::string &text);
    Mat spreadSpectrum(const Mat &watermark);

    WatermarkType type_;
    WatermarkParams params_;
    std::function<Mat(const Mat &, const Mat &)> strategy_;
};

// 独立功能函数声明
Mat optimized_embed(const Mat &src, const Mat &watermark, float alpha = 0.15f);
Mat parallel_embed(Mat &magnitude, const Mat &watermark, float alpha, Rect center);
void optimized_fft_shift(Mat &mag) noexcept;