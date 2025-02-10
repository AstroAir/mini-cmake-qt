#include <opencv2/opencv.hpp>
#include <variant>
#include <functional>

enum class WatermarkType {
    Magnitude,  // 幅度调制（默认）
    Phase,      // 相位调制
    Complex,    // 复数域嵌入
    QRCode,     // QR码水印
    Diffused    // 扩散水印
};

struct WatermarkParams {
    float alpha = 0.15f;          // 强度系数
    int qr_version = 5;           // QR码版本
    uint32_t seed = 0xDEADBEEF;   // 随机种子
    bool anti_aliasing = true;    // 抗锯齿
};

class FourierWatermarker {
public:
    explicit FourierWatermarker(WatermarkType type = WatermarkType::Magnitude, 
                              WatermarkParams params = {})
        : type_(type), params_(params) 
    {
        init_strategies();
    }

    Mat embed(const Mat& src, const Mat& watermark) {
        return std::visit([&](auto&& strategy) {
            return strategy(src, watermark);
        }, strategy_);
    }

    // QR码专用接口
    Mat embedQR(const Mat& src, const std::string& text) {
        Mat qr = generateQRCode(text);
        return std::get<4>(strategy_)(src, qr);
    }

private:
    void init_strategies() {
        switch(type_) {
            case WatermarkType::Phase:
                strategy_ = [this](const Mat& s, const Mat& w) { 
                    return phaseEmbed(s, w); 
                };
                break;
            case WatermarkType::Complex:
                strategy_ = [this](const Mat& s, const Mat& w) { 
                    return complexEmbed(s, w); 
                };
                break;
            case WatermarkType::QRCode:
                strategy_ = [this](const Mat& s, const Mat& w) { 
                    return qrEmbed(s, w); 
                };
                break;
            case WatermarkType::Diffused:
                strategy_ = [this](const Mat& s, const Mat& w) { 
                    return diffuseEmbed(s, w); 
                };
                break;
            default:
                strategy_ = [this](const Mat& s, const Mat& w) { 
                    return magnitudeEmbed(s, w); 
                };
        }
    }

    // 各水印策略实现
    Mat magnitudeEmbed(const Mat& src, const Mat& watermark) {
        // ... 基础幅度嵌入实现 ... 
    }

    Mat phaseEmbed(const Mat& src, const Mat& watermark) {
        Mat planes[2], phase_mod;
        split(complex_img_, planes);
        
        // 在相位信息嵌入水印
        Mat scaled_wm;
        watermark.convertTo(scaled_wm, CV_32F, params_.alpha * CV_2PI);
        add(planes[1](roi_), scaled_wm, planes[1](roi_));
        
        // 相位归一化
        phase(planes[1], planes[1], true);  // 自动相位卷绕
        merge(planes, 2, complex_img_);
    }

    Mat complexEmbed(const Mat& src, const Mat& watermark) {
        // 复数域联合嵌入
        Mat re_wm, im_wm;
        watermark.convertTo(re_wm, CV_32F, params_.alpha);
        flip(watermark, im_wm, 1).convertTo(im_wm, CV_32F, params_.alpha);
        
        planes[0](roi_) += re_wm;
        planes[1](roi_) += im_wm;
    }

    Mat qrEmbed(const Mat& src, const Mat& qr) {
        // QR码预处理
        Mat qr_resized;
        resize(qr, qr_resized, roi_.size(), 0, 0, 
              params_.anti_aliasing ? INTER_AREA : INTER_NEAREST);
        
        // 高频区域嵌入
        Mat high_freq = magnitude_(roi_).clone();
        bitwise_xor(high_freq, qr_resized, magnitude_(roi_));
    }

    Mat diffuseEmbed(const Mat& src, const Mat& watermark) {
        // 扩散水印生成
        Mat spread_wm = spreadSpectrum(watermark);
        
        // 随机频点嵌入
        RNG rng(params_.seed);
        parallel_for_(Range(0, spread_wm.total()), [&](const Range& range) {
            for(int i = range.start; i < range.end; ++i) {
                Point pt(rng.uniform(0, magnitude_.cols), 
                        rng.uniform(0, magnitude_.rows));
                magnitude_.at<float>(pt) += params_.alpha * spread_wm.at<float>(i);
            }
        });
    }

    Mat generateQRCode(const std::string& text) {
        // 使用OpenCV内置QR生成器
        QRCodeEncoder::Params params;
        params.version = params_.qr_version;
        Ptr<QRCodeEncoder> encoder = QRCodeEncoder::create(params);
        Mat qr;
        encoder->encode(text, qr);
        return qr;
    }

    Mat spreadSpectrum(const Mat& watermark) {
        // 扩频调制实现
        Mat spread;
        dft(watermark, spread, DFT_COMPLEX_OUTPUT);
        randShuffle(spread, 1.0, &params_.seed);
        return spread;
    }

    WatermarkType type_;
    WatermarkParams params_;
    std::variant<std::function<Mat(const Mat&, const Mat&)>, ...> strategy_;
    // ... 中间状态缓存 ...
};

#include <opencv2/opencv.hpp>
#include <execution>
#include <ranges>

namespace fs = std::filesystem;
using namespace cv;

// 预声明优化后的核心函数
void optimized_fft_shift(Mat& mag) noexcept;
Mat parallel_embed(Mat& magnitude, const Mat& watermark, float alpha, Rect center);

// 高频优化版傅里叶水印嵌入
Mat optimized_embed(const Mat& src, const Mat& watermark, float alpha = 0.15f) {
    CV_Assert(src.type() == CV_8UC3 && !src.empty());
    
    // 异步流水线处理
    Mat gray_src, padded;
    std::mutex mtx;
    std::thread([&]{
        Mat tmp;
        cvtColor(src, tmp, COLOR_BGR2GRAY);
        const auto opt_size = getOptimalDFTSize(src.size());
        copyMakeBorder(tmp, padded, 0, opt_size.height - src.rows,
                      0, opt_size.width - src.cols, BORDER_CONSTANT);
        tmp.release();  // 及时释放内存
    }).join();

    // 原地DFT优化
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complex_img;
    merge(planes, 2, complex_img);
    dft(complex_img, complex_img, DFT_COMPLEX_OUTPUT);  // 启用OpenCV内部优化

    // 并行频谱处理
    Mat magnitude, phase;
    {
        std::vector<Mat> planess(2);
        split(complex_img, planess);  // 非连续操作优化
        cartToPolar(planess[0], planess[1], magnitude, phase);
        optimized_fft_shift(magnitude);
    }

    // 水印预处理（使用并行缩放）
    Mat scaled_watermark;
    std::atomic<bool> resize_done{false};
    std::thread([&]{
        resize(watermark, scaled_watermark, magnitude.size(), 0, 0, INTER_AREA);
        scaled_watermark.convertTo(scaled_watermark, CV_32F, 1.0/255);
        resize_done.store(true, std::memory_order_release);
    }).detach();

    // 重叠计算：准备嵌入区域
    const Rect center(magnitude.cols/2 - watermark.cols/2,
                     magnitude.rows/2 - watermark.rows/2,
                     watermark.cols, watermark.rows);

    // 等待水印预处理完成
    while(!resize_done.load(std::memory_order_acquire)) {
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
    std::async(std::launch::async, [&]{
        inverse_transform(Rect(0, 0, src.cols, src.rows)).convertTo(output, CV_8U);
    }).wait();

    return output;
}

// 并行嵌入核心实现
Mat parallel_embed(Mat& magnitude, const Mat& watermark, float alpha, Rect center) {
    Mat modified_mag = magnitude.clone();
    const auto roi = modified_mag(center);
    
    // 使用C++17并行算法
    std::for_each(std::execution::par_unseq,
        roi.begin<float>(), roi.end<float>(),
        [start=watermark.begin<float>(), alpha](auto& mag_val) mutable {
            mag_val += alpha * (*start++);
        });
    
    optimized_fft_shift(modified_mag);
    return modified_mag;
}

// 优化后的FFT Shift（使用指针运算）
void optimized_fft_shift(Mat& mag) noexcept {
    const int cx = mag.cols / 2;
    const int cy = mag.rows / 2;
    
    // 使用指针直接操作内存
    auto swap_block = [](float* a, float* b, int block_size) {
        for(int i=0; i<block_size; ++i) {
            std::swap(a[i], b[i]);
        }
    };

    float* data = mag.ptr<float>();
    const int row_step = mag.step1() / sizeof(float);
    
    // 并行交换四个象限
    #pragma omp parallel sections
    {
        #pragma omp section
        swap_block(data, data + cy*row_step + cx, cx*cy);
        
        #pragma omp section
        swap_block(data + cx, data + cy*row_step, cx*cy);
    }
}