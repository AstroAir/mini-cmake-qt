#pragma once
#include <opencv2/opencv.hpp>
#include <functional>
#include <string>

struct NegativeConfig {
    float intensity = 1.0f;
    std::string channels = "RGB";
    bool save_alpha = true;
    std::string output_dir = "./output";
    cv::Rect roi = cv::Rect(0, 0, 0, 0);
    bool use_simd = true;
    bool multi_thread = true;
    
    void validate();
};

class NegativeProcessor {
public:
    explicit NegativeProcessor(const NegativeConfig& cfg);
    cv::Mat process(const cv::Mat& input, std::function<void(float)> progress_cb = nullptr);

private:
    NegativeConfig config_;
    cv::Mat lut_;

    void init_lut();
    void process_channel(cv::Mat& channel);
    void process_channel_simd(cv::Mat& channel);
};

class NegativeApp {
public:
    NegativeApp();
    int run(int argc, char** argv);

private:
    NegativeConfig config_;
    std::unique_ptr<NegativeProcessor> processor_;
    cv::Mat image_;
    cv::Mat negative_;

    void parseCommandLine(int argc, char** argv);
    void processImage();
    void saveResult(const std::string& input_path);
    void showHelp(const cv::CommandLineParser& parser);
};

void save_config(const std::string& path, const NegativeConfig& config);
void load_config(const std::string& path, NegativeConfig& config);
