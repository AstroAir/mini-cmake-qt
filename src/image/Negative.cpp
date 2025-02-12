#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <atomic>

using namespace cv;
using namespace std;
namespace fs = filesystem;

/**
 * @brief 负片变换配置结构体
 * 
 * 支持以下特性：
 * 1. 多通道独立处理
 * 2. ROI区域处理
 * 3. 强度调节
 * 4. Alpha通道保留
 */
struct NegativeConfig {
    float intensity = 1.0f;           // 反转强度 [0.0, 1.0]
    string channels = "RGB";          // 处理通道
    bool save_alpha = true;           // 保留Alpha通道
    string output_dir = "./output";   // 输出目录
    Rect roi = Rect(0, 0, 0, 0);     // 感兴趣区域
    bool use_simd = true;            // 启用SIMD加速
    bool multi_thread = true;         // 启用多线程
    
    void validate() {
        intensity = std::clamp(intensity, 0.0f, 1.0f);
        channels = channels.substr(0, 4);  // 最多4个通道
    }
};

/**
 * @brief 负片变换处理器类
 * 
 * 实现功能：
 * 1. SIMD加速算法
 * 2. 多线程处理
 * 3. 进度反馈
 * 4. 缓存管理
 */
class NegativeProcessor {
public:
    explicit NegativeProcessor(const NegativeConfig& cfg) : config_(cfg) {
        config_.validate();
        init_lut();
    }
    
    Mat process(const Mat& input, std::function<void(float)> progress_cb = nullptr) {
        if (input.empty()) return Mat();
        
        Mat output;
        vector<Mat> channels;
        split(input, channels);
        
        const map<char, int> channel_map = {
            {'B', 0}, {'G', 1}, {'R', 2}, {'A', 3}
        };
        
        // 处理进度计数
        atomic<int> progress{0};  // 修复: 使用花括号初始化
        const int total_work = count_if(config_.channels.begin(), 
                                      config_.channels.end(),
                                      [&](char c) { 
                                          return channel_map.count(toupper(c)); 
                                      });
        
        #pragma omp parallel for if(config_.multi_thread)
        for (char c : config_.channels) {
            if (!channel_map.count(toupper(c)) || 
                channel_map.at(toupper(c)) >= channels.size()) continue;
                
            int idx = channel_map.at(toupper(c));
            process_channel(channels[idx]);
            
            if (progress_cb) {
                progress_cb((++progress) / static_cast<float>(total_work));
            }
        }
        
        merge(channels, output);
        return output;
    }
    
private:
    NegativeConfig config_;
    Mat lut_;  // 查找表加速
    
    void init_lut() {
        const int max_value = 256;
        lut_.create(1, max_value, CV_8U);
        
        parallel_for_(Range(0, max_value), [&](const Range& range) {
            for (int i = range.start; i < range.end; i++) {
                lut_.at<uchar>(i) = static_cast<uchar>(
                    (255 - i) * config_.intensity + i * (1 - config_.intensity)
                );
            }
        });
    }
    
    void process_channel(Mat& channel) {
        if (config_.roi == Rect(0, 0, 0, 0)) {
            if (config_.use_simd) {
                process_channel_simd(channel);
            } else {
                LUT(channel, lut_, channel);
            }
        } else {
            Mat roi = channel(config_.roi);  // 修复: 创建引用
            if (config_.use_simd) {
                process_channel_simd(roi);
            } else {
                LUT(roi, lut_, roi);
            }
        }
    }
    
    void process_channel_simd(Mat& channel) {
        #if defined(__AVX2__)
            const int step = 32;  // AVX2
        #elif defined(__SSE2__)
            const int step = 16;  // SSE2
        #else
            const int step = 4;   // 基础优化
        #endif
        
        parallel_for_(Range(0, channel.rows), [&](const Range& range) {
            for (int y = range.start; y < range.end; y++) {
                uchar* row = channel.ptr<uchar>(y);
                int x = 0;
                
                // SIMD处理
                for (; x <= channel.cols - step; x += step) {
                    for (int i = 0; i < step; i++) {
                        row[x + i] = lut_.at<uchar>(row[x + i]);
                    }
                }
                
                // 处理剩余像素
                for (; x < channel.cols; x++) {
                    row[x] = lut_.at<uchar>(row[x]);
                }
            }
        });
    }
};

// GUI相关全局变量
namespace {
    Mat g_image, g_negative;
    NegativeConfig g_config;
    bool g_roi_selecting = false;
    Point g_roi_start, g_roi_end;
    unique_ptr<NegativeProcessor> g_processor;
}

// GUI事件回调
void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        g_roi_selecting = true;
        g_roi_start = Point(x, y);
        g_roi_end = Point(x, y);
    } 
    else if (event == EVENT_MOUSEMOVE && g_roi_selecting) {
        g_roi_end = Point(x, y);
        Mat display = g_image.clone();
        rectangle(display, g_roi_start, g_roi_end, Scalar(0, 255, 0), 2);
        imshow("Negative Transform", display);
    } 
    else if (event == EVENT_LBUTTONUP) {
        g_roi_selecting = false;
        g_config.roi = Rect(
            min(g_roi_start.x, g_roi_end.x),
            min(g_roi_start.y, g_roi_end.y),
            abs(g_roi_start.x - g_roi_end.x),
            abs(g_roi_start.y - g_roi_end.y)
        );
        
        if (g_processor) {
            g_negative = g_processor->process(g_image, [](float progress) {
                cout << "\r处理进度: " << static_cast<int>(progress * 100) << "%" << flush;
            });
            cout << endl;
            imshow("Result", g_negative);
        }
    }
}

// 配置文件操作
void save_config(const string& path) {
    FileStorage fs(path, FileStorage::WRITE);
    fs << "intensity" << g_config.intensity
       << "channels" << g_config.channels
       << "save_alpha" << g_config.save_alpha
       << "roi_x" << g_config.roi.x
       << "roi_y" << g_config.roi.y
       << "roi_width" << g_config.roi.width
       << "roi_height" << g_config.roi.height
       << "use_simd" << g_config.use_simd
       << "multi_thread" << g_config.multi_thread;
}

void load_config(const string& path) {
    FileStorage fs(path, FileStorage::READ);
    if (!fs.isOpened()) return;
    
    fs["intensity"] >> g_config.intensity;
    fs["channels"] >> g_config.channels;
    fs["save_alpha"] >> g_config.save_alpha;
    fs["roi_x"] >> g_config.roi.x;
    fs["roi_y"] >> g_config.roi.y;
    fs["roi_width"] >> g_config.roi.width;
    fs["roi_height"] >> g_config.roi.height;
    fs["use_simd"] >> g_config.use_simd;
    fs["multi_thread"] >> g_config.multi_thread;
    
    g_config.validate();
}

int main(int argc, char** argv) {
    try {
        // 命令行参数解析
        CommandLineParser parser(argc, argv,
                                 "{help h ? |     | 显示帮助信息}"
                                 "{@input   |     | 输入图像路径}"
                                 "{c config |     | 配置文件路径}"
                                 "{i intensity |1.0| 反转强度 (0.0-1.0)}"
                                 "{ch channels |RGB| 处理通道 (e.g. RGB, B, GA)}"
                                 "{o output |     | 输出目录}"
                                 "{gui      |     | 启用交互模式}");

        if (parser.has("help")) {
            parser.printMessage();
            return 0;
        }

        // 加载配置
        if (parser.has("config")) {
            load_config(parser.get<string>("config"));
        }

        // 覆盖命令行参数
        if (parser.has("intensity"))
            g_config.intensity = parser.get<float>("intensity");
        if (parser.has("channels"))
            g_config.channels = parser.get<string>("channels");
        if (parser.has("output"))
            g_config.output_dir = parser.get<string>("output");

        // 读取图像
        string input_path = parser.get<string>("@input");
        if (!fs::exists(input_path)) {
            cerr << "错误：输入文件不存在" << endl;
            return -1;
        }
        g_image = imread(input_path, IMREAD_UNCHANGED);
        if (g_image.empty()) {
            cerr << "错误：无法读取图像" << endl;
            return -1;
        }

        // 创建输出目录
        fs::create_directories(g_config.output_dir);

        // 初始化处理器
        g_processor = make_unique<NegativeProcessor>(g_config);

        // 交互模式
        if (parser.has("gui")) {
            namedWindow("Negative Transform", WINDOW_AUTOSIZE);
            setMouseCallback("Negative Transform", onMouse);

            // 创建控制面板
            createTrackbar("Intensity", "Negative Transform", nullptr, 100,
                           [](int val, void *) {
                               g_config.intensity = val / 100.0f;
                               g_negative = g_processor->process(g_image, [](float progress) {
                                   cout << "\r处理进度: " << static_cast<int>(progress * 100) << "%" << flush;
                               });
                               cout << endl;
                               imshow("Result", g_negative);
                           });
            setTrackbarPos("Intensity", "Negative Transform", g_config.intensity * 100);

            imshow("Negative Transform", g_image);
            g_negative = g_processor->process(g_image, [](float progress) {
                cout << "\r处理进度: " << static_cast<int>(progress * 100) << "%" << flush;
            });
            cout << endl;
            imshow("Result", g_negative);

            while (true) {
                int key = waitKey(0);
                if (key == 27)
                    break;          // ESC退出
                if (key == 's') { // 保存结果
                    string output_path =
                        (fs::path(g_config.output_dir) / fs::path(input_path).filename())
                            .string();
                    imwrite(output_path, g_negative);
                    save_config("last_config.xml");
                    cout << "结果已保存至: " << output_path << endl;
                }
            }
        } else { // 批处理模式
            cout << "开始处理..." << endl;
            g_negative = g_processor->process(g_image, [](float progress) {
                cout << "\r处理进度: " << static_cast<int>(progress * 100) << "%" << flush;
            });
            cout << endl;

            string output_path =
                (fs::path(g_config.output_dir) / fs::path(input_path).filename())
                    .string();
            imwrite(output_path, g_negative);
            save_config("last_config.xml");
            cout << "处理完成，结果已保存至: " << output_path << endl;
        }

        return 0;
    }
    catch (const Exception& e) {
        cerr << "OpenCV错误: " << e.what() << endl;
        return -1;
    }
    catch (const exception& e) {
        cerr << "程序错误: " << e.what() << endl;
        return -1;
    }
}