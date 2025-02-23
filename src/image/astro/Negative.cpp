#include "Negative.h"
#include <atomic>
#include <filesystem>


using namespace cv;
using namespace std;

void NegativeConfig::validate() {
  intensity = std::clamp(intensity, 0.0f, 1.0f);
  channels = channels.substr(0, 4);
}

NegativeProcessor::NegativeProcessor(const NegativeConfig &cfg) : config_(cfg) {
  config_.validate();
  init_lut();
}

void NegativeProcessor::init_lut() {
  const int max_value = 256;
  lut_.create(1, max_value, CV_8U);

  parallel_for_(Range(0, max_value), [&](const Range &range) {
    for (int i = range.start; i < range.end; i++) {
      lut_.at<uchar>(i) = static_cast<uchar>((255 - i) * config_.intensity +
                                             i * (1 - config_.intensity));
    }
  });
}

void NegativeProcessor::process_channel(Mat &channel) {
  if (config_.roi == Rect(0, 0, 0, 0)) {
    if (config_.use_simd) {
      process_channel_simd(channel);
    } else {
      LUT(channel, lut_, channel);
    }
  } else {
    Mat roi = channel(config_.roi);
    if (config_.use_simd) {
      process_channel_simd(roi);
    } else {
      LUT(roi, lut_, roi);
    }
  }
}

void NegativeProcessor::process_channel_simd(Mat &channel) {
#if defined(__AVX2__)
  const int step = 32;
#elif defined(__SSE2__)
  const int step = 16;
#else
  const int step = 4;
#endif

  parallel_for_(Range(0, channel.rows), [&](const Range &range) {
    for (int y = range.start; y < range.end; y++) {
      uchar *row = channel.ptr<uchar>(y);
      int x = 0;

      for (; x <= channel.cols - step; x += step) {
        for (int i = 0; i < step; i++) {
          row[x + i] = lut_.at<uchar>(row[x + i]);
        }
      }

      for (; x < channel.cols; x++) {
        row[x] = lut_.at<uchar>(row[x]);
      }
    }
  });
}

Mat NegativeProcessor::process(const Mat &input,
                               std::function<void(float)> progress_cb) {
  if (input.empty())
    return Mat();

  Mat output;
  vector<Mat> channels;
  split(input, channels);

  const map<char, int> channel_map = {{'B', 0}, {'G', 1}, {'R', 2}, {'A', 3}};

  atomic<int> progress{0};
  const int total_work =
      count_if(config_.channels.begin(), config_.channels.end(),
               [&](char c) { return channel_map.count(toupper(c)); });

#pragma omp parallel for if (config_.multi_thread)
  for (char c : config_.channels) {
    if (!channel_map.count(toupper(c)) ||
        channel_map.at(toupper(c)) >= channels.size())
      continue;

    int idx = channel_map.at(toupper(c));
    process_channel(channels[idx]);

    if (progress_cb) {
      progress_cb((++progress) / static_cast<float>(total_work));
    }
  }

  merge(channels, output);
  return output;
}

void save_config(const string &path, const NegativeConfig &config) {
  FileStorage fs(path, FileStorage::WRITE);
  fs << "intensity" << config.intensity << "channels" << config.channels
     << "save_alpha" << config.save_alpha << "roi_x" << config.roi.x << "roi_y"
     << config.roi.y << "roi_width" << config.roi.width << "roi_height"
     << config.roi.height << "use_simd" << config.use_simd << "multi_thread"
     << config.multi_thread;
}

void load_config(const string &path, NegativeConfig &config) {
  FileStorage fs(path, FileStorage::READ);
  if (!fs.isOpened())
    return;

  fs["intensity"] >> config.intensity;
  fs["channels"] >> config.channels;
  fs["save_alpha"] >> config.save_alpha;
  fs["roi_x"] >> config.roi.x;
  fs["roi_y"] >> config.roi.y;
  fs["roi_width"] >> config.roi.width;
  fs["roi_height"] >> config.roi.height;
  fs["use_simd"] >> config.use_simd;
  fs["multi_thread"] >> config.multi_thread;

  config.validate();
}

NegativeApp::NegativeApp() { config_ = NegativeConfig(); }

void NegativeApp::showHelp(const cv::CommandLineParser &parser) {
  parser.printMessage();
}

void NegativeApp::parseCommandLine(int argc, char **argv) {
  cv::CommandLineParser parser(argc, argv,
                               "{help h ? |     | 显示帮助信息}"
                               "{@input   |     | 输入图像路径}"
                               "{c config |     | 配置文件路径}"
                               "{i intensity |1.0| 反转强度 (0.0-1.0)}"
                               "{ch channels |RGB| 处理通道 (e.g. RGB, B, GA)}"
                               "{o output |     | 输出目录}");

  if (parser.has("help")) {
    showHelp(parser);
    throw std::runtime_error("显示帮助信息");
  }

  if (parser.has("config")) {
    load_config(parser.get<std::string>("config"), config_);
  }

  if (parser.has("intensity"))
    config_.intensity = parser.get<float>("intensity");
  if (parser.has("channels"))
    config_.channels = parser.get<std::string>("channels");
  if (parser.has("output"))
    config_.output_dir = parser.get<std::string>("output");

  std::string input_path = parser.get<std::string>("@input");
  if (!std::filesystem::exists(input_path)) {
    throw std::runtime_error("输入文件不存在");
  }

  image_ = cv::imread(input_path, cv::IMREAD_UNCHANGED);
  if (image_.empty()) {
    throw std::runtime_error("无法读取图像");
  }
}

void NegativeApp::processImage() {
  processor_ = std::make_unique<NegativeProcessor>(config_);
  std::cout << "开始处理..." << std::endl;
  negative_ = processor_->process(image_, [](float progress) {
    std::cout << "\r处理进度: " << static_cast<int>(progress * 100) << "%"
              << std::flush;
  });
  std::cout << std::endl;
}

void NegativeApp::saveResult(const std::string &input_path) {
  std::filesystem::create_directories(config_.output_dir);
  std::string output_path = (std::filesystem::path(config_.output_dir) /
                             std::filesystem::path(input_path).filename())
                                .string();
  cv::imwrite(output_path, negative_);
  save_config("last_config.xml", config_);
  std::cout << "处理完成，结果已保存至: " << output_path << std::endl;
}

int NegativeApp::run(int argc, char **argv) {
  try {
    parseCommandLine(argc, argv);
    processImage();
    saveResult(argv[1]); // argv[1] 是输入文件路径
    return 0;
  } catch (const cv::Exception &e) {
    std::cerr << "OpenCV错误: " << e.what() << std::endl;
    return -1;
  } catch (const std::exception &e) {
    std::cerr << "程序错误: " << e.what() << std::endl;
    return -1;
  }
}

int main(int argc, char **argv) {
  NegativeApp app;
  return app.run(argc, argv);
}