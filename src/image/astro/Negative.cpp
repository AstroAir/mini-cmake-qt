#include "Negative.h"
#include "ParallelConfig.hpp"
#include <atomic>
#include <filesystem>
#include <fstream>
#include <json/json.h>

using namespace cv;
using namespace std;

void NegativeConfig::validate() {
  intensity = std::clamp(intensity, 0.0f, 1.0f);
  channels = channels.substr(0, 4);

  if (intensity < 0.0f || intensity > 1.0f) {
    throw std::invalid_argument("Intensity must be between 0.0 and 1.0");
  }

  for (char ch : channels) {
    if (ch != 'R' && ch != 'G' && ch != 'B' && ch != 'A') {
      throw std::invalid_argument("Invalid channel specified");
    }
  }

  if (!std::filesystem::exists(output_dir)) {
    std::filesystem::create_directories(output_dir);
  }
}

NegativeProcessor::NegativeProcessor(const NegativeConfig &cfg) : config_(cfg) {
  config_.validate();
  init_lut();
}

void NegativeProcessor::init_lut() {
  const int max_value = 256;
  lut_.create(1, max_value, CV_8U);

  // 仅在数据量足够大时使用并行
  if (max_value >= parallel_config::MIN_PARALLEL_SIZE) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(parallel_config::DEFAULT_THREAD_COUNT)
#endif
    for (int i = 0; i < max_value; i++) {
      lut_.at<uchar>(i) = static_cast<uchar>((255 - i) * config_.intensity +
                                             i * (1 - config_.intensity));
    }
  } else {
    for (int i = 0; i < max_value; i++) {
      lut_.at<uchar>(i) = static_cast<uchar>((255 - i) * config_.intensity +
                                             i * (1 - config_.intensity));
    }
  }
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

  const int total_pixels = channel.rows * channel.cols;
  if (total_pixels >= parallel_config::MIN_PARALLEL_SIZE) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(parallel_config::DEFAULT_THREAD_COUNT)
#endif
    for (int y = 0; y < channel.rows; y++) {
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
  } else {
    for (int y = 0; y < channel.rows; y++) {
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
  }
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

  const int total_channels =
      count_if(config_.channels.begin(), config_.channels.end(),
               [&](char c) { return channel_map.count(toupper(c)); });

  if (total_channels >= parallel_config::MIN_FRAMES_PARALLEL) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(parallel_config::DEFAULT_THREAD_COUNT)
#endif
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
  } else {
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
  }

  merge(channels, output);
  return output;
}

void save_config(const string &path, const NegativeConfig &config) {
  Json::Value root;
  root["intensity"] = config.intensity;
  root["channels"] = config.channels;
  root["save_alpha"] = config.save_alpha;
  root["output_dir"] = config.output_dir;
  root["roi"] = Json::Value(Json::arrayValue);
  root["roi"].append(config.roi.x);
  root["roi"].append(config.roi.y);
  root["roi"].append(config.roi.width);
  root["roi"].append(config.roi.height);

  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open config file for writing");
  }

  Json::StyledWriter writer;
  file << writer.write(root);
}

void load_config(const string &path, NegativeConfig &config) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open config file for reading");
  }

  Json::Value root;
  Json::Reader reader;
  if (!reader.parse(file, root)) {
    throw std::runtime_error("Failed to parse config file");
  }

  config.intensity = root.get("intensity", 1.0f).asFloat();
  config.channels = root.get("channels", "RGB").asString();
  config.save_alpha = root.get("save_alpha", true).asBool();
  config.output_dir = root.get("output_dir", "./output").asString();

  if (root.isMember("roi") && root["roi"].size() == 4) {
    config.roi = cv::Rect(root["roi"][0].asInt(), root["roi"][1].asInt(),
                          root["roi"][2].asInt(), root["roi"][3].asInt());
  }

  config.validate();
}

NegativeApp::NegativeApp() : processor_(nullptr) {}

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
  if (!processor_) {
    processor_ = std::make_unique<NegativeProcessor>(config_);
  }
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
