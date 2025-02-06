#include "ImageIO.hpp"

#include <fitsio.h>

#ifdef _WIN32
#undef TBYTE
#endif

#include "spdlog/sinks/basic_file_sink.h"
#include <chrono>
#include <filesystem>
#include <future>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
std::shared_ptr<spdlog::logger> imageIOLogger =
    spdlog::basic_logger_mt("ImageIOLogger", "logs/image_io.log");

// 辅助函数：记录OpenCV异常并返回空矩阵
bool handleCvException(const std::exception &e, const std::string &context) {
  imageIOLogger->error("{}: {}", context, e.what());
  return false;
}

bool isFitsFile(const std::string &filename) {
  std::string ext = std::filesystem::path(filename).extension().string();
  return ext == ".fits" || ext == ".fit" || ext == ".fts";
}
} // namespace

cv::Mat loadFitsImage(const std::string &filename) {
  fitsfile *fptr;
  int status = 0;
  int naxis = 0;
  long naxes[2] = {0, 0};
  int bitpix = 0;

  // 打开FITS文件
  if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
    imageIOLogger->error("无法打开FITS文件: {}", filename);
    fits_report_error(stderr, status);
    return cv::Mat();
  }

  // 获取图像信息
  if (fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status)) {
    imageIOLogger->error("无法获取FITS图像参数");
    fits_close_file(fptr, &status);
    fits_report_error(stderr, status);
    return cv::Mat();
  }

  // 分配内存并读取数据
  std::vector<float> contents(naxes[0] * naxes[1]);
  long fpixel[2] = {1, 1}; // FITS uses 1-based indexing

  if (fits_read_pix(fptr, TFLOAT, fpixel, naxes[0] * naxes[1], nullptr,
                    contents.data(), nullptr, &status)) {
    imageIOLogger->error("读取FITS像素数据失败");
    fits_close_file(fptr, &status);
    fits_report_error(stderr, status);
    return cv::Mat();
  }

  // 关闭文件
  fits_close_file(fptr, &status);

  // 转换为OpenCV格式
  cv::Mat result(naxes[1], naxes[0], CV_32F);
  std::memcpy(result.data, contents.data(), contents.size() * sizeof(float));

  // 归一化到0-255范围
  cv::Mat normalized;
  cv::normalize(result, normalized, 0, 255, cv::NORM_MINMAX);
  normalized.convertTo(normalized, CV_8U);

  return normalized;
}

auto loadImage(const std::string &filename, int flags) -> cv::Mat {
  try {
    imageIOLogger->info("开始加载图像 '{}'，参数 flags={}", filename, flags);

    if (!fs::exists(filename)) {
      imageIOLogger->error("图像文件不存在: {}", filename);
      return {};
    }

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat image;
    if (isFitsFile(filename)) {
      image = loadFitsImage(filename);
    } else {
      image = cv::imread(filename, flags);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    if (image.empty()) {
      imageIOLogger->error("加载图像失败: {} (耗时: {}ms)", filename, duration);
      return {};
    }

    imageIOLogger->info("成功加载图像: {}", filename);
    imageIOLogger->info("图像属性: {}x{}, {} 通道, 类型={}, 深度={}",
                        image.cols, image.rows, image.channels(), image.type(),
                        image.depth());
    imageIOLogger->info("加载耗时: {}ms", duration);

    return image;
  } catch (const cv::Exception &e) {
    handleCvException(e, "loadImage");
    return {};
  } catch (const std::exception &e) {
    imageIOLogger->error("loadImage 异常: {}", e.what());
    return {};
  }
}

auto loadImages(const std::string &folder,
                const std::vector<std::string> &filenames, int flags)
    -> std::vector<std::pair<std::string, cv::Mat>> {
  try {
    imageIOLogger->info("开始批量加载图像，文件夹: {}", folder);
    imageIOLogger->info("目标文件数量: {}",
                        filenames.empty() ? "所有文件"
                                          : std::to_string(filenames.size()));

    if (!fs::exists(folder)) {
      imageIOLogger->error("文件夹不存在: {}", folder);
      return {};
    }

    std::vector<std::pair<std::string, cv::Mat>> images;
    auto startTotal = std::chrono::high_resolution_clock::now();
    int successCount = 0;
    int failCount = 0;

    if (filenames.empty()) {
      imageIOLogger->info("扫描目录中的所有图像文件...");
      std::vector<std::future<void>> futures;
      for (const auto &entry : fs::directory_iterator(folder)) {
        if (entry.is_regular_file()) {
          futures.emplace_back(
              std::async(std::launch::async, [&images, &entry, flags,
                                              &successCount, &failCount]() {
                std::string filepath = entry.path().string();
                auto start = std::chrono::high_resolution_clock::now();
                cv::Mat img = cv::imread(filepath, flags);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                          start)
                        .count();

                if (!img.empty()) {
                  {
                    images.emplace_back(filepath, img);
                  }
                  successCount++;
                  imageIOLogger->info("加载图像 {}: {}x{}, {} 通道 ({}ms)",
                                      filepath, img.cols, img.rows,
                                      img.channels(), duration);
                } else {
                  failCount++;
                  imageIOLogger->error("加载图像失败: {} ({}ms)", filepath,
                                       duration);
                }
              }));
        }
      }
      // 等待所有任务完成
      for (auto &fut : futures) {
        fut.get();
      }
    } else {
      imageIOLogger->info("加载指定的 {} 个图像文件...", filenames.size());
      std::vector<std::future<void>> futures;
      for (const auto &filename : filenames) {
        futures.emplace_back(std::async(std::launch::async, [&images, &folder,
                                                             &filename, flags,
                                                             &successCount,
                                                             &failCount]() {
          std::string filepath = (fs::path(folder) / filename).string();
          auto start = std::chrono::high_resolution_clock::now();
          cv::Mat img = cv::imread(filepath, flags);
          auto end = std::chrono::high_resolution_clock::now();
          auto duration =
              std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();

          if (!img.empty()) {
            {
              images.emplace_back(filepath, img);
            }
            successCount++;
            imageIOLogger->info("加载图像 {}: {}x{}, {} 通道 ({}ms)", filepath,
                                img.cols, img.rows, img.channels(), duration);
          } else {
            failCount++;
            imageIOLogger->error("加载图像失败: {} ({}ms)", filepath, duration);
          }
        }));
      }
      // 等待所有任务完成
      for (auto &fut : futures) {
        fut.get();
      }
    }

    auto endTotal = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                             endTotal - startTotal)
                             .count();

    imageIOLogger->info("批量加载完成:");
    imageIOLogger->info("  成功: {} 张图像", successCount);
    imageIOLogger->info("  失败: {} 张图像", failCount);
    imageIOLogger->info("  总耗时: {}ms", totalDuration);
    imageIOLogger->info("  每张图像平均耗时: {}ms",
                        (successCount > 0) ? totalDuration / successCount : 0);

    return images;
  } catch (const cv::Exception &e) {
    handleCvException(e, "loadImages");
    return {};
  } catch (const std::exception &e) {
    imageIOLogger->error("loadImages 异常: {}", e.what());
    return {};
  }
}

auto saveImage(const std::string &filename, const cv::Mat &image) -> bool {
  try {
    imageIOLogger->info("开始保存图像: {}", filename);
    imageIOLogger->info("图像属性: {}x{}, {} 通道, 类型={}", image.cols,
                        image.rows, image.channels(), image.type());

    if (image.empty()) {
      imageIOLogger->error("无法保存空图像: {}", filename);
      return false;
    }

    auto start = std::chrono::high_resolution_clock::now();
    bool success = cv::imwrite(filename, image);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    if (success) {
      imageIOLogger->info("成功保存图像: {} ({}ms)", filename, duration);
      try {
        auto fileSize = fs::file_size(filename);
        imageIOLogger->info("文件大小: {} 字节", fileSize);
      } catch (const fs::filesystem_error &e) {
        imageIOLogger->warn("无法获取文件大小: {}", e.what());
      }
      return true;
    }

    imageIOLogger->error("保存图像失败: {} ({}ms)", filename, duration);
    return false;
  } catch (const cv::Exception &e) {
    return handleCvException(e, "saveImage");
  } catch (const std::exception &e) {
    imageIOLogger->error("saveImage 异常: {}", e.what());
    return false;
  }
}

auto saveMatTo8BitJpg(const cv::Mat &image, const std::string &output_path)
    -> bool {
  try {
    imageIOLogger->info("开始将图像转换为8位JPG: {}x{}", image.cols,
                        image.rows);

    if (image.empty()) {
      imageIOLogger->error("输入图像为空");
      return false;
    }

    imageIOLogger->info("输入图像: 类型={}, 深度={}, 通道={}", image.type(),
                        image.depth(), image.channels());

    cv::Mat image16, outputImage;

    // 根据输入深度转换到16位
    switch (image.depth()) {
    case CV_8U:
      imageIOLogger->info("将8位转换为16位，MSB对齐");
      image.convertTo(image16, CV_16UC1, 256.0);
      break;
    case CV_16U:
      imageIOLogger->info("保持16位深度");
      image16 = image.clone();
      break;
    default:
      imageIOLogger->error("不支持的图像深度: {}", image.depth());
      return false;
    }

    // 归一化到8位范围
    cv::normalize(image16, outputImage, 0, 255, cv::NORM_MINMAX, CV_8U);

    // 配置JPEG压缩参数
    std::vector<int> compressionParams = {cv::IMWRITE_JPEG_QUALITY,
                                          95}; // 高质量JPG

    return saveImage(output_path, outputImage);
  } catch (const cv::Exception &e) {
    return handleCvException(e, "saveMatTo8BitJpg");
  } catch (const std::exception &e) {
    imageIOLogger->error("saveMatTo8BitJpg 异常: {}", e.what());
    return false;
  }
}

auto saveMatTo16BitPng(const cv::Mat &image, const std::string &output_path)
    -> bool {
  try {
    imageIOLogger->info("开始将图像转换为16位PNG: {}x{}", image.cols,
                        image.rows);

    if (image.empty()) {
      imageIOLogger->error("输入图像为空");
      return false;
    }

    cv::Mat outputImage;

    // 最优的16位转换
    if (image.depth() == CV_8U) {
      imageIOLogger->info("将8位转换为16位");
      image.convertTo(outputImage, CV_16U, 256.0);
    } else if (image.depth() == CV_16U) {
      outputImage = image.clone();
    } else {
      imageIOLogger->error("不支持的图像深度: {}", image.depth());
      return false;
    }

    // 配置PNG压缩参数
    std::vector<int> compressionParams = {cv::IMWRITE_PNG_COMPRESSION,
                                          9}; // 最大压缩

    return saveImage(output_path, outputImage);
  } catch (const cv::Exception &e) {
    return handleCvException(e, "saveMatTo16BitPng");
  } catch (const std::exception &e) {
    imageIOLogger->error("saveMatTo16BitPng 异常: {}", e.what());
    return false;
  }
}

auto saveMatToFits(const cv::Mat &image, const std::string &output_path)
    -> bool {
  try {
    imageIOLogger->info("开始将图像转换为FITS: {}x{}", image.cols, image.rows);

    if (image.empty()) {
      imageIOLogger->error("输入图像为空");
      return false;
    }

    // 确保为灰度图像
    cv::Mat grayImage;
    if (image.channels() == 3) {
      cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
      grayImage = image.clone();
    }

    // FITS特定参数
    const long NAXES[2] = {grayImage.cols, grayImage.rows};
    int status = 0;
    fitsfile *fptr = nullptr;

    // 创建FITS文件并处理错误
    std::string FITS_PATH = "!" + output_path; // 强制覆盖
    if (fits_create_file(&fptr, FITS_PATH.c_str(), &status) != 0) {
      char error_msg[84];
      fits_get_errstatus(status, error_msg);
      imageIOLogger->error("无法创建FITS文件: {} - {}", output_path, error_msg);
      return false;
    }

    // 创建图像
    if (fits_create_img(fptr, SHORT_IMG, 2, const_cast<long *>(NAXES),
                        &status) != 0) {
      char error_msg[84];
      fits_get_errstatus(status, error_msg);
      imageIOLogger->error("无法创建FITS图像结构 - {}", error_msg);
      fits_close_file(fptr, &status);
      return false;
    }

    // 写入数据
    if (fits_write_img(fptr, TSHORT, 1, grayImage.total(),
                       grayImage.ptr<short>(), &status) != 0) {
      char error_msg[84];
      fits_get_errstatus(status, error_msg);
      imageIOLogger->error("无法写入FITS图像数据 - {}", error_msg);
      fits_close_file(fptr, &status);
      return false;
    }

    // 关闭文件
    fits_close_file(fptr, &status);

    if (status != 0) {
      char error_msg[84];
      fits_get_errstatus(status, error_msg);
      imageIOLogger->error("FITS错误: {}", error_msg);
      return false;
    }

    imageIOLogger->info("成功保存FITS文件: {}", output_path);
    return true;
  } catch (const cv::Exception &e) {
    return handleCvException(e, "saveMatToFits");
  } catch (const std::exception &e) {
    imageIOLogger->error("saveMatToFits 异常: {}", e.what());
    return false;
  }
}

auto getFitsMetadata(const std::string &filepath)
    -> std::map<std::string, std::string> {
  std::map<std::string, std::string> metadata;
  int status = 0;
  fitsfile *fptr = nullptr;

  try {
    if (fits_open_file(&fptr, filepath.c_str(), READONLY, &status)) {
      imageIOLogger->error("无法打开FITS文件: {}", filepath);
      fits_report_error(stderr, status);
      throw std::runtime_error("无法打开FITS文件");
    }

    int nkeys;
    if (fits_get_hdrspace(fptr, &nkeys, nullptr, &status)) {
      imageIOLogger->error("无法获取FITS头部信息数量");
      fits_report_error(stderr, status);
      throw std::runtime_error("无法获取FITS头部信息数量");
    }

    char card[FLEN_CARD];
    char keyname[FLEN_KEYWORD];
    char value[FLEN_VALUE];
    char comment[FLEN_COMMENT];

    for (int i = 1; i <= nkeys; i++) {
      if (fits_read_record(fptr, i, card, &status)) {
        imageIOLogger->warn("无法读取FITS记录 {}", i);
        fits_report_error(stderr, status);
        status = 0; // reset status after non-critical error
        continue;
      }

      int n = 0; // 新增的参数
      if (fits_get_keyname(card, keyname, &status, &n)) {
        imageIOLogger->warn("无法获取键名 {}", i);
        fits_report_error(stderr, status);
        status = 0; // reset status after non-critical error
        continue;
      }

      if (fits_parse_value(card, value, comment, &status)) {
        imageIOLogger->warn("无法解析值 {}", i);
        fits_report_error(stderr, status);
        status = 0; // reset status after non-critical error
        continue;
      }

      std::string key(keyname);
      std::string val(value);

      // 过滤掉空值和特殊键
      if (!key.empty() && key != "SIMPLE" && key != "END" && key != "EXTEND" &&
          !val.empty()) {
        metadata[key] = val;
      }
    }

    if (fptr) {
      fits_close_file(fptr, &status);
      if (status) {
        imageIOLogger->error("关闭FITS文件时发生错误");
        fits_report_error(stderr, status);
      }
    }
    imageIOLogger->info("成功读取FITS元数据: {} 项", metadata.size());

  } catch (const std::exception &e) {
    imageIOLogger->error("读取FITS元数据失败: {}", e.what());
    if (fptr) {
      fits_close_file(fptr, &status);
      fits_report_error(stderr, status);
    }
    // 如果在清理资源后重新抛出异常，请确保状态正确
    throw;
  }

  return metadata;
}