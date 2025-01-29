#ifndef IMAGEIO_HPP
#define IMAGEIO_HPP

#include <string>
#include <vector>
#include <map>

namespace cv {
class Mat;
}
// 加载单张图像
auto loadImage(const std::string &filename, int flags = 1) -> cv::Mat;

// 从文件夹中读取所有图像
auto loadImages(const std::string &folder,
                const std::vector<std::string> &filenames = {}, int flags = 1)
    -> std::vector<std::pair<std::string, cv::Mat>>;

// 保存图像到文件
auto saveImage(const std::string &filename, const cv::Mat &image) -> bool;

auto saveMatTo8BitJpg(const cv::Mat &image, const std::string &output_path =
                                                "/dev/shm/MatTo8BitJPG.jpg")
    -> bool;

auto saveMatTo16BitPng(const cv::Mat &image, const std::string &output_path =
                                                 "/dev/shm/MatTo16BitPNG.png")
    -> bool;

auto saveMatToFits(const cv::Mat &image,
                   const std::string &output_path = "/dev/shm/MatToFITS.fits")
    -> bool;

// 获取FITS图像元数据
auto getFitsMetadata(const std::string &filepath) -> std::map<std::string, std::string>;

#endif