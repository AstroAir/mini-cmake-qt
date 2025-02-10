#ifndef HFR_HPP
#define HFR_HPP

#include <nlohmann/json.hpp>
#include <tuple>

using json = nlohmann::json;
namespace cv {
class Mat;
}
using namespace cv;
using namespace std;

/**
 * @brief Calculates the Half-Flux Radius (HFR) of an image.
 * @param inImage The input image.
 * @param radius The radius for HFR calculation.
 * @return The calculated HFR value.
 */
auto calcHfr(const cv::Mat &inImage, float radius) -> double;

/**
 * @brief Checks the dimensions of the image.
 * @param img The input image.
 * @return True if the dimensions are valid, false otherwise.
 */
auto caldim(const cv::Mat &img) -> bool;

/**
 * @brief Preprocesses the input image.
 * @param img The input image.
 * @param grayimg The output grayscale image.
 * @param rgbImg The output RGB image.
 * @param mark_img The output marked image.
 */
auto preprocessImage(const Mat &img, Mat &grayimg, Mat &rgbImg, Mat &mark_img)
    -> void;

/**
 * @brief Removes noise from the image.
 * @param map The input image.
 * @param if_removehotpixel Flag to indicate if hot pixels should be removed.
 * @param if_noiseremoval Flag to indicate if noise removal should be applied.
 */
auto removeNoise(Mat &map, bool if_removehotpixel, bool if_noiseremoval)
    -> void;

/**
 * @brief Calculates the mean and standard deviation of the image.
 * @param map The input image.
 * @param down_sample_mean_std Flag to indicate if down-sampling should be
 * applied.
 * @param medianVal The output median value.
 * @param stdDev The output standard deviation.
 */
auto calculateMeanAndStd(const Mat &map, bool down_sample_mean_std,
                         double &medianVal, double &stdDev) -> void;

/**
 * @brief Detects stars and calculates the HFR.
 * @param img The input image.
 * @param if_removehotpixel Flag to indicate if hot pixels should be removed.
 * @param if_noiseremoval Flag to indicate if noise removal should be applied.
 * @param do_star_mark Flag to indicate if stars should be marked.
 * @param down_sample_mean_std Flag to indicate if down-sampling should be
 * applied.
 * @param mark_img The output marked image.
 * @return A tuple containing the marked image, the number of stars detected,
 * the HFR value, and additional JSON data.
 */
auto starDetectAndHfr(const Mat &img, bool if_removehotpixel,
                      bool if_noiseremoval, bool do_star_mark,
                      bool down_sample_mean_std, Mat mark_img)
    -> tuple<Mat, int, double, json>;

// 添加SIMD优化相关函数声明
#ifdef __AVX2__
auto processStarsAVX2(const cv::Mat &img, float radius) -> double;
#endif

// 添加并行处理相关函数声明  
auto processStarsParallel(const cv::Mat &img, bool if_removehotpixel,
                         bool if_noiseremoval) -> void;

#endif // HFR_HPP