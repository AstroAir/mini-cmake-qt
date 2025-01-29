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

auto calcHfr(const cv::Mat &inImage, float radius) -> double;
auto caldim(const cv::Mat &img) -> bool;
auto preprocessImage(const Mat &img, Mat &grayimg, Mat &rgbImg, Mat &mark_img)
    -> void;
auto removeNoise(Mat &map, bool if_removehotpixel, bool if_noiseremoval)
    -> void;
auto calculateMeanAndStd(const Mat &map, bool down_sample_mean_std,
                         double &medianVal, double &stdDev) -> void;
auto starDetectAndHfr(const Mat &img, bool if_removehotpixel,
                      bool if_noiseremoval, bool do_star_mark,
                      bool down_sample_mean_std, Mat mark_img)
    -> tuple<Mat, int, double, json>;

#endif // HFR_HPP