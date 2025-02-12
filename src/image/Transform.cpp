#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

cv::Mat arsinhTransform(const cv::Mat& input, double alpha) {
    // 创建与输入图像相同大小的输出图像
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    // 遍历每一个像素
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            // 获取输入像素值，归一化到 [0,1]
            double pixelValue = static_cast<double>(input.at<uchar>(y, x)) / 255.0;
            // 计算反双曲正弦变换
            double transformedValue = std::asinh(alpha * pixelValue) / std::asinh(alpha);
            // 将输出值乘以 255 并放入输出图像
            output.at<uchar>(y, x) = static_cast<uchar>(std::round(transformedValue * 255));
        }
    }

    return output;
}

int main() {
    // 读取图像
    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Unable to open image file!" << std::endl;
        return -1;
    }

    // 应用反双曲正弦变换
    double alpha = 15.0; // 可以根据需要调整
    cv::Mat enhancedImage = arsinhTransform(image, alpha);

    // 显示结果
    cv::imshow("Original Image", image);
    cv::imshow("Enhanced Image", enhancedImage);
    
    cv::waitKey(0); // 等待按键输入
    cv::destroyAllWindows(); // 关闭所有窗口

    return 0;
}
