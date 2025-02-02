#pragma once
#include <immintrin.h>
#include <opencv2/core.hpp>

class SIMDHelper {
public:
    static void processImageTile(const cv::Mat& input, cv::Mat& output, 
                               const cv::Mat& kernel, const cv::Rect& roi);
    
private:
    template<typename T>
    static void convolve2DSSE(const T* input, T* output, const float* kernel,
                             int rows, int cols, int kernelSize, int stride);
                             
    template<typename T>
    static void convolve2DAVX(const T* input, T* output, const float* kernel,
                             int rows, int cols, int kernelSize, int stride);
                             
    static bool hasSSE;
    static bool hasAVX;
    static void checkCPUSupport();
};
