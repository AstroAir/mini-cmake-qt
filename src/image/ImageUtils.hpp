#pragma once

#include <QImage>
#include <opencv2/opencv.hpp>

namespace ImageUtils {
cv::Mat qtImageToMat(const QImage &img);
QImage matToQtImage(const cv::Mat &mat);

cv::UMat qtImageToUMat(const QImage &img);
QImage umatToQtImage(const cv::UMat &umat);

// RGB/BGR conversions
cv::Mat bgrToRgb(const cv::Mat &bgr);
cv::Mat rgbToBgr(const cv::Mat &rgb);

// QPixmap conversions
QPixmap matToQPixmap(const cv::Mat &mat);
cv::Mat qPixmapToMat(const QPixmap &pixmap);

// Additional format conversions
QImage grayToQImage(const cv::Mat &gray);
cv::Mat qImageToGray(const QImage &img);

} // namespace ImageUtils