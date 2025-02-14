#include "ImageUtils.hpp"

#include <QPixmap>

#include <spdlog/spdlog.h>
#include <stdexcept>

namespace ImageUtils {

cv::Mat qtImageToMat(const QImage &img) {
  try {
    return cv::Mat(img.height(), img.width(),
                   img.format() == QImage::Format_RGB32 ? CV_8UC4 : CV_8UC3,
                   const_cast<uchar *>(img.bits()),
                   static_cast<size_t>(img.bytesPerLine()));
  } catch (...) {
    spdlog::error("Image conversion failed");
    throw std::runtime_error("Image conversion failed");
  }
}

QImage matToQtImage(const cv::Mat &mat) {
  try {
    return QImage(mat.data, mat.cols, mat.rows, mat.step,
                  mat.channels() == 4 ? QImage::Format_RGB32
                                      : QImage::Format_RGB888)
        .copy();
  } catch (...) {
    spdlog::error("Mat to QImage conversion failed");
    throw std::runtime_error("Mat to QImage conversion failed");
  }
}

cv::UMat qtImageToUMat(const QImage &img) {
  cv::Mat mat = qtImageToMat(img);
  cv::UMat umat;
  mat.copyTo(umat);
  return umat;
}

QImage umatToQtImage(const cv::UMat &umat) {
  cv::Mat mat;
  umat.copyTo(mat);
  return matToQtImage(mat);
}

cv::Mat bgrToRgb(const cv::Mat &bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

cv::Mat rgbToBgr(const cv::Mat &rgb) {
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

QPixmap matToQPixmap(const cv::Mat &mat) {
    try {
        QImage img = matToQtImage(mat);
        return QPixmap::fromImage(img);
    } catch (...) {
        spdlog::error("Mat to QPixmap conversion failed");
        throw std::runtime_error("Mat to QPixmap conversion failed");
    }
}

cv::Mat qPixmapToMat(const QPixmap &pixmap) {
    try {
        QImage img = pixmap.toImage();
        return qtImageToMat(img);
    } catch (...) {
        spdlog::error("QPixmap to Mat conversion failed");
        throw std::runtime_error("QPixmap to Mat conversion failed");
    }
}

QImage grayToQImage(const cv::Mat &gray) {
    try {
        cv::Mat temp;
        if (gray.channels() == 1) {
            cv::cvtColor(gray, temp, cv::COLOR_GRAY2RGB);
            return matToQtImage(temp);
        }
        spdlog::warn("Input Mat is not grayscale");
        return matToQtImage(gray);
    } catch (...) {
        spdlog::error("Gray to QImage conversion failed");
        throw std::runtime_error("Gray to QImage conversion failed");
    }
}

cv::Mat qImageToGray(const QImage &img) {
    try {
        cv::Mat mat = qtImageToMat(img);
        cv::Mat gray;
        cv::cvtColor(mat, gray, cv::COLOR_RGB2GRAY);
        return gray;
    } catch (...) {
        spdlog::error("QImage to Gray conversion failed");
        throw std::runtime_error("QImage to Gray conversion failed");
    }
}

} // namespace ImageUtils