#include "VideoEffects.hpp"
#include <opencv2/imgproc.hpp>

bool VideoEffects::applyTransition(const cv::Mat &frame1, const cv::Mat &frame2,
                                   cv::Mat &output,
                                   const TransitionEffect &effect,
                                   float progress) {
  CV_Assert(frame1.size() == frame2.size() && frame1.type() == frame2.type());

  switch (effect.type) {
  case TransitionEffect::FADE:
    cv::addWeighted(frame1, 1.0f - progress, frame2, progress, 0, output);
    break;

  case TransitionEffect::DISSOLVE: {
    cv::Mat mask(frame1.size(), CV_8UC1);
    cv::randu(mask, 0, 255);
    cv::threshold(mask, mask, 255 * progress, 255, cv::THRESH_BINARY);

    frame1.copyTo(output);
    frame2.copyTo(output, mask);
    break;
  }

  case TransitionEffect::WIPE: {
    int cols = static_cast<int>(frame1.cols * progress);
    output = frame1.clone();
    frame2(cv::Rect(0, 0, cols, frame2.rows))
        .copyTo(output(cv::Rect(0, 0, cols, frame1.rows)));
    break;
  }

  case TransitionEffect::SLIDE: {
    output = frame1.clone();
    int offset = static_cast<int>(frame1.cols * progress);
    cv::Mat roi = output(cv::Rect(0, 0, frame1.cols - offset, frame1.rows));
    frame2(cv::Rect(offset, 0, frame1.cols - offset, frame1.rows)).copyTo(roi);
    break;
  }
  }

  return true;
}

// ... 继续实现其他方法 ...
