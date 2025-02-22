#pragma once

#include <QWidget>
#include <opencv2/videoio.hpp>

class QImage;
class QTimer;

class VideoPreview : public QWidget {
  Q_OBJECT

public:
  explicit VideoPreview(QWidget *parent = nullptr);
  void setVideo(const cv::VideoCapture &cap);
  void applyFilter(const QString &filterName);
  void applyEffect(const QString &effectName);

public slots:
  void play();
  void pause();
  void seekToFrame(int frameNumber);
  void updateFrame();

protected:
  void paintEvent(QPaintEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;

private:
  QImage matToQImage(const cv::Mat &mat);
  void updatePreviewSize();
  
  cv::VideoCapture m_capture;
  cv::Mat m_currentFrame;
  QTimer *m_timer;
  QSize m_previewSize;
  bool m_keepAspectRatio;
};
