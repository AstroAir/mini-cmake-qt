#pragma once

#include <QMainWindow>
#include <QProgressBar>
#include <QLabel>
#include <QSpinBox>
#include <QComboBox>
#include <QPushButton>
#include <QSlider>
#include <QCheckBox>
#include <QGroupBox>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

class VideoProcessWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit VideoProcessWindow(QWidget *parent = nullptr);
  ~VideoProcessWindow();

private slots:
  void openVideo();
  void saveVideo();
  void playPause();
  void stopVideo();
  void seekVideo(int position);
  void applyFilter();
  void updateFrame();
  void processVideo();
  void updateProgress(int value);

private:
  void setupUI();
  void setupConnections();
  void setupVideoPlayer();
  void updateControls();
  void displayFrame(const cv::Mat& frame);
  void initializeFilters();

  // UI组件
  QLabel *videoDisplay;
  QProgressBar *progressBar;
  QSlider *timelineSlider;
  QSpinBox *frameSpinBox;
  QComboBox *filterComboBox;
  QPushButton *openButton;
  QPushButton *saveButton;
  QPushButton *playButton;
  QPushButton *stopButton;
  QPushButton *applyButton;
  QCheckBox *useGPUCheckBox;
  QGroupBox *controlGroup;

  // 视频处理参数控件
  QSlider *brightnessSlider;
  QSlider *contrastSlider;
  QSlider *saturationSlider;
  QSpinBox *resizeWidthBox;
  QSpinBox *resizeHeightBox;
  QComboBox *codecComboBox;

  // 视频属性
  cv::VideoCapture videoCapture;
  cv::Mat currentFrame;
  QString currentVideoPath;
  bool isPlaying;
  int totalFrames;
  int currentFrameIndex;
  QTimer *playTimer;

  // 处理参数
  struct ProcessingParams {
    double brightness = 1.0;
    double contrast = 1.0;
    double saturation = 1.0;
    cv::Size newSize;
    QString codec;
    bool useGPU = false;
  } params;
};
