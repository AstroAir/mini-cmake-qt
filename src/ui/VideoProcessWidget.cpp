#include "VideoProcessWidget.hpp"
#include "../video/VideoIO.hpp"
#include "../video/filters/VideoFilters.hpp"
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QTimer>
#include <QVBoxLayout>
#include <spdlog/spdlog.h>


VideoProcessWindow::VideoProcessWindow(QWidget *parent)
    : QMainWindow(parent), isPlaying(false), totalFrames(0),
      currentFrameIndex(0) {
  setupUI();
  setupConnections();
  setupVideoPlayer();
  initializeFilters();
}

void VideoProcessWindow::setupUI() {
  // 创建中心部件
  QWidget *centralWidget = new QWidget(this);
  setCentralWidget(centralWidget);

  QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

  // 视频显示区域
  videoDisplay = new QLabel;
  videoDisplay->setMinimumSize(640, 480);
  videoDisplay->setAlignment(Qt::AlignCenter);
  videoDisplay->setStyleSheet("QLabel { background-color: black; }");
  mainLayout->addWidget(videoDisplay);

  // 时间轴控制
  timelineSlider = new QSlider(Qt::Horizontal);
  mainLayout->addWidget(timelineSlider);

  // 控制按钮组
  QHBoxLayout *controlLayout = new QHBoxLayout;
  openButton = new QPushButton("打开视频");
  playButton = new QPushButton("播放");
  stopButton = new QPushButton("停止");
  saveButton = new QPushButton("保存视频");

  controlLayout->addWidget(openButton);
  controlLayout->addWidget(playButton);
  controlLayout->addWidget(stopButton);
  controlLayout->addWidget(saveButton);
  mainLayout->addLayout(controlLayout);

  // 视频处理参数区域
  QGroupBox *processingGroup = new QGroupBox("处理参数");
  QVBoxLayout *processLayout = new QVBoxLayout(processingGroup);

  // 亮度、对比度、饱和度滑块
  QHBoxLayout *brightnessLayout = new QHBoxLayout;
  brightnessLayout->addWidget(new QLabel("亮度:"));
  brightnessSlider = new QSlider(Qt::Horizontal);
  brightnessSlider->setRange(0, 200);
  brightnessSlider->setValue(100);
  brightnessLayout->addWidget(brightnessSlider);
  processLayout->addLayout(brightnessLayout);

  QHBoxLayout *contrastLayout = new QHBoxLayout;
  contrastLayout->addWidget(new QLabel("对比度:"));
  contrastSlider = new QSlider(Qt::Horizontal);
  contrastSlider->setRange(0, 200);
  contrastSlider->setValue(100);
  contrastLayout->addWidget(contrastSlider);
  processLayout->addLayout(contrastLayout);

  QHBoxLayout *saturationLayout = new QHBoxLayout;
  saturationLayout->addWidget(new QLabel("饱和度:"));
  saturationSlider = new QSlider(Qt::Horizontal);
  saturationSlider->setRange(0, 200);
  saturationSlider->setValue(100);
  saturationLayout->addWidget(saturationSlider);
  processLayout->addLayout(saturationLayout);

  // 尺寸调整
  QHBoxLayout *resizeLayout = new QHBoxLayout;
  resizeLayout->addWidget(new QLabel("调整大小:"));
  resizeWidthBox = new QSpinBox;
  resizeWidthBox->setRange(1, 7680);
  resizeHeightBox = new QSpinBox;
  resizeHeightBox->setRange(1, 4320);
  resizeLayout->addWidget(resizeWidthBox);
  resizeLayout->addWidget(new QLabel("x"));
  resizeLayout->addWidget(resizeHeightBox);
  processLayout->addLayout(resizeLayout);

  // 编解码器选择
  QHBoxLayout *codecLayout = new QHBoxLayout;
  codecLayout->addWidget(new QLabel("编码器:"));
  codecComboBox = new QComboBox;
  codecLayout->addWidget(codecComboBox);
  processLayout->addLayout(codecLayout);

  // GPU加速选项
  useGPUCheckBox = new QCheckBox("使用GPU加速");
  processLayout->addWidget(useGPUCheckBox);

  // 滤镜选择
  QHBoxLayout *filterLayout = new QHBoxLayout;
  filterLayout->addWidget(new QLabel("滤镜:"));
  filterComboBox = new QComboBox;
  filterLayout->addWidget(filterComboBox);
  processLayout->addLayout(filterLayout);

  // 应用按钮
  applyButton = new QPushButton("应用处理");
  processLayout->addWidget(applyButton);

  // 进度条
  progressBar = new QProgressBar;
  progressBar->setVisible(false);
  processLayout->addWidget(progressBar);

  mainLayout->addWidget(processingGroup);
}

void VideoProcessWindow::setupConnections() {
  connect(openButton, &QPushButton::clicked, this,
          &VideoProcessWindow::openVideo);
  connect(playButton, &QPushButton::clicked, this,
          &VideoProcessWindow::playPause);
  connect(stopButton, &QPushButton::clicked, this,
          &VideoProcessWindow::stopVideo);
  connect(saveButton, &QPushButton::clicked, this,
          &VideoProcessWindow::saveVideo);
  connect(timelineSlider, &QSlider::valueChanged, this,
          &VideoProcessWindow::seekVideo);
  connect(applyButton, &QPushButton::clicked, this,
          &VideoProcessWindow::processVideo);

  // 参数改变时更新预览
  connect(brightnessSlider, &QSlider::valueChanged, [this](int value) {
    params.brightness = value / 100.0;
    updateFrame();
  });

  connect(contrastSlider, &QSlider::valueChanged, [this](int value) {
    params.contrast = value / 100.0;
    updateFrame();
  });

  connect(saturationSlider, &QSlider::valueChanged, [this](int value) {
    params.saturation = value / 100.0;
    updateFrame();
  });
}

void VideoProcessWindow::setupVideoPlayer() {
  playTimer = new QTimer(this);
  connect(playTimer, &QTimer::timeout, this, &VideoProcessWindow::updateFrame);
  playTimer->setInterval(33); // ~30fps
}

void VideoProcessWindow::initializeFilters() {
  filterComboBox->addItems({"无", "复古", "黑白", "素描", "卡通", "油画"});

  // 添加可用的编解码器
  auto codecs = VideoIO::getAvailableCodecs();
  for (const auto &codec : codecs) {
    codecComboBox->addItem(codec.c_str());
  }
}

void VideoProcessWindow::openVideo() {
  QString fileName = QFileDialog::getOpenFileName(
      this, "打开视频", QString(), "视频文件 (*.mp4 *.avi *.mkv)");

  if (fileName.isEmpty())
    return;

  videoCapture = VideoIO::openVideo(fileName.toStdString());
  if (!videoCapture.isOpened()) {
    QMessageBox::critical(this, "错误", "无法打开视频文件");
    return;
  }

  currentVideoPath = fileName;
  totalFrames = videoCapture.get(cv::CAP_PROP_FRAME_COUNT);
  timelineSlider->setRange(0, totalFrames - 1);

  // 设置默认的调整大小值
  resizeWidthBox->setValue(videoCapture.get(cv::CAP_PROP_FRAME_WIDTH));
  resizeHeightBox->setValue(videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));

  updateFrame();
  updateControls();
}

// ...继续实现其他成员函数...
