#include "StegoWindow.hpp"
#include "Channels.hpp"
#include "LSB.hpp"
#include "MSB.hpp"
#include "Stego.hpp"
#include "WaterMark.hpp"

#include <QFileDialog>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>
#include <QMenuBar>

#include "ElaPushButton.h"
#include "ElaSpinBox.h"
#include "ElaComboBox.h"
#include "ElaCheckBox.h"
#include "ElaDoubleSpinBox.h"
#include "ElaProgressBar.h"
#include "ElaSlider.h"
#include "ElaStatusBar.h"

StegoWindow::StegoWindow(QWidget *parent) : QMainWindow(parent) {
  setWindowTitle(tr("图像隐写工具"));
  setupUI();
  createMenus();
  resize(1200, 800);
}

void StegoWindow::setupUI() {
  QWidget *centralWidget = new QWidget(this);
  setCentralWidget(centralWidget);

  QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);
  mainLayout->setContentsMargins(5, 5, 5, 5);
  mainLayout->setSpacing(10);

  // 左侧控制面板
  QWidget *controlPanel = new QWidget;
  QVBoxLayout *controlLayout = new QVBoxLayout(controlPanel);
  controlLayout->setContentsMargins(0, 0, 0, 0);
  controlLayout->setSpacing(5);

  // 设置最小宽度而不是固定宽度，允许放大
  controlPanel->setMinimumWidth(280);
  controlPanel->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);

  // 基本操作按钮组
  QGroupBox *operationsGroup = new QGroupBox(tr("基本操作"));
  QVBoxLayout *operationsLayout = new QVBoxLayout(operationsGroup);

  loadImageButton = new ElaPushButton(tr("载入图像"));
  saveImageButton = new ElaPushButton(tr("保存图像"));
  operationsLayout->addWidget(loadImageButton);
  operationsLayout->addWidget(saveImageButton);

  controlLayout->addWidget(operationsGroup);

  // 设置显示控件组
  setupDisplayControls();
  controlLayout->addWidget(displayGroup);

  // 设置图像信息组
  setupImageInfo();
  controlLayout->addWidget(imageInfoGroup);

  // 创建标签页
  tabWidget = new QTabWidget;
  tabWidget->addTab(createLSBPanel(), tr("LSB隐写"));
  tabWidget->addTab(createFourierPanel(), tr("傅里叶隐写"));
  tabWidget->addTab(createChannelPanel(), tr("多通道隐写"));
  tabWidget->addTab(createWatermarkPanel(), tr("数字水印"));
  tabWidget->addTab(createMSBPanel(), tr("MSB压缩"));

  controlLayout->addWidget(tabWidget);
  controlLayout->addStretch();

  // 右侧预览区域
  QWidget *previewPanel = new QWidget;
  QVBoxLayout *previewLayout = new QVBoxLayout(previewPanel);
  previewLayout->setContentsMargins(0, 0, 0, 0);
  previewLayout->setSpacing(5);

  imagePreviewLabel = new QLabel;
  imagePreviewLabel->setMinimumSize(400, 300);
  imagePreviewLabel->setSizePolicy(QSizePolicy::Expanding,
                                   QSizePolicy::Expanding);
  imagePreviewLabel->setAlignment(Qt::AlignCenter);
  imagePreviewLabel->setScaledContents(false); // 使用aspectRatioMode替代

  histogramLabel = new QLabel;
  histogramLabel->setMinimumHeight(120);
  histogramLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

  previewLayout->addWidget(imagePreviewLabel, 4);
  previewLayout->addWidget(histogramLabel, 1);

  // 设置分割比例
  mainLayout->addWidget(controlPanel, 1);
  mainLayout->addWidget(previewPanel, 3);

  // 设置状态栏
  setupStatusBar();

  // 设置状态指示器
  setupStatusIndicators();
  setupPreviewControls();

  // 预览定时器设置
  previewTimer = new QTimer(this);
  previewTimer->setInterval(500); // 500ms刷新间隔
  connect(previewTimer, &QTimer::timeout, this, &StegoWindow::updatePreview);

  // 连接信号
  connect(loadImageButton, &ElaPushButton::clicked, this,
          &StegoWindow::loadImage);
  connect(saveImageButton, &ElaPushButton::clicked, this,
          &StegoWindow::saveImage);
  connect(previewModeBox, QOverload<int>::of(&ElaComboBox::currentIndexChanged),
          this, &StegoWindow::updatePreview);
  connect(displayModeBox, QOverload<int>::of(&ElaComboBox::currentIndexChanged),
          this, &StegoWindow::switchDisplayMode);
  connect(channelBox, QOverload<int>::of(&ElaComboBox::currentIndexChanged), this,
          &StegoWindow::updateChannelPreview);
  connect(bitPlaneBox, QOverload<int>::of(&ElaSpinBox::valueChanged), this,
          &StegoWindow::updateBitPlanePreview);
}

void StegoWindow::setupDisplayControls() {
  displayGroup = new QGroupBox(tr("显示设置"));
  QGridLayout *displayLayout = new QGridLayout(displayGroup);

  displayModeBox = new ElaComboBox;
  displayModeBox->addItems({tr("原始图像"), tr("处理结果"), tr("差异对比"),
                            tr("通道视图"), tr("位平面视图"), tr("频谱分析")});

  channelBox = new ElaComboBox;
  channelBox->addItems({tr("全部"), tr("红"), tr("绿"), tr("蓝"), tr("Alpha")});

  bitPlaneBox = new ElaSpinBox;
  bitPlaneBox->setRange(0, 7);

  displayLayout->addWidget(new QLabel(tr("显示模式:")), 0, 0);
  displayLayout->addWidget(displayModeBox, 0, 1);
  displayLayout->addWidget(new QLabel(tr("通道:")), 1, 0);
  displayLayout->addWidget(channelBox, 1, 1);
  displayLayout->addWidget(new QLabel(tr("位平面:")), 2, 0);
  displayLayout->addWidget(bitPlaneBox, 2, 1);
}

QWidget *StegoWindow::createLSBPanel() {
  QWidget *panel = new QWidget(this);
  QVBoxLayout *layout = new QVBoxLayout(panel);

  // 消息输入区域
  QGroupBox *messageGroup = new QGroupBox(tr("隐写消息"), panel);
  QVBoxLayout *messageLayout = new QVBoxLayout(messageGroup);
  lsbMessageEdit = new QTextEdit(messageGroup);
  messageLayout->addWidget(lsbMessageEdit);

  // 操作按钮
  QHBoxLayout *buttonLayout = new QHBoxLayout;
  lsbEmbedButton = new ElaPushButton(tr("嵌入"), panel);
  lsbExtractButton = new ElaPushButton(tr("提取"), panel);
  buttonLayout->addWidget(lsbEmbedButton);
  buttonLayout->addWidget(lsbExtractButton);

  layout->addWidget(messageGroup);
  layout->addLayout(buttonLayout);
  layout->addStretch();

  // 连接信号
  connect(lsbEmbedButton, &ElaPushButton::clicked, this,
          &StegoWindow::embedMessage);
  connect(lsbExtractButton, &ElaPushButton::clicked, this,
          &StegoWindow::extractMessage);

  return panel;
}

QWidget *StegoWindow::createFourierPanel() {
  QWidget *panel = new QWidget(this);
  QVBoxLayout *layout = new QVBoxLayout(panel);

  // 参数设置
  QGroupBox *paramsGroup = new QGroupBox(tr("参数设置"), panel);
  QHBoxLayout *paramsLayout = new QHBoxLayout(paramsGroup);
  QLabel *alphaLabel = new QLabel(tr("嵌入强度:"), paramsGroup);
  alphaSpinBox = new ElaDoubleSpinBox(paramsGroup);
  alphaSpinBox->setRange(0.01, 1.0);
  alphaSpinBox->setValue(0.1);
  alphaSpinBox->setSingleStep(0.01);
  paramsLayout->addWidget(alphaLabel);
  paramsLayout->addWidget(alphaSpinBox);
  paramsLayout->addStretch();

  // 消息区域
  QGroupBox *messageGroup = new QGroupBox(tr("隐写消息"), panel);
  QVBoxLayout *messageLayout = new QVBoxLayout(messageGroup);
  fourierMessageEdit = new QTextEdit(messageGroup);
  messageLayout->addWidget(fourierMessageEdit);

  // 操作按钮
  QHBoxLayout *buttonLayout = new QHBoxLayout;
  fourierEmbedButton = new ElaPushButton(tr("嵌入"), panel);
  fourierExtractButton = new ElaPushButton(tr("提取"), panel);
  buttonLayout->addWidget(fourierEmbedButton);
  buttonLayout->addWidget(fourierExtractButton);

  layout->addWidget(paramsGroup);
  layout->addWidget(messageGroup);
  layout->addLayout(buttonLayout);
  layout->addStretch();

  return panel;
}

QWidget *StegoWindow::createChannelPanel() {
  QWidget *panel = new QWidget(this);
  QVBoxLayout *layout = new QVBoxLayout(panel);

  // 通道选择
  QGroupBox *channelsGroup = new QGroupBox(tr("通道选择"), panel);
  QGridLayout *channelsLayout = new QGridLayout(channelsGroup);

  blueChannelBox = new ElaCheckBox(tr("蓝色通道"), channelsGroup);
  greenChannelBox = new ElaCheckBox(tr("绿色通道"), channelsGroup);
  redChannelBox = new ElaCheckBox(tr("红色通道"), channelsGroup);
  alphaChannelBox = new ElaCheckBox(tr("Alpha通道"), channelsGroup);

  channelsLayout->addWidget(blueChannelBox, 0, 0);
  channelsLayout->addWidget(greenChannelBox, 0, 1);
  channelsLayout->addWidget(redChannelBox, 1, 0);
  channelsLayout->addWidget(alphaChannelBox, 1, 1);

  // 参数设置
  QGroupBox *paramsGroup = new QGroupBox(tr("参数设置"), panel);
  QGridLayout *paramsLayout = new QGridLayout(paramsGroup);

  QLabel *bitsLabel = new QLabel(tr("每通道位数:"), paramsGroup);
  bitsPerChannelBox = new ElaSpinBox(paramsGroup);
  bitsPerChannelBox->setRange(1, 8);
  bitsPerChannelBox->setValue(1);

  QLabel *keyLabel = new QLabel(tr("混淆密钥:"), paramsGroup);
  scrambleKeyBox = new ElaDoubleSpinBox(paramsGroup);
  scrambleKeyBox->setRange(0, 1000);
  scrambleKeyBox->setDecimals(3);

  paramsLayout->addWidget(bitsLabel, 0, 0);
  paramsLayout->addWidget(bitsPerChannelBox, 0, 1);
  paramsLayout->addWidget(keyLabel, 1, 0);
  paramsLayout->addWidget(scrambleKeyBox, 1, 1);

  // 消息区域
  QGroupBox *messageGroup = new QGroupBox(tr("隐写消息"), panel);
  QVBoxLayout *messageLayout = new QVBoxLayout(messageGroup);
  channelMessageEdit = new QTextEdit(messageGroup);
  messageLayout->addWidget(channelMessageEdit);

  // 操作按钮
  QHBoxLayout *buttonLayout = new QHBoxLayout;
  channelEmbedButton = new ElaPushButton(tr("嵌入"), panel);
  channelExtractButton = new ElaPushButton(tr("提取"), panel);
  buttonLayout->addWidget(channelEmbedButton);
  buttonLayout->addWidget(channelExtractButton);

  layout->addWidget(channelsGroup);
  layout->addWidget(paramsGroup);
  layout->addWidget(messageGroup);
  layout->addLayout(buttonLayout);
  layout->addStretch();

  return panel;
}

QWidget *StegoWindow::createWatermarkPanel() {
  QWidget *panel = new QWidget(this);
  QVBoxLayout *layout = new QVBoxLayout(panel);

  // 水印图像预览和加载
  QGroupBox *watermarkGroup = new QGroupBox(tr("水印图像"));
  QVBoxLayout *watermarkLayout = new QVBoxLayout(watermarkGroup);

  watermarkImageLabel = new QLabel;
  watermarkImageLabel->setMinimumSize(200, 200);
  watermarkImageLabel->setAlignment(Qt::AlignCenter);

  loadWatermarkButton = new ElaPushButton(tr("载入水印"));

  watermarkLayout->addWidget(watermarkImageLabel);
  watermarkLayout->addWidget(loadWatermarkButton);

  // 参数控制
  QGroupBox *paramsGroup = new QGroupBox(tr("参数设置"));
  QGridLayout *paramsLayout = new QGridLayout(paramsGroup);

  watermarkAlphaBox = new ElaDoubleSpinBox;
  watermarkAlphaBox->setRange(0.01, 1.0);
  watermarkAlphaBox->setValue(0.1);
  watermarkAlphaBox->setSingleStep(0.01);

  watermarkChannelBox = new ElaComboBox;
  watermarkChannelBox->addItems(
      {tr("所有通道"), tr("亮度通道"), tr("色度通道")});

  watermarkSizeBox = new ElaSpinBox;
  watermarkSizeBox->setRange(16, 256);
  watermarkSizeBox->setValue(64);

  paramsLayout->addWidget(new QLabel(tr("嵌入强度:")), 0, 0);
  paramsLayout->addWidget(watermarkAlphaBox, 0, 1);
  paramsLayout->addWidget(new QLabel(tr("目标通道:")), 1, 0);
  paramsLayout->addWidget(watermarkChannelBox, 1, 1);
  paramsLayout->addWidget(new QLabel(tr("水印大小:")), 2, 0);
  paramsLayout->addWidget(watermarkSizeBox, 2, 1);

  // 操作按钮
  QHBoxLayout *buttonLayout = new QHBoxLayout;
  watermarkEmbedButton = new ElaPushButton(tr("嵌入水印"));
  watermarkExtractButton = new ElaPushButton(tr("提取水印"));
  buttonLayout->addWidget(watermarkEmbedButton);
  buttonLayout->addWidget(watermarkExtractButton);

  layout->addWidget(watermarkGroup);
  layout->addWidget(paramsGroup);
  layout->addLayout(buttonLayout);
  layout->addStretch();

  // 连接信号
  connect(loadWatermarkButton, &ElaPushButton::clicked, this,
          &StegoWindow::loadWatermarkImage);
  connect(watermarkEmbedButton, &ElaPushButton::clicked, this,
          &StegoWindow::embedWatermark);
  connect(watermarkExtractButton, &ElaPushButton::clicked, this,
          &StegoWindow::extractWatermark);

  return panel;
}

QWidget *StegoWindow::createMSBPanel() {
  QWidget *panel = new QWidget(this);
  QVBoxLayout *layout = new QVBoxLayout(panel);

  // 压缩参数
  QGroupBox *paramsGroup = new QGroupBox(tr("压缩参数"));
  QGridLayout *paramsLayout = new QGridLayout(paramsGroup);

  msbKeepBitsBox = new ElaSpinBox;
  msbKeepBitsBox->setRange(1, 8);
  msbKeepBitsBox->setValue(4);

  msbPreviewBox = new ElaCheckBox(tr("实时预览"));
  msbQualityBar = new ElaProgressBar;

  paramsLayout->addWidget(new QLabel(tr("保留位数:")), 0, 0);
  paramsLayout->addWidget(msbKeepBitsBox, 0, 1);
  paramsLayout->addWidget(msbPreviewBox, 1, 0, 1, 2);
  paramsLayout->addWidget(new QLabel(tr("压缩质量:")), 2, 0);
  paramsLayout->addWidget(msbQualityBar, 2, 1);

  // 操作按钮
  QHBoxLayout *buttonLayout = new QHBoxLayout;
  msbCompressButton = new ElaPushButton(tr("执行压缩"));
  msbToggleButton = new ElaPushButton(tr("切换显示"));
  buttonLayout->addWidget(msbCompressButton);
  buttonLayout->addWidget(msbToggleButton);

  layout->addWidget(paramsGroup);
  layout->addLayout(buttonLayout);
  layout->addStretch();

  // 连接信号
  connect(msbCompressButton, &ElaPushButton::clicked, this,
          &StegoWindow::compressMSB);
  connect(msbToggleButton, &ElaPushButton::clicked, this,
          &StegoWindow::toggleMSBView);
  connect(msbPreviewBox, &ElaCheckBox::toggled, this,
          &StegoWindow::updateMSBPreview);
  connect(msbKeepBitsBox, QOverload<int>::of(&ElaSpinBox::valueChanged), this,
          &StegoWindow::updateMSBPreview);

  return panel;
}

// 实现文件操作和处理逻辑
void StegoWindow::loadImage() {
  QString fileName = QFileDialog::getOpenFileName(
      this, tr("打开图像"), QString(), tr("Images (*.png *.jpg *.bmp)"));
  if (fileName.isEmpty())
    return;

  originalImage = cv::imread(fileName.toStdString(), cv::IMREAD_UNCHANGED);
  if (originalImage.empty()) {
    showError(tr("无法加载图像"));
    return;
  }

  processedImage = originalImage.clone();
  saveImageButton->setEnabled(true);
  updatePreview();
  emit onImageLoaded();  // 添加这一行，发射信号
}

void StegoWindow::saveImage() {
  if (processedImage.empty()) {
    showError(tr("没有可保存的图像"));
    return;
  }

  QString fileName = QFileDialog::getSaveFileName(this, tr("保存图像"),
                                                  QString(), tr("PNG (*.png)"));
  if (fileName.isEmpty())
    return;

  if (cv::imwrite(fileName.toStdString(), processedImage)) {
    showSuccess(tr("图像保存成功"));
  } else {
    showError(tr("图像保存失败"));
  }
}

void StegoWindow::embedMessage() {
  if (originalImage.empty()) {
    showError(tr("请先载入图像"));
    return;
  }

  try {
    switch (tabWidget->currentIndex()) {
    case 0: // LSB
      embedLSB(processedImage, lsbMessageEdit->toPlainText().toStdString());
      break;
    case 1: // Fourier
      processedImage = embed_message(
          originalImage, fourierMessageEdit->toPlainText().toStdString(),
          alphaSpinBox->value());
      break;
    case 2: { // Multi-channel
      ChannelConfig config;
      config.useBlue = blueChannelBox->isChecked();
      config.useGreen = greenChannelBox->isChecked();
      config.useRed = redChannelBox->isChecked();
      config.useAlpha = alphaChannelBox->isChecked();
      config.bitsPerChannel = bitsPerChannelBox->value();
      config.scrambleKey = scrambleKeyBox->value();

      steganograph::multi_channel_hide(
          processedImage, channelMessageEdit->toPlainText().toStdString(),
          config);
      break;
    }
    }
    updatePreview();
    showSuccess(tr("消息嵌入成功"));
  } catch (const std::exception &e) {
    showError(tr("嵌入失败: %1").arg(e.what()));
  }
}

void StegoWindow::extractMessage() {
  if (processedImage.empty()) {
    showError(tr("请先载入图像"));
    return;
  }

  try {
    std::string message;
    switch (tabWidget->currentIndex()) {
    case 0: // LSB
      message = extractLSB(processedImage);
      lsbMessageEdit->setText(QString::fromStdString(message));
      break;
    case 1: // Fourier
      message = extract_message(processedImage,
                                fourierMessageEdit->toPlainText().length(),
                                alphaSpinBox->value());
      fourierMessageEdit->setText(QString::fromStdString(message));
      break;
    case 2: { // Multi-channel
      ChannelConfig config;
      config.useBlue = blueChannelBox->isChecked();
      config.useGreen = greenChannelBox->isChecked();
      config.useRed = redChannelBox->isChecked();
      config.useAlpha = alphaChannelBox->isChecked();
      config.bitsPerChannel = bitsPerChannelBox->value();
      config.scrambleKey = scrambleKeyBox->value();

      message = steganograph::multi_channel_extract(
          processedImage, channelMessageEdit->toPlainText().length(), config);
      channelMessageEdit->setText(QString::fromStdString(message));
      break;
    }
    }
    showSuccess(tr("消息提取成功"));
  } catch (const std::exception &e) {
    showError(tr("提取失败: %1").arg(e.what()));
  }
}

void StegoWindow::updatePreview() {
  if (originalImage.empty())
    return;

  QImage display;
  switch (previewModeBox->currentIndex()) {
  case 0: // 原始图像
    display = matToQImage(originalImage);
    break;
  case 1: // 处理结果
    display = matToQImage(processedImage);
    break;
  case 2: { // 差异对比
    cv::Mat diff;
    cv::absdiff(originalImage, processedImage, diff);
    display = matToQImage(diff);
    break;
  }
  }

  // 保持宽高比的图像缩放
  QSize labelSize = imagePreviewLabel->size();
  imagePreviewLabel->setPixmap(QPixmap::fromImage(display).scaled(
      labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

QImage StegoWindow::matToQImage(const cv::Mat &mat) {
  switch (mat.type()) {
  case CV_8UC1:
    return QImage(mat.data, mat.cols, mat.rows, mat.step,
                  QImage::Format_Grayscale8);
  case CV_8UC3:
    return QImage(mat.data, mat.cols, mat.rows, mat.step,
                  QImage::Format_BGR888);
  case CV_8UC4:
    return QImage(mat.data, mat.cols, mat.rows, mat.step,
                  QImage::Format_ARGB32);
  default:
    return QImage();
  }
}

void StegoWindow::showError(const QString &message) {
  QMessageBox::critical(this, tr("错误"), message);
}

void StegoWindow::showSuccess(const QString &message) {
  QMessageBox::information(this, tr("成功"), message);
}

void StegoWindow::updateChannelPreview(int channel) {
  if (originalImage.empty())
    return;

  std::vector<cv::Mat> channels;
  cv::split(processedImage, channels);

  if (channel == 0) { // 显示所有通道
    updatePreviewImage(processedImage);
  } else if (channel <= channels.size()) {
    updatePreviewImage(channels[channel - 1]);
  }
}

void StegoWindow::updateBitPlanePreview(int plane) {
  if (originalImage.empty())
    return;

  cv::Mat bitPlane = getBitPlane(processedImage, plane);
  updatePreviewImage(bitPlane);
}

void StegoWindow::updateStatusInfo() {
  if (originalImage.empty())
    return;

  QString info = tr("尺寸: %1x%2 | 通道: %3 | 格式: %4")
                     .arg(originalImage.cols)
                     .arg(originalImage.rows)
                     .arg(originalImage.channels())
                     .arg(originalImage.depth());

  statusBar->showMessage(info);
}

void StegoWindow::resizeEvent(QResizeEvent *event) {
  QMainWindow::resizeEvent(event);
  updatePreview(); // 重新调整预览图像大小
}

// 水印相关实现
void StegoWindow::loadWatermarkImage() {
  QString fileName = QFileDialog::getOpenFileName(
      this, tr("打开水印图像"), QString(), tr("Images (*.png *.jpg *.bmp)"));

  if (!fileName.isEmpty()) {
    watermarkImage = cv::imread(fileName.toStdString(), cv::IMREAD_GRAYSCALE);
    if (!watermarkImage.empty()) {
      cv::resize(
          watermarkImage, watermarkImage,
          cv::Size(watermarkSizeBox->value(), watermarkSizeBox->value()));
      updateWatermarkPreview();
    }
  }
}

void StegoWindow::embedWatermark() {
  if (!checkImageValidity())
    return;

  startOperation(tr("水印嵌入"));

  try {
    // 更新进度条
    updateProgressBar(20);

    processedImage = ::embedWatermark(originalImage, watermarkImage,
                                      watermarkAlphaBox->value());

    updateProgressBar(80);

    // 计算嵌入质量
    double quality = compareWatermarks(
        watermarkImage,
        ::extractWatermark(processedImage, watermarkAlphaBox->value()));
    updateQualityIndicator(quality);

    updateProgressBar(100);
    finishOperation(true);
    showOperationResult(true, tr("水印嵌入成功"));
  } catch (const std::exception &e) {
    finishOperation(false);
    showOperationResult(false, tr("水印嵌入失败: %1").arg(e.what()));
  }
}

void StegoWindow::extractWatermark() {
  if (processedImage.empty()) {
    showError(tr("请先载入包含水印的图像"));
    return;
  }

  try {
    cv::Mat extractedWatermark = ::extractWatermark(
        processedImage, watermarkAlphaBox->value(), watermarkSizeBox->value());
    watermarkImage = extractedWatermark;
    updateWatermarkPreview();
    showSuccess(tr("水印提取成功"));
  } catch (const std::exception &e) {
    showError(tr("水印提取失败: %1").arg(e.what()));
  }
}

// MSB相关实现
void StegoWindow::compressMSB() {
  if (originalImage.empty()) {
    showError(tr("请先载入图像"));
    return;
  }

  try {
    compressedImage =
        MSBCompressor::compress(originalImage, msbKeepBitsBox->value());
    processedImage = compressedImage;
    updatePreview();
    showSuccess(tr("MSB压缩成功"));
  } catch (const std::exception &e) {
    showError(tr("MSB压缩失败: %1").arg(e.what()));
  }
}

void StegoWindow::toggleMSBView() {
  msbPreviewMode = !msbPreviewMode;
  updatePreview();
}

void StegoWindow::updateMSBPreview() {
  if (!msbPreviewBox->isChecked() || originalImage.empty())
    return;

  try {
    compressedImage =
        MSBCompressor::compress(originalImage, msbKeepBitsBox->value());
    processedImage = compressedImage;
    updatePreview();
  } catch (const std::exception &e) {
    // 实时预览失败时不显示错误消息
  }
}

void StegoWindow::setupStatusIndicators() {
  QWidget *statusWidget = new QWidget;
  QHBoxLayout *statusLayout = new QHBoxLayout(statusWidget);
  statusLayout->setContentsMargins(0, 0, 0, 0);

  statusLabel = new QLabel;
  qualityIndicator = new QLabel;
  operationProgress = new ElaProgressBar;
  operationProgress->setTextVisible(true);
  operationProgress->setRange(0, 100);
  operationProgress->hide();

  cancelButton = new ElaPushButton(tr("取消"));
  cancelButton->hide();

  statusLayout->addWidget(statusLabel, 1);
  statusLayout->addWidget(qualityIndicator);
  statusLayout->addWidget(operationProgress);
  statusLayout->addWidget(cancelButton);

  statusBar->addWidget(statusWidget, 1);
}

void StegoWindow::setupPreviewControls() {
  previewGroup = new QGroupBox(tr("预览控制"));
  QVBoxLayout *previewLayout = new QVBoxLayout(previewGroup);

  QHBoxLayout *buttonLayout = new QHBoxLayout;
  originalPreviewButton = new ElaPushButton(tr("原图"));
  processedPreviewButton = new ElaPushButton(tr("处理结果"));
  buttonLayout->addWidget(originalPreviewButton);
  buttonLayout->addWidget(processedPreviewButton);

  autoPreviewBox = new ElaCheckBox(tr("自动预览"));
  previewSizeSlider = new ElaSlider(Qt::Horizontal);
  previewSizeSlider->setRange(50, 200);
  previewSizeSlider->setValue(100);

  previewLayout->addLayout(buttonLayout);
  previewLayout->addWidget(autoPreviewBox);
  previewLayout->addWidget(previewSizeSlider);

  connect(originalPreviewButton, &ElaPushButton::clicked, this,
          &StegoWindow::previewOriginalImage);
  connect(processedPreviewButton, &ElaPushButton::clicked, this,
          &StegoWindow::previewProcessedImage);
  connect(previewSizeSlider, &QSlider::valueChanged, this,
          &StegoWindow::updatePreview);
}

void StegoWindow::startOperation(const QString &operation) {
  isProcessing = true;
  currentOperation = operation;
  processingTimer.start();

  operationProgress->show();
  cancelButton->show();
  enableControls(false);

  updateStatusMessage(tr("正在处理: %1...").arg(operation));
  emit onProcessingStarted();
}

void StegoWindow::finishOperation(bool success) {
  isProcessing = false;
  int elapsed = processingTimer.elapsed();

  operationProgress->hide();
  cancelButton->hide();
  enableControls(true);

  QString timeStr = formatProcessingTime();
  QString resultStr = success ? tr("成功") : tr("失败");

  updateStatusMessage(
      tr("%1 %2 - 耗时: %3").arg(currentOperation).arg(resultStr).arg(timeStr));

  if (success) {
    updateImageQuality();
    updatePreview();
  }

  emit onProcessingFinished();
}

void StegoWindow::updateQualityIndicator(double quality) {
  QString qualityText;
  QString styleSheet;

  if (quality >= 0.9) {
    qualityText = tr("极好");
    styleSheet = "color: green;";
  } else if (quality >= 0.7) {
    qualityText = tr("良好");
    styleSheet = "color: darkgreen;";
  } else if (quality >= 0.5) {
    qualityText = tr("一般");
    styleSheet = "color: orange;";
  } else {
    qualityText = tr("较差");
    styleSheet = "color: red;";
  }

  qualityIndicator->setText(tr("质量: %1").arg(qualityText));
  qualityIndicator->setStyleSheet(styleSheet);
}

void StegoWindow::showOperationResult(bool success, const QString &message) {
  if (success) {
    showSuccess(message);
  } else {
    showError(message);
  }
}

bool StegoWindow::checkImageValidity() {
  if (originalImage.empty()) {
    showError(tr("请先载入图像"));
    return false;
  }

  if (watermarkImage.empty() && tabWidget->currentIndex() == 3) {
    showError(tr("请先载入水印图像"));
    return false;
  }

  return true;
}

void StegoWindow::createMenus() {
    // 创建文件菜单
    QMenu *fileMenu = menuBar()->addMenu(tr("文件"));
    
    QAction *openAction = new QAction(tr("打开图像"), this);
    openAction->setShortcut(QKeySequence::Open);
    connect(openAction, &QAction::triggered, this, &StegoWindow::loadImage);
    
    QAction *saveAction = new QAction(tr("保存图像"), this);
    saveAction->setShortcut(QKeySequence::Save);
    saveAction->setEnabled(false);
    connect(saveAction, &QAction::triggered, this, &StegoWindow::saveImage);
    connect(this, &StegoWindow::onImageLoaded, [saveAction]() {
        saveAction->setEnabled(true);
    });
    
    QAction *exitAction = new QAction(tr("退出"), this);
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);
    
    fileMenu->addAction(openAction);
    fileMenu->addAction(saveAction);
    fileMenu->addSeparator();
    fileMenu->addAction(exitAction);

    // 创建编辑菜单
    QMenu *editMenu = menuBar()->addMenu(tr("编辑"));
    
    QAction *undoAction = new QAction(tr("撤销"), this);
    undoAction->setShortcut(QKeySequence::Undo);
    
    QAction *redoAction = new QAction(tr("重做"), this);
    redoAction->setShortcut(QKeySequence::Redo);
    
    QAction *resetAction = new QAction(tr("重置图像"), this);
    resetAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_R));
    connect(resetAction, &QAction::triggered, [this]() {
        if (!originalImage.empty()) {
            processedImage = originalImage.clone();
            updatePreview();
        }
    });
    
    editMenu->addAction(undoAction);
    editMenu->addAction(redoAction);
    editMenu->addSeparator();
    editMenu->addAction(resetAction);

    // 创建视图菜单
    QMenu *viewMenu = menuBar()->addMenu(tr("视图"));
    
    QAction *zoomInAction = new QAction(tr("放大"), this);
    zoomInAction->setShortcut(QKeySequence::ZoomIn);
    connect(zoomInAction, &QAction::triggered, [this]() {
        previewSizeSlider->setValue(previewSizeSlider->value() + 10);
    });
    
    QAction *zoomOutAction = new QAction(tr("缩小"), this);
    zoomOutAction->setShortcut(QKeySequence::ZoomOut);
    connect(zoomOutAction, &QAction::triggered, [this]() {
        previewSizeSlider->setValue(previewSizeSlider->value() - 10);
    });
    
    QAction *fitToWindowAction = new QAction(tr("适应窗口"), this);
    fitToWindowAction->setShortcut(Qt::CTRL | Qt::Key_F);
    connect(fitToWindowAction, &QAction::triggered, [this]() {
        previewSizeSlider->setValue(100);
    });
    
    viewMenu->addAction(zoomInAction);
    viewMenu->addAction(zoomOutAction);
    viewMenu->addAction(fitToWindowAction);

    // 创建帮助菜单
    QMenu *helpMenu = menuBar()->addMenu(tr("帮助"));
    
    QAction *aboutAction = new QAction(tr("关于"), this);
    connect(aboutAction, &QAction::triggered, [this]() {
        QMessageBox::about(this, tr("关于图像隐写工具"),
            tr("图像隐写工具 v1.0\n\n"
               "一个用于图像隐写和水印嵌入的工具。\n"
               "支持LSB隐写、傅里叶变换隐写、多通道隐写等功能。"));
    });
    
    QAction *helpAction = new QAction(tr("使用帮助"), this);
    helpAction->setShortcut(QKeySequence::HelpContents);
    
    helpMenu->addAction(helpAction);
    helpMenu->addSeparator();
    helpMenu->addAction(aboutAction);

    // 禁用尚未实现的功能
    undoAction->setEnabled(false);
    redoAction->setEnabled(false);
    helpAction->setEnabled(false);
}
