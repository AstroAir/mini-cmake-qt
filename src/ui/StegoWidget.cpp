#include "StegoWidget.hpp"
#include "Def.h"
#include "image/steganograph/Channels.hpp"
#include "image/steganograph/LSB.hpp"
#include "image/steganograph/MSB.hpp"
#include "image/steganograph/Stego.hpp"
#include "image/steganograph/WaterMark.hpp"

#include <QFileDialog>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>

#include "ElaCheckBox.h"
#include "ElaComboBox.h"
#include "ElaDoubleSpinBox.h"
#include "ElaLineEdit.h"
#include "ElaMenu.h"
#include "ElaMenuBar.h"
#include "ElaProgressBar.h"
#include "ElaPushButton.h"
#include "ElaSlider.h"
#include "ElaSpinBox.h"
#include "ElaStatusBar.h"
#include "ElaTabWidget.h"

StegoWindow::StegoWindow(QWidget *parent) : QMainWindow(parent) {
  setWindowTitle(tr("图像隐写工具"));
  setupUI();
  createMenus();
  resize(1200, 800);
}

void StegoWindow::setupUI() {
  // 创建中心部件
  QWidget *centralWidget = new QWidget(this);
  setCentralWidget(centralWidget);

  // 使用QHBoxLayout作为主布局
  QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);
  mainLayout->setContentsMargins(10, 10, 10, 10);
  mainLayout->setSpacing(15);

  // 左侧控制面板
  QWidget *controlPanel = new QWidget;
  QVBoxLayout *controlLayout = new QVBoxLayout(controlPanel);
  controlLayout->setContentsMargins(0, 0, 0, 0);
  controlLayout->setSpacing(10);

  // 设置左侧面板的大小策略
  controlPanel->setMinimumWidth(300);
  controlPanel->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);

  // 右侧预览区域
  QWidget *previewPanel = new QWidget;
  QVBoxLayout *previewLayout = new QVBoxLayout(previewPanel);
  previewLayout->setContentsMargins(0, 0, 0, 0);
  previewLayout->setSpacing(10);

  // 设置预览标签的大小策略
  imagePreviewLabel = new QLabel;
  imagePreviewLabel->setMinimumSize(400, 300);
  imagePreviewLabel->setSizePolicy(QSizePolicy::Expanding,
                                   QSizePolicy::Expanding);
  imagePreviewLabel->setAlignment(Qt::AlignCenter);
  imagePreviewLabel->setFrameStyle(QFrame::Panel | QFrame::Sunken);

  // 优化直方图显示
  histogramLabel = new QLabel;
  histogramLabel->setMinimumHeight(150);
  histogramLabel->setMaximumHeight(200);
  histogramLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  histogramLabel->setFrameStyle(QFrame::Panel | QFrame::Sunken);

  // 添加预览控件
  previewLayout->addWidget(imagePreviewLabel, 4);
  previewLayout->addWidget(histogramLabel, 1);

  // 设置伸缩比例
  mainLayout->addWidget(controlPanel, 1);
  mainLayout->addWidget(previewPanel, 3);

  // 优化控制面板的布局
  setupControlPanel(controlLayout);

  // 添加质量监测面板
  auto qualityMonitor = createQualityMonitorPanel();
  controlLayout->addWidget(qualityMonitor);

  // 添加处理控制选项
  setupProcessingControls();
  setupPreprocessingOptions();

  // 设置通道质量指示器
  setupChannelQualityIndicators();
}

void StegoWindow::setupControlPanel(QVBoxLayout *layout) {
  // 基本操作按钮组
  QGroupBox *operationsGroup = new QGroupBox(tr("基本操作"));
  QVBoxLayout *operationsLayout = new QVBoxLayout(operationsGroup);
  operationsLayout->setSpacing(8);

  loadImageButton = new ElaPushButton(tr("载入图像"));
  saveImageButton = new ElaPushButton(tr("保存图像"));

  // 使用网格布局优化按钮排列
  QGridLayout *buttonGrid = new QGridLayout;
  buttonGrid->addWidget(loadImageButton, 0, 0);
  buttonGrid->addWidget(saveImageButton, 0, 1);
  operationsLayout->addLayout(buttonGrid);

  // 添加显示控件组
  setupDisplayControls();

  // 添加图像信息组
  setupImageInfo();

  // 创建标签页并优化其布局
  setupTabbedPanel();

  // 按合理顺序添加组件
  layout->addWidget(operationsGroup);
  layout->addWidget(displayGroup);
  layout->addWidget(imageInfoGroup);
  layout->addWidget(tabWidget);

  // 添加弹性空间
  layout->addStretch();
}

void StegoWindow::setupDisplayControls() {
  displayGroup = new QGroupBox(tr("显示设置"));
  QGridLayout *displayLayout = new QGridLayout(displayGroup);
  displayLayout->setVerticalSpacing(8);
  displayLayout->setHorizontalSpacing(15);

  // 使用网格布局优化控件排列
  displayModeBox = new ElaComboBox;
  channelBox = new ElaComboBox;
  bitPlaneBox = new ElaSpinBox;

  int row = 0;
  displayLayout->addWidget(new QLabel(tr("显示模式:")), row, 0);
  displayLayout->addWidget(displayModeBox, row++, 1);

  displayLayout->addWidget(new QLabel(tr("通道:")), row, 0);
  displayLayout->addWidget(channelBox, row++, 1);

  displayLayout->addWidget(new QLabel(tr("位平面:")), row, 0);
  displayLayout->addWidget(bitPlaneBox, row, 1);

  // 设置列的拉伸因子
  displayLayout->setColumnStretch(1, 1);
}

void StegoWindow::resizeEvent(QResizeEvent *event) {
  QMainWindow::resizeEvent(event);

  // 计算预览图像的最佳大小
  QSize labelSize = imagePreviewLabel->size();
  if (!originalImage.empty()) {
    double aspectRatio = (double)originalImage.cols / originalImage.rows;
    int newWidth = labelSize.width();
    int newHeight = newWidth / aspectRatio;

    if (newHeight > labelSize.height()) {
      newHeight = labelSize.height();
      newWidth = newHeight * aspectRatio;
    }

    updatePreviewWithSize(QSize(newWidth, newHeight));
  }
}

void StegoWindow::updatePreviewWithSize(const QSize &size) {
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
  case 2: // 差异对比
    cv::Mat diff;
    cv::absdiff(originalImage, processedImage, diff);
    display = matToQImage(diff);
    break;
  }

  imagePreviewLabel->setPixmap(QPixmap::fromImage(display).scaled(
      size, Qt::KeepAspectRatio, Qt::SmoothTransformation));
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
  emit onImageLoaded(); // 添加这一行，发射信号
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
  if (!checkImageValidity())
    return;

  try {
    startOperation(tr("消息嵌入"));

    // 应用预处理
    if (isPreprocessingEnabled) {
      steganograph::preprocess_image(processedImage, channelConfig);
    }

    // 根据选择的模式执行嵌入
    if (isAdaptiveEnabled) {
      steganograph::adaptive_lsb_hide(
          processedImage, lsbMessageEdit->toPlainText().toStdString(),
          channelConfig);
    } else {
      steganograph::multi_channel_hide(
          processedImage, lsbMessageEdit->toPlainText().toStdString(),
          channelConfig);
    }

    // 更新质量监测
    updateQualityMonitor(originalImage, processedImage);
    updateChannelAnalysis();
    updatePreview();

    finishOperation(true);
    showOperationResult(true, tr("消息嵌入成功"));
  } catch (const std::exception &e) {
    finishOperation(false);
    showOperationResult(false, tr("消息嵌入失败: %1").arg(e.what()));
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
  // 替换menuBar
  ElaMenuBar *menuBar = new ElaMenuBar(this);
  setMenuBar(menuBar);

  // 创建文件菜单
  ElaMenu *fileMenu = menuBar->addMenu(ElaIconType::File, tr("文件"));

  // 添加带图标的文件操作
  QAction *openAction = fileMenu->addElaIconAction(
      ElaIconType::BoxOpen, tr("打开图像"), QKeySequence::Open);
  connect(openAction, &QAction::triggered, this, &StegoWindow::loadImage);

  QAction *saveAction = fileMenu->addElaIconAction(
      ElaIconType::FloppyDisk, tr("保存图像"), QKeySequence::Save);
  saveAction->setEnabled(false);
  connect(saveAction, &QAction::triggered, this, &StegoWindow::saveImage);
  connect(this, &StegoWindow::onImageLoaded,
          [saveAction]() { saveAction->setEnabled(true); });

  QAction *exitAction = fileMenu->addElaIconAction(
      ElaIconType::Xmark, tr("退出"), QKeySequence::Quit);
  connect(exitAction, &QAction::triggered, this, &QWidget::close);

  fileMenu->addSeparator();
  fileMenu->addAction(exitAction);

  // 创建编辑菜单
  ElaMenu *editMenu = menuBar->addMenu(ElaIconType::Pen, tr("编辑"));

  QAction *undoAction = editMenu->addElaIconAction(
      ElaIconType::RotateLeft, tr("撤销"), QKeySequence::Undo);
  QAction *redoAction = editMenu->addElaIconAction(
      ElaIconType::RotateReverse, tr("重做"), QKeySequence::Redo);
  QAction *resetAction =
      editMenu->addElaIconAction(ElaIconType::PowerOff, tr("重置图像"),
                                 QKeySequence(Qt::CTRL | Qt::Key_R));

  connect(resetAction, &QAction::triggered, [this]() {
    if (!originalImage.empty()) {
      processedImage = originalImage.clone();
      updatePreview();
    }
  });

  editMenu->addSeparator();
  editMenu->addAction(resetAction);

  // 创建视图菜单
  ElaMenu *viewMenu =
      menuBar->addMenu(ElaIconType::CameraViewfinder, tr("视图"));

  QAction *zoomInAction = viewMenu->addElaIconAction(
      ElaIconType::MagnifyingGlassPlus, tr("放大"), QKeySequence::ZoomIn);
  connect(zoomInAction, &QAction::triggered, [this]() {
    previewSizeSlider->setValue(previewSizeSlider->value() + 10);
  });

  QAction *zoomOutAction = viewMenu->addElaIconAction(
      ElaIconType::MagnifyingGlassMinus, tr("缩小"), QKeySequence::ZoomOut);
  connect(zoomOutAction, &QAction::triggered, [this]() {
    previewSizeSlider->setValue(previewSizeSlider->value() - 10);
  });

  QAction *fitToWindowAction = viewMenu->addElaIconAction(
      ElaIconType::Window, tr("适应窗口"), QKeySequence(Qt::CTRL | Qt::Key_F));
  connect(fitToWindowAction, &QAction::triggered,
          [this]() { previewSizeSlider->setValue(100); });

  // 创建帮助菜单
  ElaMenu *helpMenu = menuBar->addMenu(ElaIconType::Helicopter, tr("帮助"));

  QAction *helpAction = helpMenu->addElaIconAction(
      ElaIconType::HandshakeAngle, tr("使用帮助"), QKeySequence::HelpContents);

  QAction *aboutAction =
      helpMenu->addElaIconAction(ElaIconType::Info, tr("关于"));
  connect(aboutAction, &QAction::triggered, [this]() {
    QMessageBox::about(this, tr("关于图像隐写工具"),
                       tr("图像隐写工具 v1.0\n\n"
                          "一个用于图像隐写和水印嵌入的工具。\n"
                          "支持LSB隐写、傅里叶变换隐写、多通道隐写等功能。"));
  });

  helpMenu->addSeparator();
  helpMenu->addAction(aboutAction);

  // 禁用尚未实现的功能
  undoAction->setEnabled(false);
  redoAction->setEnabled(false);
  helpAction->setEnabled(false);
}

void StegoWindow::setupTabbedPanel() {
  // 创建标签页控件
  tabWidget = new ElaTabWidget;
  tabWidget->setStyleSheet("QTabWidget::pane { border: 1px solid #C2C7CB; }");

  // 添加LSB隐写标签页
  QWidget *lsbPanel = createLSBPanel();
  tabWidget->addTab(lsbPanel, tr("LSB隐写"));

  // 添加傅里叶变换隐写标签页
  QWidget *fourierPanel = createFourierPanel();
  tabWidget->addTab(fourierPanel, tr("傅里叶隐写"));

  // 添加多通道隐写标签页
  QWidget *channelPanel = createChannelPanel();
  tabWidget->addTab(channelPanel, tr("多通道隐写"));

  // 添加水印嵌入标签页
  QWidget *watermarkPanel = createWatermarkPanel();
  tabWidget->addTab(watermarkPanel, tr("水印处理"));

  // 添加MSB压缩标签页
  QWidget *msbPanel = createMSBPanel();
  tabWidget->addTab(msbPanel, tr("MSB压缩"));

  // 连接标签页切换信号
  connect(tabWidget, &QTabWidget::currentChanged, [this](int index) {
    // 切换标签页时更新界面状态
    updatePreview();
    updateStatusInfo();
  });

  // 设置标签页的最小高度
  tabWidget->setMinimumHeight(300);
}

QGroupBox *StegoWindow::createQualityMonitorPanel() {
  QGroupBox *group = new QGroupBox(tr("图像质量监测"));
  QVBoxLayout *layout = new QVBoxLayout(group);

  // PSNR指示器
  psnrLabel = new QLabel(tr("PSNR: N/A"));
  layout->addWidget(psnrLabel);

  // SSIM指示器
  ssimLabel = new QLabel(tr("SSIM: N/A"));
  layout->addWidget(ssimLabel);

  // 容量指示器
  capacityLabel = new QLabel(tr("可用容量: N/A"));
  layout->addWidget(capacityLabel);

  // 质量进度条
  qualityBar = new ElaProgressBar;
  qualityBar->setRange(0, 100);
  layout->addWidget(qualityBar);

  // 隐写检测指示器
  detectionLabel = new QLabel(tr("隐写分析: 未检测"));
  layout->addWidget(detectionLabel);

  return group;
}

void StegoWindow::setupChannelQualityIndicators() {
  channelQualityBars.clear();
  channelQualityLabels.clear();

  QStringList channelNames = {tr("蓝色"), tr("绿色"), tr("红色"), tr("Alpha")};
  for (const auto &name : channelNames) {
    ElaProgressBar *bar = new ElaProgressBar;
    QLabel *label = new QLabel(name + tr(" 通道质量: N/A"));
    channelQualityBars.push_back(bar);
    channelQualityLabels.push_back(label);
  }
}

void StegoWindow::setupProcessingControls() {
  adaptiveProcessingBox = new ElaCheckBox(tr("自适应处理"));
  preserveEdgesBox = new ElaCheckBox(tr("保护边缘"));
  qualityThresholdBox = new ElaDoubleSpinBox;
  qualityThresholdBox->setRange(0.0, 1.0);
  qualityThresholdBox->setValue(0.8);
  qualityThresholdBox->setSingleStep(0.05);

  connect(adaptiveProcessingBox, &ElaCheckBox::toggled, [this](bool checked) {
    isAdaptiveEnabled = checked;
    channelConfig.embedMode = checked ? ChannelConfig::EmbedMode::ADAPTIVE_LSB
                                      : ChannelConfig::EmbedMode::LSB;
  });

  connect(preserveEdgesBox, &ElaCheckBox::toggled,
          [this](bool checked) { channelConfig.preserveEdges = checked; });

  connect(qualityThresholdBox, &ElaDoubleSpinBox::valueChanged,
          [this](double value) { channelConfig.qualityThreshold = value; });
}

void StegoWindow::setupPreprocessingOptions() {
  compressionModeBox = new ElaComboBox;
  compressionModeBox->addItems(
      {tr("无压缩"), tr("Huffman压缩"), tr("LZW压缩")});

  encryptionBox = new ElaCheckBox(tr("启用加密"));
  encryptionKeyEdit = new ElaLineEdit;
  encryptionKeyEdit->setPlaceholderText(tr("加密密钥"));
  encryptionKeyEdit->setEnabled(false);

  connect(compressionModeBox,
          QOverload<int>::of(&ElaComboBox::currentIndexChanged),
          [this](int index) {
            channelConfig.compression =
                static_cast<ChannelConfig::CompressionMode>(index);
          });

  connect(encryptionBox, &ElaCheckBox::toggled, [this](bool checked) {
    channelConfig.useEncryption = checked;
    encryptionKeyEdit->setEnabled(checked);
  });

  connect(encryptionKeyEdit, &ElaLineEdit::textChanged,
          [this](const QString &text) {
            channelConfig.encryptionKey = text.toStdString();
          });
}

void StegoWindow::updateQualityMonitor(const cv::Mat &original,
                                       const cv::Mat &processed) {
  if (original.empty() || processed.empty())
    return;

  // 计算图像质量
  double quality = steganograph::evaluate_image_quality(original, processed);
  currentQualityScore = quality;

  // 更新界面显示
  double psnr = cv::PSNR(original, processed);
  psnrLabel->setText(tr("PSNR: %1 dB").arg(psnr, 0, 'f', 2));

  cv::Scalar ssim = MSSIM(original, processed);
  ssimLabel->setText(tr("SSIM: %1").arg(ssim[0], 0, 'f', 3));

  qualityBar->setValue(quality * 100);

  // 检测是否存在隐写
  double confidence;
  bool hasStego = steganograph::detect_steganography(processed, &confidence);
  detectionLabel->setText(tr("隐写分析: %1 (置信度: %2%)")
                              .arg(hasStego ? tr("已检测到") : tr("未检测到"))
                              .arg(confidence * 100, 0, 'f', 1));
}

void StegoWindow::updateChannelAnalysis() {
  if (originalImage.empty())
    return;

  auto qualities = steganograph::analyze_all_channels_quality(originalImage);
  for (size_t i = 0; i < qualities.size(); ++i) {
    const auto &quality = qualities[i];
    if (i < channelQualityBars.size()) {
      int qualityScore = quality.correlation * 100;
      channelQualityBars[i]->setValue(qualityScore);
      channelQualityLabels[i]->setText(tr("%1 通道质量: %2%")
                                           .arg(i == 0   ? "蓝色"
                                                : i == 1 ? "绿色"
                                                : i == 2 ? "红色"
                                                         : "Alpha")
                                           .arg(qualityScore));
    }
  }
}

void StegoWindow::updateCapacityLabel() {
  if (originalImage.empty())
    return;

  size_t capacity =
      steganograph::calculate_capacity(originalImage, channelConfig);
  capacityLabel->setText(tr("可用容量: %1 字节").arg(capacity));
}
