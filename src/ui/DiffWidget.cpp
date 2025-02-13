#include "DiffWidget.h"
#include "CropPreviewWidget.h"

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFuture>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QShortcut>
#include <QSpinBox>
#include <QStackedWidget>
#include <QStatusBar>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QtConcurrent>


DiffWidget::DiffWidget(QWidget *parent)
    : QWidget(parent), differ(std::make_unique<ImageDiff>()) {
  setupUi();
  connectSignals();
  setupShortcuts();
}

void DiffWidget::setupUi() { setupLayout(); }

void DiffWidget::setupLayout() {
  auto mainLayout = new QVBoxLayout(this);
  mainLayout->setSpacing(5);
  mainLayout->setContentsMargins(5, 5, 5, 5);

  // 工具栏
  auto toolbarArea = new QWidget;
  auto toolbarLayout = new QHBoxLayout(toolbarArea);
  toolbarLayout->setSpacing(2);
  toolbarLayout->setContentsMargins(0, 0, 0, 0);
  setupToolbar();
  mainLayout->addWidget(toolbarArea);

  // 主区域
  stackedWidget = new QStackedWidget;
  auto mainArea = new QWidget;
  auto mainAreaLayout = new QHBoxLayout(mainArea);

  // 左侧工具面板
  auto toolPanel = new QWidget;
  auto toolPanelLayout = new QVBoxLayout(toolPanel);
  toolPanelLayout->addWidget(createStrategyGroup());
  toolPanelLayout->addWidget(createParametersGroup());
  toolPanelLayout->addStretch();

  toolPanel->setFixedWidth(250);
  mainAreaLayout->addWidget(toolPanel);

  // 预览区域
  auto previewArea = new QWidget;
  auto previewLayout = new QGridLayout(previewArea);

  sourcePreview = new CropPreviewWidget(this);
  targetPreview = new CropPreviewWidget(this);
  diffPreview = new CropPreviewWidget(this);

  previewLayout->addWidget(new QLabel(tr("源图像")), 0, 0);
  previewLayout->addWidget(new QLabel(tr("目标图像")), 0, 1);
  previewLayout->addWidget(new QLabel(tr("差异结果")), 0, 2);
  previewLayout->addWidget(sourcePreview, 1, 0);
  previewLayout->addWidget(targetPreview, 1, 1);
  previewLayout->addWidget(diffPreview, 1, 2);

  mainAreaLayout->addWidget(previewArea, 1);

  stackedWidget->addWidget(mainArea);
  mainLayout->addWidget(stackedWidget, 1);

  // 进度条
  progressBar = new QProgressBar;
  progressBar->setVisible(false);
  mainLayout->addWidget(progressBar);

  // 状态栏和按钮
  statusBar = new QStatusBar;
  statusBar->setSizeGripEnabled(false);
  mainLayout->addWidget(statusBar);

  auto buttonBox = new QHBoxLayout;
  compareButton = new QPushButton(tr("比较"));
  cancelButton = new QPushButton(tr("取消"));
  buttonBox->addStretch();
  buttonBox->addWidget(compareButton);
  buttonBox->addWidget(cancelButton);
  mainLayout->addLayout(buttonBox);

  statusTimer = new QTimer(this);
  statusTimer->setSingleShot(true);
  connect(statusTimer, &QTimer::timeout, this,
          [this]() { statusBar->clearMessage(); });
}

void DiffWidget::setupToolbar() {
  zoomInBtn = new QToolButton(this);
  zoomOutBtn = new QToolButton(this);
  fitViewBtn = new QToolButton(this);
  resetBtn = new QToolButton(this);
  saveBtn = new QToolButton(this);

  zoomInBtn->setIcon(QIcon::fromTheme("zoom-in"));
  zoomOutBtn->setIcon(QIcon::fromTheme("zoom-out"));
  fitViewBtn->setIcon(QIcon::fromTheme("zoom-fit-best"));
  resetBtn->setIcon(QIcon::fromTheme("edit-undo"));
  saveBtn->setIcon(QIcon::fromTheme("document-save"));

  auto toolbar = new QHBoxLayout;
  toolbar->addWidget(zoomInBtn);
  toolbar->addWidget(zoomOutBtn);
  toolbar->addWidget(fitViewBtn);
  toolbar->addWidget(resetBtn);
  toolbar->addWidget(saveBtn);
  toolbar->addStretch();
}

QWidget *DiffWidget::createStrategyGroup() {
  auto group = new QGroupBox(tr("比较策略"));
  auto layout = new QVBoxLayout(group);

  strategyCombo = new QComboBox;
  strategyCombo->addItem(tr("像素差异"));
  strategyCombo->addItem(tr("结构相似性"));
  strategyCombo->addItem(tr("感知哈希"));
  strategyCombo->addItem(tr("直方图比较"));

  layout->addWidget(strategyCombo);

  return group;
}

QWidget *DiffWidget::createParametersGroup() {
  auto group = new QGroupBox(tr("参数设置"));
  auto layout = new QGridLayout(group);

  thresholdSpin = new QSpinBox;
  thresholdSpin->setRange(0, 255);
  thresholdSpin->setValue(32);

  sensitivitySpin = new QDoubleSpinBox;
  sensitivitySpin->setRange(0.1, 10.0);
  sensitivitySpin->setValue(1.0);
  sensitivitySpin->setSingleStep(0.1);

  layout->addWidget(new QLabel(tr("阈值:")), 0, 0);
  layout->addWidget(thresholdSpin, 0, 1);
  layout->addWidget(new QLabel(tr("灵敏度:")), 1, 0);
  layout->addWidget(sensitivitySpin, 1, 1);

  return group;
}

void DiffWidget::setupShortcuts() {
  auto zoomInShortcut = new QShortcut(QKeySequence::ZoomIn, this);
  auto zoomOutShortcut = new QShortcut(QKeySequence::ZoomOut, this);
  auto resetShortcut = new QShortcut(QKeySequence("Ctrl+R"), this);
  auto saveShortcut = new QShortcut(QKeySequence("Ctrl+S"), this);

  connect(zoomInShortcut, &QShortcut::activated, this, &DiffWidget::onZoomIn);
  connect(zoomOutShortcut, &QShortcut::activated, this, &DiffWidget::onZoomOut);
  connect(resetShortcut, &QShortcut::activated, this, &DiffWidget::onReset);
  connect(saveShortcut, &QShortcut::activated, this, &DiffWidget::onSaveResult);
}

void DiffWidget::connectSignals() {
  connect(strategyCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &DiffWidget::onStrategyChanged);
  connect(compareButton, &QPushButton::clicked, this,
          &DiffWidget::onCompareClicked);
  connect(cancelButton, &QPushButton::clicked, this,
          &DiffWidget::onCancelClicked);

  connect(zoomInBtn, &QToolButton::clicked, this, &DiffWidget::onZoomIn);
  connect(zoomOutBtn, &QToolButton::clicked, this, &DiffWidget::onZoomOut);
  connect(fitViewBtn, &QToolButton::clicked, this, &DiffWidget::onFitToView);
  connect(resetBtn, &QToolButton::clicked, this, &DiffWidget::onReset);
  connect(saveBtn, &QToolButton::clicked, this, &DiffWidget::onSaveResult);
}

void DiffWidget::setSourceImage(const QImage &image) {
  sourceImage = image;
  sourcePreview->setQImage(image); // 假设我们重命名了方法来明确接受 QImage
}

void DiffWidget::setTargetImage(const QImage &image) {
  targetImage = image;
  targetPreview->setQImage(image);
}

void DiffWidget::onStrategyChanged(int index) {
  try {
    ComparisonResult result;
    switch (index) {
    case 0:
      result = strategies.pixel.compare(sourceImage, targetImage);
      break;
    case 1:
      result = strategies.ssim.compare(sourceImage, targetImage);
      break;
    case 2:
      result = strategies.perceptual.compare(sourceImage, targetImage);
      break;
    case 3:
      result = strategies.histogram.compare(sourceImage, targetImage);
      break;
    }
    currentResult = result;
    updatePreview();
  } catch (const std::exception &e) {
    handleException(e);
  }
}

void DiffWidget::onCompareClicked() {
  if (sourceImage.isNull() || targetImage.isNull()) {
    showError(tr("请先加载源图像和目标图像"));
    return;
  }

  setState(DiffState::Processing);
  progressBar->setVisible(true);

  currentOperation = QtConcurrent::run([this]() {
    try {
      ComparisonResult result;
      int strategy = strategyCombo->currentIndex();
      switch (strategy) {
      case 0:
        result = strategies.pixel.compare(sourceImage, targetImage);
        break;
      case 1:
        result = strategies.ssim.compare(sourceImage, targetImage);
        break;
      case 2:
        result = strategies.perceptual.compare(sourceImage, targetImage);
        break;
      case 3:
        result = strategies.histogram.compare(sourceImage, targetImage);
        break;
      }
      return result;
    } catch (const std::exception &e) {
      handleException(e);
      return ComparisonResult{};
    }
  });

  currentOperation.then(this, [this](const ComparisonResult &result) {
    currentResult = result;
    updatePreview();
    setState(DiffState::Ready);
    progressBar->setVisible(false);
    emit diffFinished(result);
  });
}

void DiffWidget::onCancelClicked() {
  if (currentOperation.isRunning()) {
    currentOperation.cancel();
  }
  emit diffCanceled();
}

void DiffWidget::updatePreview() {
  if (!currentResult.differenceImage.isNull()) {
    diffPreview->setQImage(currentResult.differenceImage);
    updateStatus(
        tr("相似度: %1%").arg(currentResult.similarityPercent, 0, 'f', 2));
  }
}

void DiffWidget::setState(DiffState state) {
  currentState = state;
  switch (state) {
  case DiffState::Ready:
    compareButton->setEnabled(true);
    progressBar->setVisible(false);
    break;
  case DiffState::Processing:
    compareButton->setEnabled(false);
    progressBar->setVisible(true);
    break;
  case DiffState::Error:
    compareButton->setEnabled(true);
    progressBar->setVisible(false);
    statusBar->showMessage(lastError, 5000);
    break;
  }
}

void DiffWidget::handleException(const std::exception &e) {
  lastError = QString::fromUtf8(e.what());
  setState(DiffState::Error);
  QMessageBox::critical(this, tr("错误"), lastError);
}

void DiffWidget::updateStatus(const QString &message) {
  statusBar->showMessage(message, 3000);
}

void DiffWidget::showError(const QString &message) {
  lastError = message;
  setState(DiffState::Error);
}

void DiffWidget::updateProgress(int value) { progressBar->setValue(value); }

void DiffWidget::onZoomIn() {
  sourcePreview->zoomIn();
  targetPreview->zoomIn();
  diffPreview->zoomIn();
}

void DiffWidget::onZoomOut() {
  sourcePreview->zoomOut();
  targetPreview->zoomOut();
  diffPreview->zoomOut();
}

void DiffWidget::onFitToView() {
  sourcePreview->fitToView();
  targetPreview->fitToView();
  diffPreview->fitToView();
}

void DiffWidget::onReset() {
  sourcePreview->resetView();
  targetPreview->resetView();
  diffPreview->resetView();
  currentResult = ComparisonResult{};
  updatePreview();
}

void DiffWidget::onSaveResult() {
  if (currentResult.differenceImage.isNull()) {
    showError(tr("没有可保存的结果"));
    return;
  }

  QString fileName =
      QFileDialog::getSaveFileName(this, tr("保存差异结果"), QString(),
                                   tr("Images (*.png *.jpg);;All Files (*)"));

  if (!fileName.isEmpty()) {
    if (!currentResult.differenceImage.save(fileName)) {
      showError(tr("保存失败"));
    } else {
      updateStatus(tr("结果已保存到: %1").arg(fileName));
    }
  }
}

ComparisonResult DiffWidget::getResult() const { return currentResult; }

DiffWidget::~DiffWidget() = default;
