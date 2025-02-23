#include "DiffWidget.h"
#include "CropPreviewWidget.h"

#include <QDateTime>
#include <QFileDialog>
#include <QFuture>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QSettings>
#include <QShortcut>
#include <QSplitter>
#include <QTextStream>
#include <QTimer>
#include <QVBoxLayout>
#include <QtConcurrent>
#include <QContextMenuEvent>

#include "ElaComboBox.h"
#include "ElaDockWidget.h"
#include "ElaDoubleSpinBox.h"
#include "ElaMenu.h"
#include "ElaProgressBar.h"
#include "ElaPushButton.h"
#include "ElaSpinBox.h"
#include "ElaStatusBar.h"
#include "ElaToolButton.h"


DiffWidget::DiffWidget(QWidget *parent)
    : QWidget(parent), differ(std::make_unique<ImageDiff>()),
      splitOrientation(Qt::Horizontal), toolPanelVisible(true),
      statusBarVisible(true) {
  createActions();
  setupUi();
  connectSignals();
  setupShortcuts();
  createMenus();
  loadSettings();
}

void DiffWidget::createActions() {
  actions.splitHorizontal = new QAction(tr("水平分割"), this);
  actions.splitVertical = new QAction(tr("垂直分割"), this);
  actions.toggleToolPanel = new QAction(tr("显示工具面板"), this);
  actions.toggleStatusBar = new QAction(tr("显示状态栏"), this);
  actions.customizeToolbar = new QAction(tr("自定义工具栏"), this);
  actions.resetLayout = new QAction(tr("重置布局"), this);
  actions.exportReport = new QAction(tr("导出报告"), this);

  actions.toggleToolPanel->setCheckable(true);
  actions.toggleStatusBar->setCheckable(true);

  // 设置快捷键
  actions.splitHorizontal->setShortcut(QKeySequence("Ctrl+H"));
  actions.splitVertical->setShortcut(QKeySequence("Ctrl+V"));
  actions.toggleToolPanel->setShortcut(QKeySequence("Ctrl+T"));
  actions.toggleStatusBar->setShortcut(QKeySequence("Ctrl+B"));
}

void DiffWidget::setupUi() { setupLayout(); }

void DiffWidget::setupLayout() {
  auto mainLayout = new QVBoxLayout(this);
  mainLayout->setSpacing(0);
  mainLayout->setContentsMargins(0, 0, 0, 0);

  // 工具栏
  auto toolbarArea = new QWidget;
  auto toolbarLayout = new QHBoxLayout(toolbarArea);
  toolbarLayout->setSpacing(2);
  toolbarLayout->setContentsMargins(5, 2, 5, 2);
  setupToolbar();
  mainLayout->addWidget(toolbarArea);

  // 主分割器
  mainSplitter = new QSplitter(this);
  mainSplitter->setHandleWidth(1);

  // 工具面板
  toolPanelDock = new ElaDockWidget(tr("工具面板"), this);
  toolPanelDock->setFeatures(ElaDockWidget::DockWidgetMovable |
                             ElaDockWidget::DockWidgetFloatable);
  toolPanelDock->setWidget(createToolPanel());

  // 预览区域
  auto previewArea = new QWidget;
  auto previewLayout = new QHBoxLayout(previewArea);
  previewLayout->setContentsMargins(0, 0, 0, 0);

  sourcePreview = new CropPreviewWidget(this);
  targetPreview = new CropPreviewWidget(this);
  diffPreview = new CropPreviewWidget(this);

  mainSplitter->addWidget(sourcePreview);
  mainSplitter->addWidget(targetPreview);
  mainSplitter->addWidget(diffPreview);

  mainLayout->addWidget(mainSplitter, 1);

  // 进度条
  progressBar = new ElaProgressBar;
  progressBar->setVisible(false);
  mainLayout->addWidget(progressBar);

  // 状态栏和按钮
  statusBar = new ElaStatusBar;
  statusBar->setSizeGripEnabled(false);
  mainLayout->addWidget(statusBar);

  auto buttonBox = new QHBoxLayout;
  compareButton = new ElaPushButton(tr("比较"));
  cancelButton = new ElaPushButton(tr("取消"));
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
  zoomInBtn = new ElaToolButton(this);
  zoomOutBtn = new ElaToolButton(this);
  fitViewBtn = new ElaToolButton(this);
  resetBtn = new ElaToolButton(this);
  saveBtn = new ElaToolButton(this);

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

  strategyCombo = new ElaComboBox;
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

  thresholdSpin = new ElaSpinBox;
  thresholdSpin->setRange(0, 255);
  thresholdSpin->setValue(32);

  sensitivitySpin = new ElaDoubleSpinBox;
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

  connect(zoomInBtn, &ElaToolButton::clicked, this, &DiffWidget::onZoomIn);
  connect(zoomOutBtn, &ElaToolButton::clicked, this, &DiffWidget::onZoomOut);
  connect(fitViewBtn, &ElaToolButton::clicked, this, &DiffWidget::onFitToView);
  connect(resetBtn, &ElaToolButton::clicked, this, &DiffWidget::onReset);
  connect(saveBtn, &ElaToolButton::clicked, this, &DiffWidget::onSaveResult);
}

void DiffWidget::setSourceImage(const QImage &image) {
  sourceImage = image;
  sourcePreview->setImage(image); // 假设我们重命名了方法来明确接受 QImage
}

void DiffWidget::setTargetImage(const QImage &image) {
  targetImage = image;
  targetPreview->setImage(image);
}

void DiffWidget::onStrategyChanged(int index) {
  if (sourceImage.isNull() || targetImage.isNull()) {
    return;
  }

  auto promise = std::make_shared<QPromise<ComparisonResult>>();

  try {
    ComparisonResult result;
    switch (index) {
    case 0:
      result =
          differ->compare(sourceImage, targetImage, strategies.pixel, *promise);
      break;
    case 1:
      result =
          differ->compare(sourceImage, targetImage, strategies.ssim, *promise);
      break;
    case 2:
      result = differ->compare(sourceImage, targetImage, strategies.perceptual,
                               *promise);
      break;
    case 3:
      result = differ->compare(sourceImage, targetImage, strategies.histogram,
                               *promise);
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

  auto promise = std::make_shared<QPromise<ComparisonResult>>();
  auto future = promise->future();

  // 设置进度回调
  auto watcher = new QFutureWatcher<ComparisonResult>(this);
  watcher->setFuture(future);
  connect(watcher, &QFutureWatcher<ComparisonResult>::progressValueChanged,
          this, &DiffWidget::updateProgress);

  currentOperation = QtConcurrent::run([this, promise]() {
    try {
      int strategy = strategyCombo->currentIndex();
      ComparisonResult result;

      switch (strategy) {
      case 0:
        result = differ->compare(sourceImage, targetImage, strategies.pixel,
                                 *promise);
        break;
      case 1:
        result = differ->compare(sourceImage, targetImage, strategies.ssim,
                                 *promise);
        break;
      case 2:
        result = differ->compare(sourceImage, targetImage,
                                 strategies.perceptual, *promise);
        break;
      case 3:
        result = differ->compare(sourceImage, targetImage, strategies.histogram,
                                 *promise);
        break;
      }
      promise->addResult(result);
      promise->finish();
      return result;
    } catch (const std::exception &e) {
      promise->setException(std::current_exception());
      handleException(e);
      return ComparisonResult{};
    }
  });

  future.then(this, [this](const ComparisonResult &result) {
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
    diffPreview->setImage(currentResult.differenceImage);
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

void DiffWidget::createMenus() {
  viewMenu = new ElaMenu(tr("视图"));
  viewMenu->addAction(actions.splitHorizontal);
  viewMenu->addAction(actions.splitVertical);
  viewMenu->addSeparator();
  viewMenu->addAction(actions.toggleToolPanel);
  viewMenu->addAction(actions.toggleStatusBar);
  viewMenu->addSeparator();
  viewMenu->addAction(actions.resetLayout);

  toolsMenu = new QMenu(tr("工具"));
  toolsMenu->addAction(actions.customizeToolbar);
  toolsMenu->addAction(actions.exportReport);
}

void DiffWidget::updateLayout(Qt::Orientation orientation) {
  splitOrientation = orientation;
  mainSplitter->setOrientation(orientation);
}

void DiffWidget::loadSettings() {
  QSettings settings;
  settings.beginGroup("DiffWidget");

  splitOrientation = static_cast<Qt::Orientation>(
      settings.value("SplitOrientation", Qt::Horizontal).toInt());
  toolPanelVisible = settings.value("ToolPanelVisible", true).toBool();
  statusBarVisible = settings.value("StatusBarVisible", true).toBool();

  updateLayout(splitOrientation);
  toolPanelDock->setVisible(toolPanelVisible);
  statusBar->setVisible(statusBarVisible);

  settings.endGroup();
}

void DiffWidget::saveSettings() {
  QSettings settings;
  settings.beginGroup("DiffWidget");

  settings.setValue("SplitOrientation", static_cast<int>(splitOrientation));
  settings.setValue("ToolPanelVisible", toolPanelVisible);
  settings.setValue("StatusBarVisible", statusBarVisible);

  settings.endGroup();
}

void DiffWidget::exportReport(const QString &filePath) {
  QFile file(filePath);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
    showError(tr("无法创建报告文件"));
    return;
  }

  QTextStream out(&file);
  writeReportHeader(out);
  writeReportBody(out);
}

void DiffWidget::writeReportHeader(QTextStream &out) {
  out << "# 图像比较报告\n\n";
  out << "生成时间: " << QDateTime::currentDateTime().toString() << "\n\n";
  out << "## 比较参数\n\n";
  out << "- 比较策略: " << strategyCombo->currentText() << "\n";
  out << "- 阈值: " << thresholdSpin->value() << "\n";
  out << "- 灵敏度: " << sensitivitySpin->value() << "\n\n";
}

void DiffWidget::writeReportBody(QTextStream &out) {
  out << "## 比较结果\n\n";
  out << "- 相似度: "
      << QString::number(currentResult.similarityPercent, 'f', 2) << "%\n";
  out << "- 处理时间: " << currentResult.duration.count() << "ms\n";
  out << "- 差异区域数量: " << currentResult.differenceRegions.size() << "\n\n";

  out << "### 差异区域详情\n\n";
  for (const auto &region : currentResult.differenceRegions) {
    out << "- 位置: (" << region.x() << ", " << region.y() << "), ";
    out << "大小: " << region.width() << "x" << region.height() << "\n";
  }
}

QWidget *DiffWidget::createToolPanel() {
  auto panel = new QWidget;
  auto layout = new QVBoxLayout(panel);
  layout->setContentsMargins(2, 2, 2, 2);
  layout->setSpacing(4);

  // 添加策略组和参数组
  layout->addWidget(createStrategyGroup());
  layout->addWidget(createParametersGroup());

  // 添加伸缩项以便工具面板能够合理布局
  layout->addStretch();

  panel->setMinimumWidth(250);
  panel->setMaximumWidth(400);

  return panel;
}

void DiffWidget::onSplitHorizontally() {
    updateLayout(Qt::Horizontal);
}

void DiffWidget::onSplitVertically() {
    updateLayout(Qt::Vertical);
}

void DiffWidget::onToggleToolPanel() {
    toolPanelVisible = !toolPanelVisible;
    toolPanelDock->setVisible(toolPanelVisible);
    actions.toggleToolPanel->setChecked(toolPanelVisible);
}

void DiffWidget::onToggleStatusBar() {
    statusBarVisible = !statusBarVisible;
    statusBar->setVisible(statusBarVisible);
    actions.toggleStatusBar->setChecked(statusBarVisible);
}

void DiffWidget::onCustomizeToolbar() {
    // 显示工具栏自定义对话框
    QMessageBox::information(this, tr("自定义工具栏"), 
        tr("工具栏自定义功能将在后续版本中提供"));
}

void DiffWidget::onResetLayout() {
    // 重置为默认布局
    updateLayout(Qt::Horizontal);
    toolPanelDock->setVisible(true);
    statusBar->setVisible(true);
    actions.toggleToolPanel->setChecked(true);
    actions.toggleStatusBar->setChecked(true);
    
    // 重置分割器大小
    QList<int> sizes;
    int width = mainSplitter->width();
    sizes << width/3 << width/3 << width/3;
    mainSplitter->setSizes(sizes);
}

void DiffWidget::contextMenuEvent(QContextMenuEvent *event) {
    QMenu menu(this);
    menu.addMenu(viewMenu);
    menu.addMenu(toolsMenu);
    menu.exec(event->globalPos());
}

DiffWidget::~DiffWidget() = default;
