#include "CropWidget.h"
#include "CropPreviewWidget.h"

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QShortcut>
#include <QSlider>
#include <QSpinBox>
#include <QStackedWidget>
#include <QStatusBar>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>

CropWidget::CropWidget(QWidget *parent)
    : QWidget(parent), cropper(std::make_unique<ImageCropper>()) {
  setupUi();
  connectSignals();
  setupShortcuts();
}

void CropWidget::setupUi() {
  setupLayout();
  createActions();
}

void CropWidget::setupLayout() {
  auto mainLayout = new QVBoxLayout(this);
  mainLayout->setSpacing(5);
  mainLayout->setContentsMargins(5, 5, 5, 5);

  // 工具栏区域
  auto toolbarArea = new QWidget;
  auto toolbarLayout = new QHBoxLayout(toolbarArea);
  toolbarLayout->setSpacing(2);
  toolbarLayout->setContentsMargins(0, 0, 0, 0);
  setupToolbar();
  mainLayout->addWidget(toolbarArea);

  // 主操作区域
  stackedWidget = new QStackedWidget;
  auto mainArea = new QWidget;
  auto mainAreaLayout = new QHBoxLayout(mainArea);

  // 左侧工具面板
  auto toolPanel = new QWidget;
  auto toolPanelLayout = new QVBoxLayout(toolPanel);
  toolPanelLayout->addWidget(createStrategyGroup());
  toolPanelLayout->addWidget(createAdjustmentGroup());
  toolPanelLayout->addWidget(createPresetsGroup());
  toolPanelLayout->addStretch();

  // 设置左侧面板固定宽度
  toolPanel->setFixedWidth(250);
  mainAreaLayout->addWidget(toolPanel);

  // 中间预览区域
  auto previewArea = new QWidget;
  auto previewLayout = new QVBoxLayout(previewArea);
  previewWidget = new CropPreviewWidget(this);
  previewLayout->addWidget(previewWidget);
  mainAreaLayout->addWidget(previewArea, 1);

  stackedWidget->addWidget(mainArea);
  mainLayout->addWidget(stackedWidget, 1);

  // 底部状态栏和按钮区域
  statusBar = new QStatusBar;
  statusBar->setSizeGripEnabled(false);
  mainLayout->addWidget(statusBar);

  auto buttonBox = new QHBoxLayout;
  buttonBox->addStretch();
  buttonBox->addWidget(cropButton);
  buttonBox->addWidget(cancelButton);
  mainLayout->addLayout(buttonBox);

  // 设置状态更新定时器
  statusTimer = new QTimer(this);
  statusTimer->setSingleShot(true);
  connect(statusTimer, &QTimer::timeout, this,
          [this]() { statusBar->clearMessage(); });
}

void CropWidget::createActions() {
  actions.undo = new QAction(tr("撤销"), this);
  actions.undo->setShortcut(QKeySequence::Undo);
  actions.undo->setEnabled(false);

  actions.redo = new QAction(tr("重做"), this);
  actions.redo->setShortcut(QKeySequence::Redo);
  actions.redo->setEnabled(false);

  actions.reset = new QAction(tr("重置"), this);
  actions.reset->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_R));

  actions.help = new QAction(tr("帮助"), this);
  actions.help->setShortcut(QKeySequence::HelpContents);

  // 撤销动作
  connect(actions.undo, &QAction::triggered, this, [this]() {
    if (!undoStack.empty()) {
      redoStack.push_back(getCurrentStrategy());
      previewWidget->setStrategy(undoStack.back());
      undoStack.pop_back();
      updateUndoRedoState();
      updatePreview();
    }
  });

  // 重做动作
  connect(actions.redo, &QAction::triggered, this, [this]() {
    if (!redoStack.empty()) {
      undoStack.push_back(getCurrentStrategy());
      previewWidget->setStrategy(redoStack.back());
      redoStack.pop_back();
      updateUndoRedoState();
      updatePreview();
    }
  });

  // 重置动作
  connect(actions.reset, &QAction::triggered, this, [this]() {
    if (confirmOperation()) {
      undoStack.clear();
      redoStack.clear();
      updateUndoRedoState();
      onResetCrop();
    }
  });

  // 帮助动作
  connect(actions.help, &QAction::triggered, this, [this]() {
    QMessageBox::information(this, tr("帮助"),
                             tr("快捷键说明:\n"
                                "Ctrl+Z: 撤销\n"
                                "Ctrl+Y: 重做\n"
                                "Ctrl+R: 重置\n"
                                "Ctrl+L: 向左旋转\n"
                                "Ctrl+R: 向右旋转\n"
                                "+/-: 缩放\n"
                                "空格: 适应视图"));
  });
}

void CropWidget::setState(CropState state) {
  currentState = state;
  switch (state) {
  case CropState::Ready:
    cropButton->setEnabled(true);
    statusBar->clearMessage();
    break;
  case CropState::Processing:
    cropButton->setEnabled(false);
    statusBar->showMessage(tr("处理中..."));
    break;
  case CropState::Error:
    cropButton->setEnabled(true);
    statusBar->showMessage(lastError, 5000);
    break;
  }
}

void CropWidget::handleException(const std::exception &e) {
  lastError = QString::fromUtf8(e.what());
  setState(CropState::Error);
  QMessageBox::critical(this, tr("错误"), lastError);
}

void CropWidget::updateStatus(const QString &message) {
  statusBar->showMessage(message, 3000);
}

bool CropWidget::confirmOperation() {
  if (currentState == CropState::Processing) {
    auto result = QMessageBox::question(this, tr("确认"),
                                        tr("当前正在处理中，确定要继续吗？"),
                                        QMessageBox::Yes | QMessageBox::No);
    return result == QMessageBox::Yes;
  }
  return true;
}

void CropWidget::setupToolbar() {
  rotateLeftBtn = new QToolButton(this);
  rotateRightBtn = new QToolButton(this);
  zoomInBtn = new QToolButton(this);
  zoomOutBtn = new QToolButton(this);
  fitViewBtn = new QToolButton(this);
  resetBtn = new QToolButton(this);
  autoDetectBtn = new QToolButton(this);

  rotateLeftBtn->setIcon(QIcon::fromTheme("object-rotate-left"));
  rotateRightBtn->setIcon(QIcon::fromTheme("object-rotate-right"));
  zoomInBtn->setIcon(QIcon::fromTheme("zoom-in"));
  zoomOutBtn->setIcon(QIcon::fromTheme("zoom-out"));
  fitViewBtn->setIcon(QIcon::fromTheme("zoom-fit-best"));
  resetBtn->setIcon(QIcon::fromTheme("edit-undo"));
  autoDetectBtn->setIcon(QIcon::fromTheme("edit-find"));

  auto toolbar = new QHBoxLayout;
  toolbar->addWidget(rotateLeftBtn);
  toolbar->addWidget(rotateRightBtn);
  toolbar->addWidget(zoomInBtn);
  toolbar->addWidget(zoomOutBtn);
  toolbar->addWidget(fitViewBtn);
  toolbar->addWidget(resetBtn);
  toolbar->addWidget(autoDetectBtn);
  toolbar->addStretch();
}

void CropWidget::setupShortcuts() {
  auto rotateLeftShortcut = new QShortcut(QKeySequence("Ctrl+L"), this);
  auto rotateRightShortcut = new QShortcut(QKeySequence("Ctrl+R"), this);
  auto zoomInShortcut = new QShortcut(QKeySequence::ZoomIn, this);
  auto zoomOutShortcut = new QShortcut(QKeySequence::ZoomOut, this);
  auto resetShortcut = new QShortcut(QKeySequence("Ctrl+Reset"), this);

  connect(rotateLeftShortcut, &QShortcut::activated, this,
          &CropWidget::onRotateLeft);
  connect(rotateRightShortcut, &QShortcut::activated, this,
          &CropWidget::onRotateRight);
  connect(zoomInShortcut, &QShortcut::activated, this, &CropWidget::onZoomIn);
  connect(zoomOutShortcut, &QShortcut::activated, this, &CropWidget::onZoomOut);
  connect(resetShortcut, &QShortcut::activated, this, &CropWidget::onResetCrop);
}

void CropWidget::connectSignals() {
  connect(strategyCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &CropWidget::onStrategyChanged);
  connect(cropButton, &QPushButton::clicked, this, &CropWidget::onCropClicked);
  connect(cancelButton, &QPushButton::clicked, this,
          &CropWidget::onCancelClicked);
  connect(previewWidget, &CropPreviewWidget::strategyChanged, this,
          &CropWidget::updatePreview);

  connect(rotateLeftBtn, &QToolButton::clicked, this,
          &CropWidget::onRotateLeft);
  connect(rotateRightBtn, &QToolButton::clicked, this,
          &CropWidget::onRotateRight);
  connect(zoomInBtn, &QToolButton::clicked, this, &CropWidget::onZoomIn);
  connect(zoomOutBtn, &QToolButton::clicked, this, &CropWidget::onZoomOut);
  connect(fitViewBtn, &QToolButton::clicked, this, &CropWidget::onFitToView);
  connect(resetBtn, &QToolButton::clicked, this, &CropWidget::onResetCrop);
  connect(autoDetectBtn, &QToolButton::clicked, this,
          &CropWidget::onAutoDetect);

  connect(brightnessSlider, &QSlider::valueChanged, this, [this](int value) {
    if (!sourceImage.empty()) {
      cv::Mat adjusted;
      sourceImage.convertTo(adjusted, -1, 1, value);
      previewWidget->setImage(adjusted);
    }
  });

  connect(contrastSlider, &QSlider::valueChanged, this, [this](int value) {
    if (!sourceImage.empty()) {
      double factor = (100.0 + value) / 100.0;
      cv::Mat adjusted;
      sourceImage.convertTo(adjusted, -1, factor, 0);
      previewWidget->setImage(adjusted);
    }
  });

  // 添加撤销/重做状态更新
  connect(previewWidget, &CropPreviewWidget::strategyChanged, this, [this]() {
    if (!undoStack.empty()) {
      auto current = getCurrentStrategy();
      auto &last = undoStack.back();
      bool isDifferent = std::visit(
          [](const auto &a, const auto &b) -> bool {
            if constexpr (std::is_same_v<std::decay_t<decltype(a)>,
                                         std::decay_t<decltype(b)>>) {
              using T = std::decay_t<decltype(a)>;
              if constexpr (std::is_same_v<T, RatioCrop>) {
                return a.ratio != b.ratio;
              } else if constexpr (std::is_same_v<T, EllipseCrop>) {
                return a.center != b.center || a.axes != b.axes ||
                       a.angle != b.angle;
              } else if constexpr (std::is_same_v<T, CircleCrop>) {
                return a.center != b.center || a.radius != b.radius;
              } else {
                return !(a == b);
              }
            }
            return true;
          },
          last, current);

      if (isDifferent) {
        undoStack.push_back(std::move(current));
        redoStack.clear();
        updateUndoRedoState();
      }
    }
  });

  // 添加异常处理
  connect(previewWidget, &CropPreviewWidget::errorOccurred, this,
          [this](const QString &error) { showError(error); });
}

void CropWidget::setImage(const cv::Mat &image) {
  sourceImage = image.clone();
  previewWidget->setImage(sourceImage);
  updatePreview();
}

void CropWidget::onStrategyChanged(int index) {
  CropStrategy strategy;
  switch (index) {
  case 0: // 矩形
    strategy = cv::Rect(0, 0, sourceImage.cols, sourceImage.rows);
    break;
  case 1: // 多边形
    strategy =
        std::vector<cv::Point>{{50, 50},
                               {sourceImage.cols - 50, 50},
                               {sourceImage.cols - 50, sourceImage.rows - 50},
                               {50, sourceImage.rows - 50}};
    break;
  case 2: // 圆形
    strategy = CircleCrop{cv::Point(sourceImage.cols / 2, sourceImage.rows / 2),
                          std::min(sourceImage.cols, sourceImage.rows) / 2};
    break;
  case 3: // 椭圆
    strategy =
        EllipseCrop{cv::Point(sourceImage.cols / 2, sourceImage.rows / 2),
                    cv::Size(sourceImage.cols / 2, sourceImage.rows / 2), 0.0};
    break;
  case 4: // 比例
    strategy = RatioCrop{ratioSpin->value()};
    break;
  }
  previewWidget->setStrategy(strategy);
  updatePreview();
}

void CropWidget::onCropClicked() {
  if (!confirmOperation())
    return;

  if (sourceImage.empty()) {
    showError(tr("没有需要裁切的图像"));
    return;
  }

  setState(CropState::Processing);
  try {
    auto result =
        cropper->crop(sourceImage, getCurrentStrategy(), getAdaptiveParams());
    if (result) {
      resultImage = *result;
      updateStatus(tr("裁切成功"));
      emit cropFinished(resultImage);
      setState(CropState::Ready);
    } else {
      showError(tr("裁切失败"));
    }
  } catch (const std::exception &e) {
    handleException(e);
  }
}

void CropWidget::onCancelClicked() { emit cropCanceled(); }

void CropWidget::updatePreview() {
  if (!sourceImage.empty()) {
    auto result =
        cropper->crop(sourceImage, getCurrentStrategy(), getAdaptiveParams());
    if (result) {
      previewWidget->setImage(*result);
    }
  }
}

CropStrategy CropWidget::getCurrentStrategy() const {
  return previewWidget->getCurrentStrategy();
}

AdaptiveParams CropWidget::getAdaptiveParams() const {
  AdaptiveParams params;
  params.margin = marginSpin->value();
  return params;
}

cv::Mat CropWidget::getResult() const { return resultImage; }

void CropWidget::rotateImage(int angle) {
  if (sourceImage.empty())
    return;

  currentRotation += angle;
  cv::Point2f center(sourceImage.cols / 2.0f, sourceImage.rows / 2.0f);
  cv::Mat rotation = cv::getRotationMatrix2D(center, angle, 1.0);
  cv::Mat rotated;
  cv::warpAffine(sourceImage, rotated, rotation, sourceImage.size());
  sourceImage = rotated;
  previewWidget->setImage(sourceImage);
  updatePreview();
}

void CropWidget::onRotateLeft() { rotateImage(-90); }

void CropWidget::onRotateRight() { rotateImage(90); }

void CropWidget::onZoomIn() { previewWidget->zoomIn(); }

void CropWidget::onZoomOut() { previewWidget->zoomOut(); }

void CropWidget::onFitToView() { previewWidget->fitToView(); }

void CropWidget::onResetCrop() {
  if (sourceImage.empty())
    return;
  currentRotation = 0;
  previewWidget->setImage(sourceImage);
  onStrategyChanged(strategyCombo->currentIndex());
}

void CropWidget::onAutoDetect() {
  if (sourceImage.empty())
    return;

  try {
    auto result = cropper->cropAuto(sourceImage);
    if (result) {
      previewWidget->setImage(*result);
      // 更新裁切策略为检测到的区域
      cv::Rect detected = cv::boundingRect(*result);
      CropStrategy strategy = detected;
      previewWidget->setStrategy(strategy);
      updatePreview();
    }
  } catch (const std::exception &e) {
    QMessageBox::warning(this, "自动检测失败", e.what());
  }
}

void CropWidget::savePreset(const QString &name) {
  if (name.isEmpty())
    return;
  presets[name] = getCurrentStrategy();
  if (presetCombo->findText(name) == -1) {
    presetCombo->addItem(name);
  }
}

void CropWidget::loadPreset(const QString &name) {
  auto it = presets.find(name);
  if (it != presets.end()) {
    previewWidget->setStrategy(it->second);
    updatePreview();
  }
}

void CropWidget::showError(const QString &message) {
  lastError = message;
  setState(CropState::Error);
}

QWidget *CropWidget::createStrategyGroup() {
  auto group = new QGroupBox(tr("裁切方式"));
  auto layout = new QVBoxLayout(group);

  strategyCombo = new QComboBox;
  strategyCombo->addItem(tr("矩形"));
  strategyCombo->addItem(tr("多边形"));
  strategyCombo->addItem(tr("圆形"));
  strategyCombo->addItem(tr("椭圆"));
  strategyCombo->addItem(tr("比例"));

  auto ratioLayout = new QHBoxLayout;
  ratioSpin = new QDoubleSpinBox;
  ratioSpin->setRange(0.1, 10.0);
  ratioSpin->setValue(1.0);
  ratioSpin->setSingleStep(0.1);
  ratioLayout->addWidget(new QLabel(tr("宽高比:")));
  ratioLayout->addWidget(ratioSpin);

  auto marginLayout = new QHBoxLayout;
  marginSpin = new QSpinBox;
  marginSpin->setRange(0, 100);
  marginSpin->setValue(0);
  marginLayout->addWidget(new QLabel(tr("边距:")));
  marginLayout->addWidget(marginSpin);

  layout->addWidget(strategyCombo);
  layout->addLayout(ratioLayout);
  layout->addLayout(marginLayout);

  return group;
}

QWidget *CropWidget::createAdjustmentGroup() {
  auto group = new QGroupBox(tr("图像调整"));
  auto layout = new QVBoxLayout(group);

  brightnessSlider = new QSlider(Qt::Horizontal);
  brightnessSlider->setRange(-100, 100);
  brightnessSlider->setValue(0);

  contrastSlider = new QSlider(Qt::Horizontal);
  contrastSlider->setRange(-100, 100);
  contrastSlider->setValue(0);

  layout->addWidget(new QLabel(tr("亮度:")));
  layout->addWidget(brightnessSlider);
  layout->addWidget(new QLabel(tr("对比度:")));
  layout->addWidget(contrastSlider);

  return group;
}

QWidget *CropWidget::createPresetsGroup() {
  auto group = new QGroupBox(tr("预设"));
  auto layout = new QVBoxLayout(group);

  presetCombo = new QComboBox;
  auto btnLayout = new QHBoxLayout;

  auto loadBtn = new QPushButton(tr("加载"));
  auto saveBtn = new QPushButton(tr("保存"));

  btnLayout->addWidget(loadBtn);
  btnLayout->addWidget(saveBtn);

  layout->addWidget(presetCombo);
  layout->addLayout(btnLayout);

  connect(loadBtn, &QPushButton::clicked, this, &CropWidget::onLoadPreset);
  connect(saveBtn, &QPushButton::clicked, this, &CropWidget::onSavePreset);

  return group;
}

void CropWidget::onSavePreset() {
  bool ok;
  QString name =
      QInputDialog::getText(this, tr("保存预设"), tr("请输入预设名称:"),
                            QLineEdit::Normal, QString(), &ok);
  if (ok && !name.isEmpty()) {
    savePreset(name);
  }
}

void CropWidget::onLoadPreset() {
  QString name = presetCombo->currentText();
  if (!name.isEmpty()) {
    loadPreset(name);
  }
}

void CropWidget::updateUndoRedoState() {
  actions.undo->setEnabled(!undoStack.empty());
  actions.redo->setEnabled(!redoStack.empty());
}

CropWidget::~CropWidget() = default;
