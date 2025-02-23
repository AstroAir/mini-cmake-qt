#include "CropWidget.h"
#include "HistogramDialog.hpp"
#include "CropPreviewWidget.h"

#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QMessageBox>
#include <QScrollArea>
#include <QShortcut>
#include <QSplitter>
#include <QStackedWidget>
#include <QTimer>
#include <QVBoxLayout>

#include "ElaCheckBox.h"
#include "ElaComboBox.h"
#include "ElaDoubleSpinBox.h"
#include "ElaPushButton.h"
#include "ElaSlider.h"
#include "ElaSpinBox.h"
#include "ElaStatusBar.h"
#include "ElaToolButton.h"

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

  // 创建新的主布局
  auto mainSplitter = new QSplitter(Qt::Horizontal);

  // 左侧工具面板使用滚动区域
  auto toolScroll = new QScrollArea;
  toolScroll->setWidget(createToolPanel());
  toolScroll->setWidgetResizable(true);
  toolScroll->setFixedWidth(280);

  // 中间预览区域
  auto previewContainer = new QWidget;
  auto previewLayout = new QVBoxLayout(previewContainer);
  previewWidget = new CropPreviewWidget(this);
  previewLayout->addWidget(previewWidget);

  // 添加显示直方图的按钮
  auto histogramBtn = new ElaPushButton(tr("显示直方图"), this);
  previewLayout->addWidget(histogramBtn);
  connect(histogramBtn, &ElaPushButton::clicked, this,
          &CropWidget::showHistogram);

  // 右侧高级面板
  advancedPanel = createAdvancedPanel();
  advancedPanel->setVisible(false);

  mainSplitter->addWidget(toolScroll);
  mainSplitter->addWidget(previewContainer);
  mainSplitter->addWidget(advancedPanel);

  mainLayout->addWidget(mainSplitter);

  // 添加网格显示
  gridCheckBox = new ElaCheckBox(tr("显示网格"));
  aspectLockCheckBox = new ElaCheckBox(tr("锁定比例"));

  auto viewOptionsLayout = new QHBoxLayout;
  viewOptionsLayout->addWidget(gridCheckBox);
  viewOptionsLayout->addWidget(aspectLockCheckBox);
  viewOptionsLayout->addStretch();

  mainLayout->addLayout(viewOptionsLayout);

  // 底部状态栏和按钮区域
  statusBar = new ElaStatusBar;
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
  rotateLeftBtn = new ElaToolButton(this);
  rotateRightBtn = new ElaToolButton(this);
  zoomInBtn = new ElaToolButton(this);
  zoomOutBtn = new ElaToolButton(this);
  fitViewBtn = new ElaToolButton(this);
  resetBtn = new ElaToolButton(this);
  autoDetectBtn = new ElaToolButton(this);

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

  connect(rotateLeftBtn, &ElaToolButton::clicked, this,
          &CropWidget::onRotateLeft);
  connect(rotateRightBtn, &ElaToolButton::clicked, this,
          &CropWidget::onRotateRight);
  connect(zoomInBtn, &ElaToolButton::clicked, this, &CropWidget::onZoomIn);
  connect(zoomOutBtn, &ElaToolButton::clicked, this, &CropWidget::onZoomOut);
  connect(fitViewBtn, &ElaToolButton::clicked, this, &CropWidget::onFitToView);
  connect(resetBtn, &ElaToolButton::clicked, this, &CropWidget::onResetCrop);
  connect(autoDetectBtn, &ElaToolButton::clicked, this,
          &CropWidget::onAutoDetect);

  connect(brightnessSlider, &ElaSlider::valueChanged, this, [this](int value) {
    if (!sourceImage.empty()) {
      cv::Mat adjusted;
      sourceImage.convertTo(adjusted, -1, 1, value);
      previewWidget->setImage(adjusted);
    }
  });

  connect(contrastSlider, &ElaSlider::valueChanged, this, [this](int value) {
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
          // TODO: 添加自定义比例
    // strategy.emplace<RatioCrop>(ratioSpin->value());
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

  strategyCombo = new ElaComboBox;
  strategyCombo->addItem(tr("矩形"));
  strategyCombo->addItem(tr("多边形"));
  strategyCombo->addItem(tr("圆形"));
  strategyCombo->addItem(tr("椭圆"));
  strategyCombo->addItem(tr("比例"));

  auto ratioLayout = new QHBoxLayout;
  ratioSpin = new ElaDoubleSpinBox;
  ratioSpin->setRange(0.1, 10.0);
  ratioSpin->setValue(1.0);
  ratioSpin->setSingleStep(0.1);
  ratioLayout->addWidget(new QLabel(tr("宽高比:")));
  ratioLayout->addWidget(ratioSpin);

  auto marginLayout = new QHBoxLayout;
  marginSpin = new ElaSpinBox;
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

  brightnessSlider = new ElaSlider(Qt::Horizontal);
  brightnessSlider->setRange(-100, 100);
  brightnessSlider->setValue(0);

  contrastSlider = new ElaSlider(Qt::Horizontal);
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

  presetCombo = new ElaComboBox;
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

QWidget *CropWidget::createToolPanel() {
  auto toolPanel = new QWidget;
  auto toolPanelLayout = new QVBoxLayout(toolPanel);
  toolPanelLayout->addWidget(createStrategyGroup());
  toolPanelLayout->addWidget(createAdjustmentGroup());
  toolPanelLayout->addWidget(createPresetsGroup());
  toolPanelLayout->addStretch();
  return toolPanel;
}

QWidget *CropWidget::createAdvancedPanel() {
  auto panel = new QWidget;
  auto layout = new QVBoxLayout(panel);

  // 添加自定义比例控制
  auto ratioGroup = new QGroupBox(tr("自定义比例"));
  auto ratioLayout = new QHBoxLayout;

  customRatioWidth = new ElaDoubleSpinBox;
  customRatioHeight = new ElaDoubleSpinBox;

  customRatioWidth->setRange(0.1, 100);
  customRatioHeight->setRange(0.1, 100);

  ratioLayout->addWidget(new QLabel("宽:"));
  ratioLayout->addWidget(customRatioWidth);
  ratioLayout->addWidget(new QLabel("高:"));
  ratioLayout->addWidget(customRatioHeight);

  ratioGroup->setLayout(ratioLayout);
  layout->addWidget(ratioGroup);

  // 添加更多高级控件...
  layout->addStretch();

  return panel;
}

void CropWidget::setupTheme() {
  createThemeMenu();

  // 设置默认主题
  setTheme("system");
}

void CropWidget::setTheme(const QString &themeName) {
  currentTheme = themeName;

  // 应用主题样式
  if (themeName == "dark") {
    setStyleSheet(R"(
      QWidget { background-color: #2d2d2d; color: #ffffff; }
      QPushButton { background-color: #404040; border-radius: 4px; padding: 6px; }
      QPushButton:hover { background-color: #505050; }
      QComboBox { background-color: #404040; border-radius: 4px; padding: 4px; }
    )");
  } else if (themeName == "light") {
    setStyleSheet(R"(
      QWidget { background-color: #f0f0f0; color: #000000; }
      QPushButton { background-color: #e0e0e0; border-radius: 4px; padding: 6px; }
      QPushButton:hover { background-color: #d0d0d0; }
      QComboBox { background-color: #ffffff; border-radius: 4px; padding: 4px; }
    )");
  }
  // ...其他主题样式
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

void CropWidget::showHistogram() {
  if (sourceImage.empty()) {
    QMessageBox::warning(this, tr("警告"), tr("没有可用的图像"));
    return;
  }

  // 延迟创建直方图对话框
  if (!histogramDialog) {
    histogramDialog = std::make_unique<HistogramDialog>(this);
  }

  // 显示当前图像的直方图
  histogramDialog->showHistogram(sourceImage);
}

void CropWidget::createThemeMenu() {
    // 创建主题相关的 Actions
    actions.darkTheme = new QAction(tr("深色主题"), this);
    actions.lightTheme = new QAction(tr("浅色主题"), this);
    actions.systemTheme = new QAction(tr("系统主题"), this);
    actions.customTheme = new QAction(tr("自定义主题"), this);

    // 设置这些 actions 为可选中的
    actions.darkTheme->setCheckable(true);
    actions.lightTheme->setCheckable(true);
    actions.systemTheme->setCheckable(true);
    actions.customTheme->setCheckable(true);

    // 将这些 actions 加入到一个 action group 中
    auto themeGroup = new QActionGroup(this);
    themeGroup->addAction(actions.darkTheme);
    themeGroup->addAction(actions.lightTheme);
    themeGroup->addAction(actions.systemTheme);
    themeGroup->addAction(actions.customTheme);

    // 连接信号槽
    connect(actions.darkTheme, &QAction::triggered, this, [this]() { setTheme("dark"); });
    connect(actions.lightTheme, &QAction::triggered, this, [this]() { setTheme("light"); });
    connect(actions.systemTheme, &QAction::triggered, this, [this]() { setTheme("system"); });
    connect(actions.customTheme, &QAction::triggered, this, [this]() {
        // 实现自定义主题对话框
        // TODO: 添加自定义主题功能
    });
}

void CropWidget::enableAdvancedMode(bool enable) {
    isAdvancedMode = enable;
    if (advancedPanel) {
        advancedPanel->setVisible(enable);
    }
    
    // 更新UI元素的可见性
    if (customRatioWidth) customRatioWidth->setEnabled(enable);
    if (customRatioHeight) customRatioHeight->setEnabled(enable);
    
    emit onAdvancedModeToggled(enable);
}

void CropWidget::setPresets(const QMap<QString, CropStrategy>& newPresets) {
    presets.clear();
    presetCombo->clear();
    
    for (auto it = newPresets.begin(); it != newPresets.end(); ++it) {
        presets[it.key()] = it.value();
        presetCombo->addItem(it.key());
    }
}

void CropWidget::onAdvancedModeToggled(bool enabled) {
    // 更新高级模式相关的UI元素
    if (advancedPanel) {
        advancedPanel->setVisible(enabled);
    }
    
    // 更新自定义比例控件的状态
    if (customRatioWidth && customRatioHeight) {
        customRatioWidth->setEnabled(enabled);
        customRatioHeight->setEnabled(enabled);
    }
}

void CropWidget::onHistogramUpdate() {
    if (histogramDialog && histogramDialog->isVisible()) {
        histogramDialog->showHistogram(sourceImage);
    }
}

void CropWidget::onGridToggled(bool show) {
    isGridVisible = show;
    if (previewWidget) {
        previewWidget->setGridVisible(show);
    }
}

void CropWidget::onAspectRatioLocked(bool locked) {
    isAspectRatioLocked = locked;
    if (previewWidget) {
        previewWidget->setAspectRatioLocked(locked);
    }
}

void CropWidget::onCustomRatioChanged() {
    if (!customRatioWidth || !customRatioHeight) return;
    
    double width = customRatioWidth->value();
    double height = customRatioHeight->value();
    double ratio = width / height;
    
    if (isAspectRatioLocked && previewWidget) {
        previewWidget->setAspectRatio(ratio);
    }
}

void CropWidget::onThemeChanged() {
    // 根据当前主题更新UI元素的样式
    setupTheme();
    
    // 更新所有子控件的主题
    if (previewWidget) previewWidget->update();
    if (histogramDialog) histogramDialog->update();
    
    // 触发重绘
    update();
}

void CropWidget::createMenus() {
    // 创建主菜单
    auto menuBar = new QMenuBar(this);
    
    // 文件菜单
    auto fileMenu = menuBar->addMenu(tr("文件"));
    fileMenu->addAction(actions.reset);
    
    // 编辑菜单
    auto editMenu = menuBar->addMenu(tr("编辑"));
    editMenu->addAction(actions.undo);
    editMenu->addAction(actions.redo);
    
    // 视图菜单
    auto viewMenu = menuBar->addMenu(tr("视图"));
    viewMenu->addAction(actions.toggleAdvanced);
    viewMenu->addAction(actions.showGrid);
    
    // 主题菜单
    auto themeMenu = menuBar->addMenu(tr("主题"));
    themeMenu->addAction(actions.darkTheme);
    themeMenu->addAction(actions.lightTheme);
    themeMenu->addAction(actions.systemTheme);
    themeMenu->addAction(actions.customTheme);
    
    // 帮助菜单
    auto helpMenu = menuBar->addMenu(tr("帮助"));
    helpMenu->addAction(actions.help);
}

CropWidget::~CropWidget() = default;
