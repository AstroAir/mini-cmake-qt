#include "CropWidget.h"
#include "CropPreviewWidget.h"
#include "HistogramDialog.hpp"

#include <QAction>
#include <QApplication>
#include <QClipboard>
#include <QComboBox>
#include <QDragEnterEvent>
#include <QFileDialog>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QMenu>
#include <QMessageBox>
#include <QMimeData>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QShortcut>
#include <QSplitter>
#include <QStackedWidget>
#include <QTimer>
#include <QToolBar>
#include <QToolTip>
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
    : QWidget(parent), cropper(std::make_unique<ImageCropper>()),
      isProcessing(false), isDownscaled(false), isAutosaveEnabled(true),
      optimizedSize(1920, 1080) {

  // 添加对拖放的支持
  setAcceptDrops(true);

  // 设置对象名称便于样式表
  setObjectName("CropWidget");

  setupUi();
  connectSignals();
  setupShortcuts();

  // 创建自动保存定时器
  autosaveTimer = new QTimer(this);
  autosaveTimer->setInterval(60000); // 1分钟
  connect(autosaveTimer, &QTimer::timeout, this, &CropWidget::saveSessionState);
  autosaveTimer->start();

  // 加载上次会话状态
  loadSessionState();

  // 创建上下文菜单
  createContextMenu();

  // 初始化状态
  setState(CropState::Ready);
}

void CropWidget::setupUi() {
  // 基本UI设置
  setupLayout();
  createActions();
  setupTheme();
  createNotifications();

  // 创建工具栏
  mainToolBar = new QToolBar(this);
  mainToolBar->setObjectName("mainToolBar");
  mainToolBar->setMovable(false);
  mainToolBar->setIconSize(QSize(24, 24));
  mainToolBar->addAction(actions.undo);
  mainToolBar->addAction(actions.redo);
  mainToolBar->addSeparator();
  mainToolBar->addAction(actions.reset);
  mainToolBar->addSeparator();
  mainToolBar->addAction(actions.copyToClipboard);
  mainToolBar->addAction(actions.pasteFromClipboard);
  mainToolBar->addSeparator();
  mainToolBar->addAction(actions.showGrid);
  mainToolBar->addAction(actions.lockAspectRatio);
  mainToolBar->addAction(actions.toggleAdvanced);
  mainToolBar->addSeparator();
  mainToolBar->addAction(actions.help);

  // 设置状态栏
  statusBar = new ElaStatusBar;
  statusBar->setSizeGripEnabled(false);

  // 创建进度条
  progressBar = new QProgressBar;
  progressBar->setTextVisible(true);
  progressBar->setAlignment(Qt::AlignCenter);
  progressBar->setMaximumHeight(15);
  progressBar->hide();

  // 添加状态栏小部件
  statusBar->addPermanentWidget(progressBar);
}

void CropWidget::setupLayout() {
  // 创建主布局
  auto mainLayout = new QVBoxLayout(this);
  mainLayout->setSpacing(5);
  mainLayout->setContentsMargins(5, 5, 5, 5);

  // 添加工具栏
  mainLayout->addWidget(mainToolBar);

  // 创建主分隔器
  mainSplitter = new QSplitter(Qt::Horizontal);
  mainSplitter->setObjectName("mainSplitter");
  mainSplitter->setHandleWidth(4);
  mainSplitter->setChildrenCollapsible(false);

  // 创建左侧工具区
  auto toolScroll = new QScrollArea;
  toolScroll->setObjectName("toolPanel");
  toolScroll->setWidget(createToolPanel());
  toolScroll->setWidgetResizable(true);
  toolScroll->setMinimumWidth(250);
  toolScroll->setMaximumWidth(300);

  // 创建中间预览区
  auto previewContainer = new QWidget;
  previewContainer->setObjectName("previewContainer");
  auto previewLayout = new QVBoxLayout(previewContainer);
  previewLayout->setContentsMargins(0, 0, 0, 0);

  // 创建预览工具栏
  auto previewTools = new QHBoxLayout;

  // 添加工具按钮
  rotateLeftBtn = new ElaToolButton(this);
  rotateRightBtn = new ElaToolButton(this);
  zoomInBtn = new ElaToolButton(this);
  zoomOutBtn = new ElaToolButton(this);
  fitViewBtn = new ElaToolButton(this);
  resetBtn = new ElaToolButton(this);
  autoDetectBtn = new ElaToolButton(this);
  centerBtn = new ElaToolButton(this);
  clipboardBtn = new ElaToolButton(this);

  rotateLeftBtn->setIcon(QIcon::fromTheme("object-rotate-left"));
  rotateRightBtn->setIcon(QIcon::fromTheme("object-rotate-right"));
  zoomInBtn->setIcon(QIcon::fromTheme("zoom-in"));
  zoomOutBtn->setIcon(QIcon::fromTheme("zoom-out"));
  fitViewBtn->setIcon(QIcon::fromTheme("zoom-fit-best"));
  resetBtn->setIcon(QIcon::fromTheme("edit-undo"));
  autoDetectBtn->setIcon(QIcon::fromTheme("edit-find"));
  centerBtn->setIcon(QIcon::fromTheme("insert-image"));
  clipboardBtn->setIcon(QIcon::fromTheme("edit-copy"));

  // 添加工具提示
  rotateLeftBtn->setToolTip(tr("向左旋转 (Ctrl+L)"));
  rotateRightBtn->setToolTip(tr("向右旋转 (Ctrl+R)"));
  zoomInBtn->setToolTip(tr("放大 (+)"));
  zoomOutBtn->setToolTip(tr("缩小 (-)"));
  fitViewBtn->setToolTip(tr("适应视图 (F)"));
  resetBtn->setToolTip(tr("重置裁剪 (Esc)"));
  autoDetectBtn->setToolTip(tr("自动检测 (A)"));
  centerBtn->setToolTip(tr("中心裁剪 (C)"));
  clipboardBtn->setToolTip(tr("复制到剪贴板 (Ctrl+C)"));

  previewTools->addWidget(rotateLeftBtn);
  previewTools->addWidget(rotateRightBtn);
  previewTools->addWidget(zoomInBtn);
  previewTools->addWidget(zoomOutBtn);
  previewTools->addWidget(fitViewBtn);
  previewTools->addWidget(centerBtn);
  previewTools->addWidget(autoDetectBtn);
  previewTools->addWidget(resetBtn);
  previewTools->addWidget(clipboardBtn);
  previewTools->addStretch();

  // 添加显示直方图的按钮
  auto histogramBtn = new ElaPushButton(tr("显示直方图"), this);
  histogramBtn->setToolTip(tr("显示图像直方图 (H)"));
  previewTools->addWidget(histogramBtn);

  previewLayout->addLayout(previewTools);

  // 创建预览控件
  previewWidget = new CropPreviewWidget(this);
  previewWidget->setObjectName("previewWidget");
  previewWidget->setContextMenuPolicy(Qt::CustomContextMenu);
  previewLayout->addWidget(previewWidget, 1); // 添加伸展因子

  // 添加宽高比和网格显示控件
  auto viewOptionsLayout = new QHBoxLayout;
  gridCheckBox = new ElaCheckBox(tr("显示网格"));
  aspectLockCheckBox = new ElaCheckBox(tr("锁定比例"));

  // 添加宽高比输入框
  auto aspectRatioWidget = createAspectRatioPanel();

  viewOptionsLayout->addWidget(gridCheckBox);
  viewOptionsLayout->addWidget(aspectLockCheckBox);
  viewOptionsLayout->addWidget(aspectRatioWidget);
  viewOptionsLayout->addStretch();

  previewLayout->addLayout(viewOptionsLayout);

  // 创建高级面板（默认隐藏）
  advancedPanel = createAdvancedPanel();
  advancedPanel->setVisible(false);

  // 添加面板到分隔器
  mainSplitter->addWidget(toolScroll);
  mainSplitter->addWidget(previewContainer);
  mainSplitter->addWidget(advancedPanel);

  // 设置默认分隔器比例
  mainSplitter->setStretchFactor(0, 0); // 工具面板
  mainSplitter->setStretchFactor(1, 1); // 预览区域
  mainSplitter->setStretchFactor(2, 0); // 高级面板

  mainLayout->addWidget(mainSplitter, 1);

  // 添加状态栏
  mainLayout->addWidget(statusBar);

  // 添加底部操作按钮
  auto buttonBox = new QHBoxLayout;
  cropButton = new ElaPushButton(tr("裁剪"), this);
  cancelButton = new ElaPushButton(tr("取消"), this);

  cropButton->setObjectName("cropButton");
  cancelButton->setObjectName("cancelButton");

  buttonBox->addStretch();
  buttonBox->addWidget(cropButton);
  buttonBox->addWidget(cancelButton);
  mainLayout->addLayout(buttonBox);

  // 设置状态更新定时器
  statusTimer = new QTimer(this);
  statusTimer->setSingleShot(true);
  connect(statusTimer, &QTimer::timeout, this,
          [this]() { statusBar->clearMessage(); });

  // 连接显示直方图的按钮
  connect(histogramBtn, &ElaPushButton::clicked, this,
          &CropWidget::showHistogram);
}

QWidget *CropWidget::createAspectRatioPanel() {
  auto widget = new QWidget;
  auto layout = new QHBoxLayout(widget);
  layout->setContentsMargins(0, 0, 0, 0);

  customRatioWidth = new ElaDoubleSpinBox;
  customRatioHeight = new ElaDoubleSpinBox;

  customRatioWidth->setRange(0.1, 100);
  customRatioHeight->setRange(0.1, 100);
  customRatioWidth->setValue(16);
  customRatioHeight->setValue(9);
  customRatioWidth->setSingleStep(0.1);
  customRatioHeight->setSingleStep(0.1);

  customRatioWidth->setMaximumWidth(50);
  customRatioHeight->setMaximumWidth(50);

  layout->addWidget(new QLabel(tr("比例:")));
  layout->addWidget(customRatioWidth);
  layout->addWidget(new QLabel(":"));
  layout->addWidget(customRatioHeight);

  return widget;
}

void CropWidget::createActions() {
  // 创建基本操作
  actions.undo = new QAction(QIcon::fromTheme("edit-undo"), tr("撤销"), this);
  actions.undo->setShortcut(QKeySequence::Undo);
  actions.undo->setToolTip(tr("撤销上一步操作 (Ctrl+Z)"));
  actions.undo->setEnabled(false);

  actions.redo = new QAction(QIcon::fromTheme("edit-redo"), tr("重做"), this);
  actions.redo->setShortcut(QKeySequence::Redo);
  actions.redo->setToolTip(tr("重做操作 (Ctrl+Y)"));
  actions.redo->setEnabled(false);

  actions.reset = new QAction(QIcon::fromTheme("edit-clear"), tr("重置"), this);
  actions.reset->setShortcut(QKeySequence("Ctrl+R"));
  actions.reset->setToolTip(tr("重置所有修改 (Ctrl+R)"));

  actions.help = new QAction(QIcon::fromTheme("help-about"), tr("帮助"), this);
  actions.help->setShortcut(QKeySequence::HelpContents);
  actions.help->setToolTip(tr("显示帮助 (F1)"));

  // 新增操作
  actions.copyToClipboard =
      new QAction(QIcon::fromTheme("edit-copy"), tr("复制到剪贴板"), this);
  actions.copyToClipboard->setShortcut(QKeySequence::Copy);
  actions.copyToClipboard->setToolTip(tr("复制当前结果到剪贴板 (Ctrl+C)"));

  actions.pasteFromClipboard =
      new QAction(QIcon::fromTheme("edit-paste"), tr("从剪贴板粘贴"), this);
  actions.pasteFromClipboard->setShortcut(QKeySequence::Paste);
  actions.pasteFromClipboard->setToolTip(tr("从剪贴板粘贴图像 (Ctrl+V)"));

  actions.centerCrop =
      new QAction(QIcon::fromTheme("object-center"), tr("中心裁剪"), this);
  actions.centerCrop->setShortcut(QKeySequence("C"));
  actions.centerCrop->setToolTip(tr("使用中心点裁剪 (C)"));

  actions.fitToBounds =
      new QAction(QIcon::fromTheme("zoom-original"), tr("适应边界"), this);
  actions.fitToBounds->setShortcut(QKeySequence("B"));
  actions.fitToBounds->setToolTip(tr("裁剪区域适应图像边界 (B)"));

  // 主题相关
  actions.darkTheme = new QAction(tr("深色主题"), this);
  actions.lightTheme = new QAction(tr("浅色主题"), this);
  actions.systemTheme = new QAction(tr("系统主题"), this);
  actions.customTheme = new QAction(tr("自定义主题"), this);

  // 可选中的操作
  actions.darkTheme->setCheckable(true);
  actions.lightTheme->setCheckable(true);
  actions.systemTheme->setCheckable(true);
  actions.customTheme->setCheckable(true);

  // 视图选项
  actions.showGrid =
      new QAction(QIcon::fromTheme("show-grid"), tr("显示网格"), this);
  actions.showGrid->setCheckable(true);
  actions.showGrid->setToolTip(tr("显示/隐藏网格线 (G)"));
  actions.showGrid->setShortcut(QKeySequence("G"));

  actions.lockAspectRatio =
      new QAction(QIcon::fromTheme("object-locked"), tr("锁定比例"), this);
  actions.lockAspectRatio->setCheckable(true);
  actions.lockAspectRatio->setToolTip(tr("锁定/解锁宽高比 (L)"));
  actions.lockAspectRatio->setShortcut(QKeySequence("L"));

  actions.toggleAdvanced =
      new QAction(QIcon::fromTheme("preferences-system"), tr("高级模式"), this);
  actions.toggleAdvanced->setCheckable(true);
  actions.toggleAdvanced->setToolTip(tr("切换高级模式 (A)"));
  actions.toggleAdvanced->setShortcut(QKeySequence("A"));

  // 主题选项按钮组
  auto themeGroup = new QActionGroup(this);
  themeGroup->addAction(actions.darkTheme);
  themeGroup->addAction(actions.lightTheme);
  themeGroup->addAction(actions.systemTheme);
  themeGroup->addAction(actions.customTheme);

  // 连接撤销/重做操作
  connect(actions.undo, &QAction::triggered, this, [this]() {
    if (!undoStack.empty()) {
      redoStack.push_back(getCurrentStrategy());
      previewWidget->setStrategy(undoStack.back());
      undoStack.pop_back();
      updateUndoRedoState();
      updatePreview();
    }
  });

  connect(actions.redo, &QAction::triggered, this, [this]() {
    if (!redoStack.empty()) {
      undoStack.push_back(getCurrentStrategy());
      previewWidget->setStrategy(redoStack.back());
      redoStack.pop_back();
      updateUndoRedoState();
      updatePreview();
    }
  });

  // 连接重置操作
  connect(actions.reset, &QAction::triggered, this, [this]() {
    if (confirmOperation()) {
      undoStack.clear();
      redoStack.clear();
      updateUndoRedoState();
      onResetCrop();
    }
  });

  // 连接帮助操作
  connect(actions.help, &QAction::triggered, this, [this]() {
    QMessageBox::information(
        this, tr("帮助"),
        tr("图像裁剪工具快捷键:\n\n"
           "Ctrl+Z: 撤销\n"
           "Ctrl+Y: 重做\n"
           "Ctrl+R: 重置\n"
           "Ctrl+L/Ctrl+Right: 向左/右旋转\n"
           "+/-: 放大/缩小\n"
           "F: 适应视图\n"
           "C: 中心裁剪\n"
           "A: 自动检测\n"
           "G: 显示/隐藏网格\n"
           "L: 锁定/解锁比例\n"
           "H: 显示直方图\n"
           "Esc: 重置裁剪\n"
           "空格: 适应视图\n\n"
           "拖动鼠标可以调整裁剪区域，按住Ctrl键可以进行精细调整。"));
  });

  // 连接新增操作
  connect(actions.copyToClipboard, &QAction::triggered, this,
          &CropWidget::onSaveToClipboard);
  connect(actions.pasteFromClipboard, &QAction::triggered, this,
          &CropWidget::onPreviewClipboardImage);
  connect(actions.centerCrop, &QAction::triggered, this,
          &CropWidget::onCenterCrop);
  connect(actions.fitToBounds, &QAction::triggered, this,
          &CropWidget::onFitToBounds);

  // 连接主题操作
  connect(actions.darkTheme, &QAction::triggered, this,
          [this]() { setTheme("dark"); });
  connect(actions.lightTheme, &QAction::triggered, this,
          [this]() { setTheme("light"); });
  connect(actions.systemTheme, &QAction::triggered, this,
          [this]() { setTheme("system"); });
  connect(actions.customTheme, &QAction::triggered, this, [this]() {
    // 实现自定义主题对话框
    QString themePath =
        QFileDialog::getOpenFileName(this, tr("选择主题文件"), QDir::homePath(),
                                     tr("主题文件 (*.qss *.css)"));

    if (!themePath.isEmpty()) {
      QFile file(themePath);
      if (file.open(QFile::ReadOnly | QFile::Text)) {
        setStyleSheet(file.readAll());
        file.close();
        currentTheme = "custom";
      }
    }
  });

  // 连接显示操作
  connect(actions.showGrid, &QAction::triggered, this, [this](bool checked) {
    gridCheckBox->setChecked(checked);
    onGridToggled(checked);
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
  case CropState::Success:
    cropButton->setEnabled(true);
    statusBar->showMessage(tr("操作成功"), 3000);
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

  connect(actions.lockAspectRatio, &QAction::triggered, this,
          [this](bool checked) {
            aspectLockCheckBox->setChecked(checked);
            onAspectRatioLocked(checked);
          });

  connect(actions.toggleAdvanced, &QAction::triggered, this,
          [this](bool checked) { enableAdvancedMode(checked); });

  // 连接自定义比例控件
  connect(customRatioWidth,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          &CropWidget::onCustomRatioChanged);
  connect(customRatioHeight,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          &CropWidget::onCustomRatioChanged);

  // 连接复选框
  connect(gridCheckBox, &QCheckBox::toggled, this, &CropWidget::onGridToggled);
  connect(aspectLockCheckBox, &QCheckBox::toggled, this,
          &CropWidget::onAspectRatioLocked);

  // 连接分隔器移动信号
  connect(mainSplitter, &QSplitter::splitterMoved, this,
          &CropWidget::onSplitterMoved);

  // 连接预览控件的上下文菜单请求
  connect(previewWidget, &QWidget::customContextMenuRequested, this,
          &CropWidget::onShowContextMenu);

  // 连接新增的工具按钮
  connect(centerBtn, &QToolButton::clicked, this, &CropWidget::onCenterCrop);
  connect(clipboardBtn, &QToolButton::clicked, this,
          &CropWidget::onSaveToClipboard);

  // 添加缩放级别信号连接
  connect(previewWidget, &CropPreviewWidget::zoomChanged, this,
          &CropWidget::zoomLevelChanged);
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
    previewWidget->setStrategy(it.value());
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
  connect(actions.darkTheme, &QAction::triggered, this,
          [this]() { setTheme("dark"); });
  connect(actions.lightTheme, &QAction::triggered, this,
          [this]() { setTheme("light"); });
  connect(actions.systemTheme, &QAction::triggered, this,
          [this]() { setTheme("system"); });
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
  if (customRatioWidth)
    customRatioWidth->setEnabled(enable);
  if (customRatioHeight)
    customRatioHeight->setEnabled(enable);

  emit onAdvancedModeToggled(enable);
}

void CropWidget::setPresets(const QMap<QString, CropStrategy> &newPresets) {
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
  if (!customRatioWidth || !customRatioHeight)
    return;

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
  if (previewWidget)
    previewWidget->update();
  if (histogramDialog)
    histogramDialog->update();

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

void CropWidget::onPreviewClipboardImage() {
  QClipboard *clipboard = QApplication::clipboard();
  const QImage image = clipboard->image();

  if (image.isNull()) {
    QMessageBox::warning(this, tr("警告"), tr("剪贴板中没有可用的图像"));
    return;
  }

  try {
    // 将QImage转换为cv::Mat
    cv::Mat cvImage;
    QImage rgbImage = image.convertToFormat(QImage::Format_RGB888);
    cvImage =
        cv::Mat(rgbImage.height(), rgbImage.width(), CV_8UC3,
                const_cast<uchar *>(rgbImage.bits()), rgbImage.bytesPerLine());
    cv::cvtColor(cvImage, cvImage, cv::COLOR_RGB2BGR);

    // 检查图像大小，可能需要缩放以提高性能
    if (checkImageSize(cvImage)) {
      // 保存原始图像
      originalImage = cvImage.clone();
      isDownscaled = true;

      // 缩放到优化尺寸
      double ratio = std::min((double)optimizedSize.width() / cvImage.cols,
                              (double)optimizedSize.height() / cvImage.rows);

      if (ratio < 1.0) {
        cv::resize(cvImage, sourceImage, cv::Size(), ratio, ratio,
                   cv::INTER_AREA);
      } else {
        sourceImage = cvImage.clone();
        isDownscaled = false;
      }
    } else {
      sourceImage = cvImage.clone();
      originalImage = cvImage.clone();
      isDownscaled = false;
    }

    // 设置到预览控件
    setImage(sourceImage);

    // 更新状态提示
    updateStatus(tr("已从剪贴板加载图像"));

    // 发送图像变更信号
    emit imageChanged();

  } catch (const std::exception &e) {
    handleException(e);
  }
}

void CropWidget::onSaveToClipboard() {
  if (sourceImage.empty()) {
    QMessageBox::warning(this, tr("警告"), tr("没有可用的图像"));
    return;
  }

  try {
    // 获取当前裁剪结果
    auto result =
        cropper->crop(sourceImage, getCurrentStrategy(), getAdaptiveParams());
    if (!result) {
      showError(tr("无法获取裁剪结果"));
      return;
    }

    // 转换为QImage
    cv::Mat rgbImage;
    cv::cvtColor(*result, rgbImage, cv::COLOR_BGR2RGB);
    QImage qImage(rgbImage.data, rgbImage.cols, rgbImage.rows, rgbImage.step,
                  QImage::Format_RGB888);

    // 复制到剪贴板
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setImage(qImage.copy());

    // 更新状态
    updateStatus(tr("已复制到剪贴板"));

  } catch (const std::exception &e) {
    handleException(e);
  }
}

void CropWidget::onCenterCrop() {
  if (sourceImage.empty())
    return;

  int centerX = sourceImage.cols / 2;
  int centerY = sourceImage.rows / 2;

  // 根据当前裁剪策略和锁定比例设置中心裁剪
  std::visit(
      [&](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;

        if constexpr (std::is_same_v<T, cv::Rect>) {
          // 计算新的矩形，保持大小，移动到中心
          int newX = centerX - arg.width / 2;
          int newY = centerY - arg.height / 2;
          cv::Rect newRect(newX, newY, arg.width, arg.height);

          // 确保在图像范围内
          if (newX < 0)
            newRect.x = 0;
          if (newY < 0)
            newRect.y = 0;
          if (newX + arg.width > sourceImage.cols)
            newRect.x = sourceImage.cols - arg.width;
          if (newY + arg.height > sourceImage.rows)
            newRect.y = sourceImage.rows - arg.height;

          previewWidget->setStrategy(newRect);
        } else if constexpr (std::is_same_v<T, CircleCrop>) {
          // 移动圆心到中心位置
          CircleCrop newCircle = {cv::Point(centerX, centerY), arg.radius};
          previewWidget->setStrategy(newCircle);
        } else if constexpr (std::is_same_v<T, EllipseCrop>) {
          // 移动椭圆中心到中心位置
          EllipseCrop newEllipse = {cv::Point(centerX, centerY), arg.axes,
                                    arg.angle};
          previewWidget->setStrategy(newEllipse);
        } else if constexpr (std::is_same_v<T, std::vector<cv::Point>>) {
          // 计算多边形中心
          cv::Point oldCenter(0, 0);
          for (const auto &pt : arg) {
            oldCenter += pt;
          }
          oldCenter.x /= arg.size();
          oldCenter.y /= arg.size();

          // 计算移动向量
          cv::Point offset = cv::Point(centerX, centerY) - oldCenter;

          // 创建新的多边形
          std::vector<cv::Point> newPoints;
          for (const auto &pt : arg) {
            newPoints.push_back(pt + offset);
          }

          previewWidget->setStrategy(newPoints);
        }
      },
      getCurrentStrategy());

  updatePreview();
}

void CropWidget::onFitToBounds() {
  if (sourceImage.empty())
    return;

  // 计算图像边界内最大的安全区域
  int safeMargin = 10; // 安全边距
  cv::Rect imageBounds(safeMargin, safeMargin,
                       sourceImage.cols - 2 * safeMargin,
                       sourceImage.rows - 2 * safeMargin);

  // 根据当前锁定比例设置
  if (isAspectRatioLocked && customRatioWidth && customRatioHeight) {
    double ratio = customRatioWidth->value() / customRatioHeight->value();
    int newWidth, newHeight;

    // 计算适合比例的最大矩形
    if (imageBounds.width / ratio <= imageBounds.height) {
      // 宽度是限制因素
      newWidth = imageBounds.width;
      newHeight = static_cast<int>(newWidth / ratio);
    } else {
      // 高度是限制因素
      newHeight = imageBounds.height;
      newWidth = static_cast<int>(newHeight * ratio);
    }

    // 居中放置
    int x = imageBounds.x + (imageBounds.width - newWidth) / 2;
    int y = imageBounds.y + (imageBounds.height - newHeight) / 2;

    previewWidget->setStrategy(cv::Rect(x, y, newWidth, newHeight));
  } else {
    // 直接使用整个边界
    previewWidget->setStrategy(imageBounds);
  }

  updatePreview();
}

void CropWidget::onCropProgress(int percent) {
  if (progressBar) {
    progressBar->setValue(percent);
    if (percent < 100) {
      progressBar->show();
    } else {
      QTimer::singleShot(500, progressBar, &QProgressBar::hide);
    }
  }

  emit processingProgress(percent);
}

void CropWidget::onShowContextMenu(const QPoint &pos) {
  if (!contextMenu) {
    createContextMenu();
  }

  if (contextMenu) {
    contextMenu->popup(previewWidget->mapToGlobal(pos));
  }
}

void CropWidget::createContextMenu() {
  contextMenu = new QMenu(this);
  contextMenu->addAction(actions.copyToClipboard);
  contextMenu->addAction(actions.pasteFromClipboard);
  contextMenu->addSeparator();
  contextMenu->addAction(actions.reset);
  contextMenu->addAction(actions.centerCrop);
  contextMenu->addAction(actions.fitToBounds);
  contextMenu->addSeparator();
  contextMenu->addAction(actions.showGrid);
  contextMenu->addAction(actions.lockAspectRatio);
  contextMenu->addSeparator();
  contextMenu->addAction(actions.toggleAdvanced);
}

void CropWidget::onSplitterMoved(int pos, int index) {
  // 可以根据需要保存布局状态
  QSettings settings;
  settings.setValue("CropWidget/SplitterPosition", mainSplitter->saveState());
}

void CropWidget::onKeyPressed(QKeyEvent *event) {
  switch (event->key()) {
  case Qt::Key_Escape:
    onResetCrop();
    event->accept();
    break;
  case Qt::Key_Space:
    onFitToView();
    event->accept();
    break;
  case Qt::Key_C:
    if (!(event->modifiers() & Qt::ControlModifier)) {
      onCenterCrop();
      event->accept();
    }
    break;
  case Qt::Key_G:
    actions.showGrid->toggle();
    event->accept();
    break;
  case Qt::Key_L:
    if (!(event->modifiers() & Qt::ControlModifier)) {
      actions.lockAspectRatio->toggle();
      event->accept();
    }
    break;
  case Qt::Key_H:
    showHistogram();
    event->accept();
    break;
  case Qt::Key_F:
    onFitToView();
    event->accept();
    break;
  case Qt::Key_A:
    if (!(event->modifiers() & Qt::ControlModifier)) {
      onAutoDetect();
      event->accept();
    }
    break;
  case Qt::Key_B:
    onFitToBounds();
    event->accept();
    break;
  default:
    event->ignore();
  }
}

QWidget *CropWidget::createFilterGroup() {
  auto group = new QGroupBox(tr("图像滤镜"));
  auto layout = new QVBoxLayout(group);

  filterCombo = new ElaComboBox;
  filterCombo->addItem(tr("无"));
  filterCombo->addItem(tr("灰度"));
  filterCombo->addItem(tr("高斯模糊"));
  filterCombo->addItem(tr("锐化"));
  filterCombo->addItem(tr("边缘检测"));
  filterCombo->addItem(tr("反色"));
  filterCombo->addItem(tr("复古"));

  auto applyBtn = new ElaPushButton(tr("应用"));

  layout->addWidget(new QLabel(tr("选择滤镜:")));
  layout->addWidget(filterCombo);
  layout->addWidget(applyBtn);

  connect(applyBtn, &QPushButton::clicked, this,
          [this]() { onApplyFilter(filterCombo->currentIndex()); });

  return group;
}

void CropWidget::onApplyFilter(int filterType) {
  if (sourceImage.empty())
    return;

  try {
    // 开始处理
    startProgress("应用滤镜");

    // 保存到撤销堆栈
    if (sourceImage.data != originalImage.data) {
      undoStack.push_back(getCurrentStrategy());
      if (undoStack.size() > undoLimit) {
        undoStack.erase(undoStack.begin());
      }
      redoStack.clear();
      updateUndoRedoState();
    }

    // 获取基准图像
    cv::Mat baseImage = originalImage.empty() ? sourceImage : originalImage;
    cv::Mat result = applyFilter(baseImage, filterType);

    // 更新图像
    sourceImage = result;
    setImage(sourceImage);

    // 完成进度
    finishProgress();
    updateStatus(tr("滤镜应用成功"));

  } catch (const std::exception &e) {
    handleException(e);
  }
}

cv::Mat CropWidget::applyFilter(const cv::Mat &image, int filterType) {
  cv::Mat result;

  // 更新进度
  updateProgress(10);

  switch (filterType) {
  case 0: // 无
    result = image.clone();
    break;
  case 1: // 灰度
    cv::cvtColor(image, result, cv::COLOR_BGR2GRAY);
    cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    break;
  case 2: // 高斯模糊
    cv::GaussianBlur(image, result, cv::Size(5, 5), 0);
    break;
  case 3: // 锐化
  {
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    cv::filter2D(image, result, image.depth(), kernel);
  } break;
  case 4: // 边缘检测
  {
    cv::Mat gray, edges;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150);
    cv::cvtColor(edges, result, cv::COLOR_GRAY2BGR);
  } break;
  case 5: // 反色
    cv::bitwise_not(image, result);
    break;
  case 6: // 复古
  {
    result = image.clone();
    cv::Mat channels[3];
    cv::split(result, channels);

    // 调整各通道强度以获得复古效果
    channels[0] *= 0.8; // 减弱蓝色
    channels[1] *= 0.9; // 轻微减弱绿色
    // 保留红色强度

    cv::merge(channels, 3, result);

    // 添加轻微褐色
    cv::Mat overlay;
    cv::cvtColor(result, overlay, cv::COLOR_BGR2GRAY);
    cv::cvtColor(overlay, overlay, cv::COLOR_GRAY2BGR);
    cv::addWeighted(result, 0.9, overlay, 0.1, 0, result);
  } break;
  default:
    result = image.clone();
  }

  // 更新进度
  updateProgress(100);

  return result;
}

bool CropWidget::checkImageSize(const cv::Mat &image) {
  // 如果图像尺寸超过设定的优化尺寸，返回true表示需要优化
  return (image.cols > optimizedSize.width() ||
          image.rows > optimizedSize.height());
}

void CropWidget::startProgress(const QString &operationName) {
  isProcessing = true;
  currentOperation = operationName;
  currentProgress = 0;

  if (progressBar) {
    progressBar->setValue(0);
    progressBar->show();
  }

  setControlsEnabled(false);
  updateStatus(tr("%1...").arg(operationName));
}

void CropWidget::updateProgress(int percent) {
  currentProgress = percent;

  if (progressBar) {
    progressBar->setValue(percent);
  }

  emit processingProgress(percent);
}

void CropWidget::finishProgress() {
  isProcessing = false;
  currentProgress = 100;

  if (progressBar) {
    progressBar->setValue(100);
    QTimer::singleShot(500, progressBar, &QProgressBar::hide);
  }

  setControlsEnabled(true);
}

void CropWidget::setControlsEnabled(bool enabled) {
  cropButton->setEnabled(enabled);
  cancelButton->setEnabled(enabled);
  mainToolBar->setEnabled(enabled);
}

void CropWidget::createNotifications() {
  // 可以根据需要添加通知UI组件
}

void CropWidget::showNotification(const QString &message, int durationMs) {
  // 简单实现，使用状态栏
  updateStatus(message);

  // 高级实现可以添加弹出式通知
}

void CropWidget::saveSessionState() {
  if (!isAutosaveEnabled || sourceImage.empty())
    return;

  QSettings settings;

  // 保存当前裁剪策略
  // 注意：这里只是一个简单实现，完整实现需要序列化裁剪策略
  settings.setValue("CropWidget/LastUsedStrategy",
                    strategyCombo->currentIndex());

  // 保存界面设置
  settings.setValue("CropWidget/GridVisible", isGridVisible);
  settings.setValue("CropWidget/AspectRatioLocked", isAspectRatioLocked);
  settings.setValue("CropWidget/AdvancedMode", isAdvancedMode);

  // 保存分隔器位置
  settings.setValue("CropWidget/SplitterPosition", mainSplitter->saveState());

  // 保存比例设置
  if (customRatioWidth && customRatioHeight) {
    settings.setValue("CropWidget/RatioWidth", customRatioWidth->value());
    settings.setValue("CropWidget/RatioHeight", customRatioHeight->value());
  }
}

void CropWidget::loadSessionState() {
  QSettings settings;

  // 加载界面设置
  bool gridVisible = settings.value("CropWidget/GridVisible", false).toBool();
  bool aspectRatioLocked =
      settings.value("CropWidget/AspectRatioLocked", false).toBool();
  bool advancedMode = settings.value("CropWidget/AdvancedMode", false).toBool();

  // 应用设置
  if (gridCheckBox)
    gridCheckBox->setChecked(gridVisible);
  if (aspectLockCheckBox)
    aspectLockCheckBox->setChecked(aspectRatioLocked);
  if (actions.showGrid)
    actions.showGrid->setChecked(gridVisible);
  if (actions.lockAspectRatio)
    actions.lockAspectRatio->setChecked(aspectRatioLocked);

  onGridToggled(gridVisible);
  onAspectRatioLocked(aspectRatioLocked);
  enableAdvancedMode(advancedMode);

  // 加载分隔器位置
  QByteArray splitterState =
      settings.value("CropWidget/SplitterPosition").toByteArray();
  if (!splitterState.isEmpty()) {
    mainSplitter->restoreState(splitterState);
  }

  // 加载比例设置
  if (customRatioWidth && customRatioHeight) {
    double ratioWidth =
        settings.value("CropWidget/RatioWidth", 16.0).toDouble();
    double ratioHeight =
        settings.value("CropWidget/RatioHeight", 9.0).toDouble();

    customRatioWidth->setValue(ratioWidth);
    customRatioHeight->setValue(ratioHeight);
  }
}

void CropWidget::registerExternalActions(QMap<QString, QAction *> &actions) {
  // 连接外部提供的操作到我们的功能
  if (actions.contains("crop")) {
    connect(actions["crop"], &QAction::triggered, this,
            &CropWidget::onCropClicked);
  }

  if (actions.contains("cancel")) {
    connect(actions["cancel"], &QAction::triggered, this,
            &CropWidget::onCancelClicked);
  }

  if (actions.contains("reset")) {
    connect(actions["reset"], &QAction::triggered, this,
            &CropWidget::onResetCrop);
  }

  if (actions.contains("autoDetect")) {
    connect(actions["autoDetect"], &QAction::triggered, this,
            &CropWidget::onAutoDetect);
  }

  if (actions.contains("histogram")) {
    connect(actions["histogram"], &QAction::triggered, this,
            &CropWidget::showHistogram);
  }

  if (actions.contains("copyToClipboard")) {
    connect(actions["copyToClipboard"], &QAction::triggered, this,
            &CropWidget::onSaveToClipboard);
  }

  if (actions.contains("pasteFromClipboard")) {
    connect(actions["pasteFromClipboard"], &QAction::triggered, this,
            &CropWidget::onPreviewClipboardImage);
  }
}

void CropWidget::setZoomLevel(double level) {
  if (previewWidget) {
    previewWidget->setZoom(level);
  }
}

void CropWidget::setCropMode(int mode) {
  if (strategyCombo) {
    strategyCombo->setCurrentIndex(mode);
  }
}

bool CropWidget::hasActiveImage() const { return !sourceImage.empty(); }

void CropWidget::resetImage() {
  if (!originalImage.empty()) {
    sourceImage = originalImage.clone();
    previewWidget->setImage(sourceImage);
    updatePreview();
  } else {
    onResetCrop();
  }
}

void CropWidget::savePresetToFile(const QString &filePath) {
  QFile file(filePath);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
    QMessageBox::warning(this, tr("错误"), tr("无法保存预设文件"));
    return;
  }

  QJsonObject root;
  QJsonArray presetsArray;

  for (auto it = presets.begin(); it != presets.end(); ++it) {
    QJsonObject preset;
    preset["name"] = it.key();

    // 这里需要序列化CropStrategy
    // 简单示例，真实实现需要处理所有类型
    std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;

          if constexpr (std::is_same_v<T, cv::Rect>) {
            preset["type"] = "rect";
            preset["x"] = arg.x;
            preset["y"] = arg.y;
            preset["width"] = arg.width;
            preset["height"] = arg.height;
          } else if constexpr (std::is_same_v<T, CircleCrop>) {
            preset["type"] = "circle";
            preset["centerX"] = arg.center.x;
            preset["centerY"] = arg.center.y;
            preset["radius"] = arg.radius;
          } else if constexpr (std::is_same_v<T, EllipseCrop>) {
            preset["type"] = "ellipse";
            preset["centerX"] = arg.center.x;
            preset["centerY"] = arg.center.y;
            preset["axisWidth"] = arg.axes.width;
            preset["axisHeight"] = arg.axes.height;
            preset["angle"] = arg.angle;
          }
        },
        it.value());

    presetsArray.append(preset);
  }

  root["presets"] = presetsArray;

  QJsonDocument doc(root);
  file.write(doc.toJson());
}

void CropWidget::loadPresetFromFile(const QString &filePath) {
  QFile file(filePath);
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
    QMessageBox::warning(this, tr("错误"), tr("无法打开预设文件"));
    return;
  }

  QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
  if (doc.isNull() || !doc.isObject()) {
    QMessageBox::warning(this, tr("错误"), tr("无效的预设文件格式"));
    return;
  }

  QJsonObject root = doc.object();
  QJsonArray presetsArray = root["presets"].toArray();

  QMap<QString, CropStrategy> loadedPresets;

  for (const auto &item : presetsArray) {
    QJsonObject presetObj = item.toObject();
    QString name = presetObj["name"].toString();
    QString type = presetObj["type"].toString();

    CropStrategy strategy;

    if (type == "rect") {
      int x = presetObj["x"].toInt();
      int y = presetObj["y"].toInt();
      int width = presetObj["width"].toInt();
      int height = presetObj["height"].toInt();
      strategy = cv::Rect(x, y, width, height);
    } else if (type == "circle") {
      int centerX = presetObj["centerX"].toInt();
      int centerY = presetObj["centerY"].toInt();
      int radius = presetObj["radius"].toInt();
      strategy = CircleCrop{cv::Point(centerX, centerY), radius};
    } else if (type == "ellipse") {
      int centerX = presetObj["centerX"].toInt();
      int centerY = presetObj["centerY"].toInt();
      int axisWidth = presetObj["axisWidth"].toInt();
      int axisHeight = presetObj["axisHeight"].toInt();
      double angle = presetObj["angle"].toDouble();
      strategy = EllipseCrop{cv::Point(centerX, centerY),
                             cv::Size(axisWidth, axisHeight), angle};
    }

    loadedPresets[name] = strategy;
  }

  setPresets(loadedPresets);
  updateStatus(tr("已加载预设"));
}

void CropWidget::logOperation(const QString &operation) {
  // 记录操作到日志
  qDebug() << "CropWidget: " << operation;

  // 在生产环境可以使用更复杂的日志系统
}

CropWidget::~CropWidget() = default;
