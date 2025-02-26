#pragma once

#include "../image/Crop.h"
#include <QWidget>
#include <memory>

class QLabel;
class ElaPushButton;
class ElaComboBox;
class ElaSpinBox;
class ElaDoubleSpinBox;
class CropPreviewWidget;
class ElaToolButton;
class ElaSlider;
class QGroupBox;
class ElaStatusBar;
class QStackedWidget;
class ElaCheckBox;
class HistogramDialog;
class QProgressBar;  // 添加进度条
class QSplitter;     // 添加分隔器
class QToolBar;      // 添加工具栏
class QHBoxLayout;
class QMenu;         // 添加上下文菜单

class CropWidget : public QWidget {
  Q_OBJECT

public:
  explicit CropWidget(QWidget *parent = nullptr);
  ~CropWidget();

  void setImage(const cv::Mat &image);
  cv::Mat getResult() const;

  enum class CropState { Ready, Processing, Error, Success };

  // 公共方法
  void setTheme(const QString &themeName);
  void enableAdvancedMode(bool enable);
  void setPresets(const QMap<QString, CropStrategy> &newPresets);
  void loadPresetFromFile(const QString &filePath);
  void savePresetToFile(const QString &filePath);
  void registerExternalActions(QMap<QString, QAction*> &actions);
  void setZoomLevel(double level);
  void setCropMode(int mode);
  bool hasActiveImage() const;
  void resetImage();

signals:
  void cropFinished(const cv::Mat &result);
  void cropCanceled();
  void imageChanged();
  void zoomLevelChanged(double zoom);
  void statusUpdated(const QString &message);
  void processingProgress(int percent);

private slots:
  void onStrategyChanged(int index);
  void onCropClicked();
  void onCancelClicked();
  void updatePreview();
  void onRotateLeft();
  void onRotateRight();
  void onZoomIn();
  void onZoomOut();
  void onFitToView();
  void onResetCrop();
  void onLoadPreset();
  void onSavePreset();
  void onAutoDetect();
  void showError(const QString &message);
  void updateStatus(const QString &message);
  void setState(CropState state);
  void onThemeChanged();
  void onAdvancedModeToggled(bool enabled);
  void onHistogramUpdate();
  void onGridToggled(bool show);
  void onAspectRatioLocked(bool locked);
  void onCustomRatioChanged();
  void onImageDropped(const QString &imagePath);
  void onShowContextMenu(const QPoint &pos);
  void onPreviewClipboardImage();
  void onSaveToClipboard();
  void onCropProgress(int percent);
  void onCenterCrop();
  void onFitToBounds();
  void onKeyPressed(QKeyEvent *event);
  void onMouseWheelZoom(QWheelEvent *event);
  void onSplitterMoved(int pos, int index);
  void onApplyFilter(int filterType);

private:
  void setupUi();
  void connectSignals();
  CropStrategy getCurrentStrategy() const;
  AdaptiveParams getAdaptiveParams() const;
  void setupToolbar();
  void setupShortcuts();
  void rotateImage(int angle);
  void savePreset(const QString &name);
  void loadPreset(const QString &name);
  void createActions();
  QWidget *createStrategyGroup();
  QWidget *createAdjustmentGroup();
  QWidget *createPresetsGroup();
  QWidget *createFilterGroup(); // 添加滤镜组
  void createMenus();
  bool confirmOperation();
  void handleException(const std::exception &e);
  void setupLayout();
  void updateUndoRedoState();
  QWidget *createHistogramView();
  QWidget *createAdvancedPanel();
  void updateHistogram();
  void setupTheme();
  void createThemeMenu();
  QWidget* createToolPanel();
  void showHistogram();
  void logOperation(const QString &operation);
  void startProgress(const QString &operationName);
  void updateProgress(int percent);
  void finishProgress();
  void createContextMenu();
  cv::Mat applyFilter(const cv::Mat &image, int filterType);
  bool checkImageSize(const cv::Mat &image);
  QWidget* createAspectRatioPanel();
  void createNotifications();
  void showNotification(const QString &message, int durationMs = 3000);
  void setControlsEnabled(bool enabled);
  void saveSessionState();
  void loadSessionState();
  
  // UI组件
  QSplitter *mainSplitter;
  QToolBar *mainToolBar;
  QProgressBar *progressBar;
  QLabel *imageLabel;
  CropPreviewWidget *previewWidget;
  ElaComboBox *strategyCombo;
  ElaPushButton *cropButton;
  ElaPushButton *cancelButton;
  ElaSpinBox *marginSpin;
  ElaDoubleSpinBox *ratioSpin;

  ElaToolButton *rotateLeftBtn;
  ElaToolButton *rotateRightBtn;
  ElaToolButton *zoomInBtn;
  ElaToolButton *zoomOutBtn;
  ElaToolButton *fitViewBtn;
  ElaToolButton *resetBtn;
  ElaToolButton *autoDetectBtn;
  ElaToolButton *centerBtn;        // 新增中心裁剪按钮
  ElaToolButton *clipboardBtn;     // 新增剪贴板按钮

  ElaSlider *brightnessSlider;
  ElaSlider *contrastSlider;
  ElaSlider *saturationSlider;     // 新增饱和度滑块
  ElaSlider *sharpenSlider;        // 新增锐化滑块

  QGroupBox *presetBox;
  ElaComboBox *presetCombo;
  ElaComboBox *filterCombo;        // 新增滤镜下拉框

  ElaStatusBar *statusBar;
  QStackedWidget *stackedWidget;
  QMenu *contextMenu;              // 新增上下文菜单
  
  CropState currentState;
  QString lastError;
  QTimer *statusTimer;
  QTimer *autosaveTimer;           // 新增自动保存定时器

  struct {
    QAction *undo;
    QAction *redo;
    QAction *reset;
    QAction *help;
    QAction *darkTheme;
    QAction *lightTheme;
    QAction *systemTheme;
    QAction *customTheme;
    QAction *toggleAdvanced;
    QAction *showGrid;
    QAction *lockAspectRatio;
    QAction *copyToClipboard;      // 新增复制到剪贴板
    QAction *pasteFromClipboard;   // 新增从剪贴板粘贴
    QAction *centerCrop;           // 新增中心裁剪
    QAction *fitToBounds;          // 新增适应边界
  } actions;

  std::vector<CropStrategy> undoStack;
  std::vector<CropStrategy> redoStack;
  int undoLimit = 20;             // 设置撤销栈上限

  cv::Mat sourceImage;
  cv::Mat resultImage;
  cv::Mat originalImage;          // 保留原始图像备份
  std::unique_ptr<ImageCropper> cropper;
  CropperConfig config;
  double currentRotation = 0.0;
  QMap<QString, CropStrategy> presets;

  // 交互状态变量
  ElaCheckBox *gridCheckBox;
  ElaCheckBox *aspectLockCheckBox;
  ElaDoubleSpinBox *customRatioWidth;
  ElaDoubleSpinBox *customRatioHeight;
  QWidget *advancedPanel;
  bool isAdvancedMode;
  bool isGridVisible;
  bool isAspectRatioLocked;
  QString currentTheme;
  bool isProcessing;
  QString currentOperation;
  double currentProgress;
  bool isAutosaveEnabled;
  
  // 直方图对话框
  std::unique_ptr<HistogramDialog> histogramDialog;
  
  // 性能相关变量
  QSize optimizedSize;
  bool isDownscaled;
  cv::Mat fullResImage;
};
