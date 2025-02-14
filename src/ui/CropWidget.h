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
class HistogramDialog;  // 添加前向声明

class CropWidget : public QWidget {
  Q_OBJECT

public:
  explicit CropWidget(QWidget *parent = nullptr);
  ~CropWidget();

  void setImage(const cv::Mat &image);
  cv::Mat getResult() const;

  enum class CropState { Ready, Processing, Error };

  // 添加新的公共方法
  void setTheme(const QString &themeName);
  void enableAdvancedMode(bool enable);
  void setPresets(const QMap<QString, CropStrategy> &newPresets);

signals:
  void cropFinished(const cv::Mat &result);
  void cropCanceled();

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
  void showHistogram();  // 新增方法

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

  ElaSlider *brightnessSlider;
  ElaSlider *contrastSlider;

  QGroupBox *presetBox;
  ElaComboBox *presetCombo;

  ElaStatusBar *statusBar;
  QStackedWidget *stackedWidget;
  CropState currentState;
  QString lastError;
  QTimer *statusTimer;

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
  } actions;

  std::vector<CropStrategy> undoStack;
  std::vector<CropStrategy> redoStack;

  cv::Mat sourceImage;
  cv::Mat resultImage;
  std::unique_ptr<ImageCropper> cropper;
  CropperConfig config;
  double currentRotation = 0.0;
  std::map<QString, CropStrategy> presets;

  // 新增成员变量
  // QLabel *histogramLabel;
  ElaCheckBox *gridCheckBox;
  ElaCheckBox *aspectLockCheckBox;
  ElaDoubleSpinBox *customRatioWidth;
  ElaDoubleSpinBox *customRatioHeight;
  QWidget *advancedPanel;
  bool isAdvancedMode;
  bool isGridVisible;
  bool isAspectRatioLocked;
  QString currentTheme;
  std::unique_ptr<HistogramDialog> histogramDialog;  // 添加新的直方图对话框成员
};
