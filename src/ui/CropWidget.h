#pragma once

#include "../image/Crop.h"
#include <QWidget>
#include <memory>

class QLabel;
class QPushButton;
class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class CropPreviewWidget;
class QToolButton;
class QSlider;
class QGroupBox;
class QStatusBar;
class QStackedWidget;

class CropWidget : public QWidget {
  Q_OBJECT

public:
  explicit CropWidget(QWidget *parent = nullptr);
  ~CropWidget();

  void setImage(const cv::Mat &image);
  cv::Mat getResult() const;

  enum class CropState {
    Ready,
    Processing,
    Error
  };

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
  void showError(const QString& message);
  void updateStatus(const QString& message);
  void setState(CropState state);

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
  QWidget* createStrategyGroup();
  QWidget* createAdjustmentGroup();
  QWidget* createPresetsGroup();
  void createMenus();
  bool confirmOperation();
  void handleException(const std::exception& e);
  void setupLayout();
  void updateUndoRedoState();

  QLabel *imageLabel;
  CropPreviewWidget *previewWidget;
  QComboBox *strategyCombo;
  QPushButton *cropButton;
  QPushButton *cancelButton;
  QSpinBox *marginSpin;
  QDoubleSpinBox *ratioSpin;

  QToolButton *rotateLeftBtn;
  QToolButton *rotateRightBtn;
  QToolButton *zoomInBtn;
  QToolButton *zoomOutBtn;
  QToolButton *fitViewBtn;
  QToolButton *resetBtn;
  QToolButton *autoDetectBtn;

  QSlider *brightnessSlider;
  QSlider *contrastSlider;

  QGroupBox *presetBox;
  QComboBox *presetCombo;

  QStatusBar* statusBar;
  QStackedWidget* stackedWidget;
  CropState currentState;
  QString lastError;
  QTimer* statusTimer;

  struct {
    QAction* undo;
    QAction* redo;
    QAction* reset;
    QAction* help;
  } actions;

  std::vector<CropStrategy> undoStack;
  std::vector<CropStrategy> redoStack;

  cv::Mat sourceImage;
  cv::Mat resultImage;
  std::unique_ptr<ImageCropper> cropper;
  CropperConfig config;
  double currentRotation = 0.0;
  std::map<QString, CropStrategy> presets;
};
