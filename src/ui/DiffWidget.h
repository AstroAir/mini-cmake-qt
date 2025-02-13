#pragma once

#include "../image/Diff.hpp"
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
class QProgressBar;

class DiffWidget : public QWidget {
  Q_OBJECT

public:
  explicit DiffWidget(QWidget *parent = nullptr);
  ~DiffWidget();

  void setSourceImage(const QImage &image);
  void setTargetImage(const QImage &image);
  ComparisonResult getResult() const;

  enum class DiffState {
    Ready,
    Processing,
    Error
  };

signals:
  void diffFinished(const ComparisonResult &result);
  void diffCanceled();

private slots:
  void onStrategyChanged(int index);
  void onCompareClicked();
  void onCancelClicked();
  void updatePreview();
  void onZoomIn();
  void onZoomOut();
  void onFitToView();
  void onReset();
  void onSaveResult();
  void showError(const QString& message);
  void updateStatus(const QString& message);
  void setState(DiffState state);
  void updateProgress(int value);

private:
  void setupUi();
  void connectSignals();
  void setupToolbar();
  void setupShortcuts();
  QWidget* createStrategyGroup();
  QWidget* createParametersGroup();
  QWidget* createPreviewGroup();
  void handleException(const std::exception& e);
  void setupLayout();

  CropPreviewWidget *sourcePreview;
  CropPreviewWidget *targetPreview;
  CropPreviewWidget *diffPreview;
  
  QComboBox *strategyCombo;
  QPushButton *compareButton;
  QPushButton *cancelButton;
  QSpinBox *thresholdSpin;
  QDoubleSpinBox *sensitivitySpin;

  QToolButton *zoomInBtn;
  QToolButton *zoomOutBtn;
  QToolButton *fitViewBtn;
  QToolButton *resetBtn;
  QToolButton *saveBtn;

  QProgressBar *progressBar;
  QStatusBar *statusBar;
  QStackedWidget *stackedWidget;
  
  DiffState currentState;
  QString lastError;
  QTimer *statusTimer;

  QImage sourceImage;
  QImage targetImage;
  ComparisonResult currentResult;
  std::unique_ptr<ImageDiff> differ;

  struct {
    PixelDifferenceStrategy pixel;
    SSIMStrategy ssim;
    PerceptualHashStrategy perceptual;
    HistogramStrategy histogram;
  } strategies;

  QFuture<ComparisonResult> currentOperation;
};
