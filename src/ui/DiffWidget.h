#pragma once

#include "../image/Diff.hpp"
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
class ElaStackedWidget;
class ElaProgressBar;
class QSplitter;
class ElaDockWidget;
class ElaMenu;
class QAction;

class DiffWidget : public QWidget {
  Q_OBJECT

public:
  explicit DiffWidget(QWidget *parent = nullptr);
  ~DiffWidget();

  void setSourceImage(const QImage &image);
  void setTargetImage(const QImage &image);
  ComparisonResult getResult() const;

  void loadSettings();
  void saveSettings();
  void exportReport(const QString &filePath);

  enum class DiffState { Ready, Processing, Error };

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
  void showError(const QString &message);
  void updateStatus(const QString &message);
  void setState(DiffState state);
  void updateProgress(int value);
  void onSplitHorizontally();
  void onSplitVertically();
  void onToggleToolPanel();
  void onToggleStatusBar();
  void onCustomizeToolbar();
  void onResetLayout();
  void contextMenuEvent(QContextMenuEvent *event) override;

private:
  void setupUi();
  void connectSignals();
  void setupToolbar();
  void setupShortcuts();
  QWidget *createStrategyGroup();
  QWidget *createParametersGroup();
  QWidget *createPreviewGroup();
  QWidget *createToolPanel();
  void handleException(const std::exception &e);
  void setupLayout();
  void createActions();
  void createMenus();
  void updateLayout(Qt::Orientation orientation);
  void writeReportHeader(QTextStream &out);
  void writeReportBody(QTextStream &out);

  CropPreviewWidget *sourcePreview;
  CropPreviewWidget *targetPreview;
  CropPreviewWidget *diffPreview;

  ElaComboBox *strategyCombo;
  ElaPushButton *compareButton;
  ElaPushButton *cancelButton;
  ElaSpinBox *thresholdSpin;
  ElaDoubleSpinBox *sensitivitySpin;

  ElaToolButton *zoomInBtn;
  ElaToolButton *zoomOutBtn;
  ElaToolButton *fitViewBtn;
  ElaToolButton *resetBtn;
  ElaToolButton *saveBtn;

  ElaProgressBar *progressBar;
  ElaStatusBar *statusBar;
  ElaStackedWidget *stackedWidget;
  QSplitter *mainSplitter;
  ElaDockWidget *toolPanelDock;
  QMenu *viewMenu;
  QMenu *toolsMenu;

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

  struct {
    QAction *splitHorizontal;
    QAction *splitVertical;
    QAction *toggleToolPanel;
    QAction *toggleStatusBar;
    QAction *customizeToolbar;
    QAction *resetLayout;
    QAction *exportReport;
  } actions;

  // 保存布局状态
  Qt::Orientation splitOrientation;
  bool toolPanelVisible;
  bool statusBarVisible;
};
