#ifndef HISTOGRAMDIALOG_H
#define HISTOGRAMDIALOG_H

#include "../image/Histogram.hpp"
#include <QDialog>
#include <QtCharts/QtCharts>
#include <opencv2/opencv.hpp>
#include <vector>

class ElaComboBox;
class ElaPushButton;
class ElaSpinBox;
class ElaCheckBox;
class ElaProgressBar;
class ElaListView;

class HistogramDialog : public QDialog {
  Q_OBJECT
public:
  explicit HistogramDialog(QWidget *parent = nullptr);
  ~HistogramDialog() = default;
  void showHistogram(const cv::Mat &image, const HistogramConfig &config = {});
  void setTitle(const QString& title);
  void setOptions(bool showRGB, bool showGray, bool logScale);
  void showChannelHistograms();
  void exportHistogramData();
  void compareHistograms(const cv::Mat &image1, const cv::Mat &image2);
  void showHistogramWithEqualization(const cv::Mat &image);
  void performHistogramMatching(const cv::Mat &source,
                                const cv::Mat &reference);
  void addHistogramSeries(const cv::Mat &image, const QString &name);
  void clearHistograms();

private slots:
  void updateHistogramView();
  void saveHistogramAsImage();
  void toggleLogScale();
  void changeBinCount(int bins);
  void toggleEqualizationPreview();
  void exportToFormat();
  void removeSelectedSeries();
  void showHistogramAnalysis();
  void onChannelChanged(int index);
  void onBinsChanged(int bins);
  void onLogScaleToggled(bool checked);
  void onSaveClicked();
  void onCopyClicked();

private:
  QVBoxLayout layout;
  QChart *chart;
  QChartView *chartView;
  ElaCheckBox *logScaleCheckBox;
  ElaSpinBox *binCountSpinner;
  ElaPushButton *exportButton;
  ElaPushButton *saveImageButton;
  cv::Mat currentImage;
  HistogramConfig config; // 添加配置成员
  std::vector<double> calculateMoments();
  void updateStatistics();

  void updateChart(const std::vector<cv::Mat> &histograms);
  void setupUI();
  void calculateStatistics();
  double calculateKurtosis();
  double calculateSkewness();
  void drawStatisticsOverlay();
  void updateStatisticsDisplay(const HistogramStats &stats);

  // 添加新的成员
  ElaProgressBar *progressBar;
  QLabel *statusLabel;
  QTimer *updateTimer;
  std::atomic<bool> processingFlag;
  ElaComboBox *exportFormatCombo;
  ElaCheckBox *equalizationCheckBox;
  ElaListView *seriesList;       // 改为
  QStringListModel *seriesModel; // 添加model
  ElaPushButton *analyzeButton;
  ElaPushButton *removeSeriesButton;

  // 添加新的方法
  void showError(const QString &message);
  void showProgress(int value);
  void resetProgress();
  void enableControls(bool enable);
  void processWithProgress(const std::function<void()> &task);
  void setupAdvancedUI(QVBoxLayout *parent);
  void updateEqualizationPreview();
  cv::Mat performEqualization(const cv::Mat &input);
  void exportHistogramAs(const QString &format);
  QString generateHistogramReport();
  void showAnalysisDialog();
};

void showImageHistogram(QWidget *parent, const cv::Mat &image);

#endif // HISTOGRAMDIALOG_H