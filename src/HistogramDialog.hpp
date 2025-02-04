#ifndef HISTOGRAMDIALOG_H
#define HISTOGRAMDIALOG_H

#include <QDialog>
#include <QtCharts/QtCharts>
#include <opencv2/opencv.hpp>
#include <vector>
#include "image/Histogram.hpp"  // 添加头文件引用

class HistogramDialog : public QDialog {
  Q_OBJECT
public:
  explicit HistogramDialog(QWidget *parent = nullptr);
  void showHistogram(const cv::Mat &image, const HistogramConfig &config = {});
  void showChannelHistograms();
  void exportHistogramData();
  void compareHistograms(const cv::Mat& image1, const cv::Mat& image2);

private slots:
  void updateHistogramView();
  void saveHistogramAsImage();
  void toggleLogScale();
  void changeBinCount(int bins);

private:
  QVBoxLayout layout;
  QChart *chart;
  QChartView *chartView;
  QCheckBox* logScaleCheckBox;
  QSpinBox* binCountSpinner;
  QPushButton* exportButton;
  QPushButton* saveImageButton;
  cv::Mat currentImage;
  HistogramConfig config;  // 添加配置成员
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
  QProgressBar* progressBar;
  QLabel* statusLabel;
  QTimer* updateTimer;
  std::atomic<bool> processingFlag;
    
  // 添加新的方法
  void showError(const QString& message);
  void showProgress(int value);
  void resetProgress();
  void enableControls(bool enable);
  void processWithProgress(const std::function<void()>& task);
};

void showImageHistogram(QWidget *parent, const cv::Mat &image);

#endif // HISTOGRAMDIALOG_H