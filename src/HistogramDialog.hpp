#ifndef HISTOGRAMDIALOG_H
#define HISTOGRAMDIALOG_H

#include <QDialog>
#include <QtCharts/QtCharts>
#include <opencv2/opencv.hpp>
#include <vector>

class HistogramDialog : public QDialog {
  Q_OBJECT
public:
  explicit HistogramDialog(QWidget *parent = nullptr);
  void showHistogram(const cv::Mat &image, int bins = 256,
                     bool normalize = true);

private:
  QVBoxLayout layout;
  QChart *chart;
  QChartView *chartView;

  void updateChart(const std::vector<cv::Mat> &histograms, int bins);
};

void showImageHistogram(QWidget *parent, const cv::Mat &image);

#endif // HISTOGRAMDIALOG_H