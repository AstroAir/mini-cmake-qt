#include "HistogramDialog.hpp"
#include "image/Histogram.hpp"
#include <QMessageBox>

HistogramDialog::HistogramDialog(QWidget *parent)
    : QDialog(parent), chart(new QChart), chartView(new QChartView(chart)) {
  setWindowTitle("Image Histogram");
  setMinimumSize(800, 600);
  layout.addWidget(chartView);
  setLayout(&layout);
}

void HistogramDialog::showHistogram(const cv::Mat &image, int bins,
                                    bool normalize) {
  try {
    std::vector<cv::Mat> histograms;
    if (image.channels() == 1) {
      histograms.push_back(calculateGrayHist(image, bins, normalize));
    } else {
      histograms = calculateHist(image, bins, normalize);
    }
    updateChart(histograms, bins);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("Failed to calculate histogram: ") +
                             e.what());
  }
}

void HistogramDialog::updateChart(const std::vector<cv::Mat> &histograms,
                                  int bins) {
  chart->removeAllSeries();
  QValueAxis *axisX = new QValueAxis;
  QValueAxis *axisY = new QValueAxis;
  axisX->setRange(0, 255);
  axisY->setRange(0, 1.0);
  axisX->setTitleText("Pixel Value");
  axisY->setTitleText("Normalized Frequency");

  const QColor colors[] = {Qt::blue, Qt::green, Qt::red, Qt::gray};
  const QString names[] = {"Blue", "Green", "Red", "Gray"};

  for (size_t i = 0; i < histograms.size(); ++i) {
    QLineSeries *series = new QLineSeries;
    series->setName(names[i]);
    series->setColor(colors[i]);

    const cv::Mat &hist = histograms[i];
    for (int j = 0; j < bins; ++j) {
      series->append(j, hist.at<float>(j));
    }

    chart->addSeries(series);
    series->attachAxis(axisX);
    series->attachAxis(axisY);
  }

  chart->addAxis(axisX, Qt::AlignBottom);
  chart->addAxis(axisY, Qt::AlignLeft);
  chart->setTitle("Image Histogram");
  chart->legend()->setVisible(true);
}

void showImageHistogram(QWidget *parent, const cv::Mat &image) {
  try {
    HistogramDialog dialog(parent);
    dialog.showHistogram(image);
    dialog.exec();
  } catch (const std::exception &e) {
    QMessageBox::critical(parent, "Error",
                          QString("Histogram Error: ") + e.what());
  }
}