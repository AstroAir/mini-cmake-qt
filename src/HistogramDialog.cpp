#include "HistogramDialog.hpp"
#include "image/Histogram.hpp"
#include <QMessageBox>
#include <qboxlayout.h>

HistogramDialog::HistogramDialog(QWidget *parent)
    : QDialog(parent), chart(new QChart), chartView(new QChartView(chart)) {
  setWindowTitle("Image Histogram");
  setMinimumSize(800, 600);
  layout.addWidget(chartView);
  setLayout(&layout);
  setupUI();
  config = HistogramConfig{}; // 初始化默认配置
}

void HistogramDialog::showHistogram(const cv::Mat &image,
                                    const HistogramConfig &cfg) {
  if (processingFlag) {
    showError("正在处理中，请稍候...");
    return;
  }

  enableControls(false);
  processWithProgress([this, image, cfg]() {
    try {
      currentImage = image.clone();
      config = cfg;

      showProgress(20);
      std::vector<cv::Mat> histograms;

      if (image.channels() == 1) {
        histograms.push_back(calculateGrayHist(image, config));
      } else {
        histograms = calculateHist(image, config);
      }

      showProgress(60);
      updateChart(histograms);

      if (!histograms.empty()) {
        showProgress(80);
        HistogramStats stats = calculateHistogramStats(histograms[0]);
        updateStatisticsDisplay(stats);
      }

      showProgress(100);
      statusLabel->setText("处理完成");

    } catch (const HistogramException &e) {
      showError(QString("直方图计算错误: %1").arg(e.what()));
    } catch (const std::exception &e) {
      showError(QString("发生错误: %1").arg(e.what()));
    }
  });
}

void HistogramDialog::updateChart(const std::vector<cv::Mat> &histograms) {
  chart->removeAllSeries();
  QValueAxis *axisX = new QValueAxis;
  QValueAxis *axisY = new QValueAxis;

  // 使用配置中的范围
  axisX->setRange(config.range[0], config.range[1]);
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
    for (int j = 0; j < config.histSize; ++j) {
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

void HistogramDialog::showChannelHistograms() {
  if (!chart || chart->series().isEmpty())
    return;

  for (QAbstractSeries *series : chart->series()) {
    series->setVisible(true);
  }
}

void HistogramDialog::exportHistogramData() {
  QString filePath = QFileDialog::getSaveFileName(this, "导出直方图数据",
                                                  QString(), "CSV文件 (*.csv)");

  if (filePath.isEmpty())
    return;

  QFile file(filePath);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
    QMessageBox::critical(this, "错误", "无法创建文件");
    return;
  }

  QTextStream out(&file);
  out << "Value,Blue,Green,Red\n";

  for (int i = 0; i < 256; ++i) {
    out << i;
    for (QAbstractSeries *series : chart->series()) {
      if (QLineSeries *lineSeries = qobject_cast<QLineSeries *>(series)) {
        QPointF point = lineSeries->at(i);
        out << "," << point.y();
      }
    }
    out << "\n";
  }
}

void HistogramDialog::compareHistograms(const cv::Mat &image1,
                                        const cv::Mat &image2) {
  chart->removeAllSeries();

  std::vector<cv::Mat> hist1 = calculateHist(image1, config);
  std::vector<cv::Mat> hist2 = calculateHist(image2, config);

  const QColor colors[] = {Qt::blue, Qt::green, Qt::red};

  for (size_t i = 0; i < hist1.size(); ++i) {
    // 第一张图片的直方图
    QLineSeries *series1 = new QLineSeries;
    series1->setName(QString("Image 1 - %1")
                         .arg(i == 0   ? "Blue"
                              : i == 1 ? "Green"
                                       : "Red"));
    series1->setColor(colors[i]);

    // 第二张图片的直方图
    QLineSeries *series2 = new QLineSeries;
    series2->setName(QString("Image 2 - %1")
                         .arg(i == 0   ? "Blue"
                              : i == 1 ? "Green"
                                       : "Red"));
    series2->setColor(colors[i]);
    series2->setOpacity(0.5);

    for (int j = 0; j < 256; ++j) {
      series1->append(j, hist1[i].at<float>(j));
      series2->append(j, hist2[i].at<float>(j));
    }

    chart->addSeries(series1);
    chart->addSeries(series2);
  }

  // 更新坐标轴
  updateChart(hist1);
}

void HistogramDialog::saveHistogramAsImage() {
  QString fileName = QFileDialog::getSaveFileName(
      this, "保存直方图", QString(), "PNG图像 (*.png);;JPG图像 (*.jpg)");

  if (!fileName.isEmpty()) {
    QPixmap pixmap = chartView->grab();
    pixmap.save(fileName);
  }
}

void HistogramDialog::toggleLogScale() {
  config.useLog = logScaleCheckBox->isChecked(); // 更新配置中的对数标志
  updateHistogramView();
}

void HistogramDialog::changeBinCount(int bins) {
  config.histSize = bins; // 更新配置中的直方图大小
  updateHistogramView();
}

void HistogramDialog::setupUI() {
  setWindowFlags(windowFlags() | Qt::WindowMaximizeButtonHint |
                 Qt::WindowMinimizeButtonHint);

  // 创建主布局
  auto mainLayout = new QVBoxLayout(this);
  mainLayout->setContentsMargins(10, 10, 10, 10);
  mainLayout->setSpacing(10);

  // 创建工具栏
  auto toolBar = new QWidget(this);
  auto toolBarLayout = new QHBoxLayout(toolBar);
  toolBarLayout->setContentsMargins(0, 0, 0, 0);
  toolBarLayout->setSpacing(8);

  // 优化工具栏控件样式
  const QString buttonStyle = R"(
        QPushButton {
            background-color: #f0f0f0;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            min-width: 80px;
            color: #333333;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
        }
        QPushButton:pressed {
            background-color: #d0d0d0;
        }
    )";

  logScaleCheckBox = new QCheckBox("对数刻度", this);
  logScaleCheckBox->setStyleSheet("QCheckBox { padding: 4px; }");

  binCountSpinner = new QSpinBox(this);
  binCountSpinner->setRange(8, 512);
  binCountSpinner->setValue(256);
  binCountSpinner->setSingleStep(8);
  binCountSpinner->setStyleSheet(R"(
        QSpinBox {
            padding: 4px;
            border: 1px solid #cccccc;
            border-radius: 4px;
        }
    )");

  exportButton = new QPushButton("导出数据", this);
  saveImageButton = new QPushButton("保存图像", this);
  exportButton->setStyleSheet(buttonStyle);
  saveImageButton->setStyleSheet(buttonStyle);

  toolBarLayout->addWidget(new QLabel("直方图区间:", this));
  toolBarLayout->addWidget(binCountSpinner);
  toolBarLayout->addWidget(logScaleCheckBox);
  toolBarLayout->addStretch();
  toolBarLayout->addWidget(exportButton);
  toolBarLayout->addWidget(saveImageButton);

  // 优化图表视图
  chartView->setRenderHint(QPainter::Antialiasing);
  chartView->setStyleSheet(R"(
        QChartView {
            background-color: white;
            border: 1px solid #cccccc;
            border-radius: 4px;
        }
    )");

  // 组装布局
  mainLayout->addWidget(toolBar);
  mainLayout->addWidget(chartView, 1); // 添加拉伸因子

  // 连接信号
  connect(logScaleCheckBox, &QCheckBox::toggled, this,
          &HistogramDialog::toggleLogScale);
  connect(binCountSpinner, QOverload<int>::of(&QSpinBox::valueChanged), this,
          &HistogramDialog::changeBinCount);
  connect(exportButton, &QPushButton::clicked, this,
          &HistogramDialog::exportHistogramData);
  connect(saveImageButton, &QPushButton::clicked, this,
          &HistogramDialog::saveHistogramAsImage);

  // 添加状态栏
  auto statusBar = new QWidget(this);
  auto statusLayout = new QHBoxLayout(statusBar);

  progressBar = new QProgressBar(this);
  progressBar->setMaximum(100);
  progressBar->setMinimum(0);
  progressBar->hide();

  statusLabel = new QLabel(this);
  statusLabel->setStyleSheet("color: #666666;");

  statusLayout->addWidget(statusLabel);
  statusLayout->addWidget(progressBar);

  mainLayout->addWidget(statusBar);

  // 添加更新定时器
  updateTimer = new QTimer(this);
  updateTimer->setInterval(100);
  connect(updateTimer, &QTimer::timeout, this, [this]() {
    if (!processingFlag) {
      updateTimer->stop();
      resetProgress();
    }
  });

  // 添加高级功能UI
  setupAdvancedUI(mainLayout);
}

void HistogramDialog::setupAdvancedUI(QVBoxLayout *parent) {
  auto advancedGroup = new QGroupBox("高级功能", this);
  auto advancedLayout = new QVBoxLayout(advancedGroup);

  // 直方图均衡化预览
  equalizationCheckBox = new QCheckBox("预览直方图均衡化效果", this);

  // 导出格式选择
  exportFormatCombo = new QComboBox(this);
  exportFormatCombo->addItems(
      {"CSV", "Excel (XLSX)", "JSON", "XML", "PDF报告"});

  // 系列列表
  seriesList = new QListWidget(this);
  seriesList->setSelectionMode(QAbstractItemView::SingleSelection);

  // 分析按钮
  analyzeButton = new QPushButton("详细分析", this);
  removeSeriesButton = new QPushButton("移除所选系列", this);

  advancedLayout->addWidget(equalizationCheckBox);
  advancedLayout->addWidget(new QLabel("导出格式:", this));
  advancedLayout->addWidget(exportFormatCombo);
  advancedLayout->addWidget(new QLabel("已加载的直方图:", this));
  advancedLayout->addWidget(seriesList);
  advancedLayout->addWidget(removeSeriesButton);
  advancedLayout->addWidget(analyzeButton);

  // 添加到主布局
  parent->addWidget(advancedGroup);

  // 连接信号
  connect(equalizationCheckBox, &QCheckBox::toggled, this,
          &HistogramDialog::toggleEqualizationPreview);
  connect(exportButton, &QPushButton::clicked, this,
          &HistogramDialog::exportToFormat);
  connect(removeSeriesButton, &QPushButton::clicked, this,
          &HistogramDialog::removeSelectedSeries);
  connect(analyzeButton, &QPushButton::clicked, this,
          &HistogramDialog::showHistogramAnalysis);
}

void HistogramDialog::showHistogramWithEqualization(const cv::Mat &image) {
  cv::Mat equalized = performEqualization(image);
  addHistogramSeries(image, "原始图像");
  addHistogramSeries(equalized, "均衡化后");
}

void HistogramDialog::performHistogramMatching(const cv::Mat &source,
                                               const cv::Mat &reference) {
  processWithProgress([this, source, reference]() {
    try {
      cv::Mat matched;
      std::vector<cv::Mat> sourceChannels, refChannels, matchedChannels;

      if (source.channels() == reference.channels()) {
        cv::split(source, sourceChannels);
        cv::split(reference, refChannels);
        matchedChannels.resize(source.channels());

        for (int i = 0; i < source.channels(); ++i) {
          ::matchHistograms(sourceChannels[i], refChannels[i]);
        }

        cv::merge(matchedChannels, matched);

        addHistogramSeries(source, "源图像");
        addHistogramSeries(reference, "参考图像");
        addHistogramSeries(matched, "匹配结果");

      } else {
        showError("源图像和参考图像的通道数不匹配");
      }
    } catch (const std::exception &e) {
      showError(QString("直方图匹配失败: %1").arg(e.what()));
    }
  });
}

void HistogramDialog::addHistogramSeries(const cv::Mat &image,
                                         const QString &name) {
  std::vector<cv::Mat> hists = calculateHist(image, config);

  // 添加到图表
  const QColor colors[] = {Qt::blue, Qt::green, Qt::red};

  for (size_t i = 0; i < hists.size(); ++i) {
    QLineSeries *series = new QLineSeries;
    series->setName(QString("%1 - %2").arg(name).arg(i == 0   ? "Blue"
                                                     : i == 1 ? "Green"
                                                              : "Red"));
    series->setColor(colors[i]);

    for (int j = 0; j < config.histSize; ++j) {
      series->append(j, hists[i].at<float>(j));
    }

    chart->addSeries(series);
  }

  // 添加到列表
  seriesList->addItem(name);
  updateChart(hists);
}

void HistogramDialog::exportToFormat() {
  QString format = exportFormatCombo->currentText();
  QString filter;

  if (format == "CSV") {
    filter = "CSV文件 (*.csv)";
  } else if (format == "Excel (XLSX)") {
    filter = "Excel文件 (*.xlsx)";
  } else if (format == "JSON") {
    filter = "JSON文件 (*.json)";
  } else if (format == "XML") {
    filter = "XML文件 (*.xml)";
  } else if (format == "PDF报告") {
    filter = "PDF文件 (*.pdf)";
  }

  QString fileName =
      QFileDialog::getSaveFileName(this, "导出直方图", QString(), filter);

  if (!fileName.isEmpty()) {
    exportHistogramAs(format);
  }
}

void HistogramDialog::showHistogramAnalysis() {
  QString report = generateHistogramReport();
  showAnalysisDialog();
}

void HistogramDialog::calculateStatistics() {
  if (chart->series().isEmpty())
    return;

  double mean = 0.0, variance = 0.0;
  int totalCount = 0;

  // 计算均值
  for (QAbstractSeries *series : chart->series()) {
    if (QLineSeries *lineSeries = qobject_cast<QLineSeries *>(series)) {
      for (const QPointF &point : lineSeries->points()) {
        mean += point.x() * point.y();
        totalCount += point.y();
      }
    }
  }
  mean /= totalCount;

  // 计算方差
  for (QAbstractSeries *series : chart->series()) {
    if (QLineSeries *lineSeries = qobject_cast<QLineSeries *>(series)) {
      for (const QPointF &point : lineSeries->points()) {
        double diff = point.x() - mean;
        variance += diff * diff * point.y();
      }
    }
  }
  variance /= totalCount;

  // 显示统计信息
  QString stats =
      QString("均值: %.2f\n标准差: %.2f").arg(mean).arg(sqrt(variance));
  chart->setTitle(stats);
}

void HistogramDialog::updateHistogramView() {
  if (currentImage.empty())
    return;
  showHistogram(currentImage, config);
}

double HistogramDialog::calculateKurtosis() {
  std::vector<double> moments = calculateMoments();
  double variance = moments[1];
  double m4 = moments[3];
  return (m4 / (variance * variance)) - 3.0; // 超值峰度
}

double HistogramDialog::calculateSkewness() {
  std::vector<double> moments = calculateMoments();
  double variance = moments[1];
  double m3 = moments[2];
  return m3 / pow(sqrt(variance), 3);
}

std::vector<double> HistogramDialog::calculateMoments() {
  std::vector<double> moments(4, 0.0);
  double mean = 0.0;
  double sum = 0.0;

  // 获取第一个系列的数据点（假设是灰度图或第一个通道）
  QLineSeries *series = qobject_cast<QLineSeries *>(chart->series().first());
  if (!series)
    return moments;

  // 计算均值
  for (const QPointF &point : series->points()) {
    mean += point.x() * point.y();
    sum += point.y();
  }
  mean /= sum;
  moments[0] = mean;

  // 计算高阶矩
  double m2 = 0.0, m3 = 0.0, m4 = 0.0;
  for (const QPointF &point : series->points()) {
    double diff = point.x() - mean;
    double diff2 = diff * diff;
    m2 += diff2 * point.y();
    m3 += diff * diff2 * point.y();
    m4 += diff2 * diff2 * point.y();
  }

  moments[1] = m2 / sum; // 方差
  moments[2] = m3 / sum; // 三阶矩
  moments[3] = m4 / sum; // 四阶矩

  return moments;
}

void HistogramDialog::drawStatisticsOverlay() {
  if (!chart || chart->series().isEmpty())
    return;

  double kurtosis = calculateKurtosis();
  double skewness = calculateSkewness();
  std::vector<double> moments = calculateMoments();

  QString statsText = QString("统计信息:\n"
                              "均值: %.2f\n"
                              "标准差: %.2f\n"
                              "峰度: %.2f\n"
                              "偏度: %.2f")
                          .arg(moments[0])
                          .arg(sqrt(moments[1]))
                          .arg(kurtosis)
                          .arg(skewness);

  // 在图表上添加文本项
  QGraphicsTextItem *textItem = new QGraphicsTextItem(chart);
  textItem->setHtml(QString("<div style='background-color: "
                            "rgba(255,255,255,0.8); padding: 5px;'>%1</div>")
                        .arg(statsText));
  textItem->setPos(chart->plotArea().topRight().x() - 150,
                   chart->plotArea().top() + 10);
  textItem->setDefaultTextColor(Qt::black);

  chart->scene()->addItem(textItem);
}

void HistogramDialog::updateStatisticsDisplay(const HistogramStats &stats) {
  QString statsText = QString("统计信息:\n"
                              "均值: %.2f\n"
                              "标准差: %.2f\n"
                              "峰度: %.2f\n"
                              "偏度: %.2f\n"
                              "熵: %.2f\n"
                              "均匀性: %.2f")
                          .arg(stats.mean)
                          .arg(stats.stdDev)
                          .arg(stats.kurtosis)
                          .arg(stats.skewness)
                          .arg(stats.entropy)
                          .arg(stats.uniformity);

  // 更新图表标题或统计信息显示
  chart->setTitle(statsText);
}

void HistogramDialog::showError(const QString &message) {
  statusLabel->setText(message);
  statusLabel->setStyleSheet("color: #ff0000;");
  QMessageBox::warning(this, "错误", message);
}

void HistogramDialog::showProgress(int value) {
  progressBar->setValue(value);
  progressBar->setVisible(value < 100);
}

void HistogramDialog::resetProgress() {
  progressBar->hide();
  progressBar->setValue(0);
  statusLabel->setStyleSheet("color: #666666;");
  statusLabel->setText("就绪");
  enableControls(true);
}

void HistogramDialog::enableControls(bool enable) {
  logScaleCheckBox->setEnabled(enable);
  binCountSpinner->setEnabled(enable);
  exportButton->setEnabled(enable);
  saveImageButton->setEnabled(enable);
}

void HistogramDialog::processWithProgress(const std::function<void()> &task) {
  processingFlag = true;
  progressBar->show();
  updateTimer->start();

  QThread *worker = new QThread;
  QObject *context = new QObject;
  context->moveToThread(worker);

  connect(worker, &QThread::started, context, [task, this]() {
    task();
    processingFlag = false;
  });

  connect(worker, &QThread::finished, worker, &QThread::deleteLater);
  connect(worker, &QThread::finished, context, &QObject::deleteLater);

  worker->start();
}

void showImageHistogram(QWidget *parent, const cv::Mat &image) {
  try {
    HistogramDialog dialog(parent);
    HistogramConfig config;
    config.histSize = 256;
    config.normalize = true;
    dialog.showHistogram(image, config);
    dialog.exec();
  } catch (const std::exception &e) {
    QMessageBox::critical(parent, "Error",
                          QString("Histogram Error: ") + e.what());
  }
}