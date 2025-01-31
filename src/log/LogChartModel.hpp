#pragma once

#include <QObject>

#include <spdlog/common.h>

class QChart;
class QLineSeries;

class LogChartModel : public QObject {
  Q_OBJECT
public:
  explicit LogChartModel(QObject *parent = nullptr);
  void
  updateChart(const QVector<QPair<QDateTime, spdlog::level::level_enum>> &logs);
  QChart *chart() const { return m_chart; }

private:
  QChart *m_chart;
  QLineSeries *m_errorSeries;
  QLineSeries *m_warningSeries;
  QLineSeries *m_infoSeries;
};