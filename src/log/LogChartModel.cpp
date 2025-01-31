#include "LogChartModel.hpp"

#include <QChart>
#include <QDateTime>
#include <QLineSeries>


// LogChartModel implementation
LogChartModel::LogChartModel(QObject *parent) : QObject(parent) {
  m_chart = new QChart();
  m_errorSeries = new QLineSeries();
  m_warningSeries = new QLineSeries();
  m_infoSeries = new QLineSeries();

  m_errorSeries->setName(tr("错误"));
  m_warningSeries->setName(tr("警告"));
  m_infoSeries->setName(tr("信息"));

  m_chart->addSeries(m_errorSeries);
  m_chart->addSeries(m_warningSeries);
  m_chart->addSeries(m_infoSeries);

  m_chart->createDefaultAxes();
  m_chart->setTitle(tr("日志趋势"));
}

void LogChartModel::updateChart(
    const QVector<QPair<QDateTime, spdlog::level::level_enum>> &logs) {
  // 清除旧数据
  m_errorSeries->clear();
  m_warningSeries->clear();
  m_infoSeries->clear();

  // 按时间统计各级别日志数量
  QMap<QDateTime, int> errorCount;
  QMap<QDateTime, int> warningCount;
  QMap<QDateTime, int> infoCount;

  for (const auto &[time, level] : logs) {
    QDateTime hour = time;
    hour.setTime(QTime(time.time().hour(), 0));

    switch (level) {
    case spdlog::level::err:
      errorCount[hour]++;
      break;
    case spdlog::level::warn:
      warningCount[hour]++;
      break;
    case spdlog::level::info:
      infoCount[hour]++;
      break;
    default:
      break;
    }
  }

  // 更新图表数据
  for (auto it = errorCount.begin(); it != errorCount.end(); ++it) {
    m_errorSeries->append(it.key().toMSecsSinceEpoch(), it.value());
  }
  for (auto it = warningCount.begin(); it != warningCount.end(); ++it) {
    m_warningSeries->append(it.key().toMSecsSinceEpoch(), it.value());
  }
  for (auto it = infoCount.begin(); it != infoCount.end(); ++it) {
    m_infoSeries->append(it.key().toMSecsSinceEpoch(), it.value());
  }

  // 更新坐标轴
  m_chart->axes(Qt::Horizontal)
      .first()
      ->setRange(logs.first().first, logs.last().first);
}