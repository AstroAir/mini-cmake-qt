#include "LogAnalyzer.hpp"

#include <QBarSeries>
#include <QBarSet>
#include <QChart>
#include <QDateTime>
#include <QDateTimeAxis>
#include <QRegularExpression>
#include <QValueAxis>

LogAnalyzer::LogAnalyzer(QObject *parent) : QObject(parent) {}

LogAnalyzer::Statistics
LogAnalyzer::analyze(const QVector<QPair<QDateTime, QString>> &logs) {
  Statistics stats;
  stats.totalLogs = logs.size();

  QMap<QString, int> messageFrequency;
  qint64 totalLength = 0;

  // 正则表达式用于匹配日志级别
  QRegularExpression levelRegex("\\[(INFO|WARNING|ERROR|DEBUG)\\]");

  for (const auto &[time, message] : logs) {
    // 时间分布统计
    QDateTime startOfHour = time;
    startOfHour.setTime(QTime(time.time().hour(), 0));
    stats.timeDistribution[startOfHour]++;

    // 日志级别统计
    auto match = levelRegex.match(message);
    if (match.hasMatch()) {
      QString level = match.captured(1);
      stats.levelCounts[level]++;
    }

    // 消息频率统计
    messageFrequency[message]++;
    totalLength += message.length();
  }

  // 获取最频繁的消息
  QVector<QPair<QString, int>> sortedMessages;
  for (auto it = messageFrequency.begin(); it != messageFrequency.end(); ++it) {
    sortedMessages.append({it.key(), it.value()});
  }
  std::sort(sortedMessages.begin(), sortedMessages.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

  stats.mostFrequentMessages = QStringList();
  for (int i = 0; i < qMin(5, sortedMessages.size()); ++i) {
    stats.mostFrequentMessages << sortedMessages[i].first;
  }

  stats.averageMessageLength = totalLength / static_cast<double>(logs.size());
  return stats;
}

QChart *LogAnalyzer::createDistributionChart(const Statistics &stats) {
  auto chart = new QChart();

  // 创建时间分布柱状图
  auto series = new QBarSeries();
  auto barSet = new QBarSet("日志数量");

  QVector<QDateTime> sortedTimes = stats.timeDistribution.keys().toVector();
  std::sort(sortedTimes.begin(), sortedTimes.end());

  for (const auto &time : sortedTimes) {
    *barSet << stats.timeDistribution[time];
  }

  series->append(barSet);
  chart->addSeries(series);

  // 设置坐标轴
  auto timeAxis = new QDateTimeAxis;
  timeAxis->setFormat("MM-dd hh:00");
  timeAxis->setTitleText("时间");
  chart->addAxis(timeAxis, Qt::AlignBottom);
  series->attachAxis(timeAxis);

  auto valueAxis = new QValueAxis;
  valueAxis->setTitleText("日志数量");
  chart->addAxis(valueAxis, Qt::AlignLeft);
  series->attachAxis(valueAxis);

  if (!sortedTimes.isEmpty()) {
    timeAxis->setRange(sortedTimes.first(), sortedTimes.last());
  }

  chart->setTitle("日志时间分布");
  chart->legend()->hide();

  return chart;
}