#pragma once

#include <QMap>
#include <QObject>


class QChart;

class LogAnalyzer : public QObject {
  Q_OBJECT
public:
  explicit LogAnalyzer(QObject *parent = nullptr);
  struct Statistics {
    int totalLogs;
    QMap<QString, int> levelCounts;
    QMap<QDateTime, int> timeDistribution;
    QStringList mostFrequentMessages;
    double averageMessageLength;
  };

  Statistics analyze(const QVector<QPair<QDateTime, QString>> &logs);
  QChart *createDistributionChart(const Statistics &stats);
};