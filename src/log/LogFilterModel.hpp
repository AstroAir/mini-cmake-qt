#pragma once

#include <QBitArray>
#include <QSortFilterProxyModel>
#include <QColor>

#include <spdlog/spdlog.h>

class LogFilterModel : public QSortFilterProxyModel {
  Q_OBJECT
public:
  Q_PROPERTY(QString searchText MEMBER m_searchText NOTIFY filterChanged)
  Q_PROPERTY(QBitArray levelFilter MEMBER m_levelFilter NOTIFY filterChanged)

  explicit LogFilterModel(QObject *parent = nullptr);

signals:
  void filterChanged();

public:
  void setSearchText(const QString &text);
  void highlightSearchText(const QString &text);
  void setLevelFilter(spdlog::level::level_enum level, bool enabled);
  void setCustomFilter(const QString &pattern, bool isRegex);
  void setHighlightPatterns(const QStringList &patterns);
  bool hasHighlight(const QString &text) const;
  QColor highlightColor(const QString &text) const;

protected:
  bool filterAcceptsRow(int row, const QModelIndex &parent) const override;

private:
  QString m_searchText;
  QBitArray m_levelFilter;
  QStringList m_highlightTexts;
  struct HighlightRule {
    QString pattern;
    QColor color;
    bool isRegex;
  };
  QVector<HighlightRule> m_highlightRules;
  QString m_customFilterPattern;
  bool m_isCustomFilterRegex = false;
};