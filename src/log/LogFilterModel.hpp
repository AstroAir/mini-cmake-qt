#pragma once

#include <QBitArray>
#include <QSortFilterProxyModel>

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

protected:
  bool filterAcceptsRow(int row, const QModelIndex &parent) const override;

private:
  QString m_searchText;
  QBitArray m_levelFilter;
  QStringList m_highlightTexts;
};