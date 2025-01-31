#include "LogFilterModel.hpp"

// LogFilterModel implementation
LogFilterModel::LogFilterModel(QObject *parent)
    : QSortFilterProxyModel(parent) {
  m_levelFilter.resize(spdlog::level::n_levels);
  m_levelFilter.fill(true);
}

void LogFilterModel::setSearchText(const QString &text) {
  m_searchText = text;
  emit filterChanged();
}

bool LogFilterModel::filterAcceptsRow(int row,
                                      const QModelIndex &parent) const {
  const auto &idx = sourceModel()->index(row, 0, parent);
  const bool levelMatch = m_levelFilter.testBit(idx.data(Qt::UserRole).toInt());
  const bool textMatch =
      idx.data().toString().contains(m_searchText, Qt::CaseInsensitive);
  return levelMatch && textMatch;
}
