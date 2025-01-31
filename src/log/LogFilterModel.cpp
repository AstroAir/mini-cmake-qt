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

void LogFilterModel::setCustomFilter(const QString &pattern, bool isRegex) {
  m_customFilterPattern = pattern;
  m_isCustomFilterRegex = isRegex;
  invalidateFilter();
}

void LogFilterModel::setHighlightPatterns(const QStringList &patterns) {
  m_highlightRules.clear();

  QVector<QColor> colors = {
      QColor("#FFD700"), // Gold
      QColor("#FF69B4"), // Pink
      QColor("#98FB98"), // Green
      QColor("#87CEEB"), // Sky Blue
      QColor("#DDA0DD")  // Plum
  };

  int colorIndex = 0;
  for (const auto &pattern : patterns) {
    HighlightRule rule;
    rule.pattern = pattern;
    rule.color = colors[colorIndex % colors.size()];
    rule.isRegex = pattern.startsWith('/') && pattern.endsWith('/');
    if (rule.isRegex) {
      rule.pattern = pattern.mid(1, pattern.length() - 2);
    }
    m_highlightRules.append(rule);
    colorIndex++;
  }
  emit filterChanged();
}

bool LogFilterModel::hasHighlight(const QString &text) const {
  for (const auto &rule : m_highlightRules) {
    if (rule.isRegex) {
      QRegularExpression regex(rule.pattern,
                               QRegularExpression::CaseInsensitiveOption);
      if (regex.match(text).hasMatch())
        return true;
    } else if (text.contains(rule.pattern, Qt::CaseInsensitive)) {
      return true;
    }
  }
  return false;
}

QColor LogFilterModel::highlightColor(const QString &text) const {
  for (const auto &rule : m_highlightRules) {
    if (rule.isRegex) {
      QRegularExpression regex(rule.pattern,
                               QRegularExpression::CaseInsensitiveOption);
      if (regex.match(text).hasMatch())
        return rule.color;
    } else if (text.contains(rule.pattern, Qt::CaseInsensitive)) {
      return rule.color;
    }
  }
  return QColor();
}

bool LogFilterModel::filterAcceptsRow(int row,
                                      const QModelIndex &parent) const {
  const auto &idx = sourceModel()->index(row, 0, parent);
  const auto text = idx.data().toString();
  const auto level = idx.data(Qt::UserRole).toInt();

  // 检查日志级别过滤
  if (!m_levelFilter.testBit(level))
    return false;

  // 检查搜索文本
  if (!m_searchText.isEmpty() &&
      !text.contains(m_searchText, Qt::CaseInsensitive)) {
    return false;
  }

  // 检查自定义过滤器
  if (!m_customFilterPattern.isEmpty()) {
    if (m_isCustomFilterRegex) {
      QRegularExpression regex(m_customFilterPattern);
      if (!regex.match(text).hasMatch())
        return false;
    } else if (!text.contains(m_customFilterPattern, Qt::CaseInsensitive)) {
      return false;
    }
  }

  return true;
}
