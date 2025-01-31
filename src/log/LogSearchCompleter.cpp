#include "LogSearchCompleter.hpp"

#include <QStringListModel>

LogSearchCompleter::LogSearchCompleter(QObject *parent) : QCompleter(parent) {
  setModelSorting(QCompleter::CaseInsensitivelySortedModel);
  setFilterMode(Qt::MatchContains);
  setCaseSensitivity(Qt::CaseInsensitive);
}

void LogSearchCompleter::updateSuggestions(const QStringList &messages) {
  auto *model = new QStringListModel(messages, this);
  setModel(model);
}