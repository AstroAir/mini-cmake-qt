#pragma once

#include <QCompleter>

class LogSearchCompleter : public QCompleter {
  Q_OBJECT
public:
  explicit LogSearchCompleter(QObject *parent = nullptr);
  void updateSuggestions(const QStringList &messages);
};