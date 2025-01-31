#include "LogPanel.hpp"
#include <ranges>

// LogTableModel implementation
QVariant LogTableModel::headerData(int section, Qt::Orientation orientation,
                                   int role) const {
  if (role == Qt::DisplayRole && orientation == Qt::Horizontal) {
    switch (section) {
    case Time:
      return tr("Time");
    case Level:
      return tr("Level");
    case Message:
      return tr("Message");
    }
  }
  return QVariant();
}

// EnhancedLogPanel implementation
EnhancedLogPanel::EnhancedLogPanel(QWidget *parent) : QWidget(parent) {
  setupUI();
  setupConnections();
  m_logger = spdlog::basic_logger_mt("console", "logs/console.log");
}

std::shared_ptr<spdlog::logger> EnhancedLogPanel::logger() const noexcept {
  return m_logger;
}

bool EnhancedLogPanel::exportLogs(const QString &path, const QDateTime &start,
                                  const QDateTime &end) noexcept {
  try {
    auto logs = gatherLogs(start, end);
    return compressAndSave(logs, path);
  } catch (...) {
    handleExportError();
    return false;
  }
}

void EnhancedLogPanel::startBackgroundExport(const QString &path) {
  QFutureWatcher<bool> *watcher = new QFutureWatcher<bool>(this);
  connect(watcher, &QFutureWatcher<bool>::finished, this, [this, watcher]() {
    if (watcher->result()) {
      emit exportFinished(true);
    } else {
      emit exportFailed("Export failed");
    }
    watcher->deleteLater();
  });

  watcher->setFuture(
      QtConcurrent::run([this, path]() { return this->exportLogs(path); }));
}

void EnhancedLogPanel::updateLevelFilter(int index) {
  Q_UNUSED(index);
  m_filterModel.filterChanged();
}

void EnhancedLogPanel::setupUI() {
  auto *mainLayout = new QVBoxLayout(this);
  mainLayout->setContentsMargins(3, 3, 3, 3);
  mainLayout->setSpacing(2);

  // 设置搜索工具栏
  setupSearchWidget();

  // 主分割器
  m_verticalSplitter = new QSplitter(Qt::Vertical, this);
  m_verticalSplitter->setChildrenCollapsible(false);

  // 内容区域
  m_contentStack = new QStackedWidget(this);
  auto *logView = setupLogView();
  auto *analysisView = setupAnalysisPanel();
  m_contentStack->addWidget(logView);
  m_contentStack->addWidget(analysisView);

  // 设置停靠窗口
  setupDockWidgets();

  mainLayout->addWidget(m_filterToolbar);
  mainLayout->addWidget(m_verticalSplitter);

  // 恢复上次布局状态
  restoreState();

  // 设置快捷键
  setupShortcuts();
}

void EnhancedLogPanel::setupSearchWidget() {
  m_searchBox->setPlaceholderText(tr("搜索日志 (Ctrl+F, 支持正则表达式)"));
  m_searchCompleter = new LogSearchCompleter(this);
  m_searchBox->setCompleter(m_searchCompleter);

  auto *searchWidget = new QWidget(this);
  auto *searchLayout = new QHBoxLayout(searchWidget);
  searchLayout->setContentsMargins(0, 0, 0, 0);

  auto *prevBtn = new QToolButton(this);
  auto *nextBtn = new QToolButton(this);
  prevBtn->setIcon(QIcon::fromTheme("go-up"));
  nextBtn->setIcon(QIcon::fromTheme("go-down"));

  searchLayout->addWidget(m_searchBox);
  searchLayout->addWidget(prevBtn);
  searchLayout->addWidget(nextBtn);

  connect(prevBtn, &QToolButton::clicked, this, [this] {
    // 查找上一个匹配项
    findPrevious();
  });

  connect(nextBtn, &QToolButton::clicked, this, [this] {
    // 查找下一个匹配项
    findNext();
  });
}

QWidget *EnhancedLogPanel::setupAnalysisPanel() {
  auto *analysisWidget = new QWidget(this);
  auto *layout = new QVBoxLayout(analysisWidget);

  m_analysisDock = new QDockWidget(tr("日志分析"), this);
  m_analysisTabWidget = new QTabWidget(m_analysisDock);

  // 统计信息面板
  auto *statsWidget = new QWidget;
  auto *statsLayout = new QVBoxLayout(statsWidget);
  auto *statsChart = m_analyzer->createDistributionChart({});
  statsLayout->addWidget(new QChartView(statsChart));
  m_analysisTabWidget->addTab(statsWidget, tr("统计信息"));

  // 趋势分析面板
  auto *trendWidget = new QWidget;
  auto *trendLayout = new QVBoxLayout(trendWidget);
  auto *trendChart = new QChart;
  trendLayout->addWidget(new QChartView(trendChart));
  m_analysisTabWidget->addTab(trendWidget, tr("趋势分析"));

  m_analysisDock->setWidget(m_analysisTabWidget);
  m_analysisDock->setVisible(false);

  layout->addWidget(m_analysisDock);
  return analysisWidget;
}

void EnhancedLogPanel::setupShortcuts() {
  m_findShortcut = new QShortcut(QKeySequence::Find, this);
  connect(m_findShortcut, &QShortcut::activated, this,
          &EnhancedLogPanel::handleSearchShortcut);

  // 添加其他快捷键
  new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_E), this,
                [this] { m_exportBtn->click(); });

  new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_R), this,
                [this] { showAnalysisResults(); });
}

void EnhancedLogPanel::saveState() {
  m_settings.setValue("geometry", saveGeometry());
  m_settings.setValue("splitterSizes", m_verticalSplitter->saveState());
  m_settings.setValue("autoScroll", m_autoScroll);
  m_settings.setValue("isDarkTheme", m_isDarkTheme);
}

void EnhancedLogPanel::restoreState() {
  restoreGeometry(m_settings.value("geometry").toByteArray());
  m_verticalSplitter->restoreState(
      m_settings.value("splitterSizes").toByteArray());
  m_autoScroll = m_settings.value("autoScroll", true).toBool();
  m_isDarkTheme = m_settings.value("isDarkTheme", false).toBool();
  if (m_isDarkTheme) {
    toggleTheme();
  }
}

void EnhancedLogPanel::showAnalysisResults() {
  if (!m_isAnalysisPanelVisible) {
    auto stats = m_analyzer->analyze(getCurrentLogs());
    auto chart = m_analyzer->createDistributionChart(stats);
    // 更新分析面板内容
    updateAnalysisPanel(stats, chart);
    m_analysisDock->show();
    m_isAnalysisPanelVisible = true;
  } else {
    m_analysisDock->hide();
    m_isAnalysisPanelVisible = false;
  }
}

void EnhancedLogPanel::updateSearchSuggestions() {
  QStringList suggestions;
  const int rowCount = m_filterModel.rowCount();
  for (int i = 0; i < rowCount; ++i) {
    auto text =
        m_filterModel.data(m_filterModel.index(i, LogTableModel::Message))
            .toString();
    if (!suggestions.contains(text)) {
      suggestions.append(text);
    }
  }
  m_searchCompleter->updateSuggestions(suggestions);
}

void EnhancedLogPanel::setupToolbar() {
  m_filterToolbar = new QToolBar(this);
  m_searchBox = new QLineEdit(this);
  m_searchBox->setClearButtonEnabled(true);
  m_searchBox->setPlaceholderText(tr("搜索日志 (支持正则表达式)"));

  m_exportBtn = new QPushButton(tr("导出"), this);
  m_exportMenu = new QMenu(this);
  m_exportMenu->addAction(tr("导出为 TXT"), this,
                          [this] { exportToFormat("txt"); });
  m_exportMenu->addAction(tr("导出为 CSV"), this,
                          [this] { exportToFormat("csv"); });
  m_exportMenu->addAction(tr("导出为 HTML"), this,
                          [this] { exportToFormat("html"); });
  m_exportBtn->setMenu(m_exportMenu);

  m_clearBtn = new QPushButton(tr("清除"), this);
  m_settingsBtn = new QPushButton(tr("设置"), this);

  m_filterToolbar->addWidget(m_searchBox);
  m_filterToolbar->addWidget(m_exportBtn);
  m_filterToolbar->addWidget(m_clearBtn);
  m_filterToolbar->addWidget(m_themeBtn);
  m_filterToolbar->addWidget(m_settingsBtn);
}

void EnhancedLogPanel::setupCharts() {
  m_chartModel = new LogChartModel(this);
  m_chartView = new QChartView(m_chartModel->chart(), this);
  m_chartView->setRenderHint(QPainter::Antialiasing);
  m_chartView->setMinimumHeight(150);
}

void EnhancedLogPanel::applyModernStyle() {
  QString style = R"(
        QWidget {
            font-family: "Segoe UI", "Microsoft YaHei";
        }
        QPushButton {
            padding: 5px 15px;
            border-radius: 4px;
            background-color: #f0f0f0;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
        }
        QLineEdit {
            padding: 5px;
            border-radius: 4px;
            border: 1px solid #ccc;
        }
        QTableView {
            border: none;
            gridline-color: #ddd;
        }
    )";
  setStyleSheet(style);
}

void EnhancedLogPanel::toggleTheme() {
  m_isDarkTheme = !m_isDarkTheme;
  if (m_isDarkTheme) {
    // 设置深色主题
    QPalette darkPalette;
    darkPalette.setColor(QPalette::Window, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::WindowText, Qt::white);
    darkPalette.setColor(QPalette::Base, QColor(25, 25, 25));
    darkPalette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::Text, Qt::white);
    darkPalette.setColor(QPalette::Button, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::ButtonText, Qt::white);
    QApplication::setPalette(darkPalette);
  } else {
    // 恢复亮色主题
    QApplication::setPalette(QApplication::style()->standardPalette());
  }
  m_themeBtn->setText(m_isDarkTheme ? "🌞" : "🌛");
}

void EnhancedLogPanel::showLogDetails(const QModelIndex &index) {
  if (!index.isValid())
    return;

  QString details = QString(R"(
        <h3>日志详情</h3>
        <p><b>时间:</b> %1</p>
        <p><b>级别:</b> %2</p>
        <p><b>消息:</b></p>
        <pre>%3</pre>
    )")
                        .arg(index.siblingAtColumn(0).data().toString())
                        .arg(index.siblingAtColumn(1).data().toString())
                        .arg(index.siblingAtColumn(2).data().toString());

  m_detailsView->setHtml(details);
}

void EnhancedLogPanel::setupConnections() {
  connect(m_searchBox, &QLineEdit::textChanged, &m_filterModel,
          &LogFilterModel::setSearchText);
  connect(m_levelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &EnhancedLogPanel::updateLevelFilter);
  connect(m_exportBtn, &QPushButton::clicked, this, [this]() {
    QString path = QFileDialog::getSaveFileName(this, tr("Export Logs"), "",
                                                tr("Gzip Files (*.gz)"));
    if (!path.isEmpty()) {
      startBackgroundExport(path);
    }
  });
  setupContextMenu();
}

void EnhancedLogPanel::setupContextMenu() {
  connect(m_logTable, &QTableView::customContextMenuRequested, this,
          [this](const QPoint &pos) {
            QMenu menu(this);
            menu.addAction(tr("Copy"), [this]() {
              auto text = m_logTable->currentIndex().data().toString();
              QApplication::clipboard()->setText(text);
            });
            menu.addAction(tr("Copy All"), [this]() {
              // 复制所有可见日志
            });
            menu.addSeparator();
            menu.addAction(tr("Clear"), [this]() {
              // 清除日志
            });
            menu.exec(m_logTable->mapToGlobal(pos));
          });
}

void EnhancedLogPanel::updateStatistics() {
  int total = m_filterModel.rowCount();
  int errors = 0;
  int warnings = 0;
  m_statsLabel->setText(tr("Total: %1\nErrors: %2\nWarnings: %3")
                            .arg(total)
                            .arg(errors)
                            .arg(warnings));
}

void EnhancedLogPanel::toggleAutoScroll() {
  m_autoScroll = !m_autoScroll;
  m_pauseBtn->setText(m_autoScroll ? tr("⏸") : tr("▶"));
}

QVector<QString> EnhancedLogPanel::gatherLogs(const QDateTime &start,
                                              const QDateTime &end) {
  QVector<QString> logs;
  const int rowCount = m_filterModel.rowCount();

  auto indices = std::ranges::views::iota(0, rowCount) |
                 std::ranges::views::transform(
                     [this](int row) { return m_filterModel.index(row, 0); });

  std::ranges::for_each(indices, [&](const QModelIndex &idx) {
    if (isInTimeRange(idx, start, end)) {
      logs.append(idx.data().toString());
    }
  });

  return logs;
}

bool EnhancedLogPanel::compressAndSave(const QVector<QString> &logs,
                                       const QString &path) {
  gzFile file = gzopen(path.toUtf8().constData(), "wb9");
  if (!file)
    return false;

  struct GzFileCloser {
    void operator()(gzFile f) { gzclose(f); }
  };
  std::unique_ptr<void, GzFileCloser> closer(file);

  try {
    for (const auto &log : logs) {
      if (gzwrite(file, log.toUtf8().constData(), log.toUtf8().size()) == 0) {
        throw std::runtime_error("Gzip write failed");
      }
    }
    emit exportFinished(true);
    return true;
  } catch (...) {
    emit exportFinished(false);
    return false;
  }
}

void EnhancedLogPanel::handleExportError() {
  emit exportFailed("An error occurred during export.");
}

bool EnhancedLogPanel::isInTimeRange(const QModelIndex &idx,
                                     const QDateTime &start,
                                     const QDateTime &end) {
  QDateTime timestamp = idx.data(Qt::UserRole + 1).toDateTime();
  if (start.isValid() && timestamp < start)
    return false;
  if (end.isValid() && timestamp > end)
    return false;
  return true;
}

// LogItemDelegate implementation
LogItemDelegate::LogItemDelegate(QObject *parent)
    : QStyledItemDelegate(parent) {}

void LogItemDelegate::paint(QPainter *painter,
                            const QStyleOptionViewItem &option,
                            const QModelIndex &index) const {
  auto level =
      static_cast<spdlog::level::level_enum>(index.data(Qt::UserRole).toInt());

  QStyleOptionViewItem opt = option;
  initStyleOption(&opt, index);

  opt.palette.setColor(QPalette::Text, colorForLevel(level));
  if (level >= spdlog::level::err) {
    opt.font.setBold(true);
  }

  QStyledItemDelegate::paint(painter, opt, index);
}

QColor LogItemDelegate::colorForLevel(spdlog::level::level_enum level) const {
  if (m_levelColors.contains(level)) {
    return m_levelColors[level];
  }
  // 如果没有自定义颜色,使用默认颜色
  switch (level) {
  case spdlog::level::trace:
    return Qt::gray;
  case spdlog::level::debug:
    return Qt::blue;
  case spdlog::level::info:
    return Qt::black;
  case spdlog::level::warn:
    return Qt::darkYellow;
  case spdlog::level::err:
    return Qt::red;
  case spdlog::level::critical:
    return Qt::magenta;
  default:
    return Qt::black;
  }
}

void LogItemDelegate::updateColors(
    const QMap<spdlog::level::level_enum, QColor> &colors) {
  m_levelColors = colors;
}

QWidget *EnhancedLogPanel::setupLogView() {
  auto *container = new QWidget(this);
  auto *layout = new QVBoxLayout(container);

  m_logTable->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
  m_logTable->setSortingEnabled(true);
  m_logTable->setShowGrid(false);
  m_logTable->verticalHeader()->setVisible(false);
  m_logTable->horizontalHeader()->setStretchLastSection(true);
  m_logTable->setSelectionBehavior(QAbstractItemView::SelectRows);

  connect(m_logTable->selectionModel(), &QItemSelectionModel::currentChanged,
          this, &EnhancedLogPanel::showLogDetails);

  layout->addWidget(m_logTable);
  return container;
}

void EnhancedLogPanel::findNext() {
  if (m_searchResults.isEmpty()) {
    updateSearchResults();
  }
  if (m_searchResults.isEmpty())
    return;

  m_currentSearchIndex = (m_currentSearchIndex + 1) % m_searchResults.size();
  auto idx = m_searchResults[m_currentSearchIndex];
  m_logTable->setCurrentIndex(idx);
  m_logTable->scrollTo(idx, QAbstractItemView::PositionAtCenter);
}

void EnhancedLogPanel::findPrevious() {
  if (m_searchResults.isEmpty()) {
    updateSearchResults();
  }
  if (m_searchResults.isEmpty())
    return;

  m_currentSearchIndex = (m_currentSearchIndex - 1 + m_searchResults.size()) %
                         m_searchResults.size();
  auto idx = m_searchResults[m_currentSearchIndex];
  m_logTable->setCurrentIndex(idx);
  m_logTable->scrollTo(idx, QAbstractItemView::PositionAtCenter);
}

void EnhancedLogPanel::updateSearchResults() {
  // 清除旧的搜索结果
  m_searchResults.clear();
  m_currentSearchIndex = -1;

  QString searchText = m_searchBox->text().trimmed();
  if (searchText.isEmpty()) {
    return;
  }

  QRegularExpression regex;
  // 检查是否是正则表达式
  if (searchText.startsWith("/") && searchText.endsWith("/")) {
    // 移除前后的斜杠
    searchText = searchText.mid(1, searchText.length() - 2);
    regex.setPattern(searchText);
  } else {
    // 将普通文本转换为正则表达式安全的模式
    regex.setPattern(QRegularExpression::escape(searchText));
  }

  regex.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
  if (!regex.isValid()) {
    m_statusBar->showMessage(tr("无效的搜索表达式"), 3000);
    return;
  }

  // 在过滤后的模型中搜索
  for (int row = 0; row < m_filterModel.rowCount(); ++row) {
    QModelIndex messageIdx = m_filterModel.index(row, LogTableModel::Message);
    QString text = messageIdx.data().toString();

    if (regex.match(text).hasMatch()) {
      m_searchResults.append(messageIdx);
    }
  }

  // 更新状态栏显示搜索结果数量
  if (!m_searchResults.isEmpty()) {
    m_statusBar->showMessage(
        tr("找到 %1 个匹配项").arg(m_searchResults.size()));
    // 高亮第一个结果
    m_currentSearchIndex = 0;
    auto idx = m_searchResults.first();
    m_logTable->setCurrentIndex(idx);
    m_logTable->scrollTo(idx, QAbstractItemView::PositionAtCenter);
  } else {
    m_statusBar->showMessage(tr("未找到匹配项"), 3000);
  }
}

void EnhancedLogPanel::exportToFormat(const QString &format) {
  QString path = QFileDialog::getSaveFileName(
      this, tr("导出"), QString(),
      format.toUpper() + tr(" 文件 (*.") + format + ")");

  if (path.isEmpty())
    return;

  if (format == "html") {
    exportToHTML(path);
  } else if (format == "csv") {
    exportToCSV(path);
  } else if (format == "txt") {
    exportToTXT(path);
  }
}

void EnhancedLogPanel::exportToHTML(const QString &path) {
  QFile file(path);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
    emit exportFailed(tr("无法打开文件进行写入"));
    return;
  }

  QTextStream out(&file);
  // 写入HTML头部
  out << R"(<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        table { border-collapse: collapse; width: 100%; }
        th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
        tr:nth-child(even) { background-color: #f2f2f2; }
    </style>
</head>
<body>
<table>
<tr><th>时间</th><th>级别</th><th>消息</th></tr>)";

  // 写入日志内容
  auto logs = getCurrentLogs();
  for (const auto &[time, message] : logs) {
    out << QString("<tr><td>%1</td><td>%2</td><td>%3</td></tr>\n")
               .arg(time.toString("yyyy-MM-dd HH:mm:ss.zzz"))
               .arg(message.split("|").value(0))
               .arg(message.split("|").value(1));
  }

  out << "</table></body></html>";
  file.close();
  emit exportFinished(true);
}

void EnhancedLogPanel::exportToCSV(const QString &path) {
  QFile file(path);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
    emit exportFailed(tr("无法打开文件进行写入"));
    return;
  }

  QTextStream out(&file);
  // CSV 头部
  out << "时间,级别,消息\n";

  // 导出日志内容
  auto logs = getCurrentLogs();
  for (const auto &[time, message] : logs) {
    QString escapedMessage = message;
    // 处理CSV中的特殊字符
    escapedMessage.replace("\"", "\"\"");
    if (escapedMessage.contains(",") || escapedMessage.contains("\"") ||
        escapedMessage.contains("\n")) {
      escapedMessage = "\"" + escapedMessage + "\"";
    }

    out << QString("%1,%2,%3\n")
               .arg(time.toString("yyyy-MM-dd HH:mm:ss.zzz"))
               .arg(message.split("|").value(0))
               .arg(escapedMessage);
  }

  file.close();
  emit exportFinished(true);
}

void EnhancedLogPanel::exportToTXT(const QString &path) {
  QFile file(path);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
    emit exportFailed(tr("无法打开文件进行写入"));
    return;
  }

  QTextStream out(&file);
  auto logs = getCurrentLogs();
  for (const auto &[time, message] : logs) {
    out << QString("[%1] [%2] %3\n")
               .arg(time.toString("yyyy-MM-dd HH:mm:ss.zzz"))
               .arg(message.split("|").value(0))
               .arg(message.split("|").value(1));
  }

  file.close();
  emit exportFinished(true);
}

QVector<QPair<QDateTime, QString>> EnhancedLogPanel::getCurrentLogs() const {
  QVector<QPair<QDateTime, QString>> logs;
  const int rowCount = m_filterModel.rowCount();

  for (int i = 0; i < rowCount; ++i) {
    QModelIndex timeIdx = m_filterModel.index(i, LogTableModel::Time);
    QModelIndex levelIdx = m_filterModel.index(i, LogTableModel::Level);
    QModelIndex messageIdx = m_filterModel.index(i, LogTableModel::Message);

    QDateTime time = timeIdx.data(Qt::UserRole).toDateTime();
    QString message = QString("%1|%2")
                          .arg(levelIdx.data().toString())
                          .arg(messageIdx.data().toString());

    logs.append({time, message});
  }

  return logs;
}

void EnhancedLogPanel::handleSearchShortcut() {
  m_searchBox->setFocus();
  m_searchBox->selectAll();
}

void EnhancedLogPanel::setupDockWidgets() {
  // 创建统计信息停靠窗口
  m_statsDock = new QDockWidget(tr("统计信息"), this);
  m_statsDock->setAllowedAreas(Qt::LeftDockWidgetArea |
                               Qt::RightDockWidgetArea);
  m_statsLabel = new QLabel(m_statsDock);
  m_statsDock->setWidget(m_statsLabel);
  m_dockWidgets["stats"] = m_statsDock;

  // 创建图表停靠窗口
  auto chartDock = new QDockWidget(tr("趋势图"), this);
  chartDock->setAllowedAreas(Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
  chartDock->setWidget(m_chartView);
  m_dockWidgets["chart"] = chartDock;

  // 创建详情停靠窗口
  auto detailsDock = new QDockWidget(tr("日志详情"), this);
  detailsDock->setWidget(m_detailsView);
  m_dockWidgets["details"] = detailsDock;
}

void EnhancedLogPanel::showSettingsDialog() {
  LogSettingsDialog dialog(this);
  dialog.setSettings(m_currentSettings);

  if (dialog.exec() == QDialog::Accepted) {
    auto newSettings = dialog.getSettings();
    applySettings(newSettings);
  }
}

void EnhancedLogPanel::applySettings(
    const LogSettingsDialog::Settings &settings) {
  m_currentSettings = settings;

  // 更新日志字体
  m_logTable->setFont(settings.logFont);
  m_detailsView->setFont(settings.logFont);

  // 更新颜色方案
  auto *delegate = qobject_cast<LogItemDelegate *>(m_logTable->itemDelegate());
  if (delegate) {
    delegate->updateColors(settings.levelColors);
  }

  // 更新日志文件设置
  if (settings.enableFileLogging) {
    m_logger =
        spdlog::basic_logger_mt("console", settings.logPath.toStdString());
  }

  // 保存设置
  m_settings.setValue("settings", QVariant::fromValue(settings));

  // 刷新显示
  m_logTable->viewport()->update();
}



// LogSettingsDialog implementation
LogSettingsDialog::LogSettingsDialog(QWidget *parent) : QDialog(parent) {
  setupUI();
}

void LogSettingsDialog::setupUI() {
  auto layout = new QVBoxLayout(this);

  // 最大日志条目设置
  auto entriesLayout = new QHBoxLayout();
  entriesLayout->addWidget(new QLabel(tr("最大日志条目数:")));
  m_maxEntriesBox = new QSpinBox(this);
  m_maxEntriesBox->setRange(100, 1000000);
  m_maxEntriesBox->setValue(10000);
  entriesLayout->addWidget(m_maxEntriesBox);
  layout->addLayout(entriesLayout);

  // 日志文件路径设置
  auto pathLayout = new QHBoxLayout();
  pathLayout->addWidget(new QLabel(tr("日志文件路径:")));
  m_logPathEdit = new QLineEdit(this);
  pathLayout->addWidget(m_logPathEdit);
  auto browseBtn = new QPushButton(tr("浏览"), this);
  connect(browseBtn, &QPushButton::clicked, this, [this]() {
    QString path = QFileDialog::getSaveFileName(this, tr("选择日志文件"), "",
                                                tr("日志文件 (*.log)"));
    if (!path.isEmpty()) {
      m_logPathEdit->setText(path);
    }
  });
  pathLayout->addWidget(browseBtn);
  layout->addLayout(pathLayout);

  // 启用文件日志
  m_enableFileLogging = new QCheckBox(tr("启用文件日志"), this);
  layout->addWidget(m_enableFileLogging);

  // 字体设置
  auto fontLayout = new QHBoxLayout();
  fontLayout->addWidget(new QLabel(tr("日志字体:")));
  m_fontComboBox = new QFontComboBox(this);
  fontLayout->addWidget(m_fontComboBox);
  layout->addLayout(fontLayout);

  // 颜色设置
  auto colorGroup = new QGroupBox(tr("日志级别颜色"), this);
  auto colorLayout = new QGridLayout(colorGroup);
  int row = 0;
  for (int level = spdlog::level::trace; level <= spdlog::level::critical;
       ++level) {
    auto levelEnum = static_cast<spdlog::level::level_enum>(level);
    auto colorBtn = new QPushButton(this);
    colorBtn->setFixedSize(40, 25);
    m_colorDialogs[levelEnum] = new QColorDialog(this);
    connect(colorBtn, &QPushButton::clicked, this,
            [this, levelEnum]() { m_colorDialogs[levelEnum]->show(); });
    colorLayout->addWidget(
        new QLabel(QString::fromStdString(
            spdlog::level::to_string_view(levelEnum).data())),
        row, 0);
    colorLayout->addWidget(colorBtn, row, 1);
    row++;
  }
  layout->addWidget(colorGroup);

  // 确定取消按钮
  auto buttonBox = new QDialogButtonBox(
      QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
  connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
  layout->addWidget(buttonBox);
}

LogSettingsDialog::Settings LogSettingsDialog::getSettings() const {
  Settings settings;
  settings.maxLogEntries = m_maxEntriesBox->value();
  settings.logPath = m_logPathEdit->text();
  settings.enableFileLogging = m_enableFileLogging->isChecked();
  settings.logFont = m_fontComboBox->currentFont();

  for (auto it = m_colorDialogs.begin(); it != m_colorDialogs.end(); ++it) {
    settings.levelColors[it.key()] = it.value()->currentColor();
  }

  return settings;
}

void LogSettingsDialog::setSettings(const Settings &settings) {
  m_maxEntriesBox->setValue(settings.maxLogEntries);
  m_logPathEdit->setText(settings.logPath);
  m_enableFileLogging->setChecked(settings.enableFileLogging);
  m_fontComboBox->setCurrentFont(settings.logFont);

  for (auto it = settings.levelColors.begin(); it != settings.levelColors.end();
       ++it) {
    if (m_colorDialogs.contains(it.key())) {
      m_colorDialogs[it.key()]->setCurrentColor(it.value());
    }
  }
}
