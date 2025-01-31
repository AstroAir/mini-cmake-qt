#pragma once

#include <QtConcurrent>
#include <QtWidgets>
#include <QtCharts>
#include <array>
#include <memory>
#include <ranges>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <zlib.h>

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

// 添加新的图表模型类
class LogChartModel : public QObject {
  Q_OBJECT
public:
  explicit LogChartModel(QObject *parent = nullptr);
  void updateChart(const QVector<QPair<QDateTime, spdlog::level::level_enum>>& logs);
  QChart* chart() const { return m_chart; }

private:
  QChart *m_chart;
  QLineSeries *m_errorSeries;
  QLineSeries *m_warningSeries;
  QLineSeries *m_infoSeries;
};

// 添加新的日志分析器类
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
    
    Statistics analyze(const QVector<QPair<QDateTime, QString>>& logs);
    QChart* createDistributionChart(const Statistics& stats);
};

// 添加搜索建议器
class LogSearchCompleter : public QCompleter {
    Q_OBJECT
public:
    explicit LogSearchCompleter(QObject *parent = nullptr);
    void updateSuggestions(const QStringList& messages);
};

class LogTableModel : public QAbstractTableModel {
  Q_OBJECT
public:
  enum Columns { Time, Level, Message, ColumnCount };

  QVariant headerData(int section, Qt::Orientation orientation,
                      int role) const override;
};

// 添加设置对话框类
class LogSettingsDialog : public QDialog {
    Q_OBJECT
public:
    explicit LogSettingsDialog(QWidget* parent = nullptr);
    
    struct Settings {
        int maxLogEntries;
        QString logPath;
        bool enableFileLogging;
        QFont logFont;
        QMap<spdlog::level::level_enum, QColor> levelColors;
    };
    
    Settings getSettings() const;
    void setSettings(const Settings& settings);

private:
    void setupUI();
    QSpinBox* m_maxEntriesBox;
    QLineEdit* m_logPathEdit;
    QCheckBox* m_enableFileLogging;
    QFontComboBox* m_fontComboBox;
    QMap<spdlog::level::level_enum, QColorDialog*> m_colorDialogs;
};

class EnhancedLogPanel : public QWidget {
  Q_OBJECT
public:
  explicit EnhancedLogPanel(QWidget *parent = nullptr);

  [[nodiscard]] std::shared_ptr<spdlog::logger> logger() const noexcept;

signals:
  void exportFinished(bool success);
  void exportRequested(const QString &path);
  void exportFailed(const QString &error);

public slots:
  [[nodiscard]] bool exportLogs(const QString &path,
                                const QDateTime &start = {},
                                const QDateTime &end = {}) noexcept;
  void startBackgroundExport(const QString &path);

private slots:
  void updateLevelFilter(int index);

private:
  void setupUI();
  void setupConnections();
  void setupContextMenu();
  void updateStatistics();
  void toggleTheme();
  void toggleAutoScroll();
  QVector<QString> gatherLogs(const QDateTime &start, const QDateTime &end);
  bool compressAndSave(const QVector<QString> &logs, const QString &path);
  void handleExportError();
  void updateSearchResults();
  bool isInTimeRange(const QModelIndex &idx, const QDateTime &start,
                     const QDateTime &end);
  void setupCharts();
  void setupToolbar();
  void setupStatusBar();
  void exportToFormat(const QString& format);
  void applyModernStyle();
  void updateHighlight();
  void showLogDetails(const QModelIndex& index);
  // 新增方法
  void setupSearchWidget();
  QWidget* setupAnalysisPanel();
  void setupShortcuts();
  void updateLayout();
  void saveState();
  void restoreState();
  void showAnalysisResults();
  void handleSearchShortcut();
  void setupDockWidgets();
  void updateSearchSuggestions();
  // 添加缺失的方法
  void findNext();
  void findPrevious();
  QWidget* setupLogView();
  void updateAnalysisPanel(const LogAnalyzer::Statistics& stats, QChart* chart);
  QVector<QPair<QDateTime, QString>> getCurrentLogs() const;
  void showSettingsDialog();
  void applySettings(const LogSettingsDialog::Settings& settings);
  void exportToHTML(const QString& path);
  void exportToCSV(const QString& path);
  void exportToTXT(const QString& path);

  QTableView *m_logTable;
  QDateTimeEdit *m_startDate;
  QDateTimeEdit *m_endDate;
  QPushButton *m_themeBtn;
  QPushButton *m_pauseBtn;
  QDockWidget *m_statsDock;
  QLabel *m_statsLabel;
  bool m_autoScroll = true;
  QLineEdit *m_searchBox;
  QComboBox *m_levelCombo;
  QPushButton *m_exportBtn;
  QListView *m_logView;
  LogFilterModel m_filterModel;
  std::shared_ptr<spdlog::logger> m_logger;
  QChartView *m_chartView;
  LogChartModel *m_chartModel;
  QButtonGroup *m_levelFilterGroup;
  QStatusBar *m_statusBar;
  QToolBar *m_filterToolbar;
  QSplitter *m_mainSplitter;
  QTextEdit *m_detailsView;
  QPushButton *m_clearBtn;
  QPushButton *m_settingsBtn;
  QMenu *m_exportMenu;
  QWidget *m_filterWidget;
  bool m_isDarkTheme = false;
  // 新增成员变量
  LogAnalyzer* m_analyzer;
  LogSearchCompleter* m_searchCompleter;
  QDockWidget* m_analysisDock;
  QTabWidget* m_analysisTabWidget;
  QMap<QString, QDockWidget*> m_dockWidgets;
  QSplitter* m_verticalSplitter;
  QStackedWidget* m_contentStack;
  QShortcut* m_findShortcut;
  QSettings m_settings;
  bool m_isAnalysisPanelVisible = false;
  // 添加缺失的成员变量
  LogSettingsDialog::Settings m_currentSettings;
  int m_currentSearchIndex = -1;
  QVector<QModelIndex> m_searchResults;
  QTimer* m_searchDebounceTimer;
};

class LogItemDelegate : public QStyledItemDelegate {
  Q_OBJECT
public:
  explicit LogItemDelegate(QObject *parent = nullptr);

  void paint(QPainter *painter, const QStyleOptionViewItem &option,
             const QModelIndex &index) const override;

  void updateColors(const QMap<spdlog::level::level_enum, QColor>& colors);

private:
  QColor colorForLevel(spdlog::level::level_enum level) const;
  QMap<spdlog::level::level_enum, QColor> m_levelColors;
};