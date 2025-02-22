#ifndef FOLDERWATCHER_H
#define FOLDERWATCHER_H

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QFileSystemWatcher>
#include <QMutex>
#include <QObject>
#include <QThread>
#include <QWaitCondition>
#include <QDateTime>
#include <QTimer>
#include <QQueue>
#include <QHash>
#include <QSet>
#include <functional>
#include <spdlog/spdlog.h>

// 添加更多事件类型
enum class FileSystemEventType {
  FileChanged,
  DirectoryChanged,
  FileCreated,
  FileDeleted,
  FileRenamed,
  FileAttributeChanged,
  AccessError,
  WatcherError
};

// 添加错误代码
enum class WatcherErrorCode {
  None,
  PathNotFound,
  AccessDenied,
  WatcherLimitExceeded,
  SystemError
};

struct FileSystemEvent {
  FileSystemEventType type;
  QString path;
  QString oldPath; // for renames
  WatcherErrorCode errorCode{WatcherErrorCode::None};
  QString errorMessage;
  QDateTime timestamp{QDateTime::currentDateTime()};
};

class FolderMonitor : public QObject {
  Q_OBJECT
public:
  using Callback = std::function<void(const FileSystemEvent &)>;

  explicit FolderMonitor(QObject *parent = nullptr);
  explicit FolderMonitor(const QString &path, bool recursive = false,
                         QObject *parent = nullptr);

  void addFolder(const QString &path, bool recursive = false);
  void removeFolder(const QString &path);
  QStringList getMonitoredFolders() const;
  void setCallback(Callback callback);
  void setNameFilters(const QStringList &filters);
  void setExcludeFilters(const QStringList &filters);
  void setPauseMonitoring(bool pause);
  bool isMonitoring() const { return !monitoringPaused; }

  // 新增配置结构
  struct Config {
    int debounceMs{500};          // 事件防抖时间
    bool ignoreHiddenFiles{true}; // 是否忽略隐藏文件
    int maxWatchedFiles{1000};    // 最大监控文件数
    bool watchNewDirectories{true}; // 自动监控新创建的目录
    QStringList excludePaths;      // 排除的路径
  };

  void setConfig(const Config& config);
  bool isValidPath(const QString& path) const;
  void clearAllWatchers();
  QStringList getWatchedFiles() const;
  bool addWatchedFile(const QString& filePath);
  void removeWatchedFile(const QString& filePath);
  void checkWatcherStatus();

  // 回调函数优先级
  enum class CallbackPriority {
    Low = 0,
    Normal = 1,
    High = 2
  };

  // 回调函数信息结构
  struct CallbackInfo {
    Callback callback;
    CallbackPriority priority;
    bool enabled;
    QString description;
  };

  // 新增方法
  bool addCallback(const QString& path, Callback callback, 
                  CallbackPriority priority = CallbackPriority::Normal,
                  const QString& description = QString());
  bool removeCallback(const QString& path);
  void enableCallback(const QString& path, bool enable);
  void clearCallbacks();
  QStringList getCallbackPaths() const;
  bool hasCallback(const QString& path) const;
  void setGlobalCallback(Callback callback); // 设置全局回调

signals:
  void fileSystemEvent(const FileSystemEvent &event);
  void watcherError(const FileSystemEvent& event);
  void watcherStatusChanged(bool active);

private slots:
  void onDirectoryChanged(const QString &path);
  void onFileChanged(const QString &path);

private:
  void addRecursivePath(const QString &path);
  void removeRecursivePath(const QString &path);
  bool shouldMonitorFile(const QString &filePath) const;
  void updateDirectoryContents(const QString &path);
  QStringList getFilteredFiles(const QDir &dir) const;
  void handleError(WatcherErrorCode code, const QString& message);
  bool validatePath(const QString& path) const;
  void processPendingEvents();
  void scheduleEventProcessing();
  void invokeCallbacks(const FileSystemEvent& event);
  QString findMatchingCallbackPath(const QString& filePath) const;
  void cleanupCallbacks();

private:
  QFileSystemWatcher watcher;
  QMap<QString, bool> monitoredFolders; // 路径到递归标志的映射
  Callback callback;
  QMutex callbackMutex;
  QStringList nameFilters;      // 包含过滤器（如 *.cpp, *.h）
  QStringList excludeFilters;   // 排除过滤器
  QMap<QString, QStringList> directoryContents; // 缓存目录内容
  bool monitoringPaused{false};
  Config config;
  QTimer* debounceTimer{nullptr};
  QQueue<FileSystemEvent> eventQueue;
  QHash<QString, QDateTime> lastEventTimes;
  bool watcherActive{true};
  QMutex eventQueueMutex;
  int watchedFileCount{0};
  QSet<QString> invalidPaths;
  QMap<QString, CallbackInfo> pathCallbacks; // 路径到回调函数的映射
  Callback globalCallback; // 全局回调函数
  QMutex callbackMapMutex;
};

#endif // FOLDERWATCHER_H