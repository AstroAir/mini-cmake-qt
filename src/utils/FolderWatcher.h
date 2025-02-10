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
#include <functional>
#include <spdlog/spdlog.h>


enum class FileSystemEventType {
  FileChanged,
  DirectoryChanged,
  FileCreated,
  FileDeleted,
  FileRenamed
};

struct FileSystemEvent {
  FileSystemEventType type;
  QString path;
  QString oldPath; // for renames
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

signals:
  void fileSystemEvent(const FileSystemEvent &event);

private slots:
  void onDirectoryChanged(const QString &path);
  void onFileChanged(const QString &path);

private:
  void addRecursivePath(const QString &path);
  void removeRecursivePath(const QString &path);
  bool shouldMonitorFile(const QString &filePath) const;
  void updateDirectoryContents(const QString &path);
  QStringList getFilteredFiles(const QDir &dir) const;

private:
  QFileSystemWatcher watcher;
  QMap<QString, bool> monitoredFolders; // 路径到递归标志的映射
  Callback callback;
  QMutex callbackMutex;
  QStringList nameFilters;      // 包含过滤器（如 *.cpp, *.h）
  QStringList excludeFilters;   // 排除过滤器
  QMap<QString, QStringList> directoryContents; // 缓存目录内容
  bool monitoringPaused{false};
};

#endif // FOLDERWATCHER_H