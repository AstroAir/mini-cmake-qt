#include "FolderWatcher.h"

#include <QRegularExpression>
#include <spdlog/sinks/basic_file_sink.h>

namespace {
// 创建一个只属于 FolderWatcher 的 logger
std::shared_ptr<spdlog::logger> createFolderWatcherLogger() {
  auto logger =
      spdlog::basic_logger_mt("folder_watcher_logger", "folder_watcher.log");
  logger->set_level(spdlog::level::info); // 设置日志级别
  logger->flush_on(spdlog::level::warn);  // 设置何时自动刷新
  return logger;
}

std::shared_ptr<spdlog::logger> getFolderWatcherLogger() {
  static std::shared_ptr<spdlog::logger> logger = createFolderWatcherLogger();
  return logger;
}
} // namespace

FolderMonitor::FolderMonitor(QObject *parent) : QObject(parent) {
  connect(&watcher, &QFileSystemWatcher::directoryChanged, this,
          &FolderMonitor::onDirectoryChanged);
  connect(&watcher, &QFileSystemWatcher::fileChanged, this,
          &FolderMonitor::onFileChanged);
}

FolderMonitor::FolderMonitor(const QString &path, bool recursive,
                             QObject *parent)
    : FolderMonitor(parent) {
  addFolder(path, recursive);
}

void FolderMonitor::addFolder(const QString &path, bool recursive) {
  if (!QDir(path).exists()) {
    getFolderWatcherLogger()->error("Directory不存在: {}", path.toStdString());
    throw std::runtime_error("Directory does not exist");
  }

  monitoredFolders[path] = recursive;
  if (recursive) {
    addRecursivePath(path);
  } else {
    watcher.addPath(path);
  }

  getFolderWatcherLogger()->info("开始监控目录: {}, 递归: {}",
                                 path.toStdString(), recursive);
  updateDirectoryContents(path);
}

void FolderMonitor::removeFolder(const QString &path) {
  if (!monitoredFolders.contains(path)) {
    return;
  }

  bool recursive = monitoredFolders[path];
  if (recursive) {
    removeRecursivePath(path);
  } else {
    watcher.removePath(path);
  }

  monitoredFolders.remove(path);
  getFolderWatcherLogger()->info("停止监控目录: {}", path.toStdString());
}

QStringList FolderMonitor::getMonitoredFolders() const {
  return monitoredFolders.keys();
}

void FolderMonitor::removeRecursivePath(const QString &path) {
  watcher.removePath(path);
  QDir dir(path);
  auto subDirs =
      dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
  for (const QFileInfo &dirInfo : subDirs) {
    removeRecursivePath(dirInfo.absoluteFilePath());
  }
}

void FolderMonitor::setCallback(Callback callback) {
  QMutexLocker locker(&callbackMutex);
  this->callback = std::move(callback);
}

void FolderMonitor::onDirectoryChanged(const QString &path) {
  if (monitoringPaused)
    return;

  getFolderWatcherLogger()->info("目录变化检测到: {}", path.toStdString());
  updateDirectoryContents(path);

  if (monitoredFolders[path]) {
    // 检查是否有新的子目录被创建，并添加到监控列表
    QDir dir(path);
    auto subDirs =
        dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
    for (const QFileInfo &dirInfo : subDirs) {
      if (!watcher.directories().contains(dirInfo.absoluteFilePath())) {
        addRecursivePath(dirInfo.absoluteFilePath());
        getFolderWatcherLogger()->info(
            "新增监控目录: {}", dirInfo.absoluteFilePath().toStdString());
        emit fileSystemEvent(
            {FileSystemEventType::FileCreated, dirInfo.absoluteFilePath()});
      }
    }
  }
}

void FolderMonitor::onFileChanged(const QString &path) {
  getFolderWatcherLogger()->info("文件变化检测到: {}", path.toStdString());
  emit fileSystemEvent({FileSystemEventType::FileChanged, path});
}

void FolderMonitor::addRecursivePath(const QString &path) {
  watcher.addPath(path);
  QDir dir(path);
  auto subDirs =
      dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
  for (const QFileInfo &dirInfo : subDirs) {
    addRecursivePath(dirInfo.absoluteFilePath());
  }
}

void FolderMonitor::setNameFilters(const QStringList &filters) {
  nameFilters = filters;
  getFolderWatcherLogger()->info("设置文件包含过滤器: {}",
                                 filters.join(", ").toStdString());
}

void FolderMonitor::setExcludeFilters(const QStringList &filters) {
  excludeFilters = filters;
  getFolderWatcherLogger()->info("设置文件排除过滤器: {}",
                                 filters.join(", ").toStdString());
}

void FolderMonitor::setPauseMonitoring(bool pause) {
  monitoringPaused = pause;
  getFolderWatcherLogger()->info("监控状态: {}", pause ? "暂停" : "运行");
}

bool FolderMonitor::shouldMonitorFile(const QString &filePath) const {
  if (monitoringPaused)
    return false;

  QFileInfo fileInfo(filePath);
  QString fileName = fileInfo.fileName();

  for (const QString &pattern : excludeFilters) {
    QRegularExpression rx(pattern, QRegularExpression::CaseInsensitiveOption);
    if (rx.match(fileName).hasMatch())
      return false;
  }

  if (nameFilters.isEmpty())
    return true;

  for (const QString &pattern : nameFilters) {
    QRegularExpression rx(pattern, QRegularExpression::CaseInsensitiveOption);
    if (rx.match(fileName).hasMatch())
      return true;
  }

  return false;
}

void FolderMonitor::updateDirectoryContents(const QString &path) {
  QDir dir(path);
  QStringList currentFiles = getFilteredFiles(dir);
  QStringList &previousFiles = directoryContents[path];

  // 检测新文件
  for (const QString &file : currentFiles) {
    if (!previousFiles.contains(file)) {
      emit fileSystemEvent({FileSystemEventType::FileCreated, file});
      getFolderWatcherLogger()->info("文件创建: {}", file.toStdString());
    }
  }

  // 检测删除的文件
  for (const QString &file : previousFiles) {
    if (!currentFiles.contains(file)) {
      emit fileSystemEvent({FileSystemEventType::FileDeleted, file});
      getFolderWatcherLogger()->info("文件删除: {}", file.toStdString());
    }
  }

  previousFiles = currentFiles;
}

QStringList FolderMonitor::getFilteredFiles(const QDir &dir) const {
  QStringList files;
  QFileInfoList entries =
      dir.entryInfoList(QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot,
                        QDir::DirsFirst | QDir::Name);

  for (const QFileInfo &info : entries) {
    QString path = info.absoluteFilePath();
    if (shouldMonitorFile(path)) {
      files.append(path);
    }
  }
  return files;
}