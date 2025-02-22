#include "FolderWatcher.h"
#include <QTimer>
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

  debounceTimer = new QTimer(this);
  debounceTimer->setSingleShot(true);
  connect(debounceTimer, &QTimer::timeout, this, &FolderMonitor::processPendingEvents);
}

FolderMonitor::FolderMonitor(const QString &path, bool recursive,
                             QObject *parent)
    : FolderMonitor(parent) {
  addFolder(path, recursive);
}

void FolderMonitor::setConfig(const Config& config) {
  this->config = config;
  debounceTimer->setInterval(config.debounceMs);
  getFolderWatcherLogger()->info("更新配置: 防抖时间={}ms, 忽略隐藏文件={}", 
                                config.debounceMs, config.ignoreHiddenFiles);
}

void FolderMonitor::addFolder(const QString &path, bool recursive) {
  try {
    if (!validatePath(path)) {
      handleError(WatcherErrorCode::PathNotFound, 
                 QString("无效路径: %1").arg(path));
      return;
    }

    if (watchedFileCount >= config.maxWatchedFiles) {
      handleError(WatcherErrorCode::WatcherLimitExceeded, 
                 "已达到最大监控文件数限制");
      return;
    }

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

  } catch (const std::exception& e) {
    handleError(WatcherErrorCode::SystemError, 
               QString("添加文件夹失败: %1").arg(e.what()));
  }
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

void FolderMonitor::handleError(WatcherErrorCode code, const QString& message) {
  FileSystemEvent event;
  event.type = FileSystemEventType::WatcherError;
  event.errorCode = code;
  event.errorMessage = message;
  
  getFolderWatcherLogger()->error("监控错误: {}", message.toStdString());
  emit watcherError(event);
}

bool FolderMonitor::validatePath(const QString& path) const {
  if (invalidPaths.contains(path)) {
    return false;
  }

  QFileInfo fileInfo(path);
  if (!fileInfo.exists() || !fileInfo.isReadable()) {
    return false;
  }

  // 检查是否在排除列表中
  for (const QString& excludePath : config.excludePaths) {
    if (path.startsWith(excludePath)) {
      return false;
    }
  }

  return true;
}

void FolderMonitor::scheduleEventProcessing() {
  if (!debounceTimer->isActive()) {
    debounceTimer->start();
  }
}

void FolderMonitor::processPendingEvents() {
  QMutexLocker locker(&eventQueueMutex);
  while (!eventQueue.isEmpty()) {
    FileSystemEvent event = eventQueue.dequeue();
    
    if (config.ignoreHiddenFiles) {
      QFileInfo fileInfo(event.path);
      if (fileInfo.isHidden()) {
        continue;
      }
    }

    QString key = event.path + QString::number(static_cast<int>(event.type));
    QDateTime now = QDateTime::currentDateTime();
    if (lastEventTimes.contains(key)) {
      if (lastEventTimes[key].msecsTo(now) < config.debounceMs) {
        continue;
      }
    }
    lastEventTimes[key] = now;

    // 使用新的回调处理机制
    invokeCallbacks(event);
    emit fileSystemEvent(event);
  }
}

void FolderMonitor::checkWatcherStatus() {
  bool previousStatus = watcherActive;
  watcherActive = !watcher.directories().isEmpty() || !watcher.files().isEmpty();

  if (previousStatus != watcherActive) {
    getFolderWatcherLogger()->info("监控状态改变: {}", 
                                  watcherActive ? "活动" : "非活动");
    emit watcherStatusChanged(watcherActive);
  }
}

QStringList FolderMonitor::getWatchedFiles() const {
  QStringList files = watcher.files();
  QStringList dirs = watcher.directories();
  files.append(dirs);
  return files;
}

void FolderMonitor::clearAllWatchers() {
  watcher.removePaths(watcher.files());
  watcher.removePaths(watcher.directories());
  monitoredFolders.clear();
  directoryContents.clear();
  watchedFileCount = 0;
  getFolderWatcherLogger()->info("清除所有监控");
  checkWatcherStatus();
}

bool FolderMonitor::addCallback(const QString& path, Callback callback,
                               CallbackPriority priority,
                               const QString& description) {
  if (!validatePath(path)) {
    handleError(WatcherErrorCode::PathNotFound,
                QString("添加回调函数失败：无效路径 %1").arg(path));
    return false;
  }

  QMutexLocker locker(&callbackMapMutex);
  pathCallbacks[path] = CallbackInfo{
    std::move(callback),
    priority,
    true,
    description
  };

  getFolderWatcherLogger()->info("添加回调函数: 路径={}, 优先级={}, 描述={}",
                                path.toStdString(),
                                static_cast<int>(priority),
                                description.toStdString());
  return true;
}

bool FolderMonitor::removeCallback(const QString& path) {
  QMutexLocker locker(&callbackMapMutex);
  if (pathCallbacks.remove(path) > 0) {
    getFolderWatcherLogger()->info("移除回调函数: 路径={}", path.toStdString());
    return true;
  }
  return false;
}

void FolderMonitor::enableCallback(const QString& path, bool enable) {
  QMutexLocker locker(&callbackMapMutex);
  if (pathCallbacks.contains(path)) {
    pathCallbacks[path].enabled = enable;
    getFolderWatcherLogger()->info("{}回调函数: 路径={}",
                                  enable ? "启用" : "禁用",
                                  path.toStdString());
  }
}

void FolderMonitor::clearCallbacks() {
  QMutexLocker locker(&callbackMapMutex);
  pathCallbacks.clear();
  globalCallback = nullptr;
  getFolderWatcherLogger()->info("清除所有回调函数");
}

void FolderMonitor::setGlobalCallback(Callback callback) {
  QMutexLocker locker(&callbackMapMutex);
  globalCallback = std::move(callback);
  getFolderWatcherLogger()->info("设置全局回调函数");
}

QString FolderMonitor::findMatchingCallbackPath(const QString& filePath) const {
  // 找到最长匹配的路径
  QString bestMatch;
  int maxLength = -1;

  for (auto it = pathCallbacks.constBegin(); it != pathCallbacks.constEnd(); ++it) {
    const QString& path = it.key();
    if (filePath.startsWith(path) && path.length() > maxLength) {
      maxLength = path.length();
      bestMatch = path;
    }
  }
  return bestMatch;
}

void FolderMonitor::invokeCallbacks(const FileSystemEvent& event) {
  QMutexLocker locker(&callbackMapMutex);

  // 按优先级排序的回调函数列表
  QMultiMap<CallbackPriority, std::pair<QString, Callback>> priorityCallbacks;

  // 查找匹配的回调函数
  QString matchingPath = findMatchingCallbackPath(event.path);
  if (!matchingPath.isEmpty() && pathCallbacks.contains(matchingPath)) {
    const auto& callbackInfo = pathCallbacks[matchingPath];
    if (callbackInfo.enabled) {
      priorityCallbacks.insert(callbackInfo.priority,
                             {matchingPath, callbackInfo.callback});
    }
  }

  // 添加全局回调（如果存在）
  if (globalCallback) {
    priorityCallbacks.insert(CallbackPriority::Low,
                           {QString(), globalCallback});
  }

  // 按优先级执行回调
  try {
    for (auto it = priorityCallbacks.end(); it != priorityCallbacks.begin();) {
      --it;
      const auto& [path, callback] = it.value();
      if (callback) {
        callback(event);
        getFolderWatcherLogger()->debug("执行回调函数: 路径={}, 优先级={}",
                                      path.toStdString(),
                                      static_cast<int>(it.key()));
      }
    }
  } catch (const std::exception& e) {
    handleError(WatcherErrorCode::SystemError,
               QString("回调函数执行失败: %1").arg(e.what()));
  }
}

QStringList FolderMonitor::getCallbackPaths() const {
  QMutexLocker locker(&callbackMapMutex);
  return pathCallbacks.keys();
}

bool FolderMonitor::hasCallback(const QString& path) const {
  QMutexLocker locker(&callbackMapMutex);
  return pathCallbacks.contains(path);
}