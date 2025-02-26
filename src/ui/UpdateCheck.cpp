#include "UpdateCheck.h"
#include "../utils/AppUpdater.h"
#include <QApplication>
#include <QCryptographicHash>
#include <QDesktopServices>
#include <QFile>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QMessageBox>
#include <QMutex>
#include <QPushButton>
#include <QScreen>
#include <QStandardPaths>
#include <QTextStream>
#include <QVBoxLayout>
#include <QtNetwork>
#include <QtWidgets>
#include <semver.hpp>
#include <thread>

// 日志系统优化
class Logger {
public:
  static Logger &instance() {
    static Logger instance;
    return instance;
  }

  void log(const QString &message, QtMsgType msgType = QtDebugMsg) {
    QMutexLocker locker(&mutex);

    // 获取日志目录
    QString logDir =
        QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(logDir);
    QString logPath = QDir(logDir).filePath("update_checker.log");

    // 延迟打开日志文件直到需要
    if (!logFile.isOpen()) {
      logFile.setFileName(logPath);
      if (!logFile.open(QIODevice::WriteOnly | QIODevice::Append |
                        QIODevice::Text)) {
        qWarning() << "Failed to open log file:" << logPath;
        return;
      }
      logStream.setDevice(&logFile);
    }

    // 日志级别前缀
    QString levelStr;
    switch (msgType) {
    case QtDebugMsg:
      levelStr = "[DEBUG]";
      break;
    case QtInfoMsg:
      levelStr = "[INFO]";
      break;
    case QtWarningMsg:
      levelStr = "[WARNING]";
      break;
    case QtCriticalMsg:
      levelStr = "[CRITICAL]";
      break;
    case QtFatalMsg:
      levelStr = "[FATAL]";
      break;
    }

    // 写入日志
    logStream << QDateTime::currentDateTime().toString(Qt::ISODate) << " "
              << levelStr << " " << message << Qt::endl;
    logStream.flush();

    // 如果日志文件超过5MB，备份并创建新文件
    if (logFile.size() > 5 * 1024 * 1024) {
      logFile.close();
      QString backupName =
          logPath + "." +
          QDateTime::currentDateTime().toString("yyyyMMdd-hhmmss");
      QFile::rename(logPath, backupName);
      logFile.setFileName(logPath);
      logFile.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text);
      logStream.setDevice(&logFile);
    }
  }

private:
  Logger() {}
  ~Logger() {
    if (logFile.isOpen()) {
      logFile.close();
    }
  }

  QFile logFile;
  QTextStream logStream;
  QMutex mutex;
};

// 实现UpdateConfig类
UpdateConfig &UpdateConfig::instance() {
  static UpdateConfig config;
  return config;
}

UpdateConfig::UpdateConfig(QObject *parent)
    : QObject(parent),
      m_settings(QSettings::IniFormat, QSettings::UserScope,
                 qApp->organizationName(), qApp->applicationName()) {
  load();
}

bool UpdateConfig::isAutoCheckEnabled() const { return m_autoCheckEnabled; }

void UpdateConfig::setAutoCheckEnabled(bool enabled) {
  m_autoCheckEnabled = enabled;
}

int UpdateConfig::checkFrequency() const { return m_checkFrequency; }

void UpdateConfig::setCheckFrequency(int days) {
  m_checkFrequency = qBound(1, days, 90);
}

QString UpdateConfig::updateChannel() const { return m_updateChannel; }

void UpdateConfig::setUpdateChannel(const QString &channel) {
  m_updateChannel = channel;
}

QString UpdateConfig::getUpdateUrl() const {
  if (!m_customUpdateUrl.isEmpty()) {
    return m_customUpdateUrl;
  }

  // 默认URL基于通道
  QString baseUrl = "https://api.example.com/updates";
  if (m_updateChannel == "beta") {
    baseUrl = "https://api.example.com/beta-updates";
  } else if (m_updateChannel == "dev") {
    baseUrl = "https://api.example.com/dev-updates";
  }

  return baseUrl;
}

void UpdateConfig::setCustomUpdateUrl(const QString &url) {
  m_customUpdateUrl = url;
}

QDateTime UpdateConfig::lastCheckTime() const { return m_lastCheckTime; }

void UpdateConfig::setLastCheckTime(const QDateTime &dateTime) {
  m_lastCheckTime = dateTime;
}

QString UpdateConfig::skippedVersion() const { return m_skippedVersion; }

void UpdateConfig::setSkippedVersion(const QString &version) {
  m_skippedVersion = version;
}

QString UpdateConfig::ignoredVersion() const { return m_ignoredVersion; }

void UpdateConfig::setIgnoredVersion(const QString &version) {
  m_ignoredVersion = version;
}

void UpdateConfig::save() {
  m_settings.setValue("Updates/AutoCheck", m_autoCheckEnabled);
  m_settings.setValue("Updates/Frequency", m_checkFrequency);
  m_settings.setValue("Updates/Channel", m_updateChannel);
  m_settings.setValue("Updates/CustomUrl", m_customUpdateUrl);
  m_settings.setValue("Updates/LastCheck", m_lastCheckTime);
  m_settings.setValue("Updates/IgnoredVersion", m_ignoredVersion);
  m_settings.sync();
}

void UpdateConfig::load() {
  m_autoCheckEnabled = m_settings.value("Updates/AutoCheck", true).toBool();
  m_checkFrequency = m_settings.value("Updates/Frequency", 7).toInt();
  m_updateChannel = m_settings.value("Updates/Channel", "stable").toString();
  m_customUpdateUrl = m_settings.value("Updates/CustomUrl", "").toString();
  m_lastCheckTime = m_settings.value("Updates/LastCheck").toDateTime();
  m_ignoredVersion = m_settings.value("Updates/IgnoredVersion", "").toString();
}

// UpdateInfo 实现
UpdateInfo UpdateInfo::fromJson(const QJsonObject &json) {
  UpdateInfo info;
  info.version = json["version"].toString();
  info.releaseDate = json["releaseDate"].toString();
  info.changelog = json["changelog"].toString();
  info.downloadUrl = json["downloadUrl"].toString();
  info.isMandatory = json["mandatory"].toBool(false);

  QJsonArray components = json["affectedComponents"].toArray();
  for (const auto &comp : components) {
    info.affectedComponents.append(comp.toString());
  }

  info.fileSize = json["fileSize"].toVariant().toLongLong();
  info.sha256Checksum = json["sha256"].toString();

  return info;
}

QJsonObject UpdateInfo::toJson() const {
  QJsonObject json;
  json["version"] = version;
  json["releaseDate"] = releaseDate;
  json["changelog"] = changelog;
  json["downloadUrl"] = downloadUrl;
  json["mandatory"] = isMandatory;

  QJsonArray components;
  for (const auto &comp : affectedComponents) {
    components.append(comp);
  }
  json["affectedComponents"] = components;

  json["fileSize"] = QJsonValue::fromVariant(fileSize);
  json["sha256"] = sha256Checksum;

  return json;
}

// 更新UpdateChecker类
UpdateChecker::UpdateChecker(QObject *parent)
    : QObject(parent), manager(new QNetworkAccessManager(this)) {
  connect(manager, &QNetworkAccessManager::finished, this,
          &UpdateChecker::onUpdateInfoReceived);

  // 配置重试逻辑
  m_retryTimer.setSingleShot(true);
  connect(&m_retryTimer, &QTimer::timeout, this, [this]() {
    if (m_retryCount < m_maxRetries) {
      Logger::instance().log(QString("Retrying update check, attempt %1 of %2")
                                 .arg(m_retryCount + 1)
                                 .arg(m_maxRetries),
                             QtWarningMsg);
      checkForUpdates(true);
    }
  });

  connect(&m_network, &QNetworkAccessManager::finished, this,
          &UpdateChecker::onNetworkReply);
}

UpdateChecker::~UpdateChecker() { cancelCheck(); }

void UpdateChecker::cancelCheck() {
  if (m_currentReply) {
    m_currentReply->abort();
    m_currentReply->deleteLater();
    m_currentReply = nullptr;
  }
  m_retryTimer.stop();
  m_retryCount = 0;
}

void UpdateChecker::checkForUpdates(bool silent) {
  Logger::instance().log("Checking for updates...", QtInfoMsg);

  cancelCheck(); // 取消任何现有检查

  m_silent = silent;

  if (!silent && !shouldCheck()) {
    Logger::instance().log("Skipping update check, not due yet.", QtInfoMsg);
    emit updateNotAvailable();
    return;
  }

  QUrl updateUrl(UpdateConfig::instance().getUpdateUrl());
  QNetworkRequest request(updateUrl);
  request.setAttribute(QNetworkRequest::CacheLoadControlAttribute,
                       QNetworkRequest::AlwaysNetwork);
  request.setHeader(
      QNetworkRequest::UserAgentHeader,
      QString("%1/%2 (%3; %4)")
          .arg(qApp->applicationName(), qApp->applicationVersion(),
               QSysInfo::prettyProductName(), QLocale::system().name()));

  // 添加支持压缩的头部
  request.setRawHeader("Accept-Encoding", "gzip, deflate");

  m_currentReply = manager->get(request);

  connect(m_currentReply, &QNetworkReply::downloadProgress, this,
          &UpdateChecker::onDownloadProgress);

  connect(
      m_currentReply, &QNetworkReply::errorOccurred, this,
      [this](QNetworkReply::NetworkError code) { handleNetworkError(code); });

  // 设置请求超时
  QTimer::singleShot(15000, this, [this]() {
    if (m_currentReply && m_currentReply->isRunning()) {
      Logger::instance().log("Update check timed out", QtWarningMsg);
      m_currentReply->abort();
    }
  });
}

void UpdateChecker::asyncCheck(
    std::function<void(bool, const UpdateInfo &, const QString &)> callback) {
  // 保存回调
  m_callback = callback;

  // 使用Lambda连接信号以异步处理结果
  QObject *context = new QObject(this);

  connect(this, &UpdateChecker::updateAvailable, context,
          [callback, context](const UpdateInfo &info) {
            callback(true, info, QString());
            context->deleteLater();
          });

  connect(this, &UpdateChecker::updateNotAvailable, context,
          [callback, context]() {
            callback(false, UpdateInfo(), "No update available");
            context->deleteLater();
          });

  connect(this, &UpdateChecker::updateError, context,
          [callback, context](const QString &error) {
            callback(false, UpdateInfo(), error);
            context->deleteLater();
          });

  // 开始检查
  checkForUpdates(true);
}

bool UpdateChecker::shouldCheck() const {
  const auto &config = UpdateConfig::instance();
  if (!config.isAutoCheckEnabled()) {
    return false;
  }

  const auto lastCheck = config.lastCheckTime();
  if (!lastCheck.isValid()) {
    return true; // 从未检查过
  }

  return lastCheck.daysTo(QDateTime::currentDateTime()) >=
         config.checkFrequency();
}

void UpdateChecker::onUpdateInfoReceived(QNetworkReply *reply) {
  QScopedPointer<QNetworkReply, QScopedPointerDeleteLater> replyGuard(reply);
  m_currentReply = nullptr; // 清除当前回复指针

  if (reply->error() != QNetworkReply::NoError) {
    // 错误处理在errorOccurred信号中完成
    return;
  }

  QByteArray data = reply->readAll();

  // 处理可能的压缩响应
  if (reply->rawHeader("Content-Encoding") == "gzip") {
    data = qUncompress(data);
  } else if (reply->rawHeader("Content-Encoding") == "deflate") {
    // 简单deflate解压缩
    QByteArray uncompressed;
    // 这里应实现deflate解压缩，或使用库函数
    // 示例用简化写法
    data = uncompressed;
  }

  // 在单独线程中解析JSON以避免阻塞UI
  std::thread([this, data]() {
    const auto processResult = parseUpdateInfo(data);

    // 回到主线程处理结果
    QMetaObject::invokeMethod(
        this,
        [this, processResult]() {
          if (processResult.hasError()) {
            Logger::instance().log(processResult.error(), QtWarningMsg);
            emit updateError(processResult.error());
            return;
          }

          UpdateConfig::instance().setLastCheckTime(
              QDateTime::currentDateTime());
          UpdateConfig::instance().save();

          const auto &updateInfo = processResult.value();
          const auto &ignoredVersion =
              UpdateConfig::instance().ignoredVersion();

          // 检查是否是被忽略的版本
          if (!ignoredVersion.isEmpty() &&
              ignoredVersion == updateInfo.version) {
            Logger::instance().log(
                "Update available but version was ignored: " +
                    updateInfo.version,
                QtInfoMsg);
            emit updateNotAvailable();
          } else {
            Logger::instance().log("Update available: " + updateInfo.version,
                                   QtInfoMsg);
            emit updateAvailable(updateInfo);
          }
        },
        Qt::QueuedConnection);
  }).detach();
}

void UpdateChecker::onDownloadProgress(qint64 bytesReceived,
                                       qint64 bytesTotal) {
  if (bytesTotal > 0) {
    int percentage = static_cast<int>((bytesReceived * 100) / bytesTotal);
    emit checkProgress(percentage);
  }
}

void UpdateChecker::onNetworkReply(QNetworkReply *reply) {
  if (reply != m_currentReply)
    return;

  emit checkProgress(90); // 接近完成

  UpdateInfo info;
  QString error;
  bool hasUpdate = false;

  if (reply->error() == QNetworkReply::NoError) {
    QJsonDocument jsonResponse = QJsonDocument::fromJson(reply->readAll());

    if (!jsonResponse.isNull()) {
      info = UpdateInfo::fromJson(jsonResponse);

      if (info.isValid()) {
        QString currentVersion = getCurrentVersion();
        hasUpdate = compareVersions(info.version, currentVersion);

        // 检查是否是已跳过的版本
        if (hasUpdate &&
            info.version == UpdateConfig::instance().skippedVersion()) {
          hasUpdate = false; // 用户已跳过此版本
        }
      } else {
        error = tr("Invalid update information");
      }
    } else {
      error = tr("Invalid response from update server");
    }
  } else {
    error = reply->errorString();
  }

  emit checkProgress(100); // 完成

  // 处理结果
  if (!error.isEmpty()) {
    if (m_callback) {
      m_callback(false, info, error);
    }
    emit updateError(error);
  } else if (hasUpdate) {
    if (m_callback) {
      m_callback(true, info, QString());
    }
    emit updateAvailable(info);
  } else {
    if (m_callback) {
      m_callback(false, info, QString());
    }
    emit updateNotAvailable();
  }

  m_callback = nullptr;
  m_currentReply->deleteLater();
  m_currentReply = nullptr;
}

Result<UpdateInfo>
UpdateChecker::parseUpdateInfo(const QByteArray &data) noexcept {
  QJsonParseError parseError;
  const auto doc = QJsonDocument::fromJson(data, &parseError);
  if (parseError.error != QJsonParseError::NoError) {
    const QString errorMsg =
        tr("JSON parse error: %1").arg(parseError.errorString());
    return Result<UpdateInfo>(errorMsg);
  }

  try {
    const auto currentVersion =
        semver::version{qApp->applicationVersion().toStdString()};
    const auto latestVersion =
        semver::version{doc["version"].toString().toStdString()};

    if (latestVersion <= currentVersion) {
      return Result<UpdateInfo>(tr("Already up to date"));
    }

    return Result<UpdateInfo>(UpdateInfo::fromJson(doc.object()));
  } catch (const std::exception &e) {
    return Result<UpdateInfo>(tr("Version parse error: %1").arg(e.what()));
  }
}

void UpdateChecker::handleNetworkError(QNetworkReply::NetworkError code) {
  const QString message = [code] {
    switch (code) {
    case QNetworkReply::TimeoutError:
      return tr("Connection timeout");
    case QNetworkReply::SslHandshakeFailedError:
      return tr("SSL handshake failed");
    case QNetworkReply::OperationCanceledError:
      return tr("Operation canceled");
    default:
      return tr("Network error occurred: %1").arg(static_cast<int>(code));
    }
  }();

  Logger::instance().log("Network error: " + message, QtWarningMsg);

  if (code != QNetworkReply::OperationCanceledError) {
    scheduleRetry();
  } else {
    emit updateError(message);
  }
}

void UpdateChecker::scheduleRetry() {
  if (++m_retryCount <= m_maxRetries) {
    // 指数退避策略
    int delay = 1000 * (1 << (m_retryCount - 1)); // 1s, 2s, 4s, ...
    Logger::instance().log(QString("Scheduling retry in %1 ms").arg(delay),
                           QtInfoMsg);
    m_retryTimer.start(delay);
  } else {
    Logger::instance().log(
        QString("Max retries (%1) reached, giving up").arg(m_maxRetries),
        QtCriticalMsg);
    emit updateError(tr("Failed after %1 retry attempts").arg(m_maxRetries));
    m_retryCount = 0;
  }
}

QString UpdateChecker::getCurrentVersion() const {
  return qApp->applicationVersion();
}

QUrl UpdateChecker::getUpdateUrl() const {
  return QUrl(UpdateConfig::instance().getUpdateUrl());
}

bool UpdateChecker::compareVersions(const QString &newVer,
                                    const QString &currentVer) {
  // 简单版本比较，格式假设为 x.y.z
  QStringList newParts = newVer.split('.');
  QStringList currentParts = currentVer.split('.');

  // 确保两个版本号都至少有一部分
  if (newParts.isEmpty() || currentParts.isEmpty()) {
    return false;
  }

  // 比较每个部分
  for (int i = 0; i < qMin(newParts.size(), currentParts.size()); ++i) {
    int newVal = newParts[i].toInt();
    int currentVal = currentParts[i].toInt();

    if (newVal > currentVal) {
      return true;
    } else if (newVal < currentVal) {
      return false;
    }
  }

  // 如果进行到这里，检查是否有更多的版本部分
  return newParts.size() > currentParts.size();
}

// 更新 UpdateDialog 实现
UpdateDialog::UpdateDialog(const UpdateInfo &updateInfo, QWidget *parent)
    : QDialog(parent), m_updateInfo(updateInfo) {
  setWindowTitle(tr("Update Available"));
  setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);

  // 设置窗口图标
  setWindowIcon(QIcon(":/icons/update.png"));

  setupUI(updateInfo);
  loadUserPreferences();

  // 创建动画效果
  m_animation = new QPropertyAnimation(this, "windowOpacity");
  m_animation->setDuration(300);
  m_animation->setStartValue(0.0);
  m_animation->setEndValue(1.0);
  m_animation->start();

  // 自动调整布局以适应屏幕大小
  adjustLayoutForScreenSize();
}

UpdateDialog::~UpdateDialog() { saveUserPreferences(); }

bool UpdateDialog::showUpdateDialogIfAvailable(QWidget *parent) {
  UpdateChecker checker;
  QEventLoop loop;
  bool updateAvailable = false;

  QObject::connect(&checker, &UpdateChecker::updateAvailable,
                   [&loop, &updateAvailable, parent](const UpdateInfo &info) {
                     updateAvailable = true;
                     UpdateDialog *dialog = new UpdateDialog(info, parent);
                     dialog->setAttribute(Qt::WA_DeleteOnClose);
                     dialog->show();
                     loop.quit();
                   });

  QObject::connect(&checker, &UpdateChecker::updateNotAvailable,
                   [&loop]() { loop.quit(); });

  QObject::connect(&checker, &UpdateChecker::updateError,
                   [&loop](const QString &) { loop.quit(); });

  checker.checkForUpdates();

  // 使用超时防止无限等待
  QTimer::singleShot(10000, &loop, &QEventLoop::quit);
  loop.exec();

  return updateAvailable;
}

void UpdateDialog::setupUI(const UpdateInfo &info) {
  m_mainLayout = new QVBoxLayout(this);
  m_mainLayout->setSpacing(12);
  m_mainLayout->setContentsMargins(15, 15, 15, 15);

  // 版本信息和图标
  QHBoxLayout *headerLayout = new QHBoxLayout();
  QLabel *iconLabel = new QLabel(this);
  QPixmap updateIcon(":/icons/update_available.png");
  if (!updateIcon.isNull()) {
    iconLabel->setPixmap(updateIcon.scaled(48, 48, Qt::KeepAspectRatio,
                                           Qt::SmoothTransformation));
  } else {
    iconLabel->setText("🔄"); // 备用文本图标
    iconLabel->setStyleSheet("font-size: 24pt;");
  }

  m_titleLabel =
      new QLabel(tr("A new version %1 is available!").arg(info.version), this);
  m_titleLabel->setStyleSheet("font-weight: bold; font-size: 14pt;");
  m_titleLabel->setWordWrap(true);

  headerLayout->addWidget(iconLabel);
  headerLayout->addWidget(m_titleLabel, 1);
  m_mainLayout->addLayout(headerLayout);

  // 分割线
  QFrame *line = new QFrame(this);
  line->setFrameShape(QFrame::HLine);
  line->setFrameShadow(QFrame::Sunken);
  m_mainLayout->addWidget(line);

  // 更新日志
  QLabel *changelogHeader = new QLabel(tr("What's New"), this);
  changelogHeader->setStyleSheet("font-weight: bold;");
  m_mainLayout->addWidget(changelogHeader);

  m_changelogBrowser = new QTextBrowser(this);
  m_changelogBrowser->setReadOnly(true);
  m_changelogBrowser->setMinimumHeight(150);
  m_changelogBrowser->setOpenExternalLinks(true);
  m_changelogBrowser->setHtml(formatChangelog(info.changelog));
  m_mainLayout->addWidget(m_changelogBrowser);

  // 进度条（初始隐藏）
  m_downloadProgress = new QProgressBar(this);
  m_downloadProgress->setRange(0, 100);
  m_downloadProgress->setValue(0);
  m_downloadProgress->setVisible(false);
  m_mainLayout->addWidget(m_downloadProgress);

  // 设置选项
  QGroupBox *settingsGroup = new QGroupBox(tr("Settings"), this);
  QVBoxLayout *settingsLayout = new QVBoxLayout(settingsGroup);

  // 自动检查更新选项
  m_autoCheckBox = new QCheckBox(tr("Automatically check for updates"), this);
  m_autoCheckBox->setChecked(UpdateConfig::instance().isAutoCheckEnabled());
  connect(m_autoCheckBox, &QCheckBox::stateChanged, this,
          &UpdateDialog::onAutoCheckChanged);
  settingsLayout->addWidget(m_autoCheckBox);

  // 更新通道选择
  QHBoxLayout *channelLayout = new QHBoxLayout();
  QLabel *channelLabel = new QLabel(tr("Update Channel:"), this);
  m_channelComboBox = new QComboBox(this);
  m_channelComboBox->addItem(tr("Stable"), "stable");
  m_channelComboBox->addItem(tr("Beta"), "beta");
  m_channelComboBox->addItem(tr("Development"), "dev");

  // 设置当前通道
  int index =
      m_channelComboBox->findData(UpdateConfig::instance().updateChannel());
  m_channelComboBox->setCurrentIndex(index != -1 ? index : 0);

  connect(m_channelComboBox, &QComboBox::currentTextChanged, this,
          &UpdateDialog::onChannelChanged);

  channelLayout->addWidget(channelLabel);
  channelLayout->addWidget(m_channelComboBox);
  channelLayout->addStretch(1);
  settingsLayout->addLayout(channelLayout);

  m_mainLayout->addWidget(settingsGroup);

  // 按钮区域
  QHBoxLayout *buttonLayout = new QHBoxLayout();

  // 文件大小信息
  if (info.fileSize > 0) {
    QString sizeText =
        QString("%1 MB").arg(info.fileSize / 1024.0 / 1024.0, 0, 'f', 1);
    QLabel *sizeLabel = new QLabel(tr("Size: %1").arg(sizeText), this);
    sizeLabel->setStyleSheet("color: #666;");
    buttonLayout->addWidget(sizeLabel);
  }

  buttonLayout->addStretch(1);

  m_ignoreButton = new QPushButton(tr("Ignore Version"), this);
  m_laterButton = new QPushButton(tr("Remind Later"), this);
  m_updateButton = new QPushButton(tr("Update Now"), this);
  m_updateButton->setDefault(true);
  m_updateButton->setStyleSheet("QPushButton { font-weight: bold; }");
  buttonLayout->addWidget(m_ignoreButton);
  buttonLayout->addWidget(m_laterButton);
  buttonLayout->addWidget(m_updateButton);
  m_mainLayout->addLayout(buttonLayout);

  // 连接信号与槽
  connect(m_ignoreButton, &QPushButton::clicked, this,
          &UpdateDialog::onIgnoreClicked);
  connect(m_laterButton, &QPushButton::clicked, this,
          &UpdateDialog::onLaterClicked);
  connect(m_updateButton, &QPushButton::clicked, this,
          &UpdateDialog::onUpdateClicked);

  // 强制更新模式处理
  if (info.isMandatory) {
    m_ignoreButton->setEnabled(false);
    m_laterButton->setEnabled(false);
    m_titleLabel->setText(
        tr("Mandatory Update %1 Available").arg(info.version));
  }
}

QString UpdateDialog::formatChangelog(const QString &raw) const {
  // 将纯文本更新日志转换为HTML格式，支持Markdown风格的标记
  QString html = "<html><body>";

  // 处理Markdown样式
  const QStringList lines = raw.split('\n');
  bool inList = false;
  bool inCodeBlock = false;

  for (const QString &line : lines) {
    QString processedLine = line.trimmed();

    if (processedLine.isEmpty()) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += "<p></p>";
      continue;
    }

    // 处理代码块
    if (processedLine.startsWith("```")) {
      if (inCodeBlock) {
        html += "</code></pre>";
        inCodeBlock = false;
      } else {
        html += "<pre><code>";
        inCodeBlock = true;
      }
      continue;
    }

    if (inCodeBlock) {
      html += processedLine.toHtmlEscaped() + "<br/>";
      continue;
    }

    // 处理标题
    if (processedLine.startsWith("# ")) {
      html += "<h1>" + processedLine.mid(2).toHtmlEscaped() + "</h1>";
    } else if (processedLine.startsWith("## ")) {
      html += "<h2>" + processedLine.mid(3).toHtmlEscaped() + "</h2>";
    } else if (processedLine.startsWith("### ")) {
      html += "<h3>" + processedLine.mid(4).toHtmlEscaped() + "</h3>";
    }
    // 处理列表
    else if (processedLine.startsWith("- ") || processedLine.startsWith("* ")) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += "<li>" + processedLine.mid(2).toHtmlEscaped() + "</li>";
    }
    // 处理链接 [text](url)
    else if (processedLine.contains("[") && processedLine.contains("](")) {
      QString text = processedLine;
      QRegularExpression linkRegex("\\[([^\\]]+)\\]\\(([^\\)]+)\\)");
      QRegularExpressionMatchIterator i = linkRegex.globalMatch(processedLine);
      while (i.hasNext()) {
        QRegularExpressionMatch match = i.next();
        QString linkText = match.captured(1);
        QString url = match.captured(2);
        text.replace(match.captured(0),
                     QString("<a href=\"%1\">%2</a>").arg(url, linkText));
      }
      html += "<p>" + text + "</p>";
    }
    // 普通文本
    else {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += "<p>" + processedLine.toHtmlEscaped() + "</p>";
    }
  }

  if (inList) {
    html += "</ul>";
  }

  if (inCodeBlock) {
    html += "</code></pre>";
  }

  html += "</body></html>";
  return html;
}

void UpdateDialog::loadUserPreferences() {
  QSettings settings;

  // 加载窗口大小
  QSize savedSize =
      settings.value("UpdateDialog/size", QSize(550, 450)).toSize();
  // 确保窗口大小合理
  QSize minSize(400, 300);
  QSize finalSize(qMax(savedSize.width(), minSize.width()),
                  qMax(savedSize.height(), minSize.height()));
  resize(finalSize);
}

void UpdateDialog::saveUserPreferences() {
  QSettings settings;
  settings.setValue("UpdateDialog/size", size());
}

void UpdateDialog::adjustLayoutForScreenSize() {
  // 获取当前屏幕尺寸
  QScreen *screen = QGuiApplication::primaryScreen();
  if (!screen)
    return;

  QRect screenGeometry = screen->availableGeometry();
  int screenWidth = screenGeometry.width();
  int screenHeight = screenGeometry.height();

  // 根据屏幕尺寸调整布局
  if (screenWidth <= 800 || screenHeight <= 600) {
    // 小屏幕模式
    setMinimumSize(350, 300);
    m_mainLayout->setSpacing(6);
    m_mainLayout->setContentsMargins(10, 10, 10, 10);
    m_titleLabel->setStyleSheet("font-weight: bold; font-size: 12pt;");
    m_changelogBrowser->setMinimumHeight(120);
  } else if (screenWidth <= 1366) {
    // 中等屏幕
    setMinimumSize(450, 350);
    m_mainLayout->setSpacing(10);
    m_mainLayout->setContentsMargins(12, 12, 12, 12);
  } else {
    // 大屏幕
    setMinimumSize(550, 400);
    m_mainLayout->setSpacing(15);
    m_mainLayout->setContentsMargins(20, 20, 20, 20);
    m_titleLabel->setStyleSheet("font-weight: bold; font-size: 16pt;");
    m_changelogBrowser->setMinimumHeight(200);
  }

  // 移动窗口到屏幕中央
  move(screenGeometry.center() - frameGeometry().center());
}

void UpdateDialog::resizeEvent(QResizeEvent *event) {
  QDialog::resizeEvent(event);
  // 可以在这里进行其他响应式布局调整
}

void UpdateDialog::closeEvent(QCloseEvent *event) {
  saveUserPreferences();
  QDialog::closeEvent(event);
}

void UpdateDialog::onIgnoreClicked() {
  UpdateConfig::instance().setIgnoredVersion(m_updateInfo.version);
  UpdateConfig::instance().save();
  accept();
}

void UpdateDialog::onLaterClicked() {
  // 简单地关闭对话框，下次检查时会再次提醒
  accept();
}

void UpdateDialog::onUpdateClicked() {
  if (!m_isDownloading && !m_updateInfo.downloadUrl.isEmpty()) {
    startDownload();
  } else {
    // 如果没有直接下载链接，打开浏览器到下载页
    QDesktopServices::openUrl(
        QUrl(m_updateInfo.downloadUrl.isEmpty()
                 ? "https://github.com/yourusername/yourproject/releases/latest"
                 : m_updateInfo.downloadUrl));
    accept();
  }
}

void UpdateDialog::startDownload() {
  // 设置下载路径
  QString downloadPath =
      QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
  QString fileName = QFileInfo(m_updateInfo.downloadUrl).fileName();
  if (fileName.isEmpty()) {
    fileName = QString("%1-%2-update.zip")
                   .arg(qApp->applicationName(), m_updateInfo.version);
  }
  QString filePath = QDir(downloadPath).filePath(fileName);

  // 打开文件准备写入
  m_downloadFile.setFileName(filePath);
  if (!m_downloadFile.open(QIODevice::WriteOnly)) {
    QMessageBox::critical(this, tr("Error"),
                          tr("Could not open file for writing: %1")
                              .arg(m_downloadFile.errorString()));
    return;
  }

  // 设置下载请求
  QNetworkRequest request(QUrl(m_updateInfo.downloadUrl));
  request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                       QNetworkRequest::NoLessSafeRedirectPolicy);

  // 开始下载
  m_downloadReply.reset(m_networkManager.get(request));

  // 连接信号
  connect(m_downloadReply.data(), &QNetworkReply::readyRead, this, [this]() {
    if (m_downloadReply && m_downloadFile.isOpen()) {
      m_downloadFile.write(m_downloadReply->readAll());
    }
  });

  connect(m_downloadReply.data(), &QNetworkReply::downloadProgress, this,
          &UpdateDialog::onDownloadProgress);

  connect(m_downloadReply.data(), &QNetworkReply::errorOccurred, this,
          &UpdateDialog::onDownloadError);

  connect(m_downloadReply.data(), &QNetworkReply::finished, this,
          &UpdateDialog::onDownloadFinished);

  // 更新界面状态
  m_isDownloading = true;
  updateDialogState(DialogState::Downloading);
}

void UpdateDialog::onDownloadProgress(qint64 bytesReceived, qint64 bytesTotal) {
  if (bytesTotal > 0) {
    int percentage = static_cast<int>(bytesReceived * 100 / bytesTotal);
    m_downloadProgress->setValue(percentage);

    // 更新窗口标题显示进度
    setWindowTitle(tr("Downloading Update (%1%)").arg(percentage));
  } else {
    m_downloadProgress->setRange(0, 0); // 不确定进度
  }
}

void UpdateDialog::onDownloadFinished() {
  m_isDownloading = false;

  if (!m_downloadReply)
    return;

  m_downloadFile.close();

  if (m_downloadReply->error() != QNetworkReply::NoError) {
    // 错误处理已在onDownloadError中完成
    return;
  }

  // 验证SHA256校验和（如果提供）
  if (!m_updateInfo.sha256Checksum.isEmpty()) {
    if (!m_downloadFile.open(QIODevice::ReadOnly)) {
      QMessageBox::warning(
          this, tr("Verification Failed"),
          tr("Could not open downloaded file for verification."));
      return;
    }

    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (hash.addData(&m_downloadFile)) {
      QString fileHash = hash.result().toHex();
      m_downloadFile.close();

      if (fileHash != m_updateInfo.sha256Checksum) {
        QMessageBox::critical(
            this, tr("Verification Failed"),
            tr("The downloaded file is corrupted. Please try again."));
        m_downloadFile.remove();
        return;
      }
    }
  }

  updateDialogState(DialogState::DownloadComplete);

  // 提示安装
  QMessageBox::StandardButton result = QMessageBox::question(
      this, tr("Download Complete"),
      tr("Update downloaded to:\n%1\n\nDo you want to install it now?")
          .arg(QDir::toNativeSeparators(m_downloadFile.fileName())));

  if (result == QMessageBox::Yes) {
    // 尝试启动安装程序
    bool success = QDesktopServices::openUrl(
        QUrl::fromLocalFile(m_downloadFile.fileName()));
    if (success) {
      accept();     // 关闭对话框
      qApp->quit(); // 可选：关闭应用程序，以便安装更新
    } else {
      QMessageBox::information(
          this, tr("Information"),
          tr("Please manually run the installer at:\n%1")
              .arg(QDir::toNativeSeparators(m_downloadFile.fileName())));
    }
  }
}

void UpdateDialog::onDownloadError(QNetworkReply::NetworkError error) {
  m_isDownloading = false;
  m_downloadFile.close();

  updateDialogState(DialogState::Error);

  QString errorMessage;
  switch (error) {
  case QNetworkReply::OperationCanceledError:
    errorMessage = tr("Download cancelled");
    break;
  case QNetworkReply::TimeoutError:
    errorMessage = tr("Connection timed out");
    break;
  case QNetworkReply::ContentNotFoundError:
    errorMessage = tr("File not found on server");
    break;
  default:
    errorMessage = tr("Network error: %1").arg(m_downloadReply->errorString());
  }

  QMessageBox::warning(this, tr("Download Failed"),
                       tr("Failed to download update: %1").arg(errorMessage));
}

void UpdateDialog::updateDialogState(DialogState state) {
  m_currentState = state;

  switch (state) {
  case DialogState::Information:
    m_downloadProgress->setVisible(false);
    m_ignoreButton->setEnabled(true);
    m_laterButton->setEnabled(true);
    m_updateButton->setText(tr("Update Now"));
    break;

  case DialogState::Downloading:
    m_downloadProgress->setVisible(true);
    m_ignoreButton->setEnabled(false);
    m_laterButton->setText(tr("Cancel"));
    m_updateButton->setEnabled(false);
    break;

  case DialogState::DownloadComplete:
    m_downloadProgress->setVisible(true);
    m_downloadProgress->setValue(100);
    m_ignoreButton->setEnabled(false);
    m_laterButton->setText(tr("Later"));
    m_laterButton->setEnabled(true);
    m_updateButton->setEnabled(true);
    m_updateButton->setText(tr("Install Now"));
    break;

  case DialogState::Error:
    m_downloadProgress->setVisible(false);
    m_ignoreButton->setEnabled(true);
    m_laterButton->setEnabled(true);
    m_laterButton->setText(tr("Later"));
    m_updateButton->setEnabled(true);
    m_updateButton->setText(tr("Retry"));
    break;
  }
}

void UpdateDialog::onAutoCheckChanged(int state) {
  bool enabled = (state == Qt::Checked);
  UpdateConfig::instance().setAutoCheckEnabled(enabled);
  UpdateConfig::instance().save();
}

void UpdateDialog::onChannelChanged(const QString &channel) {
  int index = m_channelComboBox->currentIndex();
  if (index >= 0) {
    QString channelId = m_channelComboBox->itemData(index).toString();
    UpdateConfig::instance().setUpdateChannel(channelId);
    UpdateConfig::instance().save();
  }
}

// 实现 UpdateSettingsDialog
UpdateSettingsDialog::UpdateSettingsDialog(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Update Settings"));
  setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);

  setupUI();
  loadSettings();
}

void UpdateSettingsDialog::setupUI() {
  auto mainLayout = new QVBoxLayout(this);
  mainLayout->setSpacing(10);

  // 自动检查选项
  m_autoCheckBox = new QCheckBox(tr("Automatically check for updates"), this);
  mainLayout->addWidget(m_autoCheckBox);

  // 检查频率
  auto freqLayout = new QHBoxLayout();
  auto freqLabel = new QLabel(tr("Check frequency:"), this);
  m_frequencySpinBox = new QSpinBox(this);
  m_frequencySpinBox->setRange(1, 90);
  m_frequencySpinBox->setSuffix(tr(" days"));
  freqLayout->addWidget(freqLabel);
  freqLayout->addWidget(m_frequencySpinBox);
  freqLayout->addStretch(1);
  mainLayout->addLayout(freqLayout);

  // 更新通道
  auto channelLayout = new QHBoxLayout();
  auto channelLabel = new QLabel(tr("Update channel:"), this);
  m_channelCombo = new QComboBox(this);
  m_channelCombo->addItem(tr("Stable"), "stable");
  m_channelCombo->addItem(tr("Beta"), "beta");
  m_channelCombo->addItem(tr("Development"), "dev");
  channelLayout->addWidget(channelLabel);
  channelLayout->addWidget(m_channelCombo);
  channelLayout->addStretch(1);
  mainLayout->addLayout(channelLayout);

  // 自定义URL
  auto customUrlLayout = new QHBoxLayout();
  auto customUrlLabel = new QLabel(tr("Custom update URL:"), this);
  m_customUrlEdit = new QLineEdit(this);
  m_customUrlEdit->setPlaceholderText(tr("Leave empty for default URL"));
  customUrlLayout->addWidget(customUrlLabel);
  customUrlLayout->addWidget(m_customUrlEdit);
  mainLayout->addLayout(customUrlLayout);

  // 按钮
  auto buttonBox = new QDialogButtonBox(
      QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
  mainLayout->addWidget(buttonBox);

  connect(buttonBox, &QDialogButtonBox::accepted, this,
          &UpdateSettingsDialog::onAccepted);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

  // 启用/禁用逻辑
  connect(m_autoCheckBox, &QCheckBox::toggled, m_frequencySpinBox,
          &QSpinBox::setEnabled);
}

void UpdateSettingsDialog::loadSettings() {
  auto &config = UpdateConfig::instance();

  m_autoCheckBox->setChecked(config.isAutoCheckEnabled());
  m_frequencySpinBox->setValue(config.checkFrequency());
  m_frequencySpinBox->setEnabled(config.isAutoCheckEnabled());

  int channelIndex = m_channelCombo->findData(config.updateChannel());
  m_channelCombo->setCurrentIndex(channelIndex != -1 ? channelIndex : 0);

  m_customUrlEdit->setText(config.getUpdateUrl());
}

void UpdateSettingsDialog::saveSettings() {
  auto &config = UpdateConfig::instance();

  config.setAutoCheckEnabled(m_autoCheckBox->isChecked());
  config.setCheckFrequency(m_frequencySpinBox->value());

  int channelIndex = m_channelCombo->currentIndex();
  if (channelIndex >= 0) {
    config.setUpdateChannel(m_channelCombo->itemData(channelIndex).toString());
  }

  config.setCustomUpdateUrl(m_customUrlEdit->text());
  config.save();
}

void UpdateSettingsDialog::onAccepted() {
  saveSettings();
  accept();
}

void UpdateSettingsDialog::onSaveSettings() {
  saveSettings();
  accept();
}

void UpdateSettingsDialog::onCheckNowClicked() {
  // 手动检查更新
  AppUpdater::instance().checkForUpdates(true, this);
}

// ========== UpdateInfo 实现 ==========

UpdateInfo UpdateInfo::fromJson(const QJsonDocument &json) {
  UpdateInfo info;

  if (json.isObject()) {
    QJsonObject obj = json.object();

    info.version = obj["version"].toString();
    info.releaseDate = obj["releaseDate"].toString();
    info.downloadUrl = obj["downloadUrl"].toString();
    info.releaseNotes = obj["releaseNotes"].toString();
    info.fileSize = obj["fileSize"].toInt(0);
    info.minOSVersion = obj["minOSVersion"].toString();
    info.isMandatory = obj["mandatory"].toBool(false);
  }

  return info;
}
