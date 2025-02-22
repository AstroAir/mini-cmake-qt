
#include "UpdateCheck.h"
#include <QFile>
#include <QMutex>
#include <QTextStream>
#include <QtNetwork>
#include <QtWidgets>
#include <semver.hpp>

class Logger {
public:
  Logger(const QString &filename) : logFile(filename) {
    logFile.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text);
    logStream.setDevice(&logFile);
  }

  ~Logger() {
    if (logFile.isOpen()) {
      logFile.close();
    }
  }

  void log(const QString &message) {
    QMutexLocker locker(&mutex);
    logStream << QDateTime::currentDateTime().toString(Qt::ISODate) << " "
              << message << Qt::endl;
    logStream.flush();
  }

private:
  QFile logFile;
  QTextStream logStream;
  QMutex mutex;
};

static Logger logger("update_check.log");

UpdateChecker::UpdateChecker(QObject *parent)
    : QObject(parent), manager(new QNetworkAccessManager(this)) {
  connect(manager, &QNetworkAccessManager::finished, this,
          &UpdateChecker::onUpdateInfoReceived);
}

void UpdateChecker::checkForUpdates(bool force) {
  logger.log("Checking for updates..."); // Log activity
  if (!force && !shouldCheck()) {
    logger.log("Skipping update check, not due yet.");
    emit updateNotAvailable();
    return;
  }

  QUrl updateUrl("https://api.example.com/latest-version");
  QNetworkRequest request(updateUrl);
  request.setAttribute(QNetworkRequest::CacheLoadControlAttribute,
                       QNetworkRequest::AlwaysNetwork);
  request.setHeader(QNetworkRequest::UserAgentHeader,
                    QString("%1/%2").arg(qApp->applicationName(),
                                         qApp->applicationVersion()));

  auto reply = manager->get(request);
  connect(
      reply, &QNetworkReply::errorOccurred, this,
      [this](QNetworkReply::NetworkError code) { handleNetworkError(code); });
  QTimer::singleShot(15000, reply, &QNetworkReply::abort);
}

bool UpdateChecker::shouldCheck() const {
  const auto lastCheck = settings.value("lastUpdateCheck").toDateTime();
  return lastCheck.daysTo(QDateTime::currentDateTime()) > 7;
}

void UpdateChecker::onUpdateInfoReceived(QNetworkReply *reply) {
  const auto process = qScopeGuard([reply] { reply->deleteLater(); });

  if (reply->error() != QNetworkReply::NoError) {
    logger.log(QString("Network error: %1").arg(reply->errorString()));
    emit updateError(tr("Network error: %1").arg(reply->errorString()));
    return;
  }

  const auto processResult = parseUpdateInfo(reply->readAll());
  if (processResult.hasError()) {
    logger.log(processResult.error());
    emit updateError(processResult.error());
    return;
  }

  settings.setValue("lastUpdateCheck", QDateTime::currentDateTime());
  emit updateAvailable(processResult.value());
}

Result<QJsonObject>
UpdateChecker::parseUpdateInfo(const QByteArray &data) noexcept {
  QJsonParseError parseError;
  const auto doc = QJsonDocument::fromJson(data, &parseError);
  if (parseError.error != QJsonParseError::NoError) {
    const QString errorMsg =
        tr("JSON parse error: %1").arg(parseError.errorString());
    logger.log(errorMsg);
    return Result<QJsonObject>(errorMsg);
  }

  const auto currentVersion =
      semver::version{qApp->applicationVersion().toStdString()};
  const auto latestVersion =
      semver::version{doc["version"].toString().toStdString()};

  if (latestVersion <= currentVersion) {
    logger.log("Already up to date");
    return Result<QJsonObject>(tr("Already up to date"));
  }

  return Result<QJsonObject>(doc.object());
}

void UpdateChecker::handleNetworkError(QNetworkReply::NetworkError code) {
  const QString message = [code] {
    switch (code) {
    case QNetworkReply::TimeoutError:
      return tr("Connection timeout");
    case QNetworkReply::SslHandshakeFailedError:
      return tr("SSL handshake failed");
    default:
      return tr("Network error occurred");
    }
  }();
  logger.log(message);
  emit updateError(message);
}

UpdateDialog::UpdateDialog(const QJsonObject &updateInfo, QWidget *parent)
    : QDialog(parent) {
  setWindowTitle(tr("Update Available"));
  setupUI(updateInfo);
  loadUserPreferences();
}

void UpdateDialog::setupUI(const QJsonObject &info) {
  auto layout = new QVBoxLayout(this);

  // 版本信息
  auto titleLabel = new QLabel(
      tr("A new version %1 is available!").arg(info["version"].toString()),
      this);
  titleLabel->setStyleSheet("font-weight: bold;");
  layout->addWidget(titleLabel);

  // 更新日志
  auto changelogLabel = new QLabel(this);
  changelogLabel->setTextFormat(Qt::RichText);
  changelogLabel->setWordWrap(true);
  changelogLabel->setText(formatChangelog(info["changelog"].toString()));
  layout->addWidget(changelogLabel);

  // 按钮区域
  auto buttonBox = new QHBoxLayout;
  auto ignoreButton = new QPushButton(tr("Ignore This Version"), this);
  auto laterButton = new QPushButton(tr("Remind Me Later"), this);
  auto updateButton = new QPushButton(tr("Update Now"), this);

  updateButton->setDefault(true);

  buttonBox->addWidget(ignoreButton);
  buttonBox->addWidget(laterButton);
  buttonBox->addWidget(updateButton);
  layout->addLayout(buttonBox);

  connect(ignoreButton, &QPushButton::clicked, this,
          &UpdateDialog::onIgnoreClicked);
  connect(laterButton, &QPushButton::clicked, this,
          &UpdateDialog::onLaterClicked);
  connect(updateButton, &QPushButton::clicked, this,
          &UpdateDialog::onUpdateClicked);
}

QString UpdateDialog::formatChangelog(const QString &raw) const {
  // 将纯文本更新日志转换为HTML格式
  QString html = "<html><body><ul>";
  const auto lines = raw.split('\n', Qt::SkipEmptyParts);

  for (const auto &line : lines) {
    QString trimmed = line.trimmed();
    if (!trimmed.isEmpty()) {
      // 处理Markdown风格的列表项
      if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
        trimmed = trimmed.mid(2);
      }
      html += "<li>" + trimmed.toHtmlEscaped() + "</li>";
    }
  }

  html += "</ul></body></html>";
  return html;
}

void UpdateDialog::loadUserPreferences() {
  QSettings settings;
  resize(settings.value("UpdateDialog/size", QSize(400, 300)).toSize());
}

void UpdateDialog::saveUserPreferences() {
  QSettings settings;
  settings.setValue("UpdateDialog/size", size());
}

void UpdateDialog::closeEvent(QCloseEvent *event) {
  saveUserPreferences();
  QDialog::closeEvent(event);
}

void UpdateDialog::onIgnoreClicked() {
  QSettings settings;
  settings.setValue("ignoredVersion", qApp->applicationVersion());
  accept();
}

void UpdateDialog::onLaterClicked() {
  // 简单地关闭对话框，下次检查时会再次提醒
  accept();
}

void UpdateDialog::onUpdateClicked() {
  QDesktopServices::openUrl(
      QUrl("https://github.com/yourusername/yourproject/releases/latest"));
  accept();
}