#include "CrashDialog.hpp"
#include "CrashHandler.hpp"

#include <QApplication>
#include <QClipboard>
#include <QDateTime>
#include <QFileDialog>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QNetworkAccessManager>
#include <QNetworkDiskCache>
#include <QNetworkRequest>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QStandardPaths>
#include <QStyle>
#include <QToolButton>
#include <QToolTip>
#include <QVBoxLayout>
#include <QtConcurrent>

#ifdef Q_OS_WIN
#include <windows.h>
#elif defined(Q_OS_LINUX)
#include <sys/utsname.h>
#endif

QString CrashDialog::s_reportServerUrl =
    "https://crash-report.example.com/api/submit";
QMap<QString, QString> CrashDialog::s_customFields;

CrashDialog::CrashDialog(const QString &log, QWidget *parent)
    : QDialog(parent), m_fullLog(log) {
  setWindowTitle(tr("Application Crash Report"));
  setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
  setFixedSize(720, 560);

  setupUI();
  m_detailInfo->setPlainText(collectSystemInfo() + "\n\n" + log);

  initializeNetworkManager();
  setupAutoSave();
  collectSystemInfoAsync();
}

void CrashDialog::setupUI() {
  // 图标区域
  m_iconLabel = new QLabel(this);
  m_iconLabel->setPixmap(QApplication::style()
                             ->standardIcon(QStyle::SP_MessageBoxCritical)
                             .pixmap(64, 64));

  // 主提示文字
  m_mainLabel = new QLabel(
      tr("<h3>Oops! The application has crashed.</h3>"
         "<p>Please help us improve by sending the crash report.</p>"),
      this);
  m_mainLabel->setWordWrap(true);

  // 日志显示区域
  m_logView = new QTextEdit(this);
  m_logView->setReadOnly(true);
  m_logView->setPlainText(tr("Crash summary:\n") +
                          m_fullLog.section('\n', 0, 15)); // 显示前15行

  // 详细信息区域
  m_detailInfo = new QTextEdit(this);
  m_detailInfo->setReadOnly(true);
  m_detailInfo->setVisible(false); // 默认隐藏

  // 操作按钮
  m_detailsBtn = new QToolButton(this);
  m_detailsBtn->setText(tr("Technical Details ▼"));
  m_detailsBtn->setCheckable(true);
  m_detailsBtn->setChecked(false);
  m_detailsBtn->setToolButtonStyle(Qt::ToolButtonTextOnly);

  m_copyBtn = new QPushButton(tr("Copy"), this);
  m_saveBtn = new QPushButton(tr("Save..."), this);
  m_reportBtn = new QPushButton(tr("Send Report"), this);
  m_closeBtn = new QPushButton(tr("Close"), this);

  m_progressBar = new QProgressBar(this);
  m_progressBar->setVisible(false);
  m_progressBar->setRange(0, 100);

  // 布局管理
  QHBoxLayout *topLayout = new QHBoxLayout;
  topLayout->addWidget(m_iconLabel);
  topLayout->addWidget(m_mainLabel, 1);

  QHBoxLayout *btnLayout = new QHBoxLayout;
  btnLayout->addWidget(m_detailsBtn);
  btnLayout->addStretch();
  btnLayout->addWidget(m_copyBtn);
  btnLayout->addWidget(m_saveBtn);
  btnLayout->addWidget(m_reportBtn);
  btnLayout->addWidget(m_closeBtn);

  QVBoxLayout *mainLayout = new QVBoxLayout(this);
  mainLayout->addLayout(topLayout);
  mainLayout->addWidget(m_logView, 1);
  mainLayout->addWidget(m_detailInfo, 2);
  mainLayout->addWidget(m_progressBar);
  mainLayout->addLayout(btnLayout);

  // 信号连接
  connect(m_detailsBtn, &QToolButton::toggled, this,
          &CrashDialog::onDetailsToggled);
  connect(m_copyBtn, &QPushButton::clicked, this, &CrashDialog::onCopyClicked);
  connect(m_saveBtn, &QPushButton::clicked, this, &CrashDialog::onSaveClicked);
  connect(m_reportBtn, &QPushButton::clicked, this, &CrashDialog::onSendReport);
  connect(m_closeBtn, &QPushButton::clicked, this, &QDialog::accept);
}

QString CrashDialog::collectSystemInfo() const {
  QString info;
  QTextStream ts(&info);

  // 添加堆栈跟踪信息
  ts << "== Stack Trace ==\n"
     << QString::fromStdString(CrashHandler::getStackTrace()) << "\n\n"
     << "== Platform Information ==\n"
     << QString::fromStdString(CrashHandler::getPlatformInfo()) << "\n\n"
     << "== System Information ==\n"
     << "Timestamp: " << QDateTime::currentDateTime().toString(Qt::ISODate)
     << "\n"
     << "OS: " << QSysInfo::prettyProductName() << "\n"
     << "Kernel: " << QSysInfo::kernelVersion() << "\n"
     << "Arch: " << QSysInfo::currentCpuArchitecture() << "\n"
     << "App Version: " << QCoreApplication::applicationVersion() << "\n";

  // 扩展硬件信息（平台相关）
#ifdef Q_OS_WIN
  SYSTEM_INFO sysInfo;
  GetNativeSystemInfo(&sysInfo);
  ts << "CPU Cores: " << sysInfo.dwNumberOfProcessors << "\n";
#elif defined(Q_OS_LINUX)
  utsname unameInfo;
  if (uname(&unameInfo) == 0) {
    ts << "Kernel Release: " << unameInfo.release << "\n"
       << "Machine: " << unameInfo.machine << "\n";
  }
#endif

  return info;
}

void CrashDialog::onDetailsToggled(bool checked) {
  m_detailInfo->setVisible(checked);
  m_detailsBtn->setText(checked ? tr("Technical Details ▲")
                                : tr("Technical Details ▼"));
  adjustSize();
}

void CrashDialog::onCopyClicked() {
  QApplication::clipboard()->setText(m_detailInfo->toPlainText());
  QToolTip::showText(m_copyBtn->mapToGlobal(QPoint(0, 0)),
                     tr("Copied to clipboard!"), this);
}

void CrashDialog::onSaveClicked() {
  QString fileName = QFileDialog::getSaveFileName(
      this, tr("Save Crash Report"),
      QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),
      "Log Files (*.log)");

  if (!fileName.isEmpty()) {
    QFile file(fileName);
    if (file.open(QIODevice::WriteOnly)) {
      QTextStream(&file) << m_detailInfo->toPlainText();
      QMessageBox::information(this, tr("Success"),
                               tr("Report saved successfully"));
    }
  }
}

void CrashDialog::onSendReport() {
  // 创建一个 QNetworkRequest 对象
  auto request = QNetworkRequest();
  request.setUrl(QUrl(s_reportServerUrl));
  request.setAttribute(QNetworkRequest::Http2AllowedAttribute, true);
  request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

  QJsonObject reportData;
  reportData["crash_log"] = m_detailInfo->toPlainText();
  reportData["app_version"] = QCoreApplication::applicationVersion();
  reportData["platform"] = QSysInfo::prettyProductName();

  // 添加自定义字段
  for (auto it = s_customFields.constBegin(); it != s_customFields.constEnd();
       ++it) {
    reportData[it.key()] = it.value();
  }

  QJsonDocument doc(reportData);

  m_reportBtn->setEnabled(false);
  m_progressBar->setVisible(true);

  // 发送 POST 请求
  QNetworkReply *reply = m_networkManager->post(request, doc.toJson());

  connect(reply, &QNetworkReply::uploadProgress, this,
          &CrashDialog::onUploadProgress);
  connect(reply, &QNetworkReply::finished,
          [this, reply]() { this->onUploadFinished(reply); });
}

void CrashDialog::onUploadProgress(qint64 sent, qint64 total) {
  if (total > 0) {
    m_progressBar->setValue(static_cast<int>((sent * 100) / total));
  }
}

void CrashDialog::onUploadFinished(QNetworkReply *reply) {
  m_progressBar->setVisible(false);
  m_reportBtn->setEnabled(true);

  if (reply->error() == QNetworkReply::NoError) {
    QMessageBox::information(
        this, tr("Thank You"),
        tr("Crash report submitted successfully. Thank you!"));
    accept();
  } else {
    QMessageBox::critical(
        this, tr("Error"),
        tr("Failed to send report: %1").arg(reply->errorString()));
  }
  reply->deleteLater();
}

void CrashDialog::initializeNetworkManager() {
  m_networkManager = new QNetworkAccessManager(this);
  m_networkManager->setTransferTimeout(30000); // 30秒超时

  // 配置网络缓存
  auto *cache = new QNetworkDiskCache(this);
  QString cacheDir =
      QStandardPaths::writableLocation(QStandardPaths::CacheLocation) +
      "/network-cache";
  cache->setCacheDirectory(cacheDir);
  m_networkManager->setCache(cache);
}

void CrashDialog::setupAutoSave() {
  m_autoSaveTimer = new QTimer(this);
  m_autoSaveTimer->setInterval(60000); // 每分钟
  connect(m_autoSaveTimer, &QTimer::timeout, this, &CrashDialog::onAutoSave);
  m_autoSaveTimer->start();
}

void CrashDialog::collectSystemInfoAsync() {
  auto future = QtConcurrent::run([this]() { return collectSystemInfo(); });

  auto watcher = new QFutureWatcher<QString>(this);
  connect(watcher, &QFutureWatcher<QString>::finished, [this, watcher]() {
    m_detailInfo->setPlainText(watcher->result());
    watcher->deleteLater();
  });
  watcher->setFuture(future);
}

void CrashDialog::onAutoSave() {
  QSettings settings;
  settings.setValue("CrashReport/LastReport", m_detailInfo->toPlainText());
  settings.setValue("CrashReport/Timestamp", QDateTime::currentDateTime());
}

void CrashDialog::setReportServer(const QString &url) {
  s_reportServerUrl = url;
}

void CrashDialog::setCustomFields(const QMap<QString, QString> &fields) {
  s_customFields = fields;
}

void CrashDialog::onSystemInfoUpdate() {
  // 异步更新系统信息
  auto future = QtConcurrent::run([this]() { return collectSystemInfo(); });

  auto watcher = new QFutureWatcher<QString>(this);
  connect(watcher, &QFutureWatcher<QString>::finished, [this, watcher]() {
    m_detailInfo->setPlainText(watcher->result() + "\n\n" + m_fullLog);
    watcher->deleteLater();
  });
  watcher->setFuture(future);
}

void CrashDialog::onCheckUpdates() {
  QNetworkRequest request(QUrl(s_reportServerUrl + "/version"));
  request.setAttribute(QNetworkRequest::Http2AllowedAttribute, true);

  auto reply = m_networkManager->get(request);
  connect(reply, &QNetworkReply::finished, [this, reply]() {
    if (reply->error() == QNetworkReply::NoError) {
      QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
      QJsonObject obj = doc.object();

      QString latestVersion = obj["latest_version"].toString();
      QString currentVersion = QCoreApplication::applicationVersion();

      if (latestVersion > currentVersion) {
        QMessageBox::information(this, tr("Update Available"),
                                 tr("A new version (%1) is available. Please "
                                    "update your application.")
                                     .arg(latestVersion));
      }
    }
    reply->deleteLater();
  });
}