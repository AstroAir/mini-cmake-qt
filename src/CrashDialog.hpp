#pragma once

#include <QDialog>
#include <QNetworkReply>
#include <QSysInfo>
#include <QTextEdit>
#include <QTimer>
#include <QFuture>
#include <QNetworkAccessManager>

class QPushButton;
class QLabel;
class QProgressBar;
class QToolButton;

class CrashDialog : public QDialog {
  Q_OBJECT
public:
  explicit CrashDialog(const QString &log, QWidget *parent = nullptr);
  static void setReportServer(const QString& url);
  static void setCustomFields(const QMap<QString, QString>& fields);

private slots:
  void onSendReport();
  void onCopyClicked();
  void onSaveClicked();
  void onDetailsToggled(bool checked);
  void onUploadProgress(qint64 sent, qint64 total);
  void onUploadFinished(QNetworkReply *reply);
  void onAutoSave();
  void onSystemInfoUpdate();
  void onCheckUpdates();

private:
  void setupUI();
  QString collectSystemInfo() const;
  void initializeNetworkManager();
  void setupAutoSave();
  void collectSystemInfoAsync();

  QLabel *m_iconLabel;
  QLabel *m_mainLabel;
  QTextEdit *m_logView;
  QTextEdit *m_detailInfo;
  QPushButton *m_reportBtn;
  QPushButton *m_closeBtn;
  QPushButton *m_copyBtn;
  QPushButton *m_saveBtn;
  QToolButton *m_detailsBtn;
  QProgressBar *m_progressBar;
  QString m_fullLog;
  QTimer* m_autoSaveTimer;
  QNetworkAccessManager* m_networkManager;
  QFuture<QString> m_systemInfoFuture;
  static QString s_reportServerUrl;
  static QMap<QString, QString> s_customFields;
};