#pragma once

#include <QDialog>
#include <QFuture>
#include <QNetworkReply>
#include <QSysInfo>

class ElaPushButton;
class ElaText;
class ElaProgressBar;
class ElaToolButton;
class ElaPlainTextEdit;

class CrashDialog : public QDialog {
  Q_OBJECT
public:
  explicit CrashDialog(const QString &log, QWidget *parent = nullptr);
  static void setReportServer(const QString &url);
  static void setCustomFields(const QMap<QString, QString> &fields);

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

  ElaText *m_iconLabel;
  ElaText *m_mainLabel;
  ElaPlainTextEdit *m_logView;
  ElaPlainTextEdit *m_detailInfo;
  ElaPushButton *m_reportBtn;
  ElaPushButton *m_closeBtn;
  ElaPushButton *m_copyBtn;
  ElaPushButton *m_saveBtn;
  ElaToolButton *m_detailsBtn;
  ElaProgressBar *m_progressBar;
  QString m_fullLog;
  QTimer *m_autoSaveTimer;
  QNetworkAccessManager *m_networkManager;
  QFuture<QString> m_systemInfoFuture;
  static QString s_reportServerUrl;
  static QMap<QString, QString> s_customFields;
};