#pragma once

#include <QDialog>
#include <QNetworkReply>
#include <QSysInfo>
#include <QTextEdit>


class QPushButton;
class QLabel;
class QProgressBar;
class QToolButton;

class CrashDialog : public QDialog {
  Q_OBJECT
public:
  explicit CrashDialog(const QString &log, QWidget *parent = nullptr);

private slots:
  void onSendReport();
  void onCopyClicked();
  void onSaveClicked();
  void onDetailsToggled(bool checked);
  void onUploadProgress(qint64 sent, qint64 total);
  void onUploadFinished(QNetworkReply *reply);

private:
  void setupUI();
  QString collectSystemInfo() const;

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
};