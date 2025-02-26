#ifndef UPDATEDIALOG_H
#define UPDATEDIALOG_H

#include <QDialog>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QTextBrowser>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QFile>
#include "UpdateCheck.h"

/**
 * @brief 更新对话框
 * 
 * 显示更新信息并提供下载和安装选项
 */
class UpdateDialog : public QDialog {
    Q_OBJECT
    
public:
    explicit UpdateDialog(const UpdateInfo& info, QWidget* parent = nullptr);
    ~UpdateDialog();
    
private slots:
    void onDownloadUpdate();
    void onSkipVersion();
    void onRemindLater();
    void onDownloadProgress(qint64 bytesReceived, qint64 bytesTotal);
    void onDownloadFinished();
    void onInstallUpdate();
    void onOpenDownloadFolder();
    
private:
    void setupUi();
    void startDownload();
    void setupDownloadUI();
    void setupCompleteUI();
    bool runInstaller();
    QString formatFileSize(qint64 size) const;
    
    UpdateInfo m_updateInfo;
    QNetworkAccessManager m_network;
    QNetworkReply* m_currentReply{nullptr};
    QFile* m_outputFile{nullptr};
    QString m_downloadPath;
    bool m_downloadComplete{false};
    
    QLabel* m_titleLabel{nullptr};
    QLabel* m_versionLabel{nullptr};
    QTextBrowser* m_releaseNotes{nullptr};
    QLabel* m_statusLabel{nullptr};
    QProgressBar* m_progressBar{nullptr};
    QPushButton* m_downloadBtn{nullptr};
    QPushButton* m_skipBtn{nullptr};
    QPushButton* m_laterBtn{nullptr};
    QPushButton* m_installBtn{nullptr};
    QPushButton* m_openFolderBtn{nullptr};
};

#endif // UPDATEDIALOG_H
