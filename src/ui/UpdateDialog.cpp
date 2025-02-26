#include "UpdateDialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDesktopServices>
#include <QMessageBox>
#include <QDir>
#include <QStandardPaths>
#include <QProcess>
#include <QFileDialog>

UpdateDialog::UpdateDialog(const UpdateInfo& info, QWidget* parent)
    : QDialog(parent),
      m_updateInfo(info)
{
    setWindowTitle(tr("Software Update"));
    setMinimumSize(500, 400);
    
    setupUi();
    
    // 如果是强制更新，禁用跳过和稍后提醒按钮
    if (m_updateInfo.isMandatory) {
        m_skipBtn->setEnabled(false);
        m_laterBtn->setEnabled(false);
    }
}

UpdateDialog::~UpdateDialog() {
    if (m_currentReply) {
        m_currentReply->abort();
        m_currentReply->deleteLater();
        m_currentReply = nullptr;
    }
    
    if (m_outputFile && m_outputFile->isOpen()) {
        m_outputFile->close();
        delete m_outputFile;
        m_outputFile = nullptr;
    }
}

void UpdateDialog::setupUi() {
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    
    // 标题
    m_titleLabel = new QLabel(tr("A new version is available!"), this);
    QFont titleFont = m_titleLabel->font();
    titleFont.setPointSize(titleFont.pointSize() + 2);
    titleFont.setBold(true);
    m_titleLabel->setFont(titleFont);
    mainLayout->addWidget(m_titleLabel);
    
    // 版本信息
    m_versionLabel = new QLabel(
        tr("Version %1 is available - you have %2")
            .arg(m_updateInfo.version)
            .arg(qApp->applicationVersion()),
        this);
    mainLayout->addWidget(m_versionLabel);
    
    // 分隔线
    QFrame* line = new QFrame(this);
    line->setFrameShape(QFrame::HLine);
    line->setFrameShadow(QFrame::Sunken);
    mainLayout->addWidget(line);
    
    // 更新说明
    QLabel* notesLabel = new QLabel(tr("Release Notes:"), this);
    mainLayout->addWidget(notesLabel);
    
    m_releaseNotes = new QTextBrowser(this);
    m_releaseNotes->setOpenExternalLinks(true);
    m_releaseNotes->setHtml(m_updateInfo.releaseNotes);
    mainLayout->addWidget(m_releaseNotes);
    
    // 下载状态和进度
    m_statusLabel = new QLabel(this);
    m_statusLabel->setText(
        tr("Size: %1").arg(formatFileSize(m_updateInfo.fileSize)));
    mainLayout->addWidget(m_statusLabel);
    
    m_progressBar = new QProgressBar(this);
    m_progressBar->setVisible(false);
    mainLayout->addWidget(m_progressBar);
    
    // 按钮区域
    QHBoxLayout* btnLayout = new QHBoxLayout();
    
    m_skipBtn = new QPushButton(tr("Skip This Version"), this);
    m_laterBtn = new QPushButton(tr("Remind Me Later"), this);
    m_downloadBtn = new QPushButton(tr("Download"), this);
    m_installBtn = new QPushButton(tr("Install Now"), this);
    m_openFolderBtn = new QPushButton(tr("Show in Folder"), this);
    
    m_downloadBtn->setDefault(true);
    
    // 按钮布局
    btnLayout->addWidget(m_skipBtn);
    btnLayout->addWidget(m_laterBtn);
    btnLayout->addStretch();
    btnLayout->addWidget(m_downloadBtn);
    
    // 安装和打开文件夹按钮初始隐藏
    m_installBtn->setVisible(false);
    m_openFolderBtn->setVisible(false);
    
    btnLayout->addWidget(m_installBtn);
    btnLayout->addWidget(m_openFolderBtn);
    
    mainLayout->addLayout(btnLayout);
    
    // 连接信号
    connect(m_downloadBtn, &QPushButton::clicked, this, &UpdateDialog::onDownloadUpdate);
    connect(m_skipBtn, &QPushButton::clicked, this, &UpdateDialog::onSkipVersion);
    connect(m_laterBtn, &QPushButton::clicked, this, &UpdateDialog::onRemindLater);
    connect(m_installBtn, &QPushButton::clicked, this, &UpdateDialog::onInstallUpdate);
    connect(m_openFolderBtn, &QPushButton::clicked, this, &UpdateDialog::onOpenDownloadFolder);
    
    // 检查是否已经下载过该版本
    QString downloadDir = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
    QString filename = QFileInfo(m_updateInfo.downloadUrl).fileName();
    m_downloadPath = downloadDir + QDir::separator() + filename;
    
    if (QFile::exists(m_downloadPath)) {
        QFileInfo fileInfo(m_downloadPath);
        if (fileInfo.size() == m_updateInfo.fileSize) {
            m_downloadComplete = true;
            setupCompleteUI();
        }
    }
}

void UpdateDialog::onDownloadUpdate() {
    startDownload();
}

void UpdateDialog::onSkipVersion() {
    // 记录已跳过的版本
    UpdateConfig::instance().setSkippedVersion(m_updateInfo.version);
    accept();
}

void UpdateDialog::onRemindLater() {
    // 只是关闭窗口，下次检查时还会提示
    reject();
}

void UpdateDialog::startDownload() {
    setupDownloadUI();
    
    // 准备下载文件
    QString downloadDir = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
    QDir().mkpath(downloadDir); // 确保文件夹存在
    
    QString filename = QFileInfo(m_updateInfo.downloadUrl).fileName();
    if (filename.isEmpty()) {
        filename = QString("%1-%2-update.exe")
                  .arg(qApp->applicationName())
                  .arg(m_updateInfo.version);
    }
    
    m_downloadPath = downloadDir + QDir::separator() + filename;
    
    // 创建输出文件
    m_outputFile = new QFile(m_downloadPath);
    if (!m_outputFile->open(QIODevice::WriteOnly)) {
        QMessageBox::critical(this, tr("Download Error"),
            tr("Could not open output file for writing: %1").arg(m_outputFile->errorString()));
        delete m_outputFile;
        m_outputFile = nullptr;
        return;
    }
    
    // 创建网络请求
    QNetworkRequest request(m_updateInfo.downloadUrl);
    m_currentReply = m_network.get(request);
    
    // 连接信号
    connect(m_currentReply, &QNetworkReply::downloadProgress, 
            this, &UpdateDialog::onDownloadProgress);
    connect(m_currentReply, &QNetworkReply::finished, 
            this, &UpdateDialog::onDownloadFinished);
    connect(m_currentReply, QOverload<QNetworkReply::NetworkError>::of(&QNetworkReply::error),
            this, &UpdateDialog::onDownloadError);
}

void UpdateDialog::onDownloadProgress(qint64 bytesReceived, qint64 bytesTotal) {
    // 更新进度条
    if (bytesTotal > 0) {
        m_progressBar->setValue(static_cast<int>(bytesReceived * 100 / bytesTotal));
        m_statusLabel->setText(tr("Downloading: %1 of %2")
            .arg(formatFileSize(bytesReceived))
            .arg(formatFileSize(bytesTotal)));
    } else {
        m_progressBar->setRange(0, 0); // 未知进度
        m_statusLabel->setText(tr("Downloading..."));
    }
}

void UpdateDialog::onDownloadFinished() {
    // 检查是否有错误
    if (m_currentReply->error() != QNetworkReply::NoError) {
        m_statusLabel->setText(tr("Download failed: %1").arg(m_currentReply->errorString()));
        m_progressBar->setValue(0);
        
        // 重新显示下载按钮
        m_downloadBtn->setVisible(true);
        m_downloadBtn->setText(tr("Retry Download"));
    } else {
        // 下载成功
        if (m_outputFile) {
            m_outputFile->write(m_currentReply->readAll());
            m_outputFile->close();
            delete m_outputFile;
            m_outputFile = nullptr;
            
            m_downloadComplete = true;
            setupCompleteUI();
        }
    }
    
    m_currentReply->deleteLater();
    m_currentReply = nullptr;
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

void UpdateDialog::setupDownloadUI() {
    m_downloadBtn->setEnabled(false);
    m_skipBtn->setEnabled(false);
    m_laterBtn->setEnabled(false);
    m_progressBar->setValue(0);
    m_progressBar->setVisible(true);
    m_statusLabel->setText(tr("Preparing download..."));
}

void UpdateDialog::setupCompleteUI() {
    m_downloadBtn->setVisible(false);
    m_skipBtn->setVisible(false);
    m_laterBtn->setVisible(false);
    m_progressBar->setValue(100);
    
    m_installBtn->setVisible(true);
    m_openFolderBtn->setVisible(true);
    m_installBtn->setDefault(true);
    
    m_statusLabel->setText(tr("Download complete. Ready to install!"));
}

void UpdateDialog::onInstallUpdate() {
    if (runInstaller()) {
        // 如果安装程序成功启动，关闭此对话框
        accept();
    }
}

void UpdateDialog::onOpenDownloadFolder() {
    QFileInfo fileInfo(m_downloadPath);
    QDesktopServices::openUrl(QUrl::fromLocalFile(fileInfo.absolutePath()));
}

QString UpdateDialog::formatFileSize(qint64 size) const {
    const double KB = 1024;
    const double MB = KB * 1024;
    const double GB = MB * 1024;
    
    if (size < KB) {
        return tr("%1 bytes").arg(size);
    } else if (size < MB) {
        return tr("%1 KB").arg(size / KB, 0, 'f', 2);
    } else if (size < GB) {
        return tr("%1 MB").arg(size / MB, 0, 'f', 2);
    } else {
        return tr("%1 GB").arg(size / GB, 0, 'f', 2);
    }
}

bool UpdateDialog::runInstaller() {
    if (!QFile::exists(m_downloadPath)) {
        QMessageBox::critical(this, tr("Installation Error"), 
            tr("The installer file is missing: %1").arg(m_downloadPath));
        return false;
    }
    
    // 基于操作系统决定如何运行安装程序
#ifdef Q_OS_WIN
    // 在Windows上直接执行安装程序
    if (!QProcess::startDetached(m_downloadPath, QStringList())) {
        QMessageBox::critical(this, tr("Installation Error"),
            tr("Failed to start the installer."));
        return false;
    }
#else
    // 在其他操作系统上，可能需要不同的处理
    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(m_downloadPath))) {
        QMessageBox::critical(this, tr("Installation Error"),
            tr("Failed to open the installer."));
        return false;
    }
#endif

    return true;
}