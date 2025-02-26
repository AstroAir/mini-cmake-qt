#include "AppUpdater.h"
#include <QApplication>
#include <QMessageBox>
#include <QProgressDialog>

// 单例实现
AppUpdater& AppUpdater::instance() {
    static AppUpdater instance;
    return instance;
}

AppUpdater::AppUpdater(QObject* parent) : QObject(parent) {
    // 创建更新检查器
    m_checker = new UpdateChecker(this);
    
    // 连接信号槽
    connect(m_checker, &UpdateChecker::updateAvailable,
            this, &AppUpdater::onUpdateInfoReceived);
            
    connect(m_checker, &UpdateChecker::updateNotAvailable,
            this, &AppUpdater::noUpdateAvailable);
            
    connect(m_checker, &UpdateChecker::updateError,
            this, [this](const QString& error) {
                emit updateError(error);
                m_checkInProgress = false;
            });
            
    connect(m_checker, &UpdateChecker::checkProgress,
            this, [this](int progress) {
                emit updateStatus(tr("Checking for updates..."), progress);
            });
            
    // 配置更新定时器
    m_updateTimer.setInterval(86400000); // 24小时
    connect(&m_updateTimer, &QTimer::timeout,
            this, &AppUpdater::onUpdateTimer);
}

AppUpdater::~AppUpdater() {
}

void AppUpdater::initialize(bool autoCheck, QWidget* parent) {
    m_parentWindow = parent;
    
    setupAutoUpdateCheck();
    
    if (autoCheck && shouldRunAutoUpdateCheck()) {
        QTimer::singleShot(5000, this, [this] {
            // 延迟5秒后检查更新，避免影响应用启动速度
            checkForUpdates(false);
        });
    }
}

void AppUpdater::checkForUpdates(bool showProgress, QWidget* parentWindow) {
    if (m_checkInProgress) {
        return; // 避免重复检查
    }
    
    m_checkInProgress = true;
    QWidget* parent = parentWindow ? parentWindow : m_parentWindow;
    
    // 如果需要显示进度对话框
    if (showProgress) {
        QProgressDialog* progressDialog = new QProgressDialog(
            tr("Checking for updates..."),
            tr("Cancel"), 0, 0, parent);
        progressDialog->setWindowModality(Qt::WindowModal);
        progressDialog->setMinimumDuration(500); // 显示前等待500ms
        progressDialog->setAutoClose(true);
        progressDialog->setAutoReset(true);
        
        // 连接取消按钮
        connect(progressDialog, &QProgressDialog::canceled, m_checker, &UpdateChecker::cancelCheck);
        
        // 连接更新结果信号
        auto updateCallback = [progressDialog](bool hasUpdate, const UpdateInfo& info, const QString& error) {
            progressDialog->close();
            progressDialog->deleteLater();
            
            if (!error.isEmpty()) {
                QMessageBox::warning(progressDialog->parentWidget(),
                    tr("Update Error"), error);
            } else if (hasUpdate) {
                // 显示更新对话框 - 这会在UpdateChecker中自动处理
            } else {
                QMessageBox::information(progressDialog->parentWidget(),
                    tr("Software Update"), tr("You're using the latest version."));
            }
        };
        
        // 开始异步检查
        m_checker->asyncCheck(updateCallback);
    } else {
        // 直接检查，不显示UI
        m_checker->checkForUpdates(true);
    }
    
    m_lastCheckTime = QDateTime::currentDateTime();
}

void AppUpdater::showSettingsDialog(QWidget* parent) {
    QWidget* parentWindow = parent ? parent : m_parentWindow;
    UpdateSettingsDialog dialog(parentWindow);
    dialog.exec();
    
    // 可能更新了设置，更新自动检查计划
    setupAutoUpdateCheck();
}

UpdateInfo AppUpdater::latestVersion() const {
    return m_latestVersion;
}

bool AppUpdater::hasUpdate() const {
    return m_hasUpdate;
}

void AppUpdater::setupAutoUpdateCheck() {
    const auto& config = UpdateConfig::instance();
    
    if (config.isAutoCheckEnabled()) {
        // 计算检查间隔，转换为毫秒
        qint64 interval = static_cast<qint64>(config.checkFrequency()) * 24 * 60 * 60 * 1000;
        m_updateTimer.setInterval(interval);
        m_updateTimer.start();
    } else {
        m_updateTimer.stop();
    }
}

bool AppUpdater::shouldRunAutoUpdateCheck() const {
    const auto& config = UpdateConfig::instance();
    
    // 如果自动更新被禁用，不检查
    if (!config.isAutoCheckEnabled()) {
        return false;
    }
    
    // 获取上次检查时间
    auto lastCheck = config.lastCheckTime();
    if (!lastCheck.isValid()) {
        return true; // 从未检查过，应该检查
    }
    
    // 检查间隔是否已经过去
    int daysSinceLastCheck = lastCheck.daysTo(QDateTime::currentDateTime());
    return daysSinceLastCheck >= config.checkFrequency();
}

void AppUpdater::onUpdateTimer() {
    // 定时器触发时检查更新
    if (shouldRunAutoUpdateCheck()) {
        checkForUpdates(false);
    }
}

void AppUpdater::onUpdateInfoReceived(const UpdateInfo& info) {
    m_latestVersion = info;
    m_hasUpdate = true;
    m_checkInProgress = false;
    emit updateAvailable(info);
    
    // 如果有父窗口，自动显示更新对话框
    if (m_parentWindow) {
        UpdateDialog* dialog = new UpdateDialog(info, m_parentWindow);
        dialog->setAttribute(Qt::WA_DeleteOnClose);
        dialog->show();
    }
}
