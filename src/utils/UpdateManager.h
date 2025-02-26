#ifndef UPDATEMANAGER_H
#define UPDATEMANAGER_H

#include <QObject>
#include <QTimer>
#include <QDateTime>
#include <memory>
#include "../ui/UpdateCheck.h"
#include <QApplication>
#include <QMessageBox>
#include <QProgressDialog>
#include "../ui/UpdateDialog.h"

/**
 * @brief 更新管理器类
 * 
 * 这个类提供了应用程序更新系统的主要接口，包含自动更新检查、
 * 更新通知以及下载与安装更新的功能。
 */
class UpdateManager : public QObject {
    Q_OBJECT
    
public:
    static UpdateManager& instance();
    
    /**
     * @brief 初始化更新系统
     * 
     * @param autoCheck 是否在启动时自动检查更新
     * @param parent 父窗口，用于显示更新对话框
     */
    void initialize(bool autoCheck = true, QWidget* parent = nullptr);
    
    /**
     * @brief 检查更新
     * 
     * @param showUI 是否显示用户界面
     * @param parent 父窗口，用于显示更新对话框
     */
    void checkForUpdates(bool showUI = true, QWidget* parent = nullptr);
    
    /**
     * @brief 显示更新设置对话框
     * 
     * @param parent 父窗口
     */
    void showSettings(QWidget* parent = nullptr);
    
    /**
     * @brief 获取最新版本信息
     * 
     * @return 最新版本的信息
     */
    UpdateInfo getLatestVersion() const;
    
    /**
     * @brief 是否有新版本可用
     * 
     * @return 是否有新版本
     */
    bool hasUpdate() const;
    
    /**
     * @brief 设置更新检查频率
     * 
     * @param days 检查频率（天）
     */
    void setCheckFrequency(int days);
    
    /**
     * @brief 启用或禁用自动更新检查
     * 
     * @param enable 是否启用
     */
    void enableAutoCheck(bool enable);
    
signals:
    /**
     * @brief 发现更新时发出的信号
     * 
     * @param info 更新信息
     */
    void updateAvailable(const UpdateInfo& info);
    
    /**
     * @brief 无更新可用时发出的信号
     */
    void noUpdateAvailable();
    
    /**
     * @brief 更新检查错误时发出的信号
     * 
     * @param error 错误信息
     */
    void updateError(const QString& error);
    
    /**
     * @brief 更新状态变化时发出的信号
     * 
     * @param message 状态信息
     * @param progress 进度（0-100）
     */
    void updateStatusChanged(const QString& message, int progress);
    
private slots:
    void onUpdateTimer();
    void handleUpdateResult(bool hasUpdate, const UpdateInfo& info, const QString& error);
    
private:
    explicit UpdateManager(QObject* parent = nullptr);
    ~UpdateManager();
    
    void setupAutoUpdateCheck();
    bool shouldCheckForUpdates() const;
    
    // 禁止复制
    UpdateManager(const UpdateManager&) = delete;
    UpdateManager& operator=(const UpdateManager&) = delete;
    
    QTimer m_updateTimer;
    UpdateChecker* m_checker{nullptr};
    UpdateInfo m_latestVersion;
    bool m_hasUpdate{false};
    QWidget* m_parentWindow{nullptr};
    bool m_checkInProgress{false};
};

#endif // UPDATEMANAGER_H

#include "UpdateManager.h"

UpdateManager& UpdateManager::instance() {
    static UpdateManager instance;
    return instance;
}

UpdateManager::UpdateManager(QObject* parent) : QObject(parent) {
    // 初始化更新检查器
    m_checker = new UpdateChecker(this);
    
    // 连接信号
    connect(m_checker, &UpdateChecker::updateAvailable,
            this, [this](const UpdateInfo& info) {
                m_latestVersion = info;
                m_hasUpdate = true;
                m_checkInProgress = false;
                emit updateAvailable(info);
            });
            
    connect(m_checker, &UpdateChecker::updateNotAvailable,
            this, [this]() {
                m_hasUpdate = false;
                m_checkInProgress = false;
                emit noUpdateAvailable();
            });
            
    connect(m_checker, &UpdateChecker::updateError,
            this, [this](const QString& error) {
                m_checkInProgress = false;
                emit updateError(error);
            });
            
    connect(m_checker, &UpdateChecker::checkProgress,
            this, [this](int progress) {
                emit updateStatusChanged(tr("Checking for updates..."), progress);
            });
            
    // 设置更新计时器
    m_updateTimer.setSingleShot(false);
    connect(&m_updateTimer, &QTimer::timeout, this, &UpdateManager::onUpdateTimer);
}

UpdateManager::~UpdateManager() {
    // 清理资源
}

void UpdateManager::initialize(bool autoCheck, QWidget* parent) {
    m_parentWindow = parent;
    
    setupAutoUpdateCheck();
    
    if (autoCheck && shouldCheckForUpdates()) {
        // 延迟几秒钟检查，避免影响应用启动速度
        QTimer::singleShot(3000, this, [this]() {
            checkForUpdates(false, m_parentWindow);
        });
    }
}

void UpdateManager::checkForUpdates(bool showUI, QWidget* parent) {
    if (m_checkInProgress) {
        return;
    }
    
    m_checkInProgress = true;
    QWidget* parentWindow = parent ? parent : m_parentWindow;
    
    if (showUI) {
        // 显示进度对话框
        QProgressDialog* progressDialog = new QProgressDialog(
            tr("Checking for updates..."),
            tr("Cancel"),
            0, 100,
            parentWindow);
        progressDialog->setWindowTitle(tr("Software Update"));
        progressDialog->setWindowModality(Qt::WindowModal);
        progressDialog->setMinimumDuration(500);
        
        // 连接取消按钮
        connect(progressDialog, &QProgressDialog::canceled, 
                m_checker, &UpdateChecker::cancelCheck);
        
        // 连接进度更新
        connect(m_checker, &UpdateChecker::checkProgress,
                progressDialog, &QProgressDialog::setValue);
                
        // 使用异步检查和回调
        m_checker->asyncCheck([this, progressDialog, parentWindow](
            bool hasUpdate, const UpdateInfo& info, const QString& error) {
            
            progressDialog->close();
            progressDialog->deleteLater();
            
            this->handleUpdateResult(hasUpdate, info, error);
            
            if (!error.isEmpty()) {
                QMessageBox::warning(parentWindow, tr("Update Error"), error);
            } else if (hasUpdate) {
                // 显示更新对话框
                UpdateDialog* dialog = new UpdateDialog(info, parentWindow);
                dialog->setAttribute(Qt::WA_DeleteOnClose);
                dialog->show();
            } else if (progressDialog->wasCanceled()) {
                // 用户取消了检查，不显示任何消息
            } else {
                QMessageBox::information(parentWindow, 
                    tr("Software Update"), 
                    tr("You are using the latest version."));
            }
        });
    } else {
        // 静默检查
        m_checker->checkForUpdates(true);
    }
}

void UpdateManager::showSettings(QWidget* parent) {
    QWidget* parentWindow = parent ? parent : m_parentWindow;
    
    UpdateSettingsDialog dialog(parentWindow);
    dialog.exec();
    
    // 可能已经更改了设置，重新设置自动更新检查
    setupAutoUpdateCheck();
}

UpdateInfo UpdateManager::getLatestVersion() const {
    return m_latestVersion;
}

bool UpdateManager::hasUpdate() const {
    return m_hasUpdate;
}

void UpdateManager::setCheckFrequency(int days) {
    UpdateConfig& config = UpdateConfig::instance();
    config.setCheckFrequency(days);
    config.save();
    
    setupAutoUpdateCheck();
}

void UpdateManager::enableAutoCheck(bool enable) {
    UpdateConfig& config = UpdateConfig::instance();
    config.setAutoCheckEnabled(enable);
    config.save();
    
    setupAutoUpdateCheck();
}

void UpdateManager::onUpdateTimer() {
    if (shouldCheckForUpdates()) {
        checkForUpdates(false);
    }
}

void UpdateManager::handleUpdateResult(bool hasUpdate, const UpdateInfo& info, const QString& error) {
    if (!error.isEmpty()) {
        emit updateError(error);
    } else if (hasUpdate) {
        m_latestVersion = info;
        m_hasUpdate = true;
        emit updateAvailable(info);
    } else {
        m_hasUpdate = false;
        emit noUpdateAvailable();
    }
}

void UpdateManager::setupAutoUpdateCheck() {
    const auto& config = UpdateConfig::instance();
    
    if (config.isAutoCheckEnabled()) {
        // 计算检查间隔（毫秒）
        int interval = config.checkFrequency() * 24 * 60 * 60 * 1000;
        m_updateTimer.setInterval(interval);
        m_updateTimer.start();
    } else {
        m_updateTimer.stop();
    }
}

bool UpdateManager::shouldCheckForUpdates() const {
    const auto& config = UpdateConfig::instance();
    
    // 如果自动检查已禁用，则不检查
    if (!config.isAutoCheckEnabled()) {
        return false;
    }
    
    // 获取上次检查时间
    auto lastCheck = config.lastCheckTime();
    if (!lastCheck.isValid()) {
        return true; // 从未检查过，应该检查
    }
    
    // 检查时间是否已超过频率设置
    int daysPassed = lastCheck.daysTo(QDateTime::currentDateTime());
    return daysPassed >= config.checkFrequency();
}
