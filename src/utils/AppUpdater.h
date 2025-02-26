#ifndef APPUPDATER_H
#define APPUPDATER_H

#include <QObject>
#include <QTimer>
#include <QDateTime>
#include <memory>
#include "../ui/UpdateCheck.h"

/**
 * @brief 应用更新管理类
 * 
 * 该类管理应用更新的全部流程，包括：
 * 1. 检查更新
 * 2. 自动/手动更新管理
 * 3. 下载和安装更新
 * 4. 提供应用生命周期内的更新状态
 */
class AppUpdater : public QObject {
  Q_OBJECT
  
public:
  /**
   * @brief 获取全局单例实例
   * @return AppUpdater的单例引用
   */
  static AppUpdater& instance();

  /**
   * @brief 初始化更新系统
   * @param autoCheck 是否在初始化时自动检查更新
   * @param parent 父对象
   */
  void initialize(bool autoCheck = true, QWidget* parent = nullptr);
  
  /**
   * @brief 手动检查更新
   * @param showProgress 是否显示进度对话框
   * @param parentWindow 父窗口
   */
  void checkForUpdates(bool showProgress = true, QWidget* parentWindow = nullptr);
  
  /**
   * @brief 显示更新设置对话框
   * @param parent 父窗口
   */
  void showSettingsDialog(QWidget* parent = nullptr);
  
  /**
   * @brief 获取最新版本信息
   * @return 最新版本信息，如果没有则返回空对象
   */
  UpdateInfo latestVersion() const;
  
  /**
   * @brief 是否有可用更新
   * @return 是否有新版本可用
   */
  bool hasUpdate() const;

signals:
  /**
   * @brief 发现新版本时发出的信号
   * @param info 更新信息
   */
  void updateAvailable(const UpdateInfo& info);
  
  /**
   * @brief 无可用更新时发出的信号
   */
  void noUpdateAvailable();
  
  /**
   * @brief 更新检查失败时发出的信号
   * @param error 错误信息
   */
  void updateError(const QString& error);
  
  /**
   * @brief 更新过程状态变化时发出的信号
   * @param status 状态描述
   * @param progress 进度 (0-100)，-1表示不确定
   */
  void updateStatus(const QString& status, int progress);

private:
  explicit AppUpdater(QObject* parent = nullptr);
  ~AppUpdater();
  
  // 禁止复制
  AppUpdater(const AppUpdater&) = delete;
  AppUpdater& operator=(const AppUpdater&) = delete;
  
  // 定时检查
  void setupAutoUpdateCheck();
  
  // 检查是否该运行自动更新
  bool shouldRunAutoUpdateCheck() const;
  
private slots:
  void onUpdateTimer();
  void onUpdateInfoReceived(const UpdateInfo& info);
  
private:
  QTimer m_updateTimer;
  UpdateChecker* m_checker{nullptr};
  UpdateInfo m_latestVersion;
  bool m_hasUpdate{false};
  QDateTime m_lastCheckTime;
  QWidget* m_parentWindow{nullptr};
  bool m_checkInProgress{false};
};

#endif // APPUPDATER_H
