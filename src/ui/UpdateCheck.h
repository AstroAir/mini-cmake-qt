#ifndef UPDATECHECK_H
#define UPDATECHECK_H

#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDialog>
#include <QFile>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QObject>
#include <QProgressBar>
#include <QPropertyAnimation>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QTextBrowser>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>
#include <functional>

// Result 类代替 std::expected
template <typename T> class Result {
public:
  Result(const T &value) : m_value(value), m_isError(false) {}
  Result(const QString &error) : m_error(error), m_isError(true) {}

  bool hasError() const { return m_isError; }
  const QString &error() const { return m_error; }
  const T &value() const { return m_value; }

private:
  T m_value;
  QString m_error;
  bool m_isError;
};

/**
 * @brief 更新信息结构体
 *
 * 包含可用更新的完整信息
 */
struct UpdateInfo {
  QString version;                // 版本号
  QString releaseDate;            // 发布日期
  QString downloadUrl;            // 下载URL
  QString releaseNotes;           // 更新说明
  QString changelog;              // 更新日志（与releaseNotes同义）
  qint64 fileSize{0};             // 文件大小（字节）
  QString minOSVersion;           // 最小操作系统版本
  bool isMandatory{false};        // 是否强制更新
  QStringList affectedComponents; // 受影响的组件
  QString sha256Checksum;         // SHA256校验和

  bool isValid() const { return !version.isEmpty() && !downloadUrl.isEmpty(); }

  static UpdateInfo fromJson(const QJsonDocument &json);
  static UpdateInfo fromJson(const QJsonObject &json);
  QJsonObject toJson() const;
};

/**
 * @brief 更新配置管理类
 *
 * 管理更新相关的持久化配置
 */
class UpdateConfig : public QObject {
  Q_OBJECT

public:
  /**
   * @brief 获取全局单例实例
   */
  static UpdateConfig &instance();

  /**
   * @brief 是否启用自动检查更新
   */
  bool isAutoCheckEnabled() const;
  void setAutoCheckEnabled(bool enabled);

  /**
   * @brief 获取/设置检查频率（天）
   */
  int checkFrequency() const;
  void setCheckFrequency(int days);

  /**
   * @brief 获取/设置上次检查时间
   */
  QDateTime lastCheckTime() const;
  void setLastCheckTime(const QDateTime &dt);

  /**
   * @brief 获取/设置跳过的版本
   */
  QString skippedVersion() const;
  void setSkippedVersion(const QString &version);

  /**
   * @brief 获取/设置更新通道
   */
  QString updateChannel() const;
  void setUpdateChannel(const QString &channel);

  /**
   * @brief 获取更新URL
   */
  QString getUpdateUrl() const;

  /**
   * @brief 设置自定义更新URL
   */
  void setCustomUpdateUrl(const QString &url);

  /**
   * @brief 获取/设置忽略的版本
   */
  QString ignoredVersion() const;
  void setIgnoredVersion(const QString &version);

  /**
   * @brief 加载和保存配置
   */
  void load();
  void save();

private:
  explicit UpdateConfig(QObject *parent = nullptr);

  QSettings m_settings;
  bool m_autoCheck{true};
  bool m_autoCheckEnabled{true};
  int m_checkFrequency{7}; // 默认7天
  QDateTime m_lastCheckTime;
  QString m_skippedVersion;
  QString m_updateChannel{"stable"};
  QString m_customUpdateUrl;
  QString m_ignoredVersion;
};

/**
 * @brief 更新检查器类
 *
 * 处理版本检查和更新信息获取
 */
class UpdateChecker : public QObject {
  Q_OBJECT

public:
  explicit UpdateChecker(QObject *parent = nullptr);
  ~UpdateChecker();

  /**
   * @brief 检查更新
   * @param silent 是否在后台静默检查
   */
  void checkForUpdates(bool silent = false);

  /**
   * @brief 异步检查更新并通过回调返回结果
   * @param callback 结果回调函数
   */
  void asyncCheck(
      std::function<void(bool, const UpdateInfo &, const QString &)> callback);

  /**
   * @brief 取消检查
   */
  void cancelCheck();

signals:
  /**
   * @brief 发现更新时发出的信号
   */
  void updateAvailable(const UpdateInfo &info);

  /**
   * @brief 无可用更新时发出的信号
   */
  void updateNotAvailable();

  /**
   * @brief 更新检查失败时发出的信号
   */
  void updateError(const QString &error);

  /**
   * @brief 检查进度信号
   */
  void checkProgress(int percentage);

private slots:
  void onNetworkReply(QNetworkReply *reply);
  void onUpdateInfoReceived(QNetworkReply *reply);
  void onDownloadProgress(qint64 bytesReceived, qint64 bytesTotal);
  void handleNetworkError(QNetworkReply::NetworkError code);

private:
  bool compareVersions(const QString &newVer, const QString &currentVer);
  QString getCurrentVersion() const;
  QUrl getUpdateUrl() const;
  bool shouldCheck() const;
  void scheduleRetry();
  Result<UpdateInfo> parseUpdateInfo(const QByteArray &data) noexcept;

  QNetworkAccessManager m_network;
  QNetworkAccessManager *manager;
  QNetworkReply *m_currentReply{nullptr};
  bool m_silent{false};
  std::function<void(bool, const UpdateInfo &, const QString &)> m_callback{
      nullptr};
  QTimer m_retryTimer;
  int m_retryCount{0};
  int m_maxRetries{3};
};

/**
 * @brief 更新设置对话框
 */
class UpdateSettingsDialog : public QDialog {
  Q_OBJECT

public:
  explicit UpdateSettingsDialog(QWidget *parent = nullptr);

private slots:
  void onSaveSettings();
  void onCheckNowClicked();
  void onAccepted();

private:
  void setupUI();
  void loadSettings();
  void saveSettings();

  QCheckBox *m_autoCheckBox{nullptr};
  QComboBox *m_frequencyCombo{nullptr};
  QSpinBox *m_frequencySpinBox{nullptr};
  QComboBox *m_channelCombo{nullptr};
  QLineEdit *m_customUrlEdit{nullptr};
};

/**
 * @brief 更新对话框
 */
class UpdateDialog : public QDialog {
  Q_OBJECT

public:
  explicit UpdateDialog(const UpdateInfo &updateInfo,
                        QWidget *parent = nullptr);
  ~UpdateDialog();

  // 便捷静态方法显示更新对话框
  static bool showUpdateDialogIfAvailable(QWidget *parent = nullptr);

  // 枚举对话框状态
  enum class DialogState { Information, Downloading, DownloadComplete, Error };

protected:
  void closeEvent(QCloseEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;

private slots:
  void onIgnoreClicked();
  void onLaterClicked();
  void onUpdateClicked();
  void onDownloadProgress(qint64 bytesReceived, qint64 bytesTotal);
  void onDownloadFinished();
  void onDownloadError(QNetworkReply::NetworkError error);
  void onAutoCheckChanged(int state);
  void onChannelChanged(const QString &channel);

private:
  void setupUI(const UpdateInfo &info);
  QString formatChangelog(const QString &raw) const;
  void startDownload();
  void loadUserPreferences();
  void saveUserPreferences();
  void adjustLayoutForScreenSize();
  void updateDialogState(DialogState state);

  UpdateInfo m_updateInfo;
  DialogState m_currentState{DialogState::Information};

  QVBoxLayout *m_mainLayout{nullptr};
  QLabel *m_titleLabel{nullptr};
  QTextBrowser *m_changelogBrowser{nullptr};
  QProgressBar *m_downloadProgress{nullptr};
  QPushButton *m_ignoreButton{nullptr};
  QPushButton *m_laterButton{nullptr};
  QPushButton *m_updateButton{nullptr};
  QCheckBox *m_autoCheckBox{nullptr};
  QComboBox *m_channelComboBox{nullptr};
  QComboBox *m_frequencySpinBox{nullptr};
  QLineEdit *m_customUrlEdit{nullptr};
  QPropertyAnimation *m_animation{nullptr};

  QNetworkAccessManager m_networkManager;
  QScopedPointer<QNetworkReply> m_downloadReply;
  QFile m_downloadFile;
  bool m_isDownloading{false};
};

#endif // UPDATECHECK_H