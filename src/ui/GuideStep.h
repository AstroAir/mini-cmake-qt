#pragma once

#include <QDialog>
#include <QLabel>
#include <QMessageBox>
#include <QString>
#include <QVBoxLayout>
#include <QWidget>
#include <QWizard>
#include <QWizardPage>
#include <atomic>
#include <string>

class ElaCheckBox;
class ElaComboBox;
class ElaLineEdit;
class ElaSpinBox;

struct AppConfig {
  std::string workspace;
  std::string serverUrl;
  uint16_t port;
  bool enableLogging;
  std::string featureSelection;
};

// 自定义向导页面基类
class GuidePage : public QWizardPage {
  Q_OBJECT
public:
  explicit GuidePage(QWidget *parent = nullptr);
  bool validatePage() override;

protected:
  virtual bool validateInputs();
};

// 欢迎页面
class WelcomePage final : public GuidePage {
  Q_OBJECT
public:
  explicit WelcomePage(QWidget *parent = nullptr);
};

// 特性选择页面
class FeatureSelectionPage final : public GuidePage {
  Q_OBJECT
public:
  explicit FeatureSelectionPage(QWidget *parent = nullptr);

protected:
  bool validateInputs() override;

private:
  ElaComboBox *featureCombo;
};

// 配置页面
class ConfigPage final : public GuidePage {
  Q_OBJECT
public:
  explicit ConfigPage(QWidget *parent = nullptr);

protected:
  bool validateInputs() override;

private:
  ElaLineEdit *workspaceEdit;
  ElaLineEdit *serverEdit;
  ElaSpinBox *portSpin;
  void setupUI();
};

// 高级配置页面
class AdvancedConfigPage final : public GuidePage {
  Q_OBJECT
public:
  explicit AdvancedConfigPage(QWidget *parent = nullptr);

protected:
  bool validateInputs() override;

private:
  ElaCheckBox *enableAdvancedFeature;
  ElaLineEdit *advancedSettingEdit;
};

// 总结页面
class SummaryPage final : public GuidePage {
  Q_OBJECT
public:
  explicit SummaryPage(QWidget *parent = nullptr);
  void initializePage() override;

private:
  QLabel *summaryLabel;
  enum { Page_AdvancedConfig };
};

// 引导向导主类
class SetupWizard final : public QWizard {
  Q_OBJECT
public:
  explicit SetupWizard(QWidget *parent = nullptr);
  [[nodiscard]] AppConfig getConfig() const;

signals:
  void configCompleted(AppConfig);

private slots:
  void onFinished(int result);

private:
  void setupPages();
  enum {
    Page_Welcome,
    Page_FeatureSelection,
    Page_Config,
    Page_AdvancedConfig,
    Page_Summary
  };
};

// 应用初始化管理器
class AppInitializer final {
public:
  explicit AppInitializer(const AppConfig &config);
  ~AppInitializer();

private:
  std::atomic<bool> initialized{false};
  void initializeAsync(const AppConfig &config);
  void initializeWorkspace(const std::string &path);
  void testServerConnection(const std::string &url, uint16_t port);
  void configureFeatures(const std::string &featureSet);
  void shutdown() noexcept;
};
