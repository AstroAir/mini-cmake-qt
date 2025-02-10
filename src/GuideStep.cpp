#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMetaObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QObject>
#include <QPushButton>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <QWizard>
#include <QWizardPage>
#include <QtConcurrent>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <qthreadpool.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/spdlog.h>
#include <string>

namespace fs = std::filesystem;

namespace {
std::shared_ptr<spdlog::logger> guideStepLogger =
    spdlog::basic_logger_mt("GuideStepLogger", "logs/guidestep.log");
} // namespace

// 配置数据结构（C++20 聚合初始化）
struct AppConfig {
  std::string workspace;
  std::string serverUrl;
  uint16_t port;
  bool enableLogging;
  std::string featureSelection;
};

// 自定义向导页面基类（Qt Widget）
class GuidePage : public QWizardPage {
  Q_OBJECT
public:
  explicit GuidePage(QWidget *parent = nullptr) : QWizardPage(parent) {}
  virtual bool validatePage() override {
    try {
      return validateInputs();
    } catch (const std::exception &e) {
      guideStepLogger->error("Validation failed: {}", e.what());
      QMessageBox::critical(this, tr("Error"),
                            QString::fromStdString(
                                std::format("Validation error: {}", e.what())));
      return false;
    }
  }

protected:
  virtual bool validateInputs() { return true; }
};

// 欢迎页面（使用C++20 attributes）
class [[nodiscard]] WelcomePage final : public GuidePage {
public:
  WelcomePage() {
    setTitle(tr("Welcome"));
    auto *layout = new QVBoxLayout;
    layout->addWidget(new QLabel(tr("Welcome to Application Setup")));
    layout->addWidget(new QLabel(tr("This wizard will guide you through the "
                                    "necessary steps to configure the "
                                    "application.")));
    setLayout(layout);
  }
};

// 特性选择页面
class FeatureSelectionPage final : public GuidePage {
  Q_OBJECT

  QComboBox *featureCombo;

public:
  FeatureSelectionPage() {
    setTitle(tr("Feature Selection"));
    auto *layout = new QVBoxLayout(this);

    auto *featureLabel = new QLabel(tr("Choose a feature set:"), this);
    featureCombo = new QComboBox(this);
    featureCombo->addItem(tr("Basic"));
    featureCombo->addItem(tr("Advanced"));
    featureCombo->addItem(tr("Custom"));

    layout->addWidget(featureLabel);
    layout->addWidget(featureCombo);

    setLayout(layout);
    registerField("featureSelection", featureCombo, "currentText");
  }

protected:
  bool validateInputs() override {
    // 示例：根据选择的特性进行验证
    QString selectedFeature = featureCombo->currentText();
    if (selectedFeature == "Custom") {
      // 自定义特性需要进一步验证，这里简化处理
      guideStepLogger->warn(
          "Custom feature selected, further validation needed");
    }
    return true;
  }
};

// 配置页面（使用STL ranges验证）
class ConfigPage final : public GuidePage {
  QLineEdit *workspaceEdit;
  QLineEdit *serverEdit;
  QSpinBox *portSpin;

public:
  ConfigPage() {
    setupUI();
    registerField("workspace*", workspaceEdit);
    registerField("server*", serverEdit);
    registerField("port*", portSpin, "value");
  }

private:
  void setupUI() {
    auto *layout = new QVBoxLayout(this);

    auto *workspaceLabel = new QLabel(tr("Workspace:"), this);
    workspaceEdit = new QLineEdit(this);
    layout->addWidget(workspaceLabel);
    layout->addWidget(workspaceEdit);

    auto *serverLabel = new QLabel(tr("Server URL:"), this);
    serverEdit = new QLineEdit(this);
    layout->addWidget(serverLabel);
    layout->addWidget(serverEdit);

    auto *portLabel = new QLabel(tr("Port:"), this);
    portSpin = new QSpinBox(this);
    portSpin->setRange(1, 65535);
    layout->addWidget(portLabel);
    layout->addWidget(portSpin);

    setLayout(layout);
  }

  bool validateInputs() override {
    // 使用C++20 ranges验证路径
    const auto path = workspaceEdit->text().toStdString();
    if (!fs::exists(path)) {
      throw std::runtime_error("Workspace path does not exist");
    }

    // 使用STL算法验证服务器格式
    const auto server = serverEdit->text().toStdString();
    if (!std::ranges::all_of(server, [](char c) {
          return std::isalnum(c) || c == '.' || c == '-';
        })) {
      throw std::runtime_error("Invalid server format");
    }

    return true;
  }
};

// 高级配置页面
class AdvancedConfigPage final : public GuidePage {
  Q_OBJECT

  QCheckBox *enableAdvancedFeature;
  QLineEdit *advancedSettingEdit;

public:
  AdvancedConfigPage() {
    setTitle(tr("Advanced Configuration"));
    auto *layout = new QVBoxLayout(this);

    enableAdvancedFeature = new QCheckBox(tr("Enable Advanced Feature"), this);
    layout->addWidget(enableAdvancedFeature);

    auto *advancedSettingLabel = new QLabel(tr("Advanced Setting:"), this);
    advancedSettingEdit = new QLineEdit(this);
    layout->addWidget(advancedSettingLabel);
    layout->addWidget(advancedSettingEdit);

    setLayout(layout);
    registerField("enableAdvancedFeature", enableAdvancedFeature, "checked");
    registerField("advancedSetting", advancedSettingEdit, "text");
  }

protected:
  bool validateInputs() override {
    if (field("enableAdvancedFeature").toBool()) {
      QString setting = field("advancedSetting").toString();
      if (setting.isEmpty()) {
        throw std::runtime_error("Advanced setting is required when enabled");
      }
    }
    return true;
  }
};

// 总结页面
class SummaryPage final : public GuidePage {
  Q_OBJECT

  QLabel *summaryLabel;

public:
  SummaryPage() {
    setTitle(tr("Summary"));
    summaryLabel = new QLabel(this);
    summaryLabel->setWordWrap(true);

    auto *layout = new QVBoxLayout(this);
    layout->addWidget(summaryLabel);
    setLayout(layout);
  }

  void initializePage() override {
    QString summaryText = QString("<b>Workspace:</b> %1<br>"
                                  "<b>Server URL:</b> %2<br>"
                                  "<b>Port:</b> %3<br>"
                                  "<b>Feature Set:</b> %4")
                              .arg(field("workspace").toString())
                              .arg(field("server").toString())
                              .arg(field("port").toInt())
                              .arg(field("featureSelection").toString());

    if (wizard()->hasVisitedPage(Page_AdvancedConfig)) {
      summaryText +=
          QString("<br><b>Advanced Feature Enabled:</b> %1")
              .arg(field("enableAdvancedFeature").toBool() ? "Yes" : "No");
    }

    summaryLabel->setText(summaryText);
  }

private:
  enum { Page_AdvancedConfig };
};

// 引导向导主类（使用QWizard）
class SetupWizard : public QWizard {
  Q_OBJECT
public:
  SetupWizard() {
    setupPages();
    connect(this, &QWizard::finished, this, &SetupWizard::onFinished);
    setOption(QWizard::NoDefaultButton, true);
  }

  [[nodiscard]] AppConfig getConfig() const {
    return {.workspace = field("workspace").toString().toStdString(),
            .serverUrl = field("server").toString().toStdString(),
            .port = static_cast<uint16_t>(field("port").toInt()),
            .enableLogging = true,
            .featureSelection =
                field("featureSelection").toString().toStdString()};
  }

private slots:
  void onFinished(int result) {
    if (result == QDialog::Accepted) {
      Q_EMIT configCompleted(getConfig());
    }
  }

signals:
  void configCompleted(AppConfig);

private:
  void setupPages() {
    setPage(Page_Welcome, new WelcomePage);
    setPage(Page_FeatureSelection, new FeatureSelectionPage);
    setPage(Page_Config, new ConfigPage);
    setPage(Page_AdvancedConfig, new AdvancedConfigPage);
    setPage(Page_Summary, new SummaryPage);

    // 设置页面之间的依赖关系
    setStartId(Page_Welcome);
  }

  enum {
    Page_Welcome,
    Page_FeatureSelection,
    Page_Config,
    Page_AdvancedConfig,
    Page_Summary
  };
};

// 初始化管理器（使用RAII和异常安全）
class [[nodiscard]] AppInitializer {
  std::atomic<bool> initialized{false};

public:
  explicit AppInitializer(const AppConfig &config) { initializeAsync(config); }

  ~AppInitializer() {
    if (initialized)
      shutdown();
  }

private:
  void initializeAsync(const AppConfig &config) {
    QThreadPool::globalInstance()->start([this, config] {
      try {
        guideStepLogger->info("Initializing application...");

        // 示例：异步初始化操作
        initializeWorkspace(config.workspace);
        testServerConnection(config.serverUrl, config.port);
        configureFeatures(config.featureSelection);

        guideStepLogger->info("Initialization completed");
        initialized = true;
      } catch (const std::exception &e) {
        guideStepLogger->critical("Initialization failed: {}", e.what());
        QMetaObject::invokeMethod(qApp, [msg = e.what()] {
          QMessageBox::critical(nullptr, "Fatal Error",
                                QString::fromStdString(std::format(
                                    "Initialization failed:\n{}", msg)));
          qApp->exit(EXIT_FAILURE);
        });
      }
    });
  }

  void initializeWorkspace(const std::string &path) {
    // 使用C++17 filesystem API
    fs::create_directories(path);
    if (!fs::is_directory(path)) {
      throw std::runtime_error("Failed to create workspace");
    }

    // 验证写权限（使用STL）
    std::ofstream testFile(fs::path(path) / "test.tmp");
    if (!testFile)
      throw std::runtime_error("No write permission");
  }

  void testServerConnection(const std::string &url, uint16_t port) {
    // 使用Qt网络模块异步测试连接
    QEventLoop loop;
    QTimer::singleShot(5000, &loop, &QEventLoop::quit);

    auto *manager = new QNetworkAccessManager;
    QObject::connect(manager, &QNetworkAccessManager::finished,
                     [&](QNetworkReply *reply) {
                       if (reply->error() != QNetworkReply::NoError) {
                         throw std::runtime_error("Server connection failed");
                       }
                       loop.quit();
                     });

    QUrl qUrl =
        QUrl(QString::fromStdString(std::format("http://{}:{}", url, port)));
    QNetworkRequest request(qUrl);

    manager->get(request);

    loop.exec();
    if (!loop.isRunning()) { // 超时处理
      throw std::runtime_error("Connection timeout");
    }
  }

  void configureFeatures(const std::string &featureSet) {
    guideStepLogger->info("Configuring features based on selection: {}",
                          featureSet);
    if (featureSet == "Basic") {
      // 配置基本特性
      guideStepLogger->info("Configuring basic features");
    } else if (featureSet == "Advanced") {
      // 配置高级特性
      guideStepLogger->info("Configuring advanced features");
    } else {
      // 自定义配置
      guideStepLogger->warn("Custom feature configuration selected");
    }
  }

  void shutdown() noexcept {
    guideStepLogger->info("Cleaning up resources...");
    // 资源释放逻辑
  }
};

// 主程序入口
int main(int argc, char *argv[]) {
  QApplication app(argc, argv);

  auto daily_sink =
      std::make_shared<spdlog::sinks::daily_file_sink_mt>("logs/app.log", 0, 0);
  auto logger = std::make_shared<spdlog::logger>("main", daily_sink);
  spdlog::set_default_logger(logger);
  spdlog::flush_every(std::chrono::seconds(3));

  SetupWizard wizard;
  QObject::connect(
      &wizard, &SetupWizard::configCompleted, [&](AppConfig config) {
        try {
          AppInitializer initializer(config);
          QMessageBox::information(nullptr, "Success",
                                   "Application initialized successfully!");
        } catch (...) {
          // 异常已通过其他机制处理
        }
      });

  wizard.show();
  return app.exec();
}
