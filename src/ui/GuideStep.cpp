#include "GuideStep.h"

#include <QThreadPool>
#include <QMetaObject>
#include <QTimer>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QEventLoop>
#include <QApplication>
#include <QUrl>
#include <QMessageBox>
#include <stdexcept>
#include <fstream>
#include <filesystem>

#include "ElaCheckBox.h"
#include "ElaComboBox.h"
#include "ElaLineEdit.h"
#include "ElaSpinBox.h"

namespace fs = std::filesystem;

// 初始化日志记录器
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
namespace {
    std::shared_ptr<spdlog::logger> guideStepLogger =
        spdlog::basic_logger_mt("GuideStepLogger", "logs/guidestep.log");
} // namespace

////////////////////////////////////
// GuidePage 实现
////////////////////////////////////
GuidePage::GuidePage(QWidget *parent)
    : QWizardPage(parent) {}

bool GuidePage::validatePage() {
    try {
        return validateInputs();
    } catch (const std::exception &e) {
        guideStepLogger->error("Validation failed: {}", e.what());
        QMessageBox::critical(this, tr("Error"),
                              QString::fromStdString(std::format("Validation error: {}", e.what())));
        return false;
    }
}

bool GuidePage::validateInputs() {
    return true;
}

////////////////////////////////////
// WelcomePage 实现
////////////////////////////////////
WelcomePage::WelcomePage(QWidget *parent)
    : GuidePage(parent) {
    setTitle(tr("Welcome"));
    auto *layout = new QVBoxLayout;
    layout->addWidget(new QLabel(tr("Welcome to Application Setup")));
    layout->addWidget(new QLabel(tr("This wizard will guide you through the necessary steps to configure the application.")));
    setLayout(layout);
}

////////////////////////////////////
// FeatureSelectionPage 实现
////////////////////////////////////
FeatureSelectionPage::FeatureSelectionPage(QWidget *parent)
    : GuidePage(parent) {
    setTitle(tr("Feature Selection"));
    auto *layout = new QVBoxLayout(this);

    auto *featureLabel = new QLabel(tr("Choose a feature set:"), this);
    featureCombo = new ElaComboBox(this);
    featureCombo->addItem(tr("Basic"));
    featureCombo->addItem(tr("Advanced"));
    featureCombo->addItem(tr("Custom"));

    layout->addWidget(featureLabel);
    layout->addWidget(featureCombo);

    setLayout(layout);
    registerField("featureSelection", featureCombo, "currentText");
}

bool FeatureSelectionPage::validateInputs() {
    QString selectedFeature = featureCombo->currentText();
    if (selectedFeature == "Custom") {
        guideStepLogger->warn("Custom feature selected, further validation needed");
    }
    return true;
}

////////////////////////////////////
// ConfigPage 实现
////////////////////////////////////
ConfigPage::ConfigPage(QWidget *parent)
    : GuidePage(parent) {
    setupUI();
    registerField("workspace*", workspaceEdit);
    registerField("server*", serverEdit);
    registerField("port*", portSpin, "value");
}

void ConfigPage::setupUI() {
    auto *layout = new QVBoxLayout(this);

    auto *workspaceLabel = new QLabel(tr("Workspace:"), this);
    workspaceEdit = new ElaLineEdit(this);
    layout->addWidget(workspaceLabel);
    layout->addWidget(workspaceEdit);

    auto *serverLabel = new QLabel(tr("Server URL:"), this);
    serverEdit = new ElaLineEdit(this);
    layout->addWidget(serverLabel);
    layout->addWidget(serverEdit);

    auto *portLabel = new QLabel(tr("Port:"), this);
    portSpin = new ElaSpinBox(this);
    portSpin->setRange(1, 65535);
    layout->addWidget(portLabel);
    layout->addWidget(portSpin);

    setLayout(layout);
}

bool ConfigPage::validateInputs() {
    const auto path = workspaceEdit->text().toStdString();
    if (!fs::exists(path)) {
        throw std::runtime_error("Workspace path does not exist");
    }

    const auto server = serverEdit->text().toStdString();
    if (!std::ranges::all_of(server, [](char c) {
          return std::isalnum(c) || c == '.' || c == '-';
        })) {
        throw std::runtime_error("Invalid server format");
    }
    return true;
}

////////////////////////////////////
// AdvancedConfigPage 实现
////////////////////////////////////
AdvancedConfigPage::AdvancedConfigPage(QWidget *parent)
    : GuidePage(parent) {
    setTitle(tr("Advanced Configuration"));
    auto *layout = new QVBoxLayout(this);

    enableAdvancedFeature = new ElaCheckBox(tr("Enable Advanced Feature"), this);
    layout->addWidget(enableAdvancedFeature);

    auto *advancedSettingLabel = new QLabel(tr("Advanced Setting:"), this);
    advancedSettingEdit = new ElaLineEdit(this);
    layout->addWidget(advancedSettingLabel);
    layout->addWidget(advancedSettingEdit);

    setLayout(layout);
    registerField("enableAdvancedFeature", enableAdvancedFeature, "checked");
    registerField("advancedSetting", advancedSettingEdit, "text");
}

bool AdvancedConfigPage::validateInputs() {
    if (field("enableAdvancedFeature").toBool()) {
        QString setting = field("advancedSetting").toString();
        if (setting.isEmpty()) {
            throw std::runtime_error("Advanced setting is required when enabled");
        }
    }
    return true;
}

////////////////////////////////////
// SummaryPage 实现
////////////////////////////////////
SummaryPage::SummaryPage(QWidget *parent)
    : GuidePage(parent) {
    setTitle(tr("Summary"));
    summaryLabel = new QLabel(this);
    summaryLabel->setWordWrap(true);

    auto *layout = new QVBoxLayout(this);
    layout->addWidget(summaryLabel);
    setLayout(layout);
}

void SummaryPage::initializePage() {
    QString summaryText = QString("<b>Workspace:</b> %1<br>"
                                  "<b>Server URL:</b> %2<br>"
                                  "<b>Port:</b> %3<br>"
                                  "<b>Feature Set:</b> %4")
                              .arg(field("workspace").toString())
                              .arg(field("server").toString())
                              .arg(field("port").toInt())
                              .arg(field("featureSelection").toString());
    if (wizard()->hasVisitedPage(Page_AdvancedConfig)) {
        summaryText += QString("<br><b>Advanced Feature Enabled:</b> %1")
                           .arg(field("enableAdvancedFeature").toBool() ? "Yes" : "No");
    }
    summaryLabel->setText(summaryText);
}

////////////////////////////////////
// SetupWizard 实现
////////////////////////////////////
SetupWizard::SetupWizard(QWidget *parent)
    : QWizard(parent) {
    setupPages();
    connect(this, &QWizard::finished, this, &SetupWizard::onFinished);
    setOption(QWizard::NoDefaultButton, true);
}

AppConfig SetupWizard::getConfig() const {
    return AppConfig{
        .workspace = field("workspace").toString().toStdString(),
        .serverUrl = field("server").toString().toStdString(),
        .port = static_cast<uint16_t>(field("port").toInt()),
        .enableLogging = true,
        .featureSelection = field("featureSelection").toString().toStdString()
    };
}

void SetupWizard::onFinished(int result) {
    if (result == QDialog::Accepted) {
        Q_EMIT configCompleted(getConfig());
    }
}

void SetupWizard::setupPages() {
    setPage(Page_Welcome, new WelcomePage);
    setPage(Page_FeatureSelection, new FeatureSelectionPage);
    setPage(Page_Config, new ConfigPage);
    setPage(Page_AdvancedConfig, new AdvancedConfigPage);
    setPage(Page_Summary, new SummaryPage);
    setStartId(Page_Welcome);
}

////////////////////////////////////
// AppInitializer 实现
////////////////////////////////////
AppInitializer::AppInitializer(const AppConfig &config) {
    initializeAsync(config);
}

AppInitializer::~AppInitializer() {
    if (initialized)
        shutdown();
}

void AppInitializer::initializeAsync(const AppConfig &config) {
    QThreadPool::globalInstance()->start([this, config]() {
        try {
            guideStepLogger->info("Initializing application...");
            initializeWorkspace(config.workspace);
            testServerConnection(config.serverUrl, config.port);
            configureFeatures(config.featureSelection);
            guideStepLogger->info("Initialization completed");
            initialized = true;
        } catch (const std::exception &e) {
            guideStepLogger->critical("Initialization failed: {}", e.what());
            QMetaObject::invokeMethod(qApp, [msg = std::string(e.what())] {
                QMessageBox::critical(nullptr, "Fatal Error",
                                      QString::fromStdString(std::format("Initialization failed:\n{}", msg)));
                qApp->exit(EXIT_FAILURE);
            });
        }
    });
}

void AppInitializer::initializeWorkspace(const std::string &path) {
    fs::create_directories(path);
    if (!fs::is_directory(path)) {
        throw std::runtime_error("Failed to create workspace");
    }
    std::ofstream testFile(fs::path(path) / "test.tmp");
    if (!testFile)
        throw std::runtime_error("No write permission");
}

void AppInitializer::testServerConnection(const std::string &url, uint16_t port) {
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

    QUrl qUrl = QUrl(QString::fromStdString(std::format("http://{}:{}", url, port)));
    QNetworkRequest request(qUrl);
    manager->get(request);

    loop.exec();
    if (!loop.isRunning()) { // 超时处理
        throw std::runtime_error("Connection timeout");
    }
}

void AppInitializer::configureFeatures(const std::string &featureSet) {
    guideStepLogger->info("Configuring features based on selection: {}", featureSet);
    if (featureSet == "Basic") {
        guideStepLogger->info("Configuring basic features");
    } else if (featureSet == "Advanced") {
        guideStepLogger->info("Configuring advanced features");
    } else {
        guideStepLogger->warn("Custom feature configuration selected");
    }
}

void AppInitializer::shutdown() noexcept {
    guideStepLogger->info("Cleaning up resources...");
    // 资源释放逻辑
}

////////////////////////////////////
// main 函数
////////////////////////////////////
#include <chrono>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    auto daily_sink =
        std::make_shared<spdlog::sinks::daily_file_sink_mt>("logs/app.log", 0, 0);
    auto logger = std::make_shared<spdlog::logger>("main", daily_sink);
    spdlog::set_default_logger(logger);
    spdlog::flush_every(std::chrono::seconds(3));

    SetupWizard wizard;
    QObject::connect(&wizard, &SetupWizard::configCompleted, [&](AppConfig config) {
        try {
            AppInitializer initializer(config);
            QMessageBox::information(nullptr, "Success",
                                     "Application initialized successfully!");
        } catch (...) {
            // 异常通过其他机制处理
        }
    });

    wizard.show();
    return app.exec();
}