#include "MetadataDialog.h"
#include <QFileDialog>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <spdlog/sinks/basic_file_sink.h> // 需要包含文件 sink
#include <spdlog/spdlog.h>


namespace {
// 创建一个只属于 MetadataDialog.cpp 的 logger
std::shared_ptr<spdlog::logger> CreateMetadataDialogLogger() {
  try {
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        "metadata_dialog.log", true);
    auto logger =
        std::make_shared<spdlog::logger>("metadata_dialog", file_sink);
    logger->set_level(spdlog::level::debug);               // 设置日志级别
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v"); // 设置日志格式
    return logger;
  } catch (const spdlog::spdlog_ex &ex) {
    std::cerr << "Log initialization failed: " << ex.what() << std::endl;
    return nullptr;
  }
}

auto metadata_dialog_logger = CreateMetadataDialogLogger();
} // namespace

MetadataDialog::MetadataDialog(const ImageMetadata &meta, QWidget *parent)
    : QDialog(parent), metadata(meta) {
  setupUI();
  if (metadata_dialog_logger) {
    metadata_dialog_logger->debug("MetadataDialog 创建, path: {}",
                                  meta.path.string());
  }
  try {
    populateTree(metadata);
  } catch (const std::exception &e) {
    if (metadata_dialog_logger) {
      metadata_dialog_logger->error("加载元信息失败: {}", e.what());
    }
    QMessageBox::critical(this, "错误",
                          QString("加载元信息失败: %1").arg(e.what()));
    // 可选：清空 treeWidget 或其他处理
    treeWidget->clear();
  }
  resize(800, 600);
  setWindowTitle("图像元信息");
}

void MetadataDialog::setupUI() {
  auto mainLayout = new QVBoxLayout(this);

  // 添加搜索框
  searchBox = new QLineEdit(this);
  searchBox->setPlaceholderText("搜索元数据...");
  mainLayout->insertWidget(0, searchBox);

  // 添加树形视图
  treeWidget = new QTreeWidget;
  treeWidget->setHeaderLabels({"属性", "值"});
  treeWidget->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
  mainLayout->addWidget(treeWidget);

  // 添加按钮布局
  auto buttonLayout = new QHBoxLayout;
  editButton = new QPushButton("编辑自定义数据");
  saveButton = new QPushButton("保存修改");
  saveButton->setEnabled(false);

  buttonLayout->addWidget(editButton);
  buttonLayout->addWidget(saveButton);
  buttonLayout->addStretch();

  // 添加新按钮
  exportButton = new QPushButton("导出", this);
  importButton = new QPushButton("导入", this);
  resetButton = new QPushButton("重置", this);

  buttonLayout->addWidget(exportButton);
  buttonLayout->addWidget(importButton);
  buttonLayout->addWidget(resetButton);

  mainLayout->addLayout(buttonLayout);

  // 创建但默认隐藏JsonEditor
  jsonEditor = new JsonEditor(this);
  jsonEditor->hide();
  mainLayout->addWidget(jsonEditor);

  // 连接信号
  connect(editButton, &QPushButton::clicked, this,
          &MetadataDialog::editCustomData);
  connect(saveButton, &QPushButton::clicked, this,
          &MetadataDialog::saveMetadata);
  connect(searchBox, &QLineEdit::textChanged, this,
          &MetadataDialog::searchMetadata);
  connect(exportButton, &QPushButton::clicked, this,
          &MetadataDialog::exportMetadata);
  connect(importButton, &QPushButton::clicked, this,
          &MetadataDialog::importMetadata);
  connect(resetButton, &QPushButton::clicked, this,
          &MetadataDialog::resetMetadata);
}

void MetadataDialog::editCustomData() {
  if (jsonEditor->isHidden()) {
    // 显示编辑器并加载数据
    jsonEditor->show();
    jsonEditor->loadJson(metadata.custom_data);
    editButton->setText("隐藏编辑器");
    saveButton->setEnabled(true);
    if (metadata_dialog_logger) {
      metadata_dialog_logger->debug("显示 JsonEditor");
    }
  } else {
    jsonEditor->hide();
    editButton->setText("编辑自定义数据");
    if (metadata_dialog_logger) {
      metadata_dialog_logger->debug("隐藏 JsonEditor");
    }
  }
}

void MetadataDialog::saveMetadata() {
  try {
    metadata.custom_data = jsonEditor->getJson();
    updateCustomDataDisplay();
    QMessageBox::information(this, "成功", "元数据已更新");
    if (metadata_dialog_logger) {
      metadata_dialog_logger->info("元数据已更新");
    }
  } catch (const std::exception &e) {
    QMessageBox::critical(this, "错误", QString("保存失败: %1").arg(e.what()));
    if (metadata_dialog_logger) {
      metadata_dialog_logger->error("保存元数据失败: {}", e.what());
    }
  }
}

void MetadataDialog::updateCustomDataDisplay() {
  // 更新树形视图中的自定义数据显示
  // ...existing code...
}

void MetadataDialog::populateTree(const ImageMetadata &metadata) {
  // 基本信息
  auto basicInfo = new QTreeWidgetItem(treeWidget, {"基本信息"});
  new QTreeWidgetItem(
      basicInfo, {"文件路径", QString::fromStdString(metadata.path.string())});
  new QTreeWidgetItem(
      basicInfo,
      {"尺寸",
       QString("%1 x %2").arg(metadata.size.width).arg(metadata.size.height)});
  new QTreeWidgetItem(basicInfo,
                      {"通道数", QString::number(metadata.channels)});
  new QTreeWidgetItem(
      basicInfo, {"色彩空间", QString::fromStdString(metadata.color_space)});

  // 自定义标签
  if (!metadata.custom_data.empty()) {
    auto customTags = new QTreeWidgetItem(treeWidget, {"自定义标签"});
    for (const auto &[key, value] : metadata.custom_data.items()) {
      new QTreeWidgetItem(customTags, {QString::fromStdString(key),
                                       QString::fromStdString(value.dump())});
    }
  }

  treeWidget->expandAll();
}

void MetadataDialog::exportMetadata() {
  QString filePath = QFileDialog::getSaveFileName(this, "导出元数据", QString(),
                                                  "JSON文件 (*.json)");

  if (!filePath.isEmpty()) {
    try {
      saveMetadataToFile(filePath);
      QMessageBox::information(this, "成功", "元数据已成功导出");
      if (metadata_dialog_logger) {
        metadata_dialog_logger->info("元数据已成功导出到: {}",
                                     filePath.toStdString());
      }
    } catch (const std::exception &e) {
      QMessageBox::critical(this, "错误",
                            QString("导出失败: %1").arg(e.what()));
      if (metadata_dialog_logger) {
        metadata_dialog_logger->error("导出元数据失败: {}", e.what());
      }
    }
  }
}

void MetadataDialog::importMetadata() {
  QString filePath = QFileDialog::getOpenFileName(this, "导入元数据", QString(),
                                                  "JSON文件 (*.json)");

  if (!filePath.isEmpty()) {
    try {
      loadMetadataFromFile(filePath);
      populateTree(metadata);
      QMessageBox::information(this, "成功", "元数据已成功导入");
      if (metadata_dialog_logger) {
        metadata_dialog_logger->info("元数据已成功从 {} 导入",
                                     filePath.toStdString());
      }
    } catch (const std::exception &e) {
      QMessageBox::critical(this, "错误",
                            QString("导入失败: %1").arg(e.what()));
      if (metadata_dialog_logger) {
        metadata_dialog_logger->error("导入元数据失败: {}", e.what());
      }
    }
  }
}

void MetadataDialog::resetMetadata() {
  if (QMessageBox::question(this, "确认", "确定要重置所有元数据吗？") ==
      QMessageBox::Yes) {
    metadata = ImageMetadata(); // 重置为默认值
    populateTree(metadata);
    if (metadata_dialog_logger) {
      metadata_dialog_logger->warn("元数据已重置");
    }
  }
}

void MetadataDialog::searchMetadata() {
  QString searchText = searchBox->text().toLower();
  filterTreeItems(searchText);
  if (metadata_dialog_logger) {
    metadata_dialog_logger->debug("搜索元数据: {}", searchText.toStdString());
  }
}

void MetadataDialog::filterTreeItems(const QString &searchText) {
  for (int i = 0; i < treeWidget->topLevelItemCount(); ++i) {
    QTreeWidgetItem *item = treeWidget->topLevelItem(i);
    filterTreeItem(item, searchText);
  }
}

bool MetadataDialog::filterTreeItem(QTreeWidgetItem *item,
                                    const QString &searchText) {
  bool matched = false;

  // 检查当前项
  if (searchText.isEmpty() || item->text(0).toLower().contains(searchText) ||
      item->text(1).toLower().contains(searchText)) {
    matched = true;
  }

  // 递归检查子项
  for (int i = 0; i < item->childCount(); ++i) {
    if (filterTreeItem(item->child(i), searchText)) {
      matched = true;
    }
  }

  item->setHidden(!matched);
  return matched;
}

void MetadataDialog::saveMetadataToFile(const QString &filePath) {
  QFile file(filePath);
  if (!file.open(QIODevice::WriteOnly)) {
    if (metadata_dialog_logger) {
      metadata_dialog_logger->error("无法打开文件进行写入: {}",
                                    filePath.toStdString());
    }
    throw std::runtime_error("无法打开文件进行写入");
  }

  // 直接将 metadata 转换为 JSON
  nlohmann::json j = metadata.to_json();
  QJsonDocument doc =
      QJsonDocument::fromJson(QByteArray::fromStdString(j.dump()));
  file.write(doc.toJson());
}

void MetadataDialog::loadMetadataFromFile(const QString &filePath) {
  QFile file(filePath);
  if (!file.open(QIODevice::ReadOnly)) {
    if (metadata_dialog_logger) {
      metadata_dialog_logger->error("无法打开文件进行读取: {}",
                                    filePath.toStdString());
    }
    throw std::runtime_error("无法打开文件进行读取");
  }
  QByteArray content = file.readAll();
  if (content.isEmpty()) {
    // 文件为空，避免崩溃，设置为空的元数据
    metadata = ImageMetadata();
    if (metadata_dialog_logger) {
      metadata_dialog_logger->warn("文件为空: {}", filePath.toStdString());
    }
    return;
  }
  QJsonDocument doc = QJsonDocument::fromJson(content);
  if (doc.isNull() || doc.isEmpty()) {
    metadata = ImageMetadata();
    if (metadata_dialog_logger) {
      metadata_dialog_logger->warn("文件内容为空或格式不正确: {}",
                                   filePath.toStdString());
    }
    return;
  }
  nlohmann::json j = nlohmann::json::parse(doc.toJson().toStdString());
  metadata = ImageMetadata::from_json(j);
}