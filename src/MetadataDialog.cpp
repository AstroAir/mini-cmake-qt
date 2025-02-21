#include "MetadataDialog.h"

#include <QFileDialog>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QStandardItemModel>
#include <QVBoxLayout>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>


#include "ElaLineEdit.h"
#include "ElaPushButton.h"
#include "ElaTreeView.h"


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
    treeView->setModel(nullptr);
  }
  resize(800, 600);
  setWindowTitle("图像元信息");
}

void MetadataDialog::setupUI() {
  auto mainLayout = new QVBoxLayout(this);

  // 添加搜索框
  searchBox = new ElaLineEdit(this);
  searchBox->setPlaceholderText("搜索元数据...");
  mainLayout->insertWidget(0, searchBox);

  // 创建树形视图和模型
  treeView = new ElaTreeView(this);
  model = new QStandardItemModel(this);
  model->setHorizontalHeaderLabels({"属性", "值"});
  treeView->setModel(model);
  treeView->setAlternatingRowColors(true);
  treeView->setEditTriggers(ElaTreeView::NoEditTriggers);
  treeView->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
  mainLayout->addWidget(treeView);

  // 添加按钮布局
  auto buttonLayout = new QHBoxLayout;
  editButton = new ElaPushButton("编辑自定义数据");
  saveButton = new ElaPushButton("保存修改");
  saveButton->setEnabled(false);

  buttonLayout->addWidget(editButton);
  buttonLayout->addWidget(saveButton);
  buttonLayout->addStretch();

  // 添加新按钮
  exportButton = new ElaPushButton("导出", this);
  importButton = new ElaPushButton("导入", this);
  resetButton = new ElaPushButton("重置", this);

  buttonLayout->addWidget(exportButton);
  buttonLayout->addWidget(importButton);
  buttonLayout->addWidget(resetButton);

  mainLayout->addLayout(buttonLayout);

  // 创建但默认隐藏JsonEditor
  jsonEditor = new JsonEditor(this);
  jsonEditor->hide();
  mainLayout->addWidget(jsonEditor);

  // 连接信号
  connect(editButton, &ElaPushButton::clicked, this,
          &MetadataDialog::editCustomData);
  connect(saveButton, &ElaPushButton::clicked, this,
          &MetadataDialog::saveMetadata);
  connect(searchBox, &ElaLineEdit::textChanged, this,
          &MetadataDialog::searchMetadata);
  connect(exportButton, &ElaPushButton::clicked, this,
          &MetadataDialog::exportMetadata);
  connect(importButton, &ElaPushButton::clicked, this,
          &MetadataDialog::importMetadata);
  connect(resetButton, &ElaPushButton::clicked, this,
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
  model->clear();
  model->setHorizontalHeaderLabels({"属性", "值"});

  // 基本信息
  auto basicInfo = new QStandardItem("基本信息");
  model->appendRow(basicInfo);

  auto pathItem = new QStandardItem("文件路径");
  auto pathValue =
      new QStandardItem(QString::fromStdString(metadata.path.string()));
  basicInfo->appendRow({pathItem, pathValue});

  auto sizeItem = new QStandardItem("尺寸");
  auto sizeValue = new QStandardItem(
      QString("%1 x %2").arg(metadata.size.width).arg(metadata.size.height));
  basicInfo->appendRow({sizeItem, sizeValue});

  auto channelsItem = new QStandardItem("通道数");
  auto channelsValue = new QStandardItem(QString::number(metadata.channels));
  basicInfo->appendRow({channelsItem, channelsValue});

  auto colorSpaceItem = new QStandardItem("色彩空间");
  auto colorSpaceValue =
      new QStandardItem(QString::fromStdString(metadata.color_space));
  basicInfo->appendRow({colorSpaceItem, colorSpaceValue});

  // 自定义标签
  if (!metadata.custom_data.empty()) {
    auto customTags = new QStandardItem("自定义标签");
    model->appendRow(customTags);

    for (const auto &[key, value] : metadata.custom_data.items()) {
      auto keyItem = new QStandardItem(QString::fromStdString(key));
      auto valueItem = new QStandardItem(QString::fromStdString(value.dump()));
      customTags->appendRow({keyItem, valueItem});
    }
  }

  treeView->expandAll();
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
  if (!model->invisibleRootItem())
    return;

  for (int i = 0; i < model->rowCount(); ++i) {
    QStandardItem *item = model->item(i);
    filterTreeItem(item, searchText);
  }
}

bool MetadataDialog::filterTreeItem(QStandardItem *item,
                                    const QString &searchText) {
  if (!item)
    return false;
  bool matched = false;

  // 检查当前项
  if (searchText.isEmpty()) {
    matched = true;
  } else {
    QStandardItem *valueItem = item->parent()
                                   ? item->parent()->child(item->row(), 1)
                                   : model->item(item->row(), 1);

    matched = item->text().toLower().contains(searchText) ||
              (valueItem && valueItem->text().toLower().contains(searchText));
  }

  // 递归检查子项
  for (int i = 0; i < item->rowCount(); ++i) {
    if (filterTreeItem(item->child(i), searchText)) {
      matched = true;
    }
  }

  if (item->parent()) {
    treeView->setRowHidden(item->row(), item->parent()->index(), !matched);
  } else {
    treeView->setRowHidden(item->row(), QModelIndex(), !matched);
  }

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