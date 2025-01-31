#include "MetadataDialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QPushButton>
#include <QMessageBox>
#include <QFileDialog>
#include <QJsonDocument>
#include <QLineEdit>

MetadataDialog::MetadataDialog(const ImageMetadata& meta, QWidget* parent)
    : QDialog(parent), metadata(meta)
{
    setupUI();
    populateTree(metadata);
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
    connect(editButton, &QPushButton::clicked, this, &MetadataDialog::editCustomData);
    connect(saveButton, &QPushButton::clicked, this, &MetadataDialog::saveMetadata);
    connect(searchBox, &QLineEdit::textChanged, this, &MetadataDialog::searchMetadata);
    connect(exportButton, &QPushButton::clicked, this, &MetadataDialog::exportMetadata);
    connect(importButton, &QPushButton::clicked, this, &MetadataDialog::importMetadata);
    connect(resetButton, &QPushButton::clicked, this, &MetadataDialog::resetMetadata);
}

void MetadataDialog::editCustomData() {
    if (jsonEditor->isHidden()) {
        // 显示编辑器并加载数据
        jsonEditor->show();
        jsonEditor->loadJson(metadata.custom_data);
        editButton->setText("隐藏编辑器");
        saveButton->setEnabled(true);
    } else {
        jsonEditor->hide();
        editButton->setText("编辑自定义数据");
    }
}

void MetadataDialog::saveMetadata() {
    try {
        metadata.custom_data = jsonEditor->getJson();
        updateCustomDataDisplay();
        QMessageBox::information(this, "成功", "元数据已更新");
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "错误", QString("保存失败: %1").arg(e.what()));
    }
}

void MetadataDialog::updateCustomDataDisplay() {
    // 更新树形视图中的自定义数据显示
    // ...existing code...
}

void MetadataDialog::populateTree(const ImageMetadata& metadata) {
    // 基本信息
    auto basicInfo = new QTreeWidgetItem(treeWidget, {"基本信息"});
    new QTreeWidgetItem(basicInfo, {"文件路径", QString::fromStdString(metadata.path.string())});
    new QTreeWidgetItem(basicInfo, {"尺寸", QString("%1 x %2").arg(metadata.size.width).arg(metadata.size.height)});
    new QTreeWidgetItem(basicInfo, {"通道数", QString::number(metadata.channels)});
    new QTreeWidgetItem(basicInfo, {"色彩空间", QString::fromStdString(metadata.color_space)});
    
    // 自定义标签
    if (!metadata.custom_data.empty()) {
        auto customTags = new QTreeWidgetItem(treeWidget, {"自定义标签"});
        for (const auto& [key, value] : metadata.custom_data.items()) {
            new QTreeWidgetItem(customTags, {
                QString::fromStdString(key),
                QString::fromStdString(value.dump())
            });
        }
    }
    
    treeWidget->expandAll();
}

void MetadataDialog::exportMetadata() {
    QString filePath = QFileDialog::getSaveFileName(this, 
        "导出元数据", QString(), "JSON文件 (*.json)");
    
    if (!filePath.isEmpty()) {
        try {
            saveMetadataToFile(filePath);
            QMessageBox::information(this, "成功", "元数据已成功导出");
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "错误", 
                QString("导出失败: %1").arg(e.what()));
        }
    }
}

void MetadataDialog::importMetadata() {
    QString filePath = QFileDialog::getOpenFileName(this, 
        "导入元数据", QString(), "JSON文件 (*.json)");
    
    if (!filePath.isEmpty()) {
        try {
            loadMetadataFromFile(filePath);
            populateTree(metadata);
            QMessageBox::information(this, "成功", "元数据已成功导入");
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "错误", 
                QString("导入失败: %1").arg(e.what()));
        }
    }
}

void MetadataDialog::resetMetadata() {
    if (QMessageBox::question(this, "确认", "确定要重置所有元数据吗？") 
        == QMessageBox::Yes) {
        metadata = ImageMetadata(); // 重置为默认值
        populateTree(metadata);
    }
}

void MetadataDialog::searchMetadata() {
    QString searchText = searchBox->text().toLower();
    filterTreeItems(searchText);
}

void MetadataDialog::filterTreeItems(const QString& searchText) {
    for (int i = 0; i < treeWidget->topLevelItemCount(); ++i) {
        QTreeWidgetItem* item = treeWidget->topLevelItem(i);
        filterTreeItem(item, searchText);
    }
}

void MetadataDialog::saveMetadataToFile(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly)) {
        throw std::runtime_error("无法打开文件进行写入");
    }
    
    QJsonDocument doc = QJsonDocument::fromJson(
        QJsonDocument::fromVariant(metadata.toVariant()).toJson());
    file.write(doc.toJson());
}

void MetadataDialog::loadMetadataFromFile(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("无法打开文件进行读取");
    }
    
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    metadata.fromVariant(doc.toVariant().toMap());
}
