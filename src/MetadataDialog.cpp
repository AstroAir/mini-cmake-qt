#include "MetadataDialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QPushButton>
#include <QMessageBox>

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
    mainLayout->addLayout(buttonLayout);

    // 创建但默认隐藏JsonEditor
    jsonEditor = new JsonEditor(this);
    jsonEditor->hide();
    mainLayout->addWidget(jsonEditor);

    // 连接信号
    connect(editButton, &QPushButton::clicked, this, &MetadataDialog::editCustomData);
    connect(saveButton, &QPushButton::clicked, this, &MetadataDialog::saveMetadata);
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
