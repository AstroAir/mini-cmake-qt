#include "MetadataDialog.h"
#include <QVBoxLayout>
#include <QHeaderView>

MetadataDialog::MetadataDialog(const ImageMetadata& metadata, QWidget* parent)
    : QDialog(parent)
{
    setupUI();
    populateTree(metadata);
    resize(600, 400);
    setWindowTitle("图像元信息");
}

void MetadataDialog::setupUI() {
    auto layout = new QVBoxLayout(this);
    treeWidget = new QTreeWidget;
    treeWidget->setHeaderLabels({"属性", "值"});
    treeWidget->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
    layout->addWidget(treeWidget);
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
