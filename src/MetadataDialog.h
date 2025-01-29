#ifndef METADATADIALOG_H
#define METADATADIALOG_H

#include <QDialog>
#include <QTreeWidget>
#include "image/MetaData.hpp"

class MetadataDialog : public QDialog {
    Q_OBJECT

public:
    explicit MetadataDialog(const ImageMetadata& metadata, QWidget* parent = nullptr);

private:
    QTreeWidget* treeWidget;
    void setupUI();
    void populateTree(const ImageMetadata& metadata);
};

#endif
