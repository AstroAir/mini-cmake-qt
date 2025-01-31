#ifndef METADATADIALOG_H
#define METADATADIALOG_H

#include <QDialog>
#include <QTreeWidget>
#include "image/MetaData.hpp"
#include "json/JsonEditor.hpp"

class MetadataDialog : public QDialog {
    Q_OBJECT

public:
    explicit MetadataDialog(const ImageMetadata& metadata, QWidget* parent = nullptr);
    const ImageMetadata& getMetadata() const { return metadata; }

private slots:
    void editCustomData();
    void saveMetadata();

private:
    QTreeWidget* treeWidget;
    QPushButton* editButton;
    QPushButton* saveButton;
    ImageMetadata metadata;
    JsonEditor* jsonEditor;

    void setupUI();
    void populateTree(const ImageMetadata& metadata);
    void updateCustomDataDisplay();
};

#endif
