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
    void exportMetadata();
    void importMetadata();
    void resetMetadata();
    void searchMetadata();

private:
    QTreeWidget* treeWidget;
    QPushButton* editButton;
    QPushButton* saveButton;
    QPushButton* exportButton;
    QPushButton* importButton;
    QPushButton* resetButton;
    QLineEdit* searchBox;
    ImageMetadata metadata;
    JsonEditor* jsonEditor;

    void setupUI();
    void populateTree(const ImageMetadata& metadata);
    void updateCustomDataDisplay();
    void filterTreeItems(const QString& searchText);
    void saveMetadataToFile(const QString& filePath);
    void loadMetadataFromFile(const QString& filePath);
};

#endif
