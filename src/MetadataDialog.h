#ifndef METADATADIALOG_H
#define METADATADIALOG_H

#include "image/MetaData.hpp"
#include "json/JsonEditor.hpp"
#include <QDialog>

class ElaLineEdit;
class ElaPushButton;
class ElaTreeView;
class QStandardItem;
class QStandardItemModel;

class MetadataDialog : public QDialog {
  Q_OBJECT

public:
  explicit MetadataDialog(const ImageMetadata &metadata,
                          QWidget *parent = nullptr);
  const ImageMetadata &getMetadata() const { return metadata; }

private slots:
  void editCustomData();
  void saveMetadata();
  void exportMetadata();
  void importMetadata();
  void resetMetadata();
  void searchMetadata();

private:
  ElaTreeView *treeView;
  QStandardItemModel *model;
  ElaPushButton *editButton;
  ElaPushButton *saveButton;
  ElaPushButton *exportButton;
  ElaPushButton *importButton;
  ElaPushButton *resetButton;
  ElaLineEdit *searchBox;
  ImageMetadata metadata;
  JsonEditor *jsonEditor;

  void setupUI();
  void populateTree(const ImageMetadata &metadata);
  void updateCustomDataDisplay();
  void filterTreeItems(const QString &searchText);
  void saveMetadataToFile(const QString &filePath);
  void loadMetadataFromFile(const QString &filePath);
  bool filterTreeItem(QStandardItem *item, const QString &searchText);
};

#endif
