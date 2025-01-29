#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QApplication>
#include <QFileDialog>
#include <QFileSystemModel>
#include <QGraphicsDropShadowEffect>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QListView>
#include <QMainWindow>
#include <QMessageBox>
#include <QPushButton>
#include <QStandardItemModel>
#include <QStyleOption>
#include <QTreeView>
#include <QVBoxLayout>

#include <chrono>
#include <filesystem>
#include <sstream>
#include <unordered_set>

#include "ImagePreviewDialog.h"

namespace fs = std::filesystem;

struct ImageCategory {
  std::string name;
  std::vector<fs::path> images;
};

struct DateDirectory {
  std::string date;
  std::vector<ImageCategory> categories;
};

struct FolderStructure {
  std::vector<DateDirectory> dates;
};

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  MainWindow(QWidget *parent = nullptr);

private slots:
  void onScanButtonClicked();
  void onListButtonClicked();
  void onGridButtonClicked();
  void onToggleViewButtonClicked();
  void onListViewDoubleClicked(const QModelIndex &index); // 新增：处理双击事件
  void showFileInfo(const QString &filePath);             // 新增：显示文件信息
  void onGridItemDoubleClicked();
  void navigateToParentDirectory();
  void showContextMenu(const QPoint &pos);
  void showListContextMenu(const QPoint &pos);
  void showGridContextMenu(const QPoint &pos);
  void openSelectedFile();
  void openContainingFolder();
  void deleteSelectedFile();
  void renameSelectedFile();
  void showFileProperties();

private:
  QListView *listView;
  QWidget *gridView;
  QGridLayout *gridLayout;
  QPushButton *scanButton;
  QPushButton *listButton;
  QPushButton *gridButton;
  QPushButton *toggleViewButton;
  QFileSystemModel *model;
  QString currentFolderPath;
  QTreeView *treeView;

  bool isDetailedMode = false;

  FolderStructure parse_folder_structure(const std::string &root);
  bool is_image_file(const fs::path &path);
  void populateListView(const FolderStructure &structure);
  void populateGridView(const FolderStructure &structure);
  void showImagePreview(const QString &imagePath);   // 新增：图片预览
  QString formatFileSize(qint64 size);               // 新增：文件大小格式化
  QString formatDateTime(const QDateTime &datetime); // 新增：日期时间格式化
  QStandardItemModel *gridModel;
  QString currentPath;
  void updateNavigationControls();
  void loadCurrentDirectory();
  void createGridItem(const QFileInfo &fileInfo, int row, int col);
  void setupGridItemStyle(QPushButton *button);
  void createHoverEffect(QPushButton *button);
  static const int GRID_ITEM_WIDTH = 120;
  static const int GRID_ITEM_HEIGHT = 100;
  static const int GRID_SPACING = 15;
  static const int ICON_SIZE = 48;

  // 移除旧的预览相关方法
  // void showImagePreviewDialog(const QString &imagePath);
  // void updateImagePreview(QLabel *imageLabel, const QString &imagePath);

  // 添加新的成员变量
  ImagePreviewDialog *imagePreviewDialog;

  // 添加新的成员变量
  QVector<QString> currentImageList; // 当前文件夹中的所有图片
  int currentImageIndex;             // 当前显示的图片索引

  // 删除这些不再需要的方法声明
  // void showImagePreviewDialog(const QString &imagePath, bool useImageList =
  // false); void showNextImage(); void showPreviousImage(); void
  // updateImagePreview(QLabel *imageLabel, const QString &imagePath);

  // 只保留这两个实用方法
  QVector<QString> getImageListFromCurrentFolder();
  QString formatFileInfo(const QString &path);

  QMenu *contextMenu;
  void createContextMenu();
  QString getSelectedFilePath();
  void setupConnections();
};

#endif // MAINWINDOW_H