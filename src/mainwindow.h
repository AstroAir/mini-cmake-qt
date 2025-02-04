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

#include <filesystem>

#include "ImagePreviewDialog.h"
#include "utils/ThreadPool.hpp"
#include <QCache>

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
  void createThumbnailGrid();    // 新增：自动创建网格模式缩略图
  void addImageOverlay(const QPixmap &overlay);  // 新增：叠加图像（用于功能扩展）

private:
  QTreeView *fileView;  // 将 listView 改为 fileView
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
  QVector<QString> imageList;  // 当前目录下的所有图片文件列表
  void updateImageList();      // 更新图片列表
  void filterImageFiles(const QDir &dir); // 过滤图片文件

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

  // 新增：线程池和缓存相关成员
  DynamicThreadPool threadPool;
  QCache<QString, QPixmap> thumbnailCache;
  QProgressDialog *progressDialog;
  int loadingProgress = 0;
  std::atomic<bool> isLoading{false};
  
  void initThreadPool();
  void loadThumbnail(const QString &path, QPushButton *button);
  void updateLoadingProgress();
  void switchToListView();
  void switchToGridView();
  void clearCurrentView();
  
  static constexpr int THUMBNAIL_CACHE_SIZE = 1000; // 缓存大小（单位：MB）
  static constexpr int MAX_CONCURRENT_LOADS = 4;    // 最大并发加载数

  QLabel* statusLabel;      // 状态标签
  QProgressBar* progressBar;  // 进度条
  void setupStatusBar();    // 设置状态栏
  void updateStatusMessage(const QString& message); // 更新状态消息
  void updateProgress(int value, int total);       // 更新进度
  void startLoading();     // 开始加载
  void finishLoading();    // 完成加载
  
  std::atomic<bool> isCancelled{false};  // 取消标志
};

#endif // MAINWINDOW_H