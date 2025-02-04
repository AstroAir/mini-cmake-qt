#include "mainwindow.h"
#include "image/ImageIO.hpp"
#include <QDateTime>
#include <QFileInfo>
#include <QImageReader>
#include <QLabel>
#include <QScrollArea>
#include <spdlog/spdlog.h>
#include <unordered_set>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
  spdlog::info("Initializing MainWindow");
  setWindowTitle("File Scan Tool");
  resize(1024, 768); // 增加默认窗口大小

  model = new QFileSystemModel(this);
  model->setRootPath("");

  // 创建主布局容器
  auto centralWidget = new QWidget(this);
  auto mainLayout = new QVBoxLayout(centralWidget);
  mainLayout->setContentsMargins(10, 10, 10, 10); // 添加边距
  mainLayout->setSpacing(10);                     // 增加组件间距

  // 创建工具栏容器
  auto toolbarContainer = new QWidget;
  auto toolbarLayout = new QHBoxLayout(toolbarContainer);
  toolbarLayout->setContentsMargins(0, 0, 0, 0);
  toolbarLayout->setSpacing(8);

  // 优化按钮样式和布局
  scanButton = new QPushButton("选择文件夹", this);
  listButton = new QPushButton("列表显示", this);
  gridButton = new QPushButton("网格显示", this);
  toggleViewButton = new QPushButton("切换显示模式", this);

  // 设置按钮固定高度和最小宽度
  const int buttonHeight = 32;
  const int buttonMinWidth = 100;
  for (auto *btn : {scanButton, listButton, gridButton, toggleViewButton}) {
    btn->setFixedHeight(buttonHeight);
    btn->setMinimumWidth(buttonMinWidth);
    btn->setStyleSheet(R"(
          QPushButton {
              background-color: #f0f0f0;
              border: none;
              border-radius: 4px;
              padding: 6px 12px;
              color: #333333;
          }
          QPushButton:hover {
              background-color: #e0e0e0;
          }
          QPushButton:pressed {
              background-color: #d0d0d0;
          }
      )");
  }

  toolbarLayout->addWidget(scanButton);
  toolbarLayout->addWidget(listButton);
  toolbarLayout->addWidget(gridButton);
  toolbarLayout->addWidget(toggleViewButton);
  toolbarLayout->addStretch(); // 添加弹性空间

  // 创建视图容器
  auto viewContainer = new QWidget;
  auto viewLayout = new QStackedLayout(viewContainer);
  viewLayout->setContentsMargins(0, 0, 0, 0);

  // 优化树形视图
  treeView = new QTreeView;
  treeView->setStyleSheet(R"(
      QTreeView {
          background-color: white;
          border: 1px solid #cccccc;
          border-radius: 4px;
      }
      QTreeView::item {
          padding: 4px;
      }
      QTreeView::item:hover {
          background-color: #f5f5f5;
      }
      QTreeView::item:selected {
          background-color: #0078d4;
          color: white;
      }
  )");

  // 优化网格视图
  gridView = new QWidget;
  gridLayout = new QGridLayout(gridView);
  gridLayout->setContentsMargins(0, 0, 0, 0);
  gridLayout->setSpacing(12); // 增加网格间距

  // 重新初始化文件视图
  fileView = new QTreeView(this);
  fileView->setModel(model);
  fileView->setRootIndex(model->index(""));
  fileView->setSelectionMode(QAbstractItemView::SingleSelection);
  fileView->setAnimated(true);
  fileView->setSortingEnabled(true);
  fileView->setEditTriggers(QAbstractItemView::NoEditTriggers);
  fileView->setColumnWidth(0, 200);

  // 设置列宽和显示的列
  fileView->header()->setStretchLastSection(true);
  fileView->setColumnHidden(2, false); // 显示文件类型
  fileView->setColumnHidden(3, false); // 显示文件大小

  // 设置网格布局属性
  gridLayout->setSpacing(5);
  gridLayout->setAlignment(Qt::AlignLeft | Qt::AlignTop);

  // 使网格视图自动调整大小
  gridView->setMinimumWidth(400);
  gridView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

  viewLayout->addWidget(fileView);
  viewLayout->addWidget(gridView);

  // 组装主布局
  mainLayout->addWidget(toolbarContainer);
  mainLayout->addWidget(viewContainer, 1); // 添加拉伸因子

  setCentralWidget(centralWidget);

  // 设置主窗口样式
  setStyleSheet(R"(
      QMainWindow {
          background-color: #f8f9fa;
      }
      QScrollBar {
          background-color: #f0f0f0;
          width: 12px;
          border-radius: 6px;
      }
      QScrollBar::handle {
          background-color: #c0c0c0;
          border-radius: 5px;
          min-height: 20px;
      }
      QScrollBar::handle:hover {
          background-color: #a0a0a0;
      }
  )");

  connect(scanButton, &QPushButton::clicked, this,
          &MainWindow::onScanButtonClicked);
  connect(listButton, &QPushButton::clicked, this,
          &MainWindow::onListButtonClicked);
  connect(gridButton, &QPushButton::clicked, this,
          &MainWindow::onGridButtonClicked);
  connect(toggleViewButton, &QPushButton::clicked, this,
          &MainWindow::onToggleViewButtonClicked);
  connect(treeView, &QTreeView::doubleClicked, this,
          &MainWindow::onListViewDoubleClicked);

  treeView->hide();
  gridView->hide();
  spdlog::info("MainWindow initialized");

  imagePreviewDialog = new ImagePreviewDialog(this);
  createContextMenu();
  setupConnections();

  // 初始化缓存
  thumbnailCache.setMaxCost(THUMBNAIL_CACHE_SIZE * 1024);

  // 初始化状态栏
  setupStatusBar();
  
  // 初始化线程池
  initThreadPool();
  
  // 不在构造函数中加载目录
  gridView->hide();
  fileView->hide();
}

void MainWindow::initThreadPool() {
  threadPool.setMinThreads(2);
  threadPool.setMaxThreads(MAX_CONCURRENT_LOADS);
  connect(&threadPool, &DynamicThreadPool::threadCountChanged, this,
          [this](int count) {
            spdlog::debug("Thread pool size changed to: {}", count);
          });
}

void MainWindow::setupStatusBar() {
  statusLabel = new QLabel(this);
  progressBar = new QProgressBar(this);
  progressBar->setFixedWidth(200);
  progressBar->hide();
  
  statusBar()->addWidget(statusLabel);
  statusBar()->addPermanentWidget(progressBar);
  updateStatusMessage("就绪");
}

void MainWindow::updateStatusMessage(const QString& message) {
  statusLabel->setText(message);
}

void MainWindow::updateProgress(int value, int total) {
  if (!progressBar->isVisible()) progressBar->show();
  progressBar->setMaximum(total);
  progressBar->setValue(value);
  
  if (value >= total) {
    progressBar->hide();
  }
}

void MainWindow::startLoading() {
  isCancelled = false;
  updateStatusMessage("正在加载...");
  progressBar->show();
  progressBar->setValue(0);
  
  // 禁用相关按钮
  scanButton->setEnabled(false);
  listButton->setEnabled(false);
  gridButton->setEnabled(false);
}

void MainWindow::finishLoading() {
  progressBar->hide();
  updateStatusMessage("就绪");
  
  // 启用按钮
  scanButton->setEnabled(true);
  listButton->setEnabled(true);
  gridButton->setEnabled(true);
}

void MainWindow::onScanButtonClicked() {
  try {
    QString folderPath = QFileDialog::getExistingDirectory(
        this, "选择文件夹", "", QFileDialog::ShowDirsOnly);

    if (folderPath.isEmpty())
      return;

    if (isLoading) {
      isCancelled = true;
      threadPool.cancelAll();
      while (isLoading) {
        QThread::msleep(100);
      }
    }

    currentPath = folderPath;
    currentFolderPath = folderPath;

    // 开始加载
    startLoading();
    isLoading = true;

    auto future = threadPool.enqueue([this, folderPath]() {
      // 在后台线程中扫描文件
      int totalFiles = 0;
      QDir dir(folderPath);
      QFileInfoList entries = dir.entryInfoList(QDir::AllEntries | QDir::NoDotAndDotDot);
      totalFiles = entries.size();
      
      int processedFiles = 0;
      updateImageList();
      model->setRootPath(folderPath);

      for (const auto& entry : entries) {
        if (isCancelled) break;
        
        // 更新进度
        QMetaObject::invokeMethod(this, [this, processedFiles, totalFiles]() {
          updateProgress(processedFiles, totalFiles);
        }, Qt::QueuedConnection);
        
        processedFiles++;
      }

      // 切换到主线程更新UI
      QMetaObject::invokeMethod(this, [this, folderPath]() {
        treeView->setRootIndex(model->index(folderPath));
        if (treeView->isVisible()) {
          switchToListView();
        } else {
          switchToGridView();
        }
        updateNavigationControls();
        isLoading = false;
        finishLoading();
      }, Qt::QueuedConnection);
    });

  } catch (const std::exception &e) {
    spdlog::error("Error in onScanButtonClicked: {}", e.what());
    QMessageBox::critical(this, "错误", 
                         QString("加载文件夹时出错：%1").arg(e.what()));
    finishLoading();
  }
}

void MainWindow::onListButtonClicked() {
  spdlog::info("Switching to list view");
  switchToListView();
}

void MainWindow::onGridButtonClicked() {
  spdlog::info("Switching to grid view");
  switchToGridView();
}

void MainWindow::onToggleViewButtonClicked() {
  spdlog::info("Toggle view button clicked");
  isDetailedMode = !isDetailedMode;
  spdlog::info("Detailed mode set to {}", isDetailedMode);
  auto structure = parse_folder_structure(currentFolderPath.toStdString());
  populateListView(structure);
  populateGridView(structure);
}

void MainWindow::onListViewDoubleClicked(const QModelIndex &index) {
  try {
    spdlog::info("List item double clicked");
    QString filePath = model->filePath(index);
    QFileInfo fileInfo(filePath);

    if (fileInfo.isDir()) {
      // 如果是目录,进入目录
      currentFolderPath = filePath;
      treeView->setRootIndex(model->index(filePath));
      updateNavigationControls();
    } else if (fileInfo.isFile()) {
      // 如果是文件,检查是否为图片
      if (is_image_file(filePath.toStdString())) {
        // 查找当前图片在imageList中的索引
        int currentIndex = imageList.indexOf(filePath);
        if (currentIndex >= 0) {
          imagePreviewDialog->setImageList(imageList, currentIndex);
          imagePreviewDialog->show();
        }
      } else {
        // 非图片文件,仅显示文件信息
        showFileInfo(filePath);
      }
    }
  } catch (const std::exception &e) {
    spdlog::error("Error handling double click: {}", e.what());
    QMessageBox::critical(this, "错误",
                          QString("处理文件时出错：%1").arg(e.what()));
  }
}

void MainWindow::showFileInfo(const QString &filePath) {
  spdlog::info("Showing file info for: {}", filePath.toStdString());
  QFileInfo fileInfo(filePath);

  QString info = QString("文件名: %1\n"
                         "大小: %2\n"
                         "创建时间: %3\n"
                         "修改时间: %4\n"
                         "路径: %5")
                     .arg(fileInfo.fileName())
                     .arg(formatFileSize(fileInfo.size()))
                     .arg(formatDateTime(fileInfo.birthTime()))
                     .arg(formatDateTime(fileInfo.lastModified()))
                     .arg(fileInfo.absoluteFilePath());

  if (is_image_file(filePath.toStdString())) {
    // 使用新的 ImagePreviewDialog 显示图片
    int index = imageList.indexOf(filePath);
    if (index >= 0) {
      imagePreviewDialog->setImageList(imageList, index);
      imagePreviewDialog->show();
    }
  }

  QMessageBox::information(this, "文件信息", info);
}

void MainWindow::showImagePreview(const QString &imagePath) {
  spdlog::info("Showing image preview for: {}", imagePath.toStdString());
  QDialog *previewDialog = new QDialog(this);
  previewDialog->setWindowTitle("图片预览");
  previewDialog->resize(800, 600);

  QVBoxLayout *layout = new QVBoxLayout(previewDialog);
  QScrollArea *scrollArea = new QScrollArea(previewDialog);
  QLabel *imageLabel = new QLabel();

  QImageReader reader(imagePath);
  reader.setAutoTransform(true);
  QImage image = reader.read();

  if (image.isNull()) {
    spdlog::error("Failed to load image: {}", imagePath.toStdString());
    return;
  }

  // 调整图片大小以适应窗口
  QSize scaledSize = image.size();
  scaledSize.scale(780, 580, Qt::KeepAspectRatio);
  imageLabel->setPixmap(QPixmap::fromImage(image).scaled(
      scaledSize, Qt::KeepAspectRatio, Qt::SmoothTransformation));

  scrollArea->setWidget(imageLabel);
  layout->addWidget(scrollArea);

  previewDialog->exec();
}

QString MainWindow::formatFileSize(qint64 size) {
  const QStringList units = {"B", "KB", "MB", "GB", "TB"};
  int unitIndex = 0;
  double fileSize = size;

  while (fileSize >= 1024.0 && unitIndex < units.size() - 1) {
    fileSize /= 1024.0;
    unitIndex++;
  }

  return QString("%1 %2").arg(fileSize, 0, 'f', 2).arg(units[unitIndex]);
}

QString MainWindow::formatDateTime(const QDateTime &datetime) {
  return datetime.toString("yyyy-MM-dd hh:mm:ss");
}

FolderStructure MainWindow::parse_folder_structure(const std::string &root) {
  try {
    spdlog::info("Parsing folder structure for root: {}", root);
    FolderStructure result;

    QDir dir(QString::fromStdString(root));
    if (!dir.exists()) {
      throw std::runtime_error("目录不存在");
    }

    // 获取文件列表
    QFileInfoList entries =
        dir.entryInfoList(QDir::AllEntries | QDir::NoDotAndDotDot);

    DateDirectory currentDate;
    currentDate.date =
        QDateTime::currentDateTime().toString("yyyy-MM-dd").toStdString();

    ImageCategory defaultCategory;
    defaultCategory.name = "默认分类";

    for (const QFileInfo &entry : entries) {
      if (entry.isFile() && is_image_file(entry.filePath().toStdString())) {
        defaultCategory.images.push_back(
            fs::path(entry.filePath().toStdString()));
      }
    }

    if (!defaultCategory.images.empty()) {
      currentDate.categories.push_back(defaultCategory);
      result.dates.push_back(currentDate);
    }

    return result;

  } catch (const std::exception &e) {
    spdlog::error("Error parsing folder structure: {}", e.what());
    throw; // 重新抛出异常，让上层处理
  }
}

bool MainWindow::is_image_file(const fs::path &path) {
  static const std::unordered_set<std::string> valid_extensions{
      ".jpg",  ".jpeg", ".png", ".gif", ".bmp",
      ".webp", ".tiff", ".svg", ".fits"};

  if (!fs::is_regular_file(path))
    return false;

  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  bool isValid = valid_extensions.count(ext) > 0;
  spdlog::info("File {} is {}an image file", path.string(),
               isValid ? "" : "not ");
  return isValid;
}

void MainWindow::populateListView(const FolderStructure &structure) {
  spdlog::info("Populating tree view");
  QStandardItemModel *treeModel = new QStandardItemModel(this);
  treeModel->setHorizontalHeaderLabels({"名称", "修改日期", "类型", "大小"});

  for (const auto &date : structure.dates) {
    for (const auto &category : date.categories) {
      for (const auto &img : category.images) {
        QFileInfo fileInfo(QString::fromStdString(img.string()));
        QList<QStandardItem *> row;

        // 名称列
        auto nameItem = new QStandardItem();
        nameItem->setIcon(style()->standardIcon(QStyle::SP_FileIcon));
        nameItem->setText(fileInfo.fileName());
        row.append(nameItem);

        // 修改日期列
        row.append(new QStandardItem(
            fileInfo.lastModified().toString("yyyy/MM/dd HH:mm")));

        // 类型列
        row.append(new QStandardItem(
            QString("%1文件").arg(fileInfo.suffix().toUpper())));

        // 大小列
        row.append(new QStandardItem(formatFileSize(fileInfo.size())));

        treeModel->appendRow(row);
      }
    }
  }

  treeView->setModel(treeModel);
  treeView->setAlternatingRowColors(true);

  // 设置列宽
  treeView->setColumnWidth(0, 200); // 名称列
  treeView->setColumnWidth(1, 150); // 日期列
  treeView->setColumnWidth(2, 100); // 类型列
  treeView->setColumnWidth(3, 100); // 大小列
  spdlog::info("Tree view populated");
}

void MainWindow::setupGridItemStyle(QPushButton *button) {
  button->setFixedSize(GRID_ITEM_WIDTH, GRID_ITEM_HEIGHT);
  button->setStyleSheet(R"(
      QPushButton {
          background-color: white;
          border: 1px solid #e0e0e0;
          border-radius: 8px;
          padding: 8px;
          text-align: center;
          color: #333333;
      }
      QPushButton:hover {
          background-color: #f5f5f5;
          border-color: #0078d4;
      }
      QPushButton:pressed {
          background-color: #e0e0e0;
      }
  )");

  // 现代化阴影效果
  auto shadow = new QGraphicsDropShadowEffect(button);
  shadow->setBlurRadius(15);
  shadow->setColor(QColor(0, 0, 0, 30));
  shadow->setOffset(0, 2);
  button->setGraphicsEffect(shadow);
}

void MainWindow::createGridItem(const QFileInfo &fileInfo, int row, int col) {
  auto *button = new QPushButton(this);
  setupGridItemStyle(button);

  // 设置图标
  QIcon icon;
  if (fileInfo.isDir()) {
    icon = style()->standardIcon(QStyle::SP_DirIcon);
  } else {
    // 根据文件类型设置不同图标
    if (is_image_file(fileInfo.filePath().toStdString())) {
      icon = style()->standardIcon(QStyle::SP_FileIcon);
    } else {
      icon = style()->standardIcon(QStyle::SP_FileIcon);
    }
  }

  // 设置图标和文本
  button->setIcon(icon);
  button->setIconSize(QSize(ICON_SIZE, ICON_SIZE));

  // 处理长文件名
  QString fileName = fileInfo.fileName();
  if (fileName.length() > 20) {
    fileName = fileName.left(17) + "...";
  }
  button->setText(fileName);
  button->setToolTip(fileInfo.fileName());

  // 存储文件信息
  button->setProperty("filePath", fileInfo.filePath());
  button->setProperty("isDir", fileInfo.isDir());

  // 修改点击事件连接
  if (is_image_file(fileInfo.filePath().toStdString())) {
    connect(button, &QPushButton::clicked, [this, fileInfo]() {
      int index = imageList.indexOf(fileInfo.filePath());
      if (index >= 0) {
        imagePreviewDialog->setImageList(imageList, index);
        imagePreviewDialog->show();
      }
    });
  } else {
    connect(button, &QPushButton::clicked, this,
            &MainWindow::onGridItemDoubleClicked);
  }

  gridLayout->addWidget(button, row, col);
}

void MainWindow::populateGridView(const FolderStructure &structure) {
  // 清除现有布局内容
  QLayoutItem *child;
  while ((child = gridLayout->takeAt(0)) != nullptr) {
    delete child->widget();
    delete child;
  }

  // 重新设置布局属性
  gridLayout->setSpacing(GRID_SPACING);
  gridLayout->setContentsMargins(GRID_SPACING, GRID_SPACING, GRID_SPACING,
                                 GRID_SPACING);

  // 使用当前路径获取文件列表
  QDir dir(currentPath);
  if (!dir.exists()) {
    spdlog::error("Directory does not exist: {}", currentPath.toStdString());
    return;
  }

  // 计算每行可以显示的列数
  int viewWidth = gridView->width();
  int totalItemWidth = GRID_ITEM_WIDTH + GRID_SPACING;
  int columns = std::max(1, (viewWidth - GRID_SPACING) / totalItemWidth);

  int row = 0, col = 0;

  // 添加返回上级目录按钮
  if (!currentPath.isEmpty() && currentPath != QDir::rootPath()) {
    auto *upButton = new QPushButton(this);
    setupGridItemStyle(upButton);
    upButton->setIcon(style()->standardIcon(QStyle::SP_FileDialogToParent));
    upButton->setText("返回上级");
    connect(upButton, &QPushButton::clicked, this,
            &MainWindow::navigateToParentDirectory);
    gridLayout->addWidget(upButton, row, col++);
  }

  const QFileInfoList entries =
      dir.entryInfoList(QDir::AllEntries | QDir::NoDotAndDotDot);

  // 先显示文件夹，再显示文件
  QFileInfoList folders, files;
  for (const QFileInfo &info : entries) {
    if (info.isDir()) {
      folders.append(info);
    } else {
      files.append(info);
    }
  }

  // 显示文件夹
  for (const QFileInfo &folder : folders) {
    if (col >= columns) {
      col = 0;
      row++;
    }
    createGridItem(folder, row, col++);
  }

  // 显示文件
  for (const QFileInfo &file : files) {
    if (col >= columns) {
      col = 0;
      row++;
    }
    createGridItem(file, row, col++);
  }

  // 添加伸缩项以保持网格对齐
  gridLayout->setColumnStretch(columns, 1);

  // 设置网格容器的背景
  gridView->setStyleSheet(R"(
    QWidget {
      background-color: #ffffff;
      border-radius: 10px;
    }
  )");
}

void MainWindow::onGridItemDoubleClicked() {
  QPushButton *button = qobject_cast<QPushButton *>(sender());
  if (!button)
    return;

  QString filePath = button->property("filePath").toString();
  bool isDir = button->property("isDir").toBool();

  if (isDir) {
    currentPath = filePath;
    loadCurrentDirectory();
  } else {
    showFileInfo(filePath);
  }
}

void MainWindow::navigateToParentDirectory() {
  QDir dir(currentPath);
  if (dir.cdUp()) {
    currentPath = dir.absolutePath();
    loadCurrentDirectory();
  }
}

void MainWindow::loadCurrentDirectory() {
  try {
    if (currentPath.isEmpty()) {
      throw std::runtime_error("未选择有效的文件夹");
    }

    QDir dir(currentPath);
    if (!dir.exists()) {
      throw std::runtime_error("文件夹不存在");
    }

    if (!dir.isReadable()) {
      throw std::runtime_error("无法访问文件夹，请检查权限");
    }

    // 更新图片列表
    updateImageList();

    // 更新树形视图
    treeView->setRootIndex(model->index(currentPath));

    // 更新网格视图
    populateGridView(parse_folder_structure(currentPath.toStdString()));

    // 确保网格视图可见性正确
    if (gridView->isVisible()) {
      gridView->update();
    }

    // 更新导航控件
    updateNavigationControls();

    spdlog::info("Successfully loaded directory: {}",
                 currentPath.toStdString());
  } catch (const std::exception &e) {
    spdlog::error("Error in loadCurrentDirectory: {}", e.what());
    QMessageBox::warning(this, "错误",
                         QString("加载目录时出错：%1").arg(e.what()));
  }
}

void MainWindow::updateNavigationControls() {
  setWindowTitle(QString("File Scan Tool - %1").arg(currentPath));
}

QVector<QString> MainWindow::getImageListFromCurrentFolder() {
  return imageList;
}

QString MainWindow::formatFileInfo(const QString &path) {
  QFileInfo info(path);
  return QString("%1 (%2, %3)")
      .arg(info.fileName())
      .arg(info.size() / 1024.0, 0, 'f', 1)
      .arg(info.suffix().toUpper());
}

void MainWindow::createContextMenu() {
  contextMenu = new QMenu(this);
  contextMenu->addAction("打开", this, &MainWindow::openSelectedFile);
  contextMenu->addAction("打开所在文件夹", this,
                         &MainWindow::openContainingFolder);
  contextMenu->addSeparator();
  contextMenu->addAction("重命名", this, &MainWindow::renameSelectedFile);
  contextMenu->addAction("删除", this, &MainWindow::deleteSelectedFile);
  contextMenu->addSeparator();
  contextMenu->addAction("属性", this, &MainWindow::showFileProperties);
}

void MainWindow::setupConnections() {
  // 设置右键菜单触发
  treeView->setContextMenuPolicy(Qt::CustomContextMenu);
  gridView->setContextMenuPolicy(Qt::CustomContextMenu);

  // 修改连接，使用通用的 showContextMenu
  connect(treeView, &QTreeView::customContextMenuRequested, this,
          &MainWindow::showContextMenu);
  connect(gridView, &QWidget::customContextMenuRequested, this,
          &MainWindow::showContextMenu);
}

void MainWindow::showListContextMenu(const QPoint &pos) {
  QModelIndex index = treeView->indexAt(pos);
  if (index.isValid()) {
    QString filePath = model->filePath(index);
    if (!filePath.isEmpty()) {
      // 设置当前选中项
      treeView->setCurrentIndex(index);
      contextMenu->exec(treeView->viewport()->mapToGlobal(pos));
    }
  }
}

void MainWindow::showGridContextMenu(const QPoint &pos) {
  QWidget *widget = gridView->childAt(pos);
  if (QPushButton *button = qobject_cast<QPushButton *>(widget)) {
    QString filePath = button->property("filePath").toString();
    if (!filePath.isEmpty()) {
      // 设置焦点到当前按钮
      button->setFocus();
      contextMenu->exec(gridView->mapToGlobal(pos));
    }
  }
}

void MainWindow::showContextMenu(const QPoint &pos) {
  // 确定当前是列表视图还是网格视图
  if (treeView->isVisible()) {
    showListContextMenu(pos);
  } else if (gridView->isVisible()) {
    showGridContextMenu(pos);
  }
}

void MainWindow::openSelectedFile() {
  QString filePath = getSelectedFilePath();
  if (filePath.isEmpty())
    return;

  if (is_image_file(filePath.toStdString())) {
    int index = imageList.indexOf(filePath);
    if (index >= 0) {
      imagePreviewDialog->setImageList(imageList, index);
      imagePreviewDialog->show();
    }
  }
}

void MainWindow::openContainingFolder() {
  QString filePath = getSelectedFilePath();
  if (filePath.isEmpty())
    return;

  QFileInfo fileInfo(filePath);
  QDesktopServices::openUrl(QUrl::fromLocalFile(fileInfo.absolutePath()));
}

void MainWindow::deleteSelectedFile() {
  QString filePath = getSelectedFilePath();
  if (filePath.isEmpty())
    return;

  QMessageBox::StandardButton reply =
      QMessageBox::question(this, "确认删除", "确定要删除这个文件吗？",
                            QMessageBox::Yes | QMessageBox::No);

  if (reply == QMessageBox::Yes) {
    QFile file(filePath);
    if (file.remove()) {
      // 刷新视图
      loadCurrentDirectory();
    } else {
      QMessageBox::warning(this, "错误", "无法删除文件");
    }
  }
}

void MainWindow::renameSelectedFile() {
  QString filePath = getSelectedFilePath();
  if (filePath.isEmpty())
    return;

  QFileInfo fileInfo(filePath);
  bool ok;
  QString newName =
      QInputDialog::getText(this, "重命名", "输入新的文件名：",
                            QLineEdit::Normal, fileInfo.fileName(), &ok);

  if (ok && !newName.isEmpty()) {
    QString newPath = fileInfo.absolutePath() + "/" + newName;
    QFile file(filePath);
    if (file.rename(newPath)) {
      // 刷新视图
      loadCurrentDirectory();
    } else {
      QMessageBox::warning(this, "错误", "无法重命名文件");
    }
  }
}

void MainWindow::showFileProperties() {
  QString filePath = getSelectedFilePath();
  if (!filePath.isEmpty()) {
    showFileInfo(filePath);
  }
}

QString MainWindow::getSelectedFilePath() {
  if (treeView->isVisible()) {
    QModelIndex index = treeView->currentIndex();
    if (index.isValid()) {
      return model->filePath(index);
    }
  } else if (gridView->isVisible()) {
    QWidget *focusWidget = gridView->focusWidget();
    if (QPushButton *button = qobject_cast<QPushButton *>(focusWidget)) {
      return button->property("filePath").toString();
    }
  }
  return QString();
}

void MainWindow::createThumbnailGrid() {
  // 新增：采用网格布局展示缩略图
  QDialog *gridDialog = new QDialog(this);
  gridDialog->setWindowTitle("缩略图网格");
  QGridLayout *gridLayout = new QGridLayout(gridDialog);

  // 假设存在成员变量 imageList，存放所有图片路径
  int cols = 4;
  int row = 0, col = 0;
  for (const QString &imgPath : imageList) {
    QLabel *thumbLabel = new QLabel;
    QPixmap pix;
    // 如果是 FITS 文件调用现有io模块加载
    if (imgPath.endsWith(".fits", Qt::CaseInsensitive)) {
      // 利用 ImageIO 接口加载 FITS 文件（需要自行封装 QImage 从 cv::Mat 转换）
      cv::Mat mat = loadImage(imgPath.toStdString());
      if (!mat.empty()) {
        QImage qImg(mat.data, mat.cols, mat.rows, mat.step,
                    QImage::Format_Grayscale8);
        pix = QPixmap::fromImage(qImg).scaled(100, 100, Qt::KeepAspectRatio,
                                              Qt::SmoothTransformation);
      }
    } else {
      pix.load(imgPath);
      if (!pix.isNull())
        pix =
            pix.scaled(100, 100, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }
    thumbLabel->setPixmap(pix);
    gridLayout->addWidget(thumbLabel, row, col);
    if (++col >= cols) {
      col = 0;
      row++;
    }
  }
  gridDialog->setLayout(gridLayout);
  gridDialog->exec();
}

void MainWindow::addImageOverlay(const QPixmap &overlay) {
  // 获取当前显示的图片
  QString currentPath = getSelectedFilePath();
  if (currentPath.isEmpty() || !imagePreviewDialog) {
    QMessageBox::warning(this, "错误", "请先选择一张图片");
    return;
  }

  // 创建叠加设置对话框
  QDialog settingsDialog(this);
  settingsDialog.setWindowTitle("图像叠加设置");
  auto layout = new QVBoxLayout(&settingsDialog);

  // 添加透明度设置
  auto opacitySlider = new QSlider(Qt::Horizontal);
  opacitySlider->setRange(0, 100);
  opacitySlider->setValue(50);
  layout->addWidget(new QLabel("透明度:"));
  layout->addWidget(opacitySlider);

  // 添加位置调整
  auto posXSpinBox = new QSpinBox;
  auto posYSpinBox = new QSpinBox;
  posXSpinBox->setRange(-1000, 1000);
  posYSpinBox->setRange(-1000, 1000);

  auto posLayout = new QHBoxLayout;
  posLayout->addWidget(new QLabel("X:"));
  posLayout->addWidget(posXSpinBox);
  posLayout->addWidget(new QLabel("Y:"));
  posLayout->addWidget(posYSpinBox);
  layout->addLayout(posLayout);

  // 添加预览功能
  auto previewLabel = new QLabel;
  previewLabel->setMinimumSize(200, 200);
  layout->addWidget(previewLabel);

  // 添加确定和取消按钮
  auto buttonBox = new QDialogButtonBox(
      QDialogButtonBox::Ok | QDialogButtonBox::Cancel, Qt::Horizontal);
  layout->addWidget(buttonBox);

  // 实时预览效果
  auto updatePreview = [&]() {
    QImage baseImage(currentPath);
    if (baseImage.isNull())
      return;

    // 创建预览图
    QImage preview = baseImage.scaled(200, 200, Qt::KeepAspectRatio);
    QImage overlayScaled =
        overlay.toImage().scaled(preview.size(), Qt::KeepAspectRatio);

    // 应用透明度和位置偏移
    double opacity = opacitySlider->value() / 100.0;
    QPainter painter(&preview);
    painter.setOpacity(opacity);
    painter.drawImage(posXSpinBox->value(), posYSpinBox->value(),
                      overlayScaled);

    previewLabel->setPixmap(QPixmap::fromImage(preview));
  };

  // 连接信号槽
  connect(opacitySlider, &QSlider::valueChanged, updatePreview);
  connect(posXSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
          updatePreview);
  connect(posYSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
          updatePreview);
  connect(buttonBox, &QDialogButtonBox::accepted, &settingsDialog,
          &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, &settingsDialog,
          &QDialog::reject);

  // 初始预览
  updatePreview();

  // 如果用户确认设置
  if (settingsDialog.exec() == QDialog::Accepted) {
    try {
      // 加载原始图像
      QImage baseImage(currentPath);
      if (baseImage.isNull()) {
        throw std::runtime_error("无法加载基础图像");
      }

      // 创建新的图像用于叠加
      QImage result = baseImage;
      QPainter painter(&result);

      // 应用透明度设置
      painter.setOpacity(opacitySlider->value() / 100.0);

      // 调整叠加图像大小以匹配原图
      QImage overlayScaled =
          overlay.toImage().scaled(baseImage.size(), Qt::KeepAspectRatio);

      // 在指定位置绘制叠加图像
      painter.drawImage(posXSpinBox->value(), posYSpinBox->value(),
                        overlayScaled);

      // 保存结果
      QString savePath = currentPath;
      savePath.insert(savePath.lastIndexOf('.'), "_overlay");

      if (result.save(savePath)) {
        QMessageBox::information(this, "成功", "叠加图像已保存至: " + savePath);

        // 刷新图像列表和显示
        updateImageList();
        loadCurrentDirectory();
      } else {
        throw std::runtime_error("保存结果图像失败");
      }

    } catch (const std::exception &e) {
      QMessageBox::critical(this, "错误",
                            QString("图像叠加处理失败: %1").arg(e.what()));
    }
  }
}

void MainWindow::updateImageList() {
  imageList.clear();
  QDir dir(currentPath);
  filterImageFiles(dir);

  // 更新UI状态
  if (!imageList.isEmpty()) {
    spdlog::info("Found {} images in directory", imageList.size());
  }
}

void MainWindow::filterImageFiles(const QDir &dir) {
  QStringList filters;
  filters << "*.jpg" << "*.jpeg" << "*.png" << "*.gif" << "*.bmp" << "*.fits";

  QFileInfoList entries =
      dir.entryInfoList(filters, QDir::Files | QDir::NoDotAndDotDot);
  for (const QFileInfo &fileInfo : entries) {
    imageList.append(fileInfo.absoluteFilePath());
  }
}

void MainWindow::loadThumbnail(const QString &path, QPushButton *button) {
  // 检查缓存
  if (auto *cached = thumbnailCache.object(path)) {
    button->setIcon(QIcon(*cached));
    return;
  }

  // 异步加载缩略图
  threadPool.enqueue([this, path, button]() {
    QPixmap pix;
    if (path.endsWith(".fits", Qt::CaseInsensitive)) {
      cv::Mat mat = loadImage(path.toStdString());
      if (!mat.empty()) {
        QImage qImg(mat.data, mat.cols, mat.rows, mat.step,
                    QImage::Format_Grayscale8);
        pix = QPixmap::fromImage(qImg);
      }
    } else {
      pix.load(path);
    }

    if (!pix.isNull()) {
      pix = pix.scaled(ICON_SIZE, ICON_SIZE, Qt::KeepAspectRatio,
                       Qt::SmoothTransformation);

      // 缓存缩略图
      thumbnailCache.insert(path, new QPixmap(pix),
                            pix.width() * pix.height() * pix.depth() / 8 /
                                1024);

      // 在主线程中更新UI
      QMetaObject::invokeMethod(
          this, [button, pix]() { button->setIcon(QIcon(pix)); },
          Qt::QueuedConnection);
    }
  });
}

void MainWindow::switchToListView() {
  clearCurrentView();
  treeView->show(); // 显示列表视图（treeView）
  gridView->hide();
  treeView->setFocus();
  if (!currentPath.isEmpty()) {
    treeView->setRootIndex(model->index(currentPath));
  }
}

void MainWindow::switchToGridView() {
  clearCurrentView();
  gridView->show();
  treeView->hide(); // 隐藏列表视图
  fileView->hide(); // 隐藏不使用的 fileView
  loadCurrentDirectory();
}

void MainWindow::clearCurrentView() {
  // 清理网格视图
  QLayoutItem *child;
  while ((child = gridLayout->takeAt(0)) != nullptr) {
    delete child->widget();
    delete child;
  }

  // 重置进度
  loadingProgress = 0;
}
