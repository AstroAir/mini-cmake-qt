#include "mainwindow.h"
#include <QDateTime>
#include <QFileInfo>
#include <QImageReader>
#include <QLabel>
#include <QScrollArea>
#include <spdlog/spdlog.h>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
  spdlog::info("Initializing MainWindow");
  setWindowTitle("File Scan Tool");
  resize(800, 600);

  model = new QFileSystemModel(this);
  model->setRootPath("");

  treeView = new QTreeView(this);
  gridView = new QWidget(this);
  gridLayout = new QGridLayout(gridView);

  scanButton = new QPushButton("选择文件夹", this);
  listButton = new QPushButton("列表显示", this);
  gridButton = new QPushButton("网格显示", this);
  toggleViewButton = new QPushButton("切换模式 (简略/详细)", this);

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

  QVBoxLayout *mainLayout = new QVBoxLayout();
  QHBoxLayout *buttonLayout = new QHBoxLayout();
  buttonLayout->addWidget(scanButton);
  buttonLayout->addWidget(listButton);
  buttonLayout->addWidget(gridButton);
  buttonLayout->addWidget(toggleViewButton);

  mainLayout->addLayout(buttonLayout);
  mainLayout->addWidget(treeView);
  mainLayout->addWidget(gridView);

  QWidget *centralWidget = new QWidget(this);
  centralWidget->setLayout(mainLayout);
  setCentralWidget(centralWidget);

  treeView->hide();
  gridView->hide();
  spdlog::info("MainWindow initialized");

  imagePreviewDialog = new ImagePreviewDialog(this);
  createContextMenu();
  setupConnections();
}

void MainWindow::onScanButtonClicked() {
  try {
    QString folderPath = QFileDialog::getExistingDirectory(
        this, "选择文件夹", "", QFileDialog::ShowDirsOnly);

    if (folderPath.isEmpty()) {
      spdlog::warn("No folder selected");
      return;
    }

    // 验证文件夹是否可访问
    QDir dir(folderPath);
    if (!dir.exists()) {
      throw std::runtime_error("所选文件夹不存在");
    }

    if (!dir.isReadable()) {
      throw std::runtime_error("无法访问所选文件夹，请检查权限");
    }

    currentPath = folderPath;
    currentFolderPath = folderPath;

    // 设置文件系统模型
    model->setRootPath(folderPath);
    treeView->setModel(model);
    treeView->setRootIndex(model->index(folderPath));

    // 设置列宽
    treeView->setColumnWidth(0, 200);
    treeView->setAlternatingRowColors(true);

    // 自动切换到列表视图
    treeView->show();
    gridView->hide();

    spdlog::info("Successfully loaded folder: {}", folderPath.toStdString());

    // 更新窗口标题
    updateNavigationControls();

  } catch (const std::exception &e) {
    spdlog::error("Error in onScanButtonClicked: {}", e.what());
    QMessageBox::critical(this, "错误",
                          QString("加载文件夹时出错：%1").arg(e.what()));
  }
}

void MainWindow::onListButtonClicked() {
  spdlog::info("List button clicked");
  treeView->show();
  gridView->hide();
}

void MainWindow::onGridButtonClicked() {
  spdlog::info("Grid button clicked");
  treeView->hide();
  gridView->show();

  // 确保网格视图内容是最新的
  if (!currentPath.isEmpty()) {
    loadCurrentDirectory();
  }
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
        // 获取当前文件夹下的所有图片
        QDir dir = fileInfo.dir();
        QStringList filters;
        filters << "*.jpg" << "*.jpeg" << "*.png" << "*.gif" << "*.bmp";
        QFileInfoList files = dir.entryInfoList(filters, QDir::Files);
        
        // 构建图片列表
        QVector<QString> imageList;
        int currentIndex = 0;
        for(int i = 0; i < files.size(); i++) {
          imageList.append(files[i].filePath());
          if(files[i].filePath() == filePath) {
            currentIndex = i;
          }
        }
        
        // 打开图片预览对话框
        if(imagePreviewDialog) {
          imagePreviewDialog->setImageList(imageList, currentIndex); 
          imagePreviewDialog->show();
          spdlog::info("Image preview opened for: {}", filePath.toStdString());
        } else {
          throw std::runtime_error("图片预览对话框未初始化");
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
    QVector<QString> images = getImageListFromCurrentFolder();
    int index = images.indexOf(filePath);
    imagePreviewDialog->setImageList(images, index);
    imagePreviewDialog->show();
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

  // Windows 11风格的样式表
  button->setStyleSheet(R"(
        QPushButton {
            background-color: transparent;
            border: none;
            border-radius: 8px;
            padding: 8px;
            text-align: center;
            color: #000000;
        }
        QPushButton:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
        QPushButton:pressed {
            background-color: rgba(0, 0, 0, 0.1);
        }
    )");

  // 设置文本对齐和换行
  button->setStyleSheet(button->styleSheet() +
                        "\nQPushButton { white-space: normal; }");

  // 添加阴影效果
  auto *shadow = new QGraphicsDropShadowEffect(button);
  shadow->setBlurRadius(10);
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
      QVector<QString> images = getImageListFromCurrentFolder();
      int index = images.indexOf(fileInfo.filePath());
      imagePreviewDialog->setImageList(images, index);
      imagePreviewDialog->show();
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

  int row = 0, col = 0;
  const int COLS = 5; // 增加列数以适应更小的图标

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
    if (col >= COLS) {
      col = 0;
      row++;
    }
    createGridItem(folder, row, col++);
  }

  // 显示文件
  for (const QFileInfo &file : files) {
    if (col >= COLS) {
      col = 0;
      row++;
    }
    createGridItem(file, row, col++);
  }

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
  QVector<QString> imageList;
  QDir dir(currentPath);
  QFileInfoList entries = dir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);

  for (const QFileInfo &fileInfo : entries) {
    if (is_image_file(fileInfo.filePath().toStdString())) {
      imageList.append(fileInfo.filePath());
    }
  }
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
    QVector<QString> images = getImageListFromCurrentFolder();
    int index = images.indexOf(filePath);
    imagePreviewDialog->setImageList(images, index);
    imagePreviewDialog->show();
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