#include "VersionControlWidget.h"
#include "CropPreviewWidget.h"

#include "ElaComboBox.h"
#include "ElaDockWidget.h"
#include "ElaLineEdit.h"
#include "ElaListView.h"
#include "ElaPushButton.h"
#include "ElaSlider.h"
#include "ElaStatusBar.h"
#include "ElaToolButton.h"
#include "ElaTreeView.h"
#include <ElaCheckBox.h>


#include <QContextMenuEvent>
#include <QFileDialog>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProgressDialog>
#include <QSettings>
#include <QSplitter>
#include <QStandardItemModel>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>


#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

VersionControlWidget::VersionControlWidget(QWidget *parent)
    : QWidget(parent), versionControl(std::make_unique<ImageVersionControl>()),
      splitOrientation(Qt::Horizontal), historyPanelVisible(true),
      infoPanelVisible(true) {
  createActions();  // 首先创建动作
  setupModels();    // 然后设置模型
  setupLayout();    // 接着设置布局
  connectSignals(); // 最后连接信号
  createMenus();
  loadSettings();
  updateBranchList(); // 更新列表
  updateTagList();
}

void VersionControlWidget::setupUi() {
  // 删除此方法中的重复调用
  setupLayout();
}

void VersionControlWidget::setupLayout() {
  auto mainLayout = new QVBoxLayout(this);
  mainLayout->setSpacing(0);
  mainLayout->setContentsMargins(0, 0, 0, 0);

  // 工具栏区域
  auto toolbarArea = new QWidget;
  auto toolbarLayout = new QHBoxLayout(toolbarArea);
  toolbarLayout->setSpacing(2);
  toolbarLayout->setContentsMargins(5, 2, 5, 2);
  setupToolbar();
  mainLayout->addWidget(toolbarArea);

  // 添加搜索框
  auto searchWidget = setupSearchWidget();
  toolbarLayout->addWidget(searchWidget);

  // 主分割器
  mainSplitter = new QSplitter(this);
  mainSplitter->setHandleWidth(1);

  // 历史面板
  historyDock = new ElaDockWidget(tr("历史记录"), this);
  historyDock->setFeatures(ElaDockWidget::DockWidgetMovable |
                           ElaDockWidget::DockWidgetFloatable);
  historyDock->setWidget(createHistoryPanel());

  // 信息面板
  infoDock = new ElaDockWidget(tr("详细信息"), this);
  infoDock->setFeatures(ElaDockWidget::DockWidgetMovable |
                        ElaDockWidget::DockWidgetFloatable);
  infoDock->setWidget(createInfoPanel());

  // 预览区域
  auto previewArea = createPreviewArea();

  // 添加右侧分割器
  rightSplitter = new QSplitter(Qt::Vertical);
  rightSplitter->addWidget(previewArea);
  rightSplitter->addWidget(infoDock);

  mainSplitter->addWidget(historyDock);
  mainSplitter->addWidget(rightSplitter);

  mainLayout->addWidget(mainSplitter, 1);

  // 底部按钮区域
  auto buttonBox = new QHBoxLayout;
  commitButton = new ElaPushButton(tr("提交"));
  branchButton = new ElaPushButton(tr("新建分支"));
  tagButton = new ElaPushButton(tr("新建标签"));
  mergeButton = new ElaPushButton(tr("合并分支"));
  checkoutButton = new ElaPushButton(tr("检出"));

  buttonBox->addStretch();
  buttonBox->addWidget(commitButton);
  buttonBox->addWidget(branchButton);
  buttonBox->addWidget(tagButton);
  buttonBox->addWidget(mergeButton);
  buttonBox->addWidget(checkoutButton);
  mainLayout->addLayout(buttonBox);

  // 添加状态栏
  setupStatusBar();
  mainLayout->addWidget(statusBar);

  // 设置拖拽
  setupDragDrop();

  // 恢复分割器状态
  restoreSplitterState();
}

QWidget *VersionControlWidget::setupSearchWidget() {
  auto container = new QWidget;
  auto layout = new QHBoxLayout(container);
  layout->setContentsMargins(0, 0, 0, 0);

  searchBox = new ElaLineEdit;
  searchBox->setPlaceholderText(tr("搜索提交..."));
  searchBox->setClearButtonEnabled(true);

  auto searchButton = new ElaToolButton;
  searchButton->setIcon(QIcon::fromTheme("search"));

  layout->addWidget(searchBox);
  layout->addWidget(searchButton);

  connect(searchBox, &QLineEdit::textChanged, this,
          &VersionControlWidget::onFilterChanged);

  return container;
}

void VersionControlWidget::setupStatusBar() {
  statusBar = new ElaStatusBar(this);
  statusLabel = new QLabel(tr("就绪"));
  statusBar->addWidget(statusLabel);

  // 添加主题选择器
  themeSelector = new ElaComboBox;
  themeSelector->addItems({tr("浅色"), tr("深色"), tr("跟随系统")});
  statusBar->addPermanentWidget(themeSelector);

  // 添加预览质量滑块
  qualitySlider = new ElaSlider(Qt::Horizontal);
  qualitySlider->setRange(10, 100);
  qualitySlider->setValue(settings.previewQuality);
  qualitySlider->setFixedWidth(100);
  statusBar->addPermanentWidget(new QLabel(tr("预览质量:")));
  statusBar->addPermanentWidget(qualitySlider);

  connect(qualitySlider, &QSlider::valueChanged, this,
          &VersionControlWidget::updatePreviewQuality);
}

void VersionControlWidget::setupDragDrop() {
  setAcceptDrops(true);
  imagePreview->setAcceptDrops(true);
}

void VersionControlWidget::connectSignals() {
  connect(commitButton, &QPushButton::clicked, this,
          &VersionControlWidget::onCommit);
  connect(branchButton, &QPushButton::clicked, this,
          &VersionControlWidget::onCreateBranch);
  connect(tagButton, &QPushButton::clicked, this,
          &VersionControlWidget::onCreateTag);
  connect(mergeButton, &QPushButton::clicked, this,
          &VersionControlWidget::onMergeBranch);
  connect(checkoutButton, &QPushButton::clicked, this,
          &VersionControlWidget::onCheckout);

  connect(compareButton, &QToolButton::clicked, this,
          &VersionControlWidget::onCompareVersions);
  connect(exportButton, &QToolButton::clicked, this,
          &VersionControlWidget::onExportCommit);
  connect(refreshButton, &QToolButton::clicked, this,
          &VersionControlWidget::refreshHistory);

  // 更新信号连接方式
  connect(historyTree->selectionModel(), &QItemSelectionModel::currentChanged,
          this, [this](const QModelIndex &current, const QModelIndex &) {
            if (current.isValid()) {
              showCommitInfo(current.data().toString());
            }
          });

  connect(actions.resolveConflicts, &QAction::triggered, this,
          &VersionControlWidget::onResolveConflicts);
  connect(actions.viewDiff, &QAction::triggered, this,
          &VersionControlWidget::onViewDiff);
  connect(actions.addMetadata, &QAction::triggered, this,
          &VersionControlWidget::onAddMetadata);
  connect(actions.editMetadata, &QAction::triggered, this,
          &VersionControlWidget::onEditMetadata);
  connect(actions.deleteTag, &QAction::triggered, this,
          &VersionControlWidget::onDeleteTag);
  connect(actions.exportVersion, &QAction::triggered, this,
          &VersionControlWidget::onExportVersion);
  connect(actions.importVersion, &QAction::triggered, this,
          &VersionControlWidget::onImportVersion);
  connect(actions.searchHistory, &QAction::triggered, this,
          &VersionControlWidget::onSearchHistory);
  connect(actions.revertCommit, &QAction::triggered, this,
          &VersionControlWidget::onRevertCommit);
  connect(actions.createPatch, &QAction::triggered, this,
          &VersionControlWidget::onCreatePatch);
  connect(actions.applyPatch, &QAction::triggered, this,
          &VersionControlWidget::onApplyPatch);
}

void VersionControlWidget::setupModels() {
  historyModel = new QStandardItemModel(this);
  branchModel = new QStandardItemModel(this);
  tagModel = new QStandardItemModel(this);

  historyModel->setHorizontalHeaderLabels({tr("提交"), tr("作者"), tr("日期")});
  historyTree->setModel(historyModel);
  branchList->setModel(branchModel);
  tagList->setModel(tagModel);
}

void VersionControlWidget::createActions() {
  actions.splitHorizontal = new QAction(tr("水平分割"), this);
  actions.splitVertical = new QAction(tr("垂直分割"), this);
  actions.toggleHistoryPanel = new QAction(tr("显示历史面板"), this);
  actions.toggleInfoPanel = new QAction(tr("显示信息面板"), this);
  actions.refresh = new QAction(tr("刷新"), this);
  actions.settings = new QAction(tr("设置"), this);

  // 新增的动作
  actions.resolveConflicts = new QAction(tr("解决冲突"), this);
  actions.viewDiff = new QAction(tr("查看差异"), this);
  actions.addMetadata = new QAction(tr("添加元数据"), this);
  actions.editMetadata = new QAction(tr("编辑元数据"), this);
  actions.deleteTag = new QAction(tr("删除标签"), this);
  actions.exportVersion = new QAction(tr("导出版本"), this);
  actions.importVersion = new QAction(tr("导入版本"), this);
  actions.searchHistory = new QAction(tr("搜索历史"), this);
  actions.revertCommit = new QAction(tr("还原提交"), this);
  actions.createPatch = new QAction(tr("创建补丁"), this);
  actions.applyPatch = new QAction(tr("应用补丁"), this);
}

void VersionControlWidget::updateBranchList() {
  branchModel->clear();
  for (const auto &branch : versionControl->list_branches()) {
    auto item = new QStandardItem(QString::fromStdString(branch));
    branchModel->appendRow(item);
  }
}

void VersionControlWidget::updateTagList() {
  tagModel->clear();
  // 实现标签列表更新
}

void VersionControlWidget::showDiffDialog(const QString &hash1,
                                          const QString &hash2) {
  // checkout now returns a cv::Mat; convert to QImage if needed later.
  auto img1 = versionControl->checkout(hash1.toStdString());
  auto img2 = versionControl->checkout(hash2.toStdString());

  auto result = versionControl->compare_images(img1, img2);

  // 显示差异对话框
  // TODO: 实现差异可视化对话框
}

void VersionControlWidget::handleMergeConflicts(const cv::Mat &base,
                                                const cv::Mat &theirs) {
  cv::Mat current = QImageToCvMat(currentImage);
  auto merged = versionControl->merge_images(base, current, theirs);

  if (!merged.empty()) {
    QImage mergedImage = CvMatToQImage(merged);
    imagePreview->setImage(mergedImage);
    // TODO: 显示冲突解决对话框
  }
}

void VersionControlWidget::onResolveConflicts() {
  // TODO: 实现冲突解决流程
}

void VersionControlWidget::onViewDiff() {
  auto selection = historyTree->selectionModel()->selectedIndexes();
  if (selection.size() >= 2) {
    showDiffDialog(selection[0].data().toString(),
                   selection[1].data().toString());
  }
}

void VersionControlWidget::onAddMetadata() {
  auto selection = historyTree->selectionModel()->currentIndex();
  if (selection.isValid()) {
    showMetadataDialog(selection.data().toString());
  }
}

void VersionControlWidget::onEditMetadata() {
  auto selection = historyTree->selectionModel()->currentIndex();
  if (selection.isValid()) {
    showMetadataDialog(selection.data().toString());
  }
}

void VersionControlWidget::onExportVersion() {
  // TODO: 实现版本导出
}

void VersionControlWidget::onImportVersion() {
  // TODO: 实现版本导入
}

void VersionControlWidget::onSearchHistory() {
  // TODO: 实现历史搜索
}

void VersionControlWidget::onRevertCommit() {
  // TODO: 实现提交还原
}

void VersionControlWidget::onCreatePatch() {
  // TODO: 实现补丁创建
}

void VersionControlWidget::onApplyPatch() {
  // TODO: 实现补丁应用
}

void VersionControlWidget::contextMenuEvent(QContextMenuEvent *event) {
  QMenu menu(this);
  menu.addAction(actions.viewDiff); // 使用 actions 而不是 menuActions
  menu.addAction(actions.resolveConflicts);
  menu.addSeparator();
  menu.addAction(actions.addMetadata);
  menu.addAction(actions.editMetadata);
  menu.addSeparator();
  menu.addAction(actions.exportVersion);
  menu.addAction(actions.importVersion);
  menu.addSeparator();
  menu.addAction(actions.createPatch);
  menu.addAction(actions.applyPatch);

  menu.exec(event->pos());
}

// 添加转换辅助函数
namespace {
cv::Mat QImageToCvMat(const QImage &image) {
  switch (image.format()) {
  case QImage::Format_ARGB32:
  case QImage::Format_ARGB32_Premultiplied: {
    cv::Mat mat(image.height(), image.width(), CV_8UC4,
                const_cast<uchar *>(image.bits()),
                static_cast<size_t>(image.bytesPerLine()));
    cv::Mat mat_rgb;
    cv::cvtColor(mat, mat_rgb, cv::COLOR_BGRA2BGR);
    return mat_rgb;
  }
  case QImage::Format_RGB32: {
    cv::Mat mat(image.height(), image.width(), CV_8UC4,
                const_cast<uchar *>(image.bits()),
                static_cast<size_t>(image.bytesPerLine()));
    cv::Mat mat_rgb;
    cv::cvtColor(mat, mat_rgb, cv::COLOR_BGRA2BGR);
    return mat_rgb;
  }
  case QImage::Format_RGB888: {
    cv::Mat mat(image.height(), image.width(), CV_8UC3,
                const_cast<uchar *>(image.bits()),
                static_cast<size_t>(image.bytesPerLine()));
    cv::Mat mat_rgb;
    cv::cvtColor(mat, mat_rgb, cv::COLOR_BGR2RGB);
    return mat_rgb;
  }
  default:
    QImage converted = image.convertToFormat(QImage::Format_RGB888);
    cv::Mat mat(converted.height(), converted.width(), CV_8UC3,
                const_cast<uchar *>(converted.bits()),
                static_cast<size_t>(converted.bytesPerLine()));
    cv::Mat mat_rgb;
    cv::cvtColor(mat, mat_rgb, cv::COLOR_BGR2RGB);
    return mat_rgb;
  }
}

QImage CvMatToQImage(const cv::Mat &mat) {
  switch (mat.type()) {
  case CV_8UC4: {
    return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
                  QImage::Format_ARGB32);
  }
  case CV_8UC3: {
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                  QImage::Format_RGB888)
        .copy();
  }
  case CV_8UC1: {
    return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
                  QImage::Format_Grayscale8)
        .copy();
  }
  default:
    return QImage();
  }
}
} // namespace

void VersionControlWidget::showMetadataDialog(const QString &hash) {
  // 创建对话框
  QDialog dialog(this);
  dialog.setWindowTitle(tr("版本元数据"));

  auto layout = new QVBoxLayout(&dialog);
  auto form = new QFormLayout;

  auto titleEdit = new QLineEdit;
  auto descEdit = new QTextEdit;
  auto tagsEdit = new QLineEdit;

  form->addRow(tr("标题:"), titleEdit);
  form->addRow(tr("描述:"), descEdit);
  form->addRow(tr("标签:"), tagsEdit);

  layout->addLayout(form);

  auto buttonBox =
      new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  connect(buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

  layout->addWidget(buttonBox);

  if (dialog.exec() == QDialog::Accepted) {
    // 保存元数据，使用 update_metadata 替换原 set_metadata
    versionControl->update_metadata(
        hash.toStdString(),
        {{"title", titleEdit->text().toStdString()},
         {"description", descEdit->toPlainText().toStdString()},
         {"tags", tagsEdit->text().toStdString()}});
  }
}

void VersionControlWidget::createMenus() {
  auto menuBar = new QMenuBar(this);
  auto fileMenu = menuBar->addMenu(tr("文件"));
  fileMenu->addAction(actions.exportVersion);
  fileMenu->addAction(actions.importVersion);

  auto editMenu = menuBar->addMenu(tr("编辑"));
  editMenu->addAction(actions.addMetadata);
  editMenu->addAction(actions.editMetadata);

  auto viewMenu = menuBar->addMenu(tr("视图"));
  viewMenu->addAction(actions.toggleHistoryPanel);
  viewMenu->addAction(actions.toggleInfoPanel);
}

void VersionControlWidget::loadSettings() {
  QSettings settings;
  settings.beginGroup("VersionControl");

  splitOrientation = static_cast<Qt::Orientation>(
      settings.value("SplitOrientation", Qt::Horizontal).toInt());
  historyPanelVisible = settings.value("HistoryPanelVisible", true).toBool();
  infoPanelVisible = settings.value("InfoPanelVisible", true).toBool();

  settings.endGroup();

  updatePanelVisibility();
}

void VersionControlWidget::onCommit() {
  QString message = QInputDialog::getMultiLineText(this, tr("提交更改"),
                                                   tr("请输入提交信息:"));

  if (!message.isEmpty()) {
    try {
      // Convert QImage to cv::Mat before commit
      versionControl->commit(QImageToCvMat(currentImage),
                             message.toStdString());
      refreshHistory();
    } catch (const std::exception &e) {
      QMessageBox::critical(this, tr("错误"), tr("提交失败: %1").arg(e.what()));
    }
  }
}

void VersionControlWidget::onCreateBranch() {
  QString name = QInputDialog::getText(this, tr("新建分支"), tr("分支名称:"));

  if (!name.isEmpty()) {
    try {
      versionControl->create_branch(name.toStdString());
      updateBranchList();
    } catch (const std::exception &e) {
      QMessageBox::critical(this, tr("错误"),
                            tr("创建分支失败: %1").arg(e.what()));
    }
  }
}

void VersionControlWidget::onCreateTag() {
  QString name = QInputDialog::getText(this, tr("新建标签"), tr("标签名称:"));

  if (!name.isEmpty()) {
    try {
      // Fix: Pass an extra parameter (assuming "HEAD" as current commit)
      versionControl->create_tag("HEAD", name.toStdString());
      updateTagList();
    } catch (const std::exception &e) {
      QMessageBox::critical(this, tr("错误"),
                            tr("创建标签失败: %1").arg(e.what()));
    }
  }
}

void VersionControlWidget::onMergeBranch() {
  // 获取当前选中的分支
  auto current = branchList->currentIndex();
  if (!current.isValid()) {
    QMessageBox::warning(this, tr("警告"), tr("请先选择要合并的分支"));
    return;
  }

  QString branch = current.data().toString();
  try {
    versionControl->merge_branch(branch.toStdString());
    refreshHistory();
  } catch (const std::exception &e) {
    QMessageBox::critical(this, tr("错误"),
                          tr("合并分支失败: %1").arg(e.what()));
  }
}

void VersionControlWidget::onCheckout() {
  // 获取当前选中的提交
  auto current = historyTree->currentIndex();
  if (!current.isValid()) {
    QMessageBox::warning(this, tr("警告"), tr("请先选择要检出的版本"));
    return;
  }

  QString hash = current.data().toString();
  try {
    auto image = versionControl->checkout(hash.toStdString());
    currentImage = CvMatToQImage(image);
    imagePreview->setImage(currentImage);
  } catch (const std::exception &e) {
    QMessageBox::critical(this, tr("错误"), tr("检出失败: %1").arg(e.what()));
  }
}

void VersionControlWidget::onCompareVersions() {
  auto selection = historyTree->selectionModel()->selectedIndexes();
  if (selection.size() < 2) {
    QMessageBox::warning(this, tr("警告"), tr("请选择两个版本进行比较"));
    return;
  }

  showDiffDialog(selection[0].data().toString(),
                 selection[1].data().toString());
}

void VersionControlWidget::onExportCommit() {
  QString path = QFileDialog::getSaveFileName(
      this, tr("导出版本"), QString(), tr("图像文件 (*.png *.jpg *.bmp)"));

  if (!path.isEmpty()) {
    try {
      currentImage.save(path);
    } catch (const std::exception &e) {
      QMessageBox::critical(this, tr("错误"), tr("导出失败: %1").arg(e.what()));
    }
  }
}

void VersionControlWidget::refreshHistory() {
  historyModel->clear();
  historyModel->setHorizontalHeaderLabels({tr("提交"), tr("作者"), tr("日期")});

  // Fix: Use commits() instead of list_commits()
  for (const auto &commit : versionControl->commits()) {
    QList<QStandardItem *> row;
    row << new QStandardItem(QString::fromStdString(commit.hash))
        << new QStandardItem(QString::fromStdString(commit.author))
        << new QStandardItem(QString::fromStdString(commit.date));
    historyModel->appendRow(row);
  }
}

void VersionControlWidget::updatePanelVisibility() {
  historyDock->setVisible(historyPanelVisible);
  infoDock->setVisible(infoPanelVisible);
}

void VersionControlWidget::onThemeChanged() {
  bool isDark = themeSelector->currentIndex() == 1;
  settings.darkTheme = isDark;
  setupTheme();
}

void VersionControlWidget::setupTheme() {
  QString style = settings.darkTheme
                      ? "QWidget { background-color: #2d2d2d; color: #ffffff; }"
                        "QSplitter::handle { background-color: #404040; }"
                        "QHeaderView::section { background-color: #404040; }"
                      : "";
  setStyleSheet(style);
}

void VersionControlWidget::onAutoRefreshToggled(bool enabled) {
  settings.autoRefresh = enabled;
  if (enabled) {
    autoRefreshTimer->start(5000); // 5秒自动刷新
  } else {
    autoRefreshTimer->stop();
  }
}

void VersionControlWidget::onFilterChanged(const QString &filter) {
  for (int i = 0; i < historyModel->rowCount(); ++i) {
    bool show = true;
    for (int j = 0; i < historyModel->columnCount(); ++j) {
      auto item = historyModel->item(i, j);
      if (!item->text().contains(filter, Qt::CaseInsensitive)) {
        show = false;
        break;
      }
    }
    historyTree->setRowHidden(i, QModelIndex(), !show);
  }
}

void VersionControlWidget::saveSplitterState() {
  QSettings settings;
  settings.beginGroup("VersionControl");
  settings.setValue("MainSplitterState", mainSplitter->saveState());
  settings.setValue("RightSplitterState", rightSplitter->saveState());
  settings.endGroup();
}

void VersionControlWidget::restoreSplitterState() {
  QSettings settings;
  settings.beginGroup("VersionControl");
  mainSplitter->restoreState(settings.value("MainSplitterState").toByteArray());
  rightSplitter->restoreState(
      settings.value("RightSplitterState").toByteArray());
  settings.endGroup();
}

void VersionControlWidget::onShowStatistics() {
  QDialog dialog(this);
  dialog.setWindowTitle(tr("版本统计"));
  auto layout = new QVBoxLayout(&dialog);

  // 添加统计信息
  QStringList stats;
  stats << tr("总提交数: %1").arg(historyModel->rowCount())
        << tr("活跃分支数: %1").arg(branchModel->rowCount())
        << tr("标签数: %1").arg(tagModel->rowCount());

  for (const auto &stat : stats) {
    layout->addWidget(new QLabel(stat));
  }

  dialog.exec();
}

void VersionControlWidget::setupToolbar() {
  auto toolbar = new QWidget(this);
  auto layout = new QHBoxLayout(toolbar);
  layout->setContentsMargins(5, 2, 5, 2);
  layout->setSpacing(2);

  compareButton = new ElaToolButton(this);
  compareButton->setIcon(QIcon::fromTheme("compare"));
  compareButton->setToolTip(tr("比较版本"));

  exportButton = new ElaToolButton(this);
  exportButton->setIcon(QIcon::fromTheme("export"));
  exportButton->setToolTip(tr("导出"));

  refreshButton = new ElaToolButton(this);
  refreshButton->setIcon(QIcon::fromTheme("refresh"));
  refreshButton->setToolTip(tr("刷新"));

  layout->addWidget(compareButton);
  layout->addWidget(exportButton);
  layout->addWidget(refreshButton);
  layout->addStretch();
}

void VersionControlWidget::updatePreviewQuality() {
  if (!imagePreview || currentImage.isNull())
    return;

  int quality = qualitySlider->value();
  QImage scaled = currentImage.scaled(imagePreview->size(), Qt::KeepAspectRatio,
                                      quality < 100 ? Qt::SmoothTransformation
                                                    : Qt::FastTransformation);

  imagePreview->setImage(scaled);
  settings.previewQuality = quality;
}

QWidget *VersionControlWidget::createHistoryPanel() {
  auto panel = new QWidget;
  auto layout = new QVBoxLayout(panel);
  layout->setContentsMargins(0, 0, 0, 0);

  historyTree = new ElaTreeView;
  historyTree->setAlternatingRowColors(true);
  historyTree->setSortingEnabled(true);
  historyTree->setSelectionMode(QAbstractItemView::ExtendedSelection);

  branchList = new ElaListView;
  tagList = new ElaListView;

  layout->addWidget(historyTree);
  layout->addWidget(branchList);
  layout->addWidget(tagList);

  return panel;
}

QWidget *VersionControlWidget::createInfoPanel() {
  auto panel = new QWidget;
  auto layout = new QVBoxLayout(panel);
  layout->setContentsMargins(0, 0, 0, 0);

  commitInfoLabel = new QLabel;
  commitInfoLabel->setWordWrap(true);
  commitInfoLabel->setTextFormat(Qt::RichText);

  layout->addWidget(commitInfoLabel);
  layout->addStretch();

  return panel;
}

QWidget *VersionControlWidget::createPreviewArea() {
  auto area = new QWidget;
  auto layout = new QVBoxLayout(area);
  layout->setContentsMargins(0, 0, 0, 0);

  imagePreview = new CropPreviewWidget;
  imagePreview->setMinimumSize(200, 200);

  layout->addWidget(imagePreview);

  return area;
}

void VersionControlWidget::showCommitInfo(const QString &hash) {
  try {
    // Fix: Use commit_info() instead of get_commit_info()
    auto info = versionControl->commit_info(hash.toStdString());
    QString text =
        tr("<b>提交</b>: %1<br>").arg(QString::fromStdString(info.hash));
    text += tr("<b>作者</b>: %1<br>").arg(QString::fromStdString(info.author));
    text += tr("<b>日期</b>: %1<br>").arg(QString::fromStdString(info.date));
    text += tr("<b>消息</b>: %1").arg(QString::fromStdString(info.message));

    commitInfoLabel->setText(text);
  } catch (const std::exception &e) {
    commitInfoLabel->setText(tr("获取提交信息失败: %1").arg(e.what()));
  }
}

void VersionControlWidget::setImage(const QImage &image) {
  currentImage = image;
  imagePreview->setImage(image);
  updatePreviewQuality();
}

VersionControlWidget::~VersionControlWidget() {
  saveSplitterState();
  saveSettings();
}

void VersionControlWidget::saveSettings() {
  QSettings settings;
  settings.beginGroup("VersionControl");
  settings.setValue("AutoRefresh", this->settings.autoRefresh);
  settings.setValue("PreviewQuality", this->settings.previewQuality);
  settings.setValue("ShowLineNumbers", this->settings.showLineNumbers);
  settings.setValue("CompactMode", this->settings.compactMode);
  settings.setValue("DefaultExportPath", this->settings.defaultExportPath);
  settings.setValue("RecentBranches", this->settings.recentBranches);
  settings.setValue("MaxRecentBranches", this->settings.maxRecentBranches);
  settings.setValue("ShowAuthorAvatar", this->settings.showAuthorAvatar);
  settings.setValue("ThumbnailSize", this->settings.thumbnailSize);
  settings.setValue("DarkTheme", this->settings.darkTheme);
  settings.endGroup();
}

void VersionControlWidget::onShowHistory() {
  auto selection = historyTree->selectionModel()->currentIndex();
  if (!selection.isValid()) {
    return;
  }

  QDialog dialog(this);
  dialog.setWindowTitle(tr("版本历史详情"));
  dialog.resize(600, 400);

  auto layout = new QVBoxLayout(&dialog);
  auto textEdit = new QTextEdit;
  textEdit->setReadOnly(true);

  QString hash = selection.data().toString();
  try {
    auto info = versionControl->commit_info(hash.toStdString());
    QString details;
    details += tr("提交哈希: %1\n").arg(QString::fromStdString(info.hash));
    details += tr("作者: %1\n").arg(QString::fromStdString(info.author));
    details += tr("日期: %1\n").arg(QString::fromStdString(info.date));
    details += tr("\n提交信息:\n%1").arg(QString::fromStdString(info.message));

    textEdit->setText(details);
  } catch (const std::exception &e) {
    textEdit->setText(tr("获取历史信息失败: %1").arg(e.what()));
  }

  layout->addWidget(textEdit);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Close);
  connect(buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
  layout->addWidget(buttonBox);

  dialog.exec();
}

void VersionControlWidget::onFilterBranches() {
  QString filter = searchBox->text().toLower();
  for (int i = 0; i < branchModel->rowCount(); ++i) {
    auto item = branchModel->item(i);
    bool show = item->text().toLower().contains(filter);
    branchList->setRowHidden(i, !show);
  }
}

void VersionControlWidget::onMergeSelected() {
  auto selection = branchList->selectionModel()->selectedIndexes();
  if (selection.isEmpty()) {
    QMessageBox::warning(this, tr("警告"), tr("请选择要合并的分支"));
    return;
  }

  QString branchName = selection.first().data().toString();
  try {
    versionControl->merge_branch(branchName.toStdString());
    refreshHistory();
    QMessageBox::information(this, tr("成功"), tr("分支合并成功"));
  } catch (const std::exception &e) {
    QMessageBox::critical(this, tr("错误"),
                          tr("分支合并失败: %1").arg(e.what()));
  }
}

void VersionControlWidget::onShowDiffVisual() {
  auto selection = historyTree->selectionModel()->selectedIndexes();
  if (selection.size() != 2) {
    QMessageBox::warning(this, tr("警告"), tr("请选择两个版本进行对比"));
    return;
  }

  QString hash1 = selection[0].data().toString();
  QString hash2 = selection[1].data().toString();

  QDialog dialog(this);
  dialog.setWindowTitle(tr("可视化差异"));
  dialog.resize(800, 600);

  auto layout = new QHBoxLayout(&dialog);

  // 左侧图像
  auto leftPreview = new CropPreviewWidget;
  auto rightPreview = new CropPreviewWidget;

  try {
    auto img1 = versionControl->checkout(hash1.toStdString());
    auto img2 = versionControl->checkout(hash2.toStdString());

    leftPreview->setImage(CvMatToQImage(img1));
    rightPreview->setImage(CvMatToQImage(img2));

  } catch (const std::exception &e) {
    QMessageBox::critical(this, tr("错误"),
                          tr("加载图像失败: %1").arg(e.what()));
    return;
  }

  layout->addWidget(leftPreview);
  layout->addWidget(rightPreview);

  dialog.exec();
}

void VersionControlWidget::onSortCommits(int column) {
  historyModel->sort(column);
}

void VersionControlWidget::onBatchExport() {
  QString dir = QFileDialog::getExistingDirectory(
      this, tr("选择导出目录"), settings.defaultExportPath,
      QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

  if (dir.isEmpty()) {
    return;
  }

  auto selection = historyTree->selectionModel()->selectedIndexes();
  QStringList hashes;
  for (const auto &index : selection) {
    if (index.column() == 0) { // 只处理第一列的哈希值
      hashes << index.data().toString();
    }
  }

  if (hashes.isEmpty()) {
    QMessageBox::warning(this, tr("警告"), tr("请选择要导出的版本"));
    return;
  }

  QProgressDialog progress(tr("正在导出版本..."), tr("取消"), 0, hashes.size(),
                           this);
  progress.setWindowModality(Qt::WindowModal);

  int exported = 0;
  for (const QString &hash : hashes) {
    if (progress.wasCanceled())
      break;

    try {
      auto img = versionControl->checkout(hash.toStdString());
      QString filename = QString("%1/%2.png").arg(dir).arg(hash);
      CvMatToQImage(img).save(filename);
      exported++;
    } catch (const std::exception &) {
      continue;
    }

    progress.setValue(exported);
  }

  QMessageBox::information(this, tr("完成"),
                           tr("成功导出 %1 个版本").arg(exported));
}

void VersionControlWidget::updateRecentBranches(const QString &branch) {
  int index = settings.recentBranches.indexOf(branch);
  if (index != -1) {
    settings.recentBranches.removeAt(index);
  }

  settings.recentBranches.prepend(branch);

  while (settings.recentBranches.size() > settings.maxRecentBranches) {
    settings.recentBranches.removeLast();
  }
}

void VersionControlWidget::onDeleteTag() {
  auto selection = tagList->selectionModel()->selectedIndexes();
  if (selection.isEmpty()) {
    QMessageBox::warning(this, tr("警告"), tr("请选择要删除的标签"));
    return;
  }

  QString tagName = selection.first().data().toString();
  if (QMessageBox::question(
          this, tr("确认"), tr("确定要删除标签 '%1' 吗?").arg(tagName),
          QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes) {
    try {
      versionControl->delete_tag(tagName.toStdString());
      updateTagList();
    } catch (const std::exception &e) {
      QMessageBox::critical(this, tr("错误"),
                            tr("删除标签失败: %1").arg(e.what()));
    }
  }
}

void VersionControlWidget::onCustomizeColumns() {
  QDialog dialog(this);
  dialog.setWindowTitle(tr("自定义列"));
  auto layout = new QVBoxLayout(&dialog);

  QStringList columns = {tr("提交"), tr("作者"), tr("日期"), tr("消息")};

  for (const QString &column : columns) {
    auto cb = new ElaCheckBox(column);
    cb->setChecked(columnVisibility.value(column, true));
    layout->addWidget(cb);
    connect(cb, &ElaCheckBox::toggled, this, [this, column](bool checked) {
      columnVisibility[column] = checked;
      updateColumnVisibility();
    });
  }

  auto buttonBox =
      new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  connect(buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
  layout->addWidget(buttonBox);

  dialog.exec();
}

void VersionControlWidget::updateColumnVisibility() {
  for (int i = 0; i < historyModel->columnCount(); ++i) {
    QString columnName = historyModel->headerData(i, Qt::Horizontal).toString();
    historyTree->setColumnHidden(i, !columnVisibility.value(columnName, true));
  }
}

void VersionControlWidget::onSplitHorizontally() {
  updateLayout(Qt::Horizontal);
}

void VersionControlWidget::onSplitVertically() { updateLayout(Qt::Vertical); }

void VersionControlWidget::updateLayout(Qt::Orientation orientation) {
  splitOrientation = orientation;
  mainSplitter->setOrientation(orientation);
  rightSplitter->setOrientation(orientation == Qt::Horizontal ? Qt::Vertical
                                                              : Qt::Horizontal);
}

void VersionControlWidget::onToggleHistoryPanel() {
  historyPanelVisible = !historyPanelVisible;
  historyDock->setVisible(historyPanelVisible);
  actions.toggleHistoryPanel->setChecked(historyPanelVisible);
}

void VersionControlWidget::onToggleInfoPanel() {
  infoPanelVisible = !infoPanelVisible;
  infoDock->setVisible(infoPanelVisible);
  actions.toggleInfoPanel->setChecked(infoPanelVisible);
}

void VersionControlWidget::onCompactModeToggled(bool enabled) {
  settings.compactMode = enabled;
  // 调整UI元素间距和大小
  historyTree->setStyleSheet(enabled ? "QTreeView { margin: 0; padding: 0; }"
                                     : "");
  branchList->setStyleSheet(enabled ? "QListView { margin: 0; padding: 0; }"
                                    : "");
  tagList->setStyleSheet(enabled ? "QListView { margin: 0; padding: 0; }" : "");
}

void VersionControlWidget::updatePreview() {
  if (!imagePreview || currentImage.isNull()) {
    return;
  }

  QSize previewSize = imagePreview->size();
  if (previewSize.isEmpty()) {
    previewSize = QSize(640, 480); // 默认预览大小
  }

  // 根据预览质量设置缩放图像
  Qt::TransformationMode mode = settings.previewQuality < 100
                                    ? Qt::SmoothTransformation
                                    : Qt::FastTransformation;

  QImage previewImage =
      currentImage.scaled(previewSize, Qt::KeepAspectRatio, mode);

  imagePreview->setImage(previewImage);
}

cv::Mat VersionControlWidget::QImageToCvMat(const QImage &image) {
  if (image.isNull()) {
    return cv::Mat();
  }

  // 标准化图像格式为 ARGB32 或 RGB32
  QImage converted;
  switch (image.format()) {
  case QImage::Format_ARGB32:
  case QImage::Format_ARGB32_Premultiplied:
  case QImage::Format_RGB32:
    converted = image;
    break;
  default:
    converted = image.convertToFormat(QImage::Format_ARGB32);
  }

  cv::Mat mat;
  switch (converted.format()) {
  case QImage::Format_ARGB32:
  case QImage::Format_ARGB32_Premultiplied: {
    mat = cv::Mat(converted.height(), converted.width(), CV_8UC4,
                  const_cast<uchar *>(converted.bits()),
                  converted.bytesPerLine());
    cv::Mat channels[4];
    cv::split(mat, channels);
    // 重新排列通道顺序：BGRA -> RGBA
    std::swap(channels[0], channels[2]);
    cv::merge(channels, 4, mat);
    break;
  }
  case QImage::Format_RGB32: {
    mat = cv::Mat(converted.height(), converted.width(), CV_8UC4,
                  const_cast<uchar *>(converted.bits()),
                  converted.bytesPerLine());
    cv::cvtColor(mat, mat, cv::COLOR_BGRA2BGR);
    break;
  }
  default:
    return cv::Mat();
  }

  return mat.clone(); // 返回深拷贝以确保数据所有权
}

QImage VersionControlWidget::CvMatToQImage(const cv::Mat &mat) {
  if (mat.empty()) {
    return QImage();
  }

  // 根据Mat的类型选择适当的转换方式
  switch (mat.type()) {
  case CV_8UC4: // 8位无符号，4通道
  {
    cv::Mat rgba;
    // 确保通道顺序正确（BGRA -> RGBA）
    cv::Mat channels[4];
    cv::split(mat, channels);
    std::swap(channels[0], channels[2]);
    cv::merge(channels, 4, rgba);

    return QImage(rgba.data, rgba.cols, rgba.rows, rgba.step,
                  QImage::Format_RGBA8888)
        .copy();
  }

  case CV_8UC3: // 8位无符号，3通道
  {
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888)
        .copy();
  }

  case CV_8UC1: // 8位无符号，单通道
  {
    return QImage(mat.data, mat.cols, mat.rows, mat.step,
                  QImage::Format_Grayscale8)
        .copy();
  }

  default:
    // 对于其他格式，尝试转换为BGR后再处理
    cv::Mat temp;
    mat.convertTo(temp, CV_8UC3);
    cv::cvtColor(temp, temp, cv::COLOR_BGR2RGB);
    return QImage(temp.data, temp.cols, temp.rows, temp.step,
                  QImage::Format_RGB888)
        .copy();
  }
}
