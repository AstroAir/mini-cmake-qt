#include "JsonEditor.hpp"
#include "JsonModel.hpp"

#include <QApplication>
#include <QCompleter>
#include <QDialog>
#include <QFileDialog>
#include <QHeaderView>
#include <QLabel>
#include <QMenu>
#include <QMessageBox>
#include <QStringListModel>
#include <QVBoxLayout>

#include "ElaLineEdit.h"
#include "ElaMenu.h"
#include "ElaProgressBar.h"
#include "ElaPushButton.h"
#include "ElaStatusBar.h"
#include "ElaToolBar.h"
#include "ElaToolButton.h"
#include "ElaTreeView.h"

#include <fstream>

#include <spdlog/spdlog.h>

using Json = nlohmann::json;
using namespace std::string_view_literals;

JsonEditor::JsonEditor(QWidget *parent) : QWidget(parent) {
  setupUI();
  setupToolbar();
  setupStatusBar();
  setupConnections();
  setupCompleter();
  applyStyle();

  // 创建并设置语法高亮器
  highlighter = new JsonSyntaxHighlighter(treeView->viewport());
}

void JsonEditor::setupUI() {
  auto *layout = new QVBoxLayout(this);
  layout->setContentsMargins(12, 12, 12, 12);
  layout->setSpacing(8);

  // 创建顶部容器
  auto *topContainer = new QWidget(this);
  topContainer->setObjectName("topContainer");
  auto *topLayout = new QHBoxLayout(topContainer);
  topLayout->setContentsMargins(0, 0, 0, 0);
  topLayout->setSpacing(8);

  toolbar = new ElaToolBar(this);
  toolbar->setObjectName("jsonToolbar");
  searchBar = new ElaLineEdit(this);
  searchBar->setObjectName("jsonSearchBar");
  searchBar->setPlaceholderText(tr("搜索 (Ctrl+F)"));

  topLayout->addWidget(toolbar);
  topLayout->addWidget(searchBar, 1);

  // 创建树视图容器
  auto *treeContainer = new QWidget(this);
  treeContainer->setObjectName("treeContainer");
  auto *treeLayout = new QVBoxLayout(treeContainer);
  treeLayout->setContentsMargins(0, 0, 0, 0);
  treeLayout->setSpacing(0);

  treeView = new ElaTreeView(this);
  treeView->setObjectName("jsonTreeView");
  treeView->setModel(&model);
  treeView->setEditTriggers(QAbstractItemView::DoubleClicked);
  treeView->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
  treeView->setFrameStyle(QFrame::NoFrame);
  treeView->setAlternatingRowColors(true);
  treeView->setAnimated(true);
  treeView->setUniformRowHeights(true);
  treeLayout->addWidget(treeView);

  // 创建底部状态栏容器
  auto *bottomContainer = new QWidget(this);
  bottomContainer->setObjectName("bottomContainer");
  auto *bottomLayout = new QHBoxLayout(bottomContainer);
  bottomLayout->setContentsMargins(0, 0, 0, 0);
  bottomLayout->setSpacing(8);

  statusBar = new ElaStatusBar(this);
  statusBar->setObjectName("jsonStatusBar");
  progressBar = new ElaProgressBar(this);
  progressBar->setObjectName("jsonProgressBar");
  progressBar->setVisible(false);

  bottomLayout->addWidget(statusBar);
  bottomLayout->addWidget(progressBar);

  // 添加所有容器到主布局
  layout->addWidget(topContainer);
  layout->addWidget(treeContainer, 1);
  layout->addWidget(bottomContainer);

  // 设置拖放支持
  setAcceptDrops(true);

  // 连接异步加载信号
  connect(&model, &JsonModel::loadProgress, this,
          &JsonEditor::showLoadingProgress);
  connect(&model, &JsonModel::loadCompleted, this, [this]() {
    progressBar->setVisible(false);
    statusBar->showMessage(tr("加载完成"), 2000);
  });
  connect(&model, &JsonModel::loadError, this, [this](const QString &error) {
    progressBar->setVisible(false);
    QMessageBox::critical(this, tr("错误"), error);
  });
}

void JsonEditor::setupToolbar() {
  toolbar->setIconSize(QSize(16, 16));

  auto *openAct =
      toolbar->addAction(QIcon::fromTheme("document-open"), tr("打开"));
  auto *saveAct =
      toolbar->addAction(QIcon::fromTheme("document-save"), tr("保存"));
  toolbar->addSeparator();

  themeBtn = new ElaPushButton(isDarkTheme ? "🌞" : "🌛", this);
  themeBtn->setFixedSize(24, 24);
  toolbar->addWidget(themeBtn);

  auto *exportMenu = new ElaMenu(this);
  exportMenu->addAction("导出为 CSV", this, [this] { exportTo("csv"); });
  exportMenu->addAction("导出为 HTML", this, [this] { exportTo("html"); });

  auto *exportBtn = new ElaToolButton(this);
  exportBtn->setIcon(QIcon::fromTheme("document-export"));
  exportBtn->setMenu(exportMenu);
  exportBtn->setPopupMode(ElaToolButton::InstantPopup);
  toolbar->addWidget(exportBtn);

  connect(openAct, &QAction::triggered, this, &JsonEditor::openFile);
  connect(saveAct, &QAction::triggered, this, &JsonEditor::saveFile);
  connect(themeBtn, &ElaPushButton::clicked, this, &JsonEditor::toggleTheme);
}

void JsonEditor::setupStatusBar() {
  statsLabel = new QLabel(this);
  statusBar->addPermanentWidget(statsLabel);
  updateStats();
}

void JsonEditor::setupConnections() {
  connect(searchBar, &ElaLineEdit::textChanged, this,
          &JsonEditor::filterContent);
  connect(&model, &JsonModel::dataChanged, this, &JsonEditor::updateStats);

  // 添加快捷键
  auto *undoAction = new QAction(tr("撤销"), this);
  undoAction->setShortcut(QKeySequence::Undo);
  connect(undoAction, &QAction::triggered, this, [this] {
    if (model.undo()) {
      statusBar->showMessage(tr("撤销成功"), 2000);
    } else {
      statusBar->showMessage(tr("没有可撤销的操作"), 2000);
    }
  });

  auto *redoAction = new QAction(tr("重做"), this);
  redoAction->setShortcut(QKeySequence::Redo);
  connect(redoAction, &QAction::triggered, this, [this] {
    if (model.redo()) {
      statusBar->showMessage(tr("重做成功"), 2000);
    } else {
      statusBar->showMessage(tr("没有可重做的操作"), 2000);
    }
  });

  auto *searchAction = new QAction(tr("搜索"), this);
  searchAction->setShortcut(QKeySequence::Find);
  connect(searchAction, &QAction::triggered, this, [this] {
    searchBar->setFocus();
    searchBar->selectAll();
  });

  auto *saveAction = new QAction(tr("保存"), this);
  saveAction->setShortcut(QKeySequence::Save);
  connect(saveAction, &QAction::triggered, this, &JsonEditor::saveFile);

  addAction(undoAction);
  addAction(redoAction);
  addAction(searchAction);
  addAction(saveAction);
}

void JsonEditor::applyStyle() {
  setStyleSheet(R"(
      QWidget {
        font-family: "Segoe UI", "Microsoft YaHei";
        font-size: 14px;
      }
      
      #topContainer {
        background-color: rgba(30, 30, 30, 0.95);
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 8px;
      }
      
      #jsonToolbar {
        background: transparent;
        border: none;
        spacing: 8px;
        padding: 4px;
      }
      
      #jsonSearchBar {
        background-color: rgba(45, 45, 45, 0.95);
        color: #ffffff;
        padding: 8px 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        min-height: 20px;
        transition: all 0.2s ease;
      }
      
      #jsonSearchBar:focus {
        border-color: #0078d4;
        background-color: rgba(50, 50, 50, 0.95);
        box-shadow: 0 0 0 2px rgba(0, 120, 212, 0.25);
      }
      
      #treeContainer {
        background-color: rgba(30, 30, 30, 0.95);
        border-radius: 8px;
        padding: 12px;
      }
      
      #jsonTreeView {
        background-color: rgba(35, 35, 35, 0.95);
        alternate-background-color: rgba(40, 40, 40, 0.95);
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 8px;
        animation: fadeIn 0.3s ease-in-out;
      }
      
      #jsonTreeView::item {
        padding: 6px;
        margin: 2px;
        border-radius: 4px;
        transition: all 0.2s ease;
      }
      
      #jsonTreeView::item:hover {
        background-color: rgba(60, 60, 60, 0.95);
      }
      
      #jsonTreeView::item:selected {
        background-color: #0078d4;
      }
      
      #jsonTreeView::branch:has-children:!has-siblings:closed,
      #jsonTreeView::branch:closed:has-children:has-siblings {
        image: url(:/images/chevron-right.png);
      }
      
      #jsonTreeView::branch:open:has-children:!has-siblings,
      #jsonTreeView::branch:open:has-children:has-siblings {
        image: url(:/images/chevron-down.png);
      }
      
      #bottomContainer {
        background-color: rgba(30, 30, 30, 0.95);
        border-radius: 8px;
        padding: 8px;
        margin-top: 8px;
      }
      
      #jsonStatusBar {
        color: #ffffff;
        background: transparent;
        padding: 4px 8px;
        font-size: 12px;
      }
      
      #jsonProgressBar {
        background-color: rgba(45, 45, 45, 0.95);
        border-radius: 4px;
        text-align: center;
        min-height: 6px;
      }
      
      #jsonProgressBar::chunk {
        background-color: #0078d4;
        border-radius: 4px;
        width: 20px;
      }
      
      ElaPushButton {
        background-color: rgba(45, 45, 45, 0.95);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        padding: 8px 16px;
        min-height: 20px;
        font-weight: 500;
        transition: all 0.2s ease;
      }
      
      ElaPushButton:hover {
        background-color: rgba(60, 60, 60, 0.95);
        border-color: rgba(255, 255, 255, 0.2);
      }
      
      ElaPushButton:pressed {
        background-color: rgba(70, 70, 70, 0.95);
        transform: scale(0.98);
      }
      
      QScrollBar:vertical {
        background-color: rgba(45, 45, 45, 0.95);
        width: 12px;
        margin: 0px;
      }
      
      QScrollBar::handle:vertical {
        background-color: rgba(80, 80, 80, 0.95);
        border-radius: 6px;
        min-height: 20px;
        margin: 2px;
      }
      
      QScrollBar::handle:vertical:hover {
        background-color: rgba(100, 100, 100, 0.95);
      }
      
      QScrollBar::add-line:vertical,
      QScrollBar::sub-line:vertical {
        height: 0px;
      }
      
      QScrollBar:horizontal {
        background-color: rgba(45, 45, 45, 0.95);
        height: 12px;
        margin: 0px;
      }
      
      QScrollBar::handle:horizontal {
        background-color: rgba(80, 80, 80, 0.95);
        border-radius: 6px;
        min-width: 20px;
        margin: 2px;
      }
      
      QScrollBar::handle:horizontal:hover {
        background-color: rgba(100, 100, 100, 0.95);
      }
      
      QScrollBar::add-line:horizontal,
      QScrollBar::sub-line:horizontal {
        width: 0px;
      }
      
      @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
      }
      
      @media (max-width: 768px) {
        QWidget {
          font-size: 12px;
        }
        
        #jsonSearchBar,
        ElaPushButton {
          padding: 6px 12px;
        }
        
        #jsonTreeView::item {
          padding: 4px;
        }
      }
    )");
}

void JsonEditor::toggleTheme() {
  isDarkTheme = !isDarkTheme;
  QPalette darkPalette;
  if (isDarkTheme) {
    darkPalette.setColor(QPalette::Window, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::WindowText, Qt::white);
    darkPalette.setColor(QPalette::Base, QColor(25, 25, 25));
    darkPalette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::Text, Qt::white);
    darkPalette.setColor(QPalette::Button, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::ButtonText, Qt::white);
    QApplication::setPalette(darkPalette);
  } else {
    QApplication::setPalette(QApplication::style()->standardPalette());
  }
  themeBtn->setText(isDarkTheme ? "🌞" : "🌛");
}

void JsonEditor::filterContent(const QString &text) {
  QRegularExpression re(text, QRegularExpression::CaseInsensitiveOption);
  for (int i = 0; i < model.rowCount(QModelIndex()); ++i) {
    auto idx = model.index(i, 0, QModelIndex());
    treeView->setRowHidden(
        i, QModelIndex(),
        !model.data(idx, Qt::DisplayRole).toString().contains(re));
  }
}

void JsonEditor::updateStats() {
  int total = model.rowCount(QModelIndex());
  statsLabel->setText(tr("总节点数: %1").arg(total));
}

void JsonEditor::exportTo(const QString &format) {
  QString path = QFileDialog::getSaveFileName(
      this, tr("导出"), QString(),
      format.toUpper() + tr(" 文件 (*.") + format + ")");
  if (path.isEmpty())
    return;

  try {
    std::ofstream file(path.toStdString());
    if (!file) {
      throw std::runtime_error("无法打开文件进行写入");
    }

    if (format == "csv") {
      // CSV 格式导出
      file << "Key,Value,Type\n";
      for (int i = 0; i < model.rowCount(QModelIndex()); ++i) {
        auto idx = model.index(i, 0, QModelIndex());
        file << model.data(idx, Qt::DisplayRole).toString().toStdString() << ","
             << model.data(idx, Qt::DisplayRole).toString().toStdString() << ","
             << model.data(model.index(i, 1, QModelIndex()), Qt::DisplayRole)
                    .toString()
                    .toStdString()
             << "\n";
      }
    } else if (format == "html") {
      // HTML 格式导出
      file << R"(<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        table { border-collapse: collapse; width: 100%; }
        th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
        tr:nth-child(even) { background-color: #f2f2f2; }
    </style>
</head>
<body>
<table>
<tr><th>Key</th><th>Value</th><th>Type</th></tr>)";

      for (int i = 0; i < model.rowCount(QModelIndex()); ++i) {
        auto idx = model.index(i, 0, QModelIndex());
        file << "<tr><td>"
             << model.data(idx, Qt::DisplayRole).toString().toStdString()
             << "</td><td>"
             << model.data(idx, Qt::DisplayRole).toString().toStdString()
             << "</td><td>"
             << model.data(model.index(i, 1, QModelIndex()), Qt::DisplayRole)
                    .toString()
                    .toStdString()
             << "</td></tr>\n";
      }

      file << "</table></body></html>";
    }
    statusBar->showMessage(tr("导出成功"), 3000);
  } catch (const std::exception &e) {
    QMessageBox::critical(this, tr("错误"), e.what());
  }
}

// 打开 JSON 文件
void JsonEditor::openFile() {
  try {
    QString path = QFileDialog::getOpenFileName(this, "Open JSON");
    if (path.isEmpty())
      return;

    std::ifstream file(path.toStdString());
    Json data = Json::parse(file);
    model.load(data);
    updateCompleterWordList(); // 更新自动补全词列表
  } catch (const std::exception &e) {
    spdlog::error("Open file failed: {}", e.what());
    QMessageBox::critical(this, "Error", e.what());
  }
}

// 保存 JSON 文件
void JsonEditor::saveFile() {
  try {
    QString path = QFileDialog::getSaveFileName(this, "Save JSON");
    if (path.isEmpty())
      return;

    std::ofstream file(path.toStdString());
    file << model.getJson().dump(4);
  } catch (const std::exception &e) {
    spdlog::error("Save file failed: {}", e.what());
    QMessageBox::critical(this, "Error", e.what());
  }
}

void JsonEditor::loadJson(const nlohmann::json &json) {
  model.load(json);
  updateStats();
  updateCompleterWordList(); // 更新自动补全词列表
}

nlohmann::json JsonEditor::getJson() const { return model.getJson(); }

void JsonEditor::setupCompleter() {
  completer = new QCompleter(this);
  completer->setModelSorting(QCompleter::CaseInsensitivelySortedModel);
  completer->setCaseSensitivity(Qt::CaseInsensitive);
  completer->setWrapAround(false);

  // 初始化基础关键字
  wordList << "true" << "false" << "null" << "{" << "}" << "[" << "]" << ":"
           << "," << "\"\"";

  completer->setModel(new QStringListModel(wordList, completer));

  // 将自动补全器设置给编辑器
  ElaLineEdit *editor =
      qobject_cast<ElaLineEdit *>(treeView->itemDelegate()->createEditor(
          treeView, QStyleOptionViewItem(), QModelIndex()));
  if (editor) {
    editor->setCompleter(completer);
  }
}

void JsonEditor::updateCompleterWordList() {
  // 从当前JSON数据中提取关键词
  std::function<void(const Json &)> extractWords = [&](const Json &j) {
    if (j.is_object()) {
      for (auto &[key, value] : j.items()) {
        wordList << QString::fromStdString(key);
        extractWords(value);
      }
    } else if (j.is_array()) {
      for (auto &element : j) {
        extractWords(element);
      }
    } else if (j.is_string()) {
      wordList << QString::fromStdString(j.get<std::string>());
    }
  };

  extractWords(model.getJson());
  wordList.removeDuplicates();
  completer->setModel(new QStringListModel(wordList, completer));
}

void JsonEditor::setupFindReplace() {
  findReplaceDialog = new QDialog(this);
  auto *layout = new QVBoxLayout(findReplaceDialog);

  findEdit = new ElaLineEdit(findReplaceDialog);
  replaceEdit = new ElaLineEdit(findReplaceDialog);
  replaceBtn = new ElaPushButton(tr("替换"), findReplaceDialog);
  replaceAllBtn = new ElaPushButton(tr("全部替换"), findReplaceDialog);

  // ...设置查找替换对话框UI...
}

void JsonEditor::handleDroppedFile(const QString &path) {
  try {
    std::ifstream file(path.toStdString());
    Json data = Json::parse(file);
    model.loadAsync(data);
    progressBar->setVisible(true);
    addToRecentFiles(path);
  } catch (const std::exception &e) {
    QMessageBox::critical(this, tr("错误"), e.what());
  }
}

void JsonEditor::showLoadingProgress(int progress) {
  progressBar->setVisible(true);
  progressBar->setValue(progress);
  statusBar->showMessage(tr("正在加载... %1%").arg(progress));
}

void JsonEditor::addToRecentFiles(const QString &filePath) {
  const int maxRecentFiles = 5; // 最多保存5个最近文件

  QSettings settings;
  QStringList recentFiles = settings.value("recentFiles").toStringList();

  // 移除已存在的相同路径（如果有）
  recentFiles.removeAll(filePath);

  // 在开头插入新路径
  recentFiles.prepend(filePath);

  // 如果超过最大数量，移除多余的
  while (recentFiles.size() > maxRecentFiles) {
    recentFiles.removeLast();
  }

  // 保存更新后的列表
  settings.setValue("recentFiles", recentFiles);

  // 更新最近文件菜单（如果需要）
  emit recentFilesChanged(recentFiles);
}

void JsonEditor::dragEnterEvent(QDragEnterEvent *event) {
  if (event->mimeData()->hasUrls()) {
    const QList<QUrl> urls = event->mimeData()->urls();
    if (urls.size() == 1) { // 只接受单个文件
      QString filePath = urls.first().toLocalFile();
      if (filePath.endsWith(".json", Qt::CaseInsensitive)) {
        event->acceptProposedAction();
        return;
      }
    }
  }
  event->ignore();
}

void JsonEditor::dropEvent(QDropEvent *event) {
  const QList<QUrl> urls = event->mimeData()->urls();
  if (urls.isEmpty())
    return;

  QString filePath = urls.first().toLocalFile();
  handleDroppedFile(filePath);
  event->acceptProposedAction();
}