#include "JsonEditor.hpp"
#include "JsonModel.hpp"

#include <QApplication>
#include <QFileDialog>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QPushButton>
#include <QStatusBar>
#include <QToolBar>
#include <QToolButton>
#include <QTreeView>
#include <QVBoxLayout>
#include <QCompleter>
#include <QStringListModel>
#include <QProgressBar>
#include <QDialog>

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
  layout->setContentsMargins(2, 2, 2, 2);
  layout->setSpacing(2);

  treeView = new QTreeView(this);
  searchBar = new QLineEdit(this);
  statusBar = new QStatusBar(this);
  toolbar = new QToolBar(this);

  layout->addWidget(toolbar);
  layout->addWidget(searchBar);
  layout->addWidget(treeView);
  layout->addWidget(statusBar);

  treeView->setModel(&model);
  treeView->setEditTriggers(QAbstractItemView::DoubleClicked);
  treeView->header()->setSectionResizeMode(QHeaderView::ResizeToContents);

  searchBar->setPlaceholderText(tr("搜索 (Ctrl+F)"));

  // 添加进度条
  progressBar = new QProgressBar(this);
  progressBar->setVisible(false);
  statusBar->addWidget(progressBar);

  // 设置拖放支持
  setAcceptDrops(true);

  // 连接异步加载信号
  connect(&model, &JsonModel::loadProgress, this, &JsonEditor::showLoadingProgress);
  connect(&model, &JsonModel::loadCompleted, this, [this]() {
    progressBar->setVisible(false);
    statusBar->showMessage(tr("加载完成"), 2000);
  });
  connect(&model, &JsonModel::loadError, this, [this](const QString& error) {
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

  themeBtn = new QPushButton(isDarkTheme ? "🌞" : "🌛", this);
  themeBtn->setFixedSize(24, 24);
  toolbar->addWidget(themeBtn);

  auto *exportMenu = new QMenu(this);
  exportMenu->addAction("导出为 CSV", this, [this] { exportTo("csv"); });
  exportMenu->addAction("导出为 HTML", this, [this] { exportTo("html"); });

  auto *exportBtn = new QToolButton(this);
  exportBtn->setIcon(QIcon::fromTheme("document-export"));
  exportBtn->setMenu(exportMenu);
  exportBtn->setPopupMode(QToolButton::InstantPopup);
  toolbar->addWidget(exportBtn);

  connect(openAct, &QAction::triggered, this, &JsonEditor::openFile);
  connect(saveAct, &QAction::triggered, this, &JsonEditor::saveFile);
  connect(themeBtn, &QPushButton::clicked, this, &JsonEditor::toggleTheme);
}

void JsonEditor::setupStatusBar() {
  statsLabel = new QLabel(this);
  statusBar->addPermanentWidget(statsLabel);
  updateStats();
}

void JsonEditor::setupConnections() {
  connect(searchBar, &QLineEdit::textChanged, this, &JsonEditor::filterContent);
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
      }
      QTreeView {
        border: 1px solid #ccc;
        border-radius: 4px;
      }
      QLineEdit {
        padding: 4px;
        border: 1px solid #ccc;
        border-radius: 4px;
      }
      QToolBar {
        border: none;
        spacing: 4px;
      }
      QPushButton {
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 4px 8px;
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

void JsonEditor::loadJson(const nlohmann::json& json) {
    model.load(json);
    updateStats();
    updateCompleterWordList(); // 更新自动补全词列表
}

nlohmann::json JsonEditor::getJson() const {
    return model.getJson();
}

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
    QLineEdit *editor = qobject_cast<QLineEdit*>(treeView->itemDelegate()->createEditor(
        treeView, QStyleOptionViewItem(), QModelIndex()));
    if (editor) {
        editor->setCompleter(completer);
    }
}

void JsonEditor::updateCompleterWordList() {
    // 从当前JSON数据中提取关键词
    std::function<void(const Json&)> extractWords = [&](const Json& j) {
        if (j.is_object()) {
            for (auto& [key, value] : j.items()) {
                wordList << QString::fromStdString(key);
                extractWords(value);
            }
        } else if (j.is_array()) {
            for (auto& element : j) {
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
    auto* layout = new QVBoxLayout(findReplaceDialog);
    
    findEdit = new QLineEdit(findReplaceDialog);
    replaceEdit = new QLineEdit(findReplaceDialog);
    replaceBtn = new QPushButton(tr("替换"), findReplaceDialog);
    replaceAllBtn = new QPushButton(tr("全部替换"), findReplaceDialog);
    
    // ...设置查找替换对话框UI...
}

void JsonEditor::handleDroppedFile(const QString& path) {
    try {
        std::ifstream file(path.toStdString());
        Json data = Json::parse(file);
        model.loadAsync(data);
        progressBar->setVisible(true);
        addToRecentFiles(path);
    } catch (const std::exception& e) {
        QMessageBox::critical(this, tr("错误"), e.what());
    }
}