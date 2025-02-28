#include "JsonEditor.hpp"
#include "JsonModel.hpp"
#include "JsonSyntaxHighlighter.hpp"

#include <QApplication>
#include <QCompleter>
#include <QDateTime>
#include <QDialog>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFormLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMenu>
#include <QMessageBox>
#include <QMimeData>
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
#include <unordered_set>

// For brevity in modern C++20
using namespace std::string_view_literals;

JsonEditor::JsonEditor(QWidget *parent) : QWidget(parent) {
  setupUI();
  setupToolbar();
  setupStatusBar();
  setupConnections();
  setupCompleter();
  applyStyle();

  // Create and setup syntax highlighter
  highlighter = new JsonSyntaxHighlighter(treeView->viewport());
}

void JsonEditor::setupUI() {
  auto *layout = new QVBoxLayout(this);
  layout->setContentsMargins(12, 12, 12, 12);
  layout->setSpacing(8);

  // Create top container
  auto *topContainer = new QWidget(this);
  topContainer->setObjectName("topContainer");
  auto *topLayout = new QHBoxLayout(topContainer);
  topLayout->setContentsMargins(0, 0, 0, 0);
  topLayout->setSpacing(8);

  toolbar = new ElaToolBar(this);
  toolbar->setObjectName("jsonToolbar");
  searchBar = new ElaLineEdit(this);
  searchBar->setObjectName("jsonSearchBar");
  searchBar->setPlaceholderText(tr("Search (Ctrl+F)"));

  topLayout->addWidget(toolbar);
  topLayout->addWidget(searchBar, 1);

  // Create tree view container
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

  // Create bottom status bar container
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

  // Add all containers to main layout
  layout->addWidget(topContainer);
  layout->addWidget(treeContainer, 1);
  layout->addWidget(bottomContainer);

  // Setup drag and drop support
  setAcceptDrops(true);

  // Connect async loading signals
  connect(&model, &JsonModel::loadProgress, this,
          &JsonEditor::showLoadingProgress);
  connect(&model, &JsonModel::loadCompleted, this, [this]() {
    progressBar->setVisible(false);
    statusBar->showMessage(tr("Load complete"), 2000);
    emit jsonLoaded(true);
  });
  connect(&model, &JsonModel::loadError, this, [this](const QString &error) {
    progressBar->setVisible(false);
    QMessageBox::critical(this, tr("Error"), error);
    emit jsonLoaded(false);
  });
}

void JsonEditor::setupToolbar() {
  toolbar->setIconSize(QSize(16, 16));

  auto *openAct =
      toolbar->addAction(QIcon::fromTheme("document-open"), tr("Open"));
  auto *saveAct =
      toolbar->addAction(QIcon::fromTheme("document-save"), tr("Save"));
  toolbar->addSeparator();

  themeBtn = new ElaPushButton(isDarkTheme ? "🌞" : "🌛", this);
  themeBtn->setFixedSize(24, 24);
  toolbar->addWidget(themeBtn);

  auto *exportMenu = new ElaMenu(this);
  exportMenu->addAction("Export to CSV", this, [this] { exportTo("csv"); });
  exportMenu->addAction("Export to HTML", this, [this] { exportTo("html"); });

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
  connect(&model, &JsonModel::dataChanged, this, [this]() {
    updateStats();
    emit jsonChanged();
  });

  // Add keyboard shortcuts
  auto *undoAction = new QAction(tr("Undo"), this);
  undoAction->setShortcut(QKeySequence::Undo);
  connect(undoAction, &QAction::triggered, this, [this] {
    if (model.undo()) {
      statusBar->showMessage(tr("Undo successful"), 2000);
    } else {
      statusBar->showMessage(tr("Nothing to undo"), 2000);
    }
  });

  auto *redoAction = new QAction(tr("Redo"), this);
  redoAction->setShortcut(QKeySequence::Redo);
  connect(redoAction, &QAction::triggered, this, [this] {
    if (model.redo()) {
      statusBar->showMessage(tr("Redo successful"), 2000);
    } else {
      statusBar->showMessage(tr("Nothing to redo"), 2000);
    }
  });

  auto *searchAction = new QAction(tr("Search"), this);
  searchAction->setShortcut(QKeySequence::Find);
  connect(searchAction, &QAction::triggered, this, [this] {
    searchBar->setFocus();
    searchBar->selectAll();
  });

  auto *saveAction = new QAction(tr("Save"), this);
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
  if (text.isEmpty()) {
    // Show all items when filter is cleared
    treeView->expandAll();
    for (int i = 0; i < model.rowCount(QModelIndex()); ++i) {
      treeView->setRowHidden(i, QModelIndex(), false);
    }
    return;
  }

  try {
    // Use efficient regex search with case-insensitive option
    QRegularExpression re(QRegularExpression::escape(text),
                          QRegularExpression::CaseInsensitiveOption);

    // Hide/show rows based on search results
    std::function<bool(const QModelIndex &)> filterTree =
        [&](const QModelIndex &index) -> bool {
      bool matched = false;

      // Check if this node matches
      QString nodeText = model.data(index, Qt::DisplayRole).toString();
      if (re.match(nodeText).hasMatch()) {
        matched = true;
      }

      // Check child nodes
      for (int i = 0; i < model.rowCount(index); ++i) {
        QModelIndex childIndex = model.index(i, 0, index);
        if (filterTree(childIndex)) {
          // Expand parent if child matches
          treeView->expand(index);
          matched = true;
        }
      }

      // Hide/show based on match result
      if (index.parent().isValid()) {
        treeView->setRowHidden(index.row(), index.parent(), !matched);
      }

      return matched;
    };

    // Start filtering from root
    for (int i = 0; i < model.rowCount(QModelIndex()); ++i) {
      QModelIndex rootIndex = model.index(i, 0, QModelIndex());
      if (!filterTree(rootIndex)) {
        treeView->setRowHidden(i, QModelIndex(), true);
      } else {
        treeView->setRowHidden(i, QModelIndex(), false);
        treeView->expand(rootIndex);
      }
    }
  } catch (const std::exception &e) {
    qWarning("Error in filter: %s", e.what());
  }
}

void JsonEditor::updateStats() {
  try {
    // Get stats about current JSON
    const Json &json = model.getJson();
    size_t nodeCount = model.getJsonNodeCount(json);
    size_t depth = model.getJsonDepth(json);

    // Create stats string
    QString stats;
    if (json.is_object()) {
      stats = tr("Object: %1 properties, %2 nodes, depth %3")
                  .arg(json.size())
                  .arg(nodeCount)
                  .arg(depth);
    } else if (json.is_array()) {
      stats = tr("Array: %1 items, %2 nodes, depth %3")
                  .arg(json.size())
                  .arg(nodeCount)
                  .arg(depth);
    } else {
      stats = tr("Value: %1 nodes").arg(nodeCount);
    }

    statsLabel->setText(stats);
  } catch (...) {
    statsLabel->setText(tr("No valid JSON"));
  }
}

void JsonEditor::exportTo(const QString &format) {
  QString path = QFileDialog::getSaveFileName(
      this, tr("Export"), QString(),
      format.toUpper() + tr(" Files (*.") + format + ")");
  if (path.isEmpty())
    return;

  try {
    std::ofstream file(path.toStdString());
    if (!file) {
      throw std::runtime_error("Cannot open file for writing");
    }

    if (format == "csv") {
      // CSV export
      file << "Key,Value,Type\n";

      // Define recursive export function
      std::function<void(const QModelIndex &, const std::string &)> exportNode =
          [&](const QModelIndex &index, const std::string &path) {
            QString key = path.empty() ? "root" : QString::fromStdString(path);
            QString value = model.data(index, Qt::DisplayRole).toString();
            QString type =
                model
                    .data(model.index(index.row(), 1, index.parent()),
                          Qt::DisplayRole)
                    .toString();

            // Write CSV row with proper escaping
            file << "\"" << key.toStdString() << "\",\"" << value.toStdString()
                 << "\",\"" << type.toStdString() << "\"\n";

            // Process children
            for (int i = 0; i < model.rowCount(index); ++i) {
              QModelIndex child = model.index(i, 0, index);
              std::string childPath = path.empty()
                                          ? std::to_string(i)
                                          : path + "." + std::to_string(i);
              exportNode(child, childPath);
            }
          };

      // Start export from root nodes
      for (int i = 0; i < model.rowCount(QModelIndex()); ++i) {
        exportNode(model.index(i, 0, QModelIndex()), std::to_string(i));
      }
    } else if (format == "html") {
      // HTML export with CSS styling
      file << R"(<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JSON Export</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        th { background-color: #0078d4; color: white; font-weight: 600; text-align: left; }
        th, td { padding: 12px 15px; border: 1px solid #ddd; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        tr:hover { background-color: #f1f3f5; }
        .path { font-family: monospace; color: #6c757d; }
        .string { color: #22863a; }
        .number { color: #005cc5; }
        .boolean { color: #e36209; }
        .null { color: #6f42c1; }
        h1 { color: #0078d4; }
    </style>
</head>
<body>
    <h1>JSON Export</h1>
    <table>
        <tr><th>Path</th><th>Value</th><th>Type</th></tr>)";

      // Define recursive export function
      std::function<void(const QModelIndex &, const std::string &)> exportNode =
          [&](const QModelIndex &index, const std::string &path) {
            QString key = path.empty() ? "root" : QString::fromStdString(path);
            QString value = model.data(index, Qt::DisplayRole).toString();
            QString type =
                model
                    .data(model.index(index.row(), 1, index.parent()),
                          Qt::DisplayRole)
                    .toString();

            // Determine CSS class based on type
            QString cssClass = "string";
            if (type == "Number")
              cssClass = "number";
            else if (type == "Boolean")
              cssClass = "boolean";
            else if (type == "null")
              cssClass = "null";

            // Write HTML table row with CSS classes
            file << "<tr><td class=\"path\">"
                 << key.toHtmlEscaped().toStdString() << "</td><td class=\""
                 << cssClass.toStdString() << "\">"
                 << value.toHtmlEscaped().toStdString() << "</td><td>"
                 << type.toHtmlEscaped().toStdString() << "</td></tr>\n";

            // Process children
            for (int i = 0; i < model.rowCount(index); ++i) {
              QModelIndex child = model.index(i, 0, index);
              std::string childPath = path.empty()
                                          ? std::to_string(i)
                                          : path + "." + std::to_string(i);
              exportNode(child, childPath);
            }
          };

      // Start export from root nodes
      for (int i = 0; i < model.rowCount(QModelIndex()); ++i) {
        exportNode(model.index(i, 0, QModelIndex()), std::to_string(i));
      }

      // Close HTML tags
      file << "</table>\n<p>Exported on "
           << QDateTime::currentDateTime().toString().toStdString()
           << "</p>\n</body>\n</html>";
    }
    statusBar->showMessage(tr("Export successful"), 3000);
  } catch (const std::exception &e) {
    QMessageBox::critical(this, tr("Error"), e.what());
  }
}

// Open JSON file with async loading for large files
void JsonEditor::openFile() {
  try {
    QString path = QFileDialog::getOpenFileName(
        this, "Open JSON", QString(), "JSON Files (*.json);;All Files (*)");
    if (path.isEmpty())
      return;

    // Check if file exists and is readable
    QFileInfo fileInfo(path);
    if (!fileInfo.exists() || !fileInfo.isReadable()) {
      throw std::runtime_error("File does not exist or is not readable");
    }

    // Show loading indicator for large files
    if (fileInfo.size() > 10 * 1024 * 1024) { // 10MB
      progressBar->setVisible(true);
      progressBar->setValue(0);
    }

    // Open and parse file
    std::ifstream file(path.toStdString());
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file");
    }

    // Use async loading for large files
    if (fileInfo.size() > 5 * 1024 * 1024) { // 5MB
      // Read file content into memory
      std::string content((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());

      try {
        // Parse JSON and load asynchronously
        Json json = Json::parse(content);
        model.loadAsync(json);

        // Save to recent files
        addToRecentFiles(path);

        // Update UI components
        updateCompleterWordList();
        statusBar->showMessage(tr("Loading file asynchronously..."));
      } catch (const std::exception &e) {
        throw std::runtime_error(std::string("JSON parsing error: ") +
                                 e.what());
      }
    } else {
      // For smaller files, load synchronously
      try {
        Json json = Json::parse(file);
        model.load(json);

        addToRecentFiles(path);
        updateCompleterWordList();
        updateStats();

        statusBar->showMessage(tr("File loaded successfully"), 3000);
        emit jsonLoaded(true);
      } catch (const std::exception &e) {
        throw std::runtime_error(std::string("JSON parsing error: ") +
                                 e.what());
      }
    }
  } catch (const std::exception &e) {
    spdlog::error("Failed to open file: {}", e.what());
    QMessageBox::critical(this, "Error", e.what());
    emit jsonLoaded(false);
  }
}

// Save current JSON to file
void JsonEditor::saveFile() {
  try {
    QString path = QFileDialog::getSaveFileName(
        this, "Save JSON", QString(), "JSON Files (*.json);;All Files (*)");
    if (path.isEmpty())
      return;

    // Format JSON with indentation for readability
    const QString formattedJson = formatJson(4);

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
      throw std::runtime_error("Cannot open file for writing");
    }

    // Use QTextStream for efficient text writing with proper encoding
    QTextStream out(&file);
    out.setEncoding(QStringConverter::Utf8);
    out << formattedJson;

    // Add to recent files
    addToRecentFiles(path);

    statusBar->showMessage(tr("File saved successfully"), 3000);
  } catch (const std::exception &e) {
    spdlog::error("Failed to save file: {}", e.what());
    QMessageBox::critical(this, "Error", e.what());
  }
}

void JsonEditor::loadJson(const nlohmann::json &json) {
  try {
    model.load(json);
    updateStats();
    updateCompleterWordList();
    emit jsonLoaded(true);
  } catch (const std::exception &e) {
    spdlog::error("Failed to load JSON: {}", e.what());
    QMessageBox::critical(this, "Error", e.what());
    emit jsonLoaded(false);
  }
}

nlohmann::json JsonEditor::getJson() const { return model.getJson(); }

void JsonEditor::setupCompleter() {
  completer = new QCompleter(this);
  completer->setModelSorting(QCompleter::CaseInsensitivelySortedModel);
  completer->setCaseSensitivity(Qt::CaseInsensitive);
  completer->setWrapAround(false);
  completer->setCompletionMode(QCompleter::PopupCompletion);
  completer->setMaxVisibleItems(10);

  // Initialize with basic JSON keywords
  wordList = {"true", "false", "null", "{", "}", "[", "]", ":", ",", "\"\""};

  auto *model = new QStringListModel(wordList, completer);
  completer->setModel(model);

  // Apply completer to editors - use intelligent placement
  ElaLineEdit *editor =
      qobject_cast<ElaLineEdit *>(treeView->itemDelegate()->createEditor(
          treeView, QStyleOptionViewItem(), QModelIndex()));

  if (editor) {
    editor->setCompleter(completer);
  }
}

void JsonEditor::updateCompleterWordList() {
  try {
    // Extract keywords from JSON data using parallel algorithms when available
    const Json &json = model.getJson();

    // Stores unique words found in the JSON
    std::unordered_set<std::string> uniqueWords;

    // Extract words recursively from JSON
    std::function<void(const Json &)> extractWords = [&](const Json &node) {
      if (node.is_object()) {
        // Add object keys to word list
        for (const auto &[key, value] : node.items()) {
          uniqueWords.insert(key);
          extractWords(value);
        }
      } else if (node.is_array()) {
        // Process array elements
        for (const auto &element : node) {
          extractWords(element);
        }
      } else if (node.is_string()) {
        // Add string values that appear to be meaningful words (at least 3
        // chars)
        const std::string &str = node.get<std::string>();
        if (str.length() >= 3 &&
            str.find_first_of(" \t\n\r") == std::string::npos) {
          uniqueWords.insert(str);
        }
      }
    };

    // Start extraction from root
    extractWords(json);

    // Convert to QStringList and merge with base keywords
    wordList.clear();
    wordList << "true" << "false" << "null" << "{" << "}" << "[" << "]" << ":"
             << "," << "\"\"";

    // Add extracted words
    for (const auto &word : uniqueWords) {
      wordList << QString::fromStdString(word);
    }

    // Update the completer model
    completer->setModel(new QStringListModel(wordList, completer));
  } catch (const std::exception &e) {
    spdlog::error("Error updating completer list: {}", e.what());
  }
}

void JsonEditor::setupFindReplace() {
  findReplaceDialog = new QDialog(this);
  findReplaceDialog->setWindowTitle(tr("Find & Replace"));

  auto *layout = new QVBoxLayout(findReplaceDialog);

  // Create find/replace form
  auto *formLayout = new QFormLayout();

  findEdit = new ElaLineEdit(findReplaceDialog);
  replaceEdit = new ElaLineEdit(findReplaceDialog);

  formLayout->addRow(tr("Find:"), findEdit);
  formLayout->addRow(tr("Replace:"), replaceEdit);

  // Create button row
  auto *buttonLayout = new QHBoxLayout();

  auto *findNextBtn = new ElaPushButton(tr("Find Next"), findReplaceDialog);
  auto *findAllBtn = new ElaPushButton(tr("Find All"), findReplaceDialog);
  replaceBtn = new ElaPushButton(tr("Replace"), findReplaceDialog);
  replaceAllBtn = new ElaPushButton(tr("Replace All"), findReplaceDialog);
  auto *closeBtn = new ElaPushButton(tr("Close"), findReplaceDialog);

  buttonLayout->addWidget(findNextBtn);
  buttonLayout->addWidget(findAllBtn);
  buttonLayout->addWidget(replaceBtn);
  buttonLayout->addWidget(replaceAllBtn);
  buttonLayout->addWidget(closeBtn);

  layout->addLayout(formLayout);
  layout->addLayout(buttonLayout);

  // Connect signals
  connect(closeBtn, &ElaPushButton::clicked, findReplaceDialog,
          &QDialog::close);

  connect(findNextBtn, &ElaPushButton::clicked, this, [this]() {
    QString text = findEdit->text();
    if (!text.isEmpty()) {
      // Implement find functionality
    }
  });

  connect(findAllBtn, &ElaPushButton::clicked, this, [this]() {
    QString text = findEdit->text();
    if (!text.isEmpty()) {
      QStringList results = model.findAll(text);
      // Show results
    }
  });

  connect(replaceBtn, &ElaPushButton::clicked, this, [this]() {
    QString find = findEdit->text();
    QString replace = replaceEdit->text();
    if (!find.isEmpty()) {
      // Replace current match
    }
  });

  connect(replaceAllBtn, &ElaPushButton::clicked, this, [this]() {
    QString find = findEdit->text();
    QString replace = replaceEdit->text();
    if (!find.isEmpty()) {
      bool success = model.findAndReplace(find, replace);
      statusBar->showMessage(success ? tr("Replaced all occurrences")
                                     : tr("No matches found"),
                             3000);
    }
  });
}

void JsonEditor::handleDroppedFile(const QString &path) {
  try {
    QFileInfo info(path);

    // Only accept JSON files
    if (info.suffix().toLower() != "json") {
      statusBar->showMessage(tr("Only JSON files are supported"), 3000);
      return;
    }

    // Show progress for large files
    if (info.size() > 10 * 1024 * 1024) {
      progressBar->setVisible(true);
      progressBar->setValue(0);
    }

    // Read file
    std::ifstream file(path.toStdString());
    if (!file) {
      throw std::runtime_error("Cannot open dropped file");
    }

    // Parse and load
    Json data = Json::parse(file);

    // Use async loading for large files
    if (info.size() > 5 * 1024 * 1024) {
      model.loadAsync(data);
    } else {
      model.load(data);
      updateCompleterWordList();
      updateStats();
    }

    addToRecentFiles(path);
    statusBar->showMessage(tr("File dropped successfully"), 3000);
  } catch (const std::exception &e) {
    progressBar->setVisible(false);
    QMessageBox::critical(this, tr("Error"), e.what());
  }
}

void JsonEditor::showLoadingProgress(int progress) {
  if (progress < 0 || progress > 100)
    return;

  progressBar->setVisible(true);
  progressBar->setValue(progress);
  statusBar->showMessage(tr("Loading... %1%").arg(progress));

  // Process events to keep UI responsive during long loads
  QApplication::processEvents();
}

void JsonEditor::addToRecentFiles(const QString &filePath) {
  const int maxRecentFiles = 10;

  QSettings settings;
  QStringList recentFiles = settings.value("recentFiles").toStringList();

  // Remove existing entry if present
  recentFiles.removeAll(filePath);

  // Add to front
  recentFiles.prepend(filePath);

  // Limit list size
  while (recentFiles.size() > maxRecentFiles) {
    recentFiles.removeLast();
  }

  // Save updated list
  settings.setValue("recentFiles", recentFiles);

  // Notify listeners
  emit recentFilesChanged(recentFiles);
}

// Drag and drop support
void JsonEditor::dragEnterEvent(QDragEnterEvent *event) {
  // Accept drag if it contains file URLs
  if (event->mimeData()->hasUrls()) {
    const QList<QUrl> urls = event->mimeData()->urls();
    if (urls.size() == 1) {
      QString filePath = urls.first().toLocalFile();
      if (filePath.toLower().endsWith(".json")) {
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