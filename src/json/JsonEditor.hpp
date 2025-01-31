#pragma once

#include "JsonModel.hpp"
#include "JsonSyntaxHighlighter.hpp"
#include <QWidget>
#include <QCompleter>

class QTreeView;
class QLineEdit;
class QStatusBar;
class QToolBar;
class QPushButton;
class QLabel;
class QProgressBar;
class QDialog;

/**
 * @class JsonEditor
 * @brief A widget for editing JSON files.
 *
 * This class provides a user interface for viewing and editing JSON files.
 * It includes a tree view for displaying the JSON structure, a search bar
 * for filtering content, and a status bar for displaying statistics and
 * messages.
 */
class JsonEditor : public QWidget {
  QTreeView *treeView;   ///< Tree view for displaying JSON structure.
  QLineEdit *searchBar;  ///< Search bar for filtering JSON content.
  JsonModel model;       ///< Model for managing JSON data.
  QStatusBar *statusBar; ///< Status bar for displaying messages and statistics.
  QToolBar *toolbar;     ///< Toolbar for various actions.
  QPushButton *themeBtn; ///< Button for toggling theme.
  bool isDarkTheme = false; ///< Flag indicating if dark theme is enabled.
  QLabel *statsLabel;       ///< Label for displaying statistics.
  JsonSyntaxHighlighter *highlighter; ///< JSON语法高亮器
  QCompleter *completer;              ///< 自动补全器
  QStringList wordList;               ///< 自动补全词列表
  QProgressBar* progressBar; ///< Progress bar for loading progress.
  QDialog* findReplaceDialog; ///< Dialog for find and replace.
  QLineEdit* findEdit; ///< Line edit for find text.
  QLineEdit* replaceEdit; ///< Line edit for replace text.
  QPushButton* replaceBtn; ///< Button for replace.
  QPushButton* replaceAllBtn; ///< Button for replace all.

public:
  /**
   * @brief Constructor for JsonEditor.
   * @param parent The parent widget.
   */
  JsonEditor(QWidget *parent = nullptr);

  /**
   * @brief 加载JSON数据到编辑器
   * @param json 要加载的JSON数据
   */
  void loadJson(const nlohmann::json& json);

  /**
   * @brief 获取当前编辑器中的JSON数据
   * @return 当前的JSON数据
   */
  nlohmann::json getJson() const;

private:
  /**
   * @brief Sets up the user interface.
   */
  void setupUI();

  /**
   * @brief Sets up the toolbar.
   */
  void setupToolbar();

  /**
   * @brief Sets up the status bar.
   */
  void setupStatusBar();

  /**
   * @brief Sets up the connections for signals and slots.
   */
  void setupConnections();

  /**
   * @brief Applies the current style to the widget.
   */
  void applyStyle();

  /**
   * @brief Toggles between light and dark themes.
   */
  void toggleTheme();

  /**
   * @brief Filters the JSON content based on the given text.
   * @param text The text to filter the content by.
   */
  void filterContent(const QString &text);

  /**
   * @brief Updates the statistics displayed in the status bar.
   */
  void updateStats();

  /**
   * @brief Exports the JSON content to the specified format.
   * @param format The format to export to (e.g., JSON, XML).
   */
  void exportTo(const QString &format);

  /**
   * @brief Opens a JSON file.
   */
  void openFile();

  /**
   * @brief Saves the JSON content to a file.
   */
  void saveFile();

  /**
   * @brief 设置自动补全
   */
  void setupCompleter();

  /**
   * @brief 更新自动补全词列表
   */
  void updateCompleterWordList();

  /**
   * @brief Sets up the find and replace functionality.
   */
  void setupFindReplace();

  /**
   * @brief Sets up the drag and drop functionality.
   */
  void setupDragDrop();

  /**
   * @brief Creates the recent files menu.
   */
  void createRecentFilesMenu();

  /**
   * @brief Adds a file to the recent files list.
   * @param path The path of the file to add.
   */
  void addToRecentFiles(const QString& path);

  /**
   * @brief Handles a dropped file.
   * @param path The path of the dropped file.
   */
  void handleDroppedFile(const QString& path);

  /**
   * @brief Shows the loading progress.
   * @param percent The loading progress percentage.
   */
  void showLoadingProgress(int percent);

  /**
   * @brief Shows the JSON schema.
   */
  void showJsonSchema();

  /**
   * @brief Validates the JSON schema.
   */
  void validateJsonSchema();

protected:
  /**
   * @brief Handles the drag enter event.
   * @param event The drag enter event.
   */
  void dragEnterEvent(QDragEnterEvent* event) override;

  /**
   * @brief Handles the drop event.
   * @param event The drop event.
   */
  void dropEvent(QDropEvent* event) override;
};