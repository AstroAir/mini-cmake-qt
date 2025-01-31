#pragma once

#include "JsonModel.hpp"
#include <QWidget>

class QTreeView;
class QLineEdit;
class QStatusBar;
class QToolBar;
class QPushButton;
class QLabel;

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

public:
  /**
   * @brief Constructor for JsonEditor.
   * @param parent The parent widget.
   */
  JsonEditor(QWidget *parent = nullptr);

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
};