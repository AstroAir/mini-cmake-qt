#pragma once

#include <QWidget>
#include <QStringList>
#include <QCompleter>
#include <QSettings>
#include "JsonModel.hpp"
#include "JsonSyntaxHighlighter.hpp"

class QLabel;
class QDialog;
class ElaLineEdit;
class ElaPushButton;
class ElaTreeView;
class ElaToolBar;
class ElaStatusBar;
class ElaProgressBar;

/**
 * @class JsonEditor
 * @brief JSON editor widget with tree view, syntax highlighting and advanced features
 * 
 * Provides UI to edit, validate and visualize JSON data with support for:
 * - Syntax highlighting
 * - Schema validation
 * - Search/replace functionality
 * - Async loading of large files
 * - Drag and drop support
 */
class JsonEditor : public QWidget {
    Q_OBJECT

public:
    explicit JsonEditor(QWidget *parent = nullptr);
    ~JsonEditor() override = default;
    
    // Disable copying
    JsonEditor(const JsonEditor&) = delete;
    JsonEditor& operator=(const JsonEditor&) = delete;
    
    // Enable moving
    JsonEditor(JsonEditor&&) noexcept = default;
    JsonEditor& operator=(JsonEditor&&) noexcept = default;

    /**
     * @brief Load JSON data into the editor
     * @param json The JSON data to load
     */
    void loadJson(const nlohmann::json& json);
    
    /**
     * @brief Get the current JSON data from the editor
     * @return Current JSON data
     */
    [[nodiscard]] nlohmann::json getJson() const;
    
    /**
     * @brief Check if editor is currently loading data asynchronously
     * @return True if async loading is in progress
     */
    [[nodiscard]] bool isLoading() const noexcept { return model.isLoadingData(); }
    
    /**
     * @brief Format the current JSON with specified indentation
     * @param indent Number of spaces for indentation (1-8)
     * @return Formatted JSON string
     */
    [[nodiscard]] QString formatJson(int indent = 4) const noexcept {
        return model.beautifyJson(indent);
    }

public slots:
    /**
     * @brief Open a JSON file from disk
     */
    void openFile();
    
    /**
     * @brief Save current JSON to file
     */
    void saveFile();
    
    /**
     * @brief Toggle between dark and light theme
     */
    void toggleTheme();
    
    /**
     * @brief Filter tree view content based on search text
     * @param text Search text
     */
    void filterContent(const QString& text);
    
    /**
     * @brief Show loading progress in the UI
     * @param progress Progress percentage (0-100)
     */
    void showLoadingProgress(int progress);

signals:
    /**
     * @brief Signal emitted when recent files list changes
     * @param files List of recent file paths
     */
    void recentFilesChanged(const QStringList& files);
    
    /**
     * @brief Signal emitted when JSON data is changed
     */
    void jsonChanged();
    
    /**
     * @brief Signal emitted when JSON data is loaded
     * @param success Whether load was successful
     */
    void jsonLoaded(bool success);

protected:
    /**
     * @brief Process drag enter events for drag & drop support
     * @param event Drag enter event
     */
    void dragEnterEvent(QDragEnterEvent* event) override;
    
    /**
     * @brief Process drop events for drag & drop support
     * @param event Drop event
     */
    void dropEvent(QDropEvent* event) override;

private:
    // UI Setup methods
    void setupUI();
    void setupToolbar();
    void setupStatusBar();
    void setupConnections();
    void setupCompleter();
    void setupFindReplace();
    void applyStyle();
    
    // Utility methods
    void updateStats();
    void updateCompleterWordList();
    void exportTo(const QString& format);
    void handleDroppedFile(const QString& path);
    void addToRecentFiles(const QString& filePath);
    
    // Model and UI components
    JsonModel model;
    JsonSyntaxHighlighter* highlighter = nullptr;
    
    // UI elements
    ElaTreeView* treeView = nullptr;
    ElaToolBar* toolbar = nullptr;
    ElaLineEdit* searchBar = nullptr;
    ElaStatusBar* statusBar = nullptr;
    ElaProgressBar* progressBar = nullptr;
    QLabel* statsLabel = nullptr;
    ElaPushButton* themeBtn = nullptr;
    
    // Find & Replace dialog
    QDialog* findReplaceDialog = nullptr;
    ElaLineEdit* findEdit = nullptr;
    ElaLineEdit* replaceEdit = nullptr;
    ElaPushButton* replaceBtn = nullptr;
    ElaPushButton* replaceAllBtn = nullptr;
    
    // Auto-completion
    QCompleter* completer = nullptr;
    QStringList wordList;
    
    // State tracking
    bool isDarkTheme = true;
};