#pragma once

#include "../image/VersionControl.hpp"
#include <QWidget>
#include <memory>

class QLabel;
class ElaPushButton;
class ElaComboBox;
class ElaToolButton;
class CropPreviewWidget;
class ElaListView; // 替换 ElaListWidget
class ElaTreeView; // 替换 ElaTreeWidget
class QSplitter;
class ElaDockWidget;
class QMenu;
class QAction;
class QStandardItemModel;
class ElaLineEdit;
class ElaStatusBar;
class ElaSlider;

class VersionControlWidget : public QWidget {
  Q_OBJECT

public:
  explicit VersionControlWidget(QWidget *parent = nullptr);
  ~VersionControlWidget();

  void setImage(const QImage &image);
  void loadSettings();
  void saveSettings();

  // 新增自定义选项结构
  struct Settings {
    bool autoRefresh = true;
    int previewQuality = 100;
    bool showLineNumbers = true;
    bool compactMode = false;
    QString defaultExportPath;
    QStringList recentBranches;
    int maxRecentBranches = 10;
    bool showAuthorAvatar = true;
    int thumbnailSize = 64;
    bool darkTheme = false;
  };

  void setSettings(const Settings &settings);
  Settings getSettings() const;

signals:
  void commitFinished();
  void branchCreated(const QString &name);
  void checkoutCompleted(const QImage &image);

private slots:
  void onCommit();
  void onCreateBranch();
  void onCheckout();
  void onCreateTag();
  void onShowHistory();
  void onCompareVersions();
  void onMergeBranch();
  void onExportCommit();
  void onSplitHorizontally();
  void onSplitVertically();
  void onToggleHistoryPanel();
  void onToggleInfoPanel();
  void updatePreview();
  void contextMenuEvent(QContextMenuEvent *event) override;
  void onResolveConflicts();
  void onViewDiff();
  void onAddMetadata();
  void onEditMetadata();
  void onDeleteTag();
  void onExportVersion();
  void onImportVersion();
  void onSearchHistory();
  void onFilterBranches();
  void onMergeSelected();
  void onShowDiffVisual();
  void onRevertCommit();
  void onCreatePatch();
  void onApplyPatch();
  void onThemeChanged();
  void onAutoRefreshToggled(bool enabled);
  void onCompactModeToggled(bool enabled);
  void onFilterChanged(const QString &filter);
  void onShowStatistics();
  void onCustomizeColumns();
  void onBatchExport();
  void onSortCommits(int column);

private:
  void setupUi();
  void connectSignals();
  void setupToolbar();
  void createActions();
  void createMenus();
  void setupLayout();
  QWidget *createHistoryPanel();
  QWidget *createInfoPanel();
  QWidget *createPreviewArea();
  void handleException(const std::exception &e);
  void updateLayout(Qt::Orientation orientation);
  void refreshHistory();
  void showCommitInfo(const QString &hash);
  void setupModels();
  void updateBranchList();
  void updateTagList();
  void showDiffDialog(const QString &hash1, const QString &hash2);
  void showMetadataDialog(const QString &hash);
  void handleMergeConflicts(const cv::Mat &base, const cv::Mat &theirs);
  void exportToFile(const QString &hash, const QString &filepath);
  void importFromFile(const QString &filepath);
  void updateCommitInfo(const QString &hash);
  void setupTheme();
  void createShortcuts();
  void updateColumnVisibility();
  void setupStatusBar();
  QWidget * setupSearchWidget();
  void updatePreviewQuality();
  void saveSplitterState();
  void restoreSplitterState();
  void setupDragDrop();
  void updateRecentBranches(const QString &branch);

  // 视图相关
  void updatePanelVisibility();
  
  // 数据处理
  cv::Mat QImageToCvMat(const QImage &image);
  QImage CvMatToQImage(const cv::Mat &mat);


  CropPreviewWidget *imagePreview;
  ElaTreeView *historyTree; // 修改类型
  ElaListView *branchList;  // 修改类型
  ElaListView *tagList;     // 修改类型

  ElaPushButton *commitButton;
  ElaPushButton *branchButton;
  ElaPushButton *tagButton;
  ElaPushButton *mergeButton;
  ElaPushButton *checkoutButton;

  ElaToolButton *compareButton;
  ElaToolButton *exportButton;
  ElaToolButton *refreshButton;

  QSplitter *mainSplitter;
  ElaDockWidget *historyDock;
  ElaDockWidget *infoDock;
  QMenu *viewMenu;
  QMenu *toolsMenu;

  std::unique_ptr<ImageVersionControl> versionControl;
  QImage currentImage;
  QString currentBranch;

  struct {
    QAction *splitHorizontal;
    QAction *splitVertical;
    QAction *toggleHistoryPanel;
    QAction *toggleInfoPanel;
    QAction *refresh;
    QAction *settings;
    QAction *resolveConflicts;
    QAction *viewDiff;
    QAction *addMetadata;
    QAction *editMetadata;
    QAction *deleteTag;
    QAction *exportVersion;
    QAction *importVersion;
    QAction *searchHistory;
    QAction *revertCommit;
    QAction *createPatch;
    QAction *applyPatch;
  } actions;

  Qt::Orientation splitOrientation;
  bool historyPanelVisible;
  bool infoPanelVisible;

  QStandardItemModel *historyModel;
  QStandardItemModel *branchModel;
  QStandardItemModel *tagModel;
  QLabel *commitInfoLabel;

  Settings settings;
  ElaLineEdit *searchBox;
  ElaStatusBar *statusBar;
  QLabel *statusLabel;
  QTimer *autoRefreshTimer;
  ElaComboBox *themeSelector;
  ElaSlider *qualitySlider;
  QSplitter *rightSplitter;
  QHash<QString, bool> columnVisibility;
  QStringList recentBranches;
};
