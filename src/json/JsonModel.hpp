#pragma once

#include <QAbstractItemModel>
#include <nlohmann/json.hpp>
#include <shared_mutex>
#include <variant>
#include <vector>
#include <unordered_map>
#include <QFuture>
#include <QtConcurrent>

using Json = nlohmann::json;

/**
 * @class JsonModel
 * @brief A model for representing JSON data in a tree structure.
 *
 * This class provides a model for displaying and editing JSON data in a tree
 * view.
 */
class JsonModel : public QAbstractItemModel {
  struct Node {
    /**
     * @brief Data stored in the node.
     *
     * The data can be a pointer to the original JSON or a new node name.
     */
    std::variant<Json *, QString> data;
    Node *parent = nullptr; ///< Pointer to the parent node.
    std::vector<std::unique_ptr<Node>> children; ///< List of child nodes.

    /**
     * @brief Returns the display value of the node.
     * @return The display value of the node.
     */
    [[nodiscard]] QVariant display() const noexcept;
  };

  std::unique_ptr<Node> root; ///< Root node of the JSON tree.
  Json jsonData;              ///< The JSON data.
  mutable std::shared_mutex
      dataMutex; ///< Mutex for thread-safe access to the data.

  /**
   * @struct Command
   * @brief Structure representing an undo/redo command.
   */
  struct Command {
    QModelIndex index; ///< The index of the modified item.
    QVariant oldValue; ///< The old value before the modification.
    QVariant newValue; ///< The new value after the modification.
  };
  std::vector<Command> undoStack;  ///< Stack of undo commands.
  std::vector<Command> redoStack;  ///< Stack of redo commands.
  const size_t maxUndoSteps = 100; ///< Maximum number of undo steps.

  // 内存池优化
  struct MemoryPool {
    static constexpr size_t BLOCK_SIZE = 1024;
    std::vector<std::unique_ptr<char[]>> blocks;
    size_t currentIndex = 0;
    
    void* allocate(size_t size);
    void clear();
  };
  MemoryPool nodePool;

  // 节点缓存
  struct NodeCache {
    std::unordered_map<std::string, Node*> cache;
    void add(const std::string& path, Node* node);
    Node* get(const std::string& path);
    void clear();
  };
  NodeCache nodeCache;

  // 异步加载支持
  QFuture<void> loadFuture;
  bool isLoading = false;

public:
  /**
   * @brief Constructor for JsonModel.
   * @param parent The parent object.
   */
  explicit JsonModel(QObject *parent = nullptr);

  /**
   * @brief Returns the complete JSON data.
   * @return The JSON data.
   */
  [[nodiscard]] const Json &getJson() const noexcept;

  /**
   * @brief Creates and returns an index for the specified item.
   * @param row The row number.
   * @param column The column number.
   * @param parent The parent index.
   * @return The created index.
   */
  QModelIndex index(int row, int column,
                    const QModelIndex &parent) const override;

  /**
   * @brief Returns the parent index of the specified item.
   * @param index The index of the item.
   * @return The parent index.
   */
  QModelIndex parent(const QModelIndex &index) const override;

  /**
   * @brief Returns the number of rows under the given parent.
   * @param parent The parent index.
   * @return The number of rows.
   */
  int rowCount(const QModelIndex &parent) const override;

  /**
   * @brief Returns the number of columns for the children of the given parent.
   * @param parent The parent index.
   * @return The number of columns.
   */
  int columnCount(const QModelIndex &parent) const override;

  /**
   * @brief Returns the data for the specified role and section in the header
   * with the given orientation.
   * @param index The index of the item.
   * @param role The role of the data.
   * @return The data for the specified role and section.
   */
  QVariant data(const QModelIndex &index, int role) const override;

  /**
   * @brief Sets the role data for the item at the specified index.
   * @param index The index of the item.
   * @param value The value to set.
   * @param role The role of the data.
   * @return True if the data was set successfully, false otherwise.
   */
  bool setData(const QModelIndex &index, const QVariant &value,
               int role) override;

  /**
   * @brief Adds an undo command to the stack.
   * @param cmd The command to add.
   */
  void addUndoCommand(const Command &cmd);

  /**
   * @brief Undoes the last command.
   * @return True if the undo was successful, false otherwise.
   */
  bool undo();

  /**
   * @brief Redoes the last undone command.
   * @return True if the redo was successful, false otherwise.
   */
  bool redo();

  /**
   * @brief Returns the item flags for the given index.
   * @param index The index of the item.
   * @return The item flags.
   */
  Qt::ItemFlags flags(const QModelIndex &index) const override;

  /**
   * @brief Loads and parses new JSON data.
   * @param newData The new JSON data.
   */
  void load(const Json &newData);

  /**
   * @brief Validates the JSON data.
   * @param j The JSON data to validate.
   */
  void validateJson(const Json &j);

  /**
   * @brief Returns the maximum depth of the JSON data.
   * @param j The JSON data.
   * @return The maximum depth.
   */
  size_t getJsonDepth(const Json &j) const;

  /**
   * @brief Returns the total number of nodes in the JSON data.
   * @param j The JSON data.
   * @return The total number of nodes.
   */
  size_t getJsonNodeCount(const Json &j) const;

  /**
   * @brief Sets the JSON data.
   * @param json The JSON data to set.
   */
  void setJson(const nlohmann::json &json);

  // 新增公共方法
  bool validateSchema(const std::string& schemaStr);
  QString compressJson() const;
  QString beautifyJson(int indent = 4) const;
  bool findAndReplace(const QString& find, const QString& replace, 
                      Qt::CaseSensitivity cs = Qt::CaseSensitive);
  QStringList findAll(const QString& text, 
                      Qt::CaseSensitivity cs = Qt::CaseSensitive) const;
  
  // 异步加载支持
  void loadAsync(const Json& newData);
  bool isLoadingData() const { return isLoading; }

signals:
  void loadProgress(int percent);
  void loadCompleted();
  void loadError(const QString& error);

private:
  /**
   * @brief Recursively parses the JSON data and builds the tree structure.
   * @param j The JSON data.
   * @return The root node of the parsed tree.
   */
  std::unique_ptr<Node> parseJson(const Json &j);

  /**
   * @brief Finds the row number of the specified node among its siblings.
   * @param node The node to find the row number for.
   * @return The row number.
   */
  [[nodiscard]] int findRow(Node *node) const noexcept;

  /**
   * @brief Returns the type of the node as a string.
   * @param node The node to get the type for.
   * @return The type of the node as a string.
   */
  [[nodiscard]] QString typeString(Node *node) const noexcept;

  /**
   * @brief 进度回调函数类型
   */
  using ProgressCallback = std::function<void(const Json&)>;

  /**
   * @brief Recursively parses the JSON data and builds the tree structure.
   * @param j The JSON data to parse
   * @param callback Optional callback for progress reporting
   * @param depth Current recursion depth
   * @return The root node of the parsed tree
   */
  std::unique_ptr<Node> parseJson(
      const Json &j, 
      const ProgressCallback& callback,
      int depth = 0
  );

  /**
   * @brief Maximum recursion depth for parsing
   */
  static constexpr int MAX_PARSE_DEPTH = 1000;
};