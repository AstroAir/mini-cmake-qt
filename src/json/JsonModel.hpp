#pragma once

#include <QAbstractItemModel>
#include <QFuture>
#include <QtConcurrent>
#include <atomic>
#include <concepts>
#include <coroutine>
#include <nlohmann/json.hpp>
#include <qtmetamacros.h>
#include <shared_mutex>
#include <span>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>
#include <chrono> // 添加缺失的头文件


// C++20: Create JSON concepts
template <typename T>
concept JsonValue = requires(T value) {
  { nlohmann::json(value) } -> std::convertible_to<nlohmann::json>;
};

using Json = nlohmann::json;

/**
 * @class JsonModel
 * @brief A model for representing JSON data in a tree structure.
 *
 * This class provides a model for displaying and editing JSON data in a tree
 * view.
 */
class JsonModel : public QAbstractItemModel {
  Q_OBJECT

  struct Node;
  using NodePtr = std::unique_ptr<Node>;

  struct Node {
    /**
     * @brief Data stored in the node.
     *
     * The data can be a pointer to the original JSON or a new node name.
     */
    std::variant<Json *, QString> data;
    Node *parent = nullptr;        ///< Pointer to the parent node.
    std::vector<NodePtr> children; ///< List of child nodes.

    /**
     * @brief Returns the display value of the node.
     * @return The display value of the node.
     */
    [[nodiscard]] QVariant display() const noexcept;

    // C++20: Create structured node factory
    template <JsonValue T>
    [[nodiscard]] static NodePtr create(T &&value, Node *parent = nullptr) {
      auto node = std::make_unique<Node>();
      node->data = new Json(std::forward<T>(value));
      node->parent = parent;
      return node;
    }

    // Add RAII for owned Json pointers
    ~Node() {
      if (auto *jsonPtr = std::get_if<Json *>(&data)) {
        delete *jsonPtr;
      }
    }
  };

  NodePtr root;  ///< Root node of the JSON tree.
  Json jsonData; ///< The JSON data.
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

  // Memory pooling optimization
  struct MemoryPool {
    static constexpr size_t BLOCK_SIZE = 4096; // Increased for better caching
    static constexpr size_t ALIGNMENT = 16;    // For SIMD alignment

    std::vector<std::unique_ptr<char[]>> blocks;
    std::atomic<size_t> currentIndex = 0;

    void *allocate(size_t size) noexcept;
    void clear() noexcept;

    // C++20: Templated allocation with concepts
    template <std::default_initializable T> T *allocateObject() noexcept {
      void *memory = allocate(sizeof(T));
      if (!memory)
        return nullptr;
      return new (memory) T();
    }
  };
  MemoryPool nodePool;

  // Node cache with LRU eviction policy
  class NodeCache {
    struct CacheEntry {
      Node *node;
      std::chrono::steady_clock::time_point lastAccess;
    };

    std::unordered_map<std::string, CacheEntry> cache;
    std::mutex cacheMutex;
    static constexpr size_t MAX_CACHE_SIZE = 10000;
    static constexpr auto CACHE_EXPIRY = std::chrono::minutes(5);

  public:
    void add(const std::string &path, Node *node) noexcept;
    Node *get(const std::string &path) noexcept;
    void clear() noexcept;
    void evictExpired() noexcept;
  };
  NodeCache nodeCache;

  // Async loading support with improved progress tracking
  QFuture<void> loadFuture;
  std::atomic<bool> isLoading = false;
  std::atomic<int> progressValue = 0;
  std::atomic<size_t> totalNodes = 0;
  std::atomic<size_t> processedNodes = 0;

  // C++20: Task-based coroutine for async operations
  struct JsonTask {
    struct promise_type {
      JsonTask get_return_object() { return {}; }
      std::suspend_never initial_suspend() noexcept { return {}; }
      std::suspend_never final_suspend() noexcept { return {}; }
      void return_void() {}
      void unhandled_exception() {}
    };
  };

public:
  /**
   * @brief Constructor for JsonModel.
   * @param parent The parent object.
   */
  explicit JsonModel(QObject *parent = nullptr) noexcept;

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
  void addUndoCommand(const Command &cmd) noexcept;

  /**
   * @brief Undoes the last command.
   * @return True if the undo was successful, false otherwise.
   */
  bool undo() noexcept;

  /**
   * @brief Redoes the last undone command.
   * @return True if the redo was successful, false otherwise.
   */
  bool redo() noexcept;

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
  [[nodiscard]] size_t getJsonDepth(const Json &j) const noexcept;

  /**
   * @brief Returns the total number of nodes in the JSON data.
   * @param j The JSON data.
   * @return The total number of nodes.
   */
  [[nodiscard]] size_t getJsonNodeCount(const Json &j) const noexcept;

  /**
   * @brief Sets the JSON data.
   * @param json The JSON data to set.
   */
  void setJson(const nlohmann::json &json);

  // Enhanced public methods
  bool validateSchema(std::string_view schemaStr) noexcept;
  QString compressJson() const noexcept;
  QString beautifyJson(int indent = 4) const noexcept;
  bool findAndReplace(const QString &find, const QString &replace,
                      Qt::CaseSensitivity cs = Qt::CaseSensitive) noexcept;
  [[nodiscard]] QStringList
  findAll(const QString &text,
          Qt::CaseSensitivity cs = Qt::CaseSensitive) const noexcept;

  // Async loading with improved API
  void loadAsync(const Json &newData);
  void cancelLoading() noexcept;
  [[nodiscard]] bool isLoadingData() const noexcept { return isLoading; }
  [[nodiscard]] int currentProgress() const noexcept { return progressValue; }

signals:
  void loadProgress(int percent);
  void loadCompleted();
  void loadError(const QString &error);
  void nodeSelected(const QString &path);

private:
  /**
   * @brief Recursively parses the JSON data and builds the tree structure.
   * @param j The JSON data.
   * @return The root node of the parsed tree.
   */
  NodePtr parseJson(const Json &j);

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
   * @brief Progress callback function type
   */
  using ProgressCallback = std::function<void(const Json &, int depth)>;

  /**
   * @brief Recursively parses the JSON data and builds the tree structure.
   * @param j The JSON data to parse
   * @param callback Optional callback for progress reporting
   * @param depth Current recursion depth
   * @return The root node of the parsed tree
   */
  NodePtr parseJson(const Json &j, const ProgressCallback &callback,
                    int depth = 0);

  /**
   * @brief Maximum recursion depth for parsing
   */
  static constexpr int MAX_PARSE_DEPTH = 1000;

  /**
   * @brief Calculate path for a node
   * @param node The node to get the path for
   * @return Path string representing the node location
   */
  [[nodiscard]] std::string getNodePath(Node *node) const noexcept;

  /**
   * @brief Process batch of nodes in parallel
   * @param nodes Vector of JSON nodes to process
   * @param parent Parent node
   * @param callback Progress callback
   * @param depth Current depth
   */
  void processBatch(std::span<const Json> nodes, Node *parent,
                    const ProgressCallback &callback, int depth);
};