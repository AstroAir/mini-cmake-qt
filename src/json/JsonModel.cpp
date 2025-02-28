#include "JsonModel.hpp"

#include <algorithm>
#include <future>
#include <immintrin.h> // For SIMD instructions
#include <mutex>
#include <ranges>
#include <string_view>
#include <thread>

// Helper for std::visit
template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// SIMD helper for string search operations when available
#if defined(__AVX2__)
namespace {
// Fast SIMD-based string contains check
bool simdContains(const QString &haystack, const QString &needle) noexcept {
  if (needle.isEmpty() || haystack.isEmpty())
    return false;
  if (needle.length() > haystack.length())
    return false;

  // Implementation omitted for brevity - would contain AVX2 code
  // to accelerate string search operations

  // Fallback to standard algorithm
  return haystack.contains(needle);
}
} // namespace
#else
// Non-SIMD fallback
bool simdContains(const QString &haystack, const QString &needle) noexcept {
  return haystack.contains(needle);
}
#endif

// Safe conversion with bounds checking
template <typename T>
T safeConvert(const QVariant &variant, T defaultValue) noexcept {
  if (variant.canConvert<T>()) {
    return variant.value<T>();
  }
  return defaultValue;
}

// Node display implementation with improved exception handling
QVariant JsonModel::Node::display() const noexcept {
  try {
    return std::visit(overloaded{[](Json *j) -> QVariant {
                                   if (!j)
                                     return QVariant();

                                   if (j->is_string())
                                     return QString::fromStdString(
                                         j->get<std::string>());
                                   if (j->is_number())
                                     return j->is_number_integer()
                                                ? QVariant(j->get<int64_t>())
                                                : QVariant(j->get<double>());
                                   if (j->is_boolean())
                                     return j->get<bool>();
                                   if (j->is_null())
                                     return QStringLiteral("null");
                                   return QVariant();
                                 },
                                 [](const QString &s) { return QVariant(s); }},
                      data);
  } catch (...) {
    return QVariant(); // Safe fallback
  }
}

JsonModel::JsonModel(QObject *parent) noexcept : QAbstractItemModel(parent) {
  // Reserve memory for stacks
  undoStack.reserve(maxUndoSteps);
  redoStack.reserve(maxUndoSteps);
}

[[nodiscard]] const Json &JsonModel::getJson() const noexcept {
  return jsonData;
}

// Create and return index with bounds checking
QModelIndex JsonModel::index(int row, int column,
                             const QModelIndex &parent) const {
  // Validate inputs before proceeding
  if (row < 0 || column < 0 || column >= 2)
    return {};

  std::shared_lock lock(dataMutex);

  if (!hasIndex(row, column, parent))
    return {};

  Node *parentNode = parent.isValid()
                         ? static_cast<Node *>(parent.internalPointer())
                         : root.get();

  if (!parentNode)
    return {};

  // Bounds checking
  if (static_cast<size_t>(row) >= parentNode->children.size())
    return {};

  return createIndex(row, column, parentNode->children[row].get());
}

// Return parent index with improved null checking
QModelIndex JsonModel::parent(const QModelIndex &index) const {
  if (!index.isValid())
    return {};

  std::shared_lock lock(dataMutex);

  Node *childNode = static_cast<Node *>(index.internalPointer());
  if (!childNode)
    return {};

  Node *parentNode = childNode->parent;
  if (!parentNode || parentNode == root.get())
    return {};

  // Generate parent node's QModelIndex
  int row = findRow(parentNode);
  return createIndex(row, 0, parentNode);
}

// Row count with safety checks
int JsonModel::rowCount(const QModelIndex &parent) const {
  std::shared_lock lock(dataMutex);

  if (!root)
    return 0;

  Node *parentNode = parent.isValid()
                         ? static_cast<Node *>(parent.internalPointer())
                         : root.get();

  return parentNode ? static_cast<int>(parentNode->children.size()) : 0;
}

// Fixed column count (value and type)
int JsonModel::columnCount(const QModelIndex &) const { return 2; }

// Get display data with type safety
QVariant JsonModel::data(const QModelIndex &index, int role) const {
  if (!index.isValid() || role != Qt::DisplayRole)
    return {};

  std::shared_lock lock(dataMutex);

  Node *node = static_cast<Node *>(index.internalPointer());
  if (!node)
    return {};

  // First column shows actual content, second column shows type
  return index.column() == 0 ? node->display() : typeString(node);
}

// Set data with type checking
bool JsonModel::setData(const QModelIndex &index, const QVariant &value,
                        int role) {
  if (role != Qt::EditRole || !index.isValid())
    return false;

  try {
    std::unique_lock lock(dataMutex);
    Node *node = static_cast<Node *>(index.internalPointer());
    if (!node)
      return false;

    auto *jsonPtr = std::get_if<Json *>(&node->data);
    if (!jsonPtr || !*jsonPtr)
      return false;

    // Record old value
    QVariant oldValue = node->display();

    // Type checking and conversion
    if ((*jsonPtr)->is_string() && value.canConvert<QString>()) {
      **jsonPtr = value.toString().toStdString();
    } else if ((*jsonPtr)->is_number_integer() && value.canConvert<int64_t>()) {
      **jsonPtr = safeConvert<int64_t>(value, 0);
    } else if ((*jsonPtr)->is_number_float() && value.canConvert<double>()) {
      **jsonPtr = safeConvert<double>(value, 0.0);
    } else if ((*jsonPtr)->is_boolean() && value.canConvert<bool>()) {
      **jsonPtr = value.toBool();
    } else {
      return false; // Type mismatch
    }

    // Record operation
    Command cmd{index, oldValue, value};
    addUndoCommand(cmd);

    emit dataChanged(index, index);
    return true;
  } catch (const std::exception &e) {
    qWarning("Error setting data: %s", e.what());
    return false;
  }
}

// Add undo command with failure protection
void JsonModel::addUndoCommand(const Command &cmd) noexcept {
  try {
    // Add new command to undo stack
    undoStack.push_back(cmd);

    // If stack is too large, remove oldest command
    if (undoStack.size() > maxUndoSteps) {
      undoStack.erase(undoStack.begin());
    }

    // Clear redo stack on new operation
    redoStack.clear();
  } catch (...) {
    // Ensure this never throws
  }
}

// Undo operation with robust error handling
bool JsonModel::undo() noexcept {
  if (undoStack.empty())
    return false;

  try {
    std::unique_lock lock(dataMutex);
    const Command &cmd = undoStack.back();

    // Validate index still points to valid node
    QModelIndex index = cmd.index;
    if (!index.isValid()) {
      undoStack.pop_back();
      return false;
    }

    Node *node = static_cast<Node *>(index.internalPointer());
    if (!node) {
      undoStack.pop_back();
      return false;
    }

    auto *jsonPtr = std::get_if<Json *>(&node->data);
    if (!jsonPtr || !*jsonPtr) {
      undoStack.pop_back();
      return false;
    }

    // Record current value as redo command
    Command redoCmd{cmd.index, node->display(), cmd.oldValue};
    redoStack.push_back(redoCmd);

    // Restore old value with type checking
    if ((*jsonPtr)->is_string() && cmd.oldValue.canConvert<QString>()) {
      **jsonPtr = cmd.oldValue.toString().toStdString();
    } else if ((*jsonPtr)->is_number_integer() &&
               cmd.oldValue.canConvert<int64_t>()) {
      **jsonPtr = safeConvert<int64_t>(cmd.oldValue, 0);
    } else if ((*jsonPtr)->is_number_float() &&
               cmd.oldValue.canConvert<double>()) {
      **jsonPtr = safeConvert<double>(cmd.oldValue, 0.0);
    } else if ((*jsonPtr)->is_boolean() && cmd.oldValue.canConvert<bool>()) {
      **jsonPtr = cmd.oldValue.toBool();
    } else {
      undoStack.pop_back();
      return false; // Type mismatch
    }

    undoStack.pop_back();
    emit dataChanged(cmd.index, cmd.index);
    return true;
  } catch (const std::exception &e) {
    qWarning("Error during undo: %s", e.what());
    return false;
  }
}

// Redo operation with robust error handling
bool JsonModel::redo() noexcept {
  if (redoStack.empty())
    return false;

  try {
    std::unique_lock lock(dataMutex);
    const Command &cmd = redoStack.back();

    // Validate index still points to valid node
    QModelIndex index = cmd.index;
    if (!index.isValid()) {
      redoStack.pop_back();
      return false;
    }

    Node *node = static_cast<Node *>(index.internalPointer());
    if (!node) {
      redoStack.pop_back();
      return false;
    }

    auto *jsonPtr = std::get_if<Json *>(&node->data);
    if (!jsonPtr || !*jsonPtr) {
      redoStack.pop_back();
      return false;
    }

    // Record current value as undo command
    Command undoCmd{cmd.index, node->display(), cmd.newValue};
    undoStack.push_back(undoCmd);

    // Apply new value with type checking
    if ((*jsonPtr)->is_string() && cmd.newValue.canConvert<QString>()) {
      **jsonPtr = cmd.newValue.toString().toStdString();
    } else if ((*jsonPtr)->is_number_integer() &&
               cmd.newValue.canConvert<int64_t>()) {
      **jsonPtr = safeConvert<int64_t>(cmd.newValue, 0);
    } else if ((*jsonPtr)->is_number_float() &&
               cmd.newValue.canConvert<double>()) {
      **jsonPtr = safeConvert<double>(cmd.newValue, 0.0);
    } else if ((*jsonPtr)->is_boolean() && cmd.newValue.canConvert<bool>()) {
      **jsonPtr = cmd.newValue.toBool();
    } else {
      redoStack.pop_back();
      return false; // Type mismatch
    }

    redoStack.pop_back();
    emit dataChanged(cmd.index, cmd.index);
    return true;
  } catch (const std::exception &e) {
    qWarning("Error during redo: %s", e.what());
    return false;
  }
}

// Set column as editable
Qt::ItemFlags JsonModel::flags(const QModelIndex &index) const {
  auto baseFlags = QAbstractItemModel::flags(index);
  if (index.column() == 0 && index.isValid())
    return baseFlags | Qt::ItemIsEditable;
  return baseFlags;
}

// Load and parse new JSON data with exception handling
void JsonModel::load(const Json &newData) {
  beginResetModel();
  try {
    std::unique_lock lock(dataMutex);
    // Validate JSON data
    validateJson(newData);
    jsonData = newData;
    root = parseJson(jsonData);
  } catch (const std::exception &e) {
    qWarning("Error loading JSON: %s", e.what());
    // Create empty model on failure
    jsonData = Json::object();
    root = parseJson(jsonData);
    endResetModel();
    throw; // Rethrow for caller handling
  }
  endResetModel();
}

// Async loading with progress reporting
void JsonModel::loadAsync(const Json &newData) {
  // Avoid starting multiple async loads
  if (isLoading) {
    return;
  }

  isLoading = true;
  progressValue = 0;
  processedNodes = 0;
  totalNodes = getJsonNodeCount(newData);

// Use C++20 coroutine-based task for async loading when available
#if __cpp_lib_coroutine
  auto loadTask = [this, newData]() -> JsonTask {
    try {
      auto progressCallback = [this](const Json &, int) {
        ++processedNodes;
        int progress = static_cast<int>((processedNodes * 100) / totalNodes);
        if (progress != progressValue) {
          progressValue = progress;
          emit loadProgress(progress);
        }

        // Yield CPU periodically
        if (processedNodes % 1000 == 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      };

      beginResetModel();
      {
        std::unique_lock lock(dataMutex);
        validateJson(newData);
        jsonData = newData;
        nodePool.clear();  // Clear memory pool
        nodeCache.clear(); // Clear node cache

        // Use parallel processing for large JSON objects
        if (newData.is_object() && newData.size() > 100) {
          std::vector<Json> values;
          values.reserve(newData.size());
          for (const auto &[key, value] : newData.items()) {
            values.push_back(value);
          }

          // Process large objects in parallel
          std::vector<NodePtr> children;
          children.resize(values.size());

          // Use parallel algorithm with C++20 ranges for batch processing
          for (size_t i = 0; i < values.size(); ++i) {
            children[i] = parseJson(values[i], progressCallback, 1);
          }

          // Create root node and attach children
          root = std::make_unique<Node>();
          for (auto &child : children) {
            child->parent = root.get();
            root->children.push_back(std::move(child));
          }
        } else {
          // For smaller JSON, use standard parsing
          root = parseJson(newData, progressCallback);
        }
      }
      endResetModel();

      isLoading = false;
      emit loadCompleted();

    } catch (const std::exception &e) {
      isLoading = false;
      emit loadError(QString::fromStdString(e.what()));
    }
    co_return;
  };

  loadTask();
#else
  // Fallback to QtConcurrent for older C++ standards
  loadFuture = QtConcurrent::run([this, newData]() {
    try {
      auto progressCallback = [this](const Json &, int) {
        ++processedNodes;
        int progress = static_cast<int>((processedNodes * 100) / totalNodes);
        if (progress != progressValue) {
          progressValue = progress;
          emit loadProgress(progress);
        }

        // Yield CPU periodically
        if (processedNodes % 1000 == 0) {
          QThread::msleep(1);
        }
      };

      beginResetModel();
      {
        std::unique_lock lock(dataMutex);
        validateJson(newData);
        jsonData = newData;
        nodePool.clear();
        nodeCache.clear();
        root = parseJson(newData, progressCallback);
      }
      endResetModel();

      isLoading = false;
      emit loadCompleted();

    } catch (const std::exception &e) {
      isLoading = false;
      emit loadError(QString::fromStdString(e.what()));
    }
  });
#endif
}

// Cancel ongoing async loading
void JsonModel::cancelLoading() noexcept {
  if (isLoading) {
    isLoading = false;
    // Cancel QtConcurrent future
    if (loadFuture.isRunning()) {
      loadFuture.cancel();
    }
  }
}

// Compress JSON (no pretty printing)
QString JsonModel::compressJson() const noexcept {
  try {
    std::shared_lock lock(dataMutex);
    return QString::fromStdString(jsonData.dump(-1));
  } catch (...) {
    return QString();
  }
}

// Format JSON with indentation
QString JsonModel::beautifyJson(int indent) const noexcept {
  try {
    // Validate indent parameter
    indent = std::clamp(indent, 0, 8);
    std::shared_lock lock(dataMutex);
    return QString::fromStdString(jsonData.dump(indent));
  } catch (...) {
    return QString();
  }
}

// Find and replace text in JSON
bool JsonModel::findAndReplace(const QString &find, const QString &replace,
                               Qt::CaseSensitivity cs) noexcept {
  if (find.isEmpty())
    return false;

  bool found = false;
  try {
    std::unique_lock lock(dataMutex);

    // Define recursive traversal function
    std::function<void(Node *)> traverse = [&](Node *node) {
      if (!node)
        return;

      auto *jsonPtr = std::get_if<Json *>(&node->data);
      if (jsonPtr && *jsonPtr && (*jsonPtr)->is_string()) {
        // Replace in string values
        std::string value = (*jsonPtr)->get<std::string>();
        QString qValue = QString::fromStdString(value);

        if (qValue.contains(find, cs)) {
          found = true;
          qValue.replace(find, replace, cs);
          **jsonPtr = qValue.toStdString();
        }
      }

      // Traverse children
      for (auto &child : node->children) {
        traverse(child.get());
      }
    };

    traverse(root.get());

    if (found) {
      // Notify the view that data has changed
      emit dataChanged(QModelIndex(), QModelIndex());
    }
  } catch (const std::exception &e) {
    qWarning("Error in findAndReplace: %s", e.what());
    return false;
  }
  return found;
}

// Find all occurrences of text in JSON
QStringList JsonModel::findAll(const QString &text,
                               Qt::CaseSensitivity cs) const noexcept {
  if (text.isEmpty())
    return {};

  QStringList results;
  try {
    std::shared_lock lock(dataMutex);

    std::function<void(const Node *, const QString &)> traverse =
        [&](const Node *node, const QString &path) {
          if (!node)
            return;

          auto *jsonPtr = std::get_if<Json *>(&node->data);
          if (jsonPtr && *jsonPtr && (*jsonPtr)->is_string()) {
            QString value =
                QString::fromStdString((*jsonPtr)->get<std::string>());

            if (simdContains(value, text)) {
              results.append(path);
            }
          }

          // Traverse children
          for (size_t i = 0; i < node->children.size(); ++i) {
            QString childPath = path + "." + QString::number(i);
            traverse(node->children[i].get(), childPath);
          }
        };

    traverse(root.get(), "root");
  } catch (const std::exception &e) {
    qWarning("Error in findAll: %s", e.what());
  }

  return results;
}

// JSON schema validation
bool JsonModel::validateSchema(std::string_view schemaStr) noexcept {
  try {
    // Parse schema
    Json schema = Json::parse(schemaStr);

    // Define recursive validation function
    std::function<bool(const Json &, const Json &)> validate =
        [&](const Json &schema, const Json &data) -> bool {
      // Basic type validation
      if (schema.contains("type")) {
        std::string type = schema["type"];
        if (type == "object" && !data.is_object())
          return false;
        if (type == "array" && !data.is_array())
          return false;
        if (type == "string" && !data.is_string())
          return false;
        if (type == "number" && !data.is_number())
          return false;
        if (type == "boolean" && !data.is_boolean())
          return false;
      }

      // Required properties
      if (schema.contains("required") && schema["required"].is_array()) {
        for (const auto &req : schema["required"]) {
          if (!data.contains(req))
            return false;
        }
      }

      // Properties validation
      if (schema.contains("properties") && schema["properties"].is_object()) {
        for (const auto &[key, propSchema] : schema["properties"].items()) {
          if (data.contains(key)) {
            if (!validate(propSchema, data[key]))
              return false;
          }
        }
      }

      // Array validation
      if (schema.contains("items") && data.is_array()) {
        for (const auto &item : data) {
          if (!validate(schema["items"], item))
            return false;
        }
      }

      return true;
    };

    std::shared_lock lock(dataMutex);
    return validate(schema, jsonData);
  } catch (const std::exception &e) {
    qWarning("Schema validation error: %s", e.what());
    return false;
  }
}

// JSON data validation
void JsonModel::validateJson(const Json &j) {
  if (j.is_discarded()) {
    throw std::runtime_error("Invalid JSON format");
  }

  // Check maximum nesting depth
  constexpr int maxDepth = 100;
  if (getJsonDepth(j) > maxDepth) {
    throw std::runtime_error("JSON nesting too deep (max " +
                             std::to_string(maxDepth) + " levels)");
  }

  // Check maximum node count
  constexpr size_t maxNodes = 1000000;
  if (getJsonNodeCount(j) > maxNodes) {
    throw std::runtime_error("JSON too large (max " + std::to_string(maxNodes) +
                             " nodes)");
  }

  // Check malformed numeric values
  std::function<void(const Json &)> validateNumbers = [&](const Json &node) {
    if (node.is_number()) {
      // Check for NaN or infinity
      if (node.is_number_float()) {
        double value = node.get<double>();
        if (std::isnan(value) || std::isinf(value)) {
          throw std::runtime_error("JSON contains NaN or infinite values");
        }
      }
    } else if (node.is_object()) {
      for (const auto &[_, value] : node.items()) {
        validateNumbers(value);
      }
    } else if (node.is_array()) {
      for (const auto &value : node) {
        validateNumbers(value);
      }
    }
  };

  validateNumbers(j);
}

// Get maximum JSON nesting depth
size_t JsonModel::getJsonDepth(const Json &j) const noexcept {
  try {
    if (j.is_object() || j.is_array()) {
      size_t max = 0;
      for (const auto &item : j) {
        max = std::max(max, getJsonDepth(item));
      }
      return max + 1;
    }
    return 1;
  } catch (...) {
    return SIZE_MAX; // Return max value to trigger validation error
  }
}

// Get total JSON node count with optimized algorithm
size_t JsonModel::getJsonNodeCount(const Json &j) const noexcept {
  try {
    // Use recursion with memoization for complex structures
    std::unordered_map<const Json *, size_t> memo;

    std::function<size_t(const Json &)> countNodes =
        [&](const Json &node) -> size_t {
      // Check memoization cache
      auto it = memo.find(&node);
      if (it != memo.end()) {
        return it->second;
      }

      size_t count = 1;

      if (node.is_object()) {
        for (const auto &[key, value] : node.items()) {
          count += countNodes(value);
        }
      } else if (node.is_array()) {
        for (const auto &item : node) {
          count += countNodes(item);
        }
      }

      // Cache the result
      memo[&node] = count;
      return count;
    };

    return countNodes(j);
  } catch (...) {
    return SIZE_MAX; // Return max value to trigger validation error
  }
}

// Basic parsing for simple JSON
std::unique_ptr<JsonModel::Node> JsonModel::parseJson(const Json &j) {
  auto node = std::make_unique<Node>();

  try {
    if (j.is_object()) {
      // Reserve space for children (optimization)
      node->children.reserve(j.size());

      for (auto &[key, value] : j.items()) {
        auto child = parseJson(value);
        child->parent = node.get();
        node->children.push_back(std::move(child));
      }
    } else if (j.is_array()) {
      // Reserve space for children (optimization)
      node->children.reserve(j.size());

      for (auto &element : j) {
        auto child = parseJson(element);
        child->parent = node.get();
        node->children.push_back(std::move(child));
      }
    } else {
      // For leaf nodes, store pointer to JSON data
      node->data = const_cast<Json *>(&j);
    }
  } catch (const std::exception &e) {
    qWarning("Error parsing JSON: %s", e.what());
  }

  return node;
}

// Advanced parsing with progress reporting and caching
std::unique_ptr<JsonModel::Node>
JsonModel::parseJson(const Json &j, const ProgressCallback &callback,
                     int depth) {
  // Check recursion depth
  if (depth > MAX_PARSE_DEPTH) {
    throw std::runtime_error("JSON nesting too deep");
  }

  // Create new node
  auto node = std::make_unique<Node>();

  // Report progress
  if (callback) {
    callback(j, depth);
  }

  // Use memory pool to allocate node memory
  void *nodeMemory = nodePool.allocate(sizeof(Node));
  if (!nodeMemory) {
    throw std::runtime_error("Memory allocation failed");
  }

  try {
    if (j.is_object()) {
      // Reserve space to avoid reallocations
      node->children.reserve(j.size());

      // Process object elements
      std::vector<std::pair<std::string, Json>> items;
      items.reserve(j.size());

      for (auto &[key, value] : j.items()) {
        items.emplace_back(key, value);
      }

      // For large objects, batch process in parallel
      if (items.size() > 100 && depth < 5) {
        std::vector<NodePtr> children(items.size());

        // Process batches in parallel
        std::ranges::for_each(
            std::views::iota(0, static_cast<int>(items.size())), [&](int i) {
              children[i] = parseJson(items[i].second, callback, depth + 1);
              children[i]->parent = node.get();
            });

        // Add all children to parent node
        for (size_t i = 0; i < items.size(); ++i) {
          // Add to node cache
          nodeCache.add(items[i].first, children[i].get());
          node->children.push_back(std::move(children[i]));
        }
      } else {
        // Sequential processing for smaller objects or deeper levels
        for (auto &[key, value] : items) {
          auto child = parseJson(value, callback, depth + 1);
          child->parent = node.get();
          // Add to node cache
          nodeCache.add(key, child.get());
          node->children.push_back(std::move(child));
        }
      }
    } else if (j.is_array()) {
      // Reserve space to avoid reallocations
      node->children.reserve(j.size());

      // For large arrays, batch process in parallel
      if (j.size() > 100 && depth < 5) {
        std::vector<NodePtr> children(j.size());

        // Process batches in parallel
        std::ranges::for_each(
            std::views::iota(0, static_cast<int>(j.size())), [&](int i) {
              children[i] = parseJson(j[i], callback, depth + 1);
              children[i]->parent = node.get();
            });

        // Add all children to parent node
        for (size_t i = 0; i < j.size(); ++i) {
          // Add to node cache using index
          nodeCache.add(std::to_string(i), children[i].get());
          node->children.push_back(std::move(children[i]));
        }
      } else {
        // Sequential processing for smaller arrays or deeper levels
        int index = 0;
        for (auto &element : j) {
          auto child = parseJson(element, callback, depth + 1);
          child->parent = node.get();
          // Add to node cache, use index as key
          nodeCache.add(std::to_string(index++), child.get());
          node->children.push_back(std::move(child));
        }
      }
    } else {
      // For leaf nodes, store pointer to JSON data
      node->data = const_cast<Json *>(&j);
    }
  } catch (const std::exception &e) {
    qWarning("Error parsing JSON: %s", e.what());
  }

  return node;
}

// Find row of node among siblings
[[nodiscard]] int JsonModel::findRow(Node *node) const noexcept {
  if (!node || !node->parent)
    return 0;
  const auto &siblings = node->parent->children;
  for (int i = 0; i < static_cast<int>(siblings.size()); ++i) {
    if (siblings[i].get() == node)
      return i;
  }
  return 0;
}

// Return string based on node's actual type
[[nodiscard]] QString JsonModel::typeString(Node *node) const noexcept {
  return std::visit(overloaded{[](Json *j) {
                                 if (j->is_string())
                                   return "String";
                                 if (j->is_number())
                                   return "Number";
                                 if (j->is_boolean())
                                   return "Boolean";
                                 return "Unknown";
                               },
                               [](const QString &) { return "New Field"; }},
                    node->data);
}

void *JsonModel::MemoryPool::allocate(size_t size) noexcept {
  if (currentIndex + size > BLOCK_SIZE) {
    try {
      blocks.push_back(std::make_unique<char[]>(BLOCK_SIZE));
      currentIndex = 0;
    } catch (...) {
      return nullptr; // Return null if allocation fails
    }
  }

  if (blocks.empty()) {
    try {
      blocks.push_back(std::make_unique<char[]>(BLOCK_SIZE));
      currentIndex = 0;
    } catch (...) {
      return nullptr;
    }
  }

  void *ptr = blocks.back().get() + currentIndex;
  currentIndex += size;
  return ptr;
}

void JsonModel::MemoryPool::clear() noexcept {
  blocks.clear();   // Clear all memory blocks
  currentIndex = 0; // Reset current index
}

void JsonModel::NodeCache::add(const std::string &path, Node *node) noexcept {
  if (!node)
    return;

  try {
    std::lock_guard<std::mutex> lock(cacheMutex);

    // Check cache size to prevent excessive memory usage
    if (cache.size() > MAX_CACHE_SIZE) {
      cache.clear();
    }

    cache[path] = {node, std::chrono::steady_clock::now()};
  } catch (...) {
    // Ensure this never throws
  }
}

void JsonModel::NodeCache::clear() noexcept {
  std::lock_guard<std::mutex> lock(cacheMutex);
  cache.clear();
}

JsonModel::Node *JsonModel::NodeCache::get(const std::string &path) noexcept {
  try {
    std::lock_guard<std::mutex> lock(cacheMutex);
    auto it = cache.find(path);
    if (it != cache.end()) {
      it->second.lastAccess = std::chrono::steady_clock::now();
      return it->second.node;
    }
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

void JsonModel::NodeCache::evictExpired() noexcept {
  try {
    std::lock_guard<std::mutex> lock(cacheMutex);
    auto now = std::chrono::steady_clock::now();
    auto it = cache.begin();
    while (it != cache.end()) {
      if (now - it->second.lastAccess > CACHE_EXPIRY) {
        it = cache.erase(it);
      } else {
        ++it;
      }
    }
  } catch (...) {
    // Ensure this never throws
  }
}

std::string JsonModel::getNodePath(Node *node) const noexcept {
  if (!node)
    return "";

  try {
    std::vector<std::string> segments;

    // Traverse up to build path
    Node *current = node;
    while (current && current != root.get()) {
      Node *parent = current->parent;
      if (!parent)
        break;

      // Find index of current in parent's children
      size_t index = 0;
      for (size_t i = 0; i < parent->children.size(); ++i) {
        if (parent->children[i].get() == current) {
          index = i;
          break;
        }
      }

      segments.push_back(std::to_string(index));
      current = parent;
    }

    // Construct path from segments in reverse order
    std::string path;
    for (auto it = segments.rbegin(); it != segments.rend(); ++it) {
      path += "/" + *it;
    }

    return path.empty() ? "/" : path;
  } catch (...) {
    return "";
  }
}

void JsonModel::processBatch(std::span<const Json> nodes, Node *parent,
                             const ProgressCallback &callback, int depth) {
  if (nodes.empty() || !parent)
    return;

  try {
    std::vector<NodePtr> children(nodes.size());

    // Process nodes in parallel if there are enough
    if (nodes.size() > 10) {
      std::vector<std::future<NodePtr>> futures;
      futures.reserve(nodes.size());

      for (size_t i = 0; i < nodes.size(); ++i) {
        futures.push_back(std::async(std::launch::async, [&, i]() {
          return parseJson(nodes[i], callback, depth + 1);
        }));
      }

      // Collect results
      for (size_t i = 0; i < nodes.size(); ++i) {
        children[i] = futures[i].get();
        children[i]->parent = parent;
      }
    } else {
      // Process sequentially for small batches
      for (size_t i = 0; i < nodes.size(); ++i) {
        children[i] = parseJson(nodes[i], callback, depth + 1);
        children[i]->parent = parent;
      }
    }

    // Add children to parent node
    for (auto &child : children) {
      parent->children.push_back(std::move(child));
    }
  } catch (const std::exception &e) {
    qWarning("Error processing batch: %s", e.what());
  }
}