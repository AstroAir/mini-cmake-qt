#include "JsonModel.hpp"
#include <mutex>

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

QVariant JsonModel::Node::display() const noexcept {
  // 此处可安全使用 std::visit，因为 data 是 std::variant<Json*, QString>
  return std::visit(overloaded{[](Json *j) -> QVariant {
                                 if (j->is_string())
                                   return QString::fromStdString(
                                       j->get<std::string>());
                                 if (j->is_number())
                                   return j->get<double>();
                                 if (j->is_boolean())
                                   return j->get<bool>();
                                 return QVariant();
                               },
                               [](const QString &s) { return QVariant(s); }},
                    data);
}

JsonModel::JsonModel(QObject *parent) : QAbstractItemModel(parent) {}

[[nodiscard]] const Json &JsonModel::getJson() const noexcept {
  return jsonData;
}

// 创建并返回索引
QModelIndex JsonModel::index(int row, int column,
                             const QModelIndex &parent) const {
  if (!hasIndex(row, column, parent))
    return {};

  Node *parentNode = parent.isValid()
                         ? static_cast<Node *>(parent.internalPointer())
                         : root.get();
  return createIndex(row, column, parentNode->children[row].get());
}

// 返回父索引
QModelIndex JsonModel::parent(const QModelIndex &index) const {
  if (!index.isValid())
    return {};

  Node *childNode = static_cast<Node *>(index.internalPointer());
  Node *parentNode = childNode->parent;
  if (parentNode == root.get())
    return {};

  // 生成父节点对应的 QModelIndex
  int row = findRow(parentNode);
  return createIndex(row, 0, parentNode);
}

// 行数
int JsonModel::rowCount(const QModelIndex &parent) const {
  std::shared_lock lock(dataMutex);
  Node *parentNode = parent.isValid()
                         ? static_cast<Node *>(parent.internalPointer())
                         : root.get();
  return static_cast<int>(parentNode->children.size());
}

// 列数
int JsonModel::columnCount(const QModelIndex &) const { return 2; }

// 获取不可编辑的显示数据
QVariant JsonModel::data(const QModelIndex &index, int role) const {
  if (!index.isValid() || role != Qt::DisplayRole)
    return {};
  Node *node = static_cast<Node *>(index.internalPointer());
  // 第一列显示实际内容，第二列显示类型
  return index.column() == 0 ? node->display() : typeString(node);
}

// 设置数据（仅第一列可编辑）
bool JsonModel::setData(const QModelIndex &index, const QVariant &value,
                        int role) {
  if (role != Qt::EditRole || !index.isValid())
    return false;

  std::unique_lock lock(dataMutex);
  Node *node = static_cast<Node *>(index.internalPointer());
  if (auto j = std::get_if<Json *>(&node->data)) {
    // 记录旧值
    QVariant oldValue = node->display();

    // 更新值
    if (value.canConvert<QString>()) {
      (**j) = value.toString().toStdString();
    } else if (value.canConvert<double>()) {
      (**j) = value.toDouble();
    } else if (value.canConvert<bool>()) {
      (**j) = value.toBool();
    }

    // 记录操作
    Command cmd{index, oldValue, value};
    addUndoCommand(cmd);

    emit dataChanged(index, index);
    return true;
  }
  return false;
}

// 添加撤销命令
void JsonModel::addUndoCommand(const Command &cmd) {
  undoStack.push_back(cmd);
  if (undoStack.size() > maxUndoSteps) {
    undoStack.erase(undoStack.begin());
  }
  redoStack.clear(); // 新操作会清除重做栈
}

// 撤销操作
bool JsonModel::undo() {
  if (undoStack.empty())
    return false;

  std::unique_lock lock(dataMutex);
  const Command &cmd = undoStack.back();
  Node *node = static_cast<Node *>(cmd.index.internalPointer());
  if (auto j = std::get_if<Json *>(&node->data)) {
    // 记录当前值为重做命令
    Command redoCmd{cmd.index, node->display(), cmd.oldValue};
    redoStack.push_back(redoCmd);

    // 恢复旧值
    if (cmd.oldValue.canConvert<QString>()) {
      (**j) = cmd.oldValue.toString().toStdString();
    } else if (cmd.oldValue.canConvert<double>()) {
      (**j) = cmd.oldValue.toDouble();
    } else if (cmd.oldValue.canConvert<bool>()) {
      (**j) = cmd.oldValue.toBool();
    }

    undoStack.pop_back();
    emit dataChanged(cmd.index, cmd.index);
    return true;
  }
  return false;
}

// 重做操作
bool JsonModel::redo() {
  if (redoStack.empty())
    return false;

  std::unique_lock lock(dataMutex);
  const Command &cmd = redoStack.back();
  Node *node = static_cast<Node *>(cmd.index.internalPointer());
  if (auto j = std::get_if<Json *>(&node->data)) {
    // 记录当前值为撤销命令
    Command undoCmd{cmd.index, node->display(), cmd.newValue};
    undoStack.push_back(undoCmd);

    // 应用新值
    if (cmd.newValue.canConvert<QString>()) {
      (**j) = cmd.newValue.toString().toStdString();
    } else if (cmd.newValue.canConvert<double>()) {
      (**j) = cmd.newValue.toDouble();
    } else if (cmd.newValue.canConvert<bool>()) {
      (**j) = cmd.newValue.toBool();
    }

    redoStack.pop_back();
    emit dataChanged(cmd.index, cmd.index);
    return true;
  }
  return false;
}

// 设置列为可编辑
Qt::ItemFlags JsonModel::flags(const QModelIndex &index) const {
  auto baseFlags = QAbstractItemModel::flags(index);
  if (index.column() == 0)
    return baseFlags | Qt::ItemIsEditable;
  return baseFlags;
}

// 加载并解析新的 JSON 数据
void JsonModel::load(const Json &newData) {
  beginResetModel();
  {
    std::unique_lock lock(dataMutex);
    try {
      // 验证 JSON 数据
      validateJson(newData);
      jsonData = newData;
      root = parseJson(jsonData);
    } catch (const std::exception &e) {
      endResetModel();
      throw; // 重新抛出异常
    }
  }
  endResetModel();
}

// JSON 数据验证
void JsonModel::validateJson(const Json &j) {
  if (j.is_discarded()) {
    throw std::runtime_error("Invalid JSON format");
  }
  // 检查最大嵌套深度
  constexpr int maxDepth = 100;
  if (getJsonDepth(j) > maxDepth) {
    throw std::runtime_error("JSON nesting too deep (max " +
                             std::to_string(maxDepth) + " levels)");
  }
  // 检查最大节点数
  constexpr size_t maxNodes = 1000000;
  if (getJsonNodeCount(j) > maxNodes) {
    throw std::runtime_error("JSON too large (max " + std::to_string(maxNodes) +
                             " nodes)");
  }
}

// 获取 JSON 最大嵌套深度
size_t JsonModel::getJsonDepth(const Json &j) const {
  if (j.is_object() || j.is_array()) {
    size_t max = 0;
    for (const auto &item : j) {
      max = std::max(max, getJsonDepth(item));
    }
    return max + 1;
  }
  return 1;
}

// 获取 JSON 节点总数
size_t JsonModel::getJsonNodeCount(const Json &j) const {
  if (j.is_object()) {
    size_t count = 1;
    for (const auto &[key, value] : j.items()) {
      count += getJsonNodeCount(value);
    }
    return count;
  }
  if (j.is_array()) {
    size_t count = 1;
    for (const auto &item : j) {
      count += getJsonNodeCount(item);
    }
    return count;
  }
  return 1;
}

std::unique_ptr<JsonModel::Node> JsonModel::parseJson(const Json &j) {
  auto node = std::make_unique<Node>();
  if (j.is_object()) {
    for (auto &[key, value] : j.items()) {
      auto child = parseJson(value);
      child->parent = node.get();
      node->children.push_back(std::move(child));
    }
  } else if (j.is_array()) {
    for (auto &element : j) {
      auto child = parseJson(element);
      child->parent = node.get();
      node->children.push_back(std::move(child));
    }
  } else {
    node->data = const_cast<Json *>(&j);
  }
  return node;
}

// 查找节点在同级中的行号
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

// 根据节点的实际类型返回字符串
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