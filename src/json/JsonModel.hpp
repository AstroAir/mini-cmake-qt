#pragma once

#include <QAbstractItemModel>

#include <nlohmann/json.hpp>
#include <shared_mutex>

using Json = nlohmann::json;

class JsonModel : public QAbstractItemModel {
  struct Node {
    // 存储指向原始 JSON 的指针或者新增节点名
    std::variant<Json *, QString> data;
    Node *parent = nullptr;
    std::vector<std::unique_ptr<Node>> children;

    [[nodiscard]] QVariant display() const noexcept;
  };

  std::unique_ptr<Node> root;
  Json jsonData;
  mutable std::shared_mutex dataMutex;

  // 撤销/重做栈
  struct Command {
    QModelIndex index;
    QVariant oldValue;
    QVariant newValue;
  };
  std::vector<Command> undoStack;
  std::vector<Command> redoStack;
  const size_t maxUndoSteps = 100;

public:
  explicit JsonModel(QObject *parent = nullptr);

  // 供外界获取完整 JSON
  [[nodiscard]] const Json &getJson() const noexcept;

  // 创建并返回索引
  QModelIndex index(int row, int column,
                    const QModelIndex &parent) const override;

  // 返回父索引
  QModelIndex parent(const QModelIndex &index) const override;

  // 行数
  int rowCount(const QModelIndex &parent) const override;

  // 列数
  int columnCount(const QModelIndex &) const override;

  // 获取不可编辑的显示数据
  QVariant data(const QModelIndex &index, int role) const override;

  // 设置数据（仅第一列可编辑）
  bool setData(const QModelIndex &index, const QVariant &value,
               int role) override;

  // 添加撤销命令
  void addUndoCommand(const Command &cmd);

  // 撤销操作
  bool undo();

  // 重做操作
  bool redo();

  // 设置列为可编辑
  Qt::ItemFlags flags(const QModelIndex &index) const override;

  // 加载并解析新的 JSON 数据
  void load(const Json &newData);

  // JSON 数据验证
  void validateJson(const Json &j);

  // 获取 JSON 最大嵌套深度
  size_t getJsonDepth(const Json &j) const;

  // 获取 JSON 节点总数
  size_t getJsonNodeCount(const Json &j) const;

private:
  // 递归解析 JSON，构建树状结构
  std::unique_ptr<Node> parseJson(const Json &j);

  // 查找节点在同级中的行号
  [[nodiscard]] int findRow(Node *node) const noexcept;
  // 根据节点的实际类型返回字符串
  [[nodiscard]] QString typeString(Node *node) const noexcept;
};