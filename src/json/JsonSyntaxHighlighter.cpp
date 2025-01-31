#include "JsonSyntaxHighlighter.hpp"

JsonSyntaxHighlighter::JsonSyntaxHighlighter(QObject *parent)
    : QSyntaxHighlighter(parent) {
  initializeFormats();

  // 关键字规则
  QStringList keywordPatterns = {QStringLiteral("\\btrue\\b"),
                                 QStringLiteral("\\bfalse\\b"),
                                 QStringLiteral("\\bnull\\b")};

  for (const QString &pattern : keywordPatterns) {
    HighlightingRule rule;
    rule.pattern = QRegularExpression(pattern);
    rule.format = keywordFormat;
    highlightingRules.push_back(rule);
  }

  // 字符串规则
  HighlightingRule rule;
  rule.pattern = QRegularExpression(QStringLiteral("\"[^\"]*\""));
  rule.format = stringFormat;
  highlightingRules.push_back(rule);

  // 数字规则
  rule.pattern = QRegularExpression(
      QStringLiteral("\\b-?\\d+\\.?\\d*([eE][+-]?\\d+)?\\b"));
  rule.format = numberFormat;
  highlightingRules.push_back(rule);

  // 冒号规则
  rule.pattern = QRegularExpression(QStringLiteral(":"));
  rule.format = colonFormat;
  highlightingRules.push_back(rule);

  // 括号规则
  rule.pattern = QRegularExpression(QStringLiteral("[\\[\\]{}]"));
  rule.format = braceFormat;
  highlightingRules.push_back(rule);
}

void JsonSyntaxHighlighter::highlightBlock(const QString &text) {
  // 处理多行字符串
  if (previousBlockState() == 1) {
    isInMultilineString = true;
    multilineStringStartPos = 0;
  }

  // 添加属性名匹配规则
  QRegularExpression propertyRegex("\"([^\"]+)\"\\s*:");
  QRegularExpressionMatchIterator propIterator =
      propertyRegex.globalMatch(text);
  while (propIterator.hasNext()) {
    QRegularExpressionMatch match = propIterator.next();
    setFormat(match.capturedStart(1), match.capturedLength(1), propertyFormat);
  }

  // 处理注释
  QRegularExpression commentRegex("//.*$");
  QRegularExpressionMatch commentMatch = commentRegex.match(text);
  if (commentMatch.hasMatch()) {
    setFormat(commentMatch.capturedStart(), commentMatch.capturedLength(),
              commentFormat);
  }

  // 应用其他现有规则
  for (const HighlightingRule &rule : highlightingRules) {
    QRegularExpressionMatchIterator matchIterator =
        rule.pattern.globalMatch(text);
    while (matchIterator.hasNext()) {
      QRegularExpressionMatch match = matchIterator.next();
      setFormat(match.capturedStart(), match.capturedLength(), rule.format);
    }
  }
}

void JsonSyntaxHighlighter::initializeFormats() {
  // 设置关键字格式
  keywordFormat.setForeground(QColor("#0033B3"));
  keywordFormat.setFontWeight(QFont::Bold);

  // 设置属性名格式
  propertyFormat.setForeground(QColor("#116644"));
  propertyFormat.setFontWeight(QFont::Bold);

  // 设置值格式
  valueFormat.setForeground(QColor("#067D17"));

  // 设置数字格式
  numberFormat.setForeground(QColor("#1750EB"));

  // 设置字符串格式
  stringFormat.setForeground(QColor("#067D17"));

  // 设置注释格式
  commentFormat.setForeground(QColor("#808080"));
  commentFormat.setFontItalic(true);

  // 设置冒号格式
  colonFormat.setForeground(QColor("#000000"));

  // 设置逗号格式
  commaFormat.setForeground(QColor("#000000"));

  // 设置括号格式
  braceFormat.setForeground(QColor("#000000"));
  braceFormat.setFontWeight(QFont::Bold);
}
