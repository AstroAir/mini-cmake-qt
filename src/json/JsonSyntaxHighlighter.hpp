#pragma once

#include <QSyntaxHighlighter>
#include <QRegularExpression>
#include <QTextCharFormat>

/**
 * @class JsonSyntaxHighlighter
 * @brief JSON语法高亮器
 */
class JsonSyntaxHighlighter : public QSyntaxHighlighter {
    Q_OBJECT

public:
    explicit JsonSyntaxHighlighter(QObject *parent = nullptr);

protected:
    void highlightBlock(const QString &text) override;

private:
    struct HighlightingRule {
        QRegularExpression pattern;
        QTextCharFormat format;
    };
    std::vector<HighlightingRule> highlightingRules;

    QTextCharFormat keywordFormat;     // 关键字格式
    QTextCharFormat valueFormat;       // 值格式
    QTextCharFormat numberFormat;      // 数字格式
    QTextCharFormat stringFormat;      // 字符串格式
    QTextCharFormat colonFormat;       // 冒号格式
    QTextCharFormat braceFormat;       // 括号格式
    QTextCharFormat propertyFormat;    // 属性名格式
    QTextCharFormat commentFormat;     // 注释格式
    QTextCharFormat commaFormat;       // 逗号格式

    // 多行字符串状态追踪
    bool isInMultilineString;
    int multilineStringStartPos;

    void initializeFormats();
};
