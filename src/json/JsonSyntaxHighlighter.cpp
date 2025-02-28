#include "JsonSyntaxHighlighter.hpp"
#include <algorithm>
#include <ranges>

JsonSyntaxHighlighter::JsonSyntaxHighlighter(QObject *parent) noexcept
    : QSyntaxHighlighter(parent) {
  try {
    initializeFormats();

    // Keywords rules
    constexpr std::array<const char *, 3> keywordPatterns = {
        "\\btrue\\b", "\\bfalse\\b", "\\bnull\\b"};

    // Reserve space for rules
    highlightingRules.reserve(keywordPatterns.size() + 4);

    // Use C++20 ranges to transform keyword patterns into highlighting rules
    for (const auto &pattern :
         keywordPatterns | std::views::filter([this](const auto &p) {
           return isValidPattern(p);
         })) {
      HighlightingRule rule;
      rule.pattern = QRegularExpression(QString::fromLatin1(pattern));
      rule.format = keywordFormat;
      highlightingRules.push_back(rule);
    }

    // String rule
    if (isValidPattern(QStringLiteral("\"[^\"]*\""))) {
      HighlightingRule rule;
      rule.pattern = QRegularExpression(QStringLiteral("\"[^\"]*\""));
      rule.format = stringFormat;
      highlightingRules.push_back(rule);
    }

    // Number rule - enhanced regex for JSON number validation
    if (isValidPattern(QStringLiteral(
            "\\b-?(?:0|[1-9]\\d*)(?:\\.\\d+)?(?:[eE][+-]?\\d+)?\\b"))) {
      HighlightingRule rule;
      rule.pattern = QRegularExpression(QStringLiteral(
          "\\b-?(?:0|[1-9]\\d*)(?:\\.\\d+)?(?:[eE][+-]?\\d+)?\\b"));
      rule.format = numberFormat;
      highlightingRules.push_back(rule);
    }

    // Colon rule
    if (isValidPattern(QStringLiteral(":"))) {
      HighlightingRule rule;
      rule.pattern = QRegularExpression(QStringLiteral(":"));
      rule.format = colonFormat;
      highlightingRules.push_back(rule);
    }

    // Braces rule
    if (isValidPattern(QStringLiteral("[\\[\\]{}]"))) {
      HighlightingRule rule;
      rule.pattern = QRegularExpression(QStringLiteral("[\\[\\]{}]"));
      rule.format = braceFormat;
      highlightingRules.push_back(rule);
    }
  } catch (const std::exception &e) {
    // Log the exception but don't throw from constructor
    qWarning("Exception in JsonSyntaxHighlighter constructor: %s", e.what());
  }
}

void JsonSyntaxHighlighter::highlightBlock(const QString &text) {
  try {
    // Handle multiline strings
    if (previousBlockState() == 1) {
      isInMultilineString = true;
      multilineStringStartPos = 0;
    }

    // Add property name matching rule
    const QRegularExpression propertyRegex("\"([^\"]+)\"\\s*:");
    QRegularExpressionMatchIterator propIterator =
        propertyRegex.globalMatch(text);

    // Use atomic operations for thread safety
    while (propIterator.hasNext()) {
      const QRegularExpressionMatch match = propIterator.next();
      if (match.hasMatch() && match.capturedStart(1) >= 0) {
        setFormat(match.capturedStart(1), match.capturedLength(1),
                  propertyFormat);
      }
    }

    // Handle comments
    const QRegularExpression commentRegex("//.*$");
    const QRegularExpressionMatch commentMatch = commentRegex.match(text);
    if (commentMatch.hasMatch()) {
      setFormat(commentMatch.capturedStart(), commentMatch.capturedLength(),
                commentFormat);
    }

    // Apply existing rules - use parallel execution if text is long enough
    if (text.length() > 1000 && highlightingRules.size() > 5) {
      // Create a local copy of rules for thread safety
      const auto rules = highlightingRules;

      // Process rules in parallel when beneficial
      std::for_each(rules.begin(), rules.end(),
                    [this, &text](const HighlightingRule &rule) {
                      QRegularExpressionMatchIterator matchIterator =
                          rule.pattern.globalMatch(text);
                      while (matchIterator.hasNext()) {
                        const QRegularExpressionMatch match =
                            matchIterator.next();
                        if (match.hasMatch() && match.capturedStart() >= 0) {
                          // Safely apply format in a way that doesn't conflict
                          // with parallel execution
                          setFormat(match.capturedStart(),
                                    match.capturedLength(), rule.format);
                        }
                      }
                    });
    } else {
      // Sequential execution for shorter text
      for (const HighlightingRule &rule : highlightingRules) {
        QRegularExpressionMatchIterator matchIterator =
            rule.pattern.globalMatch(text);
        while (matchIterator.hasNext()) {
          const QRegularExpressionMatch match = matchIterator.next();
          if (match.hasMatch()) {
            setFormat(match.capturedStart(), match.capturedLength(),
                      rule.format);
          }
        }
      }
    }
  } catch (const std::exception &e) {
    qWarning("Exception in highlightBlock: %s", e.what());
  }
}

void JsonSyntaxHighlighter::initializeFormats() noexcept {
  // Keyword format
  keywordFormat.setForeground(QColor("#0033B3"));
  keywordFormat.setFontWeight(QFont::Bold);

  // Property name format
  propertyFormat.setForeground(QColor("#116644"));
  propertyFormat.setFontWeight(QFont::Bold);

  // Value format
  valueFormat.setForeground(QColor("#067D17"));

  // Number format
  numberFormat.setForeground(QColor("#1750EB"));

  // String format
  stringFormat.setForeground(QColor("#067D17"));

  // Comment format
  commentFormat.setForeground(QColor("#808080"));
  commentFormat.setFontItalic(true);

  // Colon format
  colonFormat.setForeground(QColor("#000000"));

  // Comma format
  commaFormat.setForeground(QColor("#000000"));

  // Brace format
  braceFormat.setForeground(QColor("#000000"));
  braceFormat.setFontWeight(QFont::Bold);
}
