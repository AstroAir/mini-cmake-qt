#pragma once

#include <QRegularExpression>
#include <QSyntaxHighlighter>
#include <QTextCharFormat>
#include <concepts>

/**
 * @class JsonSyntaxHighlighter
 * @brief A syntax highlighter for JSON data.
 *
 * This class provides syntax highlighting for JSON data in a QTextDocument.
 * It highlights keywords, values, numbers, strings, colons, braces, property
 * names, comments, and commas using different text formats.
 */
class JsonSyntaxHighlighter : public QSyntaxHighlighter {
  Q_OBJECT

public:
  /**
   * @brief Constructs a JsonSyntaxHighlighter object.
   * @param parent The parent object.
   */
  explicit JsonSyntaxHighlighter(QObject *parent = nullptr) noexcept;

  /**
   * @brief Destructor for JsonSyntaxHighlighter.
   */
  ~JsonSyntaxHighlighter() override = default;

  // Disable copying
  JsonSyntaxHighlighter(const JsonSyntaxHighlighter &) = delete;
  JsonSyntaxHighlighter &operator=(const JsonSyntaxHighlighter &) = delete;

  // Enable moving
  JsonSyntaxHighlighter(JsonSyntaxHighlighter &&) noexcept = delete;
  JsonSyntaxHighlighter &operator=(JsonSyntaxHighlighter &&) noexcept = delete;

protected:
  /**
   * @brief Highlights a block of text.
   * @param text The text to highlight.
   */
  void highlightBlock(const QString &text) override;

private:
  /**
   * @struct HighlightingRule
   * @brief A structure to define a highlighting rule.
   *
   * This structure contains a regular expression pattern and a text format
   * to apply to the text that matches the pattern.
   */
  struct HighlightingRule {
    QRegularExpression pattern; ///< The regular expression pattern.
    QTextCharFormat format;     ///< The text format to apply.
  };

  std::vector<HighlightingRule>
      highlightingRules; ///< The list of highlighting rules.

  QTextCharFormat keywordFormat;  ///< Format for JSON keywords.
  QTextCharFormat valueFormat;    ///< Format for JSON values.
  QTextCharFormat numberFormat;   ///< Format for JSON numbers.
  QTextCharFormat stringFormat;   ///< Format for JSON strings.
  QTextCharFormat colonFormat;    ///< Format for colons.
  QTextCharFormat braceFormat;    ///< Format for braces.
  QTextCharFormat propertyFormat; ///< Format for property names.
  QTextCharFormat commentFormat;  ///< Format for comments.
  QTextCharFormat commaFormat;    ///< Format for commas.

  bool isInMultilineString =
      false; ///< Flag to track if inside a multiline string.
  int multilineStringStartPos = 0; ///< Start position of the multiline string.

  /**
   * @brief Initializes the text formats for different JSON elements.
   */
  void initializeFormats() noexcept;

  /**
   * @brief Validates if a pattern is a valid regular expression.
   * @tparam T A type convertible to QString.
   * @param pattern The pattern to validate.
   * @return True if the pattern is valid, false otherwise.
   */
  template <std::convertible_to<QString> T>
  bool isValidPattern(const T &pattern) const noexcept {
    QRegularExpression re(pattern);
    return re.isValid();
  }
};