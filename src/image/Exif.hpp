#ifndef EXIF_H
#define EXIF_H

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

/**
 * @struct ExifValue
 * @brief Structure representing an EXIF value.
 *
 * This structure holds the tag, tag name, and value of an EXIF entry.
 */
struct ExifValue {
  /**
   * @brief Type alias for the value type.
   *
   * The value can be one of several types, including integers, doubles,
   * strings, and tuples representing GPS coordinates or three-component data.
   */
  using ValueType =
      std::variant<uint32_t, int32_t, double, std::string,
                   std::tuple<double, double>,        // GPS coordinates
                   std::tuple<double, double, double> // Three-component data
                   >;

  uint16_t tag;         ///< The EXIF tag.
  std::string tag_name; ///< The name of the EXIF tag.
  ValueType value;      ///< The value of the EXIF tag.
};

/**
 * @class ExifParser
 * @brief Class for parsing EXIF data from image files.
 *
 * This class provides functionality to parse EXIF data from image files.
 */
class ExifParser {
public:
  /**
   * @brief Constructor for ExifParser.
   * @param file_path The path to the image file.
   */
  explicit ExifParser(const std::string &file_path);

  /**
   * @brief Parses the EXIF data from the image file.
   * @return A vector of ExifValue structures representing the parsed EXIF data.
   */
  std::vector<ExifValue> parse();

private:
  std::span<const uint8_t> data_; ///< Span representing the image data.

  /**
   * @brief Reads a value from the data at the specified offset.
   * @tparam T The type of the value to read.
   * @param offset The offset in the data to read from.
   * @param order The byte order (endianness) of the value.
   * @return The value read from the data.
   */
  template <typename T> T read_value(size_t offset, std::endian order) const;

  /**
   * @brief Processes an IFD (Image File Directory) and extracts EXIF values.
   * @param offset The offset of the IFD in the data.
   * @param order The byte order (endianness) of the IFD.
   * @param results The vector to store the extracted EXIF values.
   * @param root_ifd Flag indicating if this is the root IFD.
   */
  void process_ifd(uint32_t offset, std::endian order,
                   std::vector<ExifValue> &results, bool root_ifd);

  /**
   * @brief Parses an EXIF entry.
   * @param tag The EXIF tag.
   * @param format The format of the EXIF entry.
   * @param components The number of components in the EXIF entry.
   * @param offset The offset of the EXIF entry in the data.
   * @param order The byte order (endianness) of the EXIF entry.
   * @return An optional ExifValue representing the parsed EXIF entry.
   */
  std::optional<ExifValue> parse_entry(uint16_t tag, uint16_t format,
                                       uint32_t components, uint32_t offset,
                                       std::endian order) const;

  /**
   * @brief Gets the size of a type based on the EXIF format.
   * @param format The EXIF format.
   * @return The size of the type in bytes.
   */
  static size_t get_type_size(uint16_t format);
};

#endif // EXIF_H