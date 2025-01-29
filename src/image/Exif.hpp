#ifndef EXIF_H
#define EXIF_H

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

struct ExifValue {
  using ValueType =
      std::variant<uint32_t, int32_t, double, std::string,
                   std::tuple<double, double>,        // GPS坐标
                   std::tuple<double, double, double> // 三分量数据
                   >;

  uint16_t tag;
  std::string tag_name;
  ValueType value;
};

class ExifParser {
public:
  explicit ExifParser(const std::string &file_path);
  std::vector<ExifValue> parse();

private:
  std::span<const uint8_t> data_;

  template <typename T> T read_value(size_t offset, std::endian order) const;

  void process_ifd(uint32_t offset, std::endian order,
                   std::vector<ExifValue> &results, bool root_ifd);

  std::optional<ExifValue> parse_entry(uint16_t tag, uint16_t format,
                                       uint32_t components, uint32_t offset,
                                       std::endian order) const;

  static size_t get_type_size(uint16_t format);
};

#endif // EXIF_H