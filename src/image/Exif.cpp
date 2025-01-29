#include "Exif.hpp"

#include <bit>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <span>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace std;

// Custom byteswap implementation if std::byteswap is not available
template <typename T> T byteswap(T value) {
  static_assert(std::is_integral_v<T>, "Integral required.");
  T result;
  auto *src = reinterpret_cast<char *>(&value);
  auto *dst = reinterpret_cast<char *>(&result);
  for (size_t i = 0; i < sizeof(T); ++i)
    dst[i] = src[sizeof(T) - 1 - i];
  return result;
}

ExifParser::ExifParser(const string &file_path) {
  spdlog::info("Opening file: {}", file_path);
  ifstream file(file_path, ios::binary);
  if (!file) {
    spdlog::error("Failed to open file: {}", file_path);
    throw runtime_error("Failed to open file");
  }

  // 直接搜索EXIF标记，避免依赖OpenCV的EXIF处理
  vector<char> buffer(1024 * 1024);
  file.read(buffer.data(), buffer.size());
  const char exif_signature[] = "Exif\0\0";
  auto exif_start = search(buffer.begin(), buffer.end(), begin(exif_signature),
                           end(exif_signature));

  if (exif_start == buffer.end()) {
    spdlog::error("EXIF signature not found in file: {}", file_path);
    throw runtime_error("EXIF signature not found");
  }

  data_ = span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(exif_start.base() + 6),
      reinterpret_cast<const uint8_t *>(buffer.data() + buffer.size()));
  spdlog::info("EXIF data found and loaded from file: {}", file_path);
}

vector<ExifValue> ExifParser::parse() {
  vector<ExifValue> results;

  if (data_.size() < 8) {
    spdlog::error("Invalid EXIF header, data size too small");
    throw runtime_error("Invalid EXIF header");
  }

  const auto byte_order = data_[0] == 'I' ? endian::little : endian::big;
  spdlog::info("Byte order detected: {}",
               byte_order == endian::little ? "Little Endian" : "Big Endian");

  constexpr array<uint8_t, 2> fixed_header{0x2a, 0x00};
  if (!equal(fixed_header.begin(), fixed_header.end(), data_.begin() + 2)) {
    spdlog::error("Invalid TIFF header");
    throw runtime_error("Invalid TIFF header");
  }

  auto ifd_offset = read_value<uint32_t>(4, byte_order);
  spdlog::info("IFD offset: {}", ifd_offset);
  process_ifd(ifd_offset, byte_order, results, true);

  return results;
}

template <typename T>
T ExifParser::read_value(size_t offset, endian order) const {
  if (offset + sizeof(T) > data_.size()) {
    spdlog::error("Read beyond EXIF data boundary at offset: {}", offset);
    throw out_of_range("Read beyond EXIF data boundary");
  }

  T value;
  memcpy(&value, data_.data() + offset, sizeof(T));
  return order == endian::native ? value : byteswap(value);
}

void ExifParser::process_ifd(uint32_t offset, endian order,
                             vector<ExifValue> &results, bool root_ifd) {
  const auto entry_count = read_value<uint16_t>(offset, order);
  spdlog::info("Processing IFD with {} entries at offset: {}", entry_count,
               offset);

  constexpr size_t entry_size = 12;
  const auto max_offset = offset + 2 + entry_count * entry_size;
  if (max_offset > data_.size()) {
    spdlog::error("IFD structure exceeds data boundary");
    throw out_of_range("IFD structure exceeds data boundary");
  }

  for (uint16_t i = 0; i < entry_count; ++i) {
    const auto entry_offset = offset + 2 + i * entry_size;
    auto entry = data_.subspan(entry_offset, entry_size);

    const auto tag = read_value<uint16_t>(0, order);
    const auto format = read_value<uint16_t>(2, order);
    const auto components = read_value<uint32_t>(4, order);
    const auto value_offset = read_value<uint32_t>(8, order);

    spdlog::info("Parsing entry: tag=0x{:04x}, format={}, components={}, "
                 "value_offset={}",
                 tag, format, components, value_offset);

    try {
      if (auto exif_value =
              parse_entry(tag, format, components, value_offset, order)) {
        results.push_back(std::move(*exif_value));

        // 处理子IFD
        if (root_ifd && tag == 0x8769) { // Exif subIFD
          spdlog::info("Processing Exif subIFD at offset: {}", value_offset);
          process_ifd(value_offset, order, results, false);
        }
      }
    } catch (const exception &e) {
      // 记录解析错误但继续处理其他条目
      spdlog::error("Error parsing tag 0x{:04x}: {}", tag, e.what());
    }
  }
}

optional<ExifValue> ExifParser::parse_entry(uint16_t tag, uint16_t format,
                                            uint32_t components,
                                            uint32_t offset,
                                            endian order) const {
  static const unordered_map<uint16_t, string> tag_names{
      {0x010e, "ImageDescription"},
      {0x0132, "DateTime"},
      {0x8769, "ExifIFD"},
      {0x8825, "GPSInfo"},
      {0x010f, "Make"},
      {0x0110, "Model"},
      {0x0112, "Orientation"},
      {0x011a, "XResolution"},
      {0x011b, "YResolution"},
      {0x0128, "ResolutionUnit"},
      {0x0131, "Software"},
      {0x013b, "Artist"},
      {0x0213, "YCbCrPositioning"},
      {0x8298, "Copyright"},
      {0x9003, "DateTimeOriginal"},
      {0x9004, "DateTimeDigitized"},
      {0x9201, "ShutterSpeedValue"},
      {0x9202, "ApertureValue"},
      {0x9204, "ExposureBiasValue"},
      {0x9205, "MaxApertureValue"},
      {0x9207, "MeteringMode"},
      {0x9208, "LightSource"},
      {0x9209, "Flash"},
      {0x920a, "FocalLength"},
      {0x927c, "MakerNote"},
      {0x9286, "UserComment"},
      {0xa002, "PixelXDimension"},
      {0xa003, "PixelYDimension"},
      {0xa005, "InteroperabilityIFD"},
      {0xa20e, "FocalPlaneXResolution"},
      {0xa20f, "FocalPlaneYResolution"},
      {0xa210, "FocalPlaneResolutionUnit"},
      {0xa217, "SensingMethod"},
      {0xa300, "FileSource"},
      {0xa301, "SceneType"},
      {0xa302, "CFAPattern"},
      {0xa401, "CustomRendered"},
      {0xa402, "ExposureMode"},
      {0xa403, "WhiteBalance"},
      {0xa404, "DigitalZoomRatio"},
      {0xa405, "FocalLengthIn35mmFilm"},
      {0xa406, "SceneCaptureType"},
      {0xa407, "GainControl"},
      {0xa408, "Contrast"},
      {0xa409, "Saturation"},
      {0xa40a, "Sharpness"},
      {0xa40c, "SubjectDistanceRange"},
      {0xa420, "ImageUniqueID"},
  };

  const size_t type_size = get_type_size(format);
  const size_t total_size = components * type_size;

  // 处理内联存储的值（<=4字节）
  const bool value_inline = total_size <= 4;
  const auto value_ptr = value_inline ? data_.data() + offset + 8
                                      : // 字段中的value字段位置
                             data_.data() + offset;

  auto read_data = [&](auto convert_fn) {
    vector<decay_t<decltype(convert_fn(0))>> values(components);
    for (uint32_t i = 0; i < components; ++i) {
      const auto byte_offset = i * type_size;
      if (offset + byte_offset + type_size > data_.size())
        throw out_of_range("Data field out of bounds");

      values[i] = convert_fn(
          read_value<decltype(convert_fn(0))>(offset + byte_offset, order));
    }
    return values;
  };

  ExifValue result{tag, tag_names.count(tag) ? tag_names.at(tag) : "Unknown"};

  switch (format) {
  case 1: { // BYTE
    auto values = read_data([](uint8_t v) { return v; });
    if (!values.empty()) {
      result.value = static_cast<uint32_t>(values[0]);
    }
    break;
  }
  case 2: { // ASCII
    string str(value_ptr, value_ptr + components);
    if (auto pos = str.find('\0'); pos != string::npos)
      str.resize(pos);
    result.value = str;
    break;
  }
  case 3: { // SHORT
    auto values = read_data([](uint16_t v) { return v; });
    if (tag == 0x8827) { // ISO感光度
      result.value = values.front();
    }
    break;
  }
  case 4: { // LONG
    auto values = read_data([](uint32_t v) { return v; });
    // 处理GPS坐标等
    break;
  }
  case 5: {                                 // RATIONAL
    if (components == 2 && tag == 0x9202) { // 纬度
      auto num = read_value<uint32_t>(offset, order);
      auto den = read_value<uint32_t>(offset + 4, order);
      result.value = make_tuple(
          static_cast<double>(num) / den,
          static_cast<double>(read_value<uint32_t>(offset + 8, order)) /
              read_value<uint32_t>(offset + 12, order));
    }
    break;
  }
  case 10: { // SRATIONAL
    // 处理有符号分数
    break;
  }
  default:
    spdlog::warn("Unsupported format: {}", format);
    return nullopt; // 忽略不支持的类型
  }
  return result;
}

size_t ExifParser::get_type_size(uint16_t format) {
  static const array<size_t, 12> sizes{0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4};
  return format < sizes.size() ? sizes[format] : 0;
}