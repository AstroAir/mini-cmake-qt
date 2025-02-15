#pragma once

#include <array>
#include <cstdint>
#include <thread>
#include <utility>

/**
 * @brief Optimized IHDR block structure (16-byte aligned).
 */
struct alignas(16) IHDRData {
  uint32_t width;      ///< Image width
  uint32_t height;     ///< Image height
  uint8_t bit_depth;   ///< Bit depth
  uint8_t color_type;  ///< Color type
  uint8_t compression; ///< Compression method
  uint8_t filter;      ///< Filter method
  uint8_t interlace;   ///< Interlace method
  uint8_t padding[3];  ///< Padding to optimize memory access
};

/**
 * @brief CRC Calculator class for computing CRC32 checksums.
 */
class CRCCalculator {
public:
  /**
   * @brief Generates the CRC lookup table.
   * @return A 2D array containing the CRC lookup table.
   */
  static constexpr auto generate_crc_table() {
    std::array<std::array<uint32_t, 256>, 16> table{};
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t crc = i;
      for (int j = 0; j < 8; j++) {
        crc = (crc >> 1) ^ ((crc & 1) * 0xEDB88320);
      }
      table[0][i] = crc;
    }

    // Generate optimized table
    for (uint32_t i = 0; i < 256; i++) {
      for (uint32_t j = 1; j < 16; j++) {
        table[j][i] = (table[j - 1][i] >> 8) ^ table[0][table[j - 1][i] & 0xFF];
      }
    }
    return table;
  }

  /**
   * @brief Computes the CRC32 checksum for the given data.
   * @param data Pointer to the data.
   * @param length Length of the data in bytes.
   * @param crc Initial CRC value (default is 0).
   * @return The computed CRC32 checksum.
   */
  static uint32_t fast_crc32(const uint8_t *data, size_t length,
                             uint32_t crc = 0);

private:
#if defined(__AVX2__)
  /**
   * @brief Processes a block of data using SIMD optimization.
   * @param data SIMD register containing the data.
   * @param crc Current CRC value.
   * @return The updated CRC value.
   */
  static uint32_t process_block_simd(__m128i data, uint32_t crc);
#endif
};

namespace crc_utils {

/**
 * @brief Performs a byte swap on a 32-bit unsigned integer using SIMD
 * optimization.
 * @param value The 32-bit unsigned integer to byte swap.
 * @return The byte-swapped value.
 */
uint32_t byteswap_uint32_simd(uint32_t value);

/**
 * @brief Performs a brute-force search to find the width and height that match
 * the target CRC.
 * @param original_ihdr The original IHDR data.
 * @param target_crc The target CRC value.
 * @param max_dim The maximum dimension to search (default is 5000).
 * @param num_threads The number of threads to use (default is hardware
 * concurrency).
 * @return A pair containing the found width and height.
 */
std::pair<int, int>
brute_force_crc(const IHDRData &original_ihdr, uint32_t target_crc,
                int max_dim = 5000,
                int num_threads = std::thread::hardware_concurrency());

/**
 * @brief Exception class for CRC validation errors.
 */
class CRCException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

namespace constants {
constexpr uint32_t PNG_CRC_POLYNOMIAL = 0xEDB88320; ///< CRC polynomial for PNG
constexpr size_t CRC_TABLE_SIZE = 256;              ///< Size of the CRC table
constexpr size_t CRC_TABLE_ROWS = 16;  ///< Number of rows in the CRC table
constexpr int DEFAULT_BLOCK_SIZE = 64; ///< Default block size for processing
} // namespace constants

namespace detail {
/**
 * @brief Checks if a pointer is aligned to a specified alignment.
 * @param ptr The pointer to check.
 * @param alignment The alignment to check against.
 * @return True if the pointer is aligned, false otherwise.
 */
inline bool is_aligned(const void *ptr, size_t alignment) {
  return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

/**
 * @brief Checks if the system supports AVX2 instructions.
 * @return True if AVX2 is supported, false otherwise.
 */
inline bool has_avx2_support() {
#if defined(__AVX2__)
  return true;
#else
  return false;
#endif
}
} // namespace detail

} // namespace crc_utils

namespace crc_inline {
/**
 * @brief Computes the CRC32 checksum for the given data (inline version).
 * @param data Pointer to the data.
 * @param length Length of the data in bytes.
 * @return The computed CRC32 checksum.
 */
inline uint32_t quick_crc32(const void *data, size_t length) {
  return CRCCalculator::fast_crc32(reinterpret_cast<const uint8_t *>(data),
                                   length);
}

/**
 * @brief Computes the CRC32 checksum for an IHDR block.
 * @param ihdr The IHDR data.
 * @return The computed CRC32 checksum.
 */
inline uint32_t calculate_ihdr_crc(const IHDRData &ihdr) {
  static constexpr uint32_t IHDR_MAGIC = 0x49484452; // "IHDR"
  uint32_t crc = CRCCalculator::fast_crc32(
      reinterpret_cast<const uint8_t *>(&IHDR_MAGIC), sizeof(IHDR_MAGIC));
  return CRCCalculator::fast_crc32(reinterpret_cast<const uint8_t *>(&ihdr),
                                   sizeof(IHDRData), crc);
}
} // namespace crc_inline