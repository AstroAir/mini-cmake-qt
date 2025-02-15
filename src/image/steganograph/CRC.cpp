#include "CRC.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <immintrin.h> // SSE/AVX 支持
#include <thread>
#include <vector>

// SIMD优化的字节序转换
inline uint32_t byteswap_uint32_simd(uint32_t value) {
#if defined(__AVX2__)
  return _bswap32(value);
#else
  return ((value & 0xFF000000) >> 24) | ((value & 0x00FF0000) >> 8) |
         ((value & 0x0000FF00) << 8) | ((value & 0x000000FF) << 24);
#endif
}

uint32_t CRCCalculator::fast_crc32(const uint8_t *data, size_t length,
                                   uint32_t crc) {
  static constexpr auto crc_tables = generate_crc_table();
  crc = ~crc;

// SIMD优化：每次处理16字节
#if defined(__AVX2__)
  while (length >= 16) {
    __m128i data_v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data));
    crc = process_block_simd(data_v, crc);
    data += 16;
    length -= 16;
  }
#endif

  // 处理剩余字节
  while (length >= 4) {
    crc ^= *reinterpret_cast<const uint32_t *>(data);
    crc = crc_tables[3][crc & 0xFF] ^ crc_tables[2][(crc >> 8) & 0xFF] ^
          crc_tables[1][(crc >> 16) & 0xFF] ^ crc_tables[0][crc >> 24];
    data += 4;
    length -= 4;
  }

  while (length--) {
    crc = (crc >> 8) ^ crc_tables[0][(crc & 0xFF) ^ *data++];
  }

  return ~crc;
}

#if defined(__AVX2__)
uint32_t CRCCalculator::process_block_simd(__m128i data, uint32_t crc) {
  static constexpr auto crc_tables = generate_crc_table();
  uint32_t tmp[4];
  _mm_storeu_si128(reinterpret_cast<__m128i *>(tmp), data);

  for (int i = 0; i < 4; ++i) {
    crc ^= tmp[i];
    crc = crc_tables[3][crc & 0xFF] ^ crc_tables[2][(crc >> 8) & 0xFF] ^
          crc_tables[1][(crc >> 16) & 0xFF] ^ crc_tables[0][crc >> 24];
  }
  return crc;
}
#endif

// 优化的并行爆破实现
std::pair<int, int>
brute_force_crc(const IHDRData &original_ihdr, uint32_t target_crc,
                int max_dim = 5000,
                int num_threads = std::thread::hardware_concurrency()) {
  std::atomic<bool> found(false);
  std::atomic<int> result_width{-1};
  std::atomic<int> result_height{-1};

  // 计算任务分块大小
  const int block_size = 64; // 缓存行大小相关
  const int num_blocks = (max_dim + block_size - 1) / block_size;
  std::atomic<int> next_block(0);

  // 工作线程函数
  auto worker = [&]() {
    IHDRData modified = original_ihdr;
    const uint32_t ihdr_crc =
        CRCCalculator::fast_crc32(reinterpret_cast<const uint8_t *>("IHDR"), 4);

    while (!found) {
      int block_id = next_block.fetch_add(1, std::memory_order_relaxed);
      if (block_id >= num_blocks)
        break;

      int start_w = block_id * block_size + 1;
      int end_w = std::min(start_w + block_size, max_dim + 1);

      for (int w = start_w; w < end_w && !found; ++w) {
        modified.width = byteswap_uint32_simd(static_cast<uint32_t>(w));

        for (int h = 1; h <= max_dim && !found; ++h) {
          modified.height = byteswap_uint32_simd(static_cast<uint32_t>(h));

          uint32_t crc = CRCCalculator::fast_crc32(
              reinterpret_cast<const uint8_t *>(&modified), sizeof(IHDRData),
              ihdr_crc);

          result_width.store(w);
          result_height.store(h);
          found = true;
          break;
          break;
        }
      }
    }
  };

  // 启动工作线程
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker);
  }

  // 等待所有线程完成
  for (auto &thread : threads) {
    thread.join();
  }
  return std::make_pair(result_width.load(), result_height.load());
}