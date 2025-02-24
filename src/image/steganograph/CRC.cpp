#include "CRC.hpp"
#include "utils/ThreadPool.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <future>
#include <immintrin.h>
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

CRCCalculator::CRCConfig CRCCalculator::s_config; // 修正类型名
thread_local std::array<uint8_t, 4096> CRCCalculator::s_buffer;

uint32_t CRCCalculator::calculate_with_progress(
    const uint8_t *data, size_t length,
    std::function<void(float)> progress_callback) {
  const size_t chunk_size = s_config.chunk_size;
  size_t processed = 0;
  uint32_t crc = 0xFFFFFFFF;

  while (processed < length) {
    size_t current_chunk = std::min(chunk_size, length - processed);

    if (s_config.opt_level == CRCConfig::OptimizationLevel::Maximum) {
      memcpy(s_buffer.data(), data + processed,
             current_chunk); // 直接使用memcpy
      crc = fast_crc32(s_buffer.data(), current_chunk, crc);
    } else {
      crc = fast_crc32(data + processed, current_chunk, crc);
    }

    processed += current_chunk;

    if (progress_callback) {
      progress_callback(static_cast<float>(processed) / length);
    }
  }

  return ~crc;
}

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

namespace crc_utils {
BruteForceResult enhanced_brute_force(const IHDRData &original_ihdr,
                                      uint32_t target_crc,
                                      const BruteForceConfig &config) {

  BruteForceResult result;
  auto start_time = std::chrono::high_resolution_clock::now();

  std::atomic<bool> found(false);
  std::atomic<uint32_t> iteration_count(0);

  // 使用DynamicThreadPool替换ThreadPool
  DynamicThreadPool pool(config.num_threads);
  std::vector<std::future<void>> futures;

  // 任务分块
  int width_range = config.max_width - config.min_width + 1;
  int block_size = std::max(1, width_range / (config.num_threads * 4));

  // 使用新的线程池API
  for (int base_w = config.min_width; base_w <= config.max_width;
       base_w += block_size) {
    futures.push_back(pool.enqueue(
        [&](int start_w) {
          IHDRData modified = original_ihdr;
          int end_w = std::min(start_w + block_size, config.max_width + 1);

          for (int w = start_w; w < end_w && !found; ++w) {
            modified.width = byteswap_uint32_simd(static_cast<uint32_t>(w));

            for (int h = config.min_height; h <= config.max_height && !found;
                 ++h) {
              modified.height = byteswap_uint32_simd(static_cast<uint32_t>(h));

              iteration_count.fetch_add(1, std::memory_order_relaxed);

              if (crc_inline::calculate_ihdr_crc(modified) == target_crc) {
                result.width = w;
                result.height = h;
                result.success = true;
                found = true;
                break;
              }

              // 进度报告
              if (config.progress_callback && (iteration_count % 1000 == 0)) {
                float progress =
                    static_cast<float>(iteration_count) /
                    (width_range * (config.max_height - config.min_height + 1));
                config.progress_callback(progress);
              }
            }
          }
        },
        base_w));
  }

  // 等待所有任务完成
  for (auto &future : futures) {
    future.wait();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  result.elapsed_time =
      std::chrono::duration<double>(end_time - start_time).count();
  result.iterations = iteration_count;

  return result;
}
} // namespace crc_utils