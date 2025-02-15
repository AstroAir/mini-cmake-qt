#pragma once

#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>

constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t DEFAULT_BLOCKS_PER_CHUNK = 8192;

template <typename T>
static constexpr size_t aligned_size =
    ((sizeof(T) + alignof(std::max_align_t) - 1) &
     ~(alignof(std::max_align_t) - 1));

template <typename T>
concept ValidPoolType = requires {
  requires std::is_trivially_destructible_v<T>;
  requires sizeof(T) >= sizeof(void *);
};

// Main memory pool implementation
template <size_t BlockSize, size_t BlocksPerChunk = DEFAULT_BLOCKS_PER_CHUNK>
class MemoryPool {
public:
  static_assert(BlockSize >= sizeof(void *),
                "Block size must be at least sizeof(void*)");

  MemoryPool() noexcept = default;

  // Prevent copying
  MemoryPool(const MemoryPool &) = delete;
  MemoryPool &operator=(const MemoryPool &) = delete;

  // Allow moving
  MemoryPool(MemoryPool &&other) noexcept {
    std::scoped_lock lock(other.mutex);
    free_list = std::exchange(other.free_list, nullptr);
    chunks = std::exchange(other.chunks, nullptr);
  }

  ~MemoryPool() {
    while (chunks) {
      Chunk *next = chunks->next;
      std::destroy_at(chunks);
      ::operator delete(chunks);
      chunks = next;
    }
  }

  [[nodiscard]] void *allocate() {
    std::scoped_lock lock(mutex);

    if (!free_list) {
      allocate_chunk();
    }

    Block *block = free_list;
    free_list = block->next;
    return block;
  }

  void deallocate(void *ptr) noexcept {
    if (!ptr)
      return;

    std::scoped_lock lock(mutex);
    Block *block = static_cast<Block *>(ptr);
    block->next = free_list;
    free_list = block;
  }

  // Statistics structure
  struct Stats {
    size_t total_chunks;
    size_t total_blocks;
    size_t free_blocks;
  };

  [[nodiscard]] Stats get_stats() const noexcept {
    std::scoped_lock lock(mutex);

    Stats stats{0, 0, 0};
    Chunk *current = chunks;
    while (current) {
      ++stats.total_chunks;
      current = current->next;
    }

    stats.total_blocks = stats.total_chunks * BlocksPerChunk;

    Block *current_free = free_list;
    while (current_free) {
      ++stats.free_blocks;
      current_free = current_free->next;
    }

    return stats;
  }

#ifdef MEMORY_POOL_DEBUG
  void dump_stats() const {
    auto stats = get_stats();
    std::cout << "Memory Pool Stats:\n"
              << "Total chunks: " << stats.total_chunks << '\n'
              << "Total blocks: " << stats.total_blocks << '\n'
              << "Free blocks: " << stats.free_blocks << '\n'
              << "Block size: " << BlockSize << '\n'
              << "Blocks per chunk: " << BlocksPerChunk << '\n';
  }
#endif

private:
  struct alignas(std::max_align_t) Block {
    union {
      Block *next;
      std::byte data[BlockSize];
    };
  };

  struct Chunk {
    std::array<Block, BlocksPerChunk> blocks;
    Chunk *next = nullptr;
  };

  void allocate_chunk() {
    Chunk *new_chunk = new Chunk();

    // Initialize free list
    for (size_t i = 0; i < BlocksPerChunk - 1; ++i) {
      new_chunk->blocks[i].next = &new_chunk->blocks[i + 1];
    }
    new_chunk->blocks[BlocksPerChunk - 1].next = free_list;

    free_list = &new_chunk->blocks[0];
    new_chunk->next = chunks;
    chunks = new_chunk;
  }

  Block *free_list = nullptr;
  Chunk *chunks = nullptr;
  mutable std::mutex mutex;
};

// Pool allocator adapter for STL containers
template <typename T, size_t BlocksPerChunk = DEFAULT_BLOCKS_PER_CHUNK>
class PoolAllocator {
public:
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;

  template <typename U> struct rebind {
    using other = PoolAllocator<U, BlocksPerChunk>;
  };

  PoolAllocator() noexcept = default;

  template <typename U> PoolAllocator(const PoolAllocator<U> &) noexcept {}

  [[nodiscard]] T *allocate(size_t n) {
    // 修改为支持多个对象的分配
    if (n > BlocksPerChunk) {
      throw std::bad_alloc();
    }

    // 为n个对象分配连续内存
    T *result = nullptr;
    void *memory = nullptr;

    try {
      memory = ::operator new(n * sizeof(T));
      result = static_cast<T *>(memory);
    } catch (...) {
      throw std::bad_alloc();
    }

    return result;
  }

  void deallocate(T *p, [[maybe_unused]] size_t n) noexcept {
    // 修改为支持多个对象的释放
    if (p) {
      ::operator delete(p);
    }
  }

  template <typename U>
  bool operator==(const PoolAllocator<U> &) const noexcept {
    return true;
  }

  template <typename U>
  bool operator!=(const PoolAllocator<U> &) const noexcept {
    return false;
  }

private:
  static MemoryPool<sizeof(T), BlocksPerChunk> pool;
};

template <typename T, size_t B>
MemoryPool<sizeof(T), B> PoolAllocator<T, B>::pool;
