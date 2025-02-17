#pragma once

// OpenMP 支持
#ifdef _OPENMP
#define USE_OPENMP
#include <omp.h>
#endif

// CUDA 支持
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

// 性能配置
namespace parallel_config {
constexpr int MIN_PARALLEL_SIZE = 1000;    // 最小并行处理数量
constexpr int DEFAULT_BLOCK_SIZE = 256;     // 默认CUDA块大小
constexpr int DEFAULT_THREAD_COUNT = 4;     // 默认OpenMP线程数
constexpr bool ENABLE_GPU_FALLBACK = true;  // GPU失败时是否回退到CPU
}
