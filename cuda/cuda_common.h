/*
 * @Description: some common functions used in CUDA kernels
 *              most are copied from tensorflow source code
 * @Author: Tianling Lyu
 * @Date: 2019-11-13 14:24:28
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-01-05 14:08:54
 */

#ifndef _CT_RECON_EXT_CUDA_COM_H_
#define _CT_RECON_EXT_CUDA_COM_H_

//#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#define FAN_BP_PIX_DRIVEN_KERNEL 1

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// simplified version of the struct in tensorflow
struct CudaLaunchConfig {
  // Number of threads per block.
  int thread_per_block = -1;
  // Number of blocks for Cuda kernel launch.
  int block_count = -1;
};

inline CudaLaunchConfig GetCudaLaunchConfig(int work_element_count) {
  CudaLaunchConfig config;
  const int thread_per_block = std::min(512, work_element_count);
  const int block_count = std::min(DivUp(work_element_count, thread_per_block), 32);

  config.thread_per_block = thread_per_block;
  config.block_count = block_count;
  return config;
}

// Helper for range-based for loop using 'delta' increments.
// Copied from tensorflow
template <typename T>
class CudaGridRange {
  struct Iterator {
    __device__ Iterator(T index, T delta) : index_(index), delta_(delta) {}
    __device__ T operator*() const { return index_; }
    __device__ Iterator& operator++() {
      index_ += delta_;
      return *this;
    }
    __device__ bool operator!=(const Iterator& other) const {
      bool greater = index_ > other.index_;
      bool less = index_ < other.index_;
      // Anything past an end iterator (delta_ == 0) is equal.
      // In range-based for loops, this optimizes to 'return less'.
      if (!other.delta_) {
        return less;
      }
      if (!delta_) {
        return greater;
      }
      return less || greater;
    }

   private:
    T index_;
    const T delta_;
  };

 public:
  __device__ CudaGridRange(T begin, T delta, T end)
      : begin_(begin), delta_(delta), end_(end) {}

  __device__ Iterator begin() const { return Iterator{begin_, delta_}; }
  __device__ Iterator end() const { return Iterator{end_, 0}; }

 private:
  T begin_;
  T delta_;
  T end_;
};

// Helper to visit indices in the range 0 <= i < count, using the x-coordinate
// of the global thread index. That is, each index i is visited by all threads
// with the same x-coordinate.
// Usage: for(int i : CudaGridRangeX(count)) { visit(i); }
template <typename T>
__device__ CudaGridRange<T> CudaGridRangeX(T count) {
  return CudaGridRange<T>(blockIdx.x * blockDim.x + threadIdx.x,
                                  gridDim.x * blockDim.x, count);
}
//#endif

#endif