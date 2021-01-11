/*
 * @Description: 
 * @Author: Tianling Lyu
 * @Date: 2021-01-09 18:06:32
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2021-01-11 18:04:07
 */

#include "numpy_ext/common.h"
#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif

namespace np_ext {

int device_ = -1;
#ifdef USE_CUDA
cudaStream_t stream_ = 0;
#endif

} // namespace np_ext

#if defined(_WIN32)
#define DLL_EXPORT _declspec(dllexport)
#else
#define DLL_EXPORT
#endif

DLL_EXPORT extern "C"
bool set_device(const int device)
{
    np_ext::device_ = device;
    if (np_ext::device_ >= 0) {
        // use GPU
#ifdef USE_CUDA
        cudaError_t err;
        err = cudaSetDevice(np_ext::device_);
        if (err != cudaSuccess) 
            throw std::runtime_error("Device not found!");
        err = cudaStreamCreate(&np_ext::stream_);
        if (err != cudaSuccess) 
            throw std::runtime_error("Stream initialization failed!");
        return true;
#else
        return false;
#endif
    }
    return true;
}

DLL_EXPORT extern "C"
bool clear()
{
    if (np_ext::device_ >= 0) {
#ifdef USE_CUDA
        if (np_ext::stream_ != 0) 
            cudaStreamDestroy(np_ext::stream_);
#endif
    }
    return true;
}