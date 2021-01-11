/*
 * @Description: implement fan weighting numpy extension library functions
 * @Author: Tianling Lyu
 * @Date: 2021-01-10 22:35:16
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2021-01-11 18:01:09
 */

#include "numpy_ext/fan_weighting_npext.h"
#include "numpy_ext/common.h"
#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif

#include <cstdio>

namespace np_ext {

#define FanWContainer OpContainer<FanWeightingNPExt, ct_recon::FanWeightingParam, FanWeightingRunParam>
FanWContainer fan_w_container_;

FanWeightingNPExt::FanWeightingNPExt(const ct_recon::FanWeightingParam& param, int device)
    : param_(param), device_(device), allocated_(false), fw_(param)
{
#ifdef USE_CUDA
    inout_ = nullptr;
    stream_ = nullptr;
#endif
}

FanWeightingNPExt::~FanWeightingNPExt()
{
    if (device_ >= 0)  {
#ifdef USE_CUDA
        if (inout_ != nullptr)
            cudaFree(inout_);
        if (stream_ != nullptr)
            cudaStreamDestroy(stream_);
#endif
    }
}

bool FanWeightingNPExt::allocate() {
    if (allocated_) return true;
    if (device_ < 0) {
        allocated_ = true;
        return true;
    } else {
#ifdef USE_CUDA
        cudaError_t err;
        err = cudaSetDevice(device_);
        if (err != cudaSuccess) 
            throw std::runtime_error("Device not found!");
        err = cudaStreamCreate(&stream_);
        if (err != cudaSuccess) 
            throw std::runtime_error("Stream initialization failed!");
        err = cudaMalloc(&inout_, param_.ns*param_.nrow*sizeof(double));
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA allocate inout array failed!");
        allocated_ = true;
        return true;
#else
        return false;
#endif
    }
}

bool FanWeightingNPExt::run(const FanWeightingRunParam& param)
{
    if (device_ < 0) {
        return fw_.calculate_on_cpu(param.in, param.out);
    } else {
        // use GPU
#ifdef USE_CUDA
        cudaMemcpy(inout_, param.in, param_.nrow*param_.ns*sizeof(double), cudaMemcpyHostToDevice);
        bool finish = fw_.calculate_on_gpu(inout_, inout_, stream_);
        cudaMemcpy(param.out, inout_, param_.nrow*param_.ns*sizeof(double), cudaMemcpyDeviceToHost);
        return finish;
#else
        return false;
#endif
    }
}

} // namespace np_ext

#if defined(_WIN32)
#define DLL_EXPORT _declspec(dllexport)
#else
#define DLL_EXPORT
#endif

#include <string>

DLL_EXPORT extern "C"
int fan_weighting_create(unsigned int ns, unsigned int nrow, double ds, 
    double dso, double dsd, int type, int device)
{
    std::string s_type;
    switch (type) {
        case 1: s_type = "fan"; break;
        case 2: s_type = "flat"; break;
        default: {
            throw std::runtime_error("Unknown filter type!");
        }
    }
    ct_recon::FanWeightingParam param(ns, nrow, ds, dso, dsd, s_type);
    int handle = np_ext::fan_w_container_.create(param, device);
    return handle;
}

DLL_EXPORT extern "C"
bool fan_weighting_run(int handle, double* in, double* out)
{
    np_ext::FanWeightingRunParam param(in, out);
    return np_ext::fan_w_container_.run(handle, param);
}

DLL_EXPORT extern "C"
bool fan_weighting_destroy(int handle)
{
    return np_ext::fan_w_container_.erase(handle);
}