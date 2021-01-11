/*
 * @Description: implement ramp filter numpy extension library functions
 * @Author: Tianling Lyu
 * @Date: 2021-01-10 19:03:19
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2021-01-11 18:13:44
 */

#include "numpy_ext/filter_npext.h"
#include "numpy_ext/common.h"
#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif

namespace np_ext {

#define RampFilterContainer OpContainer<RampFilterNPExt, ct_recon::FilterParam, RampFilterRunParam>
RampFilterContainer ramp_filter_container_;

RampFilterNPExt::RampFilterNPExt(const ct_recon::FilterParam& param)
    : param_(param), allocated_(false), flt_prep_(param), 
    flt_(param), filter_(nullptr)
{
#ifdef USE_CUDA
    in_ = nullptr;
    out_ = nullptr;
#endif
}

RampFilterNPExt::~RampFilterNPExt()
{
    if (device_ < 0)
    {
        if (filter_ != nullptr)
            delete[] filter_;
    } else {
#ifdef USE_CUDA
        if (filter_ != nullptr)
            cudaFree(filter_);
        if (in_ != nullptr)
            cudaFree(in_);
        if (out_ != nullptr)
            cudaFree(out_);
#endif
    }
}

bool RampFilterNPExt::allocate() {
    if (allocated_) return true;
    if (device_ < 0) {
        // use CPU
        filter_ = new double[2*param_.ns+1];
        allocated_ = true;
        return flt_prep_.calculate_on_cpu(filter_);
    } else {
#ifdef USE_CUDA
        cudaError_t err;
        err = cudaMalloc(&filter_, (2*param_.ns+1)*sizeof(double));
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA allocate filter failed!");
        err = cudaMalloc(&in_, param_.ns*param_.nrow*sizeof(double));
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA allocate input array failed!");
        err = cudaMalloc(&out_, param_.ns*param_.nrow*sizeof(double));
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA allocate output array failed!");
        allocated_ = true;
        return flt_prep_.calculate_on_gpu(filter_, stream_);
#else
        return false;
#endif
    }
}

bool RampFilterNPExt::run(const RampFilterRunParam& param)
{
    if (device_ < 0) {
        // use CPU
        return flt_.calculate_on_cpu(param.in, filter_, param.out);
    } else {
        // use GPU
#ifdef USE_CUDA
        cudaMemcpy(in_, param.in, param_.nrow*param_.ns*sizeof(double), cudaMemcpyHostToDevice);
        bool finish = flt_.calculate_on_gpu(in_, filter_, out_, stream_);
        cudaMemcpy(param.out, out_, param_.nrow*param_.ns*sizeof(double), cudaMemcpyDeviceToHost);
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
#include <cstdio>

DLL_EXPORT extern "C"
int ramp_filter_create(unsigned int ns, unsigned int nrow, double ds, 
    double dsd, int type)
{
    std::string s_type;
    switch (type) {
        case 0: s_type = "par"; break;
        case 1: s_type = "fan"; break;
        case 2: s_type = "flat"; break;
        default: {
            throw std::runtime_error("Unknown filter type!");
        }
    }
    ct_recon::FilterParam param(ns, nrow, ds, dsd, s_type);
    int handle = np_ext::ramp_filter_container_.create(param);
    return handle;
}

DLL_EXPORT extern "C"
bool ramp_filter_run(int handle, double* in, double* out)
{
    np_ext::RampFilterRunParam param(in, out);
    return np_ext::ramp_filter_container_.run(handle, param);
}

DLL_EXPORT extern "C"
bool ramp_filter_destroy(int handle)
{
    return np_ext::ramp_filter_container_.erase(handle);
}