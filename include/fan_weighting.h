/*
 * @Description: pre-weighting on projection data
 * @Author: Tianling Lyu
 * @Date: 2021-01-08 16:32:04
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-02-07 16:23:40
 */

#ifndef _CT_RECON_EXT_FAN_WEIGHTING_H_
#define _CT_RECON_EXT_FAN_WEIGHTING_H_

#include <string>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace ct_recon
{

// parameter structure
struct FanWeightingParam
{
    unsigned int ns; // number of channels
    unsigned int nrow; // number of rows
    double ds;
    double offset;
    double dso;
    double dsd;
    int type; // 1 for "fan", 2 for "flat"
    //std::string type; // can be "fan" or "flat"

    // Ctor
    FanWeightingParam() {}
    FanWeightingParam(unsigned int ns, unsigned int nrow, double ds,
        double offset = 0, double dso = -1, double dsd = -1, int type = 1)
        : ns(ns), nrow(nrow), ds(ds), offset(offset), dso(dso), dsd(dsd), 
        type(type)
    {}
}; // struct FanWeightingParam

template <typename T>
class FanWeighting
{
public:
    // Ctor and Dtor
    FanWeighting(const FanWeightingParam& param)
        : param_(param)
    {}
    ~FanWeighting() {}
    // utility functions
    bool calculate_on_cpu(const T* in, T* out) const;
#ifdef USE_CUDA
    bool calculate_on_gpu(const T* in, T* out, 
        cudaStream_t) const;
#endif
private:
    FanWeightingParam param_;
}; // class FanWeighting

template <typename T>
class FanWeightingGrad
{
public:
    // Ctor and Dtor
    FanWeightingGrad(const FanWeightingParam& param)
        : param_(param)
    {}
    ~FanWeightingGrad() {}
    // utility functions
    bool calculate_on_cpu(const T* in, T* out) const;
#ifdef USE_CUDA
    bool calculate_on_gpu(const T* in, T* out, 
        cudaStream_t) const;
#endif
private:
    FanWeightingParam param_;
}; // class FanWeightingGrad

} // namespace ct_recon

#endif