/*
 * @Description: ramp filtering for FBP
 * @Author: Tianling Lyu
 * @Date: 2019-11-28 11:52:58
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-12-08 14:51:55
 */

#ifndef _CT_RECON_EXT_FILTER_H_
#define _CT_RECON_EXT_FILTER_H_

#include <string>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace ct_recon
{

// parameter structure
struct FilterParam
{
    unsigned int ns; // number of channels
    unsigned int nrow; // number of rows
    double ds;
    double dsd; // only needed for fan beam arc
    std::string type; // can be "par", "fan" or "flat"
    std::string window; // apply window on the filter, not implemented yet

    // Ctor
    FilterParam() {}
    FilterParam(unsigned int ns, unsigned int nrow, double ds,
        double dsd = -1, 
        const std::string& type="par",  
        const std::string& window="None")
        : ns(ns), nrow(nrow), ds(ds), dsd(dsd), type(type), window(window)
    {}
}; // struct FilterParam

template <typename T>
class RampFilterPrep
{
public:
    // Ctor and Dtor
    RampFilterPrep(const FilterParam& param)
        : param_(param)
    {}
    ~RampFilterPrep() {}
    // utility functions
    bool calculate_on_cpu(T* filter) const;
#ifdef USE_CUDA
    bool calculate_on_gpu(T* filter, cudaStream_t) const;
#endif

private:
    FilterParam param_;
}; // class RampFilterPrep

template <typename T>
class RampFilter
{
public:
    // Ctor and Dtor
    RampFilter(const FilterParam& param)
        : param_(param)
    {}
    ~RampFilter() {}
    // utility functions
    bool calculate_on_cpu(const T* in, const T* filter, T* out) const;
#ifdef USE_CUDA
    bool calculate_on_gpu(const T* in, const T* filter, T* out, 
        cudaStream_t) const;
#endif
private:
    FilterParam param_;
}; // class RampFilter

template <typename T>
class RampFilterGrad
{
public:
    // Ctor and Dtor
    RampFilterGrad(const FilterParam& param)
        : param_(param)
    {}
    ~RampFilterGrad() {}
    // utility functions
    bool calculate_on_cpu(const T* in, const T* filter, T* out) const;
#ifdef USE_CUDA
    bool calculate_on_gpu(const T* in, const T* filter, T* out, 
        cudaStream_t) const;
#endif
private:
    FilterParam param_;
}; // class RampFilterGrad

} // namespace ct_recon

#endif