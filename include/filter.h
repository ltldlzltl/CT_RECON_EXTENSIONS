/*
 * @Description: ramp filtering for FBP
 * @Author: Tianling Lyu
 * @Date: 2019-11-28 11:52:58
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-11-30 20:09:50
 */

#ifndef _CT_REON_EXT_FILTER_H_
#define _CT_REON_EXT_FILTER_H_

#include <string>
#include <cuda_runtime.h>

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
    bool calculate_on_cpu(T* filter);
    bool calculate_on_gpu(T* filter, cudaStream_t);

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
    bool calculate_on_cpu(const T* in, const T* filter, T* out);
    bool calculate_on_gpu(const T* in, const T* filter, T* out, 
        cudaStream_t);
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
    bool calculate_on_cpu(const T* in, const T* filter, T* out);
    bool calculate_on_gpu(const T* in, const T* filter, T* out, 
        cudaStream_t);
private:
    FilterParam param_;
}; // class RampFilterGrad

} // namespace ct_recon

#endif