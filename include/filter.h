/*
 * @Description: ramp filtering for FBP
 * @Author: Tianling Lyu
 * @Date: 2019-11-28 11:52:58
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-11-28 15:21:37
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
    unsigned int ns;
    unsigned int nrow;
    double ds;
    std::string type;
    std::string window;

    // Ctor
    FilterParam() {}
    FilterParam(unsigned int ns, unsigned int nrow, double ds,
        const std::string& type="par",  
        const std::string& window="None")
        : ns(ns), nrow(nrow), ds(ds), type(type), window(window)
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

} // namespace ct_recon

#endif