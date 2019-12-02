/*
 * @Description: ramp filtering structures for tensorflow
 * @Author: Tianling Lyu
 * @Date: 2019-12-01 10:22:45
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-12-01 23:00:41
 */

#ifndef TENSORFLOW_CORE_USER_OPS_RAMP_FILTER_OPS_H_
#define TENSORFLOW_CORE_USER_OPS_RAMP_FILTER_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"

#include "include/filter.h"

namespace tensorflow
{

// Functor for ramp filter preparation and ramp filter gradient preparation
template <typename Device, typename T>
struct LaunchRampFilterPrepOp {
    bool operator()(OpKernelContext* ctx, T* filter, 
        const ct_recon::RampFilterPrep<T>* prep) {
        return prep->calculate_on_cpu(filter);
    }
}; // struct LaunchRampFilterPrepOp

// Functor for calculating ramp filter results
template <typename Device, typename T>
struct LaunchRampFilterOp {
    bool operator()(OpKernelContext* ctx, const T* in, const T* filter, 
        T* out, const ct_recon::RampFilter<T>* ramp, const int nbatch, 
        const unsigned int sizeproj) {
        bool result = true;
        const T* in_ptr = in;
        T* out_ptr = out;
        // iterate in batch
        for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
            result &= ramp->calculate_on_cpu(in_ptr, filter, out_ptr);
            in_ptr += sizeproj;
            out_ptr += sizeproj;
        }
        return result;
    }
}; // struct LaunchRampFilterOp

// Functor for calculating ramp filter gradient
template <typename Device, typename T>
struct LaunchRampFilterGradOp {
    bool operator()(OpKernelContext* ctx, const T* in, const T* filter, 
        T* out, const ct_recon::RampFilterGrad<T>* ramp, const int nbatch, 
        const unsigned int sizeproj) {
        bool result = true;
        const T* in_ptr = in;
        T* out_ptr = out;
        // iterate in batch
        for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
            result &= ramp->calculate_on_cpu(in_ptr, filter, out_ptr);
            in_ptr += sizeproj;
            out_ptr += sizeproj;
        }
        return result;
    }
}; // struct LaunchRampFilterGradOp

#if GOOGLE_CUDA


} // namespace tensorflow

#endif