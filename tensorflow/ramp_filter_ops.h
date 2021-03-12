/*
 * @Description: ramp filtering structures for tensorflow
 * @Author: Tianling Lyu
 * @Date: 2019-12-01 10:22:45
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-03-11 14:38:31
 */

#ifndef TENSORFLOW_CORE_USER_OPS_RAMP_FILTER_OPS_H_
#define TENSORFLOW_CORE_USER_OPS_RAMP_FILTER_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"

#include "include/filter.h"

namespace tensorflow
{

// Functor for ramp filter preparation and ramp filter gradient preparation
template <typename Device>
struct LaunchRampFilterPrepOp {
    bool operator()(OpKernelContext* ctx, double* filter, 
        const ct_recon::RampFilterPrep* prep) {
        return prep->calculate_on_cpu(filter);
    }
}; // struct LaunchRampFilterPrepOp

// Functor for calculating ramp filter results
template <typename Device, typename T>
struct LaunchRampFilterOp {
    bool operator()(OpKernelContext* ctx, const T* in, const double* filter, 
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
    bool operator()(OpKernelContext* ctx, const T* in, const double* filter, 
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
} // namespace tensorflow

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#ifdef TFVER_1_14
#include "tensorflow/core/util/gpu_kernel_helper.h"
#else
#include "tensorflow/core/util/cuda_kernel_helper.h"
#endif
namespace tensorflow
{
// partial specializations for GPU devices
template <>
struct LaunchRampFilterPrepOp<Eigen::GpuDevice> {
    bool operator()(OpKernelContext* ctx, double* filter, 
        const ct_recon::RampFilterPrep *prep)
    {
        auto device = ctx->eigen_gpu_device();
        return prep->calculate_on_gpu(filter, device.stream());
    }
}; // LaunchRampFilterPrepOp

template <typename T>
struct LaunchRampFilterOp<Eigen::GpuDevice, T> {
    bool operator()(OpKernelContext* ctx, const T* in, 
        const double* filter, T* out, const ct_recon::RampFilter<T> *filt, 
        const int nbatch, const unsigned int sizeproj)
    {
        auto device = ctx->eigen_gpu_device();
        bool result = true;
        const T* in_ptr = in;
        T* out_ptr = out;
        for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
            result &= filt->calculate_on_gpu(in_ptr, filter, out_ptr, 
                device.stream());
            in_ptr += sizeproj;
            out_ptr += sizeproj;
        }
        return result;
    }
}; // struct LaunchRampFilterOp

template <typename T>
struct LaunchRampFilterGradOp<Eigen::GpuDevice, T> {
    bool operator()(OpKernelContext* ctx, const T* in, const double* filter, 
        T* out, const ct_recon::RampFilterGrad<T> *grad, const int nbatch, 
        const unsigned int sizeproj)
    {
        auto device = ctx->eigen_gpu_device();
        bool result = true;
        const T* in_ptr = in;
        T* out_ptr = out;
        for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
            result &= grad->calculate_on_gpu(in_ptr, filter, out_ptr, 
                device.stream());
            in_ptr += sizeproj;
            out_ptr += sizeproj;
        }
        return result;
    }
}; // struct LaunchRampFilterGradOp
} // namespace tensorflow
#endif

#endif